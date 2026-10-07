from __future__ import annotations

import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from .assets import assert_downloads
from .common import ROOT, project_path, run, write_json
from .rl_env import GridDataset
from .video import count_frames

YOLOV5_COMMIT = "915bbf294bb74c859f0b41f1c23bc395014ea679"


def preflight(config: dict) -> None:
    if not Path("/.dockerenv").exists():
        raise RuntimeError("pipeline must run inside its pinned Docker image")
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU is mandatory and unavailable inside Docker; no CPU fallback is allowed")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device found")
    if torch.__version__.split("+")[0] != "2.2.2" or torchvision.__version__.split("+")[0] != "0.17.2":
        raise RuntimeError(
            f"unexpected PyTorch stack: torch={torch.__version__}, torchvision={torchvision.__version__}"
        )
    if torch.version.cuda != "12.1":
        raise RuntimeError(f"unexpected PyTorch CUDA build: {torch.version.cuda}")
    yolo_root = Path(os.environ["YOLOV5_ROOT"])
    if not yolo_root.is_dir():
        raise RuntimeError(f"pinned YOLOv5 source missing: {yolo_root}")
    if os.environ.get("YOLOv5_AUTOINSTALL", "").lower() != "false":
        raise RuntimeError("YOLOv5_AUTOINSTALL must be false")
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))
    yolo_head = run(["git", "-C", yolo_root, "rev-parse", "HEAD"], capture=True).stdout.strip()
    if yolo_head != YOLOV5_COMMIT:
        raise RuntimeError(f"unexpected YOLOv5 revision: {yolo_head}")
    from models.common import AutoShape, DetectMultiBackend  # noqa: F401

    for executable in ("ffmpeg", "ffprobe", "roi_encoder", "git"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable missing: {executable}")
    encoders = run(["ffmpeg", "-hide_banner", "-encoders"], capture=True).stdout
    filters = run(["ffmpeg", "-hide_banner", "-filters"], capture=True).stdout
    if "libx264" not in encoders:
        raise RuntimeError("required FFmpeg libx264 encoder is missing")
    if "libvmaf" not in filters:
        raise RuntimeError("required FFmpeg libvmaf filter is missing")
    ffmpeg_version = run(["ffmpeg", "-version"], capture=True).stdout.splitlines()[0]
    if not ffmpeg_version.startswith("ffmpeg version n6.1.1 "):
        raise RuntimeError(f"unexpected FFmpeg build: {ffmpeg_version}")
    vmaf_result = run(["vmaf", "--version"], capture=True)
    vmaf_version = (vmaf_result.stdout + vmaf_result.stderr).strip()
    if vmaf_version != "17a67b2":
        raise RuntimeError(f"unexpected libvmaf build: {vmaf_version}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("Preflight passed: Docker, CUDA, libx264, libvmaf, roi_encoder")


def _frame_types(path: Path) -> list[tuple[int, str]]:
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=key_frame,pict_type", "-of", "json", path,
    ], capture=True)
    frames = json.loads(result.stdout)["frames"]
    return [(int(frame["key_frame"]), frame["pict_type"]) for frame in frames]


def _validate_encode(encode, gop: int) -> None:
    path = ROOT / encode.encoded_path
    if count_frames(path) != encode.frame_count:
        raise RuntimeError(f"encoded frame count mismatch: {path}")
    types = _frame_types(path)
    if len(types) != encode.frame_count:
        raise RuntimeError(f"frame type count mismatch: {path}")
    for index, (key, picture_type) in enumerate(types):
        expected_key = index % gop == 0
        if expected_key and (key != 1 or picture_type != "I"):
            raise RuntimeError(f"missing IDR/I at frame {index}: {path}")
        if not expected_key and (key != 0 or picture_type != "P"):
            raise RuntimeError(f"non-P frame at frame {index}: {path}")


def validate_dataset(config: dict) -> None:
    assert_downloads(config)
    artifacts = project_path(config, "artifacts")
    processed = project_path(config, "processed_data")
    reports = project_path(config, "reports")
    required = {
        "reference": artifacts / "reference" / "manifest.parquet",
        "detections": artifacts / "metadata" / "detections.parquet",
        "semantics": artifacts / "metadata" / "gop_semantics.parquet",
        "encodes": artifacts / "encodes" / "manifest.parquet",
        "action_space": artifacts / "encodes" / "action_space.parquet",
        "grid": artifacts / "metrics" / "gop_action_grid.parquet",
        "traces": processed / "bandwidth_traces.parquet",
        "catalog": processed / "trace_catalog.parquet",
        "transitions": processed / "transitions.parquet",
        "schedules": processed / "action_schedules.parquet",
        "transition_schema": processed / "transition_schema.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"mandatory outputs missing: {missing}")
    reference = pd.read_parquet(required["reference"])
    semantics = pd.read_parquet(required["semantics"])
    encodes = pd.read_parquet(required["encodes"])
    action_space = pd.read_parquet(required["action_space"])
    grid = pd.read_parquet(required["grid"])
    traces = pd.read_parquet(required["traces"])
    catalog = pd.read_parquet(required["catalog"])
    transitions = pd.read_parquet(required["transitions"])
    schedules = pd.read_parquet(required["schedules"])
    gop = config["video"]["gop_frames"]
    action_count = len(config["encoder"]["bitrates_mbps"]) * len(config["encoder"]["roi_delta_qp"])

    if not (reference.frame_count % gop == 0).all():
        raise RuntimeError("reference contains a partial GOP")
    if not semantics.semantic_score.between(0, 1).all():
        raise RuntimeError("semantic score outside [0,1]")
    if semantics.duplicated(["video_id", "gop_idx"]).any():
        raise RuntimeError("duplicate semantic GOP key")
    if len(semantics) != int(reference.gop_count.sum()):
        raise RuntimeError("semantic GOP count does not match reference manifest")
    if len(encodes) != len(reference) * action_count:
        raise RuntimeError(f"encode manifest rows {len(encodes)} != {len(reference) * action_count}")
    expected_actions = {
        (index, float(bitrate), int(delta))
        for index, (bitrate, delta) in enumerate(
            (bitrate, delta)
            for bitrate in config["encoder"]["bitrates_mbps"]
            for delta in config["encoder"]["roi_delta_qp"]
        )
    }
    observed_actions = set(action_space[["action_id", "target_bitrate_mbps", "delta_qp_roi"]].itertuples(index=False, name=None))
    if observed_actions != expected_actions or len(action_space) != len(expected_actions):
        raise RuntimeError("action-space artifact does not match the configured Cartesian product")
    expected_grid = int(reference.gop_count.sum()) * action_count
    if len(grid) != expected_grid:
        raise RuntimeError(f"grid rows {len(grid)} != {expected_grid}")
    if grid.duplicated(["video_id", "gop_idx", "action_id"]).any():
        raise RuntimeError("duplicate gop/action key")
    per_gop_actions = grid.groupby(["video_id", "gop_idx"]).action_id.agg(["count", "nunique"])
    if not ((per_gop_actions["count"] == action_count) & (per_gop_actions["nunique"] == action_count)).all():
        raise RuntimeError("one or more GOPs do not contain the complete action grid")
    if not (grid.frame_count == gop).all() or not (grid.packet_count == gop).all() or not (grid.key_packets == 1).all():
        raise RuntimeError("grid contains an invalid GOP frame/packet/keyframe count")
    stride = config.get("metrics", {}).get("sample_stride", 1)
    if stride > 1 and ("metric_sample_count" not in grid or
                       not (grid.metric_sample_count == gop // stride).all()):
        raise RuntimeError("grid contains invalid metric sample coverage")
    mandatory_finite = ["actual_bitrate_mbps", "vmaf", "psnr", "ssim"]
    if not np.isfinite(grid[mandatory_finite].to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite mandatory metric")
    if set(catalog.split) != {"train", "validation", "test"} or set(catalog.source) != {"fcc", "norway"}:
        raise RuntimeError("FCC/Norway bandwidth split incomplete")
    if catalog.groupby("group_id").split.nunique().max() != 1:
        raise RuntimeError("trace session leaks between splits")
    video_splits = reference.set_index("video_id").split
    if not (transitions.video_id.map(video_splits) == transitions.split).all():
        raise RuntimeError("transition video leaks between splits")
    if (traces.bandwidth_mbps < 0).any():
        raise RuntimeError("negative bandwidth value")
    if transitions.empty or not transitions.semantic_score.between(0, 1).all():
        raise RuntimeError("invalid transition dataset")
    if transitions.duplicated(["episode_id", "gop_idx"]).any():
        raise RuntimeError("duplicate transition key")
    transition_finite = [
        "bw_k_minus_4", "bw_k_minus_3", "bw_k_minus_2", "bw_k_minus_1",
        "r_prev_actual_mbps", "semantic_score", "target_bitrate_mbps",
        "actual_bitrate_mbps", "reward", "next_bw_k_minus_4", "next_bw_k_minus_3",
        "next_bw_k_minus_2", "next_bw_k_minus_1", "next_r_prev_actual_mbps",
        "next_semantic_score", "vmaf", "psnr", "ssim",
    ]
    if not np.isfinite(transitions[transition_finite].to_numpy(dtype=float)).all():
        raise RuntimeError("transition dataset contains a non-finite mandatory value")
    if not transitions.action_id.between(0, action_count - 1).all():
        raise RuntimeError("transition action outside the configured action space")
    done_counts = transitions.groupby("episode_id").done.sum()
    if not (done_counts == 1).all():
        raise RuntimeError("each episode must contain exactly one terminal transition")

    current = transitions.merge(
        grid[["video_id", "gop_idx", "action_id", "actual_bitrate_mbps"]],
        on=["video_id", "gop_idx", "action_id"], suffixes=("", "_grid"), validate="many_to_one",
    )
    if not np.allclose(current.actual_bitrate_mbps, current.actual_bitrate_mbps_grid):
        raise RuntimeError("transition outcome does not match selected grid action")
    previous_schedule = schedules[["episode_id", "video_id", "gop_idx", "action_id"]].copy()
    previous_schedule["gop_idx"] += 1
    previous_schedule = previous_schedule.rename(columns={"action_id": "previous_action_id"})
    previous = transitions.merge(
        previous_schedule, on=["episode_id", "video_id", "gop_idx"], validate="one_to_one"
    )
    previous["previous_gop_idx"] = previous.gop_idx - 1
    previous = previous.merge(
        grid[["video_id", "gop_idx", "action_id", "actual_bitrate_mbps"]].rename(columns={
            "gop_idx": "previous_gop_idx", "action_id": "previous_action_id",
            "actual_bitrate_mbps": "expected_r_prev_actual_mbps",
        }),
        on=["video_id", "previous_gop_idx", "previous_action_id"], validate="many_to_one",
    )
    if not np.allclose(previous.r_prev_actual_mbps, previous.expected_r_prev_actual_mbps):
        raise RuntimeError("transition previous bitrate does not match the previous selected action")

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(lambda encode: _validate_encode(encode, gop), encodes.itertuples(index=False)):
            pass

    GridDataset(config)

    report = {
        "status": "passed", "videos": len(reference), "reference_frames": int(reference.frame_count.sum()),
        "gops": int(reference.gop_count.sum()), "actions": action_count, "grid_rows": len(grid),
        "trace_count": len(catalog), "transition_count": len(transitions),
        "detector": config["yolo"]["variant"], "detector_weights": config["yolo"]["weights"],
        "cuda_device": torch.cuda.get_device_name(0),
    }
    write_json(reports / "validation_report.json", report)
    print(json.dumps(report, indent=2))
