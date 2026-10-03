from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ROOT, project_path, run, stable_hash, write_json


def _packet_metrics(path: Path, fps: int, gop_frames: int) -> pd.DataFrame:
    probe = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts_time,size,flags", "-of", "json", path,
    ], capture=True)
    packets = json.loads(probe.stdout)["packets"]
    rows = []
    for packet in packets:
        frame_idx = int(round(float(packet["pts_time"]) * fps))
        rows.append({"frame_idx": frame_idx, "packet_bytes": int(packet["size"]), "key_packet": "K" in packet["flags"]})
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"no video packets found in {path}")
    result["gop_idx"] = result.frame_idx // gop_frames
    return result


def _run_vmaf(encoded: Path, reference: Path, log_path: Path) -> pd.DataFrame:
    log_path.unlink(missing_ok=True)
    graph = (
        "[0:v]settb=AVTB,setpts=N/(30*TB)[dist];"
        "[1:v]settb=AVTB,setpts=N/(30*TB)[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path}:"
        "model=version=vmaf_v0.6.1:feature=name=psnr|name=float_ssim:n_threads=6"
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", encoded, "-i", reference,
        "-filter_complex", graph, "-f", "null", "-",
    ])
    if not log_path.is_file():
        raise RuntimeError("libvmaf did not create its mandatory JSON log")
    data = json.loads(log_path.read_text())
    rows = []
    for frame in data["frames"]:
        metrics = frame["metrics"]
        try:
            rows.append({
                "frame_idx": int(frame["frameNum"]),
                "vmaf": float(metrics["vmaf"]),
                "psnr": float(metrics["psnr_y"]),
                "ssim": float(metrics["float_ssim"]),
            })
        except KeyError as error:
            raise RuntimeError(f"required libvmaf metric missing: {error}; available={sorted(metrics)}") from error
    if not rows:
        raise RuntimeError("libvmaf returned no per-frame metrics")
    return pd.DataFrame(rows)


def _roi_metrics(encoded: Path, reference: Path, detections: pd.DataFrame, frame_count: int) -> pd.DataFrame:
    import cv2
    from skimage.metrics import structural_similarity

    encoded_cap = cv2.VideoCapture(str(encoded))
    reference_cap = cv2.VideoCapture(str(reference))
    if not encoded_cap.isOpened() or not reference_cap.isOpened():
        raise RuntimeError(f"cannot decode ROI metric pair: {encoded}, {reference}")
    by_frame = {int(idx): group for idx, group in detections.groupby("frame_idx")}
    rows = []
    for frame_idx in tqdm(range(frame_count), desc=f"ROI metrics {encoded.stem}", leave=False, disable=True):
        ok_dist, distorted = encoded_cap.read()
        ok_ref, original = reference_cap.read()
        if not ok_dist or not ok_ref:
            raise RuntimeError(f"metric decode ended at frame {frame_idx}: {encoded}")
        group = by_frame.get(frame_idx)
        if group is None or group.empty:
            rows.append({"frame_idx": frame_idx, "has_roi": False, "roi_pixel_fraction": 0.0,
                         "roi_psnr": np.nan, "roi_ssim": np.nan})
            continue
        height, width = original.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        ssim_weighted = 0.0
        weight_sum = 0.0
        for box in group.itertuples(index=False):
            x1, y1 = max(0, int(math.floor(box.x1))), max(0, int(math.floor(box.y1)))
            x2, y2 = min(width, int(math.ceil(box.x2))), min(height, int(math.ceil(box.y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = True
            ref_crop = original[y1:y2, x1:x2]
            dist_crop = distorted[y1:y2, x1:x2]
            minimum = min(ref_crop.shape[0], ref_crop.shape[1])
            win_size = min(7, minimum if minimum % 2 == 1 else minimum - 1)
            if win_size >= 3:
                score = structural_similarity(ref_crop, dist_crop, channel_axis=2, data_range=255, win_size=win_size)
                weight = float(box.class_weight * box.confidence * (x2 - x1) * (y2 - y1))
                ssim_weighted += score * weight
                weight_sum += weight
        if not mask.any():
            rows.append({"frame_idx": frame_idx, "has_roi": False, "roi_pixel_fraction": 0.0,
                         "roi_psnr": np.nan, "roi_ssim": np.nan})
            continue
        difference = original[mask].astype(np.float32) - distorted[mask].astype(np.float32)
        mse = float(np.mean(np.square(difference)))
        roi_psnr = 100.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
        rows.append({
            "frame_idx": frame_idx, "has_roi": True,
            "roi_pixel_fraction": float(mask.mean()), "roi_psnr": roi_psnr,
            "roi_ssim": ssim_weighted / weight_sum if weight_sum else np.nan,
        })
    encoded_cap.release()
    reference_cap.release()
    return pd.DataFrame(rows)


def _metric_one(encode, reference: pd.Series, detections: pd.DataFrame,
                metrics_root: Path, fps: int, gop_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_path = ROOT / encode.encoded_path
    reference_path = ROOT / reference.reference_path
    video_dir = metrics_root / encode.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    frame_output = video_dir / f"action_{encode.action_id:02d}.frames.parquet"
    gop_output = video_dir / f"action_{encode.action_id:02d}.gops.parquet"
    sidecar = video_dir / f"action_{encode.action_id:02d}.json"
    job = {
        "encoded_sha256": encode.encoded_sha256, "reference_sha256": reference.reference_sha256,
        "metrics": ["vmaf", "psnr_y", "float_ssim", "roi_psnr", "roi_ssim"],
    }
    job_hash = stable_hash(job)
    if frame_output.is_file() and gop_output.is_file() and sidecar.is_file():
        existing = json.loads(sidecar.read_text())
        if existing.get("job_hash") != job_hash:
            raise RuntimeError(f"incompatible metric artifact: {gop_output}")
        return pd.read_parquet(frame_output), pd.read_parquet(gop_output)

    vmaf_log = video_dir / f"action_{encode.action_id:02d}.vmaf.json"
    full = _run_vmaf(encoded_path, reference_path, vmaf_log)
    roi = _roi_metrics(encoded_path, reference_path, detections, int(encode.frame_count))
    packets = _packet_metrics(encoded_path, fps, gop_frames)
    frames = full.merge(roi, on="frame_idx", validate="one_to_one")
    if len(frames) != int(encode.frame_count):
        raise RuntimeError(f"metric frame count mismatch: {encoded_path}")
    frames["video_id"] = encode.video_id
    frames["action_id"] = encode.action_id
    frames["gop_idx"] = frames.frame_idx // gop_frames
    frames.to_parquet(frame_output, index=False)

    packet_gops = packets.groupby("gop_idx", as_index=False).agg(
        payload_bytes=("packet_bytes", "sum"), key_packets=("key_packet", "sum"),
        packet_count=("packet_bytes", "size"),
    )
    quality = frames.groupby("gop_idx", as_index=False).agg(
        vmaf=("vmaf", "mean"), psnr=("psnr", "mean"), ssim=("ssim", "mean"),
        roi_psnr=("roi_psnr", "mean"), roi_ssim=("roi_ssim", "mean"),
        roi_pixel_fraction=("roi_pixel_fraction", "mean"), roi_frame_count=("has_roi", "sum"),
        frame_count=("frame_idx", "size"),
    )
    gops = quality.merge(packet_gops, on="gop_idx", validate="one_to_one")
    duration = gop_frames / fps
    gops["actual_bitrate_mbps"] = gops.payload_bytes * 8.0 / duration / 1_000_000.0
    gops["video_id"] = encode.video_id
    gops["action_id"] = encode.action_id
    gops["bitrate_index"] = encode.bitrate_index
    gops["roi_index"] = encode.roi_index
    gops["target_bitrate_mbps"] = encode.target_bitrate_mbps
    gops["delta_qp_roi"] = encode.delta_qp_roi
    gops["stream_container_overhead_bytes"] = encoded_path.stat().st_size - int(packets.packet_bytes.sum())
    gops.to_parquet(gop_output, index=False)
    write_json(sidecar, {**job, "job_hash": job_hash})
    return frames, gops


def compute_metrics(config: dict) -> None:
    artifacts = project_path(config, "artifacts")
    encode_manifest_path = artifacts / "encodes" / "manifest.parquet"
    reference_manifest_path = artifacts / "reference" / "manifest.parquet"
    detections_path = artifacts / "metadata" / "detections.parquet"
    if not all(path.is_file() for path in (encode_manifest_path, reference_manifest_path, detections_path)):
        raise RuntimeError("encode manifest, reference manifest, or detections missing")
    encodes = pd.read_parquet(encode_manifest_path)
    references = pd.read_parquet(reference_manifest_path).set_index("video_id")
    detections = pd.read_parquet(detections_path)
    detections_by_video = {video_id: frame for video_id, frame in detections.groupby("video_id")}
    fps = config["video"]["fps"]
    gop_frames = config["video"]["gop_frames"]
    metrics_root = artifacts / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    grid_parts, frame_parts = [], []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_metric_one, SimpleNamespace(**encode._asdict()),
                               references.loc[encode.video_id],
                               detections_by_video.get(encode.video_id, detections.iloc[0:0]),
                               metrics_root, fps, gop_frames)
                   for encode in encodes.itertuples(index=False)]
        for future in as_completed(futures):
            frames, gops = future.result()
            frame_parts.append(frames)
            grid_parts.append(gops)
    pd.concat(frame_parts, ignore_index=True).to_parquet(metrics_root / "frame_metrics.parquet", index=False)
    pd.concat(grid_parts, ignore_index=True).sort_values(
        ["video_id", "gop_idx", "action_id"]
    ).to_parquet(metrics_root / "gop_action_grid.parquet", index=False)
