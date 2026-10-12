#!/usr/bin/env python3
"""Host-only packaging. Run with pandas/pyarrow and FFmpeg, never on the board."""
import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from rl_runtime.common import FPS, GOP, FRAME_BYTES, FRAMES, GOPS, sha256, write_json


def copy_checked(source, destination):
    if not source.is_file():
        raise RuntimeError("required source missing: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(source) != sha256(destination):
        raise RuntimeError("refusing to replace different artifact: " + str(destination))
    if not destination.exists():
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-source", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT / "config/pipeline.yaml").read_text())
    processed = PROJECT / config["paths"]["processed_data"]
    videos = json.loads((processed / "video_manifest.json").read_text())["videos"]
    eligible = sorted((v for v in videos if v["split"] == "test" and v["duration_s"] >= FRAMES/FPS),
                      key=lambda v: v["video_id"])
    if not eligible:
        raise RuntimeError("no current test video has 32 seconds")
    video = eligible[0]
    source = PROJECT / video["source_path"]
    if sha256(source) != video["source_sha256"]:
        raise RuntimeError("video source checksum mismatch")
    trace_table = processed / "bandwidth_traces.parquet"
    traces = pd.read_parquet(trace_table)
    candidates = []
    for trace_id, group in traces[traces.split == "test"].groupby("trace_id", sort=True):
        group = group.sort_values("gop_idx")
        first = group.iloc[:GOPS]
        if (len(first) == GOPS and first.gop_idx.tolist() == list(range(GOPS))
                and first.timestamp_s.tolist() == list(range(GOPS))
                and first.t_start_s.tolist() == list(range(GOPS))
                and first.t_end_s.tolist() == list(range(1, GOPS+1))
                and np.isfinite(first.bandwidth_mbps).all() and (first.bandwidth_mbps >= 0).all()):
            candidates.append((str(trace_id), first))
    if not candidates:
        raise RuntimeError("no test trace has 32 contiguous GOPs at offset zero")
    trace_id, trace = candidates[0]
    trace_source = PROJECT / config["paths"]["raw_data"] / "network_traces" / trace.iloc[0].source_file
    if sha256(trace_source) != trace.iloc[0].raw_sha256:
        raise RuntimeError("raw test trace checksum mismatch")
    trace_path = ROOT / "bw_trace/test_segment.json"
    trace_meta = {"trace_id": trace_id, "split": "test", "offset_gop": 0,
                  "source_file": str(trace.iloc[0].source_file), "raw_sha256": str(trace.iloc[0].raw_sha256),
                  "processed_table_sha256": sha256(trace_table), "source": str(trace.iloc[0].source),
                  "path": "bw_trace/test_segment.json"}
    write_json(trace_path, {**trace_meta, "samples": trace[["gop_idx", "timestamp_s", "t_start_s", "t_end_s", "bandwidth_mbps"]].to_dict("records")})
    raw = ROOT / "videos/inference" / (video["video_id"]+"_test_32s.nv12")
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        receipt = json.loads((ROOT / "manifest.json").read_text())
        if sha256(raw) != receipt["files"][str(raw.relative_to(ROOT))]["sha256"]:
            raise RuntimeError("existing NV12 differs from packaging receipt")
    else:
        temporary = raw.with_suffix(".partial.nv12")
        if temporary.exists():
            raise RuntimeError("partial clip exists; inspect/remove explicitly before retry")
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(source), "-map", "0:v:0", "-an",
                        "-vf", "fps=30:round=near,scale=1920:1080:flags=lanczos,format=nv12,setpts=N/30/TB",
                        "-frames:v", str(FRAMES), "-c:v", "rawvideo", "-pix_fmt", "nv12",
                        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                        "-f", "rawvideo", str(temporary)], check=True)
        if temporary.stat().st_size != FRAMES*FRAME_BYTES:
            raise RuntimeError("normalization did not produce 960 NV12 frames")
        temporary.replace(raw)
    policy_src = PROJECT / "reports/benchmark/legacy_artifacts/rl/ddqn_selected.npz"
    copy_checked(policy_src, ROOT / "models/rl/ddqn_policy.npz")
    copy_checked(PROJECT / "src/vcu_surrogate/policy_numpy.py", ROOT / "rl_runtime/policy_numpy.py")
    for name in ("yolov5_nano_pt.xmodel", "yolov5_nano_pt.prototxt", "meta.json", "md5sum.txt"):
        copy_checked(args.semantic_source / "models/yolov5_nano_pt" / name, ROOT / "models/yolov5n" / name)
    from rl_runtime.policy_numpy import NumpyPolicy
    policy = NumpyPolicy(ROOT / "models/rl/ddqn_policy.npz")
    # Vendor only the pure NumPy tensor/YOLO routines, not the CPU image extractor.
    origin = args.semantic_source / "rl_env_offline/extract_dpu_metadata.py"
    code = origin.read_text()
    tree = ast.parse(code)
    names = {"get_dpu_subgraphs", "tensor_dims", "tensor_fix_point", "infer_input_layout", "dequantize_output",
             "sigmoid", "normalize_yolo_head", "decode_head", "box_iou_one_to_many", "match_anchors_to_head"}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not names.issubset(functions):
        raise RuntimeError("DPU source API changed")
    generated = '"""Vendored pure tensor/YOLO routines; original sha256: '+sha256(origin)+'"""\nimport numpy as np\n\n'
    generated += "\n\n".join(ast.get_source_segment(code, functions[name]) for name in sorted(names)) + "\n"
    (ROOT / "rl_runtime/dpu_core.py").write_text(generated)
    # Snapshot only the existing metric/reward primitives needed on the host.
    metric_sources = {
        "src/vcu_surrogate/metrics.py": ("_roi_ssim", "_roi_metrics", "_vmaf_command", "_read_vmaf", "_run_vmaf"),
        "src/vcu_surrogate/rl_env.py": ("reward_for",),
        "src/vcu_surrogate/benchmark_metrics.py": ("detection_match", "fluid_queue"),
    }
    host_code = ('from __future__ import annotations\nimport json, math, subprocess\nfrom pathlib import Path\n'
                 'import numpy as np\nimport pandas as pd\nfrom tqdm import tqdm\n'
                 'def run(command):\n    return subprocess.run([str(x) for x in command], check=True)\n\n')
    for path, selected in metric_sources.items():
        original = (PROJECT / path).read_text()
        nodes = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
        host_code += '# Source: '+path+' sha256 '+sha256(PROJECT/path)+'\n'
        host_code += "\n\n".join(ast.get_source_segment(original, nodes[name]) for name in selected)+"\n\n"
    (ROOT / "tools/host_metrics.py").write_text(host_code)
    labels = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "COCO80" for t in node.targets))
    write_json(ROOT / "models/yolov5n/labels.json", labels)
    biases = [float(x) for x in re.findall(r"^\s*biases\s*:\s*([\d.]+)\s*$",
              (ROOT / "models/yolov5n/yolov5_nano_pt.prototxt").read_text(), re.M)]
    if len(biases) != 18:
        raise RuntimeError("YOLOv5n requires 18 explicit anchor values")
    write_json(ROOT / "models/yolov5n/anchors.json", np.array(biases).reshape(3, 3, 2).tolist())
    copy_checked(PROJECT / "models/yolov5n.pt", ROOT / "models/yolov5n/offline_yolov5n.pt")
    for directory in ("results", "vcu_test"):
        (ROOT / directory).mkdir(exist_ok=True)
    files = {}
    for folder in ("bw_trace", "models", "videos", "rl_runtime", "tools", "vcu_test"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"schema_version": 1, "gop_frames": 30, "fallback": False,
                "video": {**video, "source_duration_s": video["duration_s"], "source_size_bytes": video["size_bytes"],
                          "duration_s": FRAMES/FPS, "size_bytes": FRAMES*FRAME_BYTES,
                          "path": str(raw.relative_to(ROOT)), "frames": FRAMES, "fps": FPS,
                          "width": 1920, "height": 1080, "pixel_format": "nv12", "start_frame": 0},
                "trace": trace_meta, "policy": {"source": str(policy_src.relative_to(PROJECT)), "trial": True,
                          "dataset_sha256": policy.dataset_sha256, "warmup_gops": 4, "warmup_action": 4},
                "detector": {"xmodel": "models/yolov5n/yolov5_nano_pt.xmodel", "input_rgb": [640, 640],
                             "confidence": .25, "iou": .45, "max_detections": 32,
                             "board_preprocessing": {"scaler": "v4l2convert", "scaler_rgb": [1280, 720],
                                 "arm_resize_rgb": [640, 360], "arm_resize": "OpenCV INTER_LINEAR",
                                 "letterbox_rgb": [640, 640], "pad_top_bottom": 140, "pad_value": 114},
                             "weights_equivalence_verified": False, "vendor_source_sha256": sha256(origin)},
                "strategies": ["cbr", "vbr", "fixed_roi", "rl"], "baseline_bitrate_mbps": 1.5,
                "fixed_roi_delta_qp": -6, "reward": config["reward"], "files": files}
    write_json(ROOT / "manifest.json", manifest)
    print(json.dumps({"bundle": str(ROOT), "video_bytes": raw.stat().st_size, "trace_id": trace_id,
                      "trial_policy": True, "ready_for_hardware_preflight": True}))


if __name__ == "__main__":
    main()
