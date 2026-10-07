#!/usr/bin/env python3
"""Host CUDA YOLO + PyTorch/NumPy replay + persistent x264 baselines."""
import argparse
import csv
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_runtime.common import (CLASS_WEIGHTS, FPS, FRAMES, GOP, HEIGHT, WIDTH, STRATEGIES,
                               bundle_manifest, decision, read_trace, roi_boxes, semantic, sha256, write_json)
from rl_runtime.policy_numpy import NumpyPolicy


def read_exact(pipe, count):
    result = bytearray()
    while len(result) < count:
        chunk = pipe.read(count-len(result))
        if not chunk:
            raise RuntimeError("short FFmpeg decoded reference")
        result.extend(chunk)
    return result


def detector_reference(manifest, directory):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for offline YOLO; no CPU fallback")
    yolo_root = Path(os.environ["YOLOV5_ROOT"])
    if not yolo_root.is_dir() or os.environ.get("YOLOv5_AUTOINSTALL", "").lower() != "false":
        raise RuntimeError("pinned YOLOv5 source/autoinstall contract missing")
    revision = subprocess.check_output(["git", "-C", str(yolo_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != "915bbf294bb74c859f0b41f1c23bc395014ea679":
        raise RuntimeError("unexpected YOLOv5 source revision")
    sys.path.insert(0, str(yolo_root))
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression
    model = DetectMultiBackend(str(ROOT / "models/yolov5n/offline_yolov5n.pt"), device=torch.device("cuda:0"), fp16=False, fuse=True)
    source = ROOT / manifest["video"]["path"]
    command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "nv12", "-video_size", "1920x1080",
               "-framerate", "30", "-colorspace", "bt709", "-i", str(source), "-vf",
               "scale=in_color_matrix=bt709:out_color_matrix=bt709,format=rgb24", "-f", "rawvideo", "pipe:1"]
    rows = []
    with (directory / "decode.log").open("w") as error:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error)
        try:
            with torch.inference_mode():
                model(torch.zeros((1, 3, 640, 640), device="cuda"))
                for frame in range(FRAMES):
                    rgb = np.frombuffer(read_exact(process.stdout, WIDTH*HEIGHT*3), dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
                    image = np.full((640, 640, 3), 114, dtype=np.uint8)
                    image[140:500] = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
                    inp = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to("cuda").float()[None]/255
                    pred = non_max_suppression(model(inp), conf_thres=.25, iou_thres=.45,
                                              classes=sorted(CLASS_WEIGHTS), max_det=32)[0].cpu().numpy()
                    pred[:, [0, 2]] *= 3
                    pred[:, [1, 3]] = (pred[:, [1, 3]]-140)*3
                    pred[:, [0, 2]] = np.clip(pred[:, [0, 2]], 0, WIDTH)
                    pred[:, [1, 3]] = np.clip(pred[:, [1, 3]], 0, HEIGHT)
                    rows.append({"frame_idx": frame, "gop_idx": frame//GOP, "pts_ns": frame*1_000_000_000//FPS,
                                 "detections": pred.tolist(), "roi": roi_boxes(pred), "semantic_score": semantic(pred)})
            if process.stdout.read(1) or process.wait(timeout=30) != 0:
                raise RuntimeError("reference decode coverage/error")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True, help="native host offline_encoder executable")
    args = parser.parse_args()
    manifest = bundle_manifest(ROOT)
    destination = ROOT / "results/offline_reference"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise RuntimeError("offline_reference already complete; do not overwrite")
    policy = NumpyPolicy(ROOT / "models/rl/ddqn_policy.npz")
    bandwidth = read_trace(ROOT, manifest)
    detections = detector_reference(manifest, destination)
    with (destination / "frames.jsonl").open("w") as handle:
        for row in detections:
            handle.write(json.dumps(row)+"\n")
    roi_csv = destination / "roi.csv"
    with roi_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "x1", "y1", "x2", "y2"])
        for row in detections:
            for box in row["roi"]:
                writer.writerow([row["frame_idx"], box["x"], box["y"], box["x"]+box["width"], box["y"]+box["height"]])
    # Independently execute the exported weights through PyTorch, not NumPy itself.
    layers = []
    for i, (weight, bias) in enumerate(zip(policy.weights, policy.biases)):
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.load_state_dict({"weight": torch.from_numpy(weight), "bias": torch.from_numpy(bias)})
        layers.append(layer)
        if i < 2:
            layers.append(torch.nn.ReLU())
    network = torch.nn.Sequential(*layers).eval()
    golden = []
    for strategy in STRATEGIES:
        out = destination / strategy
        out.mkdir(exist_ok=True)
        if (out / "output.h264").exists():
            raise RuntimeError("partial offline output exists: inspect before retry")
        rows, previous = [], 0.
        with (out / "encoder.log").open("w") as err:
            process = subprocess.Popen([str(args.encoder.resolve()), str(ROOT/manifest["video"]["path"]),
                                        str(roi_csv), str(out/"output.h264"), "cbr" if strategy == "cbr" else "vbr"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    for k in range(FRAMES//GOP):
                        score = float(np.mean([r["semantic_score"] for r in detections[k*GOP:(k+1)*GOP]]))
                        selected = decision(strategy, k, bandwidth, previous, score, policy)
                        if selected["raw_state"] is not None:
                            normalized = np.asarray(selected["normalized_state"], dtype=np.float32)
                            with torch.inference_mode():
                                q = network(torch.from_numpy(normalized)).numpy()
                            if not np.allclose(q, selected["q_values"], atol=1e-5, rtol=1e-5) or int(q.argmax()) != selected["action_id"]:
                                raise RuntimeError("PyTorch/NumPy policy mismatch")
                            golden.append({"gop_idx": k, "raw_state": selected["raw_state"], "q_values": q.tolist(),
                                           "action_id": int(q.argmax()), "source": "PyTorch FP32 exported weights"})
                        process.stdin.write("%d %d\n" % (round(selected["target_bitrate_mbps"]*1000), selected["delta_qp_roi"]))
                        process.stdin.flush()
                        if not selector.select(120):
                            raise RuntimeError("offline encoder GOP timeout")
                        line = process.stdout.readline()
                        if not line.strip().isdigit():
                            raise RuntimeError("offline encoder returned no GOP receipt; see " + str(out/"encoder.log"))
                        size = int(line)
                        if size <= 0:
                            raise RuntimeError("empty encoded GOP")
                        selected.update({"gop_idx": k, "frames": GOP, "encoded_bytes": size,
                                         "actual_bitrate_mbps": size*8/1e6, "bandwidth_mbps": float(bandwidth[k]),
                                         "previous_actual_bitrate_mbps": previous, "evaluation": k >= 4})
                        previous = selected["actual_bitrate_mbps"]
                        rows.append(selected)
                process.stdin.close()
                if process.wait(timeout=30) != 0:
                    raise RuntimeError("offline x264 encoder failed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            with (out / "gops.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row)+"\n")
    write_json(ROOT / "models/rl/golden_policy.json", golden)
    write_json(destination / "manifest.json", {"bundle": manifest, "detector": "YOLOv5n CUDA FP32",
               "preprocess": "BT709 RGB, 640x360 bilinear resize, square640 centered pad114",
               "encoder": "persistent single-thread libx264, dynamic bitrate/ROI, not bit-exact VCU",
               "encoder_binary_sha256": sha256(args.encoder),
               "offline_cbr": "HRD CBR, filler enabled, fixed 1500 kbit/s",
               "board_cbr": "VCU constant rate, filler disabled, fixed 1500 kbit/s",
               "weights_equivalence_verified": False})
    # Add independent replay receipt, without changing hashes for packaged code or data.
    manifest["files"]["models/rl/golden_policy.json"] = {"bytes": (ROOT/"models/rl/golden_policy.json").stat().st_size,
                                                       "sha256": sha256(ROOT/"models/rl/golden_policy.json")}
    write_json(ROOT / "manifest.json", manifest)
    print("Offline references and independent PyTorch golden replay ready: " + str(destination))


if __name__ == "__main__":
    main()
