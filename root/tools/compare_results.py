#!/usr/bin/env python3
"""Host metrics: fixed FP32 reference ROI, existing rl definitions, no metric substitutions."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_runtime.common import CLASS_WEIGHTS, FRAMES, GOP, STRATEGIES, bundle_manifest, write_json, sha256
from host_metrics import _vmaf_command, _read_vmaf, _roi_metrics, reward_for, detection_match, fluid_queue


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_stream(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                            "-show_packets", "-show_entries", "frame=key_frame,pict_type:packet=size",
                            "-of", "json", str(path)], check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    mixed = info.get("packets_and_frames", [])
    frames = info.get("frames", [r for r in mixed if r.get("type") == "frame"])
    packets = info.get("packets", [r for r in mixed if r.get("type") == "packet"])
    if len(frames) != FRAMES or len(packets) != FRAMES:
        raise RuntimeError("bitstream requires exactly 960 frames/packets: " + str(path))
    for index, frame in enumerate(frames):
        if (int(frame["key_frame"]) != int(index % GOP == 0)
                or frame["pict_type"] != ("I" if index % GOP == 0 else "P")):
            raise RuntimeError("wrong keyframe/frame type: " + str(path))
    return [sum(int(r["size"]) for r in packets[k*GOP:(k+1)*GOP]) for k in range(FRAMES//GOP)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--board-run", type=Path, help="results/<run_id> copied from board")
    mode.add_argument("--offline-only", action="store_true", help="verify real offline streams/metrics without board results")
    parser.add_argument("--sample-stride", type=int, default=5, choices=(1, 5))
    args = parser.parse_args()
    manifest = bundle_manifest(ROOT)
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], check=True, capture_output=True, text=True)
    if "libvmaf" not in filters.stdout:
        raise RuntimeError("FFmpeg libvmaf is mandatory")
    golden_dir = ROOT / "results/offline_reference"
    golden_manifest = json.loads((golden_dir / "manifest.json").read_text())
    for path in (manifest["video"]["path"], "bw_trace/test_segment.json", "models/rl/ddqn_policy.npz"):
        if golden_manifest["bundle"]["files"][path]["sha256"] != manifest["files"][path]["sha256"]:
            raise RuntimeError("offline reference belongs to a different bundle")
    reference_rows = jsonl(golden_dir / "frames.jsonl")
    if [r["frame_idx"] for r in reference_rows] != list(range(FRAMES)):
        raise RuntimeError("golden detector frame coverage mismatch")
    detections = []
    for row in reference_rows:
        for x1, y1, x2, y2, conf, cls in row["detections"]:
            detections.append({"frame_idx": row["frame_idx"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "confidence": conf, "class_weight": CLASS_WEIGHTS[int(cls)]})
    detection_df = pd.DataFrame(detections, columns=["frame_idx", "x1", "y1", "x2", "y2", "confidence", "class_weight"])
    target = (golden_dir if args.offline_only else args.board_run.resolve()) / "comparison"
    target.mkdir(parents=True, exist_ok=True)
    reference = target / "reference.mkv"
    if reference.exists():
        raise RuntimeError("comparison reference already exists; choose a new comparison run")
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "nv12",
                    "-video_size", "1920x1080", "-framerate", "30", "-colorspace", "bt709",
                    "-i", str(ROOT/manifest["video"]["path"]), "-c:v", "ffv1", "-threads", "2", "-pix_fmt", "yuv420p",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(reference)], check=True)
    summaries, all_gops, detector_rows = [], [], []
    jobs = []
    for strategy in STRATEGIES:
        jobs.append(("offline", strategy, golden_dir/strategy))
        if args.offline_only:
            continue
        board_jobs = sorted((args.board_run/strategy).glob("repeat_*"))
        if not board_jobs:
            raise RuntimeError("missing board strategy: " + strategy)
        for directory in board_jobs:
            receipt = json.loads((directory/"manifest.json").read_text())
            for path in (manifest["video"]["path"], "bw_trace/test_segment.json", "models/rl/ddqn_policy.npz",
                         manifest["detector"]["xmodel"]):
                if receipt["bundle"]["files"][path]["sha256"] != manifest["files"][path]["sha256"]:
                    raise RuntimeError("board/offline bundle checksum mismatch: " + path)
            jobs.append(("board", strategy, directory))
    offline_decisions = {s: jsonl(golden_dir/s/"gops.jsonl") for s in STRATEGIES}
    for platform, strategy, directory in jobs:
        label = platform+"_"+strategy+"_"+directory.name
        print("Measuring " + label, flush=True)
        stream = directory / "output.h264"
        logged = jsonl(directory / "gops.jsonl")
        sizes = verify_stream(stream)
        if [r["gop_idx"] for r in logged] != list(range(FRAMES//GOP)):
            raise RuntimeError("GOP log coverage mismatch")
        if platform == "board":
            receipt = json.loads((directory/"summary.json").read_text())
            if receipt["output_sha256"] != sha256(stream):
                raise RuntimeError("board bitstream checksum mismatch")
            board_frames = jsonl(directory/"frames.jsonl")
            if [r["frame_idx"] for r in board_frames] != list(range(FRAMES)):
                raise RuntimeError("board frame coverage mismatch")
            for ref, board in zip(reference_rows, board_frames):
                match = detection_match(np.asarray(ref["detections"]).reshape(-1, 6),
                                        np.asarray(board["detections"]).reshape(-1, 6))
                match.pop("matches")
                detector_rows.append({"run": label, "frame_idx": ref["frame_idx"], **match})
        log = target / (label+".vmaf.json")
        subprocess.run([str(x) for x in _vmaf_command(stream, reference, log, sample_stride=args.sample_stride)], check=True)
        quality = _read_vmaf(log)
        roi = _roi_metrics(stream, reference, detection_df, FRAMES, sample_stride=args.sample_stride)
        expected_samples = list(range(0, FRAMES, args.sample_stride))
        if quality.frame_idx.tolist() != expected_samples or roi.frame_idx.tolist() != expected_samples:
            raise RuntimeError("quality/ROI metric sample coverage mismatch")
        if not np.isfinite(quality[["vmaf", "psnr", "ssim"]].to_numpy()).all():
            raise RuntimeError("non-finite required quality metric")
        if roi.loc[roi.has_roi, "roi_ssim"].isna().any():
            raise RuntimeError("reference ROI exists but ROI SSIM is missing")
        frame_metrics = quality.merge(roi, on="frame_idx", validate="one_to_one")
        frame_metrics["gop_idx"] = frame_metrics.frame_idx // GOP
        frame_metrics.to_csv(target / (label+".frames.csv"), index=False)
        previous = None
        per_gop = []
        for k, item in enumerate(logged):
            if item["encoded_bytes"] != sizes[k]:
                raise RuntimeError("logged/parsed H.264 payload size mismatch")
            action = int(item["action_id"])
            metrics = frame_metrics[frame_metrics.gop_idx == k]
            row = pd.Series({**item, "bitrate_index": action//4, "roi_index": action%4,
                             "vmaf": float(metrics.vmaf.mean()), "psnr": float(metrics.psnr.mean()),
                             "ssim": float(metrics.ssim.mean()), "roi_frame_count": int(metrics.has_roi.sum()),
                             "roi_ssim": float(metrics.roi_ssim.mean()) if metrics.has_roi.any() else 0.})
            # Warm-up rewards are omitted; scored reward uses actual previous action.
            parts = reward_for(row, float(item["bandwidth_mbps"]), previous, manifest["reward"]) if k >= 4 else {"reward": None}
            per_gop.append({"run": label, "platform": platform, "strategy": strategy, "gop_idx": k,
                            "action_id": action, "actual_bitrate_mbps": item["actual_bitrate_mbps"],
                            "target_bitrate_mbps": item["target_bitrate_mbps"], "delta_qp_roi": item["delta_qp_roi"],
                            "bandwidth_mbps": item["bandwidth_mbps"], "semantic_score": item["semantic_score"],
                            "vmaf": row.vmaf, "psnr": row.psnr, "ssim": row.ssim,
                            "roi_ssim": row.roi_ssim if row.roi_frame_count else None,
                            "action_matches_offline": action == offline_decisions[strategy][k]["action_id"], **parts})
            previous = row
        table = pd.DataFrame(per_gop)
        arrivals, capacity = table.actual_bitrate_mbps.to_numpy(), table.bandwidth_mbps.to_numpy()
        queue = fluid_queue(arrivals, capacity)
        table["backlog_mbit"] = queue["backlog_mbit"]
        table["network_deadline_miss"] = queue["deadline_miss"]
        all_gops.extend(table.to_dict("records"))
        for name, selected in (("whole_clip", table), ("gops_4_31", table.iloc[4:])):
            summaries.append({"run": label, "platform": platform, "strategy": strategy, "window": name,
                              "mean_bitrate_mbps": selected.actual_bitrate_mbps.mean(),
                              "mean_vmaf": selected.vmaf.mean(), "mean_roi_ssim": selected.roi_ssim.mean(),
                              "mean_psnr": selected.psnr.mean(), "mean_ssim": selected.ssim.mean(),
                              "return": selected.reward.sum(min_count=1),
                              "bandwidth_violation_rate": (selected.actual_bitrate_mbps > selected.bandwidth_mbps).mean(),
                              "action_agreement": selected.action_matches_offline.mean(),
                              "network_deadline_miss_rate": selected.network_deadline_miss.mean()})
    pd.DataFrame(all_gops).to_csv(target/"gop_comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(target/"strategy_summary.csv", index=False)
    pd.DataFrame(detector_rows).to_csv(target/"detector_comparison.csv", index=False)
    write_json(target/"report.json", {"status": "validated_offline_only" if args.offline_only else "passed",
               "board_run_verified": not args.offline_only, "runs": len(jobs), "quality_sample_stride": args.sample_stride,
               "main_window": "GOP 4..31", "policy_trial": True, "weights_equivalence_verified": False,
               "detector_labels": "FP32 pseudo labels, not human ground truth",
               "network_metrics": "fluid replay of prerecorded BW, not real network transmission"})
    print("Comparison complete: " + str(target))


if __name__ == "__main__":
    main()
