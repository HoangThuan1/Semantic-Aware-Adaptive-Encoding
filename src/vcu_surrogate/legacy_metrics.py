"""Explicitly migrate unchanged legacy video outcomes into a new dataset identity."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from .common import ROOT, project_path, sha256, stable_hash, write_json


def reuse_legacy_metrics(config, reference, encodes):
    video_id = str(reference.video_id)
    current = project_path(config, "artifacts")
    old = (ROOT / "reports/benchmark/legacy_artifacts" if current == ROOT / "artifacts"
           else ROOT / "artifacts")
    if video_id.startswith("bdd_") or not (old / "metrics" / video_id).is_dir():
        return 0
    original = json.loads((old / "reference" / f"{video_id}.json").read_text())
    normalized = json.loads((current / "reference" / f"{video_id}.json").read_text())
    if any(original[k] != normalized[k] for k in ("source_sha256", "video", "keep_frames")):
        return 0
    original_detection = pd.read_parquet(old / "metadata" / f"{video_id}.detections.parquet")
    current_detection = pd.read_parquet(current / "metadata" / f"{video_id}.detections.parquet")
    if not original_detection.equals(current_detection):
        return 0
    reused = 0
    metrics = ["vmaf", "psnr_y", "float_ssim", "roi_psnr", "roi_ssim"]
    for encode in encodes.itertuples(index=False):
        prefix = f"action_{encode.action_id:02d}"
        destination = current / "metrics" / video_id / prefix
        if all(destination.with_suffix(suffix).is_file() for suffix in
               (".json", ".frames.parquet", ".gops.parquet")):
            continue
        source = old / "metrics" / video_id / prefix
        meta = json.loads(source.with_suffix(".json").read_text())
        if (meta["encoded_sha256"] != encode.encoded_sha256 or meta["metrics"] != metrics
                or meta["reference_sha256"] != original["output_sha256"]):
            continue
        if stable_hash({k: v for k, v in meta.items() if k != "job_hash"}) != meta["job_hash"]:
            raise RuntimeError(f"legacy metric job hash mismatch: {video_id}/{prefix}")
        if sha256(ROOT / encode.encoded_path) != encode.encoded_sha256:
            raise RuntimeError(f"encode checksum mismatch: {video_id}/{prefix}")
        frames_path, gops_path = source.with_suffix(".frames.parquet"), source.with_suffix(".gops.parquet")
        frames, gops = pd.read_parquet(frames_path), pd.read_parquet(gops_path)
        if (frames.frame_idx.tolist() != list(range(reference.frame_count))
                or gops.gop_idx.tolist() != list(range(reference.gop_count))
                or set(gops.action_id) != {encode.action_id}
                or set(gops.video_id) != {video_id}
                or not np.isfinite(gops[["vmaf", "psnr", "ssim", "actual_bitrate_mbps"]].to_numpy()).all()):
            raise RuntimeError(f"incomplete legacy metrics: {video_id}/{prefix}")
        destination = current / "metrics" / video_id / prefix
        destination.parent.mkdir(parents=True, exist_ok=True)
        job = {"dataset_sha256": reference.dataset_sha256, "encoded_sha256": encode.encoded_sha256,
               "reference_sha256": reference.reference_sha256, "metrics": metrics}
        shutil.copy2(frames_path, destination.with_suffix(".frames.parquet"))
        shutil.copy2(gops_path, destination.with_suffix(".gops.parquet"))
        write_json(destination.with_suffix(".json"), {**job, "job_hash": stable_hash(job),
                   "reused_from": str(source.relative_to(ROOT)), "legacy_job_hash": meta["job_hash"],
                   "legacy_reference_sha256": meta["reference_sha256"],
                   "equivalence": "same source checksum, normalization settings and detections; identical encoded checksum",
                   "legacy_frames_sha256": sha256(frames_path), "legacy_gops_sha256": sha256(gops_path)})
        reused += 1
    print(f"LEGACY METRICS {video_id}: reused {reused}/{len(encodes)} checksum-matching actions", flush=True)
    return reused
