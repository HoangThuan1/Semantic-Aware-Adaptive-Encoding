from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .common import ROOT, project_path, run, sha256, stable_hash, write_json
from .video import count_frames


def action_id(bitrate_index: int, roi_index: int, roi_count: int) -> int:
    return bitrate_index * roi_count + roi_index


def _encode_one(video, roi_path: Path, roi_hash: str, video_dir: Path,
                bitrate_index: int, roi_index: int, bitrate_mbps: float,
                delta_qp: int, config: dict) -> dict:
    encoder = config["encoder"]
    current_action = action_id(bitrate_index, roi_index, len(encoder["roi_delta_qp"]))
    delta_label = f"m{abs(delta_qp)}" if delta_qp < 0 else "0"
    output = video_dir / f"action_{current_action:02d}_b{bitrate_mbps:g}_q{delta_label}.mp4"
    sidecar = output.with_suffix(".json")
    job = {
        "video_id": video.video_id, "reference_sha256": video.reference_sha256,
        "roi_sha256": roi_hash, "action_id": current_action,
        "bitrate_mbps": float(bitrate_mbps), "delta_qp": int(delta_qp),
        "encoder": encoder, "gop_frames": config["video"]["gop_frames"],
    }
    job_hash = stable_hash(job)
    complete = False
    if output.is_file() and sidecar.is_file():
        existing = json.loads(sidecar.read_text())
        if existing.get("job_hash") != job_hash:
            raise RuntimeError(f"incompatible existing encode: {output}")
        if count_frames(output) != video.frame_count:
            raise RuntimeError(f"existing encode has wrong frame count: {output}")
        complete = True
    if not complete:
        output.unlink(missing_ok=True)
        temporary = output.with_name(output.stem + ".tmp.mp4")
        temporary.unlink(missing_ok=True)
        run([
            "roi_encoder", ROOT / video.reference_path, roi_path, temporary,
            str(int(round(float(bitrate_mbps) * 1000))), str(delta_qp),
            str(config["video"]["gop_frames"]),
        ])
        actual_frames = count_frames(temporary)
        if actual_frames != video.frame_count:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"encoded frame mismatch: {actual_frames} != {video.frame_count}")
        temporary.replace(output)
        write_json(sidecar, {**job, "job_hash": job_hash, "output_sha256": sha256(output)})
    return {
        "video_id": video.video_id, "action_id": current_action,
        "bitrate_index": bitrate_index, "roi_index": roi_index,
        "target_bitrate_mbps": float(bitrate_mbps), "delta_qp_roi": int(delta_qp),
        "encoded_path": str(output.relative_to(ROOT)), "frame_count": video.frame_count,
        "gop_count": video.gop_count, "encoded_sha256": sha256(output),
    }


def encode_grid(config: dict) -> None:
    artifacts = project_path(config, "artifacts")
    reference_manifest_path = artifacts / "reference" / "manifest.parquet"
    semantics_path = artifacts / "metadata" / "gop_semantics.parquet"
    if not reference_manifest_path.is_file() or not semantics_path.is_file():
        raise RuntimeError("reference or metadata missing; run normalize and infer first")
    reference_manifest = pd.read_parquet(reference_manifest_path)
    encoder = config["encoder"]
    bitrates = encoder["bitrates_mbps"]
    deltas = encoder["roi_delta_qp"]
    output_root = artifacts / "encodes"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    action_rows = [
        {
            "action_id": action_id(bitrate_index, roi_index, len(deltas)),
            "bitrate_index": bitrate_index, "roi_index": roi_index,
            "target_bitrate_mbps": float(bitrate_mbps), "delta_qp_roi": int(delta_qp),
        }
        for bitrate_index, bitrate_mbps in enumerate(bitrates)
        for roi_index, delta_qp in enumerate(deltas)
    ]
    pd.DataFrame(action_rows).sort_values("action_id").to_parquet(
        output_root / "action_space.parquet", index=False
    )

    for video in reference_manifest.itertuples(index=False):
        roi_path = artifacts / "metadata" / f"{video.video_id}.roi.csv"
        if not roi_path.is_file():
            raise RuntimeError(f"ROI schedule missing: {roi_path}")
        video_dir = output_root / video.video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        roi_hash = sha256(roi_path)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_encode_one, video, roi_path, roi_hash, video_dir,
                                   bitrate_index, roi_index, bitrate_mbps, delta_qp, config)
                       for bitrate_index, bitrate_mbps in enumerate(bitrates)
                       for roi_index, delta_qp in enumerate(deltas)]
            for future in as_completed(futures):
                rows.append(future.result())
    pd.DataFrame(rows).sort_values(["video_id", "action_id"]).to_parquet(
        output_root / "manifest.parquet", index=False
    )
