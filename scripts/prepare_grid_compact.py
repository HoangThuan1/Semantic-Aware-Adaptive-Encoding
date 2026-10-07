"""Measure each video's complete grid while retaining its FFV1 reference.

Source videos, manifests, encodes, metrics and benchmark caches are retained.
The regular metrics/benchmark stages subsequently aggregate these checked caches.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import pandas as pd

from vcu_surrogate.benchmark import measure_grid
from vcu_surrogate.common import ROOT, load_config, project_path, sha256, stable_hash, write_json
from vcu_surrogate.dataset import dataset_hash
from vcu_surrogate.encoding import encode_grid
from vcu_surrogate.metrics import (_metric_one, build_reference_yuv, build_reference_bgr,
                                   cached_metric, mixed_metric_results)
from vcu_surrogate.legacy_metrics import reuse_legacy_metrics
from vcu_surrogate.video import normalize_videos


def choose_batch(references, config, required=()):
    """Schedule all pending videos; prioritize existing encodes, without a size cap."""
    ordered = sorted(references, key=lambda r: (r.video_id not in required,
                     not str(r.video_id).startswith("bdd_"), r.video_id))
    return [str(r.video_id) for r in ordered], None


def ensure_reference(config, record):
    path = ROOT / record.reference_path
    if path.exists():
        return record
    artifacts = project_path(config, "artifacts")
    provenance = json.loads(path.with_suffix(".json").read_text())
    if (sha256(ROOT / record.source_path) != record.source_sha256
            or provenance["source_sha256"] != record.source_sha256
            or provenance["video"] != config["video"]
            or provenance["keep_frames"] != record.frame_count):
        raise RuntimeError("cannot regenerate reference: source/normalization provenance mismatch")
    # Only unencoded videos may be regenerated. Completed cache inputs stay immutable.
    if list((artifacts / "encodes" / record.video_id).glob("action_*.json")):
        raise RuntimeError("missing reference for already encoded video; refusing silent cache rebinding")
    normalize_videos(config, video_ids=[record.video_id])
    manifest = pd.read_parquet(artifacts / "reference/manifest.parquet")
    regenerated = next(manifest[manifest.video_id == record.video_id].itertuples(index=False))
    sidecar = artifacts / "metadata" / f"{record.video_id}.json"
    metadata = json.loads(sidecar.read_text())
    if metadata["reference_sha256"] != record.reference_sha256 or metadata["yolo"] != config["yolo"]:
        raise RuntimeError("reference/inference provenance mismatch")
    metadata["regenerated_from_reference_sha256"] = record.reference_sha256
    metadata["reference_sha256"] = regenerated.reference_sha256
    metadata["job_hash"] = stable_hash({k: metadata[k] for k in
        ("dataset_sha256", "reference_sha256", "weights_sha256", "yolo")})
    write_json(sidecar, metadata)
    return regenerated


def metric_reference(record):
    # pandas' dynamically generated itertuples class cannot cross a process queue.
    return SimpleNamespace(**record._asdict())


def checked_reference_cache(reference, raw_cache, bgr_cache, receipt):
    """Reuse a complete shared decode only with matching source and cache checksums."""
    if not receipt or not raw_cache.is_file() or not bgr_cache.is_file():
        return False
    if (receipt.get("video_id") != reference.video_id
            or receipt.get("reference_sha256") != reference.reference_sha256
            or receipt.get("frame_count") != int(reference.frame_count)
            or raw_cache.stat().st_size != int(reference.frame_count) * 3110400):
        return False
    return (sha256(raw_cache) == receipt.get("yuv_sha256")
            and sha256(bgr_cache) == receipt.get("bgr_sha256"))


def release_reference(reference, artifact_root, benchmark_root, identity, benchmark_required=True):
    """Verify results, retain FFV1, and remove disposable metric/cache files."""
    video_id = str(reference.video_id)
    encodes = pd.read_parquet(artifact_root / "encodes" / video_id / "manifest.parquet")
    if len(encodes) != 32 or set(encodes.action_id) != set(range(32)):
        raise RuntimeError(f"incomplete encode grid for {video_id}")
    for action in range(32):
        metric = artifact_root / "metrics" / video_id / f"action_{action:02d}"
        meta = json.loads(metric.with_suffix(".json").read_text())
        if meta.get("dataset_sha256") != identity or meta.get("reference_sha256") != reference.reference_sha256:
            raise RuntimeError(f"metric provenance mismatch for {video_id}/{action}")
        gops = pd.read_parquet(metric.with_suffix(".gops.parquet"))
        frames = pd.read_parquet(metric.with_suffix(".frames.parquet"), columns=["frame_idx"])
        stride = meta.get("metric_sample_stride", 1)
        if (len(gops) != reference.gop_count
                or frames.frame_idx.tolist() != list(range(0, reference.frame_count, stride))
                or (stride > 1 and not (gops.metric_sample_count == 30 // stride).all())):
            raise RuntimeError(f"incomplete metric grid for {video_id}/{action}")
        if benchmark_required:
            cache = benchmark_root / "cache" / video_id / f"action_{action:02d}.parquet"
            classes = cache.with_name(cache.stem + ".classes.parquet")
            evidence = json.loads(cache.with_suffix(".json").read_text())
            if (evidence.get("output_sha256") != sha256(cache)
                    or evidence.get("classes_sha256") != sha256(classes)
                    or evidence.get("sample_frames") != reference.gop_count * 3):
                raise RuntimeError(f"incomplete benchmark cache for {video_id}/{action}")
    path = ROOT / reference.reference_path
    if path.exists() and sha256(path) != reference.reference_sha256:
        raise RuntimeError(f"reference provenance mismatch for {video_id}")
    receipt = artifact_root / "reference" / f"{video_id}.released.json"
    write_json(receipt, {"dataset_sha256": identity, "reference_sha256": reference.reference_sha256,
                         "video_id": video_id, "reference_retained": path.exists(),
                         "benchmark_verified": benchmark_required,
                         "reason": "all 32 metrics verified; FFV1 retained; benchmark deferred" if not benchmark_required
                                   else "all 32 metrics and benchmark caches verified; FFV1 retained"})
    released = 0
    # Per-frame parquet and GOP parquet retain all VMAF measurements.
    for log in (artifact_root / "metrics" / video_id).glob("*.vmaf.json"):
        released += log.stat().st_size
        log.unlink()
    frame_cache = benchmark_root / "cache" / f"{video_id}.reference.npy"
    if frame_cache.exists():
        released += frame_cache.stat().st_size
        frame_cache.unlink()
    return released


def benchmark_ready_pairs(config, references, progress, progress_path, flush=False):
    """Benchmark two metric-complete videos at a time; preserve an interrupted pair."""
    artifacts = project_path(config, "artifacts")
    benchmark = project_path(config, "reports") / "benchmark"
    current_references = pd.read_parquet(artifacts / "reference/manifest.parquet")
    by_id = {str(r.video_id): r for r in current_references.itertuples(index=False)}
    identity = progress["dataset_sha256"]
    while True:
        batch = progress.get("benchmark_batch_video_ids", [])
        if not batch:
            pending = []
            for vid in progress["completed_videos"]:
                receipt = artifacts / "reference" / f"{vid}.released.json"
                if receipt.exists() and not json.loads(receipt.read_text()).get("benchmark_verified", True):
                    pending.append(vid)
            progress["benchmark_pending_video_ids"] = pending
            if len(pending) < 2 and not (flush and pending):
                write_json(progress_path, progress)
                return
            batch = pending[:2]
            progress["benchmark_batch_video_ids"] = batch
            write_json(progress_path, progress)
        progress["substage"] = "benchmark_cache"
        for vid in list(batch):
            reference = by_id[vid]
            progress["benchmark_current_video"] = vid
            write_json(progress_path, progress)
            measure_grid(config, benchmark, 3, 8, video_ids=[vid],
                         encode_manifest=artifacts / "encodes" / vid / "manifest.parquet")
            progress["released_bytes"] += release_reference(reference, artifacts, benchmark, identity)
            progress.setdefault("benchmark_completed_videos", [])
            if vid not in progress["benchmark_completed_videos"]:
                progress["benchmark_completed_videos"].append(vid)
            progress["benchmark_batch_video_ids"].remove(vid)
            write_json(progress_path, progress)
            print(f"BENCHMARK PAIR DONE {vid}", flush=True)
        progress.pop("benchmark_current_video", None)


def main():
    config = load_config()
    artifacts = project_path(config, "artifacts")
    reports = project_path(config, "reports")
    benchmark = reports / "benchmark"
    references = pd.read_parquet(artifacts / "reference/manifest.parquet")
    detections = pd.read_parquet(artifacts / "metadata/detections.parquet")
    identity = dataset_hash(config)
    stride = config.get("metrics", {}).get("sample_stride", 1)
    progress_path = reports / "compact_progress.json"
    progress = {"dataset_sha256": identity, "completed_videos": [], "released_bytes": 0}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("dataset_sha256") != identity:
            raise RuntimeError("compact preparation dataset mismatch")
    records = list(references.itertuples(index=False))
    # Recover an interrupted cleanup from checked metric/cache receipts, without re-encoding.
    for record in records:
        receipt = artifacts / "reference" / f"{record.video_id}.released.json"
        if receipt.exists() and record.video_id not in progress["completed_videos"]:
            evidence = json.loads(receipt.read_text())
            if evidence.get("dataset_sha256") != identity or evidence.get("reference_sha256") != record.reference_sha256:
                raise RuntimeError("cleanup receipt provenance mismatch")
            progress["released_bytes"] += release_reference(record, artifacts, benchmark, identity,
                                                           benchmark_required=evidence.get("benchmark_verified", True))
            progress["completed_videos"].append(record.video_id)
            write_json(progress_path, progress)
    pending = [r for r in records if r.video_id not in progress["completed_videos"]]
    required = [r.video_id for r in pending if list((artifacts / "encodes" / r.video_id).glob("action_*.json"))]
    batch_ids, _ = choose_batch(pending, config, required)
    current = progress.get("current_video")
    if current in batch_ids and current in required:
        batch_ids.remove(current)
        batch_ids.insert(0, current)
    progress.update(active_batch_ids=batch_ids, batch_reserved_bytes=None,
                    batch_limit_bytes=None, awaiting_compaction=False,
                    retain_ffv1_references=True, benchmark_deferred=False, benchmark_every_videos=2)
    write_json(progress_path, progress)
    benchmark_ready_pairs(config, records, progress, progress_path)
    manifests = []
    by_id = {r.video_id: r for r in pending}
    ordered = [by_id[vid] for vid in batch_ids]
    # Finish metrics and benchmark for one video before preparing the next.
    for reference in ordered:
        reference = ensure_reference(config, reference)
        video_id = str(reference.video_id)
        progress.update(current_video=video_id, substage="encode")
        write_json(progress_path, progress)
        print(f"COMPACT START {video_id}", flush=True)
        encode_grid(config, video_ids=[video_id])
        encodes = pd.read_parquet(artifacts / "encodes" / video_id / "manifest.parquet")
        manifests.append(encodes)
        reuse_legacy_metrics(config, reference, encodes)
        selected = detections[detections.video_id == video_id]
        # Migrate verified full-frame caches before deciding whether a decode is needed.
        for row in encodes.itertuples(index=False):
            cached_metric(SimpleNamespace(**row._asdict()), metric_reference(reference),
                          artifacts / "metrics", 30, 30, stride)
        progress.update(substage="metrics", metric_actions_completed=0)
        write_json(progress_path, progress)
        raw_cache = artifacts / "reference" / f"{video_id}.frames.yuv"
        bgr_cache = benchmark / "cache" / f"{video_id}.reference.npy"
        needs_metrics = any(not all((artifacts / "metrics" / video_id / f"action_{r.action_id:02d}{suffix}").is_file()
                            for suffix in (".json", ".frames.parquet", ".gops.parquet"))
                            for r in encodes.itertuples(index=False))
        try:
            if needs_metrics:
                progress["substage"] = "decode_reference_cache"
                write_json(progress_path, progress)
                if checked_reference_cache(reference, raw_cache, bgr_cache, progress.get("reference_cache")):
                    if sha256(ROOT / reference.reference_path) != reference.reference_sha256:
                        raise RuntimeError("reference checksum mismatch before shared cache reuse")
                    print(f"REFERENCE CACHE VERIFIED {video_id}; using existing YUV/BGR", flush=True)
                else:
                    build_reference_yuv(ROOT / reference.reference_path, reference.reference_sha256,
                                        int(reference.frame_count), raw_cache)
                    build_reference_bgr(ROOT / reference.reference_path, reference.reference_sha256,
                                        int(reference.frame_count), bgr_cache, raw_cache, stride)
                    progress["reference_cache"] = {
                        "video_id": video_id, "reference_sha256": reference.reference_sha256,
                        "frame_count": int(reference.frame_count), "yuv_sha256": sha256(raw_cache),
                        "bgr_sha256": sha256(bgr_cache)}
                    write_json(progress_path, progress)
            cpu_workers = config["metrics"]["cpu_workers"]
            cuda_workers = config["metrics"]["cuda_workers"]
            progress.update(substage="metrics", metric_workers=cpu_workers + cuda_workers, cpu_metric_workers=cpu_workers,
                            cuda_metric_workers=cuda_workers, vmaf_threads_per_process=2,
                            vmaf_backend="cpu+cuda" if cuda_workers else "cpu", metric_sample_stride=stride,
                            action_decode_passes=1, reference_decode_passes=1)
            progress["metrics_started_unix"] = time.time()
            metrics_started_monotonic = time.monotonic()
            write_json(progress_path, progress)
            jobs = [(SimpleNamespace(**r._asdict()), metric_reference(reference), selected,
                     artifacts / "metrics", 30, 30, raw_cache if needs_metrics else None,
                     bgr_cache if needs_metrics else None, stride)
                    for r in encodes.itertuples(index=False)
                    if not all((artifacts / "metrics" / video_id / f"action_{r.action_id:02d}{suffix}").is_file()
                               for suffix in (".json", ".frames.parquet", ".gops.parquet"))]
            progress["metric_actions_completed"] = len(encodes) - len(jobs)
            progress["new_metric_actions_completed"] = 0
            progress["new_metric_actions_by_backend"] = {"cpu": 0, "cuda": 0}
            write_json(progress_path, progress)
            for action, _ in mixed_metric_results(jobs, cpu_workers, cuda_workers):
                progress["metric_actions_completed"] += 1
                progress["new_metric_actions_completed"] += 1
                evidence = json.loads((artifacts / "metrics" / video_id / f"action_{action:02d}.json").read_text())
                progress["new_metric_actions_by_backend"][evidence["backend"]] += 1
                elapsed = time.monotonic() - metrics_started_monotonic
                progress["metric_elapsed_s"] = elapsed
                progress["new_actions_per_hour"] = progress["new_metric_actions_completed"] * 3600 / elapsed
                if progress["new_metric_actions_completed"] == 6:
                    progress["first_six_action_measurement"] = {
                        "video_id": video_id, "elapsed_s": elapsed,
                        "actions_per_hour": 6 * 3600 / elapsed,
                        "backend_counts": dict(progress["new_metric_actions_by_backend"]),
                        "pool_startup_included": True}
                write_json(progress_path, progress)
                print(f"METRIC DONE {video_id} action={action} "
                      f"{progress['metric_actions_completed']}/32 stride={stride}", flush=True)
        finally:
            raw_cache.unlink(missing_ok=True)
            bgr_cache.unlink(missing_ok=True)
            progress["reference_cache"] = None
        # Benchmarks run after two complete metrics grids; final stage aggregates all caches.
        for row in encodes.itertuples(index=False):
            if cached_metric(SimpleNamespace(**row._asdict()), metric_reference(reference),
                             artifacts / "metrics", 30, 30, stride) is None:
                raise RuntimeError(f"metric checksum/coverage mismatch before completion: {video_id}/{row.action_id}")
        released = release_reference(reference, artifacts, benchmark, identity, benchmark_required=False)
        progress["substage"] = "complete_video"
        progress["released_bytes"] += released
        if video_id not in progress["completed_videos"]:
            progress["completed_videos"].append(video_id)
        write_json(progress_path, progress)
        print(f"METRICS PREPARED {video_id}; benchmark after two videos; released {released} bytes; "
              f"videos {len(progress['completed_videos'])}/{len(references)}", flush=True)
        benchmark_ready_pairs(config, records, progress, progress_path)
    benchmark_ready_pairs(config, records, progress, progress_path, flush=True)
    all_done = len(progress["completed_videos"]) == len(references)
    if all_done:
        manifests = [pd.read_parquet(artifacts / "encodes" / r.video_id / "manifest.parquet") for r in records]
        pd.concat(manifests, ignore_index=True).sort_values(["video_id", "action_id"]).to_parquet(
            artifacts / "encodes/manifest.parquet", index=False)
    progress.update(awaiting_compaction=False, all_prepared=all_done,
                    substage="complete", active_batch_ids=[])
    write_json(progress_path, progress)
    print("ALL VIDEOS PREPARED. Continuing pipeline.", flush=True)


if __name__ == "__main__":
    main()
