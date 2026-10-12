"""Run the expanded dataset pipeline with durable stage logs and truthful status."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from vcu_surrogate.common import ROOT, load_config, portable_path, project_path, run, sha256, stable_hash, write_json
from vcu_surrogate.dataset import dataset_hash, video_manifest
from vcu_surrogate.encoding import encode_grid
from vcu_surrogate.inference import infer_metadata
from vcu_surrogate.metrics import compute_metrics
from vcu_surrogate.video import count_frames

def now():
    return datetime.now(timezone.utc).isoformat()


def smoke_and_capacity(config):
    videos = video_manifest(config)
    row = videos[videos.source == "bdd100k"].iloc[0]
    artifacts = project_path(config, "artifacts") / "smoke"
    reference = artifacts / "reference" / f"{row.video_id}.mkv"
    reference.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    probe_job = {"source_sha256": row.source_sha256, "dataset_sha256": dataset_hash(config), "frames": 180}
    probe_sidecar = reference.with_suffix(".smoke.json")
    if reference.exists():
        if not probe_sidecar.exists():
            raise RuntimeError("existing smoke reference has no provenance sidecar")
        previous = json.loads(probe_sidecar.read_text())
        if previous.get("job_hash") != stable_hash(probe_job) or previous.get("output_sha256") != sha256(reference):
            raise RuntimeError("existing smoke reference provenance mismatch")
    else:
        run(["ffmpeg", "-v", "error", "-y", "-i", ROOT / row.source_path,
             "-an", "-vf", "fps=30:round=near,scale=1920:1080:flags=lanczos,format=yuv420p,setpts=N/30/TB",
             "-frames:v", "180", "-c:v", "ffv1", "-level", "3", "-g", "1",
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", reference])
        write_json(probe_sidecar, {**probe_job, "job_hash": stable_hash(probe_job), "output_sha256": sha256(reference)})
    if count_frames(reference) != 180:
        raise RuntimeError("BDD six-second normalization smoke failed")
    seconds = float(videos.duration_s.map(int).sum())
    reference_estimate = int(reference.stat().st_size / 6 * seconds * 2)
    encoded_estimate = int(sum(config["encoder"]["bitrates_mbps"]) * 4 * 1e6 / 8 * seconds * 1.5)
    # Metric JSON/parquet, detector tables and benchmark temp frames reserve.
    required = reference_estimate + encoded_estimate + 60_000_000_000
    available = shutil.disk_usage(ROOT).free
    write_json(project_path(config, "reports") / "capacity_estimate.json", {
        "probe_bytes": reference.stat().st_size, "probe_seconds": 6,
        "probe_elapsed_s": time.monotonic() - start, "dataset_seconds": seconds,
        "reference_estimate_bytes_with_2x_margin": reference_estimate,
        "encode_estimate_bytes_with_1_5x_margin": encoded_estimate,
        "additional_reserve_bytes": 60_000_000_000, "required_bytes": required,
        "available_bytes": available})
    if available < required:
        raise RuntimeError(f"insufficient disk: estimated {required} bytes, available {available}")
    pd.DataFrame([{"video_id": row.video_id, "split": row.split,
                   "source_path": row.source_path, "source_sha256": row.source_sha256,
                   "reference_path": portable_path(reference), "reference_sha256": sha256(reference),
                   "frame_count": 180, "gop_count": 6, "source_duration_s": 6,
                   "fps": 30, "width": 1920, "height": 1080,
                   "dataset_sha256": dataset_hash(config)}]).to_parquet(reference.parent / "manifest.parquet", index=False)
    smoke = {**config, "paths": {**config["paths"], "artifacts": portable_path(artifacts)}}
    infer_metadata(smoke)
    encode_grid(smoke)
    compute_metrics(smoke)
    grid = pd.read_parquet(artifacts / "metrics/gop_action_grid.parquet")
    if len(grid) != 6 * 32 or grid.groupby("gop_idx").action_id.nunique().tolist() != [32] * 6:
        raise RuntimeError("smoke grid does not cover six GOPs and 32 actions")
    write_json(artifacts / "smoke_result.json", {"status": "passed", "grid_rows": len(grid),
                                                "dataset_sha256": dataset_hash(config)})


def main():
    config = load_config()
    reports = project_path(config, "reports")
    logs = reports / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = reports / "run_status.json"
    state = {"status": "running", "started_at_utc": now(), "stage": "wait_download", "completed_stages": []}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("config_sha256") == config["_config_sha256"]:
            state["completed_stages"] = previous.get("completed_stages", [])
            if "dataset" in previous:
                state["dataset"] = previous["dataset"]
            if "metrics_optimization" in previous:
                state["metrics_optimization"] = previous["metrics_optimization"]
    state["config_sha256"] = config["_config_sha256"]

    def save():
        state["updated_at_utc"] = now()
        write_json(state_path, state)
        lines = ["# FCC/Norway + BDD100K run", "", f"Status: **{state['status']}**; stage: `{state['stage']}`.",
                 "", f"Completed stages: {', '.join(state['completed_stages']) or 'none'}."]
        if "download_progress" in state:
            progress = state["download_progress"]
            lines += ["", f"Downloaded BDD100K: {progress['videos']} videos, {progress['bytes'] / 1e9:.3f} GB."]
        if "episode_mix" in state:
            lines += ["", "Episode mix:", "", "```json", json.dumps(state["episode_mix"], indent=2), "```"]
        if "dataset" in state:
            lines += ["", "```json", json.dumps(state["dataset"], indent=2), "```"]
        if "metrics_optimization" in state:
            lines += ["", "Metrics optimization (measured sample):", "", "```json",
                      json.dumps(state["metrics_optimization"], indent=2), "```"]
        if "error" in state:
            lines += ["", "Failure: " + state["error"]]
        (reports / "RUN_REPORT.md").write_text("\n".join(lines) + "\n")

    def stage(name, callback=None, command=None):
        state["stage"] = name
        save()
        if name in state["completed_stages"]:
            return
        print(f"START {name} {now()}", flush=True)
        log = logs / f"{name}.log"
        with log.open("a") as handle:
            if callback:
                from contextlib import redirect_stdout, redirect_stderr
                with redirect_stdout(handle), redirect_stderr(handle):
                    callback()
            else:
                result = subprocess.run(command or [sys.executable, "-m", "vcu_surrogate.cli", name],
                                        stdout=handle, stderr=subprocess.STDOUT)
                result.check_returncode()
        state["completed_stages"].append(name)
        save()
        print(f"DONE {name} {now()}", flush=True)

    try:
        save()
        deadline = time.monotonic() + 7200
        while not (project_path(config, "raw_data") / "download_manifest.json").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("download did not complete within two hours; inspect downloader log")
            journal = project_path(config, "raw_data") / "bdd_download_progress.json"
            if journal.exists():
                progress = json.loads(journal.read_text())
                state["download_progress"] = {"videos": len(progress["files"]), "bytes": progress["total_bytes"]}
                save()
            time.sleep(10)
        videos = video_manifest(config)
        trace = json.loads((project_path(config, "processed_data") / "trace_sources.json").read_text())
        if state_path.exists():
            previous_dataset = json.loads(state_path.read_text()).get("dataset", {})
            if previous_dataset.get("dataset_sha256", dataset_hash(config)) != dataset_hash(config):
                raise RuntimeError("dataset changed since previous run; use a new output namespace")
        state["dataset"] = {"video_count": len(videos), "video_splits": dict(Counter(videos.split)),
                            "bdd_bytes": int(videos.loc[videos.source == "bdd100k", "size_bytes"].sum()),
                            "bdd_videos": int((videos.source == "bdd100k").sum()),
                            "traces_by_source_split": dict(Counter(f"{r['source']}/{r['split']}" for r in trace["files"])),
                            "dataset_sha256": dataset_hash(config)}
        save()
        stage("tests", command=[sys.executable, "-m", "pytest", "tests", "-ra"])
        stage("preflight")
        stage("smoke_capacity", callback=lambda: smoke_and_capacity(config))
        for name in ("traces", "normalize", "infer", "encode-grid", "metrics", "transitions", "validate",
                     "train", "evaluate", "export-policy", "render-policy", "benchmark"):
            if name == "encode-grid":
                stage(name, command=[sys.executable, "scripts/prepare_grid_compact.py"])
            else:
                stage(name)
        lineage = []
        for seed in config["rl"]["seeds"]:
            episodes = json.loads((reports / f"rl/episode_lineage_seed_{seed}.json").read_text())
            count = Counter("synthetic" if e["synthetic"] else "real" for e in episodes)
            real_sources = {r["trace_id"]: r["source"] for r in trace["files"]}
            sources = Counter(real_sources[e["trace_id"]] for e in episodes if not e["synthetic"])
            lineage.append({"seed": seed, "episodes": len(episodes), "mix": dict(count), "real_sources": dict(sources),
                            "episode_fractions": {**{k: v / len(episodes) for k, v in sources.items()},
                                                  "synthetic": count["synthetic"] / len(episodes)}})
        state["episode_mix"] = lineage
        state["status"], state["stage"] = "complete", "complete"
        save()
    except Exception as error:
        state["status"], state["error"] = "failed", str(error)
        save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
