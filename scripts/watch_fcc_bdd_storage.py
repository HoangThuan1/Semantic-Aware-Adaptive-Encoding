"""Host-side storage telemetry: no artificial disk budget is enforced."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import yaml
from pathlib import Path

CONTAINER = "vcu-fcc-bdd-pipeline"
PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / yaml.safe_load((PROJECT / "config/pipeline.yaml").read_text())["paths"]["reports"]


def file_bytes(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0  # A writer may atomically rename/delete between scan and stat.


def main():
    while True:
        probe = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER],
                               text=True, capture_output=True)
        if probe.returncode or probe.stdout.strip() != "true":
            return
        free = shutil.disk_usage("/mnt/c").free
        status = {"host_free_bytes": free, "limits_enforced": False, "checked_unix": time.time()}
        progress_path = REPORTS / "compact_progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            ids = progress.get("active_batch_ids", [])
            size = 0
            config = yaml.safe_load((PROJECT / "config/pipeline.yaml").read_text())
            artifacts = PROJECT / config["paths"]["artifacts"]
            for video_id in ids:
                reference = artifacts / "reference" / f"{video_id}.mkv"
                size += file_bytes(reference)
                size += file_bytes(reference.with_name(reference.stem + ".tmp.mkv"))
                size += file_bytes(reference.with_name(reference.stem + ".frames.yuv"))
                for file in (artifacts / "encodes" / video_id).glob("*.mp4"):
                    size += file_bytes(file)
                for file in (artifacts / "metrics" / video_id).glob("*.vmaf.json"):
                    size += file_bytes(file)
                sampled = REPORTS / "benchmark/cache" / f"{video_id}.reference.npy"
                size += file_bytes(sampled)
            status["batch_temp_bytes"] = size
            status["batch_limit_bytes"] = None
        path = REPORTS / "storage_guard.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(path)
        time.sleep(5)


if __name__ == "__main__":
    main()
