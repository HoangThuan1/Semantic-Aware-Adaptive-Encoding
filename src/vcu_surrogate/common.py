from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(os.environ.get("VCU_PROJECT_ROOT", "/workspace"))
DEFAULT_CONFIG = ROOT / "config/pipeline.yaml"

PIPELINE_CONTRACT = {
    "video": {
        "width": 1920, "height": 1080, "fps": 30, "gop_frames": 30,
        "pixel_format": "yuv420p", "colorspace": "bt709",
    },
    "yolo": {
        "variant": "yolov5n", "weights": "models/yolov5n.pt", "image_size": 640,
    },
    "encoder": {
        "codec": "libx264", "bitrates_mbps": [0.5, 1, 1.5, 2, 3, 4, 6, 8],
        "roi_delta_qp": [0, -3, -6, -9], "preset": "veryfast",
        "tune": "zerolatency", "profile": "high", "refs": 1, "bframes": 0,
        "scenecut": 0, "rc_lookahead": 0, "vbv_seconds": 3.0, "vbv_init": 0.5,
        "qp_min": 10, "qp_max": 51, "filler": False, "threads": 6,
        "sliced_threads": False, "nal_hrd": "vbr", "force_cfr": True,
    },
    "dataset": {
        "bdd_revision": "2e48a0340eb591fa1918f3375d55601a7f0bbab3",
    },
    "traces": {
        "commit": "d28c4fd48d0e358db024cc0217b111c070815fa7",
        "url": "https://github.com/confiwent/Real-world-bandwidth-traces.git",
    },
}


def assert_pipeline_contract(config: dict[str, Any]) -> None:
    for section, expected_values in PIPELINE_CONTRACT.items():
        actual_section = config.get(section, {})
        for key, expected in expected_values.items():
            actual = actual_section.get(key)
            if actual != expected:
                raise RuntimeError(
                    f"unsupported {section}.{key}={actual!r}; required pinned value is {expected!r}"
                )


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(Path(path).resolve())
    config["_config_sha256"] = sha256(Path(path))
    assert_pipeline_contract(config)
    metrics = config.get("metrics", {})
    if metrics:
        if (metrics.get("sample_stride") != 5 or metrics.get("sample_offset") != 0
                or (metrics.get("cpu_workers"), metrics.get("cuda_workers")) not in ((6, 0), (5, 1))
                or metrics.get("vmaf_cuda_commit") != "17a67b238ce0539bdeafdc95961abac64fa16ea8"):
            raise RuntimeError("metrics must use stride 5, offset 0, six CPU workers or five CPU workers and one pinned CUDA worker")
    return config


def project_path(config: dict[str, Any], group: str) -> Path:
    return ROOT / config["paths"][group]


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("models", "raw_data", "processed_data", "artifacts", "reports"):
        project_path(config, key).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise RuntimeError(f"directory missing: {root}")
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path)})
    if not entries:
        raise RuntimeError(f"directory contains no files: {root}")
    return stable_hash(entries)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def run(command: Iterable[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [str(item) for item in command]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def probe_json(path: Path, entries: str, *, packets: bool = False) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error"]
    if packets:
        command += ["-select_streams", "v:0", "-show_packets"]
    else:
        command += ["-select_streams", "v:0", "-show_streams", "-show_format"]
    command += ["-show_entries", entries, "-of", "json", str(path)]
    return json.loads(run(command, capture=True).stdout)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def assert_complete(path: Path, sidecar: Path, expected_hash: str) -> bool:
    if not path.is_file() or not sidecar.is_file():
        return False
    with sidecar.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("job_hash") != expected_hash:
        raise RuntimeError(f"existing artifact has incompatible configuration: {path}")
    return True
