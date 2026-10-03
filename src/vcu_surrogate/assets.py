from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

from .common import ROOT, ensure_dirs, project_path, run, sha256, write_json

MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
MODEL_SHA256 = "4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3"
LTE_UPLINK_FILES = (
    "ATT-LTE-driving.up", "ATT-LTE-driving-2016.up",
    "TMobile-LTE-driving.up", "TMobile-LTE-short.up",
    "Verizon-LTE-driving.up", "Verizon-LTE-short.up",
)
LTE_UPLINK_SHA256 = {
    "ATT-LTE-driving.up": "d58a640b9bccd131a421bff6eaf727aba324d9258b35009655bbf68e087b9b29",
    "ATT-LTE-driving-2016.up": "ebb3d53674b77be4ed920ad91c4513a496e04e30da839e2ca8a627cb7269071e",
    "TMobile-LTE-driving.up": "556620f1c11a61c7766678b08325ef4fc8d1bd774fbc98fb96854ae3d3795b61",
    "TMobile-LTE-short.up": "1c0d1bf5417b07597724fd64afe676af3c1f337a4ccdde90106bf96b1bcf18b2",
    "Verizon-LTE-driving.up": "669840f626966d7490a27629d4227eb2a866d40f104a6217c53b3f4b5a207d1f",
    "Verizon-LTE-short.up": "f6cd9049f58d70f12545772a3482ded085b6d50f06a2f4ff4afcfb6fe7976dcc",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def _model_path(config: dict) -> Path:
    path = Path(config["yolo"]["weights"])
    return path if path.is_absolute() else ROOT / path


def _trace_paths(raw: Path) -> dict[str, Path]:
    return {name: raw / "mahimahi" / "traces" / name for name in LTE_UPLINK_FILES}


def download_assets(config: dict) -> None:
    ensure_dirs(config)
    raw = project_path(config, "raw_data")
    model = _model_path(config)
    if not model.exists():
        _download(MODEL_URL, model)
    if sha256(model) != MODEL_SHA256:
        raise RuntimeError(f"YOLO weight checksum mismatch: {model}")
    traces = config["traces"]
    checkout = raw / "mahimahi"
    if not (checkout / ".git").is_dir():
        if checkout.exists():
            raise RuntimeError(f"Mahimahi path exists without a Git checkout: {checkout}")
        run(["git", "clone", traces["mahimahi_url"], checkout])
    run(["git", "-C", checkout, "checkout", "--detach", traces["mahimahi_commit"]])
    head = run(["git", "-C", checkout, "rev-parse", "HEAD"], capture=True).stdout.strip()
    if head != traces["mahimahi_commit"]:
        raise RuntimeError(f"Mahimahi revision mismatch: {head}")
    digests = {}
    for name, path in _trace_paths(raw).items():
        if not path.is_file():
            raise RuntimeError(f"required LTE uplink trace missing: {path}")
        digests[name] = sha256(path)
    if digests != LTE_UPLINK_SHA256:
        raise RuntimeError("Mahimahi LTE uplink files do not match pinned SHA-256")
    write_json(raw / "download_manifest.json", {
        "yolov5n": {"url": MODEL_URL, "sha256": MODEL_SHA256},
        "mahimahi": {"url": traces["mahimahi_url"], "commit": head,
                     "files_sha256": digests},
    })


def assert_downloads(config: dict) -> None:
    raw = project_path(config, "raw_data")
    manifest = raw / "download_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("download manifest missing; run `vcu-surrogate download`")
    data = json.loads(manifest.read_text())
    if data.get("yolov5n") != {"url": MODEL_URL, "sha256": MODEL_SHA256}:
        raise RuntimeError("download manifest does not match pinned YOLO weights")
    model = _model_path(config)
    if not model.is_file() or sha256(model) != MODEL_SHA256:
        raise RuntimeError("YOLO weights no longer match download manifest")
    traces = config["traces"]
    checkout = raw / "mahimahi"
    expected = data.get("mahimahi", {})
    if expected.get("url") != traces["mahimahi_url"] or expected.get("commit") != traces["mahimahi_commit"]:
        raise RuntimeError("download manifest does not match pinned Mahimahi source")
    head = run(["git", "-C", checkout, "rev-parse", "HEAD"], capture=True).stdout.strip()
    if head != traces["mahimahi_commit"]:
        raise RuntimeError("Mahimahi checkout no longer matches pinned commit")
    actual = {}
    for name, path in _trace_paths(raw).items():
        if not path.is_file():
            raise RuntimeError(f"required LTE uplink trace missing: {path}")
        actual[name] = sha256(path)
    if actual != expected.get("files_sha256") or actual != LTE_UPLINK_SHA256:
        raise RuntimeError("Mahimahi trace checksum mismatch")
