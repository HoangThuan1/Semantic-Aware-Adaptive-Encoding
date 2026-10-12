from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

from .common import ROOT, ensure_dirs, project_path, sha256, write_json

MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
MODEL_SHA256 = "4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def _model_path(config: dict) -> Path:
    path = Path(config["yolo"]["weights"])
    return path if path.is_absolute() else ROOT / path


def download_assets(config: dict) -> None:
    from .dataset import download_bdd, prepare_video_manifest
    from .traces import download_traces
    ensure_dirs(config)
    model = _model_path(config)
    if not model.exists():
        _download(MODEL_URL, model)
    if sha256(model) != MODEL_SHA256:
        raise RuntimeError("YOLO weight checksum mismatch")
    download_traces(config)
    download_bdd(config)
    prepare_video_manifest(config)
    write_json(project_path(config, "raw_data") / "download_manifest.json", {
        "yolov5n": {"url": MODEL_URL, "sha256": MODEL_SHA256},
        "traces": config["traces"], "bdd_revision": config["dataset"]["bdd_revision"],
    })


def assert_downloads(config: dict) -> None:
    from .dataset import BDD_REVISION, video_manifest
    from .traces import assert_trace_sources
    raw = project_path(config, "raw_data")
    data = json.loads((raw / "download_manifest.json").read_text())
    if data.get("traces") != config["traces"] or data.get("bdd_revision") != BDD_REVISION:
        raise RuntimeError("download manifest source mismatch")
    if config["dataset"]["bdd_revision"] != BDD_REVISION:
        raise RuntimeError("BDD revision mismatch")
    if sha256(_model_path(config)) != MODEL_SHA256:
        raise RuntimeError("YOLO weight checksum mismatch")
    assert_trace_sources(config)
    bdd = json.loads((raw / "bdd_download_manifest.json").read_text())
    if bdd.get("revision") != BDD_REVISION or not 1_000_000_000 <= bdd["total_bytes"] <= 3_000_000_000:
        raise RuntimeError("BDD download manifest invalid")
    for row in video_manifest(config).itertuples(index=False):
        if sha256(ROOT / row.source_path) != row.source_sha256:
            raise RuntimeError(f"source video changed: {row.video_id}")
