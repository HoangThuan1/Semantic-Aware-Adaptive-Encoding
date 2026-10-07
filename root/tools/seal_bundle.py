#!/usr/bin/env python3
"""Host-only: refresh code receipts after development; never accept changed payloads."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_runtime.common import sha256, write_json


def main():
    path = ROOT / "manifest.json"
    manifest = json.loads(path.read_text())
    for relative, receipt in manifest["files"].items():
        if relative.startswith(("videos/", "bw_trace/", "models/")):
            target = ROOT / relative
            if not target.is_file() or target.stat().st_size != receipt["bytes"] or sha256(target) != receipt["sha256"]:
                raise RuntimeError("payload changed; refusing to reseal: " + relative)
    for folder in ("rl_runtime", "tools", "vcu_test"):
        for target in sorted((ROOT/folder).rglob("*")):
            if target.is_file() and "__pycache__" not in target.parts and target.suffix != ".pyc":
                manifest["files"][str(target.relative_to(ROOT))] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
    write_json(path, manifest)
    print("Code receipts sealed; payload hashes unchanged")


if __name__ == "__main__":
    main()
