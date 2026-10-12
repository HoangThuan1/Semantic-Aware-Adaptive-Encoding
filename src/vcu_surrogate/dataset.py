"""Pinned BDD100K subset, resumable ZIP range reads and video-level splits."""
from __future__ import annotations

import json
import math
import re
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, portable_path, project_path, run, sha256, stable_hash, write_json

BDD_REVISION = "2e48a0340eb591fa1918f3375d55601a7f0bbab3"
BDD_BASE = f"https://huggingface.co/datasets/linxxx3/bdd100k_videos/resolve/{BDD_REVISION}/bdd100k_videos/"
PART_SIZE = 107374182400


def read_range(url: str, start: int, size: int) -> bytes:
    if size <= 0:
        return b""
    for attempt in range(5):
        try:
            # Unique query prevents intermediary caches returning a different byte range.
            request = urllib.request.Request(f"{url}?range={start}-{start + size - 1}",
                                             headers={"Range": f"bytes={start}-{start + size - 1}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {start}-{start + size - 1}/"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise RuntimeError("server did not honor exact HTTP Range")
                payload = response.read(size + 1)
                if len(payload) != size:
                    raise IOError("incomplete range response")
                return payload
        except (OSError, IOError) as error:
            if attempt == 4:
                raise error
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def archive_range(offset: int, size: int) -> bytes:
    chunks = []
    while size:
        part, start = divmod(offset, PART_SIZE)
        count = min(size, PART_SIZE - start)
        chunks.append(read_range(BDD_BASE + f"bdd100k_videos.zip.part-{part:05d}", start, count))
        offset += count
        size -= count
    return b"".join(chunks)


def zip_entry(offset: int, reader=archive_range) -> dict:
    header = reader(offset, 30)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    signature, _, flags, method, _, _, crc, compressed, size, name_len, extra_len = fields
    if signature != 0x04034B50 or flags & 9 or method != 0:
        raise RuntimeError("expected unencrypted stored ZIP entry with known size")
    metadata = reader(offset + 30, name_len + extra_len)
    name = metadata[:name_len].decode("utf-8")
    extra = metadata[name_len:]
    pos = 0
    while pos + 4 <= len(extra):
        kind, length = struct.unpack_from("<HH", extra, pos)
        payload = extra[pos + 4:pos + 4 + length]
        if kind == 1:
            n = 0
            if size == 0xFFFFFFFF:
                size = struct.unpack_from("<Q", payload, n)[0]; n += 8
            if compressed == 0xFFFFFFFF:
                compressed = struct.unpack_from("<Q", payload, n)[0]
        pos += 4 + length
    if size != compressed:
        raise RuntimeError("stored ZIP entry sizes disagree")
    data_offset = offset + 30 + name_len + extra_len
    return {"name": name, "size_bytes": size, "crc32": crc,
            "data_offset": data_offset, "next_offset": data_offset + compressed}


def crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def download_entry(entry: dict, destination: Path, reader=archive_range) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != entry["size_bytes"] or crc32_file(destination) != entry["crc32"]:
            raise RuntimeError(f"existing BDD video CRC mismatch: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    done = temporary.stat().st_size if temporary.exists() else 0
    if done > entry["size_bytes"]:
        raise RuntimeError("partial download exceeds entry size")
    with temporary.open("ab") as handle:
        while done < entry["size_bytes"]:
            count = min(16 * 1024 * 1024, entry["size_bytes"] - done)
            handle.write(reader(entry["data_offset"] + done, count))
            handle.flush()
            done += count
    if crc32_file(temporary) != entry["crc32"]:
        temporary.unlink()
        raise RuntimeError(f"BDD ZIP CRC mismatch: {destination}")
    # Decode every frame before publishing the completed input.
    run(["ffmpeg", "-v", "error", "-xerror", "-i", temporary, "-map", "0:v:0", "-vf", "setpts=N/(30*TB)", "-fps_mode", "passthrough", "-f", "null", "-"])
    temporary.replace(destination)


def split_ids(ids: list[str], seed: int, development: float = .7, validation: float = .1) -> dict[str, str]:
    ordered = np.array(sorted(ids), dtype=object)
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("duplicate split ID")
    np.random.default_rng(seed).shuffle(ordered)
    n_dev = math.floor(development * len(ordered))
    n_val = round(validation * n_dev)
    if min(n_dev - n_val, n_val, len(ordered) - n_dev) < 1:
        raise RuntimeError("dataset too small for nonempty train/validation/test")
    return {str(value): "validation" if i < n_val else "train" if i < n_dev else "test"
            for i, value in enumerate(ordered)}


def download_bdd(config: dict) -> None:
    options = config["dataset"]
    target, maximum = int(options["target_bytes"]), int(options["max_bytes"])
    if not 1_000_000_000 <= target <= maximum <= 3_000_000_000:
        raise RuntimeError("BDD byte budget must be within 1–3 GB")
    destination = project_path(config, "videos") / "bdd100k"
    offset, total, entries = 0, 0, []
    journal = project_path(config, "raw_data") / "bdd_download_progress.json"
    if journal.exists():
        progress = json.loads(journal.read_text())
        if progress["revision"] != BDD_REVISION:
            raise RuntimeError("BDD download progress revision mismatch")
        offset, total, entries = progress["next_offset"], progress["total_bytes"], progress["files"]
        for item in entries:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError("completed BDD video checksum mismatch")
    while total < target:
        entry = zip_entry(offset)
        offset = entry["next_offset"]
        if not entry["name"].lower().endswith((".mp4", ".mov")):
            continue
        if total + entry["size_bytes"] > maximum:
            raise RuntimeError("next complete BDD video would exceed maximum budget")
        path = destination / Path(entry["name"]).name
        download_entry(entry, path)
        entries.append({**entry, "path": portable_path(path), "sha256": sha256(path)})
        total += entry["size_bytes"]
        write_json(journal, {"revision": BDD_REVISION, "next_offset": offset, "total_bytes": total, "files": entries})
        print(f"BDD100K {len(entries)} videos, {total / 1e9:.3f} GB", flush=True)
    write_json(project_path(config, "raw_data") / "bdd_download_manifest.json",
               {"revision": BDD_REVISION, "source": BDD_BASE, "total_bytes": total, "files": entries})


def prepare_video_manifest(config: dict) -> None:
    bdd = json.loads((project_path(config, "raw_data") / "bdd_download_manifest.json").read_text())
    downloaded = {item["path"]: item for item in bdd["files"]}
    rows, checksums, ids = [], set(), set()
    for path in sorted(project_path(config, "videos").rglob("*")):
        if path.suffix.lower() not in {".mp4", ".mov"}:
            continue
        digest = sha256(path)
        if digest in checksums:
            continue
        checksums.add(digest)
        video_id = ("bdd_" if portable_path(path) in downloaded else "") + path.stem
        if video_id in ids:
            raise RuntimeError(f"duplicate video ID: {video_id}")
        ids.add(video_id)
        info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "format=duration", "-of", "json", path], capture=True).stdout)
        duration = float(info["format"]["duration"])
        if duration < 6:
            raise RuntimeError(f"video too short for RL: {path}")
        rows.append({"video_id": video_id, "source_path": portable_path(path), "source_sha256": digest,
                     "size_bytes": path.stat().st_size, "duration_s": duration,
                     "source": "bdd100k" if portable_path(path) in downloaded else "existing",
                     "revision": BDD_REVISION if portable_path(path) in downloaded else "local"})
    splits = split_ids(list(ids), int(config["traces"]["split_seed"]))
    for row in rows:
        row["split"] = splits[row["video_id"]]
    path = project_path(config, "processed_data") / "video_manifest.json"
    payload = {"schema_version": 1, "seed": config["traces"]["split_seed"], "videos": rows}
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("video manifest changed; use a new output namespace")
    write_json(path, payload)


def video_manifest(config: dict) -> pd.DataFrame:
    path = project_path(config, "processed_data") / "video_manifest.json"
    data = pd.DataFrame(json.loads(path.read_text())["videos"])
    if data.video_id.duplicated().any() or data.source_sha256.duplicated().any():
        raise RuntimeError("duplicate video in split manifest")
    if set(data.split) != {"train", "validation", "test"}:
        raise RuntimeError("video split incomplete")
    return data


def dataset_hash(config: dict) -> str:
    processed = project_path(config, "processed_data")
    return stable_hash({name: sha256(processed / name) for name in ("video_manifest.json", "trace_sources.json")})


def trace_group(name: str) -> str:
    if name.startswith("trace_"):
        return "fcc_" + name.split("_")[1]
    return re.sub(r"_part\d+", "", name)
