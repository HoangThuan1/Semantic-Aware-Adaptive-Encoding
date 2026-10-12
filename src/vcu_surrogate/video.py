from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .dataset import dataset_hash, video_manifest
from .common import ROOT, assert_complete, portable_path, project_path, run, sha256, stable_hash, write_json


def _source_info(path: Path) -> dict:
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration,nb_frames",
        "-show_entries", "format=duration,size,bit_rate", "-of", "json", path,
    ], capture=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])
    return {"duration": duration, "stream": stream, "format": data["format"]}


def count_frames(path: Path, *, threads: int | None = None) -> int:
    decoder_options = [] if threads is None else ["-threads", str(threads)]
    result = run([
        "ffprobe", "-v", "error", *decoder_options, "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", path,
    ], capture=True)
    return int(result.stdout.strip())


def normalize_videos(config: dict, video_ids: list[str] | None = None) -> None:
    source_dir = project_path(config, "videos")
    output_dir = project_path(config, "artifacts") / "reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    video = config["video"]
    rows = []
    inputs = video_manifest(config).to_dict("records")
    if video_ids is not None:
        inputs = [r for r in inputs if r["video_id"] in video_ids]
        if set(r["video_id"] for r in inputs) != set(video_ids):
            raise RuntimeError("unknown video requested for reference generation")
    if not inputs:
        raise RuntimeError(f"no MP4 inputs found in {source_dir}")

    for record in inputs:
        source = ROOT / record["source_path"]
        video_id = record["video_id"]
        info = _source_info(source)
        full_gops = int(math.floor(info["duration"] * video["fps"] + 1e-6)) // video["gop_frames"]
        keep_frames = full_gops * video["gop_frames"]
        if keep_frames == 0:
            raise RuntimeError(f"video is shorter than one GOP: {source}")
        output = output_dir / f"{video_id}.mkv"
        job = {
            "source": portable_path(source), "source_sha256": sha256(source),
            "keep_frames": keep_frames, "video": video, "dataset_sha256": dataset_hash(config),
        }
        job_hash = stable_hash(job)
        sidecar = output.with_suffix(".json")
        if not assert_complete(output, sidecar, job_hash):
            temporary = output.with_name(output.stem + ".tmp.mkv")
            if temporary.exists():
                temporary.unlink()
            vf = (
                f"fps={video['fps']}:round=near,"
                f"scale={video['width']}:{video['height']}:flags=lanczos,"
                f"format={video['pixel_format']},setpts=N/{video['fps']}/TB"
            )
            run([
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", source,
                "-map", "0:v:0", "-an", "-sn", "-vf", vf, "-frames:v", str(keep_frames),
                "-c:v", "ffv1", "-level", "3", "-g", "1",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                temporary,
            ])
            actual_frames = count_frames(temporary)
            if actual_frames != keep_frames:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"normalization frame mismatch for {source}: {actual_frames} != {keep_frames}")
            temporary.replace(output)
            write_json(sidecar, {**job, "job_hash": job_hash, "output_sha256": sha256(output)})
        rows.append({
            "video_id": video_id, "split": record["split"], "dataset_sha256": dataset_hash(config), "source_path": portable_path(source),
            "reference_path": portable_path(output), "source_duration_s": info["duration"],
            "frame_count": keep_frames, "gop_count": full_gops,
            "fps": video["fps"], "width": video["width"], "height": video["height"],
            "source_sha256": sha256(source), "reference_sha256": sha256(output),
            "reference_size_bytes": output.stat().st_size,
        })
    result = pd.DataFrame(rows)
    manifest = output_dir / "manifest.parquet"
    if video_ids is not None and manifest.exists():
        previous = pd.read_parquet(manifest)
        result = pd.concat([previous[~previous.video_id.isin(video_ids)], result], ignore_index=True)
    result.to_parquet(manifest, index=False)
