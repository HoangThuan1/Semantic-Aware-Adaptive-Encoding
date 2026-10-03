from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

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


def count_frames(path: Path) -> int:
    result = run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", path,
    ], capture=True)
    return int(result.stdout.strip())


def normalize_videos(config: dict) -> None:
    source_dir = project_path(config, "videos")
    output_dir = project_path(config, "artifacts") / "reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    video = config["video"]
    rows = []
    inputs = sorted(source_dir.glob("*.mp4"))
    if not inputs:
        raise RuntimeError(f"no MP4 inputs found in {source_dir}")

    for source in inputs:
        info = _source_info(source)
        full_gops = int(math.floor(info["duration"] * video["fps"] + 1e-6)) // video["gop_frames"]
        keep_frames = full_gops * video["gop_frames"]
        if keep_frames == 0:
            raise RuntimeError(f"video is shorter than one GOP: {source}")
        output = output_dir / f"{source.stem}.mkv"
        job = {
            "source": portable_path(source), "source_sha256": sha256(source),
            "keep_frames": keep_frames, "video": video,
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
            "video_id": source.stem, "source_path": portable_path(source),
            "reference_path": portable_path(output), "source_duration_s": info["duration"],
            "frame_count": keep_frames, "gop_count": full_gops,
            "fps": video["fps"], "width": video["width"], "height": video["height"],
            "source_sha256": sha256(source), "reference_sha256": sha256(output),
        })
    pd.DataFrame(rows).to_parquet(output_dir / "manifest.parquet", index=False)
