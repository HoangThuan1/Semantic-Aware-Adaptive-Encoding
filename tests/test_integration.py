from __future__ import annotations

import csv
import json
from pathlib import Path

from vcu_surrogate.common import run
from vcu_surrogate.metrics import _run_vmaf
from vcu_surrogate.video import count_frames


def test_normalize_and_roi_encoder_gop_structure(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    source = videos / "synthetic.mp4"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=2.1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", source,
    ])
    reference = tmp_path / "reference.mkv"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source,
        "-vf", "scale=1920:1080:flags=lanczos,fps=30", "-frames:v", "60",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", reference,
    ])
    assert count_frames(reference) == 60
    roi = tmp_path / "roi.csv"
    with roi.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "x1", "y1", "x2", "y2"])
        for frame in range(60):
            writer.writerow([frame, 160, 160, 640, 640])
    encoded = tmp_path / "encoded.mp4"
    run(["roi_encoder", reference, roi, encoded, "2000", "-6", "30"])
    assert count_frames(encoded) == 60
    probe = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=key_frame,pict_type", "-of", "json", encoded,
    ], capture=True)
    frames = json.loads(probe.stdout)["frames"]
    assert [(index, frame["pict_type"]) for index, frame in enumerate(frames) if frame["key_frame"]] == [(0, "I"), (30, "I")]
    assert all(frame["pict_type"] == ("I" if index % 30 == 0 else "P") for index, frame in enumerate(frames))

    quality = _run_vmaf(encoded, reference, tmp_path / "vmaf.json")
    assert quality.frame_idx.tolist() == list(range(60))
    assert quality[["vmaf", "psnr", "ssim"]].notna().all().all()
