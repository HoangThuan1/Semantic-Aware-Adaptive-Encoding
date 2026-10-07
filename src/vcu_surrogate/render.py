from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import ROOT, project_path, run, sha256, stable_hash, write_json
from .rl_env import WARMUP_ACTION, WARMUP_GOPS
from .validation import _frame_types
from .video import count_frames


def _quote_ffconcat(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def render_policy_videos(config: dict) -> None:
    reports = project_path(config, "reports") / "rl"
    artifacts = project_path(config, "artifacts")
    summary_path = reports / "evaluation_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"DDQN evaluation is required before rendering: {summary_path}")
    summary = json.loads(summary_path.read_text())
    seed = int(summary["selected_seed"])
    if seed not in config["rl"]["seeds"]:
        raise RuntimeError(f"selected seed is outside DDQN configuration: {seed}")
    rollout_path = reports / f"test_rollout_seed_{seed}.parquet"
    encode_path = artifacts / "encodes" / "manifest.parquet"
    reference_path = artifacts / "reference" / "manifest.parquet"
    if not all(path.is_file() for path in (rollout_path, encode_path, reference_path)):
        raise RuntimeError("selected test rollout, encode manifest, or reference manifest missing")
    rollout = pd.read_parquet(rollout_path)
    encodes = pd.read_parquet(encode_path)
    references = pd.read_parquet(reference_path)
    if encodes.duplicated(["video_id", "action_id"]).any():
        raise RuntimeError("duplicate video/action encode")
    lookup = encodes.set_index(["video_id", "action_id"])
    fps = int(config["video"]["fps"])
    gop_frames = int(config["video"]["gop_frames"])
    if fps != 30 or gop_frames != 30:
        raise RuntimeError("policy renderer requires pinned 30 FPS/GOP30 contract")
    output_root = artifacts / "policy_videos" / f"seed_{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for reference in references[references.split == "test"].itertuples(index=False):
        video_id = str(reference.video_id)
        candidates = sorted(rollout.loc[(rollout.video_id == video_id) & (rollout.trace_start == 0), "trace_id"].unique())
        if not candidates:
            raise RuntimeError(f"no valid test trace for {video_id}")
        trace_id = candidates[0]
        selected = rollout[(rollout.video_id == video_id)
                           & (rollout.trace_id == trace_id)
                           & (rollout.trace_start == 0)].sort_values("gop_idx")
        expected_indices = list(range(WARMUP_GOPS, int(reference.gop_count)))
        if selected.gop_idx.tolist() != expected_indices:
            raise RuntimeError(f"incomplete selected test rollout for {video_id}/trace_start=0")
        actions = [WARMUP_ACTION] * WARMUP_GOPS + selected.action_id.astype(int).tolist()
        if len(actions) != int(reference.gop_count):
            raise RuntimeError(f"policy action sequence length mismatch: {video_id}")
        source_files = []
        playlist_lines = ["ffconcat version 1.0"]
        for k, action in enumerate(actions):
            try:
                row = lookup.loc[(video_id, action)]
            except KeyError as error:
                raise RuntimeError(f"missing encode for {video_id}/action={action}") from error
            source = ROOT / str(row.encoded_path)
            if not source.is_file():
                raise RuntimeError(f"encoded source missing: {source}")
            source_files.append({"gop_idx": k, "action_id": action,
                                 "encoded_sha256": str(row.encoded_sha256)})
            playlist_lines.extend([
                f"file {_quote_ffconcat(source)}",
                f"inpoint {k:.6f}", f"outpoint {k + 1:.6f}", "duration 1.000000",
            ])
        job_hash = stable_hash({"seed": seed, "video_id": video_id,
                                "trace_id": trace_id, "trace_start": 0,
                                "gop_frames": gop_frames, "sources": source_files})
        output = output_root / f"{video_id}_{trace_id}_start0.mp4"
        sidecar = output.with_suffix(".json")
        if output.is_file() and sidecar.is_file():
            existing = json.loads(sidecar.read_text())
            if existing.get("job_hash") != job_hash or sha256(output) != existing.get("output_sha256"):
                raise RuntimeError(f"incompatible existing policy video: {output}")
        else:
            playlist = output.with_suffix(".ffconcat")
            playlist.write_text("\n".join(playlist_lines) + "\n", encoding="utf-8")
            temporary = output.with_name(output.stem + ".tmp.mp4")
            temporary.unlink(missing_ok=True)
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", playlist, "-map", "0:v:0", "-an", "-c:v", "copy",
                 "-movflags", "+faststart", temporary])
            if count_frames(temporary) != int(reference.frame_count):
                raise RuntimeError(f"rendered policy video frame count mismatch: {temporary}")
            types = _frame_types(temporary)
            if len(types) != int(reference.frame_count):
                raise RuntimeError(f"rendered policy video frame type count mismatch: {temporary}")
            for frame_idx, (key, picture_type) in enumerate(types):
                if frame_idx % gop_frames == 0:
                    if key != 1 or picture_type != "I":
                        raise RuntimeError(f"rendered policy video missing IDR at frame {frame_idx}: {temporary}")
                elif key != 0 or picture_type != "P":
                    raise RuntimeError(f"rendered policy video has non-P frame {frame_idx}: {temporary}")
            temporary.replace(output)
            write_json(sidecar, {"job_hash": job_hash, "selected_seed": seed,
                                 "video_id": video_id, "trace_id": trace_id,
                                 "trace_start": 0, "gop_count": int(reference.gop_count),
                                 "frame_count": int(reference.frame_count), "actions": actions,
                                 "output_sha256": sha256(output),
                                 "render_method": "GOP stream-copy from offline action grid"})
        manifest.append({"video_id": video_id, "selected_seed": seed,
                         "trace_id": trace_id, "trace_start": 0,
                         "gop_count": int(reference.gop_count),
                         "frame_count": int(reference.frame_count),
                         "video_path": str(output.relative_to(ROOT)), "sha256": sha256(output)})
    pd.DataFrame(manifest).to_parquet(output_root / "manifest.parquet", index=False)
