from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .common import project_path, write_json
from .rl_env import STATE_COLUMNS, WARMUP_ACTION, WARMUP_GOPS, reward_for


def _episode_seed(base_seed: int, video_id: str, trace_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{video_id}:{trace_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _balanced_actions(length: int, action_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = []
    while sum(len(block) for block in blocks) < length:
        blocks.append(rng.permutation(action_count))
    return np.concatenate(blocks)[:length]


def build_transitions(config: dict, schedule_path: str | None = None) -> None:
    artifacts = project_path(config, "artifacts")
    processed = project_path(config, "processed_data")
    grid_path = artifacts / "metrics" / "gop_action_grid.parquet"
    semantics_path = artifacts / "metadata" / "gop_semantics.parquet"
    traces_path = processed / "bandwidth_traces.parquet"
    if not all(path.is_file() for path in (grid_path, semantics_path, traces_path)):
        raise RuntimeError("grid, semantic metadata, or bandwidth traces missing")
    grid = pd.read_parquet(grid_path)
    semantics = pd.read_parquet(semantics_path)
    traces = pd.read_parquet(traces_path)
    action_count = len(config["encoder"]["bitrates_mbps"]) * len(config["encoder"]["roi_delta_qp"])
    base_seed = int(config["traces"]["split_seed"])
    external = pd.read_csv(schedule_path) if schedule_path else None
    required_schedule_columns = {"video_id", "trace_id", "gop_idx", "action_id"}
    if external is not None and not required_schedule_columns.issubset(external.columns):
        raise RuntimeError(f"action schedule must contain {sorted(required_schedule_columns)}")
    if external is not None and external.duplicated(["video_id", "trace_id", "gop_idx"]).any():
        raise RuntimeError("action schedule contains duplicate video/trace/GOP keys")
    reward_cfg = config["reward"]
    epsilon = float(reward_cfg["bandwidth_epsilon_mbps"])
    rows = []
    schedule_rows = []
    semantic_feature_columns = [
        column for column in semantics.columns
        if column not in {"video_id", "gop_idx", "frame_start", "frame_count"}
    ]
    if "semantic_score" not in semantic_feature_columns:
        raise RuntimeError("semantic metadata does not contain semantic_score")

    grid_lookup = grid.set_index(["video_id", "gop_idx", "action_id"])
    for video_id, semantic_video in semantics.groupby("video_id", sort=True):
        semantic_video = semantic_video.sort_values("gop_idx").set_index("gop_idx")
        for trace_id, trace in traces.groupby("trace_id", sort=True):
            trace = trace.sort_values("gop_idx").set_index("gop_idx")
            horizon = min(len(semantic_video), len(trace))
            if horizon < WARMUP_GOPS + 2:
                raise RuntimeError(f"episode too short: {video_id}/{trace_id}")
            if external is None:
                actions = _balanced_actions(horizon, action_count, _episode_seed(base_seed, video_id, trace_id))
                actions[:WARMUP_GOPS] = WARMUP_ACTION
            else:
                selection = external[(external.video_id == video_id) & (external.trace_id == trace_id)].sort_values("gop_idx")
                expected = list(range(horizon))
                if selection.gop_idx.tolist()[:horizon] != expected:
                    raise RuntimeError(f"schedule is incomplete for {video_id}/{trace_id}")
                actions = selection.action_id.to_numpy(dtype=int)[:horizon]
            if np.any(actions < 0) or np.any(actions >= action_count):
                raise RuntimeError(f"action outside [0,{action_count - 1}] in {video_id}/{trace_id}")
            if not np.all(actions[:WARMUP_GOPS] == WARMUP_ACTION):
                raise RuntimeError(f"first four GOPs must use warm-up action {WARMUP_ACTION}: {video_id}/{trace_id}")
            episode_id = f"{video_id}__{trace_id}"
            for k, action in enumerate(actions):
                schedule_rows.append({"episode_id": episode_id, "video_id": video_id, "trace_id": trace_id,
                                      "gop_idx": k, "action_id": int(action)})
            for k in range(WARMUP_GOPS, horizon):
                action = int(actions[k])
                previous_action = int(actions[k - 1])
                outcome = grid_lookup.loc[(video_id, k, action)]
                previous = grid_lookup.loc[(video_id, k - 1, previous_action)]
                semantic = semantic_video.loc[k]
                next_semantic = semantic_video.loc[k + 1] if k + 1 < horizon else None
                bandwidth = float(trace.loc[k].bandwidth_mbps)
                next_bandwidth = float(trace.loc[k + 1].bandwidth_mbps) if k + 1 < horizon else 0.0
                parts = reward_for(outcome, bandwidth, previous, reward_cfg)
                state_semantics = {column: float(semantic[column]) for column in semantic_feature_columns}
                next_state_semantics = {
                    f"next_{column}": float(next_semantic[column]) if next_semantic is not None else 0.0
                    for column in semantic_feature_columns
                }
                history = [float(trace.loc[j].bandwidth_mbps) for j in range(k - 4, k)]
                next_history = [float(trace.loc[j].bandwidth_mbps) for j in range(k - 3, k + 1)]
                rows.append({
                    "episode_id": episode_id, "split": trace.iloc[0]["split"],
                    "video_id": video_id, "trace_id": trace_id, "gop_idx": k,
                    **{name: value for name, value in zip(STATE_COLUMNS[:4], history)},
                    "bw_mbps": bandwidth, "r_prev_actual_mbps": float(previous.actual_bitrate_mbps),
                    **state_semantics,
                    "action_id": action, "target_bitrate_mbps": float(outcome.target_bitrate_mbps),
                    "delta_qp_roi": int(outcome.delta_qp_roi), "actual_bitrate_mbps": float(outcome.actual_bitrate_mbps),
                    **parts,
                    "target_error_fraction": abs(float(outcome.actual_bitrate_mbps) - float(outcome.target_bitrate_mbps)) / float(outcome.target_bitrate_mbps),
                    "bandwidth_outage": bandwidth < epsilon,
                    "next_bw_mbps": next_bandwidth,
                    **{f"next_{name}": value for name, value in zip(STATE_COLUMNS[:4], next_history)},
                    "next_r_prev_actual_mbps": float(outcome.actual_bitrate_mbps),
                    **next_state_semantics,
                    "done": k == horizon - 1,
                    "vmaf": float(outcome.vmaf), "psnr": float(outcome.psnr), "ssim": float(outcome.ssim),
                    "roi_psnr": float(outcome.roi_psnr) if not pd.isna(outcome.roi_psnr) else np.nan,
                    "roi_ssim": float(outcome.roi_ssim) if not pd.isna(outcome.roi_ssim) else np.nan,
                })
    if external is not None:
        external_keys = {
            (str(row.video_id), str(row.trace_id), int(row.gop_idx), int(row.action_id))
            for row in external.itertuples(index=False)
        }
        used_keys = {
            (row["video_id"], row["trace_id"], row["gop_idx"], row["action_id"])
            for row in schedule_rows
        }
        if external_keys != used_keys:
            raise RuntimeError("action schedule must match every generated episode horizon exactly")
    pd.DataFrame(rows).to_parquet(processed / "transitions.parquet", index=False)
    pd.DataFrame(schedule_rows).to_parquet(processed / "action_schedules.parquet", index=False)
    write_json(processed / "transition_schema.json", {
        "schema_version": 1,
        "gop_frames": int(config["video"]["gop_frames"]),
        "fps": int(config["video"]["fps"]),
        "state_columns": list(STATE_COLUMNS),
        "action_columns": ["action_id", "target_bitrate_mbps", "delta_qp_roi"],
        "next_state_columns": [
            *(f"next_{column}" for column in STATE_COLUMNS),
        ],
        "reward_column": "reward",
        "reward_config": config["reward"],
        "action_count": action_count,
    })
