from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from vcu_surrogate.transitions import build_transitions  # noqa: E402


def test_transition_schema_is_causal_and_warmup_is_fixed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("vcu_surrogate.transitions.video_manifest", lambda config: pd.DataFrame({"video_id": ["clip"], "split": ["train"]}))
    monkeypatch.setattr("vcu_surrogate.transitions.dataset_hash", lambda config: "fixture-dataset")
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    (artifacts / "metrics").mkdir(parents=True)
    (artifacts / "metadata").mkdir(parents=True)
    processed.mkdir()
    rows = []
    for k in range(6):
        for action in range(32):
            rows.append({"video_id": "clip", "gop_idx": k, "action_id": action,
                         "bitrate_index": action // 4, "roi_index": action % 4,
                         "target_bitrate_mbps": 1.0, "delta_qp_roi": 0,
                         "actual_bitrate_mbps": 1.0 + k / 100 + action / 1000,
                         "roi_frame_count": 1, "roi_ssim": 0.8, "roi_psnr": 34.0,
                         "vmaf": 80.0, "psnr": 35.0, "ssim": 0.9})
    pd.DataFrame(rows).to_parquet(artifacts / "metrics" / "gop_action_grid.parquet", index=False)
    pd.DataFrame({"video_id": ["clip"] * 6, "gop_idx": range(6),
                  "semantic_score": np.linspace(0.1, 0.6, 6)}).to_parquet(
                      artifacts / "metadata" / "gop_semantics.parquet", index=False)
    pd.DataFrame({"trace_id": ["trace"] * 6, "split": ["train"] * 6,
                  "gop_idx": range(6), "bandwidth_mbps": [0., 1., 2., 3., 4., 5.]}).to_parquet(
                      processed / "bandwidth_traces.parquet", index=False)
    config = {"paths": {"artifacts": str(artifacts), "processed_data": str(processed)},
              "video": {"gop_frames": 30, "fps": 30},
              "encoder": {"bitrates_mbps": [0.5, 1, 1.5, 2, 3, 4, 6, 8],
                          "roi_delta_qp": [0, -3, -6, -9]},
              "traces": {"split_seed": 13},
              "reward": {"full_vmaf": 0.45, "roi_ssim": 0.25,
                         "bandwidth_violation": 0.2, "bandwidth_utilization": 0.05,
                         "action_switch": 0.05, "bandwidth_epsilon_mbps": 0.05,
                         "penalty_cap": 4.0}}
    build_transitions(config)
    transitions = pd.read_parquet(processed / "transitions.parquet")
    schedules = pd.read_parquet(processed / "action_schedules.parquet")
    assert schedules.action_id.iloc[:4].tolist() == [4] * 4
    assert transitions.gop_idx.tolist() == [4, 5]
    assert transitions.iloc[0][[f"bw_k_minus_{n}" for n in (4, 3, 2, 1)]].tolist() == [0., 1., 2., 3.]
    assert transitions.bw_mbps.tolist() == [4., 5.]
    assert transitions.done.tolist() == [False, True]
    assert transitions.r_prev_actual_mbps.iloc[0] == 1.034
