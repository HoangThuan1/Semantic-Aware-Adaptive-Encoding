from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcu_surrogate.rl_env import GridDataset, Normalizer, OfflineGOPEnv


class TinyGrid:
    action_count = 32
    video_by_split = {"train": ["clip"]}
    semantic_by_video = {"clip": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])}
    trace_by_split = {"train": {"trace": np.array([0., 1., 2., 3., 4., 5.])}}
    reward_config = {"bandwidth_epsilon_mbps": 0.05, "penalty_cap": 4.0,
                     "full_vmaf": 0.45, "roi_ssim": 0.25,
                     "bandwidth_violation": 0.20, "bandwidth_utilization": 0.05,
                     "action_switch": 0.05}

    def row(self, video_id: str, k: int, action: int) -> pd.Series:
        if action == 31:
            raise RuntimeError(f"missing action grid row: {video_id}/{k}/{action}")
        return pd.Series({"bitrate_index": action // 4, "roi_index": action % 4,
                          "target_bitrate_mbps": 1.0, "delta_qp_roi": 0,
                          "actual_bitrate_mbps": 1 + k * 0.01 + action / 100,
                          "vmaf": 80.0, "psnr": 35.0, "ssim": 0.9,
                          "roi_ssim": 0.8, "roi_psnr": 34.0, "roi_frame_count": 1})


def test_causal_state_warmup_and_previous_rate() -> None:
    normalizer = Normalizer(np.zeros(6), np.ones(6), 100.0)
    env = OfflineGOPEnv(TinyGrid(), "train", normalizer, 7)
    state, info = env.reset(options={"video_id": "clip", "trace_id": "trace", "trace_start": 0})
    assert info["warmup_gops"] == 4
    np.testing.assert_allclose(state, [0, 1, 2, 3, 1.07, 0.5])
    next_state, _, done, _, details = env.step(0)
    assert not done and details["bandwidth_mbps"] == 4.0
    assert details["gop_idx"] == 4
    np.testing.assert_allclose(next_state, [1, 2, 3, 4, 1.04, 0.6])
    _, _, done, _, _ = env.step(0)
    assert done


def test_missing_grid_action_fails() -> None:
    env = OfflineGOPEnv(TinyGrid(), "train", Normalizer(np.zeros(6), np.ones(6), 100), 7)
    env.reset(options={"video_id": "clip", "trace_id": "trace"})
    with pytest.raises(RuntimeError, match="missing action grid row"):
        env.step(31)


def test_normalizer_and_synthetic_use_only_training_bandwidth() -> None:
    dataset = GridDataset.__new__(GridDataset)
    dataset.trace_by_split = {
        "train": {"att": np.arange(1., 21.), "tmobile": np.arange(21., 41.)},
        "validation": {"held_out": np.full(40, 100.)},
        "test": {"verizon": np.full(40, 1000.)},
    }
    dataset.video_by_split = {"train": ["clip"]}
    dataset.trace_source = {"att": "fcc", "tmobile": "norway"}
    dataset.grid = pd.DataFrame({"video_id": ["clip"] * 3 + ["test"], "actual_bitrate_mbps": [1., 2., 3., 1000.]})
    dataset.semantics = pd.DataFrame({"video_id": ["clip"] * 3 + ["test"], "semantic_score": [0.1, 0.5, 0.9, 1.]})
    dataset.state_clip = 10.0
    dataset.block_seconds = 10
    normalizer = dataset.normalizer()
    assert normalizer.mean[0] == 20.5
    assert normalizer.mean[4] == 2.0
    assert normalizer.mean[5] == .5
    trace, lineage = dataset.synthetic(25, np.random.default_rng(13))
    assert len(trace) == 25 and len(lineage) == 3
    assert {item["trace_id"] for item in lineage} <= {"att", "tmobile"}
    assert trace.max() <= 40.
