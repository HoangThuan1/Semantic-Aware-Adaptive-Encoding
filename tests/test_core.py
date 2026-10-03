from __future__ import annotations

import numpy as np
import pytest

from vcu_surrogate.encoding import action_id
from vcu_surrogate.inference import _aligned_roi, _union_area
from vcu_surrogate.traces import packet_timestamps_to_bandwidth
from vcu_surrogate.transitions import _balanced_actions
from vcu_surrogate.rl_env import EpisodeMix


def test_action_mapping() -> None:
    assert action_id(0, 0, 4) == 0
    assert action_id(2, 3, 4) == 11
    assert action_id(7, 3, 4) == 31
    assert action_id(1, 0, 4) == 4


def test_roi_alignment_and_clipping() -> None:
    assert _aligned_roi(3, 5, 101, 99, 1920, 1080, 0.05) == (0, 0, 112, 112)
    assert _aligned_roi(1900, 1060, 1920, 1080, 1920, 1080, 0.05) == (1888, 1056, 1920, 1080)


def test_union_area_does_not_double_count_overlap() -> None:
    assert _union_area([(0, 0, 10, 10), (5, 0, 15, 10)]) == 150


def test_packet_trace_counts_repeats_and_empty_seconds() -> None:
    bandwidth = packet_timestamps_to_bandwidth(np.array([0, 0, 999, 2000, 2999]))
    assert bandwidth.tolist() == [0.036, 0.0, 0.024]
    with pytest.raises(RuntimeError, match="ordered"):
        packet_timestamps_to_bandwidth(np.array([1001, 999]))


def test_episode_mix_is_exact_70_30() -> None:
    mix = EpisodeMix(13)
    choices = [mix.next_is_synthetic() for _ in range(100)]
    assert all(sum(choices[start:start + 10]) == 3 for start in range(0, 100, 10))


def test_balanced_schedule_covers_every_action_per_block() -> None:
    actions = _balanced_actions(64, 32, 7)
    assert sorted(actions[:32]) == list(range(32))
    assert sorted(actions[32:]) == list(range(32))
