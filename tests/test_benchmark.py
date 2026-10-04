from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from vcu_surrogate.benchmark_metrics import (box_mask, detection_match, fluid_queue,
                                           frame_offsets, matched_rate_saving, region_quality)
from vcu_surrogate.benchmark import causal_action, oracle_actions, semantic_rates
from vcu_surrogate.rl_env import reward_for


def test_samples_cover_edges_without_duplicates():
    assert frame_offsets(3).tolist() == [0, 14, 29]
    assert frame_offsets(30).tolist() == list(range(30))
    with pytest.raises(ValueError):
        frame_offsets(31)


def test_matching_is_class_aware_one_to_one_and_confidence_ordered():
    ref = np.array([[0, 0, 10, 10, .8, 2], [20, 20, 30, 30, .9, 1]])
    pred = np.array([[0, 0, 10, 10, .7, 2], [0, 0, 10, 10, .6, 2], [20, 20, 30, 30, .9, 2]])
    result = detection_match(ref, pred)
    assert result['matched_count'] == 1
    assert result['extra_count'] == 2
    assert result['missed_count'] == 1
    assert result['matches'] == [(0, 0)]
    assert result['confidence_ratio_sum'] == pytest.approx(.7/.8)
    empty = detection_match(np.empty((0, 6)), pred)
    assert empty['matched_count'] == 0
    rates = semantic_rates(pd.DataFrame([result, empty]))
    assert rates.iloc[0].detection_recall == .5
    assert np.isnan(rates.iloc[1].detection_recall)


def test_nonroi_is_complement_and_local_damage_stays_in_roi():
    ref = np.full((64, 64, 3), 100, dtype=np.uint8)
    mask = box_mask((64, 64), np.array([[16, 16, 48, 48], [20, 20, 30, 30]]))
    dist = ref.copy()
    dist[mask] = 150
    metrics = region_quality(ref, dist, mask)
    assert metrics['roi_pixels'] + metrics['nonroi_pixels'] == 64*64
    assert metrics['roi_pixels'] == 32*32
    assert metrics['nonroi_psnr_y'] == 100
    assert metrics['nonroi_ssim_y'] == pytest.approx(1)
    assert metrics['roi_psnr_y'] < 20
    assert metrics['roi_ssim_y'] < 1
    absent = region_quality(ref, ref, np.zeros((64, 64), dtype=bool))
    assert np.isnan(absent['roi_psnr_y'])
    assert absent['nonroi_ssim_y'] == pytest.approx(1)


def test_queue_outage_censor_and_no_banked_idle_capacity():
    result = fluid_queue(np.array([0., 2., 2.]), np.array([100., 0., 3.]))
    assert result['backlog_mbit'].tolist() == [0, 2, 1]
    assert result['completion_latency_s'][1] == pytest.approx(1+2/3)
    assert np.isnan(result['completion_latency_s'][2])
    assert result['deadline_miss'].tolist() == [False, True, True]
    normal = fluid_queue(np.array([1., 1.]), np.array([2., 2.]))
    assert normal['completion_latency_s'].tolist() == [.5, .5]
    assert normal['queue_area_mbit_s'].tolist() == [.25, .25]
    with pytest.raises(ValueError):
        fluid_queue(np.array([-1.]), np.array([1.]))


def test_matched_quality_saving_does_not_extrapolate():
    result = matched_rate_saving(np.array([1, 2, 4]), np.array([.5, .7, .9]),
                                np.array([.5, 1, 2]), np.array([.5, .7, .9]))
    assert result['saving_percent'] == pytest.approx(50)
    assert result['quality_low'] == .5
    no_overlap = matched_rate_saving(np.array([1, 2]), np.array([.1, .2]), np.array([1, 2]), np.array([.8, .9]))
    assert no_overlap['status'] == 'no_quality_overlap'


def test_heuristic_uses_history_and_handles_zero():
    bitrates = np.array([.5, 1, 1.5, 2, 3, 4, 6, 8])
    assert causal_action(np.array([10, 10, 10, 1]), .8, bitrates, .8, 'last', 'semantic') == 3
    assert causal_action(np.array([4, 0, 4, 4]), .1, bitrates, 1, 'harmonic', 0) == 0


def test_dp_oracle_matches_bruteforce_with_switch_cost():
    cfg = dict(full_vmaf=.45, roi_ssim=.25, bandwidth_violation=.2,
               bandwidth_utilization=.05, action_switch=.05, bandwidth_epsilon_mbps=.05, penalty_cap=4)
    rng = np.random.default_rng(10)
    outcomes = []
    for k in range(2):
        outcomes.append(pd.DataFrame([dict(action_id=a, bitrate_index=a//4, roi_index=a%4,
                                          actual_bitrate_mbps=.5+a/10, vmaf=float(rng.uniform(20, 90)),
                                          roi_ssim=.8, roi_frame_count=30) for a in range(32)]))
    previous = outcomes[0].iloc[4].drop("action_id")
    bw = np.array([1.0, 2.0])
    actions = oracle_actions(outcomes, bw, previous, cfg)
    def score(sequence):
        p = previous
        total = 0
        for k, a in enumerate(sequence):
            row = outcomes[k].iloc[a]
            total += reward_for(row, bw[k], p, cfg)['reward']
            p = row
        return total
    optimum = max(score(seq) for seq in itertools.product(range(32), repeat=2))
    assert score(actions) == pytest.approx(optimum)


def test_cuda_region_quality_matches_cpu_definition():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA parity requires GPU')
    from vcu_surrogate.benchmark_metrics import region_quality_cuda
    rng = np.random.default_rng(2)
    mask = box_mask((128, 128), np.array([[16, 16, 96, 96]]))
    for reference in (np.full((128, 128, 3), 100, dtype=np.uint8),
                      rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)):
        for distorted in (reference, np.clip(reference.astype(float)+rng.normal(0, 5, reference.shape), 0, 255).astype(np.uint8)):
            cpu = region_quality(reference, distorted, mask)
            gpu = region_quality_cuda(reference, distorted, mask)
            for key in ('roi_psnr_y', 'nonroi_psnr_y', 'roi_ssim_y', 'nonroi_ssim_y'):
                assert gpu[key] == pytest.approx(cpu[key], abs=1e-3)
