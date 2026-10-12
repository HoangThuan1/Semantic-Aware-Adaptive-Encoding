from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from vcu_surrogate.benchmark_metrics import (box_mask, detection_match, fluid_queue,
                                           frame_offsets, matched_rate_saving, region_quality)
from vcu_surrogate.benchmark import oracle_actions, semantic_rates
from vcu_surrogate.rl_env import bandwidth_penalty, reward_for


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


@pytest.mark.parametrize('violation,switch', [(.2, .05), (.5, .05), (.6, .1), (1., .05)])
@pytest.mark.parametrize('shape', ['quadratic', 'linear_shoulder'])
def test_dp_oracle_matches_bruteforce_with_switch_cost(violation, switch, shape):
    cfg = dict(full_vmaf=.45, roi_ssim=.25, bandwidth_violation=violation,
               bandwidth_utilization=.05, action_switch=switch, bandwidth_epsilon_mbps=.05, penalty_cap=4,
               bandwidth_penalty_shape=shape)
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


def test_linear_shoulder_penalty_is_stronger_continuous_and_capped():
    cfg = dict(penalty_cap=4, bandwidth_penalty_shape='linear_shoulder')
    ratios = np.array([0, 1, 1.1, 1.2, 1.5, 2, 4])
    assert bandwidth_penalty(ratios, cfg) == pytest.approx([0, 0, .1, .2, .5, 1.25, 4])
    assert float(bandwidth_penalty(1.5-1e-8, cfg)) == pytest.approx(.5, abs=2e-8)
    assert float(bandwidth_penalty(1.5+1e-8, cfg)) == pytest.approx(.5, abs=2e-8)
    assert np.all(bandwidth_penalty(ratios, cfg) >= bandwidth_penalty(ratios, dict(penalty_cap=4)))
    with pytest.raises(ValueError):
        bandwidth_penalty(1.2, dict(penalty_cap=4, bandwidth_penalty_shape='unknown'))


@pytest.mark.parametrize('strategy', [
    {'cbr': 4}, {'cbr': 31},
    {'vbr': dict(base_mbps=1.5, gain=1., roi_mode='semantic')},
])
def test_baseline_candidate_returns_match_full_rollout_with_outage_and_padding(strategy):
    from vcu_surrogate.benchmark import (COUNT_COLUMNS, QUALITY_COLUMNS,
                                        baseline_validation_returns, evaluate_strategy)
    from vcu_surrogate.rl_env import GridDataset
    dataset = GridDataset.__new__(GridDataset)
    dataset.reward_config = dict(full_vmaf=.45, roi_ssim=.25, bandwidth_violation=1.,
        bandwidth_utilization=0., action_switch=.05, bandwidth_epsilon_mbps=.05,
        penalty_cap=4, bandwidth_penalty_shape='linear_shoulder')
    dataset.semantic_by_video = {'clip': np.array([.1, .2, .3, .4, .1, .8, .5, 1.])}
    dataset.trace_by_split = {'validation': {'trace': np.array([2., 0, 1, .75, 1, 0, .4, 3, 2, 1, 1])}}
    bitrates = [.5, 1, 1.5, 2, 3, 4, 6, 8]
    grid = pd.DataFrame([dict(video_id='clip', gop_idx=k, action_id=a,
        bitrate_index=a//4, roi_index=a%4, target_bitrate_mbps=bitrates[a//4],
        delta_qp_roi=[0, -3, -6, -9][a%4], actual_bitrate_mbps=.35 + a*.12,
        vmaf=40+a, roi_ssim=.7+a*.002 if k != 5 else np.nan,
        roi_frame_count=30 if k != 5 else 0,
        **{c: 1. for c in (*COUNT_COLUMNS, *QUALITY_COLUMNS)})
        for k in range(8) for a in range(32)])
    specs = [dict(split='validation', video_id='clip', trace_id='trace', trace_start=offset)
             for offset in (0, 3)]
    scores = baseline_validation_returns(dataset, grid, specs, **strategy)
    episodes, rollout = evaluate_strategy(dataset, grid, specs, 'candidate', **strategy)
    assert scores == pytest.approx(episodes['return'].to_numpy())
    if 'cbr' in strategy:
        assert (rollout.actual_bitrate_mbps >= rollout.target_bitrate_mbps).all()
        assert (rollout.padding_mbps > 0).any()
    else:
        assert (rollout.padding_mbps == 0).all()


def test_vbr_semantic_proxy_is_bounded_and_has_no_bandwidth_input():
    from vcu_surrogate.benchmark import vbr_action
    bitrates = np.array([.5, 1, 1.5, 2, 3, 4, 6, 8])
    assert vbr_action(np.array([0., .5, 1.]), bitrates, 1., 1., 'semantic').tolist() == [0, 6, 15]


def test_fixed_roi_selects_best_allowed_q_instead_of_overwriting_global_argmax():
    from vcu_surrogate.benchmark import fixed_roi_policy_action
    class Policy:
        def q_values(self, state):
            q = np.zeros((1, 32))
            q[0, 31] = 100  # Global best is not an allowed ROI action.
            q[0, 6] = 10
            q[0, 30] = 5
            return q
    assert fixed_roi_policy_action(Policy(), np.zeros(6), 2) == 6
    with pytest.raises(ValueError):
        fixed_roi_policy_action(Policy(), np.zeros(6), 4)


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


def test_streamed_policy_summary_matches_single_frame(tmp_path):
    pytest.importorskip("pyarrow")
    from vcu_surrogate.benchmark import summarize_policies
    from vcu_surrogate.benchmark import COUNT_COLUMNS, QUALITY_COLUMNS
    parts, episodes = [], []
    for name in ('ddqn_selected', 'cbr_validation_selected'):
        fields = {key: [1., 1.] for key in (*COUNT_COLUMNS, *QUALITY_COLUMNS)}
        fields.update(strategy=[name] * 2, split=['test'] * 2, video_id=['clip'] * 2,
                      trace_id=['trace'] * 2, trace_start=[0] * 2,
                      reward=[.1, .3], vmaf=[80., 82.], roi_ssim=[.8, .9],
                      actual_bitrate_mbps=[1., 1.], bandwidth_violation=[False, True],
                      penalty_action_switch=[0., .25], deadline_miss=[False, True],
                      uncompleted=[False, True], completion_latency_s=[.5, np.nan],
                      backlog_mbit=[0., 1.], excess_mbit=[0., 1.], served_mbit=[1., 0.],
                      unused_capacity_mbit=[1., 0.], bandwidth_mbps=[2., 0.], post_drop=[False, True])
        parts.append(pd.DataFrame(fields))
        episodes.append(dict(strategy=name, split='test', video_id='clip', trace_id='trace', trace_start=0,
                             gops=2, return_value=.4, reward_per_gop=.2, vmaf=81.,
                             actual_bitrate_mbps=1., bandwidth_violation=.5, deadline_miss=.5))
    episode_table = pd.DataFrame(episodes).rename(columns={'return_value': 'return'})
    combined = pd.concat(parts, ignore_index=True)
    a = summarize_policies(episode_table, combined, tmp_path / 'single')
    b = summarize_policies(episode_table, iter(parts), tmp_path / 'streamed')
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'streamed/policy_rollouts.parquet'), combined)
    comparisons = pd.read_csv(tmp_path / 'streamed/paired_policy_comparisons.csv')
    assert (comparisons.mean_paired_episode_difference == 0).all()


def test_nms_timeout_retries_every_image_without_treating_empty_as_complete():
    import logging
    from types import SimpleNamespace
    from vcu_surrogate.benchmark import _predict_complete
    logger=logging.getLogger('nms-timeout-test')
    calls=[]
    def model(images,size):
        calls.append(list(images))
        if len(images)>1:
            logger.warning('WARNING NMS time limit 0.900s exceeded')
            return SimpleNamespace(xyxy=[np.array([images[0]]),np.empty(0),np.empty(0)])
        return SimpleNamespace(xyxy=[np.array([images[0]])])
    original_handlers=list(logger.handlers)
    output,retried=_predict_complete(model,[10,20,30],640,logger)
    assert retried and calls==[[10,20,30],[10],[20],[30]]
    assert [x.tolist() for x in output]==[[10],[20],[30]]
    assert logger.handlers==original_handlers


def test_nms_normal_empty_predictions_are_kept_and_handler_cleans_up_on_error():
    import logging
    from types import SimpleNamespace
    from vcu_surrogate.benchmark import _predict_complete
    logger=logging.getLogger('nms-normal-test')
    calls=[]
    def model(images,size):
        calls.append(images)
        return SimpleNamespace(xyxy=[np.empty((0,6)) for image in images])
    output,retried=_predict_complete(model,[10,20],640,logger)
    assert not retried and len(calls)==1 and len(output)==2
    original_handlers=list(logger.handlers)
    def broken(images,size):raise RuntimeError('model failed')
    with pytest.raises(RuntimeError,match='model failed'):
        _predict_complete(broken,[10],640,logger)
    assert logger.handlers==original_handlers
