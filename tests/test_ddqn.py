from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from vcu_surrogate.ddqn import QNetwork, _check_policy_parity, _export_npz, double_dqn_targets  # noqa: E402
from vcu_surrogate.rl_env import Normalizer  # noqa: E402
from vcu_surrogate.policy_numpy import NumpyPolicy  # noqa: E402


class FixedQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, states):
        return self.values.expand(len(states), -1)


def test_double_dqn_uses_online_argmax_and_target_value() -> None:
    online = FixedQ([1, 9, 2])
    target = FixedQ([10, 3, 20])
    states = torch.zeros((2, 6))
    values = double_dqn_targets(online, target, states, torch.tensor([2., 4.]),
                                torch.tensor([0., 1.]), 0.5)
    assert values.tolist() == [3.5, 4.0]


def test_numpy_policy_matches_pytorch(tmp_path) -> None:
    torch.manual_seed(13)
    network = QNetwork()
    normalizer = Normalizer(np.array([1., 2., 3., 4., 5., 0.5]),
                            np.array([2., 2., 2., 2., 3., 0.25]), 10.0)
    config = {"video": {"gop_frames": 30, "fps": 30},
              "encoder": {"bitrates_mbps": [0.5, 1, 1.5, 2, 3, 4, 6, 8],
                          "roi_delta_qp": [0, -3, -6, -9]}}
    policy = tmp_path / "policy.npz"
    _export_npz(policy, network, normalizer, config)
    raw_states = np.random.default_rng(13).normal(size=(128, 6))
    assert _check_policy_parity(network, policy, raw_states, normalizer) < 1e-5
    action, bitrate, delta = NumpyPolicy(policy).act(raw_states[0])
    assert 0 <= action < 32
    assert bitrate == config["encoder"]["bitrates_mbps"][action // 4]
    assert delta == config["encoder"]["roi_delta_qp"][action % 4]
