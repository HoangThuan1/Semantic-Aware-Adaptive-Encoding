from __future__ import annotations

from pathlib import Path

import numpy as np

STATE_COLUMNS = ("bw_k_minus_4", "bw_k_minus_3", "bw_k_minus_2", "bw_k_minus_1",
                 "r_prev_actual_mbps", "semantic_score")
ACTION_TABLE = np.array([(bitrate, delta) for bitrate in (0.5, 1, 1.5, 2, 3, 4, 6, 8)
                         for delta in (0, -3, -6, -9)], dtype=np.float32)


class NumpyPolicy:
    """Board-side DDQN inference; requires only NumPy."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            if (int(data["schema_version"]) != 1 or int(data["gop_frames"]) != 30
                    or int(data["fps"]) != 30):
                raise RuntimeError("unsupported DDQN policy schema or GOP")
            self.state_columns = tuple(str(value) for value in data["state_columns"])
            self.mean = data["normalizer_mean"].astype(np.float64)
            self.std = data["normalizer_std"].astype(np.float64)
            self.clip = float(data["normalizer_clip"])
            self.action_table = data["action_table"].astype(np.float32)
            self.warmup_gops = int(data["warmup_gops"])
            self.warmup_action = int(data["warmup_action"])
            self.weights = tuple(data[f"weight_{i}"].astype(np.float32) for i in range(3))
            self.biases = tuple(data[f"bias_{i}"].astype(np.float32) for i in range(3))
        if (self.state_columns != STATE_COLUMNS or self.mean.shape != (6,) or self.std.shape != (6,)
                or self.action_table.shape != (32, 2) or self.warmup_gops != 4
                or self.warmup_action != 4 or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all() or (self.std <= 0).any()
                or not np.array_equal(self.action_table, ACTION_TABLE)
                or tuple(weight.shape for weight in self.weights) != ((128, 6), (128, 128), (32, 128))
                or tuple(bias.shape for bias in self.biases) != ((128,), (128,), (32,))):
            raise RuntimeError("invalid DDQN policy contents")

    def q_values(self, raw_states: np.ndarray) -> np.ndarray:
        states = np.asarray(raw_states, dtype=np.float64)
        if states.ndim == 1:
            states = states[None, :]
        if states.ndim != 2 or states.shape[1] != 6 or not np.isfinite(states).all():
            raise ValueError("policy input must have shape (batch, 6) and be finite")
        x = np.clip((states - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if index < 2:
                x = np.maximum(x, 0)
        return x

    def act(self, raw_state: np.ndarray) -> tuple[int, float, int]:
        action = int(self.q_values(raw_state)[0].argmax())
        bitrate, delta_qp = self.action_table[action]
        return action, float(bitrate), int(delta_qp)
