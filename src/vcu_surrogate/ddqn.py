from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common import project_path, write_json
from .policy_numpy import NumpyPolicy
from .rl_env import (EpisodeMix, GridDataset, Normalizer, OfflineGOPEnv,
                     STATE_COLUMNS, WARMUP_ACTION, WARMUP_GOPS)


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, 32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def double_dqn_targets(online: QNetwork, target: QNetwork, next_states: torch.Tensor,
                       rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    with torch.no_grad():
        best_actions = online(next_states).argmax(dim=1, keepdim=True)
        next_values = target(next_states).gather(1, best_actions).squeeze(1)
        return rewards + gamma * (1.0 - dones) * next_values


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.states = np.empty((capacity, 6), dtype=np.float32)
        self.next_states = np.empty((capacity, 6), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.integers(0, self.size, size=batch_size)
        return tuple(torch.as_tensor(values[indices], device=device) for values in
                     (self.states, self.actions, self.rewards, self.next_states, self.dones))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _greedy(network: QNetwork, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        tensor = torch.as_tensor(state[None, :], device=device)
        return int(network(tensor).argmax(dim=1).item())


def _run_evaluation(dataset: GridDataset, normalizer: Normalizer, network: QNetwork,
                    split: str, device: torch.device, seed: int) -> tuple[dict, pd.DataFrame]:
    env = OfflineGOPEnv(dataset, split, normalizer, seed)
    rows = []
    episode_returns = []
    for episode_index, spec in enumerate(dataset.evaluation_specs(split)):
        state, _ = env.reset(options=spec)
        total = 0.0
        done = False
        while not done:
            action = _greedy(network, state, device)
            state, reward, done, _, info = env.step(action)
            info["episode_index"] = episode_index
            info["episode_id"] = f"{info['video_id']}__{info['trace_id']}__{info['trace_start']}"
            rows.append(info)
            total += reward
        episode_returns.append(total)
    result = pd.DataFrame(rows)
    if not np.isfinite(episode_returns).all():
        raise RuntimeError(f"non-finite {split} episode return")
    valid_roi = result.roi_ssim.dropna()
    summary = {
        "split": split, "episodes": len(episode_returns),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_vmaf": float(result.vmaf.mean()),
        "mean_roi_ssim": float(valid_roi.mean()) if len(valid_roi) else None,
        "mean_actual_bitrate_mbps": float(result.actual_bitrate_mbps.mean()),
        "bandwidth_violation_rate": float((result.actual_bitrate_mbps > result.bandwidth_mbps).mean()),
        "action_switch_rate": float((result.penalty_action_switch > 0).mean()),
        "action_counts": {str(k): int(v) for k, v in sorted(Counter(result.action_id).items())},
    }
    return summary, result


def _checkpoint(path: Path, network: QNetwork, target: QNetwork,
                optimizer: torch.optim.Optimizer, normalizer: Normalizer,
                config: dict, seed: int, step: int, score: float, env: OfflineGOPEnv,
                mix: EpisodeMix, replay: ReplayBuffer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "online": network.state_dict(), "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std,
        "normalizer_clip": normalizer.clip, "config": config, "seed": seed,
        "step": step, "validation_mean_return": score,
        "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_env": env.rng.bit_generator.state, "rng_mix": mix.rng.bit_generator.state,
        "rng_replay": replay.rng.bit_generator.state,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _export_npz(path: Path, network: QNetwork, normalizer: Normalizer, config: dict) -> None:
    weights = [module for module in network.net if isinstance(module, nn.Linear)]
    bitrates = config["encoder"]["bitrates_mbps"]
    deltas = config["encoder"]["roi_delta_qp"]
    action_table = np.array([(b, q) for b in bitrates for q in deltas], dtype=np.float32)
    arrays = {"schema_version": np.array(1, dtype=np.int32),
              "state_columns": np.array(STATE_COLUMNS),
              "normalizer_mean": normalizer.mean.astype(np.float64),
              "normalizer_std": normalizer.std.astype(np.float64),
              "normalizer_clip": np.array(normalizer.clip, dtype=np.float32),
              "action_table": action_table,
              "gop_frames": np.array(config["video"]["gop_frames"], dtype=np.int32),
              "fps": np.array(config["video"]["fps"], dtype=np.int32),
              "warmup_gops": np.array(WARMUP_GOPS, dtype=np.int32),
              "warmup_action": np.array(WARMUP_ACTION, dtype=np.int32)}
    for index, module in enumerate(weights):
        arrays[f"weight_{index}"] = module.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"bias_{index}"] = module.bias.detach().cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def _check_policy_parity(network: QNetwork, npz: Path, states: np.ndarray,
                         normalizer: Normalizer) -> float:
    normalized = np.stack([normalizer.transform(state) for state in states])
    reference = QNetwork()
    reference.load_state_dict(network.state_dict())
    reference.eval()
    with torch.no_grad():
        expected = reference(torch.as_tensor(normalized, dtype=torch.float32)).numpy()
    actual = NumpyPolicy(npz).q_values(states)
    difference = float(np.max(np.abs(expected - actual)))
    if difference > 1e-5 or not np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1)):
        raise RuntimeError(f"PyTorch/NPZ policy mismatch: max Q error={difference}")
    return difference


def _test_states(rollouts: pd.DataFrame) -> np.ndarray:
    return np.stack([np.asarray(state, dtype=np.float64) for state in rollouts.state])


def train(config: dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for DDQN training; no CPU fallback")
    dataset = GridDataset(config)
    normalizer = dataset.normalizer()
    rl = config["rl"]
    reports = project_path(config, "reports") / "rl"
    models = project_path(config, "artifacts") / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    outcomes = []
    for seed in rl["seeds"]:
        seed = int(seed)
        _seed_all(seed)
        env = OfflineGOPEnv(dataset, "train", normalizer, seed)
        mix = EpisodeMix(seed)
        replay = ReplayBuffer(int(rl["replay_capacity"]), seed)
        online = QNetwork().to(device)
        target = QNetwork().to(device)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=float(rl["learning_rate"]))
        state, episode_info = env.reset(options={"synthetic": mix.next_is_synthetic()})
        episodes = [{"episode_index": 0, "start_step": 1, **episode_info,
                     "bandwidth_mbps": env.bandwidth.tolist() if env.synthetic else None}]
        best_score = -float("inf")
        history = []
        seed_path = models / f"ddqn_seed_{seed}.pt"
        for step in range(1, int(rl["steps_per_seed"]) + 1):
            epsilon = 1.0 - (1.0 - float(rl["epsilon_final"])) * min(
                step, int(rl["epsilon_decay_steps"])) / int(rl["epsilon_decay_steps"])
            action = int(env.rng.integers(0, 32)) if env.rng.random() < epsilon else _greedy(online, state, device)
            next_state, reward, done, _, _ = env.step(action)
            replay.add(state, action, reward, next_state, done)
            state = next_state
            if replay.size >= int(rl["learning_starts"]):
                states, actions, rewards, next_states, dones = replay.sample(int(rl["batch_size"]), device)
                target_values = double_dqn_targets(online, target, next_states, rewards, dones, float(rl["gamma"]))
                estimate = online(states).gather(1, actions[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(estimate, target_values)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
            if step % int(rl["target_sync_interval"]) == 0:
                target.load_state_dict(online.state_dict())
            if done and step < int(rl["steps_per_seed"]):
                state, episode_info = env.reset(options={"synthetic": mix.next_is_synthetic()})
                episodes.append({"episode_index": len(episodes), "start_step": step + 1,
                                 **episode_info,
                                 "bandwidth_mbps": env.bandwidth.tolist() if env.synthetic else None})
            if step % int(rl["validation_interval"]) == 0:
                summary, _ = _run_evaluation(dataset, normalizer, online, "validation", device, seed)
                score = summary["mean_return"]
                history.append({"seed": seed, "step": step, "epsilon": epsilon, **summary})
                if score > best_score:
                    best_score = score
                    _checkpoint(seed_path, online, target, optimizer, normalizer, config,
                                seed, step, score, env, mix, replay)
                print(f"seed={seed} step={step} validation_return={score:.4f} best={best_score:.4f}", flush=True)
        pd.DataFrame(history).to_csv(reports / f"learning_seed_{seed}.csv", index=False)
        if not seed_path.is_file() or not np.isfinite(best_score):
            raise RuntimeError(f"training produced no valid validation checkpoint for seed {seed}")
        pd.DataFrame(episodes).to_json(reports / f"episode_lineage_seed_{seed}.json",
                                       orient="records", indent=2)
        outcomes.append({"seed": seed, "best_validation_mean_return": best_score,
                         "checkpoint": str(seed_path)})
    write_json(reports / "training_summary.json", {"seeds": outcomes,
               "selected_seed": max(outcomes, key=lambda x: x["best_validation_mean_return"])["seed"]})


def evaluate(config: dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for DDQN evaluation; no CPU fallback")
    dataset = GridDataset(config)
    reports = project_path(config, "reports") / "rl"
    models = project_path(config, "artifacts") / "rl"
    training = json.loads((reports / "training_summary.json").read_text())
    device = torch.device("cuda:0")
    summaries = []
    for seed in config["rl"]["seeds"]:
        path = models / f"ddqn_seed_{seed}.pt"
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload["seed"] != seed or payload["step"] > config["rl"]["steps_per_seed"]:
            raise RuntimeError(f"invalid DDQN checkpoint: {path}")
        if payload["config"]["_config_sha256"] != config["_config_sha256"]:
            raise RuntimeError(f"DDQN checkpoint configuration mismatch: {path}")
        normalizer = Normalizer(payload["normalizer_mean"], payload["normalizer_std"],
                                payload["normalizer_clip"])
        online = QNetwork().to(device)
        online.load_state_dict(payload["online"])
        online.eval()
        validation, validation_rows = _run_evaluation(dataset, normalizer, online, "validation", device, seed)
        if abs(validation["mean_return"] - payload["validation_mean_return"]) > 1e-6:
            raise RuntimeError(f"validation checkpoint score changed: {path}")
        test, test_rows = _run_evaluation(dataset, normalizer, online, "test", device, seed)
        validation_rows.to_parquet(reports / f"validation_rollout_seed_{seed}.parquet", index=False)
        test_rows.to_parquet(reports / f"test_rollout_seed_{seed}.parquet", index=False)
        policy_path = models / f"ddqn_seed_{seed}.npz"
        _export_npz(policy_path, online, normalizer, config)
        parity = _check_policy_parity(online, policy_path, _test_states(test_rows), normalizer)
        summary = {"seed": seed, "checkpoint_step": payload["step"],
                   "validation": validation, "test": test, "max_policy_q_error": parity}
        write_json(reports / f"evaluation_seed_{seed}.json", summary)
        summaries.append(summary)
    chosen = max(summaries, key=lambda x: x["validation"]["mean_return"])
    if chosen["seed"] != training["selected_seed"]:
        raise RuntimeError("selected policy differs from validation-only training selection")
    metrics = ["mean_return", "mean_vmaf", "mean_roi_ssim", "mean_actual_bitrate_mbps",
               "bandwidth_violation_rate", "action_switch_rate"]
    aggregate = {}
    for name in metrics:
        values = [s["test"][name] for s in summaries]
        if all(value is None for value in values):
            aggregate[name] = {"mean": None, "std": None}
        elif any(value is None for value in values):
            raise RuntimeError(f"inconsistent test metric across seeds: {name}")
        else:
            aggregate[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    selected_policy = export_policy(config)
    write_json(reports / "evaluation_summary.json", {
        "selected_seed": chosen["seed"], "selected_policy": str(selected_policy),
        "test_aggregate": aggregate, "seeds": summaries,
    })


def export_policy(config: dict) -> Path:
    reports = project_path(config, "reports") / "rl"
    models = project_path(config, "artifacts") / "rl"
    training = json.loads((reports / "training_summary.json").read_text())
    seed = int(training["selected_seed"])
    if seed not in config["rl"]["seeds"]:
        raise RuntimeError("selected seed not in DDQN configuration")
    payload = torch.load(models / f"ddqn_seed_{seed}.pt", map_location="cpu", weights_only=False)
    if payload["seed"] != seed or payload["config"]["_config_sha256"] != config["_config_sha256"]:
        raise RuntimeError("selected DDQN checkpoint does not match training configuration")
    network = QNetwork()
    network.load_state_dict(payload["online"])
    network.eval()
    normalizer = Normalizer(payload["normalizer_mean"], payload["normalizer_std"],
                            payload["normalizer_clip"])
    path = models / "ddqn_selected.npz"
    _export_npz(path, network, normalizer, config)
    print(f"exported selected policy: {path}")
    return path
