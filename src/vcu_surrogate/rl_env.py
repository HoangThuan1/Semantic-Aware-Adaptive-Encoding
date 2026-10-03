from __future__ import annotations

import json
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .assets import LTE_UPLINK_SHA256, assert_downloads
from .common import project_path
from .traces import TRACE_SPLITS, parse_mahimahi_trace

STATE_COLUMNS = ("bw_k_minus_4", "bw_k_minus_3", "bw_k_minus_2", "bw_k_minus_1",
                 "r_prev_actual_mbps", "semantic_score")
WARMUP_GOPS = 4
WARMUP_ACTION = 4


class EpisodeMix:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.block: list[bool] = []

    def next_is_synthetic(self) -> bool:
        if not self.block:
            self.block = [False] * 7 + [True] * 3
            self.rng.shuffle(self.block)
        return self.block.pop()


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    clip: float

    def transform(self, state: np.ndarray) -> np.ndarray:
        return np.clip((state - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)


class GridDataset:
    def __init__(self, config: dict):
        assert_downloads(config)
        artifacts = project_path(config, "artifacts")
        processed = project_path(config, "processed_data")
        schema_path = processed / "transition_schema.json"
        if not schema_path.is_file():
            raise RuntimeError(f"transition schema missing: {schema_path}")
        schema = json.loads(schema_path.read_text())
        if (schema.get("schema_version") != 1 or schema.get("gop_frames") != 30 or schema.get("fps") != 30
                or schema.get("state_columns") != list(STATE_COLUMNS)
                or schema.get("action_count") != 32):
            raise RuntimeError("transition schema does not match GOP30 DDQN contract")
        self.grid = pd.read_parquet(artifacts / "metrics" / "gop_action_grid.parquet")
        self.semantics = pd.read_parquet(artifacts / "metadata" / "gop_semantics.parquet")
        self.traces = pd.read_parquet(processed / "bandwidth_traces.parquet")
        catalog = pd.read_parquet(processed / "trace_catalog.parquet")
        expected_catalog = {(name.removesuffix(".up"), name, TRACE_SPLITS.get(name, "excluded"), digest)
                            for name, digest in LTE_UPLINK_SHA256.items()}
        observed_catalog = set(catalog[["trace_id", "source_file", "split", "raw_sha256"]]
                               .itertuples(index=False, name=None))
        if observed_catalog != expected_catalog or len(catalog) != len(expected_catalog):
            raise RuntimeError("trace catalog does not match pinned LTE uplink split and SHA-256")
        action_space = pd.read_parquet(artifacts / "encodes" / "action_space.parquet")
        expected_actions = {(i * len(config["encoder"]["roi_delta_qp"]) + j, float(bitrate), int(delta))
                            for i, bitrate in enumerate(config["encoder"]["bitrates_mbps"])
                            for j, delta in enumerate(config["encoder"]["roi_delta_qp"])}
        observed_actions = set(action_space[["action_id", "target_bitrate_mbps", "delta_qp_roi"]]
                               .itertuples(index=False, name=None))
        if observed_actions != expected_actions or len(action_space) != len(expected_actions):
            raise RuntimeError("action space does not match pinned 32-action contract")
        self.action_count = len(config["encoder"]["bitrates_mbps"]) * len(config["encoder"]["roi_delta_qp"])
        if self.action_count != 32:
            raise RuntimeError("DDQN requires the pinned 32-action grid")
        required_grid = {"video_id", "gop_idx", "action_id", "bitrate_index", "roi_index",
                         "target_bitrate_mbps", "delta_qp_roi", "actual_bitrate_mbps", "vmaf",
                         "psnr", "ssim", "roi_ssim", "roi_psnr", "roi_frame_count"}
        if not required_grid.issubset(self.grid.columns):
            raise RuntimeError(f"action grid lacks columns: {sorted(required_grid - set(self.grid.columns))}")
        if self.grid.duplicated(["video_id", "gop_idx", "action_id"]).any():
            raise RuntimeError("duplicate action grid key")
        if not np.isfinite(self.grid[["actual_bitrate_mbps", "vmaf", "psnr", "ssim"]].to_numpy(dtype=float)).all():
            raise RuntimeError("action grid contains non-finite mandatory outcomes")
        roi_rows = self.grid.roi_frame_count > 0
        if roi_rows.any() and not np.isfinite(self.grid.loc[roi_rows, "roi_ssim"].to_numpy(dtype=float)).all():
            raise RuntimeError("action grid contains non-finite ROI SSIM for present ROI")
        if (not (self.grid.frame_count == 30).all()
                or not (self.grid.packet_count == 30).all()
                or not (self.grid.key_packets == 1).all()):
            raise RuntimeError("action grid has incompatible GOP structure")
        self.grid_lookup = self.grid.set_index(["video_id", "gop_idx", "action_id"])
        if set(self.grid.video_id) != set(self.semantics.video_id):
            raise RuntimeError("grid/semantic video IDs differ")
        if self.semantics.duplicated(["video_id", "gop_idx"]).any():
            raise RuntimeError("duplicate semantic GOP key")
        self.semantic_by_video = {}
        for video_id, frame in self.semantics.groupby("video_id"):
            frame = frame.sort_values("gop_idx")
            if not frame.semantic_score.between(0, 1).all():
                raise RuntimeError(f"invalid semantic score: {video_id}")
            if frame.gop_idx.tolist() != list(range(len(frame))):
                raise RuntimeError(f"non-contiguous semantic GOPs: {video_id}")
            if len(frame) < WARMUP_GOPS + 2:
                raise RuntimeError(f"video too short for warm-up: {video_id}")
            self.semantic_by_video[str(video_id)] = frame.semantic_score.to_numpy(dtype=np.float64)
            keys = self.grid[self.grid.video_id == video_id]
            counts = keys.groupby("gop_idx").action_id.nunique()
            if (len(counts) != len(frame) or counts.index.tolist() != list(range(len(frame)))
                    or not (counts == self.action_count).all()
                    or set(keys.action_id) != set(range(self.action_count))):
                raise RuntimeError(f"incomplete action grid: {video_id}")
        self.trace_by_split: dict[str, dict[str, np.ndarray]] = {}
        for (split, trace_id), frame in self.traces.groupby(["split", "trace_id"]):
            frame = frame.sort_values("gop_idx")
            if frame.gop_idx.tolist() != list(range(len(frame))):
                raise RuntimeError(f"non-contiguous bandwidth trace: {trace_id}")
            values = frame.bandwidth_mbps.to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise RuntimeError(f"invalid bandwidth trace: {trace_id}")
            self.trace_by_split.setdefault(str(split), {})[str(trace_id)] = values
        if set(self.trace_by_split) != {"train", "validation", "test"}:
            raise RuntimeError("required train/validation/test traces missing")
        raw = project_path(config, "raw_data")
        for name, split in TRACE_SPLITS.items():
            trace_id = name.removesuffix(".up")
            expected = parse_mahimahi_trace(raw / "mahimahi" / "traces" / name)
            actual = self.trace_by_split[split].get(trace_id)
            if actual is None or not np.array_equal(actual, expected):
                raise RuntimeError(f"processed bandwidth trace differs from pinned source: {name}")
        self.reward_config = config["reward"]
        self.block_seconds = int(config["rl"]["synthetic_block_seconds"])
        if self.block_seconds != 10:
            raise RuntimeError("synthetic block size must be 10 seconds")
        self.state_clip = float(config["rl"]["state_clip"])

    def normalizer(self) -> Normalizer:
        train_bw = np.concatenate(list(self.trace_by_split["train"].values()))
        real = self.grid.actual_bitrate_mbps.to_numpy(dtype=np.float64)
        semantic = self.semantics.semantic_score.to_numpy(dtype=np.float64)
        mean = np.array([*[float(train_bw.mean())] * 4, float(real.mean()), float(semantic.mean())])
        std = np.array([*[float(train_bw.std())] * 4, float(real.std()), float(semantic.std())])
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise RuntimeError("cannot fit state normalizer from training data")
        return Normalizer(mean, std, self.state_clip)

    def row(self, video_id: str, k: int, action: int) -> pd.Series:
        try:
            return self.grid_lookup.loc[(video_id, k, action)]
        except KeyError as error:
            raise RuntimeError(f"missing action grid row: {video_id}/{k}/{action}") from error

    def synthetic(self, length: int, rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
        sources = self.trace_by_split["train"]
        blocks, lineage = [], []
        while sum(len(block) for block in blocks) < length:
            trace_id = str(rng.choice(list(sources)))
            source = sources[trace_id]
            if len(source) < self.block_seconds:
                raise RuntimeError(f"trace too short for synthetic block: {trace_id}")
            start = int(rng.integers(0, len(source) - self.block_seconds + 1))
            blocks.append(source[start:start + self.block_seconds])
            lineage.append({"trace_id": trace_id, "start_gop": start, "length": self.block_seconds})
        return np.concatenate(blocks)[:length].copy(), lineage

    def evaluation_specs(self, split: str) -> list[dict]:
        if split not in {"validation", "test"}:
            raise ValueError("evaluation split must be validation or test")
        specs = []
        for video_id, semantic in sorted(self.semantic_by_video.items()):
            length = len(semantic)
            for trace_id, trace in sorted(self.trace_by_split[split].items()):
                if len(trace) < length:
                    raise RuntimeError(f"trace shorter than video: {trace_id}/{video_id}")
                for offset in range(0, len(trace) - length + 1, length):
                    specs.append({"video_id": video_id, "trace_id": trace_id,
                                  "trace_start": offset, "synthetic": False})
        if not specs:
            raise RuntimeError(f"no {split} evaluation episodes")
        return specs


def reward_for(row: pd.Series, bandwidth: float, previous: pd.Series, config: dict) -> dict:
    epsilon = float(config["bandwidth_epsilon_mbps"])
    cap = float(config["penalty_cap"])
    ratio = float(row.actual_bitrate_mbps) / max(bandwidth, epsilon)
    violation = min(max(0.0, ratio - 1.0) ** 2, cap)
    utilization = min(ratio, cap)
    switch = 0.5 * (abs(int(row.bitrate_index) - int(previous.bitrate_index)) / 7.0
                    + abs(int(row.roi_index) - int(previous.roi_index)) / 3.0)
    roi_present = int(row.roi_frame_count) > 0
    q_roi = float(row.roi_ssim) if roi_present else 0.0
    parts = {
        "reward_full_quality": float(row.vmaf) / 100.0,
        "reward_roi_quality": q_roi,
        "reward_roi_present": roi_present,
        "penalty_bandwidth_violation": violation,
        "penalty_bandwidth_utilization": utilization,
        "penalty_action_switch": switch,
    }
    parts["reward"] = (config["full_vmaf"] * parts["reward_full_quality"]
                       + config["roi_ssim"] * q_roi
                       - config["bandwidth_violation"] * violation
                       - config["bandwidth_utilization"] * utilization
                       - config["action_switch"] * switch)
    return parts


class OfflineGOPEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: GridDataset, split: str, normalizer: Normalizer, seed: int):
        super().__init__()
        if split not in dataset.trace_by_split:
            raise ValueError(f"unknown split: {split}")
        self.dataset = dataset
        self.split = split
        self.normalizer = normalizer
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(dataset.action_count)
        self.observation_space = spaces.Box(-normalizer.clip, normalizer.clip,
                                             shape=(6,), dtype=np.float32)
        self.k = 0
        self.done = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        video_id = str(options.get("video_id", self.rng.choice(list(self.dataset.semantic_by_video))))
        if video_id not in self.dataset.semantic_by_video:
            raise RuntimeError(f"unknown video: {video_id}")
        self.video_id = video_id
        self.semantics = self.dataset.semantic_by_video[video_id]
        self.synthetic = bool(options.get("synthetic", False))
        if self.synthetic:
            if self.split != "train":
                raise RuntimeError("synthetic traces are train-only")
            self.bandwidth, self.lineage = self.dataset.synthetic(len(self.semantics), self.rng)
            self.trace_id = "synthetic"
            self.trace_start = 0
        else:
            traces = self.dataset.trace_by_split[self.split]
            self.trace_id = str(options.get("trace_id", self.rng.choice(list(traces))))
            if self.trace_id not in traces:
                raise RuntimeError(f"trace not in {self.split}: {self.trace_id}")
            source = traces[self.trace_id]
            length = len(self.semantics)
            if len(source) < length:
                raise RuntimeError(f"trace shorter than video: {self.trace_id}/{video_id}")
            self.trace_start = int(options.get("trace_start", self.rng.integers(0, len(source) - length + 1)))
            if self.trace_start < 0 or self.trace_start + length > len(source):
                raise RuntimeError("trace_start outside trace bounds")
            self.bandwidth = source[self.trace_start:self.trace_start + length]
            self.lineage = []
        self.previous = None
        for k in range(WARMUP_GOPS):
            self.previous = self.dataset.row(video_id, k, WARMUP_ACTION)
        self.k = WARMUP_GOPS
        self.done = False
        return self._observation(), {"video_id": video_id, "trace_id": self.trace_id,
                                     "trace_start": self.trace_start, "synthetic": self.synthetic,
                                     "lineage": self.lineage, "warmup_gops": WARMUP_GOPS}

    def _raw_state(self) -> np.ndarray:
        return np.array([*self.bandwidth[self.k - 4:self.k],
                         float(self.previous.actual_bitrate_mbps), float(self.semantics[self.k])],
                        dtype=np.float64)

    def _observation(self) -> np.ndarray:
        return self.normalizer.transform(self._raw_state())

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode already terminated")
        if not self.action_space.contains(action):
            raise ValueError(f"action outside [0,{self.dataset.action_count - 1}]: {action}")
        k = self.k
        raw_state = self._raw_state()
        row = self.dataset.row(self.video_id, k, int(action))
        parts = reward_for(row, float(self.bandwidth[k]), self.previous, self.dataset.reward_config)
        info = {
            "video_id": self.video_id, "trace_id": self.trace_id, "trace_start": self.trace_start,
            "split": self.split, "synthetic": self.synthetic, "lineage": self.lineage,
            "gop_idx": k, "state": raw_state.tolist(), "action_id": int(action),
            "target_bitrate_mbps": float(row.target_bitrate_mbps),
            "delta_qp_roi": int(row.delta_qp_roi),
            "actual_bitrate_mbps": float(row.actual_bitrate_mbps),
            "bandwidth_mbps": float(self.bandwidth[k]), "vmaf": float(row.vmaf),
            "psnr": float(row.psnr), "ssim": float(row.ssim),
            "roi_psnr": float(row.roi_psnr) if pd.notna(row.roi_psnr) else None,
            "roi_ssim": float(row.roi_ssim) if pd.notna(row.roi_ssim) else None,
            **parts,
        }
        self.previous = row
        self.k += 1
        self.done = self.k >= len(self.semantics)
        observation = np.zeros(6, dtype=np.float32) if self.done else self._observation()
        info["done"] = self.done
        info["next_state"] = None if self.done else self._raw_state().tolist()
        return observation, float(parts["reward"]), self.done, False, info
