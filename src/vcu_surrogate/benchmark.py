"""Four-axis benchmark of the existing surrogate grid; no training or encode mutation."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_metrics import (box_mask, detection_match, fluid_queue, frame_offsets,
                               matched_rate_saving, region_quality_cuda, semantic_mass)
from .common import ROOT, project_path, sha256, stable_hash, write_json
from .inference import _load_model
from .policy_numpy import NumpyPolicy
from .rl_env import GridDataset, WARMUP_ACTION, WARMUP_GOPS, reward_for

VERSION = 1
COUNT_COLUMNS = ["reference_count", "predicted_count", "matched_count", "extra_count", "missed_count",
                 "match_iou_sum", "confidence_delta_sum", "confidence_ratio_sum",
                 "reference_mass", "retained_mass"]
QUALITY_COLUMNS = ["roi_psnr_y", "nonroi_psnr_y", "roi_ssim_y", "nonroi_ssim_y", "semantic_score_abs_error"]


def _save_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _sample_video(path: Path, expected_frames: int, offsets: np.ndarray):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open benchmark video: {path}")
    try:
        wanted = set(offsets.tolist())
        for idx in range(expected_frames):
            if not cap.grab():
                raise RuntimeError(f"benchmark decode ended early: {path}:{idx}")
            if idx % 30 in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"benchmark retrieve failed: {path}:{idx}")
                yield idx, frame
        if cap.grab():
            raise RuntimeError(f"unexpected extra benchmark frame: {path}")
    finally:
        cap.release()


def measure_grid(config: dict, output: Path, frames_per_gop: int, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import cv2
    import torch
    from .assets import assert_downloads
    assert_downloads(config)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    offsets = frame_offsets(frames_per_gop)
    artifacts = project_path(config, "artifacts")
    refs = pd.read_parquet(artifacts / "reference/manifest.parquet")
    encodes = pd.read_parquet(artifacts / "encodes/manifest.parquet")
    detections_path = artifacts / "metadata/detections.parquet"
    detections = pd.read_parquet(detections_path)
    source_hash = stable_hash({"implementation": sha256(Path(__file__)),
                              "metrics_implementation": sha256(Path(__file__).with_name("benchmark_metrics.py")),
                              "detections_sha256": sha256(detections_path), "config": config["_config_sha256"],
                              "frames_per_gop": frames_per_gop, "version": VERSION})
    model = None
    quality_pool = ThreadPoolExecutor(max_workers=4)
    samples, class_parts = [], []
    jobs = len(encodes)
    done = 0
    for reference in refs.sort_values("video_id").itertuples(index=False):
        video_id = str(reference.video_id)
        reference_path = ROOT / reference.reference_path
        if sha256(reference_path) != reference.reference_sha256:
            raise RuntimeError(f"reference hash mismatch: {video_id}")
        # Mmap is bounded in resident memory and avoids decoding FFV1 32 times.
        ref_cache = output / "cache" / f"{video_id}.reference.npy"
        ref_cache.parent.mkdir(parents=True, exist_ok=True)
        cached_reference = None
        reference_rows = detections[detections.video_id == video_id]
        by_frame = {int(k): g[["x1", "y1", "x2", "y2", "confidence", "class_id"]].to_numpy(dtype=float)
                    for k, g in reference_rows.groupby("frame_idx")}
        for encode in encodes[encodes.video_id == video_id].sort_values("action_id").itertuples(index=False):
            path = ROOT / encode.encoded_path
            if sha256(path) != encode.encoded_sha256:
                raise RuntimeError(f"encode hash mismatch: {path}")
            destination = output / "cache" / video_id / f"action_{encode.action_id:02d}.parquet"
            classes_path = destination.with_name(destination.stem + ".classes.parquet")
            sidecar = destination.with_suffix(".json")
            job_hash = stable_hash({"source": source_hash, "reference": reference.reference_sha256,
                                    "encode": encode.encoded_sha256})
            if destination.is_file() and classes_path.is_file() and sidecar.is_file():
                meta = json.loads(sidecar.read_text())
                if (meta.get("job_hash") == job_hash and meta.get("output_sha256") == sha256(destination)
                        and meta.get("classes_sha256") == sha256(classes_path)):
                    rows = pd.read_parquet(destination)
                    class_rows = pd.read_parquet(classes_path)
                    if len(rows) != reference.gop_count*len(offsets):
                        raise RuntimeError(f"benchmark cache coverage mismatch: {destination}")
                    samples.append(rows)
                    class_parts.append(class_rows)
                    done += 1
                    print(f"benchmark cached {done}/{jobs} {video_id} action={encode.action_id}", flush=True)
                    continue
            if model is None:
                model = _load_model(config)
                names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
                class_weights = {idx: float(config["yolo"]["classes"][name]) for idx, name in names.items()
                                 if name in config["yolo"]["classes"]}
            if cached_reference is None:
                n = int(reference.gop_count)*len(offsets)
                cached_reference = np.lib.format.open_memmap(ref_cache, mode="w+", dtype=np.uint8,
                                                             shape=(n, 1080, 1920, 3))
                for j, (_, frame) in enumerate(_sample_video(reference_path, int(reference.frame_count), offsets)):
                    cached_reference[j] = frame
                cached_reference.flush()
            rows, class_rows, pending = [], [], []
            started = time.monotonic()

            def process(batch):
                indices, images = zip(*batch)
                with torch.inference_mode():
                    predictions = model([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images],
                                        size=config["yolo"]["image_size"]).xyxy
                quality_jobs = []
                for (sample_idx, frame_idx), image in zip(indices, images):
                    ref_boxes = by_frame.get(frame_idx, np.empty((0, 6)))
                    region = box_mask((1080, 1920), ref_boxes[:, :4])
                    quality_jobs.append(quality_pool.submit(region_quality_cuda, cached_reference[sample_idx], image, region))
                for (sample_idx, frame_idx), distorted, tensor, quality_job in zip(indices, images, predictions, quality_jobs):
                    original = cached_reference[sample_idx]
                    ref = by_frame.get(frame_idx, np.empty((0, 6)))
                    pred = tensor.detach().cpu().numpy().astype(float)
                    pred = pred[np.argsort(-pred[:, 4], kind="stable")]
                    pred[:, [0, 2]] = np.clip(pred[:, [0, 2]], 0, 1920)
                    pred[:, [1, 3]] = np.clip(pred[:, [1, 3]], 0, 1080)
                    match = detection_match(ref, pred)
                    matched_refs = [pair[0] for pair in match.pop("matches")]
                    rm = semantic_mass(ref, class_weights, 1920*1080)
                    pm = semantic_mass(pred, class_weights, 1920*1080)
                    retained = semantic_mass(ref[matched_refs], class_weights, 1920*1080)
                    keys = {"video_id": video_id, "action_id": int(encode.action_id),
                            "gop_idx": frame_idx//30, "frame_idx": frame_idx}
                    rows.append({**keys, **quality_job.result(), **match,
                                 "reference_mass": rm, "retained_mass": retained,
                                 "semantic_score_abs_error": abs(math.exp(-rm)-math.exp(-pm))})
                    for class_id in sorted(set(ref[:, 5].astype(int)) | set(pred[:, 5].astype(int))):
                        cm = detection_match(ref[ref[:, 5] == class_id], pred[pred[:, 5] == class_id])
                        cm.pop("matches")
                        class_rows.append({**keys, "class_id": class_id, "class_name": names[class_id], **cm})
            for sample_idx, (idx, image) in enumerate(_sample_video(path, int(reference.frame_count), offsets)):
                pending.append(((sample_idx, idx), image))
                if len(pending) == batch_size:
                    process(pending)
                    pending = []
            if pending:
                process(pending)
            frame = pd.DataFrame(rows)
            classes = pd.DataFrame(class_rows, columns=["video_id", "action_id", "gop_idx", "frame_idx",
                                                       "class_id", "class_name", *COUNT_COLUMNS[:8]])
            _save_table(destination, frame)
            _save_table(classes_path, classes)
            write_json(sidecar, {"job_hash": job_hash, "output_sha256": sha256(destination),
                                 "classes_sha256": sha256(classes_path), "sample_frames": len(frame),
                                 "elapsed_s": time.monotonic()-started})
            samples.append(frame)
            class_parts.append(classes)
            done += 1
            print(f"benchmark measured {done}/{jobs} {video_id} action={encode.action_id} "
                  f"frames={len(frame)} seconds={time.monotonic()-started:.1f}", flush=True)
        if cached_reference is not None:
            del cached_reference
        ref_cache.unlink(missing_ok=True)
    quality_pool.shutdown(wait=True)
    return pd.concat(samples, ignore_index=True), pd.concat(class_parts, ignore_index=True)


def semantic_rates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name, numerator, denominator in (("detection_recall", "matched_count", "reference_count"),
                                          ("detection_precision", "matched_count", "predicted_count"),
                                          ("semantic_retention", "retained_mass", "reference_mass"),
                                          ("matched_mean_iou", "match_iou_sum", "matched_count"),
                                          ("matched_confidence_delta", "confidence_delta_sum", "matched_count"),
                                          ("matched_confidence_ratio", "confidence_ratio_sum", "matched_count")):
        if numerator in frame and denominator in frame:
            frame[name] = frame[numerator] / frame[denominator].replace(0, np.nan)
    frame["detection_f1"] = (2*frame.matched_count / (frame.reference_count+frame.predicted_count).replace(0, np.nan))
    return frame


def aggregate_samples(samples: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    aggregations = {c: "sum" for c in COUNT_COLUMNS if c in samples}
    aggregations.update({c: "mean" for c in QUALITY_COLUMNS if c in samples})
    aggregations["frame_idx"] = "count"
    result = samples.groupby(keys, as_index=False).agg(aggregations).rename(columns={"frame_idx": "sample_frames"})
    return semantic_rates(result)


def rate_distortion(grid: pd.DataFrame, output: Path) -> pd.DataFrame:
    metrics = ["vmaf", "roi_ssim", "ssim", *QUALITY_COLUMNS[:-1], "detection_recall", "detection_f1", "semantic_retention"]
    columns = ["actual_bitrate_mbps", *metrics, "target_error_fraction"]
    curves = grid.groupby(["video_id", "action_id", "target_bitrate_mbps", "delta_qp_roi"], as_index=False)[columns].mean()
    _save_table(output / "rate_distortion.parquet", curves)
    curves.to_csv(output / "rate_distortion.csv", index=False)
    comparisons = []
    for video_id, video in curves.groupby("video_id"):
        base = video[video.delta_qp_roi == 0].sort_values("target_bitrate_mbps")
        for delta in (-3, -6, -9):
            enhanced = video[video.delta_qp_roi == delta].sort_values("target_bitrate_mbps")
            for metric in metrics:
                saving = matched_rate_saving(base.actual_bitrate_mbps.to_numpy(), base[metric].to_numpy(),
                                             enhanced.actual_bitrate_mbps.to_numpy(), enhanced[metric].to_numpy())
                comparisons.append({"video_id": video_id, "delta_qp_roi": delta, "metric": metric, **saving})
    pd.DataFrame(comparisons).to_csv(output / "matched_quality_savings.csv", index=False)
    baseline = grid[grid.delta_qp_roi == 0].set_index(["video_id", "gop_idx", "bitrate_index"])
    paired = []
    for row in grid[grid.delta_qp_roi != 0].itertuples(index=False):
        original = baseline.loc[(row.video_id, row.gop_idx, row.bitrate_index)]
        paired.append({"video_id": row.video_id, "gop_idx": row.gop_idx,
                       "target_bitrate_mbps": row.target_bitrate_mbps, "delta_qp_roi": row.delta_qp_roi,
                       "actual_rate_delta_mbps": row.actual_bitrate_mbps-original.actual_bitrate_mbps,
                       **{f"delta_{m}": getattr(row, m)-original[m] for m in metrics}})
    _save_table(output / "roi_vs_nonroi_paired.parquet", pd.DataFrame(paired))
    return curves


def causal_action(history: np.ndarray, semantics: float, bitrates: np.ndarray, safety: float,
                  estimator: str, roi_mode: int | str) -> int:
    history = np.asarray(history, dtype=float)
    if estimator == "last":
        estimate = history[-1]
    elif estimator == "harmonic":
        estimate = 0.0 if np.any(history <= 0) else len(history)/np.sum(1/history)
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    feasible = np.flatnonzero(bitrates <= safety*estimate)
    bitrate_index = int(feasible[-1]) if len(feasible) else 0
    roi_index = int(semantics >= 0.25) + int(semantics >= 0.5) + int(semantics >= 0.75) if roi_mode == "semantic" else int(roi_mode)
    return bitrate_index*4 + roi_index


def oracle_actions(outcomes: list[pd.DataFrame], bandwidth: np.ndarray, previous: pd.Series,
                   config: dict) -> np.ndarray:
    """Exact finite-horizon undiscounted reward oracle via dynamic programming.

    Knows all future bandwidth/grid outcomes; NOT a deployable baseline.
    The switch cost is the only dependence on the previous action.
    """
    n = len(outcomes)
    actions = np.arange(32)
    switch = 0.5*(abs(actions[:, None]//4-actions[None, :]//4)/7
                  + abs(actions[:, None]%4-actions[None, :]%4)/3)*config["action_switch"]
    value = np.zeros(32)
    choices = np.empty((n, 32), dtype=int)
    for k in range(n-1, -1, -1):
        base = np.array([reward_for(outcomes[k].iloc[a], float(bandwidth[k]), outcomes[k].iloc[a], config)["reward"]
                         for a in range(32)])
        candidates = base[None, :] - switch + value[None, :]
        choices[k] = candidates.argmax(axis=1)
        value = candidates.max(axis=1)
    prev_action = int(previous.action_id)
    result = []
    for k in range(n):
        prev_action = int(choices[k, prev_action])
        result.append(prev_action)
    return np.array(result)


def evaluate_strategy(dataset: GridDataset, enriched: pd.DataFrame, specs: list[dict], name: str,
                      policy: NumpyPolicy | None = None, heuristic: dict | None = None,
                      fixed: int | None = None, oracle: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = enriched.set_index(["video_id", "gop_idx", "action_id"])
    by_gop = {(str(v), int(k)): g.sort_values("action_id").reset_index(drop=True)
              for (v, k), g in enriched.groupby(["video_id", "gop_idx"])}
    bitrates = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8])
    episode_rows, rollout = [], []
    for spec in specs:
        video_id, trace_id, offset = spec["video_id"], spec["trace_id"], spec["trace_start"]
        semantic = dataset.semantic_by_video[video_id]
        if "bandwidth" in spec:
            bw = np.array(spec["bandwidth"], dtype=float)
        else:
            bw = dataset.trace_by_split[spec["split"]][trace_id][offset:offset+len(semantic)]
        previous = lookup.loc[(video_id, 3, WARMUP_ACTION)]
        payloads = [float(lookup.loc[(video_id, k, WARMUP_ACTION)].actual_bitrate_mbps) for k in range(4)]
        local = []
        exact = (oracle_actions([by_gop[(video_id, k)] for k in range(4, len(semantic))], bw[4:], previous,
                                dataset.reward_config) if oracle else None)
        for k in range(4, len(semantic)):
            state = np.array([*bw[k-4:k], previous.actual_bitrate_mbps, semantic[k]])
            if policy is not None:
                action = policy.act(state)[0]
            elif exact is not None:
                action = int(exact[k-4])
            elif fixed is not None:
                action = fixed
            else:
                action = causal_action(bw[k-4:k], float(semantic[k]), bitrates, **heuristic)
            row = lookup.loc[(video_id, k, action)]
            reward = reward_for(row, float(bw[k]), previous, dataset.reward_config)
            payloads.append(float(row.actual_bitrate_mbps))
            local.append({"strategy": name, "split": spec["split"], "video_id": video_id, "trace_id": trace_id,
                          "trace_start": offset, "gop_idx": k, "action_id": action,
                          "bandwidth_mbps": float(bw[k]), **{c: row[c] for c in
                          ["actual_bitrate_mbps", "target_bitrate_mbps", "delta_qp_roi", "vmaf", "roi_ssim", *QUALITY_COLUMNS,
                           *COUNT_COLUMNS]}, **reward})
            previous = row
        q = fluid_queue(np.array(payloads), bw)
        frame = pd.DataFrame(local)
        for field, values in q.items():
            frame[field] = values[4:]
        frame["bandwidth_violation"] = frame.actual_bitrate_mbps > frame.bandwidth_mbps
        frame["excess_mbit"] = np.maximum(0, frame.actual_bitrate_mbps-frame.bandwidth_mbps)
        frame["unused_capacity_mbit"] = np.maximum(0, frame.bandwidth_mbps-frame.actual_bitrate_mbps)
        frame["post_drop"] = bw[4:] < 0.5*bw[3:-1]
        fields = ["vmaf", "roi_ssim", *QUALITY_COLUMNS, "actual_bitrate_mbps", "bandwidth_violation",
                  "excess_mbit", "backlog_mbit", "queue_area_mbit_s", "deadline_miss", "uncompleted"]
        summary = {"strategy": name, "split": spec["split"], "video_id": video_id, "trace_id": trace_id,
                   "trace_start": offset, "gops": len(frame), "return": float(frame.reward.sum()),
                   "reward_per_gop": float(frame.reward.mean()),
                   **{c: float(frame[c].mean()) for c in fields},
                   "max_backlog_mbit": float(frame.backlog_mbit.max()),
                   "p95_completed_latency_s": float(frame.completion_latency_s.quantile(.95)),
                   "action_switch_rate": float((frame.penalty_action_switch > 0).mean()),
                   "outage_gops": int((frame.bandwidth_mbps < .05).sum()),
                   "post_drop_gops": int(frame.post_drop.sum()),
                   "post_drop_violation_rate": float(frame.loc[frame.post_drop, "bandwidth_violation"].mean())}
        for field in COUNT_COLUMNS:
            summary[field] = float(frame[field].sum())
        episode_rows.append(summary)
        rollout.append(frame)
    return semantic_rates(pd.DataFrame(episode_rows)), pd.concat(rollout, ignore_index=True)


def bootstrap_mean(values: np.ndarray, seed: int = 20261004) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [.025, .975]))


def summarize_policies(episodes: pd.DataFrame, rollouts: pd.DataFrame, output: Path) -> pd.DataFrame:
    records = []
    for (split, strategy), frame in rollouts.groupby(["split", "strategy"]):
        ep = episodes[(episodes.split == split) & (episodes.strategy == strategy)]
        record = {"split": split, "strategy": strategy, "episodes": len(ep), "gops": len(frame),
                  "mean_return": float(ep["return"].mean()), "reward_per_gop": float(frame.reward.mean()),
                  "vmaf": float(frame.vmaf.mean()), "roi_ssim": float(frame.roi_ssim.mean()),
                  **{c: float(frame[c].mean()) for c in QUALITY_COLUMNS},
                  "actual_bitrate_mbps": float(frame.actual_bitrate_mbps.mean()),
                  "bandwidth_violation_rate": float(frame.bandwidth_violation.mean()),
                  "action_switch_rate": float((frame.penalty_action_switch > 0).mean()),
                  "deadline_miss_rate": float(frame.deadline_miss.mean()),
                  "uncompleted_rate": float(frame.uncompleted.mean()),
                  "p95_completed_latency_s": float(frame.completion_latency_s.quantile(.95)),
                  "mean_backlog_mbit": float(frame.backlog_mbit.mean()),
                  "excess_mbit_total": float(frame.excess_mbit.sum()),
                  "post_drop_violation_rate": float(frame.loc[frame.post_drop, "bandwidth_violation"].mean())}
        record.update({c: float(frame[c].sum()) for c in COUNT_COLUMNS})
        # Cluster by video; same content and correlated trace windows make GOP iid CIs misleading.
        video_means = ep.groupby("video_id")["reward_per_gop"].mean().to_numpy()
        record["reward_video_bootstrap_low"], record["reward_video_bootstrap_high"] = bootstrap_mean(video_means)
        records.append(record)
    summary = semantic_rates(pd.DataFrame(records))
    summary.to_csv(output / "policy_summary.csv", index=False)
    _save_table(output / "policy_episodes.parquet", episodes)
    _save_table(output / "policy_rollouts.parquet", rollouts)
    comparisons = []
    for split in ("test", "stress"):
        selected = episodes[(episodes.split == split) & (episodes.strategy == "ddqn_selected")]
        for strategy in episodes[episodes.split == split].strategy.unique():
            if strategy == "ddqn_selected":
                continue
            other = episodes[(episodes.split == split) & (episodes.strategy == strategy)]
            joined = selected.merge(other, on=["split", "video_id", "trace_id", "trace_start"], suffixes=("_ddqn", "_baseline"), validate="one_to_one")
            for metric in ("reward_per_gop", "vmaf", "actual_bitrate_mbps", "bandwidth_violation", "deadline_miss"):
                joined["difference"] = joined[f"{metric}_ddqn"]-joined[f"{metric}_baseline"]
                low, high = bootstrap_mean(joined.groupby("video_id").difference.mean().to_numpy())
                comparisons.append({"split": split, "baseline": strategy, "metric": metric,
                                    "mean_paired_episode_difference": float(joined.difference.mean()),
                                    "mean_paired_video_difference": float(joined.groupby("video_id").difference.mean().mean()),
                                    "video_cluster_ci_low": low, "video_cluster_ci_high": high,
                                    "paired_episodes": len(joined), "video_clusters": joined.video_id.nunique()})
    pd.DataFrame(comparisons).to_csv(output / "paired_policy_comparisons.csv", index=False)
    return summary


def stress_specs(dataset: GridDataset) -> list[dict]:
    specs = []
    for video_id, semantic in sorted(dataset.semantic_by_video.items()):
        n = len(semantic)
        t = np.arange(n)
        # Diagnostics only: no tuning or selected-checkpoint decision on these traces.
        patterns = {"step_down_up": np.where(t < n//3, 8.0, np.where(t < 2*n//3, .6, 4.0)),
                    "outage_recovery": np.where((t >= n//3) & (t < n//3+max(3, n//6)), 0.0, 3.0),
                    "rapid_oscillation": np.where((t//3)%2 == 0, .5, 6.0)}
        for name, bw in patterns.items():
            specs.append({"split": "stress", "video_id": video_id, "trace_id": name,
                          "trace_start": 0, "bandwidth": bw.tolist()})
    return specs


def benchmark_policies(config: dict, dataset: GridDataset, grid: pd.DataFrame, output: Path) -> pd.DataFrame:
    specs = [{**s, "split": split} for split in ("validation", "test") for s in dataset.evaluation_specs(split)]
    specs.extend(stress_specs(dataset))
    policies = project_path(config, "artifacts") / "rl"
    strategies = [(f"fixed_action_{a:02d}", {"fixed": a}) for a in range(32)]
    strategies += [(f"ddqn_seed_{seed}", {"policy": NumpyPolicy(policies / f"ddqn_seed_{seed}.npz")})
                   for seed in config["rl"]["seeds"]]
    strategies.append(("ddqn_selected", {"policy": NumpyPolicy(policies / "ddqn_selected.npz")}))
    # Hyperparameters fixed beforehand and validation selects one; test is untouched.
    for estimator in ("last", "harmonic"):
        for safety in (.6, .8, 1.0):
            for roi in (0, 1, 2, 3, "semantic"):
                strategies.append((f"heuristic_{estimator}_{safety:g}_roi{roi}",
                                   {"heuristic": {"estimator": estimator, "safety": safety, "roi_mode": roi}}))
    episodes, rollouts = [], []
    validation_specs = [s for s in specs if s["split"] == "validation"]
    candidates = []
    selected_strategies = []
    for name, kwargs in strategies:
        # Validation for selection, then benchmark ALL fixed actions and DDQN seeds.
        e, r = evaluate_strategy(dataset, grid, validation_specs, name, **kwargs)
        episodes.append(e)
        rollouts.append(r)
        candidates.append({"strategy": name, "validation_mean_return": float(e["return"].mean()),
                           "family": "fixed" if "fixed" in kwargs else "heuristic" if "heuristic" in kwargs else "ddqn"})
        if not name.startswith("heuristic"):
            selected_strategies.append((name, kwargs))
    candidate_frame = pd.DataFrame(candidates)
    selections = {}
    for family in ("fixed", "heuristic"):
        candidate = candidate_frame[candidate_frame.family == family].sort_values(["validation_mean_return", "strategy"], ascending=[False, True]).iloc[0]
        winner = next(kwargs for name, kwargs in strategies if name == candidate.strategy)
        selections[family] = candidate.to_dict()
        selected_strategies.append((f"{family}_validation_selected", winner))
        e, r = evaluate_strategy(dataset, grid, validation_specs, f"{family}_validation_selected", **winner)
        episodes.append(e)
        rollouts.append(r)
    candidate_frame.to_csv(output / "validation_candidates.csv", index=False)
    write_json(output / "baseline_selection.json", selections)
    # Include two prespecified causal baselines with/without semantic ROI for ablation.
    selected_strategies.extend([(f"causal_harmonic_roi{roi}", {"heuristic": {"estimator": "harmonic", "safety": .8, "roi_mode": roi}})
                                for roi in (0, "semantic")])
    other_specs = [s for s in specs if s["split"] != "validation"]
    for idx, (name, kwargs) in enumerate(selected_strategies):
        print(f"benchmark policies {idx+1}/{len(selected_strategies)+1} {name}", flush=True)
        e, r = evaluate_strategy(dataset, grid, other_specs, name, **kwargs)
        episodes.append(e)
        rollouts.append(r)
    e, r = evaluate_strategy(dataset, grid, specs, "oracle_reward_dp", oracle=True)
    episodes.append(e)
    rollouts.append(r)
    return summarize_policies(pd.concat(episodes, ignore_index=True), pd.concat(rollouts, ignore_index=True), output)


def plots_and_report(output: Path, curves: pd.DataFrame, summary: pd.DataFrame, coverage: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("vmaf", "roi_ssim", "nonroi_ssim_y", "detection_recall", "semantic_retention"):
        fig, axes = plt.subplots(1, curves.video_id.nunique(), figsize=(20, 4), squeeze=False)
        for ax, (video, frame) in zip(axes[0], curves.groupby("video_id")):
            for delta, group in frame.groupby("delta_qp_roi"):
                group = group.sort_values("actual_bitrate_mbps")
                ax.plot(group.actual_bitrate_mbps, group[metric], marker=".", label=f"delta {delta:g}")
            ax.set_title(video, fontsize=9)
            ax.set_xlabel("Actual Mbps")
            ax.set_ylabel(metric)
            ax.grid(alpha=.3)
        axes[0, -1].legend()
        fig.tight_layout()
        fig.savefig(output / f"rd_{metric}.png", dpi=150)
        plt.close(fig)
    names = ["ddqn_selected", "fixed_validation_selected", "heuristic_validation_selected",
             "causal_harmonic_roi0", "causal_harmonic_roisemantic", "oracle_reward_dp"]
    test = summary[(summary.split == "test") & summary.strategy.isin(names)].set_index("strategy").reindex(names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ("vmaf", "bandwidth_violation_rate", "deadline_miss_rate")):
        ax.bar(np.arange(len(test)), test[metric])
        ax.set_xticks(np.arange(len(test)), test.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(output / "policy_comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# Kết quả benchmark video/ROI/bandwidth", "", "## Độ phủ", "",
             f"- {coverage['videos']} video; {coverage['gops']} GOP; {coverage['grid_rows']} GOP/action (32 action).",
             f"- Đo bổ sung {coverage['sample_frames']} frame encoded, offsets {coverage['frame_offsets']} trong mỗi GOP.",
             "- VMAF, full PSNR/SSIM và ROI metric gốc dùng mọi frame; non-ROI và detection dùng frame sample nêu trên.",
             "", "## So sánh trên trace test Verizon", "",
             "| Strategy | Reward/GOP | VMAF | ROI SSIM | Non-ROI SSIM Y | Detection recall | Mbps | Vượt BW | Trễ deadline mô phỏng |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in test.iterrows():
        lines.append(f"| {name} | {row.reward_per_gop:.4f} | {row.vmaf:.2f} | {row.roi_ssim:.4f} | {row.nonroi_ssim_y:.4f} | "
                     f"{row.detection_recall:.4f} | {row.actual_bitrate_mbps:.3f} | {row.bandwidth_violation_rate:.2%} | {row.deadline_miss_rate:.2%} |")
    lines += ["", "## Cách diễn giải", "",
              "- ROI/non-ROI bổ sung dùng union box reference, grayscale OpenCV, SSIM Gaussian 11×11 sigma 1.5. SSIM chỉ lấy tâm cửa sổ nằm hoàn toàn trong vùng; PSNR dùng toàn pixel vùng. Không trộn với ROI SSIM BGR crop của pipeline gốc.",
              "- Detection precision/recall/F1 là mức đồng thuận với detection YOLO trên reference, same-class IoU≥0.5, matching một-một theo confidence. Không phải mAP hoặc accuracy với nhãn con người; extra detection không chắc là false positive thực.",
              "- Semantic retention là tỷ lệ reference semantic mass của đối tượng được match; confidence ratio chỉ tính trên đối tượng match nên có survivor bias. Xem recall cùng confidence.",
              "- Rate-distortion dùng bitrate thực. Matched-quality saving nội suy tuyến tính trên Pareto front trong miền chất lượng giao nhau, không ngoại suy và không gọi là BD-rate. Giá trị dương là tiết kiệm, âm là tốn thêm bit.",
              "- Fixed/heuristic tốt nhất được chọn chỉ bằng validation return, không bằng test. Oracle DP biết trước toàn trace và outcome, tối đa hóa tổng reward không discount; là trần reward trong grid, không phải thuật toán triển khai.",
              "- Mọi strategy dùng warm-up action 4 trong 4 GOP. Quality/return bỏ warm-up, nhưng backlog của warm-up được giữ khi mô phỏng hàng đợi.",
              "- Queue là FIFO không giới hạn, payload cả GOP đến đầu giây, link phục vụ đều trong giây; không RTT, loss, overhead hay pacing encoder. Deadline 1 giây. Uncompleted là chưa truyền xong khi trace episode kết thúc; latency của chúng bị censor, p95 chỉ trên GOP hoàn tất.",
              "- Bootstrap ghép cặp theo episode rồi gom theo video, 2.000 bootstrap với seed cố định. Chỉ có 5 video cluster, trace windows giữa video có thể tương quan; CI mô tả biến thiên nội dung trong dataset này, không chứng minh tổng quát hóa LTE.",
              "- Grid encode action cố định còn domain gap với encode liên tục khi chuyển action. Semantic S[k] cần toàn GOP như pipeline gốc. Video train/test dùng chung; chưa có kiểm chứng VCU hoặc mạng thật.",
              "", "## Artifact", "",
              "- `grid_enriched.parquet`, `frame_samples.parquet`, `semantic_by_class.csv`: metric bổ sung và raw counts.",
              "- `rate_distortion.csv`, `matched_quality_savings.csv`, `roi_vs_nonroi_paired.parquet`: hiệu quả bitrate/ROI.",
              "- `policy_summary.csv`, `policy_episodes.parquet`, `policy_rollouts.parquet`: validation/test/stress và trace quyết định.",
              "- `baseline_selection.json`, `validation_candidates.csv`: bằng chứng chọn baseline trên validation.",
              "- `paired_policy_comparisons.csv`: difference DDQN−baseline và CI theo video.",
              "- `manifest.json`: SHA-256 input/code, tham số, độ phủ, thời gian và môi trường.",
              "- `rd_*.png`, `policy_comparison.png`: biểu đồ độc lập để xem/chia sẻ."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_benchmark(config: dict, frames_per_gop: int = 3, batch_size: int = 8) -> None:
    from .validation import preflight
    preflight(config)
    start = time.monotonic()
    output = project_path(config, "reports") / "benchmark"
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)
    dataset = GridDataset(config)
    samples, classes = measure_grid(config, output, frames_per_gop, batch_size)
    _save_table(output / "frame_samples.parquet", samples)
    gops = aggregate_samples(samples, ["video_id", "gop_idx", "action_id"])
    expected = len(dataset.grid)*frames_per_gop
    if len(samples) != expected or len(gops) != len(dataset.grid) or not (gops.sample_frames == frames_per_gop).all():
        raise RuntimeError("benchmark coverage incomplete")
    grid = dataset.grid.merge(gops, on=["video_id", "gop_idx", "action_id"], validate="one_to_one")
    grid["target_error_fraction"] = abs(grid.actual_bitrate_mbps-grid.target_bitrate_mbps)/grid.target_bitrate_mbps
    _save_table(output / "grid_enriched.parquet", grid)
    class_summary = aggregate_samples(classes, ["video_id", "action_id", "class_id", "class_name"])
    class_summary.to_csv(output / "semantic_by_class.csv", index=False)
    _save_table(output / "semantic_class_samples.parquet", classes)
    curves = rate_distortion(grid, output)
    summary = benchmark_policies(config, dataset, grid, output)
    coverage = {"videos": grid.video_id.nunique(), "gops": grid[["video_id", "gop_idx"]].drop_duplicates().shape[0],
                "grid_rows": len(grid), "sample_frames": len(samples), "frame_offsets": frame_offsets(frames_per_gop).tolist()}
    plots_and_report(output, curves, summary, coverage)
    import torch
    inputs = [Path(config["_config_path"]), project_path(config, "artifacts") / "metrics/gop_action_grid.parquet",
              project_path(config, "artifacts") / "metadata/detections.parquet",
              project_path(config, "artifacts") / "metadata/gop_semantics.parquet",
              project_path(config, "artifacts") / "encodes/manifest.parquet",
              project_path(config, "artifacts") / "reference/manifest.parquet",
              project_path(config, "processed_data") / "bandwidth_traces.parquet",
              project_path(config, "processed_data") / "trace_catalog.parquet",
              project_path(config, "reports") / "rl/training_summary.json",
              project_path(config, "artifacts") / "rl/ddqn_selected.npz"]
    inputs += [project_path(config, "artifacts") / f"rl/ddqn_seed_{seed}.npz" for seed in config["rl"]["seeds"]]
    manifest = {"status": "complete", "benchmark_version": VERSION, "coverage": coverage,
                "frames_per_gop": frames_per_gop, "batch_size": batch_size, "elapsed_s": time.monotonic()-start,
                "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
                "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), Path(__file__).with_name("benchmark_metrics.py"),
                                                                                 Path(__file__).with_name("rl_env.py"), Path(__file__).with_name("policy_numpy.py"))},
                "python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "bootstrap_seed": 20261004, "bootstrap_replicates": 2000,
                "limits": ["sampled extra frame metrics", "reference detector pseudo labels", "shared videos across splits",
                           "fixed-action encode surrogate", "fluid queue simulation; no real network/VCU measurement"]}
    write_json(output / "manifest.json", manifest)
    print(f"Benchmark complete: {output / 'REPORT.md'} ({time.monotonic()-start:.1f}s)", flush=True)
