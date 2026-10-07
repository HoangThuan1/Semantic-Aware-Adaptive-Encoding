from __future__ import annotations
import json, math, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
def run(command):
    return subprocess.run([str(x) for x in command], check=True)

# Source: src/vcu_surrogate/metrics.py sha256 0847226107e6b053c4e569902a5ae312a8a003b6d7dad86c7bbe747a6369e3a8
def _roi_ssim(reference: np.ndarray, distorted: np.ndarray, win_size: int) -> float:
    """skimage's uniform-window RGB SSIM, using native float64 box filters.

    Same sample covariance, K1/K2, channel pooling and interior crop; only
    the filter implementation differs, within float64 rounding (1e-10).
    """
    import cv2

    if reference.shape != distorted.shape or reference.dtype != np.uint8 or distorted.dtype != np.uint8:
        raise ValueError("ROI SSIM requires matching uint8 RGB/BGR images")
    if reference.ndim != 3 or reference.shape[2] != 3 or win_size < 3 or win_size % 2 != 1:
        raise ValueError("ROI SSIM requires 3 channels and an odd window >= 3")
    if min(reference.shape[:2]) < win_size:
        raise ValueError("ROI SSIM window exceeds crop")
    pad = (win_size - 1) // 2
    cov_norm = win_size ** 2 / (win_size ** 2 - 1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores = np.empty(3, dtype=np.float64)
    def box(image):
        return cv2.boxFilter(image, cv2.CV_64F, (win_size, win_size),
                             normalize=True, borderType=cv2.BORDER_REFLECT)
    for channel in range(3):
        x = reference[:, :, channel].astype(np.float64)
        y = distorted[:, :, channel].astype(np.float64)
        ux, uy = box(x), box(y)
        vx = cov_norm * (box(x * x) - ux * ux)
        vy = cov_norm * (box(y * y) - uy * uy)
        vxy = cov_norm * (box(x * y) - ux * uy)
        score = ((2 * ux * uy + c1) * (2 * vxy + c2)
                 / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)))
        scores[channel] = score[pad:-pad, pad:-pad].mean(dtype=np.float64)
    return float(scores.mean())

def _roi_metrics(encoded: Path, reference: Path, detections: pd.DataFrame, frame_count: int,
                 reference_bgr: Path | None = None, distorted_frames=None,
                 sample_stride: int = 1) -> pd.DataFrame:
    import cv2

    # Bound each worker's decoder buffers and avoid competing OpenCV thread pools.
    cv2.setNumThreads(1)
    decode_params = [cv2.CAP_PROP_N_THREADS, 2]
    encoded_cap = cv2.VideoCapture(str(encoded), cv2.CAP_FFMPEG, decode_params) if distorted_frames is None else None
    cached = np.load(reference_bgr, mmap_mode="r") if reference_bgr is not None else None
    if cached is not None and (cached.dtype != np.uint8 or cached.shape not in
            ((frame_count, 1080, 1920, 3), (len(range(0, frame_count, sample_stride)), 1080, 1920, 3))):
        raise RuntimeError("invalid decoded reference cache coverage/shape")
    reference_cap = cv2.VideoCapture(str(reference), cv2.CAP_FFMPEG, decode_params) if cached is None else None
    if ((encoded_cap is not None and not encoded_cap.isOpened()) or
            (reference_cap is not None and not reference_cap.isOpened())):
        raise RuntimeError(f"cannot decode ROI metric pair: {encoded}, {reference}")
    by_frame = {int(idx): group for idx, group in detections.groupby("frame_idx")}
    rows = []
    indices = range(0, frame_count, sample_stride) if distorted_frames is not None else range(frame_count)
    for frame_idx in tqdm(indices, desc=f"ROI metrics {encoded.stem}", leave=False, disable=True):
        if encoded_cap is not None:
            ok_dist, distorted = encoded_cap.read()
        else:
            distorted = next(distorted_frames, None)
            ok_dist = distorted is not None
        ok_ref, original = reference_cap.read() if reference_cap is not None else (True, cached[frame_idx if len(cached) == frame_count else frame_idx // sample_stride])
        if not ok_dist or not ok_ref:
            raise RuntimeError(f"metric decode ended at frame {frame_idx}: {encoded}")
        if frame_idx % sample_stride:
            continue
        group = by_frame.get(frame_idx)
        if group is None or group.empty:
            rows.append({"frame_idx": frame_idx, "has_roi": False, "roi_pixel_fraction": 0.0,
                         "roi_psnr": np.nan, "roi_ssim": np.nan})
            continue
        height, width = original.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        ssim_weighted = 0.0
        weight_sum = 0.0
        for box in group.itertuples(index=False):
            x1, y1 = max(0, int(math.floor(box.x1))), max(0, int(math.floor(box.y1)))
            x2, y2 = min(width, int(math.ceil(box.x2))), min(height, int(math.ceil(box.y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = True
            ref_crop = original[y1:y2, x1:x2]
            dist_crop = distorted[y1:y2, x1:x2]
            minimum = min(ref_crop.shape[0], ref_crop.shape[1])
            win_size = min(7, minimum if minimum % 2 == 1 else minimum - 1)
            if win_size >= 3:
                score = _roi_ssim(ref_crop, dist_crop, win_size)
                weight = float(box.class_weight * box.confidence * (x2 - x1) * (y2 - y1))
                ssim_weighted += score * weight
                weight_sum += weight
        if not mask.any():
            rows.append({"frame_idx": frame_idx, "has_roi": False, "roi_pixel_fraction": 0.0,
                         "roi_psnr": np.nan, "roi_ssim": np.nan})
            continue
        difference = original[mask].astype(np.float32) - distorted[mask].astype(np.float32)
        mse = float(np.mean(np.square(difference)))
        roi_psnr = 100.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
        rows.append({
            "frame_idx": frame_idx, "has_roi": True,
            "roi_pixel_fraction": float(mask.mean()), "roi_psnr": roi_psnr,
            "roi_ssim": ssim_weighted / weight_sum if weight_sum else np.nan,
        })
    if encoded_cap is not None:
        encoded_cap.release()
    if reference_cap is not None:
        reference_cap.release()
    return pd.DataFrame(rows)

def _vmaf_command(encoded: Path, reference: Path, log_path: Path,
                  reference_yuv: Path | None = None, raw_distorted: bool = False,
                  sample_stride: int = 1, backend: str = "cpu") -> list:
    if backend == "cuda":
        if reference_yuv is None or not raw_distorted:
            raise RuntimeError("CUDA VMAF requires shared raw reference and a streamed action")
        return ["/opt/vmaf-cuda/bin/vmaf", "--reference", str(reference_yuv),
                "--distorted", "/dev/stdin", "--width", "1920", "--height", "1080",
                "--pixel_format", "420", "--bitdepth", "8", "--model", "version=vmaf_v0.6.1",
                "--feature", "psnr", "--feature", "float_ssim", "--threads", "0",
                "--subsample", str(sample_stride), "--json", "--output", str(log_path), "--quiet"]
    if backend != "cpu":
        raise ValueError(f"unsupported metric backend: {backend}")
    graph = (
        "[0:v]settb=AVTB,setpts=N/(30*TB)[dist];"
        "[1:v]settb=AVTB,setpts=N/(30*TB)[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path}:"
        f"model=version=vmaf_v0.6.1:feature=name=psnr|name=float_ssim:n_threads=2:n_subsample={sample_stride}"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-filter_complex_threads", "1"]
    if raw_distorted:
        command += ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "1920x1080",
                    "-framerate", "30", "-i", "pipe:0"]
    else:
        command += ["-threads", "2", "-i", encoded]
    if reference_yuv is None:
        command += ["-threads", "2", "-i", reference]
    else:
        command += ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "1920x1080",
                    "-framerate", "30", "-i", reference_yuv]
    return command + ["-filter_complex", graph, "-f", "null", "-"]

def _read_vmaf(log_path: Path) -> pd.DataFrame:
    if not log_path.is_file():
        raise RuntimeError("libvmaf did not create its mandatory JSON log")
    data = json.loads(log_path.read_text())
    rows = []
    for frame in data["frames"]:
        metrics = frame["metrics"]
        try:
            rows.append({
                "frame_idx": int(frame["frameNum"]),
                "vmaf": float(metrics["vmaf"]),
                "psnr": float(metrics["psnr_y"]),
                "ssim": float(metrics["float_ssim"]),
            })
        except KeyError as error:
            raise RuntimeError(f"required libvmaf metric missing: {error}; available={sorted(metrics)}") from error
    if not rows:
        raise RuntimeError("libvmaf returned no per-frame metrics")
    return pd.DataFrame(rows)

def _run_vmaf(encoded: Path, reference: Path, log_path: Path,
              reference_yuv: Path | None = None) -> pd.DataFrame:
    log_path.unlink(missing_ok=True)
    run(_vmaf_command(encoded, reference, log_path, reference_yuv))
    return _read_vmaf(log_path)

# Source: src/vcu_surrogate/rl_env.py sha256 2e47d1a0d62df39383c28bb58afc63a41534bf5c6ab2450c9221067a0f3d1e87
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

# Source: src/vcu_surrogate/benchmark_metrics.py sha256 62d00e8722a2dca8c59b7b567369d8f39a55d4cf9cdc39e4163a82cdad730572
def detection_match(reference: np.ndarray, predicted: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Rows xyxy, confidence, class. Confidence-ordered one-to-one same-class matching.

    Reference detections are pseudo labels, NOT human ground truth.
    """
    ref = np.asarray(reference, dtype=float).reshape(-1, 6)
    pred = np.asarray(predicted, dtype=float).reshape(-1, 6)
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    matched: set[int] = set()
    ious, changes, ratios, match_rows = [], [], [], []
    order = np.argsort(-pred[:, 4], kind="stable")
    for idx in order:
        p = pred[idx]
        candidates = [j for j in range(len(ref)) if j not in matched and ref[j, 5] == p[5]]
        best, best_iou = None, 0.0
        for j in candidates:
            r = ref[j]
            wh = np.maximum(0, np.minimum(r[2:4], p[2:4]) - np.maximum(r[:2], p[:2]))
            intersection = float(np.prod(wh))
            union = float(np.prod(np.maximum(0, r[2:4]-r[:2])) + np.prod(np.maximum(0, p[2:4]-p[:2])) - intersection)
            overlap = intersection / union if union > 0 else 0.0
            if overlap > best_iou:
                best, best_iou = j, overlap
        if best is not None and best_iou >= iou_threshold:
            matched.add(best)
            ious.append(best_iou)
            changes.append(float(p[4] - ref[best, 4]))
            ratios.append(float(p[4] / max(ref[best, 4], 1e-12)))
            match_rows.append((best, int(idx)))
    return {"reference_count": len(ref), "predicted_count": len(pred), "matched_count": len(matched),
            "extra_count": len(pred)-len(matched), "missed_count": len(ref)-len(matched),
            "match_iou_sum": float(sum(ious)), "confidence_delta_sum": float(sum(changes)),
            "confidence_ratio_sum": float(sum(ratios)), "matches": match_rows}

def fluid_queue(arrivals_mbit: np.ndarray, bandwidth_mbps: np.ndarray) -> dict[str, np.ndarray]:
    """FIFO, unbounded buffer, GOP payload arrives at bin start, constant service per 1s bin.

    No service can be saved in idle bins. Completion times after trace end
    are right-censored (NaN), never replaced by an invented tail capacity.
    """
    arrivals = np.asarray(arrivals_mbit, dtype=float)
    capacity = np.asarray(bandwidth_mbps, dtype=float)
    if arrivals.ndim != 1 or arrivals.shape != capacity.shape or not np.isfinite(arrivals).all() or not np.isfinite(capacity).all() or (arrivals < 0).any() or (capacity < 0).any():
        raise ValueError("queue needs finite, nonnegative equal-length 1D arrays")
    backlog, served, area = [], [], []
    queue = 0.0
    for payload, rate in zip(arrivals, capacity):
        initial = queue + payload
        service = min(initial, rate)
        queue = max(0.0, initial-service)
        backlog.append(queue)
        served.append(service)
        area.append((initial+queue)/2 if queue > 0 else (initial*initial/(2*rate) if rate > 0 else 0.0))
    cumulative_service = np.cumsum(served)
    demands = np.cumsum(arrivals)
    completion = np.full(len(arrivals), np.nan)
    for k, demand in enumerate(demands):
        if arrivals[k] == 0:
            completion[k] = float(k)
            continue
        t = int(np.searchsorted(cumulative_service, demand-1e-10, side="left"))
        if t < len(arrivals) and capacity[t] > 0:
            before = cumulative_service[t-1] if t else 0.0
            completion[k] = t + (demand-before)/capacity[t]
    latency = completion - np.arange(len(arrivals))
    return {"backlog_mbit": np.asarray(backlog), "queue_area_mbit_s": np.asarray(area),
            "completion_latency_s": latency, "uncompleted": ~np.isfinite(latency),
            "deadline_miss": (~np.isfinite(latency)) | (latency > 1+1e-9)}

