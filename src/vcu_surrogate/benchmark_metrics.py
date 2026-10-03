"""Deterministic region quality and detector-agreement benchmark primitives."""
from __future__ import annotations

import numpy as np


def frame_offsets(count: int, gop: int = 30) -> np.ndarray:
    if not 1 <= count <= gop:
        raise ValueError(f"frames-per-gop must be between 1 and {gop}")
    return np.unique(np.linspace(0, gop - 1, count).round().astype(int))


def box_mask(shape: tuple[int, int], boxes: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for box in boxes:
        x1, y1 = np.floor(box[:2]).astype(int)
        x2, y2 = np.ceil(box[2:4]).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def region_quality(reference: np.ndarray, distorted: np.ndarray, roi: np.ndarray) -> dict:
    """BT.601 OpenCV grayscale SSIM (11x11 Gaussian sigma 1.5), pooled by centers.

    Remove 5-pixel frame border and erode each region 5px so SSIM windows
    never cross ROI/non-ROI boundaries. PSNR pools un-eroded region pixels.
    """
    import cv2

    if reference.shape != distorted.shape or reference.ndim != 3 or roi.shape != reference.shape[:2]:
        raise ValueError("reference/distorted image and ROI shapes must agree")
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb = np.maximum(blur(a*a) - ma*ma, 0), np.maximum(blur(b*b) - mb*mb, 0)
    cov = blur(a*b) - ma*mb
    ssim = ((2*ma*mb + 6.5025) * (2*cov + 58.5225)
            / ((ma*ma + mb*mb + 6.5025) * (va + vb + 58.5225)))
    return _pool_region_quality(a, b, ssim, roi)


def _pool_region_quality(a, b, ssim, roi):
    import cv2
    error = (a - b) ** 2
    valid = np.zeros(roi.shape, dtype=bool)
    valid[5:-5, 5:-5] = True
    result = {"roi_fraction": float(roi.mean())}
    for name, mask in (("roi", roi), ("nonroi", ~roi)):
        n = int(mask.sum())
        interior = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool) & valid
        mse = float(error[mask].mean()) if n else np.nan
        result.update({f"{name}_pixels": n, f"{name}_mse_y": mse,
                       f"{name}_psnr_y": (100.0 if mse == 0 else float(10*np.log10(255**2/mse))) if n else np.nan,
                       f"{name}_ssim_y": float(ssim[interior].mean()) if interior.any() else np.nan,
                       f"{name}_ssim_centers": int(interior.sum())})
    return result


def region_quality_cuda(reference: np.ndarray, distorted: np.ndarray, roi: np.ndarray) -> dict:
    """Same regional definition, separable Gaussian convolutions on CUDA FP32.

    OpenCV grayscale conversion and region pooling remain identical. GPU/CPU
    SSIM agreement is tested with tolerance 1e-3 for FP32 reduction ordering.
    """
    import cv2
    import torch
    import torch.nn.functional as functional
    if reference.shape != distorted.shape or reference.ndim != 3 or roi.shape != reference.shape[:2]:
        raise ValueError("reference/distorted image and ROI shapes must agree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for benchmark region quality")
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY).astype(np.float32)
    with torch.inference_mode():
        moments = torch.as_tensor(np.stack((a, b, a*a, b*b, a*b))[None], device="cuda")
        kernel = torch.as_tensor(cv2.getGaussianKernel(11, 1.5).astype(np.float32), device="cuda")
        horizontal = kernel.T.reshape(1, 1, 1, 11).repeat(5, 1, 1, 1)
        vertical = kernel.reshape(1, 1, 11, 1).repeat(5, 1, 1, 1)
        blurred = functional.conv2d(functional.pad(moments, (5, 5, 0, 0), mode="reflect"), horizontal, groups=5)
        blurred = functional.conv2d(functional.pad(blurred, (0, 0, 5, 5), mode="reflect"), vertical, groups=5)
        ma, mb, aa, bb, ab = blurred[0]
        va, vb = (aa-ma*ma).clamp(min=0), (bb-mb*mb).clamp(min=0)
        cov = ab-ma*mb
        ssim = ((2*ma*mb + 6.5025)*(2*cov + 58.5225)
                / ((ma*ma + mb*mb + 6.5025)*(va + vb + 58.5225))).cpu().numpy()
    return _pool_region_quality(a, b, ssim, roi)


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


def semantic_mass(detections: np.ndarray, weights: dict[int, float], area: int) -> float:
    mass = 0.0
    for x1, y1, x2, y2, confidence, class_id in np.asarray(detections).reshape(-1, 6):
        mass += weights[int(class_id)] * confidence * np.sqrt(max(0, x2-x1)*max(0, y2-y1)/area)
    return float(mass)


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


def matched_rate_saving(base_rates: np.ndarray, base_quality: np.ndarray,
                        roi_rates: np.ndarray, roi_quality: np.ndarray) -> dict:
    """Linear rate interpolation on monotone Pareto fronts, shared measured quality only."""
    def front(rates, qualities):
        pairs = sorted((float(r), float(q)) for r, q in zip(rates, qualities) if np.isfinite(r) and np.isfinite(q))
        accepted = []
        for r, q in pairs:
            if not accepted or q > accepted[-1][1] + 1e-10:
                accepted.append((r, q))
        return np.array(accepted).reshape(-1, 2)
    a, b = front(base_rates, base_quality), front(roi_rates, roi_quality)
    if len(a) < 2 or len(b) < 2:
        return {"status": "insufficient_front", "saving_percent": np.nan}
    low, high = max(a[0, 1], b[0, 1]), min(a[-1, 1], b[-1, 1])
    if high <= low:
        return {"status": "no_quality_overlap", "saving_percent": np.nan}
    quality = np.linspace(low, high, 101)
    ra, rb = np.interp(quality, a[:, 1], a[:, 0]), np.interp(quality, b[:, 1], b[:, 0])
    return {"status": "ok", "saving_percent": float(np.mean(1-rb/ra)*100),
            "quality_low": float(low), "quality_high": float(high)}
