from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .assets import assert_downloads
from .common import ROOT, project_path, sha256, stable_hash, write_json


DETECTION_COLUMNS = [
    "video_id", "frame_idx", "pts_s", "class_id", "class_name", "class_weight", "confidence",
    "x1", "y1", "x2", "y2", "area_fraction", "roi_x1", "roi_y1", "roi_x2", "roi_y2",
]


def _union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    if not rectangles:
        return 0.0
    xs = sorted({x for rect in rectangles for x in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((top, bottom) for x1, top, x2, bottom in rectangles if x1 < right and x2 > left)
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for top, bottom in intervals[1:]:
                if top > end:
                    covered += end - start
                    start, end = top, bottom
                else:
                    end = max(end, bottom)
            covered += end - start
        area += (right - left) * covered
    return area


def _aligned_roi(x1: float, y1: float, x2: float, y2: float, width: int, height: int, padding: float):
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, x1 - box_w * padding)
    y1 = max(0, y1 - box_h * padding)
    x2 = min(width, x2 + box_w * padding)
    y2 = min(height, y2 + box_h * padding)
    return (
        max(0, int(math.floor(x1 / 16) * 16)),
        max(0, int(math.floor(y1 / 16) * 16)),
        min(width, int(math.ceil(x2 / 16) * 16)),
        min(height, int(math.ceil(y2 / 16) * 16)),
    )


def _load_model(config: dict):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is mandatory; CPU inference is not permitted")
    yolo_root = Path(os.environ["YOLOV5_ROOT"])
    if not yolo_root.is_dir():
        raise RuntimeError(f"pinned YOLOv5 source missing: {yolo_root}")
    if os.environ.get("YOLOv5_AUTOINSTALL", "").lower() != "false":
        raise RuntimeError("YOLOv5_AUTOINSTALL must be false; runtime dependency installation is prohibited")
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))
    from models.common import AutoShape, DetectMultiBackend

    weights = ROOT / config["yolo"]["weights"]
    device = torch.device("cuda:0")
    backend = DetectMultiBackend(str(weights), device=device, dnn=False, fp16=False, fuse=True)
    if not backend.pt:
        raise RuntimeError("YOLO weights did not load as the required PyTorch backend")
    model = AutoShape(backend, verbose=False).to(device).eval()
    model.conf = config["yolo"]["confidence"]
    model.iou = config["yolo"]["iou"]
    model.max_det = config["yolo"]["max_detections"]
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    target_classes = set(config["yolo"]["classes"])
    missing_classes = target_classes - set(names.values())
    if missing_classes:
        raise RuntimeError(f"configured classes are absent from YOLOv5 COCO labels: {sorted(missing_classes)}")
    model.classes = sorted(class_id for class_id, name in names.items() if name in target_classes)
    return model


def infer_metadata(config: dict) -> None:
    import cv2

    assert_downloads(config)
    reference_dir = project_path(config, "artifacts") / "reference"
    manifest_path = reference_dir / "manifest.parquet"
    if not manifest_path.is_file():
        raise RuntimeError("normalized reference manifest missing; run normalize first")
    manifest = pd.read_parquet(manifest_path)
    metadata_dir = project_path(config, "artifacts") / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(config)
    yolo = config["yolo"]
    class_weights = yolo["classes"]
    width, height = config["video"]["width"], config["video"]["height"]
    fps, gop_frames = config["video"]["fps"], config["video"]["gop_frames"]
    all_detections = []
    all_gops = []

    for item in manifest.itertuples(index=False):
        output = metadata_dir / f"{item.video_id}.detections.parquet"
        gop_output = metadata_dir / f"{item.video_id}.gops.parquet"
        roi_output = metadata_dir / f"{item.video_id}.roi.csv"
        sidecar = metadata_dir / f"{item.video_id}.json"
        job = {
            "reference_sha256": item.reference_sha256,
            "weights_sha256": sha256(ROOT / yolo["weights"]),
            "yolo": yolo,
        }
        job_hash = stable_hash(job)
        if output.exists() and gop_output.exists() and roi_output.exists() and sidecar.exists():
            import json
            existing = json.loads(sidecar.read_text())
            if existing.get("job_hash") != job_hash:
                raise RuntimeError(f"incompatible inference artifact: {output}")
            all_detections.append(pd.read_parquet(output))
            all_gops.append(pd.read_parquet(gop_output))
            continue

        cap = cv2.VideoCapture(str(ROOT / item.reference_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open normalized video: {item.reference_path}")
        rows = []
        for frame_idx in tqdm(range(item.frame_count), desc=f"YOLO {item.video_id}"):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"early EOF at frame {frame_idx}: {item.video_id}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = model(frame_rgb, size=yolo["image_size"])
            tensor = prediction.xyxy[0].detach().cpu().numpy()
            candidates = []
            for x1, y1, x2, y2, confidence, class_id_float in tensor:
                class_id = int(class_id_float)
                class_name = model.names[class_id]
                if class_name not in class_weights:
                    raise RuntimeError(f"YOLO returned an unconfigured class despite class filtering: {class_name}")
                candidates.append((float(confidence), class_id, class_name, float(x1), float(y1), float(x2), float(y2)))
            candidates.sort(reverse=True)
            for confidence, class_id, class_name, x1, y1, x2, y2 in candidates[: yolo["max_detections"]]:
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(width), x2), min(float(height), y2)
                area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
                roi = _aligned_roi(x1, y1, x2, y2, width, height, yolo["roi_padding_fraction"])
                rows.append({
                    "video_id": item.video_id, "frame_idx": frame_idx, "pts_s": frame_idx / fps,
                    "class_id": class_id, "class_name": class_name,
                    "class_weight": float(class_weights[class_name]), "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area_fraction": area_fraction,
                    "roi_x1": roi[0], "roi_y1": roi[1], "roi_x2": roi[2], "roi_y2": roi[3],
                })
        cap.release()
        detections = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
        detections.to_parquet(output, index=False)
        with roi_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_idx", "x1", "y1", "x2", "y2"])
            for row in detections.itertuples(index=False):
                writer.writerow([row.frame_idx, row.roi_x1, row.roi_y1, row.roi_x2, row.roi_y2])

        by_frame = {int(idx): group for idx, group in detections.groupby("frame_idx")}
        frame_features = []
        safe_classes = {name: name.replace(" ", "_") for name in class_weights}
        for frame_idx in range(item.frame_count):
            group = by_frame.get(frame_idx)
            feature = {"frame_idx": frame_idx, "semantic_score": 0.0, "roi_union_fraction": 0.0}
            for class_name, safe in safe_classes.items():
                feature[f"count_{safe}"] = 0
                feature[f"confidence_sum_{safe}"] = 0.0
                feature[f"area_fraction_{safe}"] = 0.0
            if group is not None:
                semantic_mass = float(np.sum(group.class_weight * group.confidence * np.sqrt(group.area_fraction)))
                feature["semantic_score"] = 1.0 - math.exp(-semantic_mass)
                feature["roi_union_fraction"] = _union_area(
                    list(group[["x1", "y1", "x2", "y2"]].itertuples(index=False, name=None))
                ) / (width * height)
                for class_name, class_group in group.groupby("class_name"):
                    safe = safe_classes[class_name]
                    feature[f"count_{safe}"] = len(class_group)
                    feature[f"confidence_sum_{safe}"] = float(class_group.confidence.sum())
                    feature[f"area_fraction_{safe}"] = float(class_group.area_fraction.sum())
            frame_features.append(feature)
        frames = pd.DataFrame(frame_features)
        frames["gop_idx"] = frames.frame_idx // gop_frames
        aggregate = {column: "mean" for column in frames.columns if column not in ("frame_idx", "gop_idx")}
        for column in frames.columns:
            if column.startswith("count_"):
                aggregate[column] = "sum"
        gops = frames.groupby("gop_idx", as_index=False).agg(aggregate)
        gops.insert(0, "video_id", item.video_id)
        gops["frame_start"] = gops.gop_idx * gop_frames
        gops["frame_count"] = gop_frames
        gops.to_parquet(gop_output, index=False)
        write_json(sidecar, {**job, "job_hash": job_hash, "detections": len(detections)})
        all_detections.append(detections)
        all_gops.append(gops)

    pd.concat(all_detections, ignore_index=True).to_parquet(metadata_dir / "detections.parquet", index=False)
    pd.concat(all_gops, ignore_index=True).to_parquet(metadata_dir / "gop_semantics.parquet", index=False)
