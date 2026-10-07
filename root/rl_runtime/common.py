"""Contracts shared by packaging, board execution and host comparison."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

WIDTH, HEIGHT, FPS, GOP, FRAMES = 1920, 1080, 30, 30, 960
GOPS = FRAMES // GOP
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2
STRATEGIES = ("cbr", "vbr", "fixed_roi", "rl")
BITRATES = (0.5, 1, 1.5, 2, 3, 4, 6, 8)
DELTAS = (0, -3, -6, -9)
CLASS_WEIGHTS = {0: 1., 1: 1., 2: .8, 3: 1., 5: .8, 7: .8, 9: .7, 11: .7}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bundle_manifest(root, verify=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["video"]["frames"], manifest["video"]["fps"], manifest["gop_frames"]) != (FRAMES, FPS, GOP):
        raise RuntimeError("bundle requires exactly 960 frames, 30 FPS, GOP30")
    if verify:
        for relative, receipt in manifest["files"].items():
            path = (root / relative).resolve()
            if root.resolve() not in path.parents:
                raise RuntimeError("manifest path escapes bundle")
            if not path.is_file() or path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
                raise RuntimeError("missing/changed artifact: " + relative)
    if (root / manifest["video"]["path"]).stat().st_size != FRAMES * FRAME_BYTES:
        raise RuntimeError("NV12 frame coverage mismatch")
    return manifest


def read_trace(root, manifest):
    rows = json.loads((Path(root) / manifest["trace"]["path"]).read_text())["samples"]
    if len(rows) != GOPS or [r["gop_idx"] for r in rows] != list(range(GOPS)):
        raise RuntimeError("trace requires 32 consecutive GOPs from offset zero")
    values = np.array([r["bandwidth_mbps"] for r in rows], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("non-finite/negative bandwidth")
    return values


def semantic(detections):
    mass = 0.0
    for x1, y1, x2, y2, confidence, cls in np.asarray(detections).reshape(-1, 6):
        mass += CLASS_WEIGHTS[int(cls)] * confidence * math.sqrt(max(0., x2-x1) * max(0., y2-y1) / (WIDTH*HEIGHT))
    return 1. - math.exp(-mass)


def roi_boxes(detections):
    result = []
    for x1, y1, x2, y2, confidence, cls in np.asarray(detections).reshape(-1, 6):
        dx, dy = (x2-x1)*.05, (y2-y1)*.05
        left = max(0, int(math.floor((x1-dx)/16))*16)
        top = max(0, int(math.floor((y1-dy)/16))*16)
        right = min(WIDTH, int(math.ceil((x2+dx)/16))*16)
        bottom = min(HEIGHT, int(math.ceil((y2+dy)/16))*16)
        if right > left and bottom > top:
            result.append({"class_id": int(cls), "x": left, "y": top,
                           "width": right-left, "height": bottom-top})
    return result


def decision(strategy, gop, bandwidth, previous, score, policy):
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy")
    state, q = None, None
    if strategy == "rl":
        if gop < policy.warmup_gops:
            action = policy.warmup_action
        else:
            state = np.array([*bandwidth[gop-4:gop], previous, score], dtype=np.float64)
            q = policy.q_values(state)[0]
            action = int(q.argmax())
    else:
        action = 10 if strategy == "fixed_roi" else 8
    target, delta = policy.action_table[action]
    return {"action_id": action, "target_bitrate_mbps": float(target), "delta_qp_roi": int(delta),
            "raw_state": None if state is None else state.tolist(),
            "normalized_state": None if state is None else np.clip((state-policy.mean)/policy.std, -policy.clip, policy.clip).astype(np.float32).tolist(),
            "q_values": None if q is None else q.tolist(), "semantic_score": float(score),
            "control_rate": "constant" if strategy == "cbr" else "variable"}


def annexb_nal_types(payload):
    """Parse start codes, including SPS/PPS/AUD sharing one access unit."""
    result, i = [], 0
    while i + 3 < len(payload):
        size = 4 if payload[i:i+4] == b"\x00\x00\x00\x01" else (3 if payload[i:i+3] == b"\x00\x00\x01" else 0)
        if size:
            if i + size < len(payload):
                result.append(payload[i+size] & 31)
            i += size + 1
        else:
            i += 1
    return result


def validate_au(payload, frame, pts):
    expected = frame * 1_000_000_000 // FPS
    if abs(int(pts) - expected) > 1:
        raise RuntimeError("encoded PTS/frame alignment mismatch")
    types = annexb_nal_types(payload)
    if not any(t in (1, 5) for t in types):
        raise RuntimeError("parser emitted a header-only buffer, expected one AU/frame")
    if (5 in types) != (frame % GOP == 0):
        raise RuntimeError("H.264 IDR boundary mismatch")
    return 5 in types


def memory_budget(model_bytes, policy_bytes, output_limit=64*1024*1024):
    return {"input": FRAMES*FRAME_BYTES, "gop_buffers": 2*GOP*FRAME_BYTES,
            "model_policy": int(model_bytes)+int(policy_bytes), "output_limit": output_limit,
            "hardware_runtime_reserve": 256*1024*1024, "safety_reserve": 256*1024*1024}
