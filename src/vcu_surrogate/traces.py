from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .assets import LTE_UPLINK_FILES, assert_downloads
from .common import project_path, sha256

TRACE_SPLITS = {
    "ATT-LTE-driving.up": "train",
    "TMobile-LTE-driving.up": "train",
    "ATT-LTE-driving-2016.up": "validation",
    "Verizon-LTE-driving.up": "test",
}


def packet_timestamps_to_bandwidth(timestamps_ms: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if timestamps.size == 0 or timestamps[0] < 0 or np.any(np.diff(timestamps) < 0):
        raise RuntimeError("Mahimahi trace must have nonempty, ordered, nonnegative timestamps")
    seconds = timestamps // 1000
    counts = np.bincount(seconds, minlength=int(seconds[-1]) + 1)
    return counts.astype(np.float64) * 1500.0 * 8.0 / 1_000_000.0


def parse_mahimahi_trace(path: Path) -> np.ndarray:
    values = []
    with path.open(encoding="ascii") as handle:
        for line_no, line in enumerate(handle, 1):
            token = line.strip()
            if not token or not token.isdecimal():
                raise RuntimeError(f"invalid Mahimahi timestamp at {path}:{line_no}")
            values.append(int(token))
    return packet_timestamps_to_bandwidth(np.asarray(values, dtype=np.int64))


def prepare_traces(config: dict) -> None:
    assert_downloads(config)
    if config["video"]["gop_frames"] != config["video"]["fps"]:
        raise RuntimeError("Mahimahi one-second bins require a one-second GOP")
    raw = project_path(config, "raw_data")
    output = project_path(config, "processed_data")
    output.mkdir(parents=True, exist_ok=True)
    rows, catalog = [], []
    for name in LTE_UPLINK_FILES:
        path = raw / "mahimahi" / "traces" / name
        values = parse_mahimahi_trace(path)
        split = TRACE_SPLITS.get(name, "excluded")
        trace_id = name.removesuffix(".up")
        catalog.append({
            "trace_id": trace_id, "source_file": name, "source": "mahimahi_sprout_lte_uplink",
            "split": split, "raw_sha256": sha256(path), "gop_samples": len(values),
            "duration_s": len(values), "mean_mbps": float(values.mean()),
            "min_mbps": float(values.min()), "max_mbps": float(values.max()),
        })
        if split == "excluded":
            continue
        for k, value in enumerate(values):
            rows.append({"trace_id": trace_id, "source": "mahimahi_sprout_lte_uplink",
                         "source_file": name, "split": split, "gop_idx": k,
                         "t_start_s": float(k), "t_end_s": float(k + 1),
                         "bandwidth_mbps": float(value)})
    traces = pd.DataFrame(rows)
    if traces.empty or set(traces.split) != {"train", "validation", "test"}:
        raise RuntimeError("Mahimahi trace split incomplete")
    traces.to_parquet(output / "bandwidth_traces.parquet", index=False)
    pd.DataFrame(catalog).to_parquet(output / "trace_catalog.parquet", index=False)
