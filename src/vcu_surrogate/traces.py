from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import project_path, run, sha256, write_json
from .dataset import split_ids, trace_group


def packet_timestamps_to_bandwidth(timestamps_ms: np.ndarray) -> np.ndarray:
    """Legacy packet parser retained for fixtures and external trace callers."""
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if timestamps.size == 0 or timestamps[0] < 0 or np.any(np.diff(timestamps) < 0):
        raise RuntimeError("packet timestamps must be nonempty, ordered and nonnegative")
    return np.bincount(timestamps // 1000).astype(float) * 1500.0 * 8.0 / 1_000_000.0


def throughput_to_bandwidth(samples: np.ndarray) -> np.ndarray:
    """Pensieve convention: sample i serves interval [time i-1, time i)."""
    values = np.asarray(samples, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 2
            or not np.isfinite(values).all() or (values < 0).any()
            or values[0, 0] != 0 or (np.diff(values[:, 0]) <= 0).any()):
        raise RuntimeError("invalid timestamp/throughput trace")
    bins = math.floor(values[-1, 0])
    if bins < 1:
        raise RuntimeError("trace lacks a complete second")
    result = np.zeros(bins)
    for i in range(1, len(values)):
        start, end = values[i - 1, 0], min(values[i, 0], bins)
        while start < end:
            k = math.floor(start)
            finish = min(end, k + 1)
            result[k] += (finish - start) * values[i, 1]
            start = finish
    return result


def parse_throughput_trace(path: Path) -> np.ndarray:
    return throughput_to_bandwidth(read_timestamp_mbps(path))


def read_timestamp_mbps(path: Path) -> np.ndarray:
    """Read exactly timestamp seconds / throughput Mbps; reject malformed input."""
    try:
        samples = np.loadtxt(path, ndmin=2)
        throughput_to_bandwidth(samples)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"invalid timestamp/Mbps trace {path}: {error}") from error
    return samples


def download_traces(config: dict) -> None:
    traces = config["traces"]
    checkout = project_path(config, "raw_data") / "network_traces"
    if not checkout.exists():
        run(["git", "clone", "--no-checkout", traces["url"], checkout])
    run(["git", "-C", checkout, "checkout", "--detach", traces["commit"]])
    paths = sorted((checkout / "fcc_ori/cooked_traces").iterdir())
    records = []
    for path in paths:
        if not path.is_file() or not (path.name.startswith("trace_") or path.suffix == ".log"):
            continue
        source = "fcc" if path.name.startswith("trace_") else "norway"
        records.append({"trace_id": f"{source}_{path.stem}", "source_file": str(path.relative_to(checkout)),
                        "source": source, "group_id": trace_group(path.name), "raw_sha256": sha256(path)})
    for source in ("fcc", "norway"):
        groups = sorted({r["group_id"] for r in records if r["source"] == source})
        splits = split_ids(groups, int(traces["split_seed"]), .85, .15 / .85)
        for row in records:
            if row["source"] == source:
                row["split"] = splits[row["group_id"]]
    manifest = {"url": traces["url"], "commit": traces["commit"], "seed": traces["split_seed"], "files": records}
    output = project_path(config, "processed_data") / "trace_sources.json"
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise RuntimeError("trace manifest changed; use a new output namespace")
    write_json(output, manifest)


def assert_trace_sources(config: dict) -> list[dict]:
    manifest = json.loads((project_path(config, "processed_data") / "trace_sources.json").read_text())
    if any(manifest.get(key) != config["traces"][key] for key in ("url", "commit")):
        raise RuntimeError("trace source revision mismatch")
    checkout = project_path(config, "raw_data") / "network_traces"
    if run(["git", "-C", checkout, "rev-parse", "HEAD"], capture=True).stdout.strip() != manifest["commit"]:
        raise RuntimeError("trace checkout revision mismatch")
    records = manifest["files"]
    if not records or len({row["trace_id"] for row in records}) != len(records):
        raise RuntimeError("empty or duplicate trace source manifest")
    for row in records:
        if sha256(checkout / row["source_file"]) != row["raw_sha256"]:
            raise RuntimeError(f"trace checksum mismatch: {row['trace_id']}")
    return records


def prepare_traces(config: dict) -> None:
    from .assets import assert_downloads
    assert_downloads(config)
    if config["video"]["gop_frames"] != config["video"]["fps"]:
        raise RuntimeError("trace bins require one-second GOPs")
    checkout = project_path(config, "raw_data") / "network_traces"
    rows, catalog, timestamp_rows = [], [], []
    for record in assert_trace_sources(config):
        samples = read_timestamp_mbps(checkout / record["source_file"])
        values = throughput_to_bandwidth(samples)
        timestamp_rows.extend({**record, "timestamp_s": float(t), "mbps": float(rate)}
                              for t, rate in samples)
        catalog.append({**record, "gop_samples": len(values), "duration_s": len(values),
                        "mean_mbps": float(values.mean()), "min_mbps": float(values.min()),
                        "max_mbps": float(values.max())})
        rows.extend({**record, "gop_idx": k, "timestamp_s": float(k), "t_start_s": float(k), "t_end_s": float(k + 1),
                     "bandwidth_mbps": float(value)} for k, value in enumerate(values))
    output = project_path(config, "processed_data")
    pd.DataFrame(rows).to_parquet(output / "bandwidth_traces.parquet", index=False)
    pd.DataFrame(timestamp_rows).to_parquet(output / "trace_timestamp_mbps.parquet", index=False)
    pd.DataFrame(catalog).to_parquet(output / "trace_catalog.parquet", index=False)
    write_json(output / "bandwidth_schema.json", {
        "source_columns": ["timestamp_s", "mbps"], "timestamp_unit": "seconds", "throughput_unit": "Mbps",
        "source_table": "trace_timestamp_mbps.parquet", "gop_table": "bandwidth_traces.parquet",
        "gop_columns": ["timestamp_s", "bandwidth_mbps"], "bin_seconds": 1,
        "integration": "sample i serves [timestamp i-1, timestamp i); time-weighted integral",
        "outage_mbps": 0, "partial_last_second": "discard", "fallback": False,
    })
