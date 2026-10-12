from __future__ import annotations

import argparse
import subprocess

from .assets import download_assets
from .benchmark import run_benchmark
from .common import DEFAULT_CONFIG, load_config
from .ddqn import evaluate, export_policy, train
from .encoding import encode_grid
from .inference import infer_metadata
from .metrics import compute_metrics
from .render import render_policy_videos
from .traces import prepare_traces
from .transitions import build_transitions
from .validation import preflight, validate_dataset
from .video import normalize_videos


def main() -> None:
    parser = argparse.ArgumentParser(prog="vcu-surrogate")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "download", "normalize", "infer", "encode-grid", "metrics", "traces", "validate", "train", "evaluate", "export-policy", "render-policy", "test", "all"):
        subparsers.add_parser(name)
    transitions = subparsers.add_parser("transitions")
    transitions.add_argument("--schedule", default=None, help="CSV with video_id,trace_id,gop_idx,action_id")
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--frames-per-gop", type=int, default=3,
                           help="Extra quality/detection samples per GOP, 1..30; 30 measures all frames")
    benchmark.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    commands = {
        "preflight": lambda: preflight(config),
        "download": lambda: download_assets(config),
        "normalize": lambda: normalize_videos(config),
        "infer": lambda: infer_metadata(config),
        "encode-grid": lambda: encode_grid(config),
        "metrics": lambda: compute_metrics(config),
        "traces": lambda: prepare_traces(config),
        "transitions": lambda: build_transitions(config, args.schedule),
        "validate": lambda: validate_dataset(config),
        "train": lambda: train(config),
        "evaluate": lambda: evaluate(config),
        "export-policy": lambda: export_policy(config),
        "render-policy": lambda: render_policy_videos(config),
        "test": lambda: subprocess.run(["python", "-m", "pytest", "tests"], check=True),
        "benchmark": lambda: run_benchmark(config, getattr(args, "frames_per_gop", 3), getattr(args, "batch_size", 8)),
    }
    if args.command == "all":
        for name in ("preflight", "download", "normalize", "infer", "encode-grid", "metrics", "traces", "transitions", "validate", "train", "evaluate", "export-policy", "render-policy", "benchmark"):
            commands[name]()
    else:
        commands[args.command]()


if __name__ == "__main__":
    main()
