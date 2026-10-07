#!/bin/bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$bundle_dir${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m rl_runtime.board --root "$bundle_dir" --smoke-only "$@"
