#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_runtime.policy_numpy import NumpyPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "models/rl/golden_policy.json")
    args = parser.parse_args()
    policy = NumpyPolicy(ROOT / "models/rl/ddqn_policy.npz")
    rows = json.loads(args.golden.read_text())
    errors = []
    for row in rows:
        q = policy.q_values(row["raw_state"])[0]
        expected = np.asarray(row["q_values"], dtype=np.float32)
        error = float(np.max(np.abs(q-expected)))
        if not np.allclose(q, expected, rtol=1e-5, atol=1e-5) or int(q.argmax()) != row["action_id"]:
            raise RuntimeError("policy replay mismatch at GOP " + str(row["gop_idx"]))
        errors.append(error)
    if not rows:
        raise RuntimeError("empty golden replay")
    print(json.dumps({"status": "passed", "states": len(rows), "max_q_error": max(errors)}))


if __name__ == "__main__":
    main()
