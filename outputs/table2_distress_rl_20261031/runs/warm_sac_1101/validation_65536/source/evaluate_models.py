"""Evaluate explicit standalone models without granting access to sealed cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, contracts, file_hash, read_json
from evaluate import run_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--training-smoke", action="store_true")
    parser.add_argument("--macro", choices=["rule", "full_residual"], default="rule")
    args = parser.parse_args()
    protocol = contracts()
    protocol["workers"] = args.workers
    protocol["code_hashes"] = {
        name: file_hash(ROOT / name)
        for name in (
            "common.py",
            "controllers.py",
            "history_policy.py",
            "evaluate.py",
            "evaluate_models.py",
        )
    }
    policies = [
        {
            "id": path.parent.name,
            "hardware": "history_sac",
            "macro": args.macro,
            "checkpoint": {"path": str(path), "sha256": file_hash(path)},
        }
        for path in args.checkpoints
    ]
    if len({row["id"] for row in policies}) != len(policies):
        raise ValueError("Checkpoint parent directories must uniquely identify models")
    cells = protocol["development_cells"]
    if args.training_smoke:
        cell = read_json(ROOT / "demonstrations_v1/summary.json")["results"][0][
            "condition"
        ]
        if cell["start_day"] not in protocol["train_days"]:
            raise ValueError("Smoke must use training support only")
        cells = [cell]
    run_panel(protocol, policies, cells, args.output)


if __name__ == "__main__":
    main()
