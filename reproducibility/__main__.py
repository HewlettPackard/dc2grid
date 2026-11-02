"""Verify frozen sources, rebuild numeric tables, and replay consumed cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import artifacts, inputs
from .runtime import inside, limit_threads, read_json, verify_files


def emit(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Check all public release bytes")
    verify.add_argument("--code-only", action="store_true")
    data = commands.add_parser("inputs", help="Check/import exact caller-supplied CSVs")
    data_commands = data.add_subparsers(dest="input_command", required=True)
    data_commands.add_parser("status")
    importer = data_commands.add_parser("import")
    importer.add_argument("--source-root", type=Path, required=True)
    tables = commands.add_parser("tables", help="Recompute all 12 numeric tables")
    tables.add_argument("--output", type=Path, required=True)
    tracking = commands.add_parser(
        "tracking", help="Replay one archived hardware-policy case"
    )
    tracking.add_argument("--snapshot", required=True)
    tracking.add_argument("--policy", required=True)
    tracking.add_argument("--day", type=int, required=True)
    tracking.add_argument("--seed", type=int, required=True)
    tracking.add_argument("--bias", type=float, required=True, choices=(0.0, 0.5))
    tracking.add_argument("--output", type=Path, required=True)
    economic = commands.add_parser(
        "economic", help="Replay one consumed economic/PID case"
    )
    economic.add_argument("--case-id", required=True)
    economic.add_argument("--output", type=Path, required=True)
    semantics = commands.add_parser(
        "pid-semantics", help="Replay an original PID/reference pair"
    )
    semantics.add_argument(
        "--mode", choices=("legacy_v0", "physical_v1"), required=True
    )
    semantics.add_argument("--day", type=int, choices=(207, 248, 249), required=True)
    semantics.add_argument("--seed", type=int, choices=range(100, 105), required=True)
    semantics.add_argument("--output", type=Path, required=True)
    archive = commands.add_parser(
        "artifacts", help="Plan or restore the OneDrive-pending evidence"
    )
    archive.add_argument("action", choices=("plan", "restore", "verify"))
    archive.add_argument(
        "--tier",
        action="append",
        required=True,
        choices=("tables", "reevaluate", "audit", "train"),
    )
    archive.add_argument("--cache", type=Path)
    archive.add_argument("--local-bundles", type=Path)
    archive.add_argument("--locations", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    limit_threads()

    if args.command == "inputs":
        if args.input_command == "import":
            emit(inputs.import_inputs(root, args.source_root))
        else:
            result = inputs.status(root)
            emit(result)
            if not result["ready_for_frozen_replay"]:
                raise SystemExit(
                    "Scientific CSV inputs must be explicitly supplied before replay."
                )
        return
    if args.command == "artifacts":
        if args.action == "plan":
            emit(artifacts.plan(root, args.tier))
        elif args.action == "verify":
            emit(artifacts.verify_restored(root, args.tier))
        else:
            emit(
                artifacts.restore(
                    root,
                    args.tier,
                    args.cache or root / ".repro-cache",
                    local_bundles=args.local_bundles,
                    locations=read_json(args.locations) if args.locations else None,
                )
            )
        return

    verification = verify_files(root)
    if args.command == "verify":
        result = {"public_release": verification}
        if not args.code_only:
            result["input_data"] = inputs.status(root)
        emit(result)
        if not args.code_only and not result["input_data"]["ready_for_frozen_replay"]:
            raise SystemExit(
                "Public code is valid; scientific inputs are missing or differ."
            )
        return
    if args.command == "tables":
        output = (
            args.output if args.output.is_absolute() else inside(root, str(args.output))
        )
        command = [
            sys.executable,
            "-B",
            str(inside(root, "provenance/rebuild_numeric_tables.py")),
            "--root",
            str(root),
            "--check-source-hashes",
            "--output-dir",
            str(output.resolve()),
            "--public-output-dir",
            "provenance/numeric_tables",
        ]
    else:
        readiness = inputs.status(root)
        if not readiness["ready_for_frozen_replay"]:
            emit(readiness)
            raise SystemExit(
                "Import the five exact scientific CSVs before frozen replay."
            )
        output = (
            args.output if args.output.is_absolute() else inside(root, str(args.output))
        )
        if args.command == "tracking":
            command = [
                sys.executable,
                "-B",
                str(inside(root, "reproducibility/tracking/tracking_release.py")),
                "evaluate",
                "--root",
                str(root),
                "--snapshot",
                args.snapshot,
                "--policy",
                args.policy,
                "--day",
                str(args.day),
                "--seed",
                str(args.seed),
                "--bias",
                str(args.bias),
                "--output",
                str(output),
            ]
        elif args.command == "economic":
            command = [
                sys.executable,
                "-B",
                str(inside(root, "reproducibility/economic/adapter.py")),
                "evaluate",
                "--root",
                str(root),
                "--case-id",
                args.case_id,
                "--output",
                str(output),
            ]
        else:
            command = [
                sys.executable,
                "-B",
                str(inside(root, "reproducibility/tracking/pid_semantics_replay.py")),
                "evaluate",
                "--root",
                str(root),
                "--mode",
                args.mode,
                "--day",
                str(args.day),
                "--seed",
                str(args.seed),
                "--output",
                str(output),
            ]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
