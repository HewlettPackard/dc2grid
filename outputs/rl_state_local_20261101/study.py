"""Bootstrap a new study without modifying the committed scientific baseline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASELINE = PROJECT / "outputs/table2_distress_rl_20261031"
SOURCE = PROJECT / "outputs/online_stability_pilot_20261031/source"
HIGH_BASELINE = PROJECT / "outputs/high_macro_recovery_pilot_20261031_v3"
BASELINE_COMMIT = "5c7e057caf686853934a19f5c56e848dbc96ae68"
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASELINE))

# isort: off
from common import cap_cpu, file_hash, identity, read_json, write_json  # noqa: E402
from history_policy import ACTION_HIGH, ACTION_LOW, FEATURE_CONTRACT  # noqa: E402
from prepare_study import verify_protocol as verify_baseline  # noqa: E402
# isort: on

VARIANTS = ("fixed_reference_kl", "replay_reference_kl")


def baseline_protocol():
    protocol = read_json(BASELINE / "registration.json")
    verify_baseline(protocol)
    return protocol


def verify_registration(protocol):
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise ValueError("New-study registration identity changed")
    for path, expected in protocol["source_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Registered source changed: {path}")
    for path, expected in protocol["input_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Registered input changed: {path}")
    baseline_protocol()


def implementation_commit(protocol):
    paths = [
        str(Path(path).relative_to(PROJECT))
        for path in protocol["source_hashes"]
        if Path(path).is_relative_to(ROOT)
    ] + [str((ROOT / "registration.json").relative_to(PROJECT))]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=PROJECT,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


__all__ = [
    "ROOT",
    "PROJECT",
    "BASELINE",
    "SOURCE",
    "HIGH_BASELINE",
    "BASELINE_COMMIT",
    "VARIANTS",
    "ACTION_LOW",
    "ACTION_HIGH",
    "FEATURE_CONTRACT",
    "cap_cpu",
    "file_hash",
    "identity",
    "read_json",
    "write_json",
    "baseline_protocol",
    "verify_registration",
    "implementation_commit",
]
