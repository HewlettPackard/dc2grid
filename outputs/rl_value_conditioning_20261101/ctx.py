"""Isolated follow-up; the first pilot and its rejection decision stay immutable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PRIOR = PROJECT / "outputs/rl_state_local_20261101"
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))

# isort: off
from study import (  # noqa: E402,F401
    BASELINE,
    SOURCE,
    cap_cpu,
    file_hash,
    identity,
    read_json,
    write_json,
    verify_registration as verify_source_hashes,
    FEATURE_CONTRACT,
    ACTION_LOW,
    ACTION_HIGH,
)
# isort: on


def verify(protocol):
    verify_source_hashes(protocol)
    prior = read_json(PRIOR / "registration.json")
    if protocol["parent_registration_id"] != prior["registration_id"]:
        raise ValueError("Follow-up is detached from its original pilot")
    reference = read_json(PRIOR / "development_references.json")
    reference_body = {
        key: value for key, value in reference.items() if key != "reference_id"
    }
    if (
        identity(reference_body) != reference["reference_id"]
        or reference["reference_id"] != protocol["frozen_reference_id"]
        or file_hash(reference["panel_summary"]["path"])
        != reference["panel_summary"]["sha256"]
    ):
        raise ValueError("Historical development comparator lock changed")


def committed(protocol):
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
