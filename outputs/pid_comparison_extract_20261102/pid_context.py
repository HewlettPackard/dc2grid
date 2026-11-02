"""Separate post-confirmation PID references; no frozen study is modified."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CONFIRMATION = PROJECT / "outputs/camera_ready_confirmation_20261102"
ECONOMIC = PROJECT / "outputs/rl_grid_cash_ppo_20261101"
BASELINE = "9e2496a62c75f6593181d47c42ae117f9801fdf6"
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
sys.dont_write_bytecode = True
sys.path.insert(0, str(CONFIRMATION))

# isort: off
from confirmation_context import (  # noqa: E402,F401
    cap_cpu,
    case_key,
    effective_source_hashes,
    file_hash,
    identity,
    initialize_worker,
    read_json,
    write_json,
)
# isort: on

os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")


def verify(protocol, *, committed=True):
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise ValueError("PID reference registration changed")
    for group in ("source_hashes", "input_hashes"):
        for path, digest in protocol[group].items():
            if file_hash(path) != digest:
                raise ValueError(f"Frozen PID reference input changed: {path}")
    if (ECONOMIC / "final_opening_grid_cash.json").exists():
        raise RuntimeError("The older unused reservation must remain closed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"],
        cwd=PROJECT,
        check=True,
    )
    if committed:
        paths = [
            str(Path(path).relative_to(PROJECT))
            for path in protocol["source_hashes"]
            if Path(path).is_relative_to(ROOT)
        ] + [str((ROOT / "registration.json").relative_to(PROJECT))]
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *paths],
            cwd=PROJECT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *paths],
            cwd=PROJECT,
            check=True,
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        text=True,
    ).strip()
