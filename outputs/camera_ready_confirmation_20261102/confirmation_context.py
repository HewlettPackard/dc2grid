"""Independent frozen-policy confirmation; historical studies remain immutable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = PROJECT / "outputs/rl_grid_cash_ppo_20261101"
BASELINE_COMMIT = "b2978771d7e03c0e16ccfbba2712e64b97557bbb"
AUTHOR_SNAPSHOT = Path(
    "/root/.vscode-server/data/agentSessionData/"
    "a18d3a20-df5f-4f65-9b30-5e7269163294/attachments/"
    "9b22d005-5765-4f73-80f8-bd81956b8c85/file_camera_ready.tex"
)
sys.dont_write_bytecode = True
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
for path in (
    PREVIOUS / "track_b_fraction",
    PREVIOUS / "track_b",
    PREVIOUS / "track_a",
    PREVIOUS,
):
    sys.path.append(str(path))

# isort: off
import economy_context  # noqa: E402,F401
from common import (  # noqa: E402,F401
    cap_cpu,
    case_key,
    file_hash,
    identity,
    read_json,
    write_json,
)
# isort: on

os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")


def initialize_worker(queue) -> None:
    cap_cpu(queue.get())


def verify(protocol: dict, *, committed: bool = True) -> str:
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise ValueError("Confirmation registration identity changed")
    for name in ("source_hashes", "input_hashes"):
        for path, digest in protocol[name].items():
            if file_hash(path) != digest:
                raise ValueError(f"Frozen confirmation input changed: {path}")
    if (PREVIOUS / "final_opening_grid_cash.json").exists():
        raise RuntimeError(
            "The previous failed study's reserved panel must stay unopened"
        )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
        cwd=PROJECT,
        check=True,
    )
    if committed:
        paths = [
            str(Path(path).relative_to(PROJECT))
            for path in protocol["source_hashes"]
            if Path(path).is_relative_to(ROOT)
        ]
        paths += [str((ROOT / "registration.json").relative_to(PROJECT))]
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
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        text=True,
    ).strip()
