"""New experiments, with the completed studies and physical model kept frozen."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PRIOR = PROJECT / "outputs/rl_state_local_20261101"
VERIFIED = PROJECT / "outputs/rl_value_conditioning_20261101"
SNAPSHOT_COMMIT = "12c362245f51a4f143e76d2d458231bef4ede398"
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))

# isort: off
from study import baseline_protocol  # noqa: E402
from common import (  # noqa: E402,F401
    assert_default_plant,
    cap_cpu,
    case_key,
    file_hash,
    identity,
    read_json,
    write_json,
)
# isort: on


def verify(protocol: dict, *, require_commit: bool = True) -> str:
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise ValueError("Experiment registration identity changed")
    baseline_protocol()
    for group in ("source_hashes", "input_hashes"):
        for path, digest in protocol[group].items():
            if file_hash(path) != digest:
                raise ValueError(f"Registered {group} artifact changed: {path}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SNAPSHOT_COMMIT, "HEAD"],
        cwd=PROJECT,
        check=True,
    )
    if require_commit:
        paths = [
            str(Path(path).relative_to(PROJECT))
            for path in protocol["source_hashes"]
            if Path(path).is_relative_to(ROOT)
        ]
        paths.append(str((ROOT / protocol["registration_file"]).relative_to(PROJECT)))
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
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
