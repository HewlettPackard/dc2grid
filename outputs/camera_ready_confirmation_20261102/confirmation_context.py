"""Independent frozen-policy confirmation; historical studies remain immutable."""

from __future__ import annotations

import hashlib
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


def effective_source_hashes(protocol: dict) -> dict:
    hashes = dict(protocol["source_hashes"])
    path = ROOT / "technical_amendment.json"
    if not path.exists():
        return hashes
    amendment = read_json(path)
    body = {key: value for key, value in amendment.items() if key != "amendment_id"}
    if (
        identity(body) != amendment["amendment_id"]
        or amendment["registration_id"] != protocol["registration_id"]
    ):
        raise ValueError("The technical amendment identity changed")
    permitted = {
        str(ROOT / name)
        for name in (
            "confirmation_context.py",
            "confirmation_audit.py",
            "confirmation_analysis.py",
            "run_confirmation.py",
            "test_confirmation.py",
        )
    }
    overrides = amendment["source_hash_overrides"]
    if set(overrides) - permitted:
        raise ValueError("The technical amendment cannot change policy or task code")
    if amendment["new_source_files"] != {
        str(ROOT / "register_technical_amendment.py"): file_hash(
            ROOT / "register_technical_amendment.py"
        ),
    }:
        raise ValueError("The amendment source generator changed")
    commit = amendment["original_registration_commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=PROJECT,
        check=True,
    )
    for name in overrides:
        original = subprocess.check_output(
            ["git", "show", f"{commit}:{Path(name).relative_to(PROJECT)}"],
            cwd=PROJECT,
        )
        if hashlib.sha256(original).hexdigest() != hashes[name]:
            raise ValueError("The original registered source was not preserved in Git")
    for name in ("policies", "panels", "analysis", "task_protocol"):
        if (
            identity(protocol[name])
            != amendment["unchanged_scientific_spec_hashes"][name]
        ):
            raise ValueError("A scientific decision changed in a technical amendment")
    inventory = amendment["retained_known_result_inventory"]
    if file_hash(inventory["path"]) != inventory["sha256"]:
        raise ValueError("The pre-amendment evidence inventory changed")
    for artifact in read_json(inventory["path"]):
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("A completed pre-amendment outcome changed")
    hashes.update(overrides)
    hashes.update(amendment["new_source_files"])
    return hashes


def technical_amendment_id() -> str | None:
    path = ROOT / "technical_amendment.json"
    return read_json(path)["amendment_id"] if path.exists() else None


def verify(protocol: dict, *, committed: bool = True) -> str:
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise ValueError("Confirmation registration identity changed")
    sources = effective_source_hashes(protocol)
    for hashes in (sources, protocol["input_hashes"]):
        for path, digest in hashes.items():
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
            for path in sources
            if Path(path).is_relative_to(ROOT)
        ]
        paths += [str((ROOT / "registration.json").relative_to(PROJECT))]
        if (ROOT / "technical_amendment.json").exists():
            paths.append(str((ROOT / "technical_amendment.json").relative_to(PROJECT)))
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
