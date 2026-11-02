"""Import five exact caller-supplied CSVs; never substitute or download inputs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath

from .runtime import file_hash, inside, read_json, verify_files

INPUTS = "reproducibility/input_data_manifest.json"


def specifications(root: Path) -> list[dict]:
    verify_files(root, [INPUTS])
    record = read_json(inside(root, INPUTS))
    if record["schema"] != "dc2grid-user-supplied-input-data-v1":
        raise ValueError("Unsupported scientific input manifest")
    if record["redistributed_in_public_git"] is not False:
        raise ValueError("The release must not redistribute external input datasets")
    rows = record["files"]
    if len(rows) != 5 or len({row["path"] for row in rows}) != 5:
        raise ValueError("This release requires exactly five distinct processed CSVs")
    for row in rows:
        path = PurePosixPath(row["path"])
        if path.parts[:2] != ("data", "processed") or path.suffix != ".csv":
            raise ValueError("Scientific inputs must remain in the declared data tree")
        inside(root, row["path"])
    return rows


def status(root: Path) -> dict:
    missing, mismatched, verified = [], [], []
    for row in specifications(root):
        path = inside(root, row["path"])
        if not path.is_file():
            missing.append(row["path"])
        elif path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            mismatched.append(row["path"])
        else:
            verified.append(row["path"])
    return {
        "ready_for_frozen_replay": not missing and not mismatched,
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
        "datasets_distributed_in_public_git": False,
        "automatic_download_or_synthetic_substitution": False,
    }


def import_inputs(root: Path, source_root: Path) -> dict:
    rows = specifications(root)
    source_root = source_root.resolve(strict=True)
    sources = {}
    for row in rows:
        source = source_root / row["path"]
        if not source.resolve().is_relative_to(source_root):
            raise ValueError(f"Input source escapes the supplied root: {row['path']}")
        if not source.is_file():
            raise FileNotFoundError(
                f"Caller-supplied scientific input is absent: {source}"
            )
        if source.stat().st_size != row["bytes"] or file_hash(source) != row["sha256"]:
            raise ValueError(
                f"Caller-supplied input does not match the recorded bytes: {source}"
            )
        destination = inside(root, row["path"])
        if destination.exists() and (
            destination.stat().st_size != row["bytes"]
            or file_hash(destination) != row["sha256"]
        ):
            raise ValueError(
                f"Refusing to overwrite an existing different input: {destination}"
            )
        sources[row["path"]] = source
    copied = 0
    for row in rows:
        target = inside(root, row["path"])
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".importing")
        with sources[row["path"]].open("rb") as source, temporary.open("xb") as out:
            shutil.copyfileobj(source, out, length=1024 * 1024)
        if (
            temporary.stat().st_size != row["bytes"]
            or file_hash(temporary) != row["sha256"]
        ):
            raise ValueError(f"Input changed during copying; retained at {temporary}")
        os.replace(temporary, target)
        copied += 1
    result = status(root)
    if not result["ready_for_frozen_replay"]:
        raise ValueError("Input closure failed after the explicit local import")
    return {
        **result,
        "copied": copied,
        "input_rights_remain_the_callers_responsibility": True,
        "redistribution_permission_granted_by_this_tool": False,
    }
