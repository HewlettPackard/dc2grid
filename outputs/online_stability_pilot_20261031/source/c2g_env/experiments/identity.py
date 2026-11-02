from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
import warnings
from pathlib import Path
from typing import Any, Mapping


CONDITION_COLUMNS = ("simulation_mode", "plant_profile", "facility_capacity_mw", "config_id", "thermal_overrides")


def check_policy_semantics(model: Any, mode: str, *, allow_mismatch: bool = False) -> str:
    space_mode = getattr(getattr(model, "observation_space", None), "c2g_simulation_mode", "legacy_v0")
    trained_mode = getattr(model, "c2g_context", {}).get("simulation_mode", space_mode)
    if trained_mode != mode:
        message = (
            f"Checkpoint semantics {trained_mode!r} differ from environment {mode!r}. "
            "Retrain for the new semantics, or explicitly allow a mismatch for frozen-policy diagnosis."
        )
        if not allow_mismatch:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return trained_mode


def result_condition(row: Mapping[str, Any]) -> dict[str, Any]:
    condition = {key: row[key] for key in CONDITION_COLUMNS if key in row}
    thermal = condition.get("thermal_overrides")
    if isinstance(thermal, str):
        condition["thermal_overrides"] = json.loads(thermal)
    return condition


def group_conditions(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    groups = {}
    for row in rows:
        condition = result_condition(row)
        key = configuration_id(condition)
        if key not in groups:
            groups[key] = (condition, [])
        groups[key][1].append(row)
    return list(groups.values())


def configuration_id(configuration: Mapping[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def source_id(root: Path) -> str:
    sources = {}
    for directory in ("c2g_env", "baselines", "evaluation", "scripts"):
        for path in sorted((root / directory).rglob("*.py")):
            sources[path.relative_to(root).as_posix()] = file_fingerprint(path)["sha256"]
    return configuration_id(sources)


def capture_source(root: Path, output_dir: Path) -> dict[str, Any]:
    paths = [path for directory in ("c2g_env", "baselines", "evaluation", "scripts") for path in (root / directory).rglob("*.py")]
    paths.extend((root / "conf").rglob("*.yaml"))
    paths.append(root / "c2g_env" / "config.yaml")
    checksums = {path.relative_to(root).as_posix(): file_fingerprint(path)["sha256"] for path in sorted(paths)}
    identity = configuration_id(checksums)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"source_{identity}.zip"
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                archive.write(path, path.relative_to(root).as_posix())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "c2g_env", "baselines", "evaluation", "scripts", "conf"],
                            cwd=root, text=True, capture_output=True, check=True).stdout
    return {"source_id": identity, "code_revision": revision, "local_modifications": status.splitlines(),
            "source_archive": str(archive_path.resolve())}