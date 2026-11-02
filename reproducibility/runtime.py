"""Verify scientific bytes and resolve historical paths without changing records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

ORIGINAL_ROOT = "/workspace/IBM/temp/dc2grid-repos/dc2grid"
MANIFEST_PATH = "reproducibility/release_manifest.json"


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or "\x00" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe release-relative path: {value!r}")
    return path


def inside(root: Path, value: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*relative_path(value).parts)
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes the release checkout: {value}")
    return candidate


def manifest(root: Path) -> dict:
    data = read_json(inside(root, MANIFEST_PATH))
    if data["schema"] != "dc2grid-clean-comparison-release-v1":
        raise ValueError("Unsupported scientific release manifest")
    if data["repository"] != "https://github.com/HewlettPackard/dc2grid":
        raise ValueError("Unexpected release repository")
    return data


def resolve_recorded(root: Path, value: str, aliases: dict | None = None) -> Path:
    aliases = {} if aliases is None else aliases
    for prefix, destination in sorted(
        aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if value == prefix or value.startswith(prefix.rstrip("/") + "/"):
            remainder = value[len(prefix) :].lstrip("/")
            relative = str(PurePosixPath(destination) / remainder)
            return inside(root, relative)
    if value.startswith(ORIGINAL_ROOT + "/"):
        return inside(root, value[len(ORIGINAL_ROOT) + 1 :])
    if not PurePosixPath(value).is_absolute():
        return inside(root, value)
    raise ValueError(f"Unmapped historical path: {value}")


def verify_files(root: Path, names: list[str] | None = None) -> dict:
    record = manifest(root)
    selected = record["files"] if names is None else names
    count = 0
    total = 0
    for name in selected:
        if name not in record["files"]:
            raise ValueError(f"File has no published release identity: {name}")
        expected = record["files"][name]
        path = inside(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is absent: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Release file size differs: {name}")
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Release file content differs: {name}")
        count += 1
        total += expected["bytes"]
    return {
        "verified_files": count,
        "verified_bytes": total,
        "release_tag": record["release_tag"],
    }


def limit_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def new_output(path: Path, root: Path) -> Path:
    path = path.resolve()
    frozen = manifest(root)["files"]
    if path == root.resolve() or any(
        inside(root, name).is_relative_to(path) for name in frozen
    ):
        raise ValueError("Output must not contain or replace frozen release files")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {path}")
    path.mkdir(parents=True)
    return path
