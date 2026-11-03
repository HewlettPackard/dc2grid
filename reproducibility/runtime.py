"""Shared path, identity and source-binding rules for table reproduction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

ORIGINAL_ROOT = "/workspace/IBM/temp/dc2grid-repos/dc2grid"
CORE_COMMIT = "910343302e6671b16112047bc98f81d530fe6cf0"
CORE_SOURCE_AMENDMENTS = {
    "c2g_env/env_high_level.py": {
        "producer_sha256": (
            "6c5034698fdd011168d99b5e2fd99b13d2825c4af37eaab563fa6acf75381842"
        ),
        "integration_sha256": (
            "28fc67d639dffa74aa3c57d8ec056fd1701381f3b9f3a5f2eb8fb61c5b958c97"
        ),
        "reason": (
            "Construct Box after finalizing bounds, fixing cached sampling metadata. "
            "Numeric bounds, observations, actions, rewards and old ZIPs are unchanged."
        ),
    }
}


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def integration_sources(producer_sources: dict[str, str]) -> dict[str, str]:
    result = dict(producer_sources)
    for path, amendment in CORE_SOURCE_AMENDMENTS.items():
        if result.get(path) != amendment["producer_sha256"]:
            raise ValueError(
                f"Unexpected producer source for the core amendment: {path}"
            )
        result[path] = amendment["integration_sha256"]
    return result


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError(f"Unsafe release-relative path: {value!r}")
    return path


def inside(root: Path, value: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*relative_path(value).parts)
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes the release checkout: {value}")
    return candidate


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
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError("Output must be a new directory inside this checkout")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path
