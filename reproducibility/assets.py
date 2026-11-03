"""Fetch only a table's hash-pinned public evidence; never download input datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from . import inputs
from .runtime import CORE_COMMIT, file_hash, inside, read_json

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SNAPSHOTS = {
    "table1": "927c527adf19e5240caaf5580579646f4381d290",
    "table2": "82acb0072804dd27d11890bcbf9ebdc3d0a8cb7e",
}


def manifest(root: Path, study: str) -> dict:
    if study not in PUBLIC_SNAPSHOTS:
        raise ValueError("Only Table 1 and Table 2 integration manifests are supported")
    record = read_json(inside(root, f"reproducibility/{study}/integration.json"))
    if (
        record["schema"] != "dc2grid-table-integration-v1"
        or record["study"] != study
        or record["source_public_commit"] != PUBLIC_SNAPSHOTS[study]
        or record["core_commit"] != CORE_COMMIT
    ):
        raise ValueError("Unexpected public snapshot or shared-core identity")
    if set(record["sources"]) & set(record["assets"]):
        raise ValueError("A source file cannot also be a downloadable asset")
    for group in ("sources", "assets"):
        for name, spec in record[group].items():
            path = inside(root, name)
            if (
                not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"])
                or type(spec["bytes"]) is not int
                or spec["bytes"] < 0
            ):
                raise ValueError(f"Invalid file identity: {name}")
            if path.is_symlink():
                raise ValueError(f"Refusing a symlinked scientific file: {name}")
            if group == "assets" and (
                not name.startswith("outputs/")
                or path.suffix not in {".json", ".csv", ".zip", ".pdf"}
            ):
                raise ValueError(f"Not an approved result/checkpoint asset: {name}")
    return record


def verify_file(root: Path, name: str, spec: dict) -> None:
    path = inside(root, name)
    if not path.is_file():
        raise FileNotFoundError(f"Required scientific file is missing: {name}")
    if path.stat().st_size != spec["bytes"] or file_hash(path) != spec["sha256"]:
        raise ValueError(f"Scientific file differs from its pinned identity: {name}")


def verify(root: Path, study: str, *, sources_only: bool = False) -> dict:
    record = manifest(root, study)
    groups = ("sources",) if sources_only else ("sources", "assets")
    for group in groups:
        for name, spec in record[group].items():
            verify_file(root, name, spec)
    return {
        "verified": True,
        "study": study,
        "source_public_commit": record["source_public_commit"],
        "core_commit": CORE_COMMIT,
        "source_files": len(record["sources"]),
        "asset_files": 0 if sources_only else len(record["assets"]),
        "asset_bytes": 0
        if sources_only
        else sum(spec["bytes"] for spec in record["assets"].values()),
        "source_binding_sha256": file_hash(
            inside(root, f"reproducibility/{study}/integration.json")
        ),
        "input_datasets_checked": False,
    }


def fetch(root: Path, study: str, *, source_root: Path | None = None) -> dict:
    root = root.resolve(strict=True)
    record = manifest(root, study)
    verify(root, study, sources_only=True)
    if source_root is not None:
        source_root = source_root.resolve(strict=True)
    copied = 0
    for name, spec in record["assets"].items():
        target = inside(root, name)
        if target.exists():
            verify_file(root, name, spec)
            continue
        if source_root is not None:
            verify_file(source_root, name, spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=target.name + ".",
            suffix=".fetching",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            if source_root is None:
                url = (
                    "https://raw.githubusercontent.com/HewlettPackard/dc2grid/"
                    f"{record['source_public_commit']}/{quote(name, safe='/')}"
                )
                request = Request(url, headers={"User-Agent": "dc2grid-reproduction"})
                source = urlopen(request, timeout=120)
            else:
                source = inside(source_root, name).open("rb")
            with source, temporary_path.open("wb") as stream:
                remaining = spec["bytes"] + 1
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    stream.write(block)
                    remaining -= len(block)
            verify_file(root, str(temporary_path.relative_to(root)), spec)
            # Atomic publication cannot replace a concurrent file.
            os.link(temporary_path, target)
            verify_file(root, name, spec)
        finally:
            temporary_path.unlink()
        copied += 1
    return {
        **verify(root, study),
        "copied": copied,
        "datasets_downloaded": False,
        "source": "public-immutable-commit"
        if source_root is None
        else "verified-local",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("study", choices=tuple(PUBLIC_SNAPSHOTS))
        if command == "fetch":
            sub.add_argument("--source-root", type=Path)
        else:
            sub.add_argument("--sources-only", action="store_true")
    data = commands.add_parser("inputs")
    data.add_argument("action", choices=("status", "import"))
    data.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        result = fetch(ROOT, args.study, source_root=args.source_root)
    elif args.command == "verify":
        result = verify(ROOT, args.study, sources_only=args.sources_only)
    elif args.action == "import":
        if args.source_root is None:
            parser.error("inputs import requires --source-root")
        result = inputs.import_inputs(ROOT, args.source_root)
    else:
        if args.source_root is not None:
            parser.error("--source-root is only supported by inputs import")
        result = inputs.status(ROOT)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("ready_for_frozen_replay") is False:
        raise SystemExit("Supply the exact authorized CSVs before replay; no fallback.")


if __name__ == "__main__":
    main()
