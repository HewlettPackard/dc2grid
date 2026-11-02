"""Plan, verify and restore content-addressed scientific evidence artifacts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from .runtime import file_hash, inside, read_json, relative_path, verify_files

INDEX = "reproducibility/artifact_manifest.json"
CHUNK = 1024 * 1024


def inventory(root: Path) -> dict:
    verify_files(root, [INDEX])
    data = read_json(inside(root, INDEX))
    if data["schema"] != "dc2grid-content-addressed-artifacts-v1":
        raise ValueError("Unsupported scientific artifact manifest")
    bundles = {row["name"]: row for row in data["bundles"]}
    if len(bundles) != len(data["bundles"]):
        raise ValueError("Artifact bundle names are not unique")
    referenced = set()
    for path, entry in data["files"].items():
        parts = relative_path(path).parts
        if not (
            parts[0] in ("outputs", "data")
            or parts[:2] == ("reproducibility", "evidence")
        ):
            raise ValueError(f"Artifact destination is outside scientific data: {path}")
        obj = data["objects"][entry["sha256"]]
        if obj["bytes"] != entry["bytes"]:
            raise ValueError(f"Object size differs from file inventory: {path}")
        if obj["bundle"] not in bundles:
            raise ValueError(f"Object has no downloadable bundle: {path}")
        if not set(entry["tiers"]) <= set(data["tiers"]):
            raise ValueError(f"Artifact has an unknown reproduction tier: {path}")
        referenced.add(entry["sha256"])
    if referenced != set(data["objects"]):
        raise ValueError("Artifact inventory contains missing or unreferenced objects")
    return data


def selected_files(data: dict, tiers: list[str]) -> dict:
    allowed = set(data["tiers"])
    if not tiers or not set(tiers) <= allowed:
        raise ValueError(f"Choose one or more declared tiers: {sorted(allowed)}")
    return {
        path: entry
        for path, entry in data["files"].items()
        if set(entry["tiers"]) & set(tiers)
    }


def plan(root: Path, tiers: list[str]) -> dict:
    data = inventory(root)
    selected = selected_files(data, tiers)
    objects = {entry["sha256"] for entry in selected.values()}
    needed = {data["objects"][digest]["bundle"] for digest in objects}
    bundles = [row for row in data["bundles"] if row["name"] in needed]
    return {
        "tiers": tiers,
        "files": len(selected),
        "unique_objects": len(objects),
        "restored_unique_bytes": sum(
            data["objects"][digest]["bytes"] for digest in objects
        ),
        "bundle_download_bytes": sum(row["bytes"] for row in bundles),
        "bundles": bundles,
    }


def download_bundle(
    record: dict, cache: Path, locations: dict[str, str] | None = None
) -> Path:
    name = relative_path(record["name"])
    if len(name.parts) != 1:
        raise ValueError("Bundle names must not contain directories")
    target = cache / name.name
    if target.exists():
        if (
            target.stat().st_size != record["bytes"]
            or file_hash(target) != record["sha256"]
        ):
            raise ValueError(f"Existing bundle is corrupt: {target}")
        return target
    url = (locations or {}).get(record["name"]) or record["download_url"]
    if not url:
        raise RuntimeError(
            "The original audit archive awaits the author's OneDrive upload. "
            "Use --local-bundles or provide an explicit artifact_locations.json; "
            "no remote availability is implied by the checksum inventory."
        )
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("Artifact locations must be explicit HTTPS download URLs")
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Remove or inspect the interrupted download: {partial}")
    digest = hashlib.sha256()
    size = 0
    with (
        urllib.request.urlopen(url, timeout=600) as response,
        partial.open("xb") as out,
    ):
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size > record["bytes"]:
                raise ValueError(f"Download exceeded the declared size: {partial}")
    if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError(f"Downloaded bundle is corrupt; retained at {partial}")
    os.replace(partial, target)
    return target


def unpack_objects(
    bundle: Path,
    record: dict,
    expected: dict,
    wanted: set[str],
    store: Path,
) -> set[str]:
    if (
        bundle.stat().st_size != record["bytes"]
        or file_hash(bundle) != record["sha256"]
    ):
        raise ValueError(f"Bundle identity differs: {bundle}")
    found = set()
    seen = set()
    with tarfile.open(bundle, mode="r|*") as archive:
        for member in archive:
            parts = relative_path(member.name).parts
            if (
                len(parts) != 3
                or parts[0] != "objects"
                or len(parts[2]) != 64
                or any(letter not in "0123456789abcdef" for letter in parts[2])
                or parts[1] != parts[2][:2]
                or not member.isfile()
            ):
                raise ValueError(f"Invalid content-addressed tar member: {member.name}")
            digest = parts[2]
            if digest in seen or digest not in expected:
                raise ValueError(f"Duplicate or undeclared tar object: {digest}")
            seen.add(digest)
            metadata = expected[digest]
            if metadata["bundle"] != record["name"] or metadata["bytes"] != member.size:
                raise ValueError(f"Object manifest differs: {digest}")
            if digest not in wanted:
                continue
            target = store / digest
            if target.exists():
                if target.stat().st_size != member.size or file_hash(target) != digest:
                    raise ValueError(f"Cached object is corrupt: {digest}")
                found.add(digest)
                continue
            partial = target.with_name(target.name + ".partial")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Declared regular object has no content: {digest}")
            measured = hashlib.sha256()
            size = 0
            with source, partial.open("xb") as out:
                while chunk := source.read(CHUNK):
                    measured.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
            if size != member.size or measured.hexdigest() != digest:
                raise ValueError(f"Object content is corrupt; retained at {partial}")
            os.replace(partial, target)
            found.add(digest)
    required = {
        digest
        for digest, metadata in expected.items()
        if metadata["bundle"] == record["name"]
    }
    if seen != required:
        raise ValueError(f"Bundle member set is incomplete: {record['name']}")
    return found


def restore(
    root: Path,
    tiers: list[str],
    cache: Path,
    *,
    local_bundles: Path | None = None,
    locations: dict[str, str] | None = None,
) -> dict:
    data = inventory(root)
    selected = selected_files(data, tiers)
    wanted = {row["sha256"] for row in selected.values()}
    needed = {data["objects"][digest]["bundle"] for digest in wanted}
    bundles = [row for row in data["bundles"] if row["name"] in needed]
    cache.mkdir(parents=True, exist_ok=True)
    store = cache / "objects"
    store.mkdir(exist_ok=True)
    found = set()
    for record in bundles:
        if local_bundles is None:
            bundle = download_bundle(record, cache, locations)
        else:
            bundle = inside(local_bundles, record["name"])
        found |= unpack_objects(bundle, record, data["objects"], wanted, store)
    if found != wanted:
        raise ValueError("Artifact object closure is incomplete")
    copied = linked = present = 0
    for relative, record in selected.items():
        target = inside(root, relative)
        digest = record["sha256"]
        if target.exists():
            if target.stat().st_size != record["bytes"] or file_hash(target) != digest:
                raise ValueError(
                    f"Refusing to overwrite differing scientific data: {target}"
                )
            present += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(store / digest, target)
            linked += 1
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            with (store / digest).open("rb") as source, target.open("xb") as out:
                shutil.copyfileobj(source, out, length=CHUNK)
            if file_hash(target) != digest:
                raise ValueError(f"Artifact copy verification failed: {target}")
            copied += 1
    receipt = {
        "tiers": tiers,
        "files": len(selected),
        "unique_objects": len(wanted),
        "bundles_verified": len(bundles),
        "already_present": present,
        "hardlinked": linked,
        "copied": copied,
        "overwritten_existing_files": 0,
    }
    (cache / "restoration_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def verify_restored(root: Path, tiers: list[str]) -> dict:
    selected = selected_files(inventory(root), tiers)
    for relative, record in selected.items():
        path = inside(root, relative)
        if not path.is_file():
            raise FileNotFoundError(
                f"Required scientific artifact is absent: {relative}"
            )
        if (
            path.stat().st_size != record["bytes"]
            or file_hash(path) != record["sha256"]
        ):
            raise ValueError(f"Scientific artifact identity differs: {relative}")
    return {"verified_artifact_files": len(selected), "tiers": tiers}
