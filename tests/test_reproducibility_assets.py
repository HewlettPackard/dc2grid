"""Offline tests for the shared, fail-closed evidence fetcher."""

import hashlib
import io
import json
from unittest.mock import patch

import pytest

from reproducibility import assets
from reproducibility.runtime import (
    CORE_COMMIT,
    CORE_SOURCE_AMENDMENTS,
    inside,
    integration_sources,
    new_output,
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "checkout"
    directory = root / "reproducibility/table2"
    directory.mkdir(parents=True)
    payload = b"published evidence\n"
    spec = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    name = "outputs/example/receipt.json"
    record = {
        "schema": "dc2grid-table-integration-v1",
        "study": "table2",
        "source_public_commit": assets.PUBLIC_SNAPSHOTS["table2"],
        "core_commit": CORE_COMMIT,
        "sources": {},
        "assets": {name: spec},
    }
    manifest = directory / "integration.json"
    manifest.write_text(json.dumps(record))
    return root, payload, name, record, manifest


def test_pinned_public_fetch_and_reuse(bundle):
    root, payload, name, _, _ = bundle
    with patch.object(assets, "urlopen", return_value=io.BytesIO(payload)) as download:
        assert assets.fetch(root, "table2")["copied"] == 1
        assert download.call_args.args[0].full_url == (
            "https://raw.githubusercontent.com/HewlettPackard/dc2grid/"
            f"{assets.PUBLIC_SNAPSHOTS['table2']}/{name}"
        )
    with patch.object(
        assets, "urlopen", side_effect=AssertionError("Unexpected network")
    ):
        assert assets.fetch(root, "table2")["copied"] == 0


def test_bad_download_is_not_installed(bundle):
    root, _, name, _, _ = bundle
    with patch.object(assets, "urlopen", return_value=io.BytesIO(b"wrong")):
        with pytest.raises(ValueError, match="pinned identity"):
            assets.fetch(root, "table2")
    assert not (root / name).exists()
    assert not list((root / "outputs/example").glob("*.fetching"))


def test_existing_different_asset_is_not_overwritten(bundle):
    root, _, name, _, _ = bundle
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"keep this")
    with pytest.raises(ValueError, match="pinned identity"):
        assets.fetch(root, "table2")
    assert path.read_bytes() == b"keep this"


def test_local_source_is_still_hash_checked(bundle, tmp_path):
    root, payload, name, _, _ = bundle
    source = tmp_path / "source"
    path = source / name
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with patch.object(
        assets, "urlopen", side_effect=AssertionError("Unexpected network")
    ):
        result = assets.fetch(root, "table2", source_root=source)
    assert result["copied"] == 1
    assert result["source"] == "verified-local"


@pytest.mark.parametrize(
    "name",
    [
        "../outside.json",
        "/tmp/outside.json",
        "outputs/../outside.json",
        r"outputs\file.json",
    ],
)
def test_unsafe_paths_are_rejected(bundle, name):
    root, _, _, _, _ = bundle
    with pytest.raises(ValueError):
        inside(root, name)


@pytest.mark.parametrize("name", ["data/processed/input.csv", "outputs/run.py"])
def test_datasets_and_executable_sources_are_not_downloadable(bundle, name):
    root, _, old, record, manifest = bundle
    record["assets"][name] = record["assets"].pop(old)
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="approved result/checkpoint"):
        assets.manifest(root, "table2")


def test_source_drift_blocks_fetch(bundle):
    root, _, name, record, manifest = bundle
    record["sources"]["missing.py"] = record["assets"][name]
    manifest.write_text(json.dumps(record))
    with pytest.raises(FileNotFoundError, match="missing.py"):
        assets.fetch(root, "table2")


def test_wrong_snapshot_is_not_accepted(bundle):
    root, _, _, record, manifest = bundle
    record["source_public_commit"] = "0" * 40
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="snapshot"):
        assets.manifest(root, "table2")


def test_asset_symlink_is_rejected(bundle):
    root, payload, name, _, _ = bundle
    source = root / "source"
    source.write_bytes(payload)
    path = root / name
    path.parent.mkdir(parents=True)
    path.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        assets.manifest(root, "table2")


def test_known_core_amendment_is_explicit_and_exclusive():
    path, amendment = next(iter(CORE_SOURCE_AMENDMENTS.items()))
    original = {path: amendment["producer_sha256"], "other.py": "unmodified"}
    amended = integration_sources(original)
    assert amended[path] == amendment["integration_sha256"]
    assert amended["other.py"] == original["other.py"]
    assert original[path] == amendment["producer_sha256"]
    with pytest.raises(ValueError, match="producer source"):
        integration_sources({path: "unexpected"})


def test_numeric_outputs_cannot_overwrite_or_escape(bundle, tmp_path):
    root, _, _, _, _ = bundle
    with pytest.raises(FileExistsError):
        new_output(root / "reproducibility", root)
    with pytest.raises(ValueError):
        new_output(tmp_path / "outside", root)
    assert new_output(root / "outputs/numeric", root).is_dir()
