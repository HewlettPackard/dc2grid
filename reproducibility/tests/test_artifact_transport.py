import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from reproducibility.artifacts import download_bundle, plan, restore, verify_restored
from reproducibility.runtime import file_hash, inside


@pytest.fixture
def release(tmp_path: Path):
    root = tmp_path / "checkout"
    metadata = root / "reproducibility"
    metadata.mkdir(parents=True)
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    payload = b"bounded numeric scientific evidence\n"
    digest = hashlib.sha256(payload).hexdigest()
    archive_path = bundles / "audit-0001.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo(f"objects/{digest[:2]}/{digest}")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    data = {
        "schema": "dc2grid-content-addressed-artifacts-v1",
        "tiers": ["audit"],
        "files": {
            "outputs/example/trace.csv": {
                "sha256": digest,
                "bytes": len(payload),
                "tiers": ["audit"],
            }
        },
        "objects": {digest: {"bytes": len(payload), "bundle": archive_path.name}},
        "bundles": [
            {
                "name": archive_path.name,
                "sha256": file_hash(archive_path),
                "bytes": archive_path.stat().st_size,
                "download_url": (
                    "https://github.com/HewlettPackard/dc2grid/releases/download/"
                    "repro%2F20261102%2Fpid-comparison/audit-0001.tar"
                ),
            }
        ],
    }

    def save():
        path = metadata / "artifact_manifest.json"
        path.write_text(json.dumps(data))
        (metadata / "release_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "dc2grid-clean-comparison-release-v1",
                    "repository": "https://github.com/HewlettPackard/dc2grid",
                    "release_tag": "repro/20261102/pid-comparison",
                    "files": {
                        "reproducibility/artifact_manifest.json": {
                            "sha256": file_hash(path),
                            "bytes": path.stat().st_size,
                        }
                    },
                }
            )
        )

    save()
    return root, bundles, data, payload, save


def test_restoration_is_complete_and_idempotent(release, tmp_path: Path):
    root, bundles, _, payload, _ = release
    expected = plan(root, ["audit"])
    assert expected["files"] == expected["unique_objects"] == 1
    first = restore(root, ["audit"], tmp_path / "cache", local_bundles=bundles)
    assert first["hardlinked"] == 1 and first["overwritten_existing_files"] == 0
    assert (root / "outputs/example/trace.csv").read_bytes() == payload
    second = restore(root, ["audit"], tmp_path / "cache", local_bundles=bundles)
    assert second["already_present"] == 1
    assert verify_restored(root, ["audit"])["verified_artifact_files"] == 1


def test_corrupt_bundle_cannot_be_restored(release, tmp_path: Path):
    root, bundles, data, _, _ = release
    path = bundles / data["bundles"][0]["name"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        restore(root, ["audit"], tmp_path / "cache", local_bundles=bundles)
    assert not (root / "outputs/example/trace.csv").exists()


def test_existing_scientific_file_is_never_overwritten(release, tmp_path: Path):
    root, bundles, _, _, _ = release
    path = root / "outputs/example/trace.csv"
    path.parent.mkdir(parents=True)
    path.write_text("unrelated existing result")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        restore(root, ["audit"], tmp_path / "cache", local_bundles=bundles)
    assert path.read_text() == "unrelated existing result"


def test_tar_links_are_not_extracted(release, tmp_path: Path):
    root, bundles, data, _, save = release
    digest = next(iter(data["objects"]))
    path = bundles / data["bundles"][0]["name"]
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo(f"objects/{digest[:2]}/{digest}")
        info.type = tarfile.SYMTYPE
        info.linkname = "/outside/checkout"
        archive.addfile(info)
    data["bundles"][0].update(
        sha256=file_hash(path),
        bytes=path.stat().st_size,
    )
    save()
    with pytest.raises(ValueError, match="Invalid content-addressed tar member"):
        restore(root, ["audit"], tmp_path / "cache", local_bundles=bundles)


def test_inventory_rejects_git_metadata_destinations(release):
    root, _, data, _, save = release
    data["files"][".git/config"] = data["files"].pop("outputs/example/trace.csv")
    save()
    with pytest.raises(ValueError, match="outside scientific data"):
        plan(root, ["audit"])


def test_escape_and_unmapped_tiers_are_rejected(release):
    root, _, _, _, _ = release
    with pytest.raises(ValueError, match="Unsafe release-relative"):
        inside(root, "../other")
    with pytest.raises(ValueError, match="declared tiers"):
        plan(root, ["unknown"])


def test_pending_onedrive_location_is_not_a_successful_download(tmp_path: Path):
    with pytest.raises(RuntimeError, match="awaits the author's OneDrive upload"):
        download_bundle(
            {
                "name": "audit-001.tar.gz",
                "download_url": None,
                "bytes": 100,
                "sha256": "a" * 64,
            },
            tmp_path,
        )
    assert not list(tmp_path.iterdir())
