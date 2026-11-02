import hashlib
import json
from pathlib import Path

import pytest

from reproducibility.inputs import import_inputs, status


@pytest.fixture
def inputs(tmp_path: Path):
    root = tmp_path / "release"
    records = root / "reproducibility"
    records.mkdir(parents=True)
    source = tmp_path / "supplied"
    rows = []
    for index in range(5):
        relative = f"data/processed/example/input-{index}.csv"
        payload = f"Time,Load\n0,{index}\n".encode()
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    data = {
        "schema": "dc2grid-user-supplied-input-data-v1",
        "redistributed_in_public_git": False,
        "files": rows,
    }
    manifest = records / "input_data_manifest.json"
    manifest.write_text(json.dumps(data))
    (records / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema": "dc2grid-clean-comparison-release-v1",
                "repository": "https://github.com/HewlettPackard/dc2grid",
                "release_tag": "repro/20261102/pid-comparison",
                "files": {
                    "reproducibility/input_data_manifest.json": {
                        "bytes": manifest.stat().st_size,
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    return root, source, rows


def test_missing_inputs_are_explicit_and_never_synthesized(inputs):
    root, _, _ = inputs
    result = status(root)
    assert result["ready_for_frozen_replay"] is False
    assert len(result["missing"]) == 5
    assert not (root / "data").exists()


def test_explicit_import_checks_all_bytes_and_is_idempotent(inputs):
    root, source, _ = inputs
    first = import_inputs(root, source)
    assert first["copied"] == 5 and first["ready_for_frozen_replay"]
    assert import_inputs(root, source)["copied"] == 0


def test_wrong_source_is_rejected_before_any_copy(inputs):
    root, source, rows = inputs
    (source / rows[-1]["path"]).write_text("wrong source")
    with pytest.raises(ValueError, match="recorded bytes"):
        import_inputs(root, source)
    assert not (root / "data").exists()


def test_existing_different_data_is_preserved(inputs):
    root, source, rows = inputs
    target = root / rows[0]["path"]
    target.parent.mkdir(parents=True)
    target.write_text("existing different data")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        import_inputs(root, source)
    assert target.read_text() == "existing different data"
