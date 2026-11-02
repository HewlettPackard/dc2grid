import finish_value
import pytest
from ctx import file_hash, identity, write_json


def test_final_admission_never_drops_a_failing_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_value, "ROOT", tmp_path)
    protocol = {
        "variants": ["union_original", "union_conditioned"],
        "seeds": [1100, 1101, 1102, 1103],
        "registration_id": "test",
        "source_hashes": {},
    }
    write_json(tmp_path / "extension_decision.json", {"extend_both_arms": True})
    jobs = []
    for variant in protocol["variants"]:
        for seed in protocol["seeds"]:
            directory = tmp_path / f"runs/extended/{variant}_{seed}"
            directory.mkdir(parents=True)
            checkpoint = directory / "step_552960.zip"
            checkpoint.write_bytes(b"hash-only fixture; never loaded for inference")
            write_json(
                directory / "completed.json",
                {
                    "status": "completed",
                    "stage": "extended",
                    "seed": seed,
                    "variant": variant,
                    "registration_id": "test",
                    "source_hashes": {},
                    "online_transitions": 552960,
                    "final": {
                        "timesteps": 552960,
                        "extension_qualified": not (
                            variant == "union_original" and seed == 1102
                        ),
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": file_hash(checkpoint),
                        },
                    },
                },
            )
            jobs.append({"variant": variant, "seed": seed, "status": "completed"})
    write_json(tmp_path / "extended_status.json", {"state": "finished", "jobs": jobs})
    slots = finish_value.primary_inventory(protocol)
    assert (
        sum(slot["available"] for slot in slots if slot["family"] == "union_original")
        == 0
    )
    assert (
        sum(
            slot["available"] for slot in slots if slot["family"] == "union_conditioned"
        )
        == 4
    )


def test_missing_regression_case_cannot_open_the_final_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_value, "ROOT", tmp_path)
    monkeypatch.setattr(finish_value, "PRIOR", tmp_path / "prior")
    policy = {
        "id": "rule__union_original_1100",
        "family": "union_original",
        "macro": "rule",
        "available": True,
    }
    lock = {
        "registration_id": "test",
        "inventory": [policy],
        "producer_sha256": file_hash(finish_value.__file__),
    }
    lock["candidate_lock_id"] = identity(lock)
    write_json(tmp_path / "candidate_lock.json", lock)
    write_json(
        tmp_path / "peak_regression/summary.json",
        {
            "episodes": [],
            "runtime_failures": [],
        },
    )
    write_json(
        tmp_path / "regression_registration.json",
        {
            "regression_cells": [
                {"start_day": 207, "seed": 100, "regd_load_bias": 0.5}
            ],
        },
    )
    with pytest.raises(ValueError, match="lost a locked policy"):
        finish_value.open_final(
            {"registration_id": "test", "variants": ["union_original"]}, []
        )
    assert not (tmp_path / "prior/final_lock.json").exists()
    assert not (tmp_path / "final_lock.json").exists()
