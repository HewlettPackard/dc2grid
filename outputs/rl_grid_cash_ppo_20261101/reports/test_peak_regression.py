from pathlib import Path

import pytest
from peak_regression import ROOT, audit_primary, evaluate, read_json


@pytest.mark.parametrize(
    ("source", "family"),
    (
        ("track_b", "analytic"),
        ("track_b", "economic_ppo"),
        ("track_b_fraction", "potential_sac"),
    ),
)
def test_regression_coordinator_on_training_cell(source, family, tmp_path):
    protocol_path = ROOT / source / "registration.json"
    if family == "analytic":
        policies = read_json(protocol_path)["baseline_policies"]
    else:
        directory = "development_full" if source == "track_b" else "development"
        policies = read_json(ROOT / source / directory / "manifest.json")["policies"]
    policy = next(row for row in policies if row["family"] == family)
    entry = {"id": "training_fixture", "source": source, "policy": policy}
    cell = {"start_day": 16, "seed": 97000040, "regd_load_bias": 0.5}
    plan = {
        "registration_id": "unit-training-cell",
        "role": "training_fixture_not_new_evaluation",
        "source_protocols": {source: {"path": str(protocol_path)}},
        "cells": [cell],
    }
    destination = Path(tmp_path) / "episode"
    result = evaluate(plan, entry, cell, str(destination))
    assert result["complete"]
    assert result["service_fast_steps"] == 17280
    assert result["recovery_fast_steps"] == 1440
    assert result["evaluation_registration_id"] == "unit-training-cell"
    assert not result["potential_used_at_deployment"]
    assert audit_primary(destination, result) > 190
