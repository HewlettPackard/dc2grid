# isort: off
from study import baseline_protocol
from panel import evaluate_case
from evaluate import run_case
from market_policy import maximum_native_quantity
import numpy as np
import pandas as pd
import pytest
# isort: on


def test_generic_panel_preserves_frozen_baseline_rollout(tmp_path):
    protocol = baseline_protocol()
    protocol["source_hashes"] = {}
    protocol["service_floor"] = 0.9
    policy = {"id": "analytic", "hardware": "cooling_first", "macro": "rule"}
    cell = {"start_day": 16, "regd_load_bias": 0.5, "seed": 730001}
    first = run_case(policy, cell, protocol, tmp_path / "baseline")
    second = evaluate_case(protocol, policy, cell, tmp_path / "new")
    for key in (
        "complete",
        "qualified",
        "service_checks",
        "initial",
        "service_terminal",
        "service",
        "grid",
        "regulation_revenue_usd",
        "cashflow_26h_usd",
    ):
        assert first[key] == second[key], key
    for name in ("hardware.csv.gz", "recovery.csv.gz"):
        a = pd.read_csv(tmp_path / "baseline" / name, float_precision="round_trip")
        b = pd.read_csv(tmp_path / "new" / name, float_precision="round_trip")
        columns = a.columns.intersection(b.columns)
        pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)
    hardware = pd.read_csv(tmp_path / "new" / "hardware.csv.gz")
    macros = pd.read_csv(tmp_path / "new" / "macro.csv.gz")
    expected_means = (
        hardware.observed_regd_signal.to_numpy().reshape(96, 180).mean(axis=1)
    )
    np.testing.assert_allclose(
        macros.signed_regd_mean, expected_means, atol=1e-12, rtol=0
    )


def test_maximum_quantity_rounds_inward():
    for residual in np.linspace(5.1, 40, 1000):
        assert 5 <= maximum_native_quantity(residual) <= min(30, residual)
    with pytest.raises(ValueError):
        maximum_native_quantity(float("nan"))


def test_final_case_cannot_instantiate_a_policy_without_a_lock(tmp_path, monkeypatch):
    import panel

    monkeypatch.setattr(panel, "ROOT", tmp_path)
    cell = {"start_day": 12, "seed": 94001, "regd_load_bias": 0.5}
    with pytest.raises(RuntimeError, match="remain sealed"):
        evaluate_case({"final_cells": [cell]}, {}, cell, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
