import json

import numpy as np
import pandas as pd
import pytest
import yaml
from run_matrix import (
    COMPONENTS,
    OUTPUT,
    REWARDS,
    SOURCE,
    pair_metrics,
    run_episode,
    verify_physical_rows,
)


def trace(power, battery=0.0):
    power = np.asarray(power, dtype=float)
    n = len(power)
    data = {
        "timestamp": [f"t{i}" for i in range(n)],
        "grid_load_mw": np.array([100, 200, 300, 400], dtype=float)[:n],
        "regd_signal": np.zeros(n),
        "T_amb": np.full(n, 25.0),
        "lmp": np.full(n, 20.0),
        "load_anomaly": np.linspace(-1, 1, 4)[:n],
        "work_arrived_kwh": np.ones(n),
        "work_served_kwh": np.ones(n),
        "backlog_work_kwh": np.zeros(n),
        "p_base_norm": np.ones(n),
        "p_flex_norm": np.ones(n),
        "is_spike": np.zeros(n),
        "p_facility_mw": power,
        "bess_actual_kw": np.full(n, battery * 1000),
        "p_grid_import_mw": power - battery,
        "p_it_A_mw": power - 20,
        "p_it_B_mw": np.full(n, 5.0),
        "p_hvac_mw": np.full(n, 5.0),
        "p_pump_mw": np.ones(n),
        "p_chiller_mw": np.ones(n),
        "p_ups_loss_mw": np.ones(n),
        "p_pdu_loss_mw": np.ones(n),
        "p_xfmr_loss_mw": np.ones(n),
        "p_aux_mw": np.full(n, 5.0),
        "reward": np.zeros(n),
        **{key: np.zeros(n) for key in REWARDS},
    }
    return pd.DataFrame(data)


def summary(frame, revenue=8.0):
    return {
        "expected_ticks": 4,
        "dt_seconds": 5.0,
        "complete": len(frame) == 4,
        "constraint_qualified": True,
        "stress_threshold_mw": 370.0,
        "annual_load_median_mw": 150.0,
        "committed_max_mw": 30.0,
        "revenue_usd": revenue,
        "electricity_cost_usd": float(
            (frame.lmp * frame.p_grid_import_mw).sum() * 5 / 3600
        ),
        "revenue_before_tracking_discount_usd": revenue + 1,
        "tracking_discount_usd": 1.0,
        "mean_committed_mw": 10.0,
        "terminal": {
            "nominal_stored_mwh": 7.5,
            "backlog_kwh": 0.0,
            "temp_A": 30.0,
            "temp_B": 25.0,
        },
        "battery_throughput_mwh": float(frame.bess_actual_kw.abs().sum() * 5 / 3.6e6),
        "battery_age_fraction_delta": 0.0,
    }


def test_component_and_cashflow_identity():
    a, b = trace([90, 100, 110, 120], 2), trace([100, 110, 120, 130])
    verify_physical_rows(a)
    verify_physical_rows(b)
    result = pair_metrics(a, b, summary(a, 10), summary(b))
    assert result["stress_facility_relief_mw"] == 10
    assert result["stress_pcc_relief_mw"] == 12
    assert result["component_stress_relief_mw"]["it_A"] == 10
    assert result["stress_battery_support_mw"] == 2
    assert result["SPDR_proxy_pct"] == pytest.approx(1000 / 530)
    assert result["electricity_cost_delta_usd"] == pytest.approx(
        -12 * 20 * 4 * 5 / 3600
    )
    assert result["cashflow_delta_before_recovery_usd"] == pytest.approx(
        2 + 12 * 20 * 4 * 5 / 3600
    )
    assert not result["recovery_and_degradation_monetized"]
    candidate_difference = pair_metrics(
        a,
        b,
        summary(a, 10),
        summary(b),
        include_relief_ratios=False,
    )
    assert "SPDR_proxy_pct" not in candidate_difference
    zero = pair_metrics(b, b, summary(b), summary(b))
    assert zero["stress_facility_relief_mw"] == 0
    assert zero["cashflow_delta_before_recovery_usd"] == 0
    assert not zero["rebound_defined"]


def test_stress_average_is_not_actual_peak_reduction():
    a, b = trace([100, 100, 200, 90]), trace([100, 100, 100, 100])
    result = pair_metrics(a, b, summary(a), summary(b))
    assert result["stress_facility_relief_mw"] == 10
    assert result["SPDR_proxy_pct"] == 2
    assert result["actual_system_facility_peak_reduction_mw"] == 0


@pytest.mark.parametrize(
    "key", ["timestamp", "grid_load_mw", "lmp", "work_arrived_kwh", "regd_signal"]
)
def test_pairing_rejects_exogenous_mismatch(key):
    a, b = trace([90, 100, 110, 120]), trace([100, 110, 120, 130])
    a.loc[0, key] = "other" if key == "timestamp" else 999.0
    with pytest.raises(ValueError, match=key):
        pair_metrics(a, b, summary(a), summary(b))


def test_no_common_prefix_score():
    a, b = trace([90, 100]), trace([100, 110, 120, 130])
    result = pair_metrics(a, b, summary(a), summary(b))
    assert not result["pair_complete"]
    assert "stress_facility_relief_mw" not in result
    assert "cashflow_delta_before_recovery_usd" not in result


def test_bias_comparison_requires_same_innovations():
    a, b = trace([90, 100, 110, 120]), trace([100, 110, 120, 130])
    a["regd_signal"] = 0.5 * a.load_anomaly
    result = pair_metrics(a, b, summary(a, 9), summary(b, 10), bias_pair=True)
    assert result["regulation_revenue_change_pct"] == pytest.approx(-10)
    assert "SPDR_proxy_pct" not in result
    a.loc[0, "regd_signal"] += 0.1
    with pytest.raises(AssertionError):
        pair_metrics(a, b, summary(a), summary(b), bias_pair=True)


def test_actual_environment_trace_and_settlement(tmp_path):
    config = yaml.safe_load((SOURCE / "c2g_env/config.yaml").read_text())
    config["global"]["episode_ticks"] = 12
    config_path = tmp_path / "environment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    registration = {
        "registration_id": "smoke-only",
        "inventory": {
            "selection_gate": {
                "min_terminal_soc": 0.2,
                "max_backlog_fraction": 0.01,
                "max_thermal_warning_fraction": 0.01,
            }
        },
    }
    cell = {"id": "smoke", "start_day": 196, "seed": 90001, "regd_load_bias": 0.0}
    policy = {
        "id": "nominal",
        "hardware": "none",
        "macro": "rule_macro",
        "hardware_checkpoint": None,
        "macro_checkpoint": None,
        "prior_selection_qualified": None,
        "diagnostic_only": False,
    }
    path = run_episode(
        policy,
        cell,
        registration,
        config_path=config_path,
        directory=tmp_path / "nominal",
    )
    with open(path) as handle:
        report = json.load(handle)
    hardware = pd.read_csv(
        tmp_path / "nominal/hardware.csv.gz", float_precision="round_trip"
    )
    macro = pd.read_csv(tmp_path / "nominal/macro.csv.gz", float_precision="round_trip")
    assert report["complete"] and report["observed_ticks"] == 12
    assert report["electricity_cost_usd"] == pytest.approx(macro.electricity_cost.sum())
    assert report["revenue_usd"] > 0
    np.testing.assert_allclose(
        hardware.p_reference_mw, hardware.p_facility_mw, atol=1e-8
    )
    assert report["commitment_observation_mae_mw"] < 1e-5
    assert set(COMPONENTS.values()) <= set(hardware.columns)


def test_actual_learned_pair_loads_with_exact_frozen_hardware(tmp_path):
    inventory = json.loads((OUTPUT / "policy_inventory.json").read_text())
    policy = next(p for p in inventory["policies"] if p["id"] == "old_pair_1006")
    cell = next(c for c in inventory["cells"] if c["original_validation_cell"])
    config = yaml.safe_load((SOURCE / "c2g_env/config.yaml").read_text())
    config["global"]["episode_ticks"] = 12
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(config))
    registration = {"registration_id": "smoke-only", "inventory": inventory}
    result_path = run_episode(
        policy,
        cell,
        registration,
        config_path=path,
        directory=tmp_path / "learned_pair",
    )
    with open(result_path) as handle:
        result = json.load(handle)
    assert result["complete"] and result["observed_ticks"] == 12
    assert result["policy"]["matched_rule_id"] == "old_low_1006"
    assert result["commitment_observation_mae_mw"] < 1e-5
    assert np.isfinite(result["tracking_rmse_kw"])
