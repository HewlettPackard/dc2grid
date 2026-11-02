"""Focused tests for physical continuation, accounting and selection guards."""

import copy
import json
from pathlib import Path

# Keep the frozen-source bootstrap ahead of simulator imports.
# isort: off
from recovery import (
    MATCHED,
    RECOVERY,
    SOURCE,
    battery_dispatch,
    begin_recovery,
    closure,
    continuation_identity,
    matched,
    plant_state,
    read_frame,
    recovery_pair,
    run_episode,
    verify_service_replay,
)
from run_evaluation import mean_sd, policy_inventory

import numpy as np
import pytest
import yaml

from c2g_env import C2GMacroEnv
from c2g_env.physics.bess import _SimpleBESSModel
# isort: on


def configuration(tmp_path, ticks=12):
    config = yaml.safe_load((SOURCE / "c2g_env/config.yaml").read_text())
    config["global"]["episode_ticks"] = ticks
    for key in ("trace_dir", "energy_dir", "weather_dir"):
        config["global"][key] = str((SOURCE / config["global"][key]).resolve())
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("soc", [0.1, 0.2, 0.499999, 0.500001, 0.8, 0.95])
def test_battery_recovery_is_bounded_and_does_not_reset_age(soc):
    battery = _SimpleBESSModel(dt_seconds=5)
    battery.set_initial_soc(soc)
    battery._age_frac = 0.01
    energy = 0.0
    for _ in range(1440):
        before = vars(battery).copy()
        command = battery_dispatch(battery, 0.5)
        assert vars(battery) == before
        assert -1 <= command <= 1
        output = battery.step(command * battery.P_MAX_MW)
        energy += abs(output["actual_power_mw"]) * 5 / 3600
    assert battery.soc_fraction == pytest.approx(0.5, abs=RECOVERY["soc_tolerance"])
    assert battery._age_frac == pytest.approx(0.01 + energy / (15 * 1000), abs=1e-12)


def test_continuation_preserves_state_and_original_bias_normalization(tmp_path):
    env = C2GMacroEnv(
        scenario="default",
        simulation_mode="physical_v1",
        config_path=configuration(tmp_path),
        start_day=196,
        regd_load_bias=0.5,
        inner_action_fn=lambda obs, macro: np.array(
            [0.0, 0.7, 0.7, 1.0], dtype=np.float32
        ),
    )
    cell = {"start_day": 196, "seed": 90001, "regd_load_bias": 0.5}
    try:
        env.reset(seed=cell["seed"])
        initial = plant_state(env._fast_env)
        workload = env._fast_env._workload
        index = (workload._tick // workload._steps_per_trace_tick) % workload._n
        workload._p_flex_arr[index] = workload.p_flex_idle_kw + 1000
        _, _, term, trunc, info = env.step(np.array([0.5, 0.1], dtype=np.float32))
        assert trunc and not term
        fast = env._fast_env
        service_state = plant_state(fast)
        assert service_state["soc"] < initial["soc"]
        assert service_state["backlog_kwh"] > 0
        anomaly = fast._grid._load_anomaly.copy()
        identity = continuation_identity(fast)
        extension = begin_recovery(fast, cell, info["last_inner_info"], RECOVERY)
        assert extension["recovery_ticks"] == 1440
        assert continuation_identity(fast) == identity
        np.testing.assert_array_equal(fast._grid._load_anomaly[:13], anomaly)
        assert plant_state(fast) == service_state
        assert fast.committed_mw == 0
        _, _, term, trunc, next_info = fast.step(
            np.array([1, 0.7, 0.7, 0], dtype=np.float32)
        )
        assert next_info["tick"] == 13 and not term and not trunc
        assert next_info["delta_p_demanded_kw"] == 0
        assert np.datetime64(next_info["timestamp"]) == np.datetime64(
            "2023-07-16T00:01:00"
        )
    finally:
        env.close()


def test_recovery_never_revives_a_physical_fault(tmp_path):
    env = C2GMacroEnv(
        config_path=configuration(tmp_path), simulation_mode="physical_v1"
    )
    try:
        env.reset(seed=100)
        with pytest.raises(ValueError, match="healthy"):
            begin_recovery(
                env._fast_env, {}, {"truncated": False, "terminated": True}, RECOVERY
            )
    finally:
        env.close()


@pytest.mark.parametrize("policy_id", ["nominal", "old_pair_1006"])
def test_full_recovery_and_exact_short_service_parity(tmp_path, policy_id):
    inventory = json.loads((MATCHED / "policy_inventory.json").read_text())
    policy = next(p for p in inventory["policies"] if p["id"] == policy_id)
    cell = inventory["cells"][0]
    config = configuration(tmp_path)
    old_directory = tmp_path / "old"
    matched.run_episode(
        policy,
        cell,
        {"registration_id": "test", "inventory": inventory},
        config_path=config,
        directory=old_directory,
    )
    registration = {
        "stage": "test",
        "registration_id": "test",
        "config_path": str(config),
        "recovery": RECOVERY,
        "selection_gate": inventory["selection_gate"],
    }
    directory = tmp_path / "new"
    path = run_episode(policy, cell, registration, directory=directory)
    result = json.loads(Path(path).read_text())
    assert result["complete"] and result["observed_ticks"] == 12
    assert result["recovery"]["complete"] and result["recovery"]["closed"]
    assert result["recovery"]["observed_ticks"] == 1440
    assert result["recovery"]["regulation_revenue_usd"] == 0
    assert abs(result["recovery"]["work_conservation_error_kwh"]) < 1e-4
    service = read_frame(directory / "hardware.csv.gz")
    macros = read_frame(directory / "macro.csv.gz")
    # CSV input loses the original float32 dtype of one macro column.
    macros["bess_soc_end"] = macros.bess_soc_end.astype(np.float32)
    assert verify_service_replay(service, macros, old_directory)["verified"]
    recovery = read_frame(directory / "recovery.csv.gz")
    assert (recovery.committed_actual_mw == 0).all()
    assert (recovery.delta_p_demanded_kw == 0).all()
    assert result["recovery"]["observed_energy_metrics"][
        "electricity_cost_usd"
    ] == pytest.approx(
        (recovery.lmp * recovery.p_grid_import_mw * recovery.duration_hours).sum()
    )


def test_inventory_retains_all_seed_slots_and_excludes_failed_macro():
    policies, _, cells = policy_inventory()
    assert len(policies) == 20 and len(cells) == 8
    assert sum(p["available"] for p in policies) == 16
    failed = next(p for p in policies if p["id"] == "old_pair_1002")
    assert failed["checkpoint_available"] and not failed["available"]
    assert failed["matched_validation_qualified_cells"] == 6
    assert len([p for p in policies if p["group"] == "old_low"]) == 8
    assert len([p for p in policies if p["group"] == "old_pair"]) == 8
    identifiers = {policy["id"] for policy in policies}
    assert all(
        policy["matched_rule_id"] in identifiers
        for policy in policies
        if policy["macro"] == "sac_macro"
    )


def test_cohort_summary_cannot_drop_failures():
    assert mean_sd([{"value": 1}, {"value": None}], "value", 2) == (None, None)
    assert mean_sd([{"value": 1}], "value", 2) == (None, None)
    assert mean_sd([{"value": 1}, {"value": 3}], "value", 2)[0] == 2
    with pytest.raises(ValueError, match="Non-finite"):
        mean_sd([{"value": float("nan")}], "value", 1)


def test_closure_requires_full_horizon_and_retains_usable_energy_difference():
    state = {
        "soc": 0.5,
        "nominal_stored_mwh": 7.5,
        "usable_energy_property_mwh": 7.49,
        "backlog_kwh": 0,
        "temp_A": 29,
        "temp_B": 25,
        "reference_temp_A": 29,
        "reference_temp_B": 25,
    }
    initial = {**state, "usable_energy_property_mwh": 7.5}
    assert not closure(state, initial, RECOVERY, complete=False)["closed"]
    result = closure(state, initial, RECOVERY, complete=True)
    assert result["closed"]
    assert result["usable_energy_property_change_mwh"] == pytest.approx(-0.01)
    state["temp_B"] = 25.2
    assert not closure(state, initial, RECOVERY, complete=True)["closed"]


def test_recovery_cashflow_and_wear_signs_are_explicit():
    def episode(revenue, cost, recovery_cost, throughput):
        return {
            "complete": True,
            "overall_qualified": True,
            "revenue_usd": revenue,
            "recovery": {
                "complete": True,
                "closed": True,
                "observed_energy_metrics": {"electricity_cost_usd": recovery_cost},
                "total_observed_energy_metrics": {
                    "electricity_cost_usd": cost,
                    "battery_throughput_mwh": throughput,
                    "backlog_exposure_kwh_hours": 0,
                    "work_served_kwh": 10,
                },
                "total_battery_age_fraction_delta": throughput / 15000,
                "terminal": {"usable_energy_property_mwh": 7.5},
            },
        }

    a, b = episode(110, 95, 5, 4), episode(100, 100, 10, 6)
    result = recovery_pair(a, b, RECOVERY)
    assert result["cashflow_delta_after_closed_recovery_usd"] == 15
    assert result["recovery_electricity_cost_delta_usd"] == -5
    assert result["wear_adjusted_cashflow_usd"]["10.0"] == 35
    assert result["wear_sensitivity_slope_mwh"] == 2
    assert result["wear_break_even_usd_per_mwh"] is None
    unclosed = copy.deepcopy(a)
    unclosed["recovery"]["closed"] = False
    result = recovery_pair(unclosed, b, RECOVERY)
    assert result["cashflow_delta_after_fixed_recovery_usd"] == 15
    assert result["cashflow_delta_after_closed_recovery_usd"] is None
    assert all(value is None for value in result["wear_adjusted_cashflow_usd"].values())


def test_end_to_end_reports_retain_unavailable_slots(tmp_path, monkeypatch):
    import run_evaluation

    policies, gate, cells = policy_inventory()
    chosen = {"nominal", "pid", "old_low_1000", "old_low_1006", "old_pair_1006"}
    policies = [policy for policy in policies if policy["id"] in chosen]
    registration = {
        "stage": "test",
        "registration_id": "test",
        "config_path": str(configuration(tmp_path)),
        "recovery": RECOVERY,
        "selection_gate": gate,
        "policies": policies,
        "cells": cells[:2],
    }
    monkeypatch.setattr(run_evaluation, "ROOT", tmp_path)
    for policy in policies:
        if policy["available"]:
            for cell in registration["cells"]:
                run_episode(
                    policy,
                    cell,
                    registration,
                    directory=tmp_path / "test/traces" / policy["id"] / cell["id"],
                )
    readback = run_evaluation.summarize(registration)
    assert readback["planned_policy_cells"] == 10
    assert readback["recorded_rollouts"] == 8
    assert readback["unavailable_policy_cells"] == 2
    assert readback["recovery_closed"] == 8
    assert readback["comparison_records"] == 21
    assert readback["complete_service_pairs"] == 16
    assert readback["closed_recovery_pairs"] == 16
    summaries = json.loads((tmp_path / "test/reports/policy_summary.json").read_text())
    unavailable = next(row for row in summaries if row["policy_id"] == "old_low_1000")
    assert unavailable["planned_cells"] == 2
    assert unavailable["rmse_kw"] is None
    assert unavailable["cash_vs_nominal_after_closed_recovery_usd"] is None


def test_recovery_fault_stays_incomplete_even_with_zero_initial_debt(
    tmp_path, monkeypatch
):
    from c2g_env.physics.weather import WeatherLoader

    policies, gate, cells = policy_inventory()
    nominal = next(policy for policy in policies if policy["id"] == "nominal")
    original_temperature = WeatherLoader.temp_c

    def temperature_shock(self, tick):
        return original_temperature(self, tick) if tick < 12 else 10_000.0

    monkeypatch.setattr(WeatherLoader, "temp_c", temperature_shock)
    registration = {
        "stage": "test",
        "registration_id": "fault-injection-test",
        "config_path": str(configuration(tmp_path)),
        "recovery": RECOVERY,
        "selection_gate": gate,
    }
    result = json.loads(
        Path(
            run_episode(nominal, cells[0], registration, directory=tmp_path / "fault")
        ).read_text()
    )
    assert result["complete"]
    assert result["recovery"]["performed"]
    assert result["recovery"]["terminated"]
    assert result["recovery"]["observed_ticks"] < 1440
    assert not result["recovery"]["complete"]
    assert not result["recovery"]["closed"]
    assert not result["overall_qualified"]


def test_recovery_prices_use_the_declared_weighting():
    import pandas as pd
    from recovery import energy_metrics

    frame = pd.DataFrame(
        {
            "duration_hours": [1.0, 1.0],
            "lmp": [10.0, 100.0],
            "p_grid_import_mw": [10.0, 20.0],
            "p_facility_mw": [9.0, 17.0],
            "bess_actual_kw": [-1000.0, -3000.0],
            "backlog_work_kwh": [0.0, 0.0],
            "work_arrived_kwh": [1.0, 1.0],
            "work_served_kwh": [1.0, 1.0],
            "thermal_warning": [False, False],
        }
    )
    result = energy_metrics(frame)
    assert result["electricity_cost_usd"] == 2100
    assert result["lmp_time_weighted_mean_usd_mwh"] == 55
    assert result["battery_charge_energy_weighted_price_usd_mwh"] == 77.5
    assert result["battery_discharge_energy_weighted_price_usd_mwh"] is None
