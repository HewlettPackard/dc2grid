from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import shutil
import sys
from types import SimpleNamespace
from uuid import uuid4

import gymnasium as gym
import numpy as np
import pytest
import yaml

from outputs.table1_corrected_existing_20261102 import table1_eval as te


@pytest.fixture
def scratch():
    directory = te.ROOT / f".evaluator-contract-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def protocol():
    return {
        "registration_id": "contract-tests-only-not-a-production-registration",
        "panel": copy.deepcopy(te.PANEL_SPEC),
        "policies": copy.deepcopy(te.ROW_SPECS),
        "models": copy.deepcopy(te.MODEL_SPECS),
        "config": te.file_receipt(te.PROJECT / "c2g_env/config.yaml"),
        "source_files": {
            path: te.file_hash(te.PROJECT / path) for path in te.runtime_source_paths()
        },
        "data_files": {path: te.file_hash(te.PROJECT / path) for path in te.DATA_PATHS},
        "evaluator_sha256": te.file_hash(te.__file__),
        "recovery_ops": dict(te.RECOVERY_OPS_SPEC),
        "recovery": copy.deepcopy(te.RECOVERY_SPEC),
        "service_gate": dict(te.SERVICE_GATE_SPEC),
    }


def state(age=0.01, soc=0.5):
    return {
        "tick": 0,
        "soc": soc,
        "nominal_stored_mwh": soc * 150.0,
        "usable_energy_property_mwh": soc * 150.0,
        "bess_age_fraction": age,
        "backlog_kwh": 0.0,
        "temp_A": 30.0,
        "temp_B": 30.0,
        "reference_temp_A": 30.0,
        "reference_temp_B": 30.0,
    }


def fast_stub():
    return SimpleNamespace(
        committed_mw=5.0,
        _committed_mw_max=30.0,
        _reference_thermal=SimpleNamespace(temp_A=30.0, temp_B=30.0),
        action_space=gym.spaces.Box(
            np.array([0, 0, 0, -1], dtype=np.float32),
            np.ones(4, dtype=np.float32),
            dtype=np.float32,
        ),
        _bess=SimpleNamespace(P_MAX_MW=50.0),
    )


def info_row(
    error=3.0, reward=1.0, tick=1, temperature=30.0, terminated=False, truncated=False
):
    return {
        "tick": tick,
        "timestamp": f"2023-01-01T00:00:{5 * (tick - 1):02d}",
        "duration_hours": 5 / 3600,
        "p_reference_mw": 100.0,
        "p_grid_import_mw": 99.0,
        "p_facility_mw": 99.0,
        "delta_p_demanded_kw": 1000.0 + error,
        "delta_p_actual_kw": 1000.0,
        "tracking_err_kw": abs(error),
        "bess_actual_kw": 0.0,
        "bess_age_frac": 0.01 + 0.005 * tick,
        "lmp": 10.0,
        "temp_A": temperature,
        "temp_B": 30.0,
        "work_arrived_kwh": 1.0,
        "work_served_kwh": 1.0,
        "backlog_work_kwh": 0.0,
        "reward": reward,
        "regd_signal": 0.3,
        "thermal_warning": temperature > 33.0,
        "thermal_fault": False,
        "freq_fault": False,
        "voltage_fault": False,
        "terminated": terminated,
        "truncated": truncated,
        "freq_dev_hz": 0.01,
        "v_pcc_pu": 1.0,
    }


def collected(info, reward=None, phase="service", fast=None):
    return te.hardware_row(
        np.zeros(18),
        np.array([1.0, 0.7, 0.7, 0.0]),
        np.zeros(18),
        info["reward"] if reward is None else reward,
        info,
        fast or fast_stub(),
        phase,
        1,
    )


def macro_row(substeps=1, reward=2.0, error_sq=9.0, revenue=5.0):
    return {
        "sub_steps_run": substeps,
        "macro_reward": reward,
        "regulation_revenue": revenue,
        "electricity_cost": 10.0 * 99.0 * substeps * 5 / 3600,
        "tracking_error_sq_sum": error_sq,
        "duration_hours": substeps * 5 / 3600,
        "committed_mw": 5.0,
        "bid_accepted": True,
    }


def short_panel(ticks=2, calls=2):
    return {
        **te.PANEL_SPEC,
        "service_seconds": ticks * 5,
        "service_fast_ticks": ticks,
        "service_macro_calls": calls,
        "macro_seconds": ticks // calls * 5,
    }


def test_closed_roster_and_semantic_limits():
    assert len(te.ROW_SPECS) == 7
    assert len(te.ROW_SPECS) * len(te.PANEL_SPEC["seeds"]) == 35
    assert te.ROW_SPECS[-1]["hardware"] == te.ROW_SPECS[-2]["hardware"] == "sac"
    assert te.PANEL_SPEC["start_day_was_in_training"]
    assert not te.PANEL_SPEC["original_producing_run_conditions_proven"]
    assert te.SERVICE_GATE_SPEC["capacity_admission_floor"] is None
    assert te.SERVICE_GATE_SPEC["credit_guard"] is None
    assert te.SERVICE_GATE_SPEC["service_hardware_action_override"] is None
    assert not any("ppo" in row["id"] or "llm" in row["id"] for row in te.ROW_SPECS)


def test_primary_definitions_include_first_tick_and_use_fast_rms():
    totals = te.HardwareTotals()
    totals.add(collected(info_row(error=3, reward=-1, temperature=33.0)))
    totals.add(collected(info_row(error=4, reward=3, tick=2)))
    initial, terminal = state(age=0.02), state(age=0.05)
    summary = te.service_summary(
        totals,
        [macro_row(reward=2), macro_row(reward=8, error_sq=16)],
        initial,
        terminal,
        short_panel(),
        False,
        True,
    )
    metrics = summary["table1_service_metrics"]
    assert summary["complete"]
    assert metrics["tracking_rmse_kw"] == pytest.approx(math.sqrt(12.5))
    assert metrics["tracking_rmse_kw"] != pytest.approx(3.5)
    assert metrics["macro_mean_reward"] == 5.0
    assert metrics["low_mean_reward"] == 1.0
    assert metrics["regulation_revenue_usd"] == 10.0
    assert metrics["thermal_warning_fraction"] == 0.5
    assert metrics["bess_aging_x1e4"] == pytest.approx(300.0)
    assert metrics["survived"] == 1
    assert not summary["supplemental_service_gate"]["qualified"]
    assert metrics["tracking_rmse_kw"] is not None


@pytest.mark.parametrize("terminated,partial", [(True, False), (False, True)])
def test_last_macro_count_cannot_masquerade_as_survival(terminated, partial):
    totals = te.HardwareTotals()
    totals.add(collected(info_row()))
    macros = [macro_row(substeps=1)]
    panel = short_panel(ticks=2 if partial else 1, calls=1)
    result = te.service_summary(
        totals, macros, state(), state(), panel, terminated, True
    )
    assert not result["complete"]
    assert result["table1_service_metrics"]["survived"] == 0
    assert all(
        result["table1_service_metrics"][key] is None for key in te.SERVICE_METRIC_KEYS
    )
    assert result["observed_service_metrics"]["tracking_rmse_kw"] == 3.0


def test_physical_proxy_mismatch_is_not_silently_reported():
    row = collected(info_row())
    row["tracking_err_kw"] = 0.0
    with pytest.raises(AssertionError, match="metered error"):
        te.HardwareTotals().add(row)


def test_supplemental_gate_does_not_require_capacity_or_revenue():
    totals = te.HardwareTotals()
    totals.add(collected(info_row()))
    macro = {**macro_row(), "committed_mw": 0.0, "regulation_revenue": 0.0}
    summary = te.service_summary(
        totals, [macro], state(), state(), short_panel(ticks=1, calls=1), False, True
    )
    assert summary["supplemental_service_gate"]["qualified"]
    assert summary["descriptive_market_metrics_not_gates"]["mean_committed_mw"] == 0.0
    assert summary["table1_service_metrics"]["regulation_revenue_usd"] == 0.0


def test_random_actions_are_case_seeded_without_consuming_environment_rng():
    fast = fast_stub()
    fast.action_space.seed(5)
    before = te.identity(fast.action_space.np_random.bit_generator.state)
    a = te._RandomController(fast.action_space, 100)
    b = te._RandomController(fast.action_space, 100)
    c = te._RandomController(fast.action_space, 101)
    aa = np.array([a.predict(None)[0] for _ in range(10)])
    bb = np.array([b.predict(None)[0] for _ in range(10)])
    cc = np.array([c.predict(None)[0] for _ in range(10)])
    np.testing.assert_array_equal(aa, bb)
    assert not np.array_equal(aa, cc)
    assert te.identity(fast.action_space.np_random.bit_generator.state) == before


def test_classical_controllers_are_fresh_and_none_is_explicit():
    fast = fast_stub()
    first = te._hardware_controller("bang_bang", fast, 100, {})
    first._pump_on = first._hvac_on = True
    second = te._hardware_controller("bang_bang", fast, 101, {})
    assert second is not first
    assert second._pump_on is False and second._hvac_on is False
    pid1 = te._hardware_controller("pid", fast, 100, {})
    pid2 = te._hardware_controller("pid", fast, 101, {})
    assert pid1 is not pid2
    nominal = te._hardware_controller("none", fast, 100, {})
    np.testing.assert_array_equal(
        nominal.predict(None)[0], np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)
    )
    rule = te._hardware_controller("rule_based", fast, 100, {})
    assert rule.bess_gain == 30.0 / 50.0


def test_preflight_validates_closed_contract_and_missing_inputs(scratch):
    contract = protocol()
    checked = te.verify_protocol(contract, te.ROW_SPECS[0], 100)
    assert checked["row"]["id"] == "rule_macro+none"
    for seed in (99, 105, True):
        with pytest.raises(ValueError, match="Seed"):
            te.verify_protocol(contract, te.ROW_SPECS[0], seed)
    changed = copy.deepcopy(contract)
    changed["models"]["low"]["path"] = str(scratch / "final_model.zip")
    with pytest.raises(ValueError, match="selected low"):
        te.verify_protocol(changed, te.ROW_SPECS[0], 100)
    with pytest.raises(ValueError, match="Missing or changed"):
        te._verify_receipt({"path": str(scratch / "absent.csv"), "sha256": "0" * 64})
    changed = copy.deepcopy(contract)
    changed["service_gate"]["capacity_admission_floor"] = 0.9
    with pytest.raises(ValueError, match="supplemental physical gates"):
        te.verify_protocol(changed, te.ROW_SPECS[0], 100)


def test_selected_model_loading_and_shapes_only():
    import torch

    torch.set_num_threads(1)
    for role, observation_dim, action_dim in (("low", 18, 4), ("high", 19, 2)):
        model = te._load_model(role, te.MODEL_SPECS[role])
        assert model.observation_space.shape == (observation_dim,)
        assert model.num_timesteps == te.MODEL_SPECS[role]["selected_timesteps"]
        action, _ = model.predict(
            np.zeros(observation_dim, dtype=np.float32), deterministic=True
        )
        assert action.shape == (action_dim,) and np.isfinite(action).all()
        assert not model.policy.training


def test_native_default_reset_contract_and_inherited_recovery_configuration():
    from c2g_env import C2GMacroEnv

    parent = json.loads(
        (
            te.PROJECT
            / "outputs/physical_recovery_eval_20261031/heldout_registration.json"
        ).read_text()
    )
    assert te.RECOVERY_SPEC == parent["recovery"]
    env = C2GMacroEnv(
        scenario="default",
        config_path=te.PROJECT / "c2g_env/config.yaml",
        simulation_mode="physical_v1",
        start_day=0,
        regd_load_bias=0.0,
    )
    try:
        env.reset(seed=100)
        actual = te._validate_environment(env, {"panel": te.PANEL_SPEC})
        assert actual["bess_backend"] == "_SimpleBESSModel"
        assert actual["electrical_backend"] == "DatacenterElectrical"
        assert env._fast_env._tick == 0
    finally:
        env.close()


def test_native_two_window_collector_and_unchanged_recovery_extension(scratch):
    from baselines.rule_based_macro import RuleBasedMacroController
    from c2g_env import C2GMacroEnv

    config = yaml.safe_load((te.PROJECT / "c2g_env/config.yaml").read_text())
    config["global"]["episode_ticks"] = 360
    config_path = scratch / "short-contract.yaml"
    config_path.write_text(yaml.safe_dump(config))
    path_before = list(sys.path)
    ops = te.load_recovery_ops()
    assert sys.path == path_before
    env = C2GMacroEnv(
        scenario="default",
        config_path=config_path,
        simulation_mode="physical_v1",
        start_day=0,
        regd_load_bias=0.0,
    )
    try:
        observation, _ = env.reset(seed=100)
        fast = env._fast_env
        initial = ops.plant_state(fast)
        totals = te.HardwareTotals()
        rows = []
        controller = te._NominalController()
        env._inner_action_fn = lambda obs, macro: controller.predict(obs)[0]

        def collect(pre, action, post, reward, done, info):
            row = te.hardware_row(pre, action, post, reward, info, fast, "service", 1)
            rows.append(row)
            totals.add(row)

        env._sub_step_callback = collect
        macro = RuleBasedMacroController()
        macros = []
        for _ in range(2):
            observation, reward, terminated, truncated, info = env.step(
                macro.predict(observation)[0]
            )
            macros.append({**info, "macro_reward": reward})
        assert len(rows) == 360 and truncated and not terminated
        panel = short_panel(ticks=360, calls=2)
        summary = te.service_summary(
            totals, macros, initial, ops.plant_state(fast), panel, terminated, truncated
        )
        assert summary["complete"]
        assert totals.error_sq_sum == pytest.approx(
            sum(row["tracking_error_sq_sum"] for row in macros)
        )
        assert totals.electricity_cost_usd == pytest.approx(
            sum(row["electricity_cost"] for row in macros)
        )
        service_snapshot = te.identity(summary)
        extension = ops.begin_recovery(
            fast,
            {"start_day": 0, "regd_load_bias": 0.0, "seed": 100},
            info["last_inner_info"],
            te.RECOVERY_SPEC,
        )
        assert extension["recovery_ticks"] == 1440
        assert (
            extension["state_preserved"]
            and extension["service_replay_prefix_preserved"]
        )
        for _ in range(2):
            action = np.array(
                [1.0, 0.7, 0.7, ops.battery_dispatch(fast._bess, initial["soc"])],
                dtype=np.float32,
            )
            _, _, terminated, truncated, recovery_info = fast.step(action)
            assert recovery_info["delta_p_demanded_kw"] == 0.0
            assert not terminated and not truncated
        assert fast._tick == 362
        assert te.identity(summary) == service_snapshot
    finally:
        env.close()


def fake_case(monkeypatch, *, fail=False, terminal_fault=False):
    import c2g_env

    fast = fast_stub()
    fast.age = 0.01
    fast.current_state = lambda: state(age=fast.age)

    def recovery_step(action):
        fast.age = 0.20
        data = info_row(error=-1000.0, reward=999.0, tick=3, truncated=True)
        data["delta_p_demanded_kw"] = 0.0
        fast.committed_mw = 0.0
        return np.zeros(18), 999.0, False, True, data

    fast.step = recovery_step

    class FakeEnv:
        def __init__(self, **kwargs):
            self._fast_env = fast
            self._last_hw_obs = np.zeros(18)
            self._last_hw_reset_info = {"test_only": True}
            self.closed = False

        def reset(self, seed):
            return np.zeros(19), {"test_only": True}

        def step(self, action):
            for tick, (error, reward) in enumerate(((3.0, 1.0), (4.0, 3.0)), 1):
                data = info_row(
                    error=error,
                    reward=reward,
                    tick=tick,
                    terminated=terminal_fault and tick == 2,
                    truncated=tick == 2,
                )
                if fail:
                    data["freq_fault"] = True
                if terminal_fault and tick == 2:
                    data["thermal_fault"] = True
                requested = self._inner_action_fn(self._last_hw_obs, action)
                self._sub_step_callback(
                    self._last_hw_obs,
                    requested,
                    self._last_hw_obs,
                    reward,
                    tick == 2,
                    data,
                )
                if fail:
                    raise RuntimeError(
                        "intentional contract-test failure after raw tick"
                    )
            fast.age = 0.02
            result = macro_row(substeps=2, reward=7.0, error_sq=25.0)
            result["last_inner_info"] = data
            return np.zeros(19), 7.0, terminal_fault, True, result

        def close(self):
            self.closed = True

    monkeypatch.setattr(c2g_env, "C2GMacroEnv", FakeEnv)
    monkeypatch.setattr(
        te, "verify_protocol", lambda p, policy, seed: {"row": te.ROW_SPECS[0]}
    )
    monkeypatch.setattr(te, "_validate_environment", lambda env, p: {"test_only": True})
    monkeypatch.setattr(te, "_verify_import_origins", lambda p: None)
    monkeypatch.setattr(
        te, "_hardware_controller", lambda *args: te._NominalController()
    )
    monkeypatch.setattr(
        te,
        "load_recovery_ops",
        lambda receipt: SimpleNamespace(
            plant_state=lambda plant: plant.current_state(),
            begin_recovery=lambda *args: {"recovery_ticks": 1},
            battery_dispatch=lambda *args: 0.0,
            closure=lambda *args, **kwargs: {"closed": True},
        ),
    )
    return {
        "registration_id": "fake-contract-only",
        "panel": short_panel(ticks=2, calls=1),
        "recovery_ops": te.RECOVERY_OPS_SPEC,
        "recovery": te.RECOVERY_SPEC,
        "models": te.MODEL_SPECS,
        "config": {"path": "unused-contract-test"},
    }


def test_run_case_keeps_recovery_out_of_service_columns(scratch, monkeypatch):
    contract = fake_case(monkeypatch)
    result = te.run_case(contract, "rule_macro+none", 100, scratch / "case")
    service = result["service"]["table1_service_metrics"]
    assert service["tracking_rmse_kw"] == pytest.approx(math.sqrt(12.5))
    assert service["low_mean_reward"] == 2.0
    assert service["macro_mean_reward"] == 7.0
    assert service["regulation_revenue_usd"] == 5.0
    assert service["bess_aging_x1e4"] == pytest.approx(100.0)
    assert (
        result["supplemental_recovery"]["terminal_state"]["bess_age_fraction"] == 0.20
    )
    assert result["supplemental_recovery"]["observed_ticks"] == 1
    assert result["supplemental_accounting"]["not_part_of_table1_service_columns"]
    for name in (
        "service_hardware.csv.gz",
        "service_macro.csv.gz",
        "recovery_hardware.csv.gz",
    ):
        assert te.file_hash(scratch / "case" / name) == result["trace_hashes"][name]


def test_native_fault_keeps_prefix_without_recovery_or_primary_imputation(
    scratch, monkeypatch
):
    contract = fake_case(monkeypatch, terminal_fault=True)
    result = te.run_case(contract, "rule_macro+none", 100, scratch / "fault")
    assert result["status"] == "service_incomplete"
    assert not result["supplemental_recovery"]["performed"]
    assert result["service"]["observed_service_metrics"]["low_mean_reward"] == 2.0
    assert result["service"]["table1_service_metrics"]["low_mean_reward"] is None
    assert result["service"]["faults"]["counts"]["thermal_fault"] == 1
    assert not (scratch / "fault/recovery_hardware.csv.gz").exists()


def test_exception_retains_partial_trace_faults_and_prevents_retry(
    scratch, monkeypatch
):
    contract = fake_case(monkeypatch, fail=True)
    destination = scratch / "failure"
    with pytest.raises(RuntimeError, match="intentional contract-test"):
        te.run_case(contract, "rule_macro+none", 100, destination)
    failure = json.loads((destination / "failure.json").read_text())
    assert failure["observed_service_ticks"] == 1
    assert failure["observed_macro_calls_returned"] == 0
    assert failure["observed_service_faults"]["counts"]["freq_fault"] == 1
    assert failure["no_retry_or_full_horizon_imputation"]
    with gzip.open(destination / "service_hardware.csv.gz", "rt") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["freq_fault"] == "True"
    digest = te.file_hash(destination / "failure.json")
    with pytest.raises(FileExistsError):
        te.run_case(contract, "rule_macro+none", 100, destination)
    assert te.file_hash(destination / "failure.json") == digest
