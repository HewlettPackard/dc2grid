"""Focused training-cell contracts; no fitting, development or final panel."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

# isort: off
from economy_macro import (
    CREDIT_RESERVE_MWH,
    DT_HOURS,
    EVALUATION_SEEDS,
    EXTERNAL_SERVICE_FLOOR,
    GUARD_SERVICE_TARGET,
    NOMINAL,
    OBSERVATION_SCHEMA,
    REQUIRED_GAMMA,
    ROOT,
    CapacityErrorBudget,
    EconomicDispatchController,
    EconomicDispatchTask,
    excess_cash_bound,
    excess_transition,
    requested_extra_response,
    service_capacity_reference,
)
from economy_context import PRIOR
from economy_eval import ConstantResidual, evaluate_episode
from controllers import CoolingFirstController, FullResidualMacro, ResidualController
from market_policy import maximum_native_quantity
from c2g_env import C2GMacroEnv
from c2g_env.obs_indices import Fast, Macro

import numpy as np
import pandas as pd
import pytest
# isort: on


@pytest.fixture(scope="module")
def protocol():
    original = json.loads((PRIOR / "registration.json").read_text())
    keys = (
        "config_path",
        "train_days",
        "train_biases",
        "n_envs",
        "normalization",
        "service_gate",
        "recovery",
        "service_floor",
        "registration_id",
        "source_hashes",
    )
    return {key: copy.deepcopy(original[key]) for key in keys}


@pytest.fixture(scope="module")
def training_cell(protocol):
    return {
        "start_day": protocol["train_days"][0],
        "seed": 97000001,
        "regd_load_bias": 0.5,
    }


@pytest.fixture(scope="module")
def diagnostic_directory():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run these tests with track_b as the working directory")
    directory = Path(".validation") / f"focused-{os.getpid()}"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        if not any(directory.parent.iterdir()):
            directory.parent.rmdir()


def finish(task, action):
    rewards, flags, infos = [], [], []
    for _ in range(96):
        observation, reward, terminated, truncated, info = task.step(action)
        assert task.observation_space.contains(observation)
        rewards.append(reward)
        flags.append((terminated, truncated))
        infos.append(info)
        if terminated or truncated:
            break
    return rewards, flags, infos


@pytest.fixture(scope="module")
def zero_episode(protocol, training_cell, diagnostic_directory):
    policy = {
        "id": "training_smoke_only",
        "family": "cooling_first",
        "hardware": "cooling_first",
        "macro": "full_residual",
    }
    frozen_dir = diagnostic_directory / "frozen_zero_reference"
    frozen = evaluate_episode(protocol, policy, training_cell, frozen_dir)
    task = EconomicDispatchTask(
        protocol, seed=4200, fixed_cell=training_cell, record_trace=True
    )
    try:
        task.reset()
        rewards, flags, infos = finish(task, [0.0])
        yield task, frozen, frozen_dir, rewards, flags, infos
    finally:
        task.close()


@pytest.fixture(scope="module")
def budgeted_episode(protocol, training_cell):
    task = EconomicDispatchTask(
        protocol, seed=4200, fixed_cell=training_cell, record_trace=True
    )
    try:
        task.reset()
        rewards, flags, infos = finish(task, [1.0])
        yield task, rewards, flags, infos
    finally:
        task.close()


def test_zero_overlay_exact_full_26h_parity(zero_episode):
    task, frozen, frozen_dir, _, _, _ = zero_episode
    current = task.episode_result
    assert current["complete"] and current["qualified"]
    assert frozen["complete"] and frozen["qualified"]
    for key in (
        "initial",
        "service_terminal",
        "service",
        "grid",
        "regulation_revenue_usd",
        "cashflow_26h_usd",
        "cash_vs_nominal_26h_usd",
        "maximum_reference_mw_hours",
        "performance_weighted_mw_hours",
        "committed_mw_hours",
        "delivered_service_ratio",
    ):
        assert current[key] == frozen[key], key
    for key, value in frozen["recovery"].items():
        assert current["recovery"][key] == value, key
    for name, rows in (
        ("hardware.csv.gz", task.service_rows),
        ("macro.csv.gz", task.macro_rows),
        ("recovery.csv.gz", task.recovery_rows),
    ):
        reference = pd.read_csv(frozen_dir / name)
        actual = pd.DataFrame(rows)
        pd.testing.assert_frame_equal(
            actual[reference.columns].reset_index(drop=True),
            reference,
            check_exact=False,
            check_dtype=False,
            atol=1e-10,
            rtol=1e-13,
        )
    assert task.service.steps == 17280
    assert task.recovery.steps == 1440
    assert task.guard_binding_ticks == 0
    assert all(row["guarded_extra_response_mw"] == 0 for row in task.service_rows)


def test_requested_one_mw_matches_constant_probe_without_binding(
    protocol, training_cell
):
    rows = []
    probe = ResidualController(ConstantResidual([0, 0, 1]))
    core = C2GMacroEnv(
        scenario="default",
        simulation_mode="physical_v1",
        config_path=protocol["config_path"],
        start_day=training_cell["start_day"],
        regd_load_bias=training_cell["regd_load_bias"],
        inner_action_fn=lambda observation, _: probe.predict(observation)[0],
        sub_step_callback=lambda pre, action, post, reward, done, info: rows.append(
            (action.copy(), post.copy(), dict(info))
        ),
    )
    task = EconomicDispatchTask(
        protocol, seed=1, fixed_cell=training_cell, record_trace=True
    )
    try:
        original_observation, _ = core.reset(seed=training_cell["seed"])
        task.reset()
        original = core.step(FullResidualMacro().predict(original_observation)[0])
        current = task.step([1 / 6])
        assert not original[2] and not original[3]
        assert not current[2] and not current[3]
        assert task.guard_binding_ticks == 0
        assert len(rows) == len(task.service_rows) == 180
        np.testing.assert_array_equal(original[0], current[0][:19])
        for (_, _, old), new in zip(rows, task.service_rows, strict=True):
            assert new["desired_extra_response_mw"] == 1
            assert new["guarded_extra_response_mw"] == 1
            for key, value in old.items():
                assert new[key] == value, key
        assert current[4]["bid_mw"] == original[4]["bid_mw"]
        assert task.revenue == original[4]["regulation_revenue"]
    finally:
        task.close()
        core._fast_env.close()
        core.close()


def test_scoped_controller_matches_frozen_algebra_in_thermal_compensation(
    protocol, training_cell
):
    task = EconomicDispatchTask(protocol, seed=1, fixed_cell=training_cell)
    try:
        task.reset()
        observation = task.core._last_hw_obs.copy()
        observation[Fast.TEMP_A] = 33.2 / 35
        observation[Fast.TEMP_B] = 32.4 / 35
        observation[Fast.SOC] = 0.55
        for extra in (0.0, 0.4, 1.0):
            frozen = CoolingFirstController()
            overlay = EconomicDispatchController()
            frozen.features(observation)
            overlay.features(observation)
            residual = np.array([0, 0, extra], dtype=np.float32)
            expected = frozen.action(residual)
            actual = overlay.action(float(residual[2]))
            np.testing.assert_array_equal(actual, expected)
            assert overlay.last_diagnostics["soc_target"] == 0.5
            assert actual[1] > NOMINAL
        overlay = EconomicDispatchController()
        overlay.features(observation)
        actual = overlay.action(6.0)
        assert actual[0] == 1
        assert overlay.last_diagnostics["extra_response_mw"] == 6
        assert task.fast.action_space.contains(actual)
    finally:
        task.close()


@pytest.mark.parametrize("fixture_name", ("zero_episode", "budgeted_episode"))
def test_exact_reward_cash_reference_identity(request, fixture_name):
    data = request.getfixturevalue(fixture_name)
    task = data[0]
    rewards = data[3] if fixture_name == "zero_episode" else data[1]
    episode = task.episode_result
    assert episode["qualified"]
    assert REQUIRED_GAMMA == task.required_gamma == 1.0
    expected = (
        episode["cashflow_26h_usd"] - episode["exogenous_reference_cash_usd"]
    ) / 100
    assert sum(rewards) == pytest.approx(expected, abs=1e-8)
    assert episode["reward_sum"] == pytest.approx(expected, abs=1e-8)
    service = pd.DataFrame(task.service_rows)
    recovery = pd.DataFrame(task.recovery_rows)
    macros = pd.DataFrame(task.macro_rows)
    revenue = macros.regulation_revenue.sum()
    ideal = np.dot(macros.rate * macros.committed_mw, macros.duration_hours)
    demand = np.dot(
        service.lmp * service.delta_p_demanded_kw / 1000, service.duration_hours
    )
    electricity = sum(
        np.dot(frame.lmp * frame.p_grid_import_mw, frame.duration_hours)
        for frame in (service, recovery)
    )
    nominal = sum(
        np.dot(frame.lmp * frame.p_reference_mw, frame.duration_hours)
        for frame in (service, recovery)
    )
    assert episode["cashflow_26h_usd"] == pytest.approx(revenue - electricity, abs=1e-6)
    assert episode["exogenous_reference_cash_usd"] == pytest.approx(
        ideal + demand - nominal, abs=1e-6
    )
    assert episode["ideal_regulation_reference_usd"] == pytest.approx(ideal)
    assert episode["contractual_demand_electricity_usd"] == pytest.approx(demand)
    totals = episode["reward_components"]
    assert totals["invalid_adjustment_usd"] == 0
    assert totals["excess_cash_usd"] == pytest.approx(
        totals["actual_regulation_revenue_usd"]
        - totals["ideal_regulation_reference_usd"]
        + totals["service_centered_electricity_usd"]
        - totals["contractual_demand_electricity_usd"]
        + totals["recovery_centered_electricity_usd"],
        abs=1e-8,
    )
    assert macros.recovery_centered_electricity_usd.iloc[:-1].eq(0).all()
    assert macros.recovery_centered_electricity_usd.iloc[-1] == pytest.approx(
        episode["recovery"]["ledger"]["nominal_electricity_usd"]
        - episode["recovery"]["ledger"]["electricity_usd"]
    )


def test_true_terminal_and_full_fixed_recovery(zero_episode):
    task, _, _, _, flags, infos = zero_episode
    assert flags == [(False, False)] * 95 + [(True, False)]
    assert task.episode_result is task.last_episode
    episode = task.episode_result
    assert episode["true_terminal"] and not episode["timeout_bootstrap"]
    assert not episode["physical_terminated"]
    assert episode["raw_service_truncated"]
    assert infos[-1]["recovery_fast_steps"] == 1440
    assert infos[-1]["episode_result"] is episode
    assert "TimeLimit.truncated" not in infos[-1]
    assert task.fast._tick == 17280 + 1440
    assert episode["recovery"]["extension"]["state_preserved"]
    assert episode["recovery"]["extension"]["service_replay_prefix_preserved"]
    for row in task.recovery_rows:
        assert row["throttle_batch"] == 1
        assert row["pump_speed_A"] == NOMINAL
        assert row["hvac_effort"] == NOMINAL
        assert row["committed_mw"] == row["delta_p_demanded_kw"] == 0
        assert -1 <= row["bess_dispatch"] <= 1
        assert row["p_hvac_mw"] == pytest.approx(50 * NOMINAL)
    assert episode["recovery"]["regulation_revenue_usd"] == 0
    with pytest.raises(RuntimeError, match="Reset"):
        task.step([0])


def test_reference_is_action_independent_and_contracts_never_mutated(
    zero_episode, budgeted_episode
):
    zero = zero_episode[0]
    budgeted = budgeted_episode[0]
    for key in ("ideal_regulation_reference_usd", "exogenous_reference_cash_usd"):
        assert zero.episode_result[key] == budgeted.episode_result[key]
    assert budgeted.guard_binding_ticks > 0
    assert budgeted.episode_result["delivered_service_ratio"] >= 0.90
    zero_rows = pd.DataFrame(zero.service_rows)
    extra_rows = pd.DataFrame(budgeted.service_rows)
    for name in (
        "delta_p_demanded_kw",
        "regd_signal",
        "lmp",
        "grid_load_mw",
        "committed_mw",
        "p_reference_mw",
        "work_arrived_kwh",
        "work_served_kwh",
    ):
        np.testing.assert_array_equal(zero_rows[name], extra_rows[name])
    np.testing.assert_allclose(
        extra_rows.delta_p_demanded_kw,
        extra_rows.committed_mw * 1000 * extra_rows.regd_signal,
        atol=1e-9,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        extra_rows.tracking_err_kw,
        abs(extra_rows.delta_p_demanded_kw - extra_rows.delta_p_actual_kw),
        atol=1e-9,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        extra_rows.p_reference_mw - extra_rows.p_grid_import_mw,
        extra_rows.delta_p_actual_kw / 1000,
        atol=1e-9,
        rtol=1e-12,
    )
    assert extra_rows.soc_target.eq(0.5).all()
    assert extra_rows.throttle_batch.eq(1).all()


def test_credit_audit_replays_every_tick_without_future_data(budgeted_episode):
    task = budgeted_episode[0]
    credit = 0.0
    for row in task.service_rows:
        remaining = (180 - (row["tick"] - 1) % 180) * DT_HOURS
        assert row["remaining_window_hours"] == remaining
        cap = min(
            6.0,
            max(
                0.0,
                (1 - 0.92) * row["committed_mw"] + (credit - 0.02) / remaining,
            ),
        )
        assert row["credit_before_mwh"] == credit
        assert row["guard_cap_mw"] == cap
        assert row["guarded_extra_response_mw"] == min(
            row["desired_extra_response_mw"], cap
        )
        credit += (
            (1 - 0.92) * row["committed_mw"] - abs(row["tracking_err_kw"]) / 1000
        ) * DT_HOURS
        assert row["credit_after_mwh"] == credit
    assert task.budget.credit_mwh == credit
    assert task.budget.steps == 17280
    assert task.episode_result["credit"]["final_mwh"] == credit
    assert not task.episode_result["credit"]["absolute_service_guarantee_claimed"]
    macros = pd.DataFrame(task.macro_rows)
    denominator = sum(
        service_capacity_reference(row) * row["duration_hours"]
        for row in task.macro_rows
    )
    numerator = np.dot(macros.committed_mw * macros.perf_score, macros.duration_hours)
    assert task.episode_result["maximum_reference_mw_hours"] == denominator
    assert task.episode_result["delivered_service_ratio"] == pytest.approx(
        numerator / denominator
    )


def test_budget_uses_realized_absolute_error_current_window_and_no_clipping():
    budget = CapacityErrorBudget()
    assert GUARD_SERVICE_TARGET == 0.92
    assert CREDIT_RESERVE_MWH == 0.02
    assert EXTERNAL_SERVICE_FLOOR == 0.90
    assert budget.cap(5.0, 0.25) == pytest.approx(0.32)
    assert budget.cap(30.0, 0.25) == pytest.approx(2.32)
    assert budget.cap(5.0, DT_HOURS) == 0
    budget.update(5.0, -1000.0, DT_HOURS)
    assert budget.credit_mwh == pytest.approx((0.4 - 1.0) * DT_HOURS)
    assert budget.minimum_credit_mwh < 0
    before = budget.credit_mwh
    budget.update(30.0, 0.0, DT_HOURS)
    assert budget.credit_mwh == pytest.approx(before + 2.4 * DT_HOURS)
    budget.credit_mwh = 100.0
    assert budget.cap(5, DT_HOURS) == 6
    for capacity, remaining in ((0, 0.25), (31, 0.25), (5, 24), (5, 0)):
        with pytest.raises(ValueError):
            budget.cap(capacity, remaining)
    with pytest.raises(ValueError):
        budget.update(5, float("nan"), DT_HOURS)
    with pytest.raises(ValueError):
        budget.update(5, 1000, 0.25)


def test_rejected_window_denominator_and_true_native_rounding(zero_episode):
    task = zero_episode[0]
    rejected = [row for row in task.macro_rows if not row["bid_accepted"]]
    assert rejected, "This fixed training smoke must exercise natural rejection"
    for row in rejected:
        assert row["committed_mw"] == 5
        assert row["rate"] == 5
        assert row["reference_capacity_mw"] == 5
        assert row["ideal_regulation_reference_usd"] == 5 * 5 * 0.25
        assert row["desired_extra_response_mw"] == 0
    for row in task.macro_rows:
        maximum = maximum_native_quantity(row["residual_mw"])
        assert maximum <= min(30, row["residual_mw"])
        assert row["reference_capacity_mw"] == (maximum if row["bid_accepted"] else 5)
    synthetic = {"residual_mw": 29.999999, "bid_accepted": False, "committed_mw": 5}
    assert service_capacity_reference(synthetic) == 5
    synthetic["bid_accepted"] = True
    assert service_capacity_reference(synthetic) == maximum_native_quantity(29.999999)
    synthetic.update(bid_accepted=False, committed_mw=0)
    with pytest.raises(ValueError, match="5MW"):
        service_capacity_reference(synthetic)
    assert requested_extra_response([-1]) == requested_extra_response([0]) == 0


def test_full_excess_bound_and_finite_fault_clawback(protocol):
    bound = excess_cash_bound(protocol["normalization"])
    assert bound["centered_cash_bound_usd"] == 2824000
    assert bound["ideal_regulation_bound_usd"] == 360000
    assert bound["signed_demand_electricity_bound_usd"] == pytest.approx(85262.461776)
    assert bound["excess_cash_prefix_bound_usd"] == 3270000
    assert bound["invalid_episode_return"] == -65400
    assert bound["valid_return_absolute_bound"] == 32700
    assert bound["invalid_episode_return"] < -bound["valid_return_absolute_bound"]
    enclosure = bound["excess_cash_prefix_bound_usd"]
    for before, increment in ((0, 25), (1e6, -2e6), (-1e6, 2e6), (0, -enclosure)):
        reward, adjustment, after = excess_transition(
            before, increment, enclosure, terminal=True, valid=False
        )
        assert math.isfinite(reward)
        assert before / 100 + reward == pytest.approx(-65400)
        assert after == before + increment
        assert adjustment < 0
        assert (increment + adjustment) / 100 == pytest.approx(reward)
        valid_reward, adjustment, _ = excess_transition(
            before, increment, enclosure, terminal=True, valid=True
        )
        assert valid_reward == pytest.approx(increment / 100)
        assert adjustment == 0
    for key in ("cash_prefix_bound_usd", "lmp_abs_usd_mwh"):
        changed = copy.deepcopy(protocol["normalization"])
        changed[key] *= 2
        with pytest.raises(ValueError):
            excess_cash_bound(changed)
    missing = dict(protocol["normalization"])
    del missing["cash_prefix_bound_usd"]
    with pytest.raises(KeyError):
        excess_cash_bound(missing)
    with pytest.raises(ValueError):
        excess_transition(0, enclosure + 1, enclosure, terminal=False, valid=True)


def test_real_physical_fault_is_terminal_not_timeout_and_not_complete(
    protocol, training_cell
):
    task = EconomicDispatchTask(protocol, seed=1, fixed_cell=training_cell)
    try:
        task.reset()
        _, first_reward, term, trunc, _ = task.step([0])
        assert not term and not trunc
        original_threshold = task.fast._thermal.T_safe
        task.fast._thermal.temp_B = original_threshold + 0.1
        _, reward, term, trunc, info = task.step([0])
        assert task.fast._thermal.T_safe == original_threshold == 35
        assert term and not trunc
        assert info["physical_terminated"] and info["true_terminal"]
        assert not info["raw_service_truncated"]
        episode = task.episode_result
        assert episode["service_last_faults"]["thermal_fault"]
        assert not episode["qualified"] and not episode["complete"]
        assert episode["cashflow_26h_usd"] is None
        assert episode["exogenous_reference_cash_usd"] is None
        assert not episode["recovery"]["performed"]
        assert 180 < task.service.steps < 360
        assert first_reward + reward == pytest.approx(-65400)
        assert episode["reward_components"]["invalid_adjustment_usd"] < 0
        assert not episode["timeout_bootstrap"]
    finally:
        task.close()


def test_recovery_fault_does_not_fabricate_complete_26h_cash(
    protocol, training_cell, monkeypatch
):
    task = EconomicDispatchTask(protocol, seed=1, fixed_cell=training_cell)
    original_recover = task._recover

    def inject_fault_at_recovery(last_info):
        assert task.service.steps == 17280
        assert task.fast._thermal.T_safe == 35
        task.fast._thermal.temp_B = 35.1
        original_recover(last_info)

    monkeypatch.setattr(task, "_recover", inject_fault_at_recovery)
    try:
        task.reset()
        rewards, flags, infos = finish(task, [0.0])
        episode = task.episode_result
        assert episode["service_complete"] and not episode["complete"]
        assert not episode["service_physical_terminated"]
        assert episode["physical_terminated"]
        assert episode["termination_reason"] == "recovery_physical_fault"
        assert episode["recovery"]["performed"]
        assert not episode["recovery"]["complete"]
        assert episode["recovery"]["ledger"]["steps"] == 1
        assert episode["recovery"]["last_faults"]["thermal_fault"]
        assert episode["cashflow_26h_usd"] is None
        assert episode["cashflow_closed_recovery_usd"] is None
        assert episode["exogenous_reference_cash_usd"] is None
        assert sum(rewards) == pytest.approx(-65400)
        assert flags[-1] == (True, False)
        assert infos[-1]["physical_terminated"]
    finally:
        task.close()


def test_early_native_time_limit_is_invalid_not_timeout_bootstrap(
    protocol, training_cell
):
    task = EconomicDispatchTask(protocol, seed=1, fixed_cell=training_cell)
    try:
        task.reset()
        task.core._episode_macro_ticks = 1  # Inject a clock mismatch, not a fault.
        _, reward, term, trunc, info = task.step([0])
        assert term and not trunc
        assert not info["physical_terminated"]
        assert task.last_episode["termination_reason"] == "service_horizon_mismatch"
        assert task.last_episode["cashflow_26h_usd"] is None
        assert not task.last_episode["recovery"]["performed"]
        assert not task.last_episode["timeout_bootstrap"]
        assert reward == -65400
    finally:
        task.close()


@pytest.mark.parametrize(
    "action",
    (0.0, [], [0, 0], [1.00001], [-1.00001], [np.nan], [np.inf], ["0"], [True], [1j]),
)
def test_action_validation(action):
    with pytest.raises(ValueError):
        requested_extra_response(action)


def test_observation_schema_current_data_no_internal_reference_feature(
    protocol, training_cell, monkeypatch
):
    task = EconomicDispatchTask(protocol, seed=1, fixed_cell=training_cell)
    try:
        observation, _ = task.reset()
        assert observation.shape == (23,)
        assert observation.dtype == np.float32
        assert len(OBSERVATION_SCHEMA) == 23
        assert (Macro.REGD, Macro.LMP, Macro.RMCP, Macro.REG_NEED) == (5, 6, 17, 18)
        assert (Fast.REGD, Fast.LMP) == (6, 7)
        assert task.action_space.shape == (1,)
        np.testing.assert_array_equal(task.action_space.low, [-1])
        np.testing.assert_array_equal(task.action_space.high, [1])
        np.testing.assert_array_equal(observation[:19], task._macro_observation)
        assert observation[19] == task.core._last_hw_obs[Fast.REGD]
        assert observation[20] == task.core._last_hw_obs[Fast.LMP]
        assert observation[21] == 0 and observation[22] == 1

        class NoReferenceAccess:
            def __getattribute__(self, name):
                raise AssertionError(f"Hidden reference read: {name}")

        with monkeypatch.context() as patch:
            patch.setattr(task.fast, "_reference_thermal", NoReferenceAccess())
            np.testing.assert_array_equal(task._observation(), observation)
            task._window_end_tick = 180
            action = task._inner_action(task.core._last_hw_obs.copy(), np.zeros(2))
            assert task.fast.action_space.contains(action)
        with pytest.raises(ValueError):
            task.controller.features(np.zeros(19))
        with pytest.raises(ValueError):
            task.controller.features(np.full(18, np.nan))
        with pytest.raises(ValueError):
            task.controller.action(6.01)
        with pytest.raises(ValueError):
            task.step([np.nan])
        assert task.service.steps == 0 and not task.done
    finally:
        task.close()


def test_schedule_matches_standalone_convention_and_excludes_evaluation(
    protocol, training_cell
):
    task = EconomicDispatchTask(protocol, seed=4200, rank=2)
    try:
        for ordinal in (0, 1, 7, 8, 29, 100):
            task.episode_ordinal = ordinal
            slot = ordinal * protocol["n_envs"] + 2
            cycle, position = divmod(slot, len(protocol["train_days"]))
            rng = np.random.default_rng(np.random.SeedSequence([4200, 713, cycle]))
            index = int(rng.permutation(len(protocol["train_days"]))[position])
            expected_seed = int(
                np.random.SeedSequence([4200, 157, cycle, position]).generate_state(1)[
                    0
                ]
            )
            condition = task.condition()
            assert condition == {
                "start_day": protocol["train_days"][index],
                "regd_load_bias": protocol["train_biases"][
                    (position + cycle) % len(protocol["train_biases"])
                ],
                "seed": expected_seed,
                "curriculum_cycle": cycle,
                "curriculum_position": position,
            }
            assert condition["seed"] not in EVALUATION_SEEDS
            assert task.condition() == condition
    finally:
        task.close()
    for changed in (
        {**training_cell, "start_day": -1},
        {**training_cell, "seed": 90001},
        {**training_cell, "regd_load_bias": 0.75},
    ):
        with pytest.raises(ValueError):
            EconomicDispatchTask(protocol, seed=1, fixed_cell=changed)
    changed = copy.deepcopy(protocol)
    changed["final_cells"] = [{"seed": training_cell["seed"]}]
    with pytest.raises(ValueError, match="evaluation"):
        EconomicDispatchTask(changed, seed=1, fixed_cell=training_cell)


def test_evaluation_mode_uses_only_fixed_allowlisted_training_fixture(
    protocol, training_cell, zero_episode, monkeypatch
):
    registered = [copy.deepcopy(training_cell)]
    fixed = copy.deepcopy(training_cell)
    configured = copy.deepcopy(protocol)
    configured["evaluation_cells"] = copy.deepcopy(registered)
    before = copy.deepcopy(configured)
    with pytest.raises(ValueError, match="excluded evaluation seed"):
        EconomicDispatchTask(configured, seed=1, fixed_cell=fixed)
    task = EconomicDispatchTask(
        configured,
        seed=1,
        rank=protocol["n_envs"] + 1,
        fixed_cell=fixed,
        evaluation_mode=True,
        evaluation_cells=registered,
    )
    try:
        registered[0]["seed"] += 1
        fixed["start_day"] += 1
        assert configured == before
        assert task.protocol["train_days"] == protocol["train_days"]
        assert task.protocol["train_biases"] == protocol["train_biases"]
        with pytest.raises(AttributeError):
            task.evaluation_mode = False

        def no_training_rng(*args, **kwargs):
            raise AssertionError("Evaluation entered random training scheduling")

        with monkeypatch.context() as patch:
            patch.setattr(np.random, "default_rng", no_training_rng)
            for ordinal in (0, 1, 100):
                task.episode_ordinal = ordinal
                assert task.condition() == training_cell
            fixed_copy = task.fixed_cell
            task.fixed_cell = None
            with pytest.raises(ValueError, match="training scheduling"):
                task.condition()
            task.fixed_cell = fixed_copy
        observation, info = task.reset()
        assert info["condition"] == training_cell
        assert info["condition_role"] == "evaluation_only"
        assert task.observation_space.contains(observation)
        _, flags, infos = finish(task, [0])
        assert flags == [(False, False)] * 95 + [(True, False)]
        info = infos[-1]
        assert info["condition"] == training_cell
        assert info["condition_role"] == "evaluation_only"
        assert task.progress()["condition_role"] == "evaluation_only"
        assert not task.record_trace
        assert not task.service_rows and not task.macro_rows and not task.recovery_rows
        assert task.episode_result["qualified"]
        for key in ("service", "grid", "cashflow_26h_usd", "reward_sum"):
            assert task.episode_result[key] == zero_episode[0].episode_result[key]
        assert task.episode_result["grid"] is not None
        assert configured == before
    finally:
        task.close()
    assert task.episode_result["condition_role"] == "evaluation_only"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("start_day", 17),
        ("seed", 97000002),
        ("regd_load_bias", 0.25),
    ),
)
def test_evaluation_mode_rejects_unallowlisted_condition(
    protocol, training_cell, field, value
):
    cell = {**training_cell, field: value}
    with pytest.raises(ValueError, match="explicit evaluation_cells allowlist"):
        EconomicDispatchTask(
            protocol,
            seed=1,
            fixed_cell=cell,
            evaluation_mode=True,
            evaluation_cells=[training_cell],
        )


@pytest.mark.parametrize(
    "case",
    (
        "missing_cell",
        "missing_allowlist",
        "empty_allowlist",
        "implicit_mode",
        "nonboolean_mode",
        "duplicate_condition",
        "malformed_condition",
        "nonfinite_condition",
    ),
)
def test_evaluation_mode_requires_explicit_valid_allowlist(
    protocol, training_cell, case
):
    kwargs = {
        "fixed_cell": training_cell,
        "evaluation_mode": True,
        "evaluation_cells": [training_cell],
    }
    if case == "missing_cell":
        kwargs["fixed_cell"] = None
    elif case == "missing_allowlist":
        kwargs["evaluation_cells"] = None
    elif case == "empty_allowlist":
        kwargs["evaluation_cells"] = []
    elif case == "implicit_mode":
        kwargs["evaluation_mode"] = False
    elif case == "nonboolean_mode":
        kwargs["evaluation_mode"] = "true"
    elif case == "duplicate_condition":
        kwargs["evaluation_cells"] = [training_cell, training_cell]
    elif case == "malformed_condition":
        kwargs["evaluation_cells"] = [{"start_day": training_cell["start_day"]}]
    elif case == "nonfinite_condition":
        kwargs["evaluation_cells"] = [{**training_cell, "regd_load_bias": float("nan")}]
    with pytest.raises(ValueError):
        EconomicDispatchTask(protocol, seed=1, **kwargs)


def test_evaluation_mode_keeps_frozen_physical_and_qualification_guards(
    protocol, training_cell
):
    for changed in (
        {**protocol, "service_floor": 0.89},
        {**protocol, "recovery": {**protocol["recovery"], "hvac": 1.0}},
    ):
        with pytest.raises(ValueError):
            EconomicDispatchTask(
                changed,
                seed=1,
                fixed_cell=training_cell,
                evaluation_mode=True,
                evaluation_cells=[training_cell],
            )


def test_terminal_trace_export_uses_public_snapshots(
    zero_episode, diagnostic_directory
):
    task = zero_episode[0]
    snapshot = task.terminal_traces()
    assert set(snapshot) == {"hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz"}
    assert [len(rows) for rows in snapshot.values()] == [17280, 96, 1440]
    destination = diagnostic_directory / "public_trace_export"
    destination.mkdir()
    for filename, rows in snapshot.items():
        path = destination / filename
        pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
        assert len(pd.read_csv(path)) == len(rows)
    snapshot["hardware.csv.gz"][0]["guarded_extra_response_mw"] = 999
    assert task.service_rows[0]["guarded_extra_response_mw"] == 0
    snapshot["macro.csv.gz"].clear()
    assert len(task.macro_rows) == 96


def test_last_episode_exact_terminal_keys(zero_episode):
    expected = set(
        """
        task_schema condition cell condition_role decisions service_fast_steps
        recovery_fast_steps complete service_complete service_physical_terminated
        physical_terminated raw_service_truncated termination_reason true_terminal
        truncated timeout_bootstrap service_checks physical_qualified qualified
        delivered_service_floor_required delivered_service_floor
        delivered_service_gate delivered_service_ratio committed_mw_hours
        performance_weighted_mw_hours maximum_reference_mw_hours initial
        service_terminal service recovery grid regulation_revenue_usd
        ideal_regulation_reference_usd contractual_demand_electricity_usd
        observed_cashflow_usd cashflow_26h_usd cashflow_closed_recovery_usd
        cash_vs_nominal_26h_usd exogenous_reference_cash_usd credit bound_derivation
        original_service_reward_sum original_macro_reward_sum
        original_recovery_reward_sum inventory_accumulation_is_not_an_action
        fixed_offsets_are_efficiency_dispatch_not_time_shifting service_last_faults
        reward_sum excess_cash_usd centered_cash_usd reward_components
        invalid_score_is_not_money required_gamma reward_scale_usd
        """.split()
    )
    assert len(expected) == 54
    assert set(zero_episode[0].episode_result) == expected


def test_terminal_trace_export_rejects_disabled_or_active_capture(
    protocol, training_cell
):
    task = EconomicDispatchTask(
        protocol, seed=1, fixed_cell=training_cell, record_trace=True
    )
    no_trace = EconomicDispatchTask(protocol, seed=1, fixed_cell=training_cell)
    try:
        with pytest.raises(RuntimeError, match="terminal/closed"):
            task.terminal_traces()
        task.reset()
        with pytest.raises(RuntimeError, match="terminal/closed"):
            task.terminal_traces()
        with pytest.raises(ValueError, match="record_trace=True"):
            no_trace.terminal_traces()
        task.step([0])
        task.close()
        partial = task.terminal_traces()
        assert len(partial["hardware.csv.gz"]) == 180
        assert len(partial["macro.csv.gz"]) == 1
        assert partial["recovery.csv.gz"] == []
        assert task.episode_result["administrative_abort"]
        assert task.episode_result["cashflow_26h_usd"] is None
    finally:
        task.close()
        no_trace.close()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("recovery", "duration_seconds", 5),
        ("recovery", "hvac", 1.0),
        ("recovery", "pump", 0.8),
        ("recovery", "temperature_tolerance_C", 0.2),
        ("service_gate", "max_backlog_fraction", 0.02),
    ),
)
def test_cannot_relax_original_service_or_recovery(protocol, section, key, value):
    changed = copy.deepcopy(protocol)
    changed[section][key] = value
    with pytest.raises(ValueError, match="Frozen"):
        EconomicDispatchTask(changed, seed=1)


def test_no_reset_escape_clean_close_and_no_implicit_output(protocol, training_cell):
    task = EconomicDispatchTask(protocol, seed=1, fixed_cell=training_cell)
    task.reset()
    before_files = set(ROOT.rglob("*"))
    task.step([-1.0])
    assert not task.service_rows and not task.macro_rows and not task.recovery_rows
    assert task.progress()["cashflow_26h_usd"] is None
    with pytest.raises(RuntimeError, match="incomplete"):
        task.reset()
    task.close()
    task.close()
    assert task.episode_result["administrative_abort"]
    assert not task.episode_result["true_terminal"]
    assert task.episode_result["cashflow_26h_usd"] is None
    assert not task.episode_result["invalid_penalty_returned"]
    assert set(ROOT.rglob("*")) == before_files
    with pytest.raises(RuntimeError, match="closed"):
        task.reset()
    with pytest.raises(RuntimeError):
        task.step([0])


def test_frozen_registered_sources_unchanged():
    registration = json.loads((ROOT.parent / "headroom_registration.json").read_text())
    for path, digest in registration["source_hashes"].items():
        with Path(path).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == digest, path
