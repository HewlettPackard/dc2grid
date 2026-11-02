"""Training-fixture contracts for the adaptive fractional-window follow-up."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess

# isort: off
from fraction_macro import (
    ACTION_SEMANTICS,
    DT_HOURS,
    OBSERVATION_SCHEMA,
    POTENTIAL_PRICE_USD_PER_MWH,
    ROOT,
    TASK_SCHEMA,
    V1_SNAPSHOT,
    CreditPotentialEnv,
    FractionalDispatchTask,
    credit_potential,
    window_fraction,
)
from track_b.economy_macro import (
    CapacityErrorBudget,
    EconomicDispatchController,
    EconomicDispatchTask,
)
from economy_context import PRIOR
from c2g_env.obs_indices import Fast

import numpy as np
import pandas as pd
import pytest
# isort: on


@pytest.fixture(scope="module", autouse=True)
def frozen_dependencies_are_unchanged():
    project = ROOT.parents[2]
    paths = set(ROOT.parent.glob("*.py"))
    paths.update(ROOT.parent.glob("*registration*.json"))
    for directory in ("track_a", "track_b"):
        paths.update((ROOT.parent / directory).glob("*.py"))
        paths.update((ROOT.parent / directory).glob("*registration*.json"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    base = ROOT.parent / "track_b/economy_macro.py"
    committed = subprocess.check_output(
        ["git", "--no-pager", "show", f"{V1_SNAPSHOT}:{base.relative_to(project)}"],
        cwd=project,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    assert hashlib.sha256(committed).hexdigest() == before[base]
    yield
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    } == before


@pytest.fixture(scope="module")
def protocol():
    original = json.loads((PRIOR / "registration.json").read_text())
    return {
        key: copy.deepcopy(original[key])
        for key in (
            "config_path",
            "train_days",
            "train_biases",
            "n_envs",
            "normalization",
            "service_gate",
            "recovery",
            "service_floor",
        )
    }


@pytest.fixture(scope="module")
def cell(protocol):
    return {
        "start_day": protocol["train_days"][0],
        "seed": 97000001,
        "regd_load_bias": 0.5,
    }


def run_episode(env, action):
    observation, reset_info = env.reset()
    assert observation.shape == (23,)
    rewards, flags, infos, observations = [], [], [], []
    for _ in range(96):
        observation, reward, term, trunc, info = env.step(action)
        assert env.observation_space.contains(observation)
        rewards.append(reward)
        flags.append((term, trunc))
        infos.append(info)
        observations.append(observation.copy())
        if term or trunc:
            break
    return {
        "reset_info": reset_info,
        "rewards": rewards,
        "flags": flags,
        "infos": infos,
        "observations": observations,
    }


@pytest.fixture(scope="module")
def zero_pair(protocol, cell):
    base = EconomicDispatchTask(protocol, seed=1, fixed_cell=cell, record_trace=True)
    fraction = FractionalDispatchTask(
        protocol, seed=1, fixed_cell=cell, record_trace=True
    )
    try:
        base_run = run_episode(base, [0])
        fraction_run = run_episode(fraction, [0])
        yield base, fraction, base_run, fraction_run
    finally:
        base.close()
        fraction.close()


@pytest.fixture(scope="module")
def potential_pair(protocol, cell):
    bare = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell, record_trace=True)
    inner = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell, record_trace=True)
    shaped = CreditPotentialEnv(inner, coefficient_usd_per_mwh=10.0)
    try:
        bare_run = run_episode(bare, [0.5])
        shaped_run = run_episode(shaped, [0.5])
        yield bare, shaped, bare_run, shaped_run
    finally:
        bare.close()
        shaped.close()


def test_zero_action_full_26h_exact_v1_analytic_parity(zero_pair):
    base, fraction, old, new = zero_pair
    assert fraction.episode_result["complete"] and fraction.episode_result["qualified"]
    assert old["rewards"] == new["rewards"]
    assert new["flags"] == [(False, False)] * 95 + [(True, False)]
    np.testing.assert_array_equal(old["observations"], new["observations"])
    assert isinstance(fraction.controller, EconomicDispatchController)
    assert type(fraction.controller) is type(base.controller)
    assert type(fraction.budget) is CapacityErrorBudget
    for key, value in base.episode_result.items():
        if key != "task_schema":
            assert fraction.episode_result[key] == value, key
    for name in ("hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz"):
        previous = pd.DataFrame(base.terminal_traces()[name])
        current = pd.DataFrame(fraction.terminal_traces()[name])
        pd.testing.assert_frame_equal(
            previous, current[previous.columns], check_exact=True
        )
    assert fraction.episode_result["task_schema"] == TASK_SCHEMA
    assert new["reset_info"]["task_schema"] == TASK_SCHEMA
    assert fraction.episode_result["action_semantics"] == ACTION_SEMANTICS
    assert fraction.episode_result["fraction_window_count"] == 96
    assert fraction.episode_result["adaptive_dev_informed_followup"]
    assert not fraction.episode_result["fresh_validation_claimed"]
    assert fraction.service.steps == 17280 and fraction.recovery.steps == 1440


def test_constant_fractions_spend_different_window_budgets(protocol, cell):
    summaries = []
    for fraction in (0.25, 0.5, 1.0):
        task = FractionalDispatchTask(
            protocol, seed=1, fixed_cell=cell, record_trace=True
        )
        try:
            task.reset()
            assert task.fast.committed_mw == 5
            _, _, term, trunc, info = task.step([fraction])
            assert not term and not trunc
            assert len(task.fraction_windows) == 1
            frozen = task.fraction_windows[0]
            assert frozen["window_committed_mw"] == info["committed_mw"] > 5
            allowance = CapacityErrorBudget().cap(info["committed_mw"], 0.25)
            assert frozen["initial_allowance_mw"] == allowance
            assert frozen["window_fraction"] == fraction
            fixed = fraction * allowance
            assert info["desired_extra_response_mw"] == fixed
            assert task.macro_rows[0]["desired_extra_response_mw"] == fixed
            assert task.requested_extra_mwh == pytest.approx(fixed * 0.25)
            assert all(
                row["fixed_window_request_mw"] == fixed
                and row["desired_extra_response_mw"] == fixed
                and row["initial_allowance_mw"] == allowance
                for row in task.service_rows
            )
            if fraction == 0.25:
                first, last = task.service_rows[0], task.service_rows[-1]
                assert last["guard_cap_mw"] > first["guard_cap_mw"] + 1
                assert last["guarded_extra_response_mw"] == fixed
                assert last["guarded_extra_response_mw"] != pytest.approx(
                    fraction * last["guard_cap_mw"]
                )
            summaries.append((task.requested_extra_mwh, task.budget.credit_mwh))
        finally:
            task.close()
    assert summaries[0][0] < summaries[1][0] < summaries[2][0]
    assert summaries[0][1] > summaries[1][1] > summaries[2][1]


def test_allowance_frozen_once_per_window_and_refreshed_after_fault_reset(
    protocol, cell
):
    task = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell, record_trace=True)
    try:
        task.reset()
        for index, fraction in enumerate((0.25, 0.75, -1.0)):
            credit = task.budget.credit_mwh
            _, _, term, trunc, info = task.step([fraction])
            assert not term and not trunc
            assert len(task.fraction_windows) == index + 1
            frozen = task.fraction_windows[-1]
            assert frozen["dispatch_window_index"] == index
            assert frozen["window_start_tick"] == index * 180
            assert frozen["window_credit_start_mwh"] == credit
            assert frozen["full_current_window_hours"] == 0.25
            assert frozen["window_fraction"] == max(fraction, 0)
            expected = CapacityErrorBudget(credit_mwh=credit).cap(
                info["committed_mw"], 0.25
            )
            assert frozen["initial_allowance_mw"] == expected
            for row in task.service_rows[-180:]:
                for key, value in frozen.items():
                    assert row[key] == value, key
        task.fast._thermal.temp_B = 35.1
        _, _, term, trunc, _ = task.step([0.5])
        assert term and not trunc
        assert len(task.fraction_windows) == 4
        assert task.episode_result["fraction_window_count"] == 4
        assert task.episode_result["cashflow_26h_usd"] is None
        task.reset()
        assert not task.fraction_windows and task.budget.credit_mwh == 0
        _, _, term, trunc, info = task.step([0.5])
        assert not term and not trunc
        assert len(task.fraction_windows) == 1
        assert info["window_start_tick"] == 0
        assert info["window_credit_start_mwh"] == 0
    finally:
        task.close()


def test_original_guard_and_actual_credit_remain_under_delivery_error(protocol, cell):
    task = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell, record_trace=True)
    try:
        task.reset()
        # Inject a measured warm state, not a changed threshold or demand.
        task.fast._thermal.temp_B = 32.49
        task.core._last_hw_obs[Fast.TEMP_B] = 32.49 / 35
        task.step([1.0])
        assert task.fast._thermal.T_safe == 35
        assert any(row["guard_applied"] for row in task.service_rows)
        frozen = task.fraction_windows[0]
        budget = CapacityErrorBudget()
        for row in task.service_rows:
            remaining = (180 - (row["tick"] - 1) % 180) * DT_HOURS
            cap = budget.cap(row["committed_mw"], remaining)
            assert row["guard_cap_mw"] == cap
            assert row["fixed_window_request_mw"] == frozen["fixed_window_request_mw"]
            assert row["guarded_extra_response_mw"] == min(
                frozen["fixed_window_request_mw"], cap
            )
            assert row["credit_before_mwh"] == budget.credit_mwh
            budget.update(row["committed_mw"], row["tracking_err_kw"], DT_HOURS)
            assert row["credit_after_mwh"] == budget.credit_mwh
            assert row["delta_p_demanded_kw"] == pytest.approx(
                row["committed_mw"] * 1000 * row["regd_signal"], abs=1e-9
            )
        assert task.budget.credit_mwh == budget.credit_mwh
        assert task.budget.absolute_error_mwh == budget.absolute_error_mwh
    finally:
        task.close()


def test_nonzero_fraction_does_not_mutate_contracts_work_or_nominal_reference(
    zero_pair, potential_pair
):
    zero = zero_pair[1]
    positive = potential_pair[0]
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
        np.testing.assert_array_equal(
            [row[name] for row in zero.service_rows],
            [row[name] for row in positive.service_rows],
        )
    for old, new in zip(zero.macro_rows, positive.macro_rows, strict=True):
        for key in ("bid_mw", "bid_price", "bid_accepted", "rate", "residual_mw"):
            assert old[key] == new[key], key
    assert all(row["throttle_batch"] == 1 for row in positive.service_rows)
    assert all(row["soc_target"] == 0.5 for row in positive.service_rows)
    assert (
        positive.episode_result["exogenous_reference_cash_usd"]
        == (zero.episode_result["exogenous_reference_cash_usd"])
    )


def assert_potential_terminal_identity(shaped, run):
    terminal = shaped.episode_result
    raw = shaped.env.episode_result
    for key, value in raw.items():
        assert terminal[key] == value, key
    assert terminal["raw_return"] == pytest.approx(raw["reward_sum"], abs=1e-8)
    assert terminal["shaped_return"] == pytest.approx(raw["reward_sum"], abs=1e-8)
    assert sum(run["rewards"]) == pytest.approx(raw["reward_sum"], abs=1e-8)
    assert terminal["potential_shaping"]["returned_terminal"]
    assert terminal["potential_shaping"]["telescoping_complete"]
    assert terminal["potential_shaping"]["initial_phi"] == 0
    assert terminal["potential_shaping"]["final_phi"] == 0
    assert terminal["potential_shaping"]["potential_delta_sum"] == pytest.approx(
        0, abs=1e-8
    )
    assert run["infos"][-1]["episode_result"] is terminal
    assert run["flags"][-1] == (True, False)
    for reward, info in zip(run["rewards"], run["infos"], strict=True):
        components = info["potential_shaping"]
        assert reward == components["raw_reward"] + components["potential_delta"]
        assert components["potential_delta"] == (
            components["phi_after"] - components["phi_before"]
        )


def test_potential_preserves_healthy_actual_cash_and_complete_return(potential_pair):
    bare, shaped, original, modified = potential_pair
    assert bare.episode_result["qualified"] and shaped.episode_result["qualified"]
    assert bare.episode_result == shaped.env.episode_result
    np.testing.assert_array_equal(original["observations"], modified["observations"])
    assert original["flags"] == modified["flags"]
    assert bare.terminal_traces() == shaped.env.terminal_traces()
    assert_potential_terminal_identity(shaped, modified)
    expected = (
        bare.episode_result["cashflow_26h_usd"]
        - bare.episode_result["exogenous_reference_cash_usd"]
    ) / 100
    assert sum(original["rewards"]) == pytest.approx(expected, abs=1e-8)
    assert sum(modified["rewards"]) == pytest.approx(expected, abs=1e-8)
    assert any(
        abs(raw - augmented) > 1e-6
        for raw, augmented in zip(original["rewards"], modified["rewards"], strict=True)
    )
    assert credit_potential(modified["observations"][-1]) > 0
    assert modified["infos"][-1]["potential_shaping"]["phi_after"] == 0
    assert len(shaped.terminal_traces()["potential.csv.gz"]) == 96
    assert "potential.csv.gz" not in bare.terminal_traces()


def test_potential_preserves_invalid_fault_cash_and_return(protocol, cell):
    bare = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell)
    inner = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell)
    shaped = CreditPotentialEnv(inner)
    try:
        runs = []
        for env, task in ((bare, bare), (shaped, inner)):
            env.reset()
            run = {"rewards": [], "infos": [], "flags": []}
            _, reward, term, trunc, info = env.step([0.5])
            assert not term and not trunc
            run["rewards"].append(reward)
            run["infos"].append(info)
            run["flags"].append((term, trunc))
            task.fast._thermal.temp_B = 35.1
            _, reward, term, trunc, info = env.step([0.5])
            run["rewards"].append(reward)
            run["infos"].append(info)
            run["flags"].append((term, trunc))
            runs.append(run)
        assert bare.episode_result == inner.episode_result
        assert inner.episode_result["physical_terminated"]
        assert not inner.episode_result["qualified"]
        assert inner.episode_result["cashflow_26h_usd"] is None
        assert inner.episode_result["reward_sum"] == pytest.approx(-65400)
        assert_potential_terminal_identity(shaped, runs[1])
        shaped.reset()
        assert shaped.phi == shaped.initial_phi == 0
        assert shaped.shaped_return == shaped.raw_return == 0
        assert not inner.fraction_windows
    finally:
        bare.close()
        shaped.close()


def test_recovery_fault_also_zeroes_terminal_potential(protocol, cell, monkeypatch):
    task = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell)
    original_recover = task._recover

    def fault_in_recovery(last_info):
        assert task.service.steps == 17280
        task.fast._thermal.temp_B = 35.1
        original_recover(last_info)

    monkeypatch.setattr(task, "_recover", fault_in_recovery)
    shaped = CreditPotentialEnv(task)
    try:
        run = run_episode(shaped, [0.0])
        assert task.episode_result["termination_reason"] == "recovery_physical_fault"
        assert task.episode_result["service_complete"]
        assert not task.episode_result["complete"]
        assert task.recovery.steps == 1
        assert task.episode_result["cashflow_26h_usd"] is None
        assert task.episode_result["reward_sum"] == pytest.approx(-65400)
        assert_potential_terminal_identity(shaped, run)
    finally:
        shaped.close()


@pytest.mark.parametrize(
    "action",
    (0, [], [0, 0], [np.nan], [np.inf], [-1.01], [1.01], ["0"], [True], [1j]),
)
def test_fraction_action_validation(action):
    with pytest.raises(ValueError):
        window_fraction(action)


@pytest.mark.parametrize("action", (-1.0, 0.0, 0.25, 1.0))
def test_valid_fraction_coordinates(action):
    assert window_fraction([action]) == max(action, 0)


def test_potential_uses_only_observed_credit_and_units():
    assert len(OBSERVATION_SCHEMA) == 23
    assert POTENTIAL_PRICE_USD_PER_MWH == 10.0
    observation = np.zeros(23, dtype=np.float32)
    observation[21] = 0.5
    assert credit_potential(observation) == 1.5
    observation[21] = -0.5
    assert credit_potential(observation) == -1.5
    observation[:21] = np.arange(21)
    observation[22] = 0.75
    assert credit_potential(observation) == -1.5
    for invalid in (np.zeros(22), np.full(23, np.nan), np.zeros(23, dtype=bool)):
        with pytest.raises(ValueError):
            credit_potential(invalid)


def test_evaluation_allowlists_unchanged_and_potential_training_only(protocol, cell):
    configured = copy.deepcopy(protocol)
    configured["evaluation_cells"] = [copy.deepcopy(cell)]
    original = copy.deepcopy(configured)
    with pytest.raises(ValueError, match="evaluation seed"):
        FractionalDispatchTask(configured, seed=1, fixed_cell=cell)
    for changed in (
        {**cell, "start_day": 17},
        {**cell, "seed": cell["seed"] + 1},
        {**cell, "regd_load_bias": 0.25},
    ):
        with pytest.raises(ValueError, match="evaluation_cells allowlist"):
            FractionalDispatchTask(
                configured,
                seed=1,
                fixed_cell=changed,
                evaluation_mode=True,
                evaluation_cells=[cell],
            )
    with pytest.raises(ValueError, match="fixed_cell"):
        FractionalDispatchTask(
            configured, seed=1, evaluation_mode=True, evaluation_cells=[cell]
        )
    evaluation = FractionalDispatchTask(
        configured,
        seed=1,
        fixed_cell=cell,
        evaluation_mode=True,
        evaluation_cells=[cell],
    )
    try:
        with pytest.raises(ValueError, match="training-only"):
            CreditPotentialEnv(evaluation)
        observation, info = evaluation.reset()
        assert observation.shape == (23,) and evaluation.action_space.shape == (1,)
        assert info["condition_role"] == "evaluation_only"
        _, _, _, _, info = evaluation.step([0.5])
        assert info["condition"] == cell
        assert info["window_fraction"] == 0.5
        assert configured == original
    finally:
        evaluation.close()


@pytest.mark.parametrize("gamma", (0.99, 0.0, True, np.nan))
def test_potential_requires_gamma_one(protocol, cell, gamma):
    task = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell)
    try:
        with pytest.raises(ValueError, match="gamma=1"):
            CreditPotentialEnv(task, gamma=gamma)
    finally:
        task.close()


@pytest.mark.parametrize("coefficient", (0, 5, 20, True, np.nan))
def test_potential_coefficient_is_fixed_at_registered_ten(protocol, cell, coefficient):
    task = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell)
    try:
        with pytest.raises(ValueError, match="10 USD/MWh"):
            CreditPotentialEnv(task, coefficient_usd_per_mwh=coefficient)
    finally:
        task.close()


def test_public_capture_close_and_partial_return_do_not_claim_invariance(
    protocol, cell
):
    task = FractionalDispatchTask(protocol, seed=1, fixed_cell=cell, record_trace=True)
    shaped = CreditPotentialEnv(task)
    try:
        shaped.reset()
        shaped.step([0.25])
        with pytest.raises(RuntimeError, match="incomplete"):
            shaped.reset()
        with pytest.raises(RuntimeError, match="terminal/closed"):
            shaped.terminal_traces()
        shaped.close()
        shaped.close()
        episode = shaped.episode_result
        assert episode["administrative_abort"]
        assert episode["cashflow_26h_usd"] is None
        assert not episode["potential_shaping"]["telescoping_complete"]
        assert episode["shaped_return"] != episode["raw_return"]
        traces = shaped.terminal_traces()
        assert len(traces["hardware.csv.gz"]) == 180
        assert len(traces["macro.csv.gz"]) == 1
        assert len(traces["potential.csv.gz"]) == 1
        assert traces["recovery.csv.gz"] == []
        traces["potential.csv.gz"][0]["raw_reward"] = 999
        assert shaped.potential_rows[0]["raw_reward"] != 999
        with pytest.raises(RuntimeError):
            shaped.step([0])
    finally:
        shaped.close()
