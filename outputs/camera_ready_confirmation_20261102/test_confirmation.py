"""Focused mechanics, physical parity, and conditional-inference regressions."""

from __future__ import annotations

import copy

# isort: off
from confirmation_context import (
    PREVIOUS,
    cap_cpu,
    case_key,
    file_hash,
    read_json,
    write_json,
)
from controls import load_model, make_environment, scripted_action, scripted_family
from confirmation_eval import evaluate_case
from confirmation_audit import (
    EXOGENOUS_COLUMNS,
    MARKET_COLUMNS,
    METRICS,
    audit_record,
    inputs_digest,
    peak_reductions,
)
from confirmation_analysis import (
    confirmation_inference,
    noise_indices,
    paired_estimate,
    summarize,
)
from prepare_confirmation import historical_exclusions, make_cells, select_dates
from economy_macro import EconomicDispatchTask
from headroom_report import episode_metrics
import numpy as np
import pandas as pd
import pytest
import run_confirmation
# isort: on


@pytest.fixture(scope="module")
def task_protocol():
    cap_cpu(32)
    return read_json(PREVIOUS / "track_b_fraction/registration.json")


@pytest.fixture(scope="module")
def neural_policies(task_protocol):
    manifest = read_json(PREVIOUS / "track_b_fraction/development/manifest.json")
    return [
        {
            **entry,
            "id": "model__" + entry["id"],
            "kind": "neural",
            "source_policy_id": entry["id"],
            "training_registration_id": task_protocol["registration_id"],
        }
        for entry in manifest["policies"]
    ]


def test_closed_scripted_family_and_units():
    policies = scripted_family()
    assert len(policies) == len({row["id"] for row in policies}) == 27
    observation = np.zeros(23, dtype=np.float32)
    observation[20], observation[17], observation[22] = 0.25, 0.25, 1.0
    for policy in policies:
        action = scripted_action(policy, observation)
        assert action.shape == (1,) and action.dtype == np.float32
        assert 0 <= action[0] <= 1
        if policy["mode"] != "credit":
            assert action[0] == policy["fraction"]


def test_margin_deadline_and_credit_causality():
    policies = {row["id"]: row for row in scripted_family()}
    observation = np.zeros(23, dtype=np.float32)
    observation[20], observation[17], observation[22] = 0.25, 0.4375, 1.0
    assert scripted_action(policies["script_f100_margin_10"], observation)[0] == 0
    assert scripted_action(policies["script_deadline_margin10"], observation)[0] == 0
    observation[22] = 0.5
    assert scripted_action(policies["script_deadline_margin10"], observation)[0] == 1
    assert scripted_action(policies["script_credit_one"], observation)[0] == 0
    observation[21] = 1 / 30
    assert scripted_action(policies["script_credit_one"], observation)[0] == 1
    observation[21] = 0.999 / 30
    assert scripted_action(policies["script_credit_one"], observation)[0] == 0


@pytest.mark.parametrize(
    "bad", [np.zeros(22), np.full(23, np.nan), np.full(23, np.inf)]
)
def test_invalid_observation_is_an_error(bad):
    with pytest.raises(ValueError):
        scripted_action(scripted_family()[0], bad)


def test_all_checkpoint_metadata_and_shapes(neural_policies):
    for policy in neural_policies:
        model = load_model(policy)
        assert model.gamma == 1
        assert model.observation_space.shape == (23,)
        assert model.action_space.shape == (1,)


def test_unwrapped_script_and_once_wrapped_ppo(task_protocol, neural_policies):
    cell = {
        "start_day": task_protocol["train_days"][0],
        "seed": 19000011,
        "regd_load_bias": 0.5,
    }
    script = next(row for row in scripted_family() if row["id"] == "script_f100_always")
    task, environment = make_environment(task_protocol, cell, [cell], script)
    try:
        assert task is environment
        environment.reset()
        environment.step(np.array([1], dtype=np.float32))
        assert task.fraction_windows[0]["window_fraction"] == 1.0
    finally:
        environment.close()
    policy = next(row for row in neural_policies if row["algorithm"] == "ppo")
    task, environment = make_environment(task_protocol, cell, [cell], policy)
    try:
        assert environment.env is task
        environment.reset()
        environment.step(np.array([1], dtype=np.float32))
        assert task.fraction_windows[0]["window_fraction"] == pytest.approx(
            np.tanh(np.float32(1)),
            abs=1e-7,
        )
    finally:
        environment.close()


def smoke_protocol(task_protocol, policy):
    cell = {
        "start_day": task_protocol["train_days"][0],
        "seed": 19000011,
        "regd_load_bias": 0.5,
    }
    return {
        "registration_id": "pre-registration-training-fixture-only",
        "task_protocol": {"path": str(PREVIOUS / "track_b_fraction/registration.json")},
        "policies": [policy],
        "panels": {"smoke": {"cells": [cell], "scope": "previous_training_date_only"}},
    }, cell


def test_complete_zero_overlay_matches_absolute_task(task_protocol, tmp_path):
    policy = next(row for row in scripted_family() if row["id"] == "script_zero")
    protocol, cell = smoke_protocol(task_protocol, policy)
    record = evaluate_case(protocol, "smoke", policy, cell, tmp_path / "fraction")
    old = EconomicDispatchTask(
        task_protocol,
        seed=0,
        fixed_cell=cell,
        record_trace=True,
        evaluation_mode=True,
        evaluation_cells=[cell],
    )
    try:
        old.reset()
        while True:
            _, _, terminal, truncated, _ = old.step(np.zeros(1, dtype=np.float32))
            assert not truncated
            if terminal:
                break
        audit = audit_record(
            tmp_path / "fraction",
            record,
            tracking_floor_kw=old.fast._macro_tracking_floor_kw,
        )
        assert audit["complete"] and audit["qualified"]
        assert audit["raw_checks"] > 1000
        assert old.episode_result["qualified"]
        for metric, value in episode_metrics(old.episode_result).items():
            assert audit["metrics"][metric] == pytest.approx(value, abs=1e-8)
    finally:
        old.close()


@pytest.mark.parametrize("family", ["fraction_ppo", "potential_sac"])
def test_complete_frozen_neural_deployment(
    family,
    neural_policies,
    task_protocol,
    tmp_path,
):
    policy = next(row for row in neural_policies if row["family"] == family)
    protocol, cell = smoke_protocol(task_protocol, policy)
    record = evaluate_case(protocol, "smoke", policy, cell, tmp_path / family)
    task, environment = make_environment(task_protocol, cell, [cell], policy)
    try:
        environment.reset()
        audit = audit_record(
            tmp_path / family,
            record,
            tracking_floor_kw=task.fast._macro_tracking_floor_kw,
        )
    finally:
        environment.close()
    assert audit["complete"] and audit["qualified"]
    assert not record["potential_used_at_evaluation"]


def test_exact_exclusions_and_new_streams(task_protocol):
    old, consumed, reserved, invalid, forbidden = historical_exclusions()
    days, tiers, eligible, _ = select_dates(consumed, reserved, invalid)
    assert len(eligible) == 133
    assert days == [7, 46, 69, 107, 135, 214, 216, 307]
    assert [
        list(tiers.values()).count(tier) for tier in ("p75_not_p90", "middle", "low")
    ] == [2, 3, 3]
    assert all(abs(a - b) >= 2 for a in days for b in consumed | reserved)
    assert all(abs(a - b) >= 2 for a in days for b in days if a != b)
    assert old["eligible_starts_before_reservation"] == 149
    used = set()
    calendar, _ = make_cells(days, 810021, forbidden, used)
    peak, _ = make_cells([207, 248, 249], 810022, forbidden, used)
    assert len(calendar) == 160 and len(peak) == 60 and len(used) == 110
    assert not used.intersection(forbidden)
    assert not used.intersection(row["seed"] for row in old["cells"])


def test_recovery_can_set_the_actual_boundary_peak():
    service = pd.DataFrame(
        {
            "grid_load_mw": [10, 12],
            "p_reference_mw": [2, 2],
            "p_grid_import_mw": [1, 1],
            "p_facility_mw": [1, 1],
        }
    )
    recovery = pd.DataFrame(
        {
            "grid_load_mw": [15],
            "p_reference_mw": [2],
            "p_grid_import_mw": [4],
            "p_facility_mw": [4],
        }
    )
    metrics = peak_reductions(service, recovery)
    assert metrics["pcc_peak_reduction_24h_mw"] == 1
    assert metrics["pcc_peak_reduction_26h_mw"] == -2
    assert metrics["pcc_peak_phase_26h"] == "recovery"


def test_paired_constant_effect_has_zero_monte_carlo_uncertainty():
    cells = [
        {"start_day": day, "seed": 1000 + day * 10 + seed, "regd_load_bias": bias}
        for day in range(8)
        for seed in range(10)
        for bias in (0.0, 0.5)
    ]
    dates, seeds, indices = noise_indices(cells, 10000, 2026110201, 11)
    assert len(dates) == 8 and all(len(values) == 10 for values in seeds.values())
    summary, samples = paired_estimate(np.full((8, 10), 7.0), indices)
    assert summary["mean"] == summary["lower_025"] == summary["upper_975"] == 7
    assert summary["bootstrap_standard_error"] == 0
    assert np.array_equal(samples, np.full(10000, 7.0))
    assert np.array_equal(indices, noise_indices(cells, 10000, 2026110201, 11)[2])
    with pytest.raises(ValueError, match="date-unique"):
        noise_indices(
            [
                {"start_day": day, "seed": seed, "regd_load_bias": 0.0}
                for day in (1, 2)
                for seed in (100, 101)
            ],
            50,
            1,
            11,
        )


def inference_fixture():
    panels = {}
    for panel, days in (("calendar", [1, 3, 5]), ("peak_noise", [10, 12, 14])):
        panels[panel] = {
            "cells": [
                {
                    "start_day": day,
                    "seed": 1000 + day * 10 + seed,
                    "regd_load_bias": bias,
                }
                for day in days
                for seed in range(3)
                for bias in (0.0, 0.5)
            ]
        }
    policies = [
        {"id": f"candidate_{seed}", "family": "potential_sac"} for seed in range(4)
    ] + [{"id": "script_f100_always", "family": "script_f100_always"}]
    protocol = {
        "panels": panels,
        "policies": policies,
        "candidate_family": "potential_sac",
        "primary_comparator_id": "script_f100_always",
        "analysis": {
            "bootstrap_replicates": 100,
            "bootstrap_seed": 42,
            "physical_gate_panels": ["calendar", "peak_noise"],
            "primary_metrics": ["stress_pcc_relief_mw", "cashflow_26h_usd"],
            "point_thresholds": {
                "stress_pcc_relief_mw": 0.001,
                "cashflow_26h_usd": 5.0,
            },
        },
    }
    records = []
    for panel, info in panels.items():
        for policy in policies:
            for cell in info["cells"]:
                candidate = policy["family"] == "potential_sac"
                records.append(
                    {
                        "panel": panel,
                        "cell": cell,
                        "policy_id": policy["id"],
                        "family": policy["family"],
                        "qualified": True,
                        "complete": True,
                        "metrics": {
                            **dict.fromkeys(METRICS, 0.0),
                            "stress_pcc_relief_mw": 1.0 if candidate else 0.0,
                            "cashflow_26h_usd": 10.0 if candidate else 0.0,
                        },
                    }
                )
    return protocol, records


def test_joint_rule_and_no_survivor_repair():
    protocol, records = inference_fixture()
    result = confirmation_inference(protocol, records)
    assert result["status"] == "supported_on_the_prespecified_conditional_panel"
    damaged = copy.deepcopy(records)
    damaged[0]["qualified"] = False
    result = confirmation_inference(protocol, damaged)
    assert result["status"] == "not_admitted_physical_or_service_gate_failure"
    summary = summarize(protocol, damaged, ["calendar"])
    group = summary["calendar"]["0.0"]["families"]["potential_sac"]
    assert not group["all_qualified"]
    assert all(values["mean"] is None for values in group["metrics"].values())
    damaged = copy.deepcopy(records)
    next(row for row in damaged if row["policy_id"] == "script_f100_always")[
        "qualified"
    ] = False
    assert confirmation_inference(protocol, damaged)["status"].startswith(
        "not_admitted"
    )


def test_zero_bias_guard_and_two_endpoint_threshold():
    protocol, records = inference_fixture()
    for row in records:
        if row["family"] == "potential_sac" and row["cell"]["regd_load_bias"] == 0:
            row["metrics"]["cashflow_26h_usd"] = -1
    assert confirmation_inference(protocol, records)["status"] == (
        "joint_improvement_not_established_by_bounds_or_zero_bias_guard"
    )
    for row in records:
        if row["family"] == "potential_sac":
            row["metrics"]["stress_pcc_relief_mw"] = 0
    assert confirmation_inference(protocol, records)["status"] == (
        "joint_improvement_not_supported_by_point_estimates"
    )


def test_real_cross_policy_pairing_excludes_only_endogenous_frequency():
    plan = read_json(PREVIOUS / "peak_regression_registration.json")
    selected = [
        entry
        for entry in plan["policies"]
        if entry["policy"]["family"] in {"analytic", "greedy_budget"}
        or (
            entry["policy"]["family"] == "potential_sac"
            and entry["policy"]["training_seed"] == 1200
        )
    ]
    assert len(selected) == 3
    cell = {"start_day": 207, "seed": 100, "regd_load_bias": 0.5}
    sets = []
    for entry in selected:
        directory = PREVIOUS / "original_peak_regression" / entry["id"] / case_key(cell)
        sets.append(
            [
                pd.read_csv(directory / name)
                for name in ("hardware.csv.gz", "recovery.csv.gz", "macro.csv.gz")
            ]
        )
    assert "f_grid_hz" not in EXOGENOUS_COLUMNS
    assert all("f_grid_hz" in frames[0] for frames in sets)
    assert len({inputs_digest(*frames) for frames in sets}) == 1
    assert any(
        not np.array_equal(sets[0][0].f_grid_hz, frames[0].f_grid_hz)
        for frames in sets[1:]
    )
    original = inputs_digest(*sets[0])
    for frame_index, columns in ((0, EXOGENOUS_COLUMNS), (2, MARKET_COLUMNS)):
        for column in columns:
            changed = [frame.copy() for frame in sets[0]]
            changed[frame_index][column] = changed[frame_index][column].astype(float)
            changed[frame_index].loc[0, column] = np.nextafter(
                changed[frame_index].loc[0, column],
                np.inf,
            )
            assert inputs_digest(*changed) != original
    modified_frequency = [frame.copy() for frame in sets[0]]
    modified_frequency[0]["f_grid_hz"] += 0.1
    assert inputs_digest(*modified_frequency) == original


def cached_failure_fixture(tmp_path):
    policy = next(row for row in scripted_family() if row["id"] == "script_zero")
    cell = {"start_day": 16, "seed": 19000011, "regd_load_bias": 0.5}
    protocol = {"registration_id": "resume-test-only"}
    destination = tmp_path / "case"
    destination.mkdir()
    trace = destination / "hardware.csv.gz"
    trace.write_bytes(b"immutable unit-test trace, not a physical rollout")
    record = {
        "evaluation_registration_id": protocol["registration_id"],
        "panel": "smoke",
        "policy": policy,
        "cell": cell,
        "outcome": {"complete": False, "qualified": False},
        "trace_hashes": {trace.name: file_hash(trace)},
    }
    write_json(destination / "episode.json", record)
    return protocol, policy, cell, destination, record


def test_resume_never_rerolls_completed_physical_failure(tmp_path, monkeypatch):
    protocol, policy, cell, destination, _ = cached_failure_fixture(tmp_path)

    def forbidden_rollout(*args, **kwargs):
        raise AssertionError("A completed physical failure must never be re-rolled")

    monkeypatch.setattr(run_confirmation, "evaluate_case", forbidden_rollout)
    before = file_hash(destination / "episode.json")
    result = run_confirmation.execute(protocol, "smoke", policy, cell, destination)
    assert not result["qualified"] and result["reused_completed_slot"]
    assert result["sha256"] == before
    assert file_hash(destination / "episode.json") == before
    assert not (destination / "attempts").exists()


def test_resume_rejects_tampered_and_duplicate_completed_evidence(tmp_path):
    protocol, policy, cell, destination, record = cached_failure_fixture(tmp_path)
    trace = destination / "hardware.csv.gz"
    original = trace.read_bytes()
    trace.write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable raw trace changed"):
        run_confirmation.completed_case(protocol, "smoke", policy, cell, destination)
    trace.write_bytes(original)
    duplicate = destination / "attempts/attempt_0001"
    duplicate.mkdir(parents=True)
    write_json(duplicate / "episode.json", record)
    with pytest.raises(ValueError, match="Multiple completed"):
        run_confirmation.completed_case(protocol, "smoke", policy, cell, destination)


def test_partial_attempts_and_old_receipts_are_preserved(tmp_path):
    directory = tmp_path / "partial"
    directory.mkdir()
    partial = directory / "hardware.csv.gz"
    partial.write_bytes(b"retained interrupted attempt")
    target = run_confirmation.next_attempt(directory)
    assert target == directory / "attempts/attempt_0001"
    target.mkdir(parents=True)
    assert (
        run_confirmation.next_attempt(directory) == directory / "attempts/attempt_0002"
    )
    assert partial.read_bytes() == b"retained interrupted attempt"
    receipt = directory / "completion.jsonl"
    receipt.write_text("retained prior receipt\n")
    fresh = run_confirmation.next_completion_log(directory)
    assert fresh.name == "completion_resume_0001.jsonl"
    assert receipt.read_text() == "retained prior receipt\n"


def test_new_record_cannot_silently_take_historical_schema(tmp_path):
    with pytest.raises(ValueError, match="requires a nested outcome"):
        audit_record(tmp_path, {}, tracking_floor_kw=100)
