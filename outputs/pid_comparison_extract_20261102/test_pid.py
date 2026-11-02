"""Verify PID variants, historical parity, matched inputs and failure visibility."""

from __future__ import annotations

import copy

# isort: off
from pid_context import CONFIRMATION, ECONOMIC, PROJECT, cap_cpu, case_key, read_json
from pid_reference import FullWorkPID, execute, hardware_factory
from audit_pid import METRICS, audit_episode, summarize
from baselines.pid_controller import PIDController
import numpy as np
import pandas as pd
import pytest
# isort: on


@pytest.fixture(scope="module")
def fixture_protocol():
    cap_cpu(32)
    protocol = copy.deepcopy(read_json(ECONOMIC / "track_b_fraction/registration.json"))
    protocol["registration_id"] = "pid-reference-known-fixture-only"
    protocol["confirmation_registration_id"] = "known-data-parity-test"
    protocol["role"] = "unit_test_on_already_consumed_case"
    protocol["policies"] = [
        {
            "id": "pid_native",
            "family": "pid_native",
            "hardware": "pid",
            "macro": "full_residual",
            "full_work": False,
        },
        {
            "id": "pid_full_work",
            "family": "pid_full_work",
            "hardware": "pid",
            "macro": "full_residual",
            "full_work": True,
        },
    ]
    protocol["panels"] = {
        "history": {
            "cells": [
                {"start_day": 207, "seed": 100, "regd_load_bias": 0.5},
            ]
        }
    }
    return protocol


def test_full_work_wrapper_changes_only_executed_throttle():
    native = PIDController()
    full_work = FullWorkPID(PIDController())
    observation = np.zeros(18, dtype=np.float32)
    observation[0:2] = 30 / 35
    observation[2] = 0.5
    observation[6] = 1.0
    observation[17] = 1.0
    first, _ = native.predict(observation)
    second, _ = full_work.predict(observation)
    assert first[0] < 1 and second[0] == 1
    np.testing.assert_array_equal(first[1:], second[1:])
    assert full_work.base.bess_gain == 6
    assert full_work.base.flex_gain == 0.12
    assert full_work.base.cool_gain == 0.10
    native.reset()
    full_work.reset()
    np.testing.assert_array_equal(
        native.predict(observation)[0][1:],
        full_work.predict(observation)[0][1:],
    )


def test_factory_does_not_replace_native_pid(fixture_protocol):
    native, fixed = fixture_protocol["policies"]
    assert type(hardware_factory(native, fixture_protocol)) is PIDController
    wrapper = hardware_factory(fixed, fixture_protocol)
    assert type(wrapper) is FullWorkPID and type(wrapper.base) is PIDController
    with pytest.raises(ValueError):
        hardware_factory({**native, "hardware": "unregistered"}, fixture_protocol)


@pytest.mark.parametrize("policy_index", [0, 1])
def test_complete_known_case_and_frozen_pairing(
    policy_index,
    fixture_protocol,
    tmp_path,
):
    cell = fixture_protocol["panels"]["history"]["cells"][0]
    policy = fixture_protocol["policies"][policy_index]
    result = execute(fixture_protocol, "history", policy, cell, tmp_path / "case")
    outcome = read_json(result["path"])
    prior = read_json(CONFIRMATION / "reports/development_cases.json")
    expected = next(
        row["inputs_sha256"]
        for row in prior
        if row["panel"] == "regression"
        and row["policy_id"] == "script_zero"
        and row["cell"] == cell
    )
    audit = audit_episode(
        tmp_path / "case",
        outcome,
        fixture_protocol,
        expected_inputs=expected,
    )
    assert audit["qualified"] == outcome["qualified"]
    service = pd.read_csv(tmp_path / "case/hardware.csv.gz")
    if policy["full_work"]:
        np.testing.assert_array_equal(service.throttle_batch, np.ones(len(service)))
    else:
        original = read_json(
            PROJECT
            / "outputs/table2_distress_rl_20261031/final_regression"
            / "full_residual__pid"
            / case_key(cell)
            / "episode.json",
        )
        assert outcome["complete"] == original["complete"]
        np.testing.assert_allclose(
            outcome["cashflow_26h_usd"],
            original["cashflow_26h_usd"],
            rtol=0,
            atol=1e-6,
        )
        for key in ("stress_facility_relief_mw", "stress_pcc_relief_mw"):
            np.testing.assert_allclose(
                outcome["grid"][key],
                original["grid"][key],
                rtol=0,
                atol=1e-8,
            )
        assert service.throttle_batch.min() < 1
    before = result["sha256"]
    reused = execute(fixture_protocol, "history", policy, cell, tmp_path / "case")
    assert reused["reused_completed_slot"] and reused["sha256"] == before


def test_failed_complete_cohort_is_not_hidden(fixture_protocol):
    protocol = copy.deepcopy(fixture_protocol)
    protocol["policies"] = protocol["policies"][:1]
    protocol["panels"] = {
        "history": {
            "cells": [
                {"start_day": 207, "seed": seed, "regd_load_bias": bias}
                for seed in (100, 101)
                for bias in (0.0, 0.5)
            ]
        }
    }
    records = [
        {
            "panel": "history",
            "policy_id": "pid_native",
            "cell": cell,
            "complete": True,
            "qualified": cell["seed"] == 100,
            "metrics": dict.fromkeys(METRICS, 1.0 if cell["seed"] == 100 else 3.0),
            "failure_reasons": [] if cell["seed"] == 100 else ["service_floor"],
        }
        for cell in protocol["panels"]["history"]["cells"]
    ]
    result = summarize(records, protocol)["history"]["0.5"]["pid_native"]
    assert result["qualified"] == 1 and result["episodes"] == 2
    assert result["metrics"]["cashflow_26h_usd"]["mean"] == 2
    assert result["metrics"]["cashflow_26h_usd"]["admissible_mean"] is None
    records[-1]["complete"] = False
    result = summarize(records, protocol)["history"]["0.5"]["pid_native"]
    assert result["metrics"]["cashflow_26h_usd"]["mean"] is None


def test_new_reference_panels_do_not_relabel_the_primary_study():
    prior = read_json(CONFIRMATION / "registration.json")
    result = read_json(CONFIRMATION / "reports/final_report.json")
    assert len(prior["policies"]) == 43
    assert result["episode_count"] == 9460
    assert all(
        row["id"] not in {"pid_native", "pid_full_work"} for row in prior["policies"]
    )
