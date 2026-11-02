from __future__ import annotations

import copy

import pytest

# isort: off
from economy_context import PRIOR, cap_cpu, read_json
from economy_eval import ConstantResidual, evaluate_episode
from headroom_report import aggregate, audit
from panel import evaluate_case
import numpy as np
import pandas as pd
# isort: on


@pytest.mark.parametrize("command", ([0, 0], [0, 0, 1.01], [0, float("nan"), 0]))
def test_invalid_residual_is_rejected(command):
    with pytest.raises(ValueError):
        ConstantResidual(command)


def test_current_signal_gate_is_causal_and_does_not_mutate_inputs():
    actor = ConstantResidual([0, 1, 1], positive_regd_only=True)
    observation = np.zeros(24, dtype=np.float32)
    observation[6] = -0.1
    before = observation.copy()
    np.testing.assert_array_equal(actor.predict(observation)[0], [0, 1, 0])
    np.testing.assert_array_equal(observation, before)
    observation[6] = 0.1
    np.testing.assert_array_equal(actor.predict(observation)[0], [0, 1, 1])
    np.testing.assert_array_equal(actor.command, [0, 1, 1])


@pytest.mark.parametrize("macro", ("full_residual", "rule"))
def test_complete_26_hour_evaluator_matches_frozen_implementation(tmp_path, macro):
    cap_cpu()
    protocol = copy.deepcopy(read_json(PRIOR / "registration.json"))
    protocol["final_cells"] = []
    policy = {
        "id": "parity_control",
        "family": "cooling_first",
        "hardware": "cooling_first",
        "macro": macro,
    }
    cell = {
        "start_day": protocol["train_days"][0],
        "seed": 97000001,
        "regd_load_bias": 0.5,
    }
    previous = evaluate_case(protocol, policy, cell, tmp_path / "previous")
    current = evaluate_episode(protocol, policy, cell, tmp_path / "current")
    assert previous["complete"] and current["complete"]
    assert previous["qualified"] and current["qualified"]
    for key in (
        "initial",
        "service_terminal",
        "service",
        "recovery",
        "grid",
        "regulation_revenue_usd",
        "cashflow_26h_usd",
        "maximum_reference_mw_hours",
        "performance_weighted_mw_hours",
        "delivered_service_ratio",
    ):
        assert current[key] == previous[key], key
    for name in ("hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz"):
        old_frame = pd.read_csv(tmp_path / "previous" / name)
        new_frame = pd.read_csv(tmp_path / "current" / name)
        pd.testing.assert_frame_equal(new_frame, old_frame, check_exact=True)
    assert current["service"]["steps"] == 17280
    assert current["recovery"]["ledger"]["steps"] == 1440
    assert current["delivered_service_floor_required"] == (macro == "full_residual")
    assert current["decomposition"]["facility_equals_request_plus_error_minus_battery"]
    if macro == "full_residual":
        frames = {
            name: pd.read_csv(tmp_path / "current" / name)
            for name in current["trace_hashes"]
        }
        references = {
            name: pd.read_csv(tmp_path / "previous" / name)
            for name in previous["trace_hashes"]
        }
        assert audit(current, frames, references, protocol) > 200


def test_aggregation_keeps_failed_cases_and_rejects_duplicates():
    cells = [{"start_day": day, "seed": 1, "regd_load_bias": 0.5} for day in (16, 19)]
    episodes = []
    for cell in cells:
        for seed in (1, 2):
            episodes.append(
                {
                    "policy": {"id": f"seed_{seed}", "family": "fixture"},
                    "cell": cell,
                    "complete": True,
                    "qualified": True,
                    "delivered_service_gate": True,
                    "delivered_service_ratio": 1.0,
                    "cashflow_26h_usd": float(cell["start_day"] + seed),
                    "regulation_revenue_usd": 1.0,
                    "grid": {
                        "stress_facility_relief_mw": 1.0,
                        "stress_pcc_relief_mw": 1.0,
                        "actual_pcc_boundary_peak_reduction_mw": 1.0,
                    },
                    "service": {
                        "tracking_rmse_kw": 1.0,
                        "battery_throughput_mwh": 0.0,
                    },
                    "recovery": {
                        "complete": True,
                        "ledger": {"battery_throughput_mwh": 0.0},
                    },
                }
            )
    summary = aggregate(episodes, cells)["fixture"]
    assert summary["strata"]["all"]["metrics"]["cashflow_26h_usd"]["mean"] == 19.0
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(episodes + [episodes[0]], cells)
    episodes[0]["complete"] = False
    episodes[0]["qualified"] = False
    summary = aggregate(episodes, cells)["fixture"]
    assert not summary["complete_cohort"]
    assert summary["qualified_policy_cases"] == 3
    assert summary["policy_case_count"] == 4
    assert summary["strata"]["all"]["metrics"]["cashflow_26h_usd"]["mean"] is None
