import numpy as np
import pandas as pd
import pytest
from analyze_final import aggregate, capacity_floor, paired_contrasts


def fixture_cohort(missing=False):
    policies = [
        {
            "id": f"rule__guided_sac_{seed}",
            "macro": "rule",
            "family": "guided_sac",
            "available": not (missing and seed == 3),
        }
        for seed in range(4)
    ]
    cells = [
        {"start_day": day, "seed": 99, "regd_load_bias": bias}
        for day in (1, 2)
        for bias in (0.0, 0.5)
    ]
    rows = [
        {
            "panel": "sealed",
            "macro": "rule",
            "family": "guided_sac",
            "start_day": day,
            "env_seed": 99,
            "bias": bias,
            "complete_26h": True,
            "qualified": True,
            "stress_facility_relief_mw": 1 + 2 * seed + 2 * (day - 1),
        }
        for day in (1, 2)
        for bias in (0.0, 0.5)
        for seed in range(3 if missing else 4)
    ]
    return rows, {"policies": policies, "panels": {"sealed": cells}}


def test_training_seeds_are_averaged_before_case_dispersion():
    rows, registration = fixture_cohort()
    result = aggregate(rows, registration)[0]
    assert result["complete_cohort"]
    metric = result["metrics"]["stress_facility_relief_mw"]
    assert metric["mean"] == 5
    assert metric["sd"] == pytest.approx(np.sqrt(2))
    assert metric["defined_calendar_cases"] == 2


def test_unavailable_training_seed_cannot_become_survivor_mean():
    rows, registration = fixture_cohort(missing=True)
    result = aggregate(rows, registration)[0]
    assert result["admitted_models"] == 3 and result["planned_models"] == 4
    assert not result["complete_cohort"]
    assert not result["metrics"]


def test_revenue_ratio_is_formed_for_each_policy_before_averaging():
    rows = []
    for seed, unbiased, biased in [(1, 100, 90), (2, 1000, 1100)]:
        for bias, revenue in [(0, unbiased), (0.5, biased)]:
            rows.append(
                {
                    "panel": "sealed",
                    "macro": "rule",
                    "policy_id": f"rule__bc_only_{seed}",
                    "family": "bc_only",
                    "training_seed": seed,
                    "start_day": 72,
                    "env_seed": 99,
                    "bias": bias,
                    "complete_26h": True,
                    "regulation_revenue_usd": revenue,
                }
            )
    paired_contrasts(rows)
    ratios = [
        row["matched_bias_revenue_change_pct"] for row in rows if row["bias"] == 0.5
    ]
    assert ratios == pytest.approx([-10, 10])
    assert np.mean(ratios) == pytest.approx(0, abs=1e-12)


def test_capacity_floor_uses_same_admission_and_maximum_offer():
    macros = pd.DataFrame(
        [
            {
                "residual_mw": 15.0,
                "bid_accepted": True,
                "committed_mw": 14.999999,
                "perf_score": 0.95,
                "duration_hours": 0.25,
            },
            {
                "residual_mw": 18.0,
                "bid_accepted": False,
                "committed_mw": 5.0,
                "perf_score": 1.0,
                "duration_hours": 0.25,
            },
        ]
    )
    assert capacity_floor(macros) == pytest.approx(
        (14.999999 * 0.95 * 0.25 + 5 * 0.25) / (15 * 0.25 + 5 * 0.25)
    )
