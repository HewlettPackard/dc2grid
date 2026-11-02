# isort: off
from loan_policy import ExpectedSurplusQuantity, PRIOR
from loan_eval import evaluate
from study import read_json
from panel import evaluate_case
from market_policy import maximum_native_quantity
import numpy as np
import pandas as pd
# isort: on


def test_new_runner_exactly_preserves_the_full_residual_control(tmp_path):
    base = read_json(PRIOR / "registration.json")
    policy = {
        "id": "analytic_full",
        "hardware": "cooling_first",
        "macro": "full_residual",
    }
    cell = {"start_day": 16, "seed": 730003, "regd_load_bias": 0.5}
    old = evaluate_case(base, policy, cell, tmp_path / "old")
    new = evaluate(policy, cell, base, tmp_path / "new", "unit-test")
    for key in (
        "grid",
        "service",
        "service_checks",
        "complete",
        "qualified",
        "recovery",
        "cashflow_26h_usd",
        "delivered_service_ratio",
    ):
        assert old[key] == new[key], key
    for name in ("hardware.csv.gz", "recovery.csv.gz"):
        a = pd.read_csv(tmp_path / "old" / name)
        b = pd.read_csv(tmp_path / "new" / name)
        pd.testing.assert_frame_equal(a[b.columns], b, check_exact=True)


def test_future_credit_is_never_booked_and_vanishes_at_last_window():
    base = read_json(PRIOR / "registration.json")
    forecast = read_json(PRIOR / "forecast_v1/report.json")
    actor = ExpectedSurplusQuantity(
        {
            "forecaster": "ridge",
            "margin_threshold_usd_mw_h": 0,
            "forecast_checkpoint": forecast["model"],
        },
        base,
    )
    macro, hardware = np.zeros(19, np.float32), np.zeros(18, np.float32)
    macro[17], macro[18] = 0.01, 0.5
    hardware[6], hardware[7] = -1, 0.5
    action = actor.choose(macro, hardware)
    assert 5 <= float(action[0]) * 30 < 5.001
    assert actor.delivered == 0 and actor.maximum == 0
    assert actor.last_diagnostics["expected_future_credit_mw_hours"] > 0
    actor.window = 95
    action = actor.choose(macro, hardware)
    assert float(action[0]) * 30 > 14.9
    assert actor.last_diagnostics["expected_future_credit_mw_hours"] == 0


def test_rolling_conditional_credit_handles_multiple_zero_delivery_withholds():
    base = read_json(PRIOR / "registration.json")
    forecast = read_json(PRIOR / "forecast_v1/report.json")
    actor = ExpectedSurplusQuantity(
        {
            "forecaster": "ridge",
            "margin_threshold_usd_mw_h": 0,
            "forecast_checkpoint": forecast["model"],
        },
        base,
    )
    macro, hardware = np.zeros(19, np.float32), np.zeros(18, np.float32)
    macro[17], macro[18] = 0.01, 1.0
    hardware[6], hardware[7] = -1, 0.5
    withholds = 0
    for _ in range(96):
        action = actor.choose(macro, hardware)
        withheld = actor.last_diagnostics["quantity_reduced"]
        withholds += int(withheld)
        actor.observe(
            {
                "residual_mw": 30.0,
                "bid_accepted": True,
                "duration_hours": 0.25,
                "committed_mw": float(action[0]) * 30,
                "perf_score": 0.0 if withheld else 0.98,
            },
            -1,
        )
        remaining = (96 - actor.window) * 0.25
        conditional_balance = (
            actor.delivered - 0.92 * actor.maximum + (0.98 - 0.92) * 5 * remaining
        )
        assert conditional_balance >= -1e-9
    assert withholds >= 2
    assert actor.maximum == maximum_native_quantity(30) * 24
    assert actor.delivered / actor.maximum >= 0.92
