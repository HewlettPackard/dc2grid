# isort: off
from forecast import episode_examples, fit_ridge, predict_ridge
from market_policy import BudgetedQuantity
import numpy as np
# isort: on


def test_forecast_features_do_not_read_the_current_future_window():
    history = np.zeros((17280, 24), dtype=np.float32)
    history[:, 23] = 1 - np.arange(17280) / 17280
    history[:, 6] = np.sin(np.arange(17280) / 501)
    x, y = episode_examples(history)
    changed = history.copy()
    changed[181:360, 6] += 0.25
    altered_x, altered_y = episode_examples(changed)
    np.testing.assert_array_equal(x[1], altered_x[1])
    assert y[1] != altered_y[1]
    assert x[2, 1] != altered_x[2, 1]
    assert y[-1] == np.mean(history[-180:, 6], dtype=np.float64)


def test_frozen_ridge_batch_and_single_features_match():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(400, 6))
    y = 0.7 * x[:, 0] + 0.2 * x[:, 1]
    model = fit_ridge(x, y, 10)
    for index in range(10):
        np.testing.assert_allclose(
            predict_ridge(model, x)[index], predict_ridge(model, x[index]), atol=1e-14
        )


def test_quantity_rule_cannot_spend_unobserved_future_credit():
    protocol = {
        "high_level": {
            "margin_thresholds_usd_mw_h": [0, 5],
            "guard_ratio": 0.92,
            "decay_rho": 0.9963,
            "rmcp_normalizer": 100,
            "planning_performance": 0.98,
        }
    }
    controller = BudgetedQuantity(
        {"forecaster": "decay", "margin_threshold_usd_mw_h": 0}, protocol
    )
    macro = np.zeros(19, dtype=np.float32)
    macro[17], macro[18] = 0.1, 0.5
    hardware = np.zeros(18, dtype=np.float32)
    hardware[6], hardware[7] = -1, 0.5
    full = controller.choose(macro, hardware)
    assert float(full[0]) * 30 > 14.9
    controller.delivered, controller.maximum = 50, 50
    reduced = controller.choose(macro, hardware)
    assert 5 <= float(reduced[0]) * 30 < 5.001
    assert controller.last_diagnostics["quantity_reduced"]
    controller.observe(
        {
            "residual_mw": 15,
            "bid_accepted": False,
            "duration_hours": 0.25,
            "committed_mw": 5,
            "perf_score": 1,
        },
        -0.8,
    )
    assert controller.maximum == 51.25
    assert controller.delivered == 51.25
    assert controller.previous_mean == -0.8
