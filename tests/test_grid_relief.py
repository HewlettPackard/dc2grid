import pandas as pd
import pytest

from evaluation.grid_relief import _episode_diagnostics


def test_episode_diagnostics_accounting():
    hardware = pd.DataFrame(
        {
            "bess_actual_kw": [1000.0, -2000.0],
            "bess_soc": [0.5, 0.4],
            "temp_A": [30.0, 31.0],
            "temp_B": [29.0, 30.0],
            "thermal_warning": [False, True],
            "regd_observed": [0.5, 1.0],
            "committed_actual_mw": [10.0, 20.0],
            "committed_observed_mw": [5.0, 20.0],
        }
    )
    macro = pd.DataFrame(
        {
            "duration_hours": [0.25, 0.125],
            "settlement_hours": [0.25, 0.25],
            "rate": [10.0, 20.0],
            "committed_mw": [10.0, 20.0],
            "perf_score": [0.8, 0.5],
            "regulation_revenue": [20.0, 50.0],
            "bid_accepted": [True, False],
        }
    )
    result = _episode_diagnostics(hardware, macro, 5.0)
    assert result["mean_committed_mw"] == pytest.approx(40 / 3)
    assert result["committed_mwh"] == pytest.approx(5.0)
    assert result["mean_paid_perf_score"] == pytest.approx(0.7)
    assert result["revenue_before_discount_usd"] == pytest.approx(125.0)
    assert result["tracking_discount_usd"] == pytest.approx(55.0)
    assert result["bess_discharge_mwh"] == pytest.approx(5 / 3600)
    assert result["bess_charge_mwh"] == pytest.approx(10 / 3600)
    assert result["soc_min"] == 0.4
    assert result["soc_final"] == 0.4
    assert result["thermal_warning_fraction"] == 0.5
    assert result["regd_clipping_fraction"] == 0.5
    assert result["commitment_observation_mae_mw"] == 2.5
