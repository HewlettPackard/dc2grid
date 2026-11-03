import pandas as pd
import pytest
import numpy as np

from evaluation.grid_relief import _aggregate_results, _episode_diagnostics, _metrics


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


def _trace(power):
    n = len(power)
    frame = pd.DataFrame({
        "p_fac_mw": power, "p_grid_import_mw": np.asarray(power) - 1,
        "load_mw": [100, 200, 300, 400][:n],
        "timestamp": [f"t{i}" for i in range(n)],
        "regd_observed": np.zeros(n), "work_arrived_kwh": np.ones(n),
        "work_served_kwh": np.ones(n), "backlog_work_kwh": np.zeros(n),
        "throughput_ratio": np.ones(n), "f_dev_hz": np.arange(n) * 0.01,
        "dp_act_kw": np.zeros(n), "dp_dem_kw": np.zeros(n),
        "err_kw": np.zeros(n), "throttle": np.ones(n), "backlog_kw": np.zeros(n),
    })
    frame.attrs.update(
        simulation_mode="physical_v1", expected_ticks=4, stress_threshold_mw=370,
        dt_s=5, load_median=150, lmp_base=20, lmp_slope=10, c_max=30, revenue_usd=1,
        terminated=False,
    )
    return frame


def test_physical_full_horizon_relief():
    result = _metrics(_trace([90, 100, 110, 120]), _trace([100, 110, 120, 130]))
    assert result["qualified"]
    assert result["dP_stress_mw"] == 10
    assert result["pcc_relief_stress_mw"] == 10
    assert result["SPDR_pct"] == pytest.approx(100 * 10 / 530)
    assert result["work_served_kwh"] == 4


def test_incomplete_trajectory_has_no_performance_estimate():
    with pytest.warns(RuntimeWarning, match="no common-prefix"):
        result = _metrics(_trace([90, 100]), _trace([100, 110, 120, 130]))
    assert not result["qualified"]
    assert result["expected_ticks"] == 4
    assert result["agent_ticks"] == 2
    assert np.isnan(result["dP_stress_mw"])
    assert np.isnan(result["rmse_kw"])


def test_terminated_full_length_is_not_qualified():
    agent = _trace([90, 100, 110, 120])
    agent.attrs["terminated"] = True
    with pytest.warns(RuntimeWarning, match="Unqualified"):
        result = _metrics(agent, _trace([100, 110, 120, 130]))
    assert not result["qualified"]


@pytest.mark.parametrize("column", ["timestamp", "load_mw", "regd_observed", "work_arrived_kwh"])
def test_paired_exogenous_mismatch_rejected(column):
    agent, baseline = _trace([90, 100, 110, 120]), _trace([100, 110, 120, 130])
    agent.loc[0, column] = "different" if column == "timestamp" else 900
    with pytest.raises(ValueError, match=column):
        _metrics(agent, baseline)


def test_aggregation_preserves_failed_attempt_denominator():
    rows = pd.DataFrame([
        {"controller": "sac", "seed": 1, "qualified": True, "dP_stress_mw": 10},
        {"controller": "sac", "seed": 2, "qualified": False, "dP_stress_mw": np.nan},
    ])
    result = _aggregate_results(rows, ["controller"], strict=True)
    assert result.loc["sac", "attempts"] == 2
    assert result.loc["sac", "qualified_count"] == 1
    assert np.isnan(result.loc["sac", "dP_stress_mw"])
