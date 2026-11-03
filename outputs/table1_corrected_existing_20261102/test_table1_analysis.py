import math

from analyze_table1 import METRICS, aggregate


def fixture():
    policy = "rule_macro+pid"
    protocol = {"policies": [{"id": policy}], "panel": {"seeds": list(range(100, 105))}}
    cases = [
        {
            "policy": policy,
            "seed": 100 + index,
            "complete_service": True,
            "qualified": True,
            "metrics": {key: float(index) for key in METRICS},
            "mean_committed_mw": 10.0,
            "closed_26h_cash_usd": float(index),
            "battery_throughput_mwh": 1.0,
            "service_end_soc": 0.5,
        }
        for index in range(5)
    ]
    return protocol, cases


def test_all_five_cases_and_sample_sd_are_used():
    protocol, cases = fixture()
    result = aggregate(cases, protocol)[0]
    assert result["metrics"]["tracking_rmse_kw"]["mean"] == 2.0
    assert math.isclose(result["metrics"]["tracking_rmse_kw"]["sd"], math.sqrt(2.5))
    assert result["qualified"] == 5


def test_qualification_failure_does_not_create_survivor_means():
    protocol, cases = fixture()
    cases[-1]["qualified"] = False
    result = aggregate(cases, protocol)[0]
    assert result["metrics"]["tracking_rmse_kw"]["mean"] == 2.0
    assert result["qualified"] == 4
    assert result["diagnostic_not_admissible"]


def test_incomplete_service_suppresses_every_primary_estimate():
    protocol, cases = fixture()
    cases[-1]["complete_service"] = False
    cases[-1]["qualified"] = False
    cases[-1]["closed_26h_cash_usd"] = None
    cases[-1]["metrics"] = {key: None for key in METRICS}
    result = aggregate(cases, protocol)[0]
    assert result["complete_service"] == 4
    assert not result["all_primary_means_available"]
    assert all(
        value == {"mean": None, "sd": None} for value in result["metrics"].values()
    )
    assert result["mean_cash_26h_usd"] is None
