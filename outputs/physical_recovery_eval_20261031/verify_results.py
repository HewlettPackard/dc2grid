"""Independent raw-ledger readback and post-test diagnostics; never select policies."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter

# isort: off
from run_evaluation import ROOT, load_json, verify_registration
from recovery import matched
import numpy as np
import pandas as pd
# isort: on


def main():
    os.sched_setaffinity(0, {59})
    registration = load_json(ROOT / "heldout_registration.json")
    verify_registration(registration)
    selection = load_json(ROOT / "selection.json")
    reports = ROOT / "heldout/reports"
    episodes = load_json(reports / "episodes.json")
    comparisons = load_json(reports / "comparisons.json")
    readback = load_json(reports / "readback.json")
    lookup = {(row["policy_id"], row["cell"]["id"]): row for row in episodes}
    expected = {
        (policy["id"], cell["id"])
        for policy in registration["policies"]
        for cell in registration["cells"]
    }
    assert len(lookup) == len(episodes) == len(expected) == 600
    assert set(lookup) == expected
    assert sum("recovery" in row for row in episodes) == readback["recorded_rollouts"]
    assert sum(row["status"] == "unavailable" for row in episodes) == 120
    assert sum(bool(row.get("constraint_qualified")) for row in episodes) == 412

    failure_rows = []
    gate = registration["selection_gate"]
    for row in episodes:
        if "recovery" not in row:
            continue
        metrics = row["observed_hardware_metrics"]
        reasons = []
        if not row["complete"]:
            reasons.append("incomplete_service")
        if metrics["terminal_soc"] < gate["min_terminal_soc"]:
            reasons.append("terminal_soc")
        if metrics["thermal_warning_fraction"] > gate["max_thermal_warning_fraction"]:
            reasons.append("thermal_warning_fraction")
        if (
            metrics["terminal_backlog_kwh"]
            > gate["max_backlog_fraction"] * metrics["work_arrived_kwh"] + 1e-6
        ):
            reasons.append("terminal_work_debt")
        if abs(metrics["work_conservation_error_kwh"]) >= 1e-4:
            reasons.append("work_conservation")
        assert row["constraint_qualified"] == (not reasons)
        if reasons:
            failure_rows.append(
                {
                    "policy_id": row["policy_id"],
                    "cell_id": row["cell"]["id"],
                    "reasons": ",".join(reasons),
                    "terminal_soc": metrics["terminal_soc"],
                    "thermal_warning_fraction": metrics["thermal_warning_fraction"],
                    "terminal_backlog_kwh": metrics["terminal_backlog_kwh"],
                    "recovery_closed": row["recovery"]["closed"],
                }
            )
    pd.DataFrame(failure_rows).to_csv(
        reports / "qualification_failures.csv", index=False
    )

    participation = []
    for stage in ("development", "heldout"):
        stage_registration = load_json(ROOT / f"{stage}_registration.json")
        for policy in stage_registration["policies"]:
            if not policy["available"]:
                continue
            frames = [
                pd.read_csv(
                    ROOT
                    / stage
                    / "traces"
                    / policy["id"]
                    / cell["id"]
                    / "macro.csv.gz",
                    float_precision="round_trip",
                )
                for cell in stage_registration["cells"]
            ]
            frame = pd.concat(frames, ignore_index=True)
            assert (frame.rmcp_usd > 0).all()
            participation.append(
                {
                    "stage": stage,
                    "policy_id": policy["id"],
                    "episodes": len(frames),
                    "macro_intervals": len(frame),
                    "accepted_intervals": int(frame.bid_accepted.sum()),
                    "acceptance_fraction": float(frame.bid_accepted.mean()),
                    "mean_committed_mw": float(
                        np.average(frame.committed_mw, weights=frame.duration_hours)
                    ),
                    "mean_bid_mw": float(frame.bid_mw.mean()),
                    "mean_bid_price": float(frame.bid_price.mean()),
                    "mean_posted_rmcp": float(frame.rmcp_usd.mean()),
                    "mean_bid_price_over_rmcp": float(
                        (frame.bid_price / frame.rmcp_usd).mean()
                    ),
                    "fraction_bid_price_above_rmcp": float(
                        (frame.bid_price > frame.rmcp_usd).mean()
                    ),
                    "mean_paid_performance": float(frame.perf_score.mean()),
                    "mean_episode_regulation_revenue_usd": math.fsum(
                        frame.regulation_revenue
                    )
                    / len(frames),
                }
            )
    pd.DataFrame(participation).to_csv(
        reports / "market_participation.csv", index=False
    )

    primary_ids = {
        "nominal",
        "pid",
        selection["primary_hardware"],
        selection["primary_macro_pair"],
        selection["macro_exact_hardware_reference"],
    }
    columns = [
        "timestamp",
        "duration_hours",
        "grid_load_mw",
        "regd_signal",
        "T_amb",
        "lmp",
        "p_grid_import_mw",
        "p_facility_mw",
        "bess_actual_kw",
        "work_arrived_kwh",
        "work_served_kwh",
        "backlog_work_kwh",
        "bess_soc",
        "terminated",
        "truncated",
    ]
    raw_facts = {}
    numeric_checks = 0

    def near(actual, expected):
        nonlocal numeric_checks
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6), (
            actual,
            expected,
        )
        numeric_checks += 1

    for policy_id in sorted(primary_ids):
        for cell in registration["cells"]:
            row = lookup[policy_id, cell["id"]]
            assert row["complete"] and row["recovery"]["complete"]
            directory = ROOT / "heldout/traces" / policy_id / cell["id"]
            service, recovery = [
                pd.read_csv(
                    directory / filename, usecols=columns, float_precision="round_trip"
                )
                for filename in ("hardware.csv.gz", "recovery.csv.gz")
            ]
            assert len(service) == 17280 and len(recovery) == 1440
            combined = pd.concat([service, recovery], ignore_index=True)
            timestamps = np.asarray(combined.timestamp, dtype="datetime64[ns]")
            assert (np.diff(timestamps) == np.timedelta64(5, "s")).all()
            assert not combined.terminated.any()
            assert service.truncated.sum() == recovery.truncated.sum() == 1
            assert service.truncated.iloc[-1] and recovery.truncated.iloc[-1]
            np.testing.assert_allclose(
                combined.p_grid_import_mw,
                combined.p_facility_mw - combined.bess_actual_kw / 1000,
                rtol=1e-12,
                atol=1e-8,
            )
            cost_service = math.fsum(
                service.lmp * service.p_grid_import_mw * service.duration_hours
            )
            cost_recovery = math.fsum(
                recovery.lmp * recovery.p_grid_import_mw * recovery.duration_hours
            )
            throughput = math.fsum(
                combined.bess_actual_kw.abs() * combined.duration_hours / 1000
            )
            macro = pd.read_csv(
                directory / "macro.csv.gz", float_precision="round_trip"
            )
            revenue = math.fsum(macro.regulation_revenue)
            near(cost_service, row["electricity_cost_usd"])
            near(
                cost_recovery,
                row["recovery"]["observed_energy_metrics"]["electricity_cost_usd"],
            )
            near(
                cost_service + cost_recovery,
                row["recovery"]["total_observed_energy_metrics"][
                    "electricity_cost_usd"
                ],
            )
            near(cost_service, math.fsum(macro.electricity_cost))
            near(
                throughput,
                row["recovery"]["total_observed_energy_metrics"][
                    "battery_throughput_mwh"
                ],
            )
            near(revenue, row["revenue_usd"])
            near(
                math.fsum(combined.work_arrived_kwh - combined.work_served_kwh),
                combined.backlog_work_kwh.iloc[-1],
            )
            near(recovery.bess_soc.iloc[-1], row["initial"]["soc"])
            near(math.fsum(service.duration_hours), 24.0)
            near(math.fsum(recovery.duration_hours), 2.0)
            near(service.bess_soc.iloc[-1], row["terminal"]["soc"])
            near(
                service.bess_soc.iloc[-1],
                row["observed_hardware_metrics"]["terminal_soc"],
            )
            signature = hashlib.sha256(timestamps.tobytes())
            for key in (
                "grid_load_mw",
                "lmp",
                "regd_signal",
                "T_amb",
                "work_arrived_kwh",
            ):
                signature.update(combined[key].to_numpy(dtype=np.float64).tobytes())
            raw_facts[policy_id, cell["id"]] = {
                "cost_usd": cost_service + cost_recovery,
                "revenue_usd": revenue,
                "throughput_mwh": throughput,
                "exogenous_sha256": signature.hexdigest(),
                "modeled_system_facility_peak_mw": float(
                    (combined.grid_load_mw + combined.p_facility_mw).max()
                ),
                "modeled_system_pcc_peak_mw": float(
                    (combined.grid_load_mw + combined.p_grid_import_mw).max()
                ),
            }
    peaks = []
    for comparison in comparisons:
        if not (
            (
                comparison["a"] == selection["primary_hardware"]
                and comparison["contrast"] == "low_vs_pid"
            )
            or (
                comparison["a"] == selection["primary_macro_pair"]
                and comparison["contrast"] == "macro_vs_matched_rule"
            )
        ):
            continue
        cell_id = comparison["cell"]["id"]
        a, b = raw_facts[comparison["a"], cell_id], raw_facts[comparison["b"], cell_id]
        assert a["exogenous_sha256"] == b["exogenous_sha256"]
        near(
            a["revenue_usd"] - b["revenue_usd"] - (a["cost_usd"] - b["cost_usd"]),
            comparison["cashflow_delta_after_closed_recovery_usd"],
        )
        near(
            a["throughput_mwh"] - b["throughput_mwh"],
            comparison["total_battery_throughput_delta_mwh"],
        )
        peaks.append(
            {
                "a": comparison["a"],
                "b": comparison["b"],
                "cell_id": cell_id,
                "phase": "supplemental_complete_26h_window",
                "modeled_facility_boundary_system_peak_reduction_mw": b[
                    "modeled_system_facility_peak_mw"
                ]
                - a["modeled_system_facility_peak_mw"],
                "modeled_pcc_boundary_system_peak_reduction_mw": b[
                    "modeled_system_pcc_peak_mw"
                ]
                - a["modeled_system_pcc_peak_mw"],
            }
        )
    pd.DataFrame(peaks).to_csv(
        reports / "primary_26h_peak_comparisons.csv", index=False
    )
    verification = {
        "registration_id": registration["registration_id"],
        "planned_policy_cell_keys_verified": len(expected),
        "qualification_flags_recomputed": readback["recorded_rollouts"],
        "qualification_failures": len(failure_rows),
        "failure_criteria": dict(Counter(row["reasons"] for row in failure_rows)),
        "primary_raw_26h_episodes_audited": len(raw_facts),
        "raw_ledger_scalar_checks": numeric_checks,
        "paired_primary_cashflow_and_throughput_checks": len(peaks),
        "frozen_source_and_checkpoint_verification": "passed",
        "extra_policy_rollouts": 0,
        "training_steps": 0,
        "auditor_sha256": matched.file_hash(__file__),
        "interpretation_limits": [
            "No selection changes were made after seeing held-out outcomes.",
            "The primary macro's low RMSE accompanies much lower participation; "
            "it is not an overall success.",
            "Bid/rejection diagnostics describe the observed policy; they do not "
            "prove that repricing alone fixes the hierarchy.",
            "Supplemental 26h peaks do not replace registered 24h stress metrics.",
            "These are isolated 26h episodes, not continuous multi-day operation; "
            "adjacent dates can overlap in their service/recovery windows.",
            "The five seeds repeat across dates; sample SD is descriptive, "
            "not evidence of 15 independent calendar conditions.",
            "After guiding further development, these peak dates cannot again "
            "be called an untouched test set.",
        ],
    }
    verify_registration(registration)
    matched.write_json(reports / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
