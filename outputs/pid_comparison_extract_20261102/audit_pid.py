"""Audit matched PID references, retaining complete-but-disqualified diagnostics."""

from __future__ import annotations

from pathlib import Path

# isort: off
from pid_context import CONFIRMATION, case_key, file_hash, read_json
from confirmation_audit import close, inputs_digest, peak_reductions
from evaluate_low import audit_primary
from market_policy import maximum_native_quantity
import numpy as np
import pandas as pd
# isort: on

METRICS = (
    "stress_facility_relief_mw",
    "stress_pcc_relief_mw",
    "cashflow_26h_usd",
    "tracking_rmse_kw",
    "battery_throughput_26h_mwh",
    "actual_pcc_boundary_peak_reduction_mw",
    "pcc_peak_reduction_26h_mw",
    "delivered_service_ratio",
    "mean_throttle",
    "maximum_backlog_kwh",
    "service_terminal_backlog_kwh",
    "recovery_terminal_backlog_kwh",
)


def audit_episode(directory, outcome, protocol, expected_inputs=None):
    directory = Path(directory)
    if outcome["study_registration_id"] != protocol["registration_id"]:
        raise ValueError("Wrong PID study identity")
    checks = audit_primary(directory, outcome)
    complete = bool(outcome["complete"] and outcome["recovery"]["complete"])
    if not complete:
        return {
            "complete": False,
            "qualified": False,
            "raw_checks": checks,
            "metrics": dict.fromkeys(METRICS),
            "inputs_sha256": None,
            "failure_reasons": [
                key for key, value in outcome["service_checks"].items() if not value
            ]
            + (["recovery_incomplete"] if not outcome["recovery"]["complete"] else []),
        }
    service = pd.read_csv(directory / "hardware.csv.gz")
    recovery = pd.read_csv(directory / "recovery.csv.gz")
    market = pd.read_csv(directory / "macro.csv.gz")
    if (market.committed_mw < 5).any():
        raise AssertionError("The fixed 100 kW fee floor must not exceed commitment")
    full = pd.concat([service, recovery], ignore_index=True)
    close(full.tick, np.arange(1, 18721), tolerance=0)
    close(full.duration_hours, 5 / 3600, tolerance=1e-15)
    if outcome["policy"]["full_work"]:
        close(service.throttle_batch, 1.0, tolerance=0)
    close(recovery.throttle_batch, 1.0, tolerance=0)
    close(recovery.pump_speed_A, 0.7, tolerance=1e-7)
    close(recovery.hvac_effort, 0.7, tolerance=1e-7)
    close(recovery.committed_mw, 0.0, tolerance=0)
    checks += 7
    expected_times = (
        pd.Timestamp("2023-01-01")
        + pd.Timedelta(days=outcome["cell"]["start_day"])
        + pd.to_timedelta(np.arange(18720) * 5, unit="s")
    )
    if not np.array_equal(pd.to_datetime(full.timestamp), expected_times):
        raise AssertionError("PID timestamps do not match the fixed case")
    offers = np.array([maximum_native_quantity(x) for x in market.residual_mw])
    accepted = market.bid_accepted.to_numpy(dtype=bool)
    close(market.bid_price, 0.0, tolerance=0)
    close(market.bid_mw, offers, tolerance=2e-6)
    close(market.rate, np.where(accepted, market.rmcp_usd, 5.0))
    close(market.committed_mw, np.where(accepted, market.bid_mw, 5.0))
    market["reference_capacity_mw"] = np.where(accepted, offers, 5.0)
    delivered = float(
        np.dot(market.committed_mw * market.perf_score, market.duration_hours)
    )
    maximum = float(np.dot(market.reference_capacity_mw, market.duration_hours))
    ratio = delivered / maximum
    close(delivered, outcome["performance_weighted_mw_hours"], tolerance=1e-6)
    close(maximum, outcome["maximum_reference_mw_hours"], tolerance=1e-6)
    close(ratio, outcome["delivered_service_ratio"])
    if bool(ratio >= protocol["service_floor"]) != outcome["delivered_service_gate"]:
        raise AssertionError("PID service gate is not the measured actual ratio")
    checks += 9
    last_service, last = service.iloc[-1], recovery.iloc[-1]
    gate = protocol["service_gate"]
    service_checks = {
        "complete": not bool(service.terminated.any()),
        "terminal_soc": bool(last_service.bess_soc >= gate["min_terminal_soc"]),
        "terminal_work": bool(
            last_service.backlog_work_kwh
            <= gate["max_backlog_fraction"] * service.work_arrived_kwh.sum() + 1e-6
        ),
        "thermal_warnings": bool(
            service.thermal_warning.mean() <= gate["max_thermal_warning_fraction"]
        ),
    }
    if service_checks != outcome["service_checks"]:
        raise AssertionError("PID physical service gates disagree with raw traces")
    config = protocol["recovery"]
    closure_checks = {
        "soc_closed": abs(last.bess_soc - outcome["initial"]["soc"])
        <= config["soc_tolerance"],
        "work_closed": last.backlog_work_kwh <= config["backlog_tolerance_kwh"],
        "thermal_A_closed": abs(last.temp_A - last.reference_temp_A)
        <= config["temperature_tolerance_C"],
        "thermal_B_closed": abs(last.temp_B - last.reference_temp_B)
        <= config["temperature_tolerance_C"],
    }
    if any(
        bool(value) != outcome["recovery"][key] for key, value in closure_checks.items()
    ):
        raise AssertionError("PID recovery closure differs from the ledger")
    closed = bool(all(closure_checks.values()) and not recovery.terminated.any())
    physical = bool(all(service_checks.values()) and closed)
    qualified = bool(physical and ratio >= protocol["service_floor"])
    if physical != outcome["physical_qualified"] or qualified != outcome["qualified"]:
        raise AssertionError(
            "PID qualification was not derived from all declared gates"
        )
    work_error = (
        outcome["initial"]["backlog_kwh"]
        + full.work_arrived_kwh.sum()
        - full.work_served_kwh.sum()
        - last.backlog_work_kwh
    )
    if abs(work_error) >= config["work_conservation_tolerance_kwh"]:
        raise AssertionError("PID work conservation failed")
    checks += 4
    peaks = peak_reductions(service, recovery)
    close(
        peaks["pcc_peak_reduction_24h_mw"],
        outcome["grid"]["actual_pcc_boundary_peak_reduction_mw"],
    )
    battery = float(np.dot(full.bess_actual_kw.abs() / 1000, full.duration_hours))
    close(
        battery,
        outcome["service"]["battery_throughput_mwh"]
        + outcome["recovery"]["ledger"]["battery_throughput_mwh"],
    )
    digest = inputs_digest(service, recovery, market)
    if expected_inputs is not None and digest != expected_inputs:
        raise AssertionError("PID and frozen confirmation received different inputs")
    failure_reasons = [key for key, value in service_checks.items() if not value]
    failure_reasons += [key for key, value in closure_checks.items() if not value]
    if ratio < protocol["service_floor"]:
        failure_reasons.append("paid_service_below_90_percent")
    return {
        "complete": True,
        "qualified": qualified,
        "physical_qualified": physical,
        "raw_checks": checks + 3,
        "inputs_sha256": digest,
        "failure_reasons": failure_reasons,
        "work_conservation_error_kwh": float(work_error),
        "metrics": {
            "stress_facility_relief_mw": outcome["grid"]["stress_facility_relief_mw"],
            "stress_pcc_relief_mw": outcome["grid"]["stress_pcc_relief_mw"],
            "cashflow_26h_usd": outcome["cashflow_26h_usd"],
            "tracking_rmse_kw": outcome["service"]["tracking_rmse_kw"],
            "battery_throughput_26h_mwh": battery,
            "actual_pcc_boundary_peak_reduction_mw": peaks["pcc_peak_reduction_24h_mw"],
            "pcc_peak_reduction_26h_mw": peaks["pcc_peak_reduction_26h_mw"],
            "delivered_service_ratio": ratio,
            "mean_throttle": float(service.throttle_batch.mean()),
            "maximum_backlog_kwh": float(service.backlog_work_kwh.max()),
            "service_terminal_backlog_kwh": float(last_service.backlog_work_kwh),
            "recovery_terminal_backlog_kwh": float(last.backlog_work_kwh),
        },
        "grid_metrics": outcome["grid"],
        "peak_phases": {key: value for key, value in peaks.items() if "phase" in key},
    }


def frozen_inputs():
    cases = read_json(CONFIRMATION / "reports/final_cases.json")
    return {
        (row["panel"], case_key(row["cell"])): row["inputs_sha256"]
        for row in cases
        if row["policy_id"] == "script_zero"
    }


def audit_slot(protocol, artifact, expected_inputs):
    path = Path(artifact["path"])
    if file_hash(path) != artifact["sha256"]:
        raise ValueError("PID episode changed after completion")
    outcome = read_json(path)
    policy = next(
        row for row in protocol["policies"] if row["id"] == artifact["policy_id"]
    )
    if outcome["policy"] != policy or outcome["cell"] != artifact["cell"]:
        raise ValueError("PID slot identity mismatch")
    return {
        **artifact,
        "family": policy["family"],
        **audit_episode(path.parent, outcome, protocol, expected_inputs),
    }


def summarize(records, protocol):
    summary = {}
    for panel in protocol["panels"]:
        summary[panel] = {}
        for bias in (0.0, 0.5):
            summary[panel][str(bias)] = {}
            for policy in protocol["policies"]:
                rows = [
                    row
                    for row in records
                    if row["panel"] == panel
                    and row["policy_id"] == policy["id"]
                    and row["cell"]["regd_load_bias"] == bias
                ]
                expected = sum(
                    cell["regd_load_bias"] == bias
                    for cell in protocol["panels"][panel]["cells"]
                )
                if not rows or len(rows) != expected:
                    raise ValueError("A PID stratum is missing planned cases")
                complete = all(row["complete"] for row in rows)
                qualified = all(row["qualified"] for row in rows)
                metrics = {
                    name: {
                        "mean": float(np.mean([row["metrics"][name] for row in rows]))
                        if complete
                        else None,
                        "sd": float(
                            np.std([row["metrics"][name] for row in rows], ddof=1)
                        )
                        if complete
                        else None,
                        "admissible_mean": float(
                            np.mean([row["metrics"][name] for row in rows])
                        )
                        if qualified
                        else None,
                    }
                    for name in METRICS
                }
                summary[panel][str(bias)][policy["id"]] = {
                    "episodes": len(rows),
                    "complete": sum(row["complete"] for row in rows),
                    "qualified": sum(row["qualified"] for row in rows),
                    "all_qualified": qualified,
                    "all_complete": complete,
                    "metrics": metrics,
                    "complete_but_failed_means_are_diagnostic_only": not qualified,
                    "failure_counts": {
                        reason: sum(reason in row["failure_reasons"] for row in rows)
                        for reason in sorted(
                            {
                                reason
                                for row in rows
                                for reason in row["failure_reasons"]
                            }
                        )
                    },
                }
    return summary
