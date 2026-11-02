"""Independent raw physical, settlement, allocation, and pairing checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

# isort: off
from confirmation_context import file_hash
from evaluate_low import audit_primary
from headroom_report import METRICS as PREVIOUS_METRICS, episode_metrics
from market_policy import maximum_native_quantity
import numpy as np
import pandas as pd
# isort: on

METRICS = (
    *PREVIOUS_METRICS,
    "actual_facility_boundary_peak_reduction_mw",
    "pcc_peak_reduction_26h_mw",
    "facility_peak_reduction_26h_mw",
    "delivered_service_ratio",
    "stress_pcc_import_mw",
    "electricity_26h_usd",
    "fraction_mean",
    "fraction_zero_share",
    "fraction_one_share",
)
EXOGENOUS_COLUMNS = (
    "tick",
    "grid_load_mw",
    "lmp",
    "T_amb",
    "p_reference_mw",
    "reference_temp_A",
    "reference_temp_B",
    "work_arrived_kwh",
    "regd_signal",
    "load_anomaly",
)
MARKET_COLUMNS = (
    "macro_tick",
    "committed_mw",
    "bid_mw",
    "bid_price",
    "bid_accepted",
    "rate",
    "rmcp_usd",
    "residual_mw",
    "maximum_reference_offer_mw",
    "reference_capacity_mw",
)


def close(actual, expected, *, tolerance=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def inputs_digest(service, recovery, market):
    digest = hashlib.sha256()
    for frame, columns in (
        (service, EXOGENOUS_COLUMNS),
        (recovery, EXOGENOUS_COLUMNS),
        (market, MARKET_COLUMNS),
    ):
        values = frame.loc[:, list(columns)].to_numpy(dtype="<f8")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite exogenous or reference input")
        digest.update(str(values.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def peak_reductions(service, recovery):
    output = {}
    for suffix, frames in (
        ("24h", [service]),
        ("26h", [service, recovery]),
    ):
        frame = pd.concat(frames, ignore_index=True)
        baseline = frame.grid_load_mw + frame.p_reference_mw
        for boundary, column in (
            ("pcc", "p_grid_import_mw"),
            ("facility", "p_facility_mw"),
        ):
            actual = frame.grid_load_mw + frame[column]
            output[f"{boundary}_peak_reduction_{suffix}_mw"] = float(
                baseline.max() - actual.max(),
            )
            output[f"{boundary}_peak_phase_{suffix}"] = (
                "service" if int(actual.argmax()) < len(service) else "recovery"
            )
    return output


def audit_record(directory, record, *, tracking_floor_kw, historical=False):
    directory = Path(directory)
    if historical:
        if "outcome" in record:
            raise ValueError("Historical evidence must retain its declared flat schema")
        outcome = record
    else:
        if "outcome" not in record:
            raise ValueError("New confirmation evidence requires a nested outcome")
        outcome = record["outcome"]
    hashes = record["trace_hashes"]
    for name, digest in hashes.items():
        if name not in {"hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz"}:
            raise ValueError("Unexpected raw trace file")
        if file_hash(directory / name) != digest:
            raise ValueError("Raw trace digest changed")
    if (
        not outcome["true_terminal"]
        or outcome["truncated"]
        or outcome["timeout_bootstrap"]
        or outcome["required_gamma"] != 1.0
    ):
        raise AssertionError("Economic terminal/gamma semantics changed")
    if not historical:
        expected_stack = (
            ["MacroLatentEnv", "FractionalDispatchTask"]
            if record["policy"].get("algorithm") == "ppo"
            else ["FractionalDispatchTask"]
        )
        if record["actual_deployment_environment_stack"] != expected_stack:
            raise AssertionError("Unexpected deployment wrapper stack")
        if record["training_performed"] or record["potential_used_at_evaluation"]:
            raise AssertionError("Confirmation is inference-only without potential")
    complete = bool(outcome["complete"] and outcome["recovery"]["complete"])
    if not complete:
        if outcome["qualified"] or outcome["cashflow_26h_usd"] is not None:
            raise AssertionError(
                "Incomplete episode treated as qualified/cash-complete"
            )
        return {
            "complete": False,
            "qualified": False,
            "metrics": {key: None for key in METRICS},
            "raw_checks": 1,
            "raw_file_hashes_checked": len(hashes),
            "full_horizon_raw_audit_available": False,
            "incomplete_horizon_not_hidden_or_averaged": True,
            "inputs_sha256": None,
        }
    if set(hashes) != {"hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz"}:
        raise ValueError("Complete episode lacks one of its raw traces")
    frames = {
        name: pd.read_csv(directory / name)
        for name in ("hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz")
    }
    service, market, recovery = (
        frames["hardware.csv.gz"],
        frames["macro.csv.gz"],
        frames["recovery.csv.gz"],
    )
    if tracking_floor_kw > 5000 or (market.committed_mw < 5).any():
        raise AssertionError("The inherited fee identity's denominator is not valid")
    checks = audit_primary(directory, {**outcome, "trace_hashes": hashes}) + 2
    full = pd.concat([service, recovery], ignore_index=True)
    close(full.tick, np.arange(1, 18721), tolerance=0)
    close(full.duration_hours, 5 / 3600, tolerance=1e-15)
    expected_times = (
        pd.Timestamp("2023-01-01")
        + pd.Timedelta(days=record["cell"]["start_day"])
        + pd.to_timedelta(np.arange(18720) * 5, unit="s")
    )
    if not np.array_equal(pd.to_datetime(full.timestamp), expected_times):
        raise AssertionError("Raw clock does not match the registered 26-hour cell")
    close(full.throttle_batch, 1.0, tolerance=0)
    close(recovery.pump_speed_A, 0.7, tolerance=1e-7)
    close(recovery.hvac_effort, 0.7, tolerance=1e-7)
    if recovery.committed_mw.abs().max() != 0:
        raise AssertionError("Recovery retained a regulation obligation")
    checks += 7
    cap = np.minimum(
        6.0,
        np.maximum(
            0.0,
            0.08 * service.committed_mw
            + (service.credit_before_mwh - 0.02) / service.remaining_window_hours,
        ),
    )
    close(service.guard_cap_mw, cap)
    close(
        service.guarded_extra_response_mw,
        np.minimum(service.desired_extra_response_mw, cap),
    )
    close(
        service.credit_after_mwh,
        service.credit_before_mwh
        + (0.08 * service.committed_mw - service.tracking_err_kw.abs() / 1000)
        * service.duration_hours,
    )
    close(service.credit_before_mwh.iloc[1:], service.credit_after_mwh.iloc[:-1])
    close(service.credit_before_mwh.iloc[0], 0.0)
    close(outcome["credit"]["final_mwh"], service.credit_after_mwh.iloc[-1])
    checks += 6
    windows = outcome["fraction_windows"]
    if len(windows) != 96 or len(market) != 96:
        raise AssertionError("Unexpected number of complete dispatch windows")
    for index, window in enumerate(windows):
        part = service[service.dispatch_window_index == index]
        if len(part) != 180 or window["dispatch_window_index"] != index:
            raise AssertionError("Fraction windows do not cover 180 physical ticks")
        if not 0 <= window["window_fraction"] <= 1:
            raise AssertionError("Fraction is outside the native action space")
        allowance = min(
            6,
            max(
                0,
                0.08 * window["window_committed_mw"]
                + (window["window_credit_start_mwh"] - 0.02)
                / window["full_current_window_hours"],
            ),
        )
        close(window["initial_allowance_mw"], allowance)
        close(
            window["fixed_window_request_mw"],
            window["window_fraction"] * allowance,
        )
        close(part.fixed_window_request_mw, window["fixed_window_request_mw"])
        close(part.window_fraction, window["window_fraction"])
        close(part.window_start_tick, index * 180, tolerance=0)
        close(part.desired_extra_response_mw, window["fixed_window_request_mw"])
        close(market.iloc[index].window_fraction, window["window_fraction"])
        checks += 9
    close(market.bid_price, 0.0, tolerance=0)
    offers = np.array([maximum_native_quantity(x) for x in market.residual_mw])
    close(market.bid_mw, offers, tolerance=2e-6)
    accepted = market.bid_accepted.to_numpy(dtype=bool)
    close(market.rate, np.where(accepted, market.rmcp_usd, 5.0))
    close(market.committed_mw, np.where(accepted, market.bid_mw, 5.0))
    close(market.reference_capacity_mw, np.where(accepted, offers, 5.0), tolerance=2e-6)
    delivered = float(
        np.dot(
            market.committed_mw * market.perf_score,
            market.duration_hours,
        )
    )
    maximum = float(np.dot(market.reference_capacity_mw, market.duration_hours))
    ratio = delivered / maximum
    close(delivered, outcome["performance_weighted_mw_hours"], tolerance=1e-6)
    close(maximum, outcome["maximum_reference_mw_hours"], tolerance=1e-6)
    close(ratio, outcome["delivered_service_ratio"])
    if bool(ratio >= 0.9) != outcome["delivered_service_gate"]:
        raise AssertionError("Service gate does not match the actual paid capacity")
    checks += 9
    last_service, last = service.iloc[-1], recovery.iloc[-1]
    service_checks = {
        "complete": not bool(service.terminated.any()),
        "terminal_soc": bool(last_service.bess_soc >= 0.20),
        "terminal_work": bool(
            last_service.backlog_work_kwh
            <= 0.01 * service.work_arrived_kwh.sum() + 1e-6
        ),
        "thermal_warnings": bool(service.thermal_warning.mean() <= 0.01),
    }
    if service_checks != outcome["service_checks"]:
        raise AssertionError("Raw service physical gates disagree with the ledger")
    work_error = (
        outcome["initial"]["backlog_kwh"]
        + full.work_arrived_kwh.sum()
        - full.work_served_kwh.sum()
        - last.backlog_work_kwh
    )
    if abs(work_error) >= 1e-4:
        raise AssertionError("Work is not conserved within the registered tolerance")
    closure = {
        "soc_closed": abs(last.bess_soc - outcome["initial"]["soc"]) <= 1e-6,
        "work_closed": last.backlog_work_kwh <= 1e-6,
        "thermal_A_closed": abs(last.temp_A - last.reference_temp_A) <= 0.1,
        "thermal_B_closed": abs(last.temp_B - last.reference_temp_B) <= 0.1,
    }
    if any(bool(value) != outcome["recovery"][key] for key, value in closure.items()):
        raise AssertionError("Raw recovery closure gates disagree with the ledger")
    closed = bool(all(closure.values()) and not recovery.terminated.any())
    if closed != outcome["recovery"]["closed"]:
        raise AssertionError("Recovery closure differs from its physical trace")
    qualified = bool(all(service_checks.values()) and closed and ratio >= 0.9)
    if qualified != outcome["qualified"]:
        raise AssertionError("Qualification is not the conjunction of physical gates")
    checks += 5
    peaks = peak_reductions(service, recovery)
    close(
        peaks["pcc_peak_reduction_24h_mw"],
        outcome["grid"]["actual_pcc_boundary_peak_reduction_mw"],
    )
    close(
        peaks["facility_peak_reduction_24h_mw"],
        outcome["grid"]["actual_facility_boundary_peak_reduction_mw"],
    )
    metrics = episode_metrics(outcome)
    battery = float(np.dot(full.bess_actual_kw.abs() / 1000, full.duration_hours))
    close(battery, metrics["battery_throughput_26h_mwh"])
    electricity = float(np.dot(full.p_grid_import_mw * full.lmp, full.duration_hours))
    close(
        market.regulation_revenue.sum() - electricity,
        metrics["cashflow_26h_usd"],
        tolerance=1e-6,
    )
    stress = service.grid_load_mw >= service.grid_load_mw.quantile(0.9)
    fractions = market.window_fraction.to_numpy()
    metrics.update(
        {
            "actual_facility_boundary_peak_reduction_mw": peaks[
                "facility_peak_reduction_24h_mw"
            ],
            "pcc_peak_reduction_26h_mw": peaks["pcc_peak_reduction_26h_mw"],
            "facility_peak_reduction_26h_mw": peaks["facility_peak_reduction_26h_mw"],
            "delivered_service_ratio": ratio,
            "stress_pcc_import_mw": float(
                service.loc[stress, "p_grid_import_mw"].mean()
            ),
            "electricity_26h_usd": electricity,
            "fraction_mean": float(fractions.mean()),
            "fraction_zero_share": float(np.mean(fractions <= 1e-7)),
            "fraction_one_share": float(np.mean(fractions >= 1 - 1e-7)),
        }
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("A complete case produced a non-finite metric")
    accepted_summary = []
    for value in (False, True):
        rows = market[market.bid_accepted == value]
        accepted_summary.append(
            {
                "bid_accepted": value,
                "windows": len(rows),
                "mean_fraction": float(rows.window_fraction.mean())
                if len(rows)
                else None,
                "revenue_usd": float(rows.regulation_revenue.sum()),
                "requested_extra_mwh": float(
                    np.dot(rows.fixed_window_request_mw, rows.duration_hours),
                ),
            }
        )
    return {
        "complete": True,
        "qualified": qualified,
        "metrics": metrics,
        "raw_checks": checks + 4,
        "raw_file_hashes_checked": len(hashes),
        "full_horizon_raw_audit_available": True,
        "inputs_sha256": inputs_digest(service, recovery, market),
        "stress_mask_sha256": hashlib.sha256(stress.to_numpy().tobytes()).hexdigest(),
        "realized_minimum_credit_mwh": float(service.credit_after_mwh.min()),
        "zero_initial_allowance_windows": int(
            (market.initial_allowance_mw < 1e-9).sum()
        ),
        "fraction_histogram_0_025_050_075_100": np.histogram(
            fractions,
            bins=[0, 0.25, 0.5, 0.75, 1],
        )[0].tolist(),
        "accepted_vs_fallback": accepted_summary,
        "peak_phases": {key: value for key, value in peaks.items() if "phase" in key},
        "work_conservation_error_kwh": float(work_error),
    }
