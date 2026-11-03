"""Derive comparator metrics from immutable raw traces without new rollouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import SOURCE, file_hash, read_json, write_json  # noqa: E402
from evaluate import grid_metrics  # noqa: E402


def read_frame(path):
    return pd.read_csv(path, float_precision="round_trip")


def verify_and_derive(directory, episode, annual_median):
    checks = 0
    for name, expected in episode["trace_hashes"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Changed raw comparator trace: {directory / name}")
    hardware = read_frame(directory / "hardware.csv.gz")
    macros = read_frame(directory / "macro.csv.gz")
    recovery = read_frame(directory / "recovery.csv.gz")
    for frame in (hardware, recovery):
        np.testing.assert_allclose(
            frame.p_grid_import_mw,
            frame.p_facility_mw - frame.bess_actual_kw / 1000,
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            frame.delta_p_actual_kw,
            (frame.p_reference_mw - frame.p_facility_mw) * 1000 + frame.bess_actual_kw,
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            frame.tracking_err_kw,
            np.abs(frame.delta_p_demanded_kw - frame.delta_p_actual_kw),
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(frame.duration_hours, 5 / 3600, rtol=0, atol=1e-15)
        checks += 4
    if len(hardware) != 17280 or len(recovery) != 1440:
        raise AssertionError("Comparator service/recovery horizons changed")
    times = pd.to_datetime(pd.concat([hardware, recovery]).timestamp)
    if not (times.diff().dropna().dt.total_seconds() == 5).all():
        raise AssertionError("Non-contiguous 26-hour comparator")
    if not (recovery.delta_p_demanded_kw == 0).all():
        raise AssertionError("Recovery contains a service request")
    checks += 3
    cursor = 0
    for row in macros.itertuples():
        count = round(row.duration_hours * 3600 / 5)
        frame = hardware.iloc[cursor : cursor + count]
        performance = max(
            0, 1 - frame.tracking_err_kw.mean() / max(row.committed_mw * 1000, 1)
        )
        np.testing.assert_allclose(row.perf_score, performance, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            row.regulation_revenue,
            row.rate * row.committed_mw * performance * row.settlement_hours,
            atol=1e-8,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            row.electricity_cost,
            np.dot(frame.lmp * frame.p_grid_import_mw, frame.duration_hours),
            atol=1e-6,
            rtol=1e-12,
        )
        cursor += count
        checks += 3
    if cursor != 17280 or len(macros) != 96:
        raise AssertionError("Macro/fast ledger partition differs")
    cash = float(
        macros.regulation_revenue.sum()
        - np.dot(hardware.lmp * hardware.p_grid_import_mw, hardware.duration_hours)
        - np.dot(recovery.lmp * recovery.p_grid_import_mw, recovery.duration_hours)
    )
    np.testing.assert_allclose(cash, episode["cashflow_26h_usd"], atol=1e-6, rtol=1e-12)
    np.testing.assert_allclose(
        np.sqrt(np.square(hardware.tracking_err_kw).mean()),
        episode["service"]["tracking_rmse_kw"],
        atol=1e-8,
        rtol=1e-12,
    )
    metrics = grid_metrics(hardware, annual_median=annual_median)
    for key, value in episode["grid"].items():
        if key in metrics and value is not None:
            np.testing.assert_allclose(metrics[key], value, atol=1e-8, rtol=1e-12)
            checks += 1
    return {
        "policy_id": episode["policy"]["id"],
        "cell": episode["cell"],
        "qualified": episode["qualified"],
        "rmse_kw": episode["service"]["tracking_rmse_kw"],
        "regulation_revenue_usd": float(macros.regulation_revenue.sum()),
        "cashflow_26h_usd": cash,
        "cashflow_closed_recovery_usd": cash if episode["recovery"]["closed"] else None,
        "battery_throughput_26h_mwh": float(
            np.dot(np.abs(hardware.bess_actual_kw) / 1000, hardware.duration_hours)
            + np.dot(np.abs(recovery.bess_actual_kw) / 1000, recovery.duration_hours)
        ),
        "raw_trace_hashes": episode["trace_hashes"],
        "checks": checks + 2,
        **metrics,
    }


def main():
    load_path = SOURCE / "data/processed/energy/NYC.csv"
    annual_median = float(pd.read_csv(load_path)["Load"].median())
    rows, inputs = [], {}
    for panel in ("development_baselines_v1", "bc_development_v1"):
        directory = ROOT / panel
        summary = read_json(directory / "summary.json")
        inputs[panel] = file_hash(directory / "summary.json")
        if summary["runtime_failures"]:
            raise ValueError("Comparator panel contains a runtime failure")
        manifest = read_json(directory / "manifest.json")
        expected_count = len(manifest["policies"]) * len(manifest["cells"])
        if len(summary["episodes"]) != expected_count:
            raise ValueError("Comparator cohort is incomplete")
        for episode in summary["episodes"]:
            cell = episode["cell"]
            key = (
                f"d{cell['start_day']}_s{cell['seed']}_b{cell['regd_load_bias']:g}"
            ).replace(".", "p")
            path = directory / episode["policy"]["id"] / key
            rows.append(verify_and_derive(path, episode, annual_median))
    output = ROOT / "reports/comparator_metrics.json"
    record = {
        "method": "deterministic derivation from preserved raw 26h traces",
        "new_rollouts": 0,
        "source_panels": inputs,
        "load_file_sha256": file_hash(load_path),
        "annual_load_median_mw": annual_median,
        "metric_implementation_sha256": file_hash(ROOT / "evaluate.py"),
        "verifier_sha256": file_hash(__file__),
        "episodes": len(rows),
        "scalar_or_array_checks": sum(row["checks"] for row in rows),
        "rows": rows,
    }
    if output.exists():
        raise FileExistsError("Refusing to overwrite comparator verification")
    write_json(output, record)
    print(
        json.dumps(
            {key: value for key, value in record.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
