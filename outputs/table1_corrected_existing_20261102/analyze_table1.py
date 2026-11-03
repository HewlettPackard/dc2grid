"""Recompute the closed Table 1 cohort from raw traces before rendering results."""

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_table1 import ROOT, verify
from table1_eval import file_hash

METRICS = (
    "tracking_rmse_kw",
    "regulation_revenue_usd",
    "macro_mean_reward",
    "low_mean_reward",
    "thermal_warning_fraction",
    "bess_aging_x1e4",
)
LABELS = {
    "rule_macro+none": "Rule macro + no hardware agent",
    "rule_macro+random": "Rule macro + Random",
    "rule_macro+bang_bang": "Rule macro + Bang-bang",
    "rule_macro+rule_based": "Rule macro + Rule hardware",
    "rule_macro+pid": "Rule macro + PID",
    "rule_macro+sac": "Rule macro + Retrained SAC",
    "sac_macro+sac": "Retrained SAC macro + same SAC low",
}
EXOGENOUS = (
    "regd_signal",
    "grid_load_mw",
    "lmp",
    "T_amb",
    "work_arrived_kwh",
    "p_flex_nom_kw",
    "p_reference_mw",
    "reference_temp_A",
    "reference_temp_B",
)


def read(path: Path):
    return json.loads(path.read_text())


class Audit:
    def __init__(self):
        self.scalar_checks = 0
        self.trace_files = []

    def equal(self, actual, expected, name: str, *, atol=1e-7, rtol=1e-10):
        left, right = np.asarray(actual), np.asarray(expected)
        if left.shape != right.shape or not np.all(np.isfinite(left)):
            raise ValueError(f"{name}: incompatible or non-finite audit values")
        if not np.allclose(left, right, atol=atol, rtol=rtol):
            raise ValueError(f"{name}: independently reconstructed values differ")
        self.scalar_checks += left.size

    def truth(self, value: bool, name: str):
        if not value:
            raise ValueError(name)
        self.scalar_checks += 1


def flags(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype.kind == "b":
        return values
    decoded = []
    for value in values:
        if value in (True, "True", "true", 1, 1.0):
            decoded.append(True)
        elif value in (False, "False", "false", 0, 0.0):
            decoded.append(False)
        else:
            raise ValueError("A native fault flag is missing or not boolean")
    return np.asarray(decoded)


def read_trace(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The 'float_precision' argument is deprecated."
        )
        return pd.read_csv(path, float_precision="round_trip")


def audit_case(protocol: dict, directory: Path, audit: Audit):
    record = read(directory / "episode.json")
    audit.truth(
        record["registration_id"] == protocol["registration_id"],
        "Case registration changed",
    )
    for name, digest in record["trace_hashes"].items():
        path = directory / name
        audit.truth(file_hash(path) == digest, f"Raw trace changed: {path}")
        audit.trace_files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    hardware = read_trace(directory / "service_hardware.csv.gz")
    macro = read_trace(directory / "service_macro.csv.gz")
    recovery_record = record["supplemental_recovery"]
    recovery = (
        read_trace(directory / "recovery_hardware.csv.gz")
        if recovery_record["performed"]
        else pd.DataFrame()
    )
    service = record["service"]
    audit.equal(len(hardware), service["observed_fast_ticks"], "Service length", atol=0)
    audit.equal(len(macro), service["observed_macro_calls"], "Macro length", atol=0)
    audit.equal(macro.sub_steps_run.sum(), len(hardware), "Macro/fast counts", atol=0)
    response = 1000 * (hardware.p_reference_mw - hardware.p_grid_import_mw)
    error = hardware.delta_p_demanded_kw - response
    audit.equal(
        response.to_numpy(),
        hardware.delta_p_actual_kw.to_numpy(),
        "Physical response",
        atol=1e-6,
    )
    audit.equal(
        abs(error).to_numpy(),
        hardware.tracking_err_kw.to_numpy(),
        "Physical error",
        atol=1e-6,
    )
    audit.equal(
        response.to_numpy(),
        hardware.metered_response_kw.to_numpy(),
        "Collector response",
        atol=1e-6,
    )
    audit.equal(
        error.to_numpy(),
        hardware.metered_signed_error_kw.to_numpy(),
        "Collector signed error",
        atol=1e-6,
    )
    audit.equal(
        hardware.duration_hours.to_numpy(),
        np.full(len(hardware), 5 / 3600),
        "Tick duration",
        atol=0,
    )
    warning = (hardware.temp_A >= 33) | (hardware.temp_B >= 33)
    complete = bool(
        len(hardware) == 17280
        and len(macro) == 96
        and (macro.sub_steps_run == 180).all()
        and not flags(hardware.terminated).any()
        and not flags(macro.terminated).any()
        and flags(hardware.truncated)[-1]
        and not any(
            flags(hardware[name]).any()
            for name in ("thermal_fault", "freq_fault", "voltage_fault")
        )
    )
    initial = service["initial_state"]
    metrics = {
        "tracking_rmse_kw": float(np.sqrt(np.mean(error**2))),
        "regulation_revenue_usd": float(macro.regulation_revenue.sum()),
        "macro_mean_reward": float(macro.macro_reward.mean()),
        "low_mean_reward": float(hardware.low_reward.mean()),
        "thermal_warning_fraction": float(warning.mean()),
        "bess_aging_x1e4": float(
            (hardware.bess_age_frac.iloc[-1] - initial["bess_age_fraction"]) * 10000
        ),
    }
    audit.truth(complete == service["complete"], "Native service completion differs")
    for key, value in metrics.items():
        audit.equal(value, service["observed_service_metrics"][key], key)
        primary = service["table1_service_metrics"][key]
        if complete:
            audit.equal(value, primary, "Full-service " + key)
        else:
            audit.truth(
                primary is None, "Incomplete horizon cannot have a primary metric"
            )
    audit.equal(
        service["table1_service_metrics"]["survived"],
        int(complete),
        "Strict survival",
        atol=0,
    )
    components = [
        "reward_backlog",
        "reward_freq",
        "reward_soc",
        "reward_thermal",
        "reward_throughput",
        "reward_tracking",
        "reward_volt",
    ]
    audit.equal(
        hardware[components].sum(axis=1).to_numpy(),
        hardware.low_reward.to_numpy(),
        "Low reward components",
    )
    audit.equal(
        (
            macro.reward_regulation
            + macro.reward_sub
            + macro.reward_elec
            + macro.reward_churn
        ).to_numpy(),
        macro.macro_reward.to_numpy(),
        "Macro reward components",
    )
    service_cost = float(
        (hardware.lmp * hardware.p_grid_import_mw * hardware.duration_hours).sum()
    )
    audit.equal(
        service_cost, macro.electricity_cost.sum(), "Actual import bill", atol=1e-6
    )
    audit.equal(
        (
            macro.rate * macro.committed_mw * macro.perf_score * macro.settlement_hours
        ).to_numpy(),
        macro.regulation_revenue.to_numpy(),
        "Actual revenue settlement",
    )
    for _, window in macro.iterrows():
        ticks = hardware[hardware.macro_call == window.macro_call]
        audit.equal(len(ticks), window.sub_steps_run, "Window length", atol=0)
        audit.equal(
            ticks.duration_hours.sum(),
            window.settlement_hours,
            "Executed settlement duration",
        )
        local_error = ticks.delta_p_demanded_kw - 1000 * (
            ticks.p_reference_mw - ticks.p_grid_import_mw
        )
        audit.equal(
            float(np.sum(local_error**2)),
            window.tracking_error_sq_sum,
            "Window error squares",
            atol=0.1,
            rtol=1e-10,
        )
        audit.equal(
            float(np.sqrt(np.mean(local_error**2))),
            window.tracking_rmse_kw,
            "Window RMSE",
            atol=1e-6,
        )
    arrived = float(hardware.work_arrived_kwh.sum())
    served = float(hardware.work_served_kwh.sum())
    backlog = float(hardware.backlog_work_kwh.iloc[-1])
    conservation = abs(served + backlog - arrived - initial["backlog_kwh"])
    checks = {
        "physical_completion": complete,
        "service_end_soc": bool(hardware.bess_soc.iloc[-1] >= 0.2),
        "service_end_backlog": bool(backlog <= 0.01 * arrived),
        "thermal_warning_fraction": bool(warning.mean() <= 0.01),
        "work_conservation": bool(conservation < 1e-4),
    }
    audit.truth(
        checks == service["supplemental_service_gate"]["checks"],
        "Service physical gates differ",
    )
    recovery_cost = 0.0
    throughput = float(
        (abs(hardware.bess_actual_kw) / 1000 * hardware.duration_hours).sum()
    )
    closed = False
    if recovery_record["performed"]:
        audit.equal(
            len(recovery),
            recovery_record["observed_ticks"],
            "Recovery duration",
            atol=0,
        )
        audit.equal(
            recovery.committed_actual_mw.to_numpy(),
            np.zeros(len(recovery)),
            "Recovery commitment",
            atol=0,
        )
        recovery_cost = float(
            (recovery.lmp * recovery.p_grid_import_mw * recovery.duration_hours).sum()
        )
        throughput += float(
            (abs(recovery.bess_actual_kw) / 1000 * recovery.duration_hours).sum()
        )
        last = recovery.iloc[-1]
        complete_recovery = bool(
            len(recovery) == 1440
            and not flags(recovery.terminated).any()
            and not any(
                flags(recovery[name]).any()
                for name in ("thermal_fault", "freq_fault", "voltage_fault")
            )
        )
        closed = bool(
            complete_recovery
            and abs(last.bess_soc - initial["soc"]) <= 1e-6
            and last.backlog_work_kwh <= 1e-6
            and abs(last.temp_A - last.reference_temp_A) <= 0.1
            and abs(last.temp_B - last.reference_temp_B) <= 0.1
        )
        audit.truth(closed == recovery_record["closed"], "Recovery closure differs")
        audit.equal(
            recovery_cost,
            recovery_record["energy"]["electricity_cost_usd"],
            "Recovery bill",
            atol=1e-6,
        )
        audit.truth(
            recovery_record["extension"]["state_digest_before"]
            == recovery_record["extension"]["state_digest_after"],
            "Recovery must not reset physical/reference/random state",
        )
        all_arrived = arrived + float(recovery.work_arrived_kwh.sum())
        all_served = served + float(recovery.work_served_kwh.sum())
        audit.truth(
            abs(
                all_served
                + last.backlog_work_kwh
                - all_arrived
                - initial["backlog_kwh"]
            )
            < 1e-4,
            "Full continuing work conservation",
        )
    qualified = bool(all(checks.values()) and closed)
    accounting = record["supplemental_accounting"]
    audit.truth(
        qualified == accounting["all_supplemental_gates_pass"],
        "Combined qualification differs",
    )
    cash = metrics["regulation_revenue_usd"] - service_cost - recovery_cost
    audit.equal(
        cash, accounting["observed_net_grid_cash_usd"], "Observed net cash", atol=1e-6
    )
    audit.equal(
        throughput,
        accounting["observed_battery_throughput_mwh"],
        "Observed battery throughput",
        atol=1e-6,
    )
    return (
        {
            "policy": record["policy"]["id"],
            "seed": record["seed"],
            "complete_service": complete,
            "qualified": qualified,
            "checks": checks,
            "recovery_closed": closed,
            "metrics": metrics if complete else {key: None for key in METRICS},
            "observed_prefix_metrics": metrics,
            "fast_ticks": len(hardware),
            "recovery_ticks": len(recovery),
            "service_end_soc": float(hardware.bess_soc.iloc[-1]),
            "service_end_backlog_kwh": backlog,
            "mean_committed_mw": service["descriptive_market_metrics_not_gates"][
                "mean_committed_mw"
            ],
            "observed_cash_usd": cash,
            "closed_26h_cash_usd": accounting["closed_recovery_net_grid_cash_usd"],
            "battery_throughput_mwh": throughput,
            "episode_sha256": file_hash(directory / "episode.json"),
            "episode_path": str((directory / "episode.json").relative_to(ROOT)),
        },
        hardware[list(EXOGENOUS)].to_numpy(),
    )


def aggregate(cases: list[dict], protocol: dict) -> list[dict]:
    result = []
    for policy in protocol["policies"]:
        rows = [row for row in cases if row["policy"] == policy["id"]]
        if len(rows) != 5 or {row["seed"] for row in rows} != set(
            protocol["panel"]["seeds"]
        ):
            raise ValueError("A planned row is incomplete or duplicated")
        complete = all(row["complete_service"] for row in rows)
        qualified = all(row["qualified"] for row in rows)
        metrics = {}
        for key in METRICS:
            values = [row["metrics"][key] for row in rows]
            metrics[key] = (
                {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
                if complete
                else {"mean": None, "sd": None}
            )
        result.append(
            {
                "id": policy["id"],
                "label": LABELS[policy["id"]],
                "planned": 5,
                "complete_service": sum(row["complete_service"] for row in rows),
                "qualified": sum(row["qualified"] for row in rows),
                "all_primary_means_available": complete,
                "diagnostic_not_admissible": complete and not qualified,
                "metrics": metrics,
                "mean_committed_mw": float(
                    np.mean([row["mean_committed_mw"] for row in rows])
                ),
                "mean_cash_26h_usd": (
                    float(np.mean([row["closed_26h_cash_usd"] for row in rows]))
                    if all(row["closed_26h_cash_usd"] is not None for row in rows)
                    else None
                ),
                "mean_observed_battery_throughput_mwh": float(
                    np.mean([row["battery_throughput_mwh"] for row in rows])
                ),
                "service_end_soc_min": min(row["service_end_soc"] for row in rows),
            }
        )
    return result


def main():
    protocol = verify(require_committed=True)
    audit = Audit()
    cases = []
    exogenous = {}
    for policy in protocol["policies"]:
        for seed in protocol["panel"]["seeds"]:
            directory = (
                ROOT / "cases" / policy["id"].replace("+", "__") / f"seed_{seed}"
            )
            result, exo = audit_case(protocol, directory, audit)
            if result["policy"] != policy["id"] or result["seed"] != seed:
                raise ValueError("A case identity does not match the closed panel")
            if policy["id"] == "rule_macro+none":
                exogenous[seed] = exo
            else:
                baseline = exogenous[seed]
                audit.truth(
                    len(exo) <= len(baseline), "Reference does not cover the case"
                )
                audit.equal(
                    exo,
                    baseline[: len(exo)],
                    "Paired exogenous/reference prefix",
                    atol=1e-8,
                    rtol=1e-12,
                )
            cases.append(result)
    rows = aggregate(cases, protocol)
    report = {
        "schema": "table1-corrected-existing-results-v1",
        "registration_id": protocol["registration_id"],
        "planned": 35,
        "recorded": len(cases),
        "complete_service": sum(row["complete_service"] for row in cases),
        "qualified_supplemental": sum(row["qualified"] for row in cases),
        "statistic": "equal mean of all five fixed evaluation cases; sample SD ddof=1",
        "case_conditions_are_reconstructed_defaults_not_original_producing_proof": True,
        "primary_metrics_exclude_recovery": True,
        "no_survivor_only_primary_means": True,
        "independent_training_seeds": False,
        "no_training_or_post_result_checkpoint_selection": True,
        "no_capacity_admission_floor": True,
        "rows": rows,
        "cases": cases,
        "raw_audit": {
            "scalar_checks": int(audit.scalar_checks),
            "trace_files": len(audit.trace_files),
            "trace_bytes": sum(row["bytes"] for row in audit.trace_files),
            "exogenous_fields": list(EXOGENOUS),
            "endogenous_frequency_not_in_pairing": True,
            "implementation_sha256": file_hash(Path(__file__)),
        },
    }
    for row in report["rows"]:
        for metric in row["metrics"].values():
            if metric["mean"] is not None and not math.isfinite(metric["mean"]):
                raise ValueError("A non-finite result cannot be published")
    reports = ROOT / "reports"
    (reports / "table1_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (reports / "raw_trace_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(audit.trace_files)
    with (reports / "table1_summary.csv").open("w", newline="") as stream:
        names = ["policy", "complete_service", "qualified", "diagnostic"]
        names += [f"{key}_{suffix}" for key in METRICS for suffix in ("mean", "sd")]
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {
                "policy": row["id"],
                "complete_service": row["complete_service"],
                "qualified": row["qualified"],
                "diagnostic": row["diagnostic_not_admissible"],
            }
            output.update(
                {
                    f"{key}_{suffix}": row["metrics"][key][suffix]
                    for key in METRICS
                    for suffix in ("mean", "sd")
                }
            )
            writer.writerow(output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "planned",
                    "recorded",
                    "complete_service",
                    "qualified_supplemental",
                    "raw_audit",
                )
            },
            indent=2,
        )
    )
    for row in rows:
        print(
            row["id"],
            json.dumps(row["metrics"]),
            f"survival={row['complete_service']}/5 qualified={row['qualified']}/5",
        )


if __name__ == "__main__":
    main()
