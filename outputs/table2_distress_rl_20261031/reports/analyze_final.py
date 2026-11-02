"""Audit locked final outcomes and produce cohort-complete publication statistics."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# isort: off
from common import SOURCE, file_hash, identity, read_json, write_json  # noqa: E402
from recompute_comparators import read_frame, verify_and_derive  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
# isort: on

COMPONENTS = {
    "it_A": "p_it_A_mw",
    "it_B": "p_it_B_mw",
    "hvac": "p_hvac_mw",
    "pump": "p_pump_mw",
    "chiller": "p_chiller_mw",
    "ups_loss": "p_ups_loss_mw",
    "pdu_loss": "p_pdu_loss_mw",
    "transformer_loss": "p_xfmr_loss_mw",
    "auxiliary": "p_aux_mw",
}
METRICS = (
    "stress_facility_relief_mw",
    "LFR_pct",
    "SPDR_pct",
    "LWSR_pct",
    "LWCF_pct",
    "rmse_kw",
    "matched_bias_revenue_change_pct",
    "cash_delta_vs_pid_usd",
    "cashflow_26h_usd",
    "battery_throughput_26h_mwh",
    "actual_facility_boundary_peak_reduction_mw",
    "actual_pcc_boundary_peak_reduction_mw",
    "stress_pcc_relief_mw",
    "max_temperature_A_C",
    "max_temperature_B_C",
    "max_B_reference_heat_excess_gj",
    "thermal_warning_fraction",
    "delivered_service_ratio",
)


def key(cell):
    return (
        f"d{cell['start_day']}_s{cell['seed']}_b{cell['regd_load_bias']:g}"
    ).replace(".", "p")


@lru_cache(maxsize=8)
def nominal_frame(panel, macro, cell_name):
    return read_frame(
        ROOT / f"final_{panel}" / f"{macro}__nominal" / cell_name / "hardware.csv.gz"
    )


def capacity_floor(macros):
    reference = 0.0
    delivered = 0.0
    for row in macros.itertuples():
        upper = np.float32(min(30, row.residual_mw) / 30)
        if float(upper) * 30 > min(30, row.residual_mw):
            upper = np.nextafter(upper, np.float32(0))
        committed = float(upper) * 30 if row.bid_accepted else 5.0
        reference += committed * row.duration_hours
        delivered += row.committed_mw * row.perf_score * row.duration_hours
    if reference <= 0:
        raise ValueError("Undefined delivered-service denominator")
    return delivered / reference


def verify_prefix(directory, episode):
    for name, expected in episode["trace_hashes"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Changed incomplete trace: {directory / name}")
    frame = read_frame(directory / "hardware.csv.gz")
    np.testing.assert_allclose(
        frame.p_grid_import_mw,
        frame.p_facility_mw - frame.bess_actual_kw / 1000,
        rtol=0,
        atol=1e-10,
    )
    if episode["qualified"]:
        raise AssertionError("Incomplete case was labelled qualified")
    return {
        "qualified": False,
        "complete": episode["complete"],
        "incomplete_case_retained": True,
        "observed_service_steps": len(frame),
        "checks": 1,
    }


def collect(panel, registration, annual_median):
    directory = ROOT / f"final_{panel}"
    summary = read_json(directory / "summary.json")
    if summary["runtime_failures"]:
        raise ValueError("Final panel has unresolved runtime failures")
    expected = {
        (policy["id"], key(cell))
        for policy in registration["policies"]
        if policy["available"]
        for cell in registration["panels"][panel]
    }
    observed = {(row["policy"]["id"], key(row["cell"])) for row in summary["episodes"]}
    if observed != expected or len(summary["episodes"]) != len(expected):
        raise AssertionError("Final policy/case coverage mismatch")
    rows = []
    for episode in summary["episodes"]:
        policy, cell = episode["policy"], episode["cell"]
        path = directory / policy["id"] / key(cell)
        complete = episode["complete"] and episode["recovery"]["complete"]
        derived = (
            verify_and_derive(path, episode, annual_median)
            if complete
            else verify_prefix(path, episode)
        )
        row = {
            **derived,
            "panel": panel,
            "policy_id": policy["id"],
            "family": policy["family"],
            "macro": policy["macro"],
            "training_seed": policy["training_seed"],
            "start_day": cell["start_day"],
            "env_seed": cell["seed"],
            "bias": cell["regd_load_bias"],
            "complete_26h": bool(complete),
            "physical_gate_qualified": bool(episode["qualified"]),
            "qualified": bool(episode["qualified"]),
            "thermal_warning_fraction": episode["service"]["thermal_warning_fraction"],
            "terminal_soc": episode["service_terminal"]["soc"],
            "terminal_backlog_kwh": episode["service_terminal"]["backlog_kwh"],
            "work_conservation_error_kwh": episode["service"][
                "work_conservation_error_kwh"
            ],
        }
        if complete:
            hardware = read_frame(path / "hardware.csv.gz")
            reference = nominal_frame(panel, policy["macro"], key(cell))
            if len(reference) == len(hardware):
                for column in (
                    "timestamp",
                    "grid_load_mw",
                    "regd_signal",
                    "work_arrived_kwh",
                ):
                    np.testing.assert_array_equal(hardware[column], reference[column])
                stress = hardware.grid_load_mw >= row["stress_threshold_mw"]
                for name, column in COMPONENTS.items():
                    row[f"stress_component_{name}_mw"] = float(
                        (reference[column] - hardware[column])[stress].mean()
                    )
                np.testing.assert_allclose(
                    sum(row[f"stress_component_{name}_mw"] for name in COMPONENTS),
                    row["stress_facility_relief_mw"],
                    atol=1e-8,
                    rtol=1e-12,
                )
                for name in ("throughput", "soc", "thermal"):
                    row[f"delta_reward_{name}"] = float(
                        (
                            hardware[f"reward_{name}"] - reference[f"reward_{name}"]
                        ).mean()
                    )
                row["daily_served_work_delta_kwh"] = float(
                    hardware.work_served_kwh.sum() - reference.work_served_kwh.sum()
                )
            else:
                row["component_comparison_unavailable"] = (
                    "Incomplete nominal trajectory"
                )
            macros = read_frame(path / "macro.csv.gz")
            row["mean_committed_mw"] = float(
                np.dot(macros.committed_mw, macros.duration_hours) / 24
            )
            row["mean_paid_performance"] = float(
                np.dot(macros.perf_score, macros.duration_hours) / 24
            )
            if policy["macro"] == "full_residual":
                row["delivered_service_ratio"] = capacity_floor(macros)
                row["delivered_service_gate"] = row["delivered_service_ratio"] >= 0.90
                if policy["family"] != "nominal":
                    row["qualified"] = bool(
                        row["qualified"] and row["delivered_service_gate"]
                    )
        rows.append(row)
    return rows


def paired_contrasts(rows):
    by_key = {
        (
            row["panel"],
            row["macro"],
            row["policy_id"],
            row["start_day"],
            row["env_seed"],
            row["bias"],
        ): row
        for row in rows
    }
    for row in rows:
        prefix = (row["panel"], row["macro"])
        suffix = (row["start_day"], row["env_seed"], row["bias"])
        pid = by_key.get((*prefix, f"{row['macro']}__pid", *suffix))
        if (
            pid is not None
            and row["complete_26h"]
            and pid["complete_26h"]
            and row.get("cashflow_closed_recovery_usd") is not None
            and pid.get("cashflow_closed_recovery_usd") is not None
        ):
            row["cash_delta_vs_pid_usd"] = (
                row["cashflow_closed_recovery_usd"]
                - pid["cashflow_closed_recovery_usd"]
            )
            row["rmse_delta_vs_pid_kw"] = row["rmse_kw"] - pid["rmse_kw"]
            row["relief_delta_vs_pid_mw"] = (
                row["stress_facility_relief_mw"] - pid["stress_facility_relief_mw"]
            )
        if row["bias"] == 0.5:
            unbiased = by_key.get(
                (*prefix, row["policy_id"], row["start_day"], row["env_seed"], 0.0)
            )
            if (
                unbiased is not None
                and row["complete_26h"]
                and unbiased["complete_26h"]
                and unbiased["regulation_revenue_usd"] != 0
            ):
                row["matched_bias_revenue_change_pct"] = 100 * (
                    row["regulation_revenue_usd"] / unbiased["regulation_revenue_usd"]
                    - 1
                )
        if row["family"] in ("warm_sac", "guided_sac"):
            bc = by_key.get(
                (*prefix, f"{row['macro']}__bc_only_{row['training_seed']}", *suffix)
            )
            if bc is not None and row["complete_26h"] and bc["complete_26h"]:
                for name, field in (
                    ("rmse_kw", "rmse_kw"),
                    ("relief_mw", "stress_facility_relief_mw"),
                    ("cash_usd", "cashflow_26h_usd"),
                    ("battery_mwh", "battery_throughput_26h_mwh"),
                ):
                    row[f"paired_bc_delta_{name}"] = row[field] - bc[field]


def aggregate(rows, registration):
    summaries = []
    for panel in registration["panels"]:
        panel_cells = registration["panels"][panel]
        for macro in ("rule", "full_residual"):
            families = sorted(
                {p["family"] for p in registration["policies"] if p["macro"] == macro}
            )
            for family in families:
                policies = [
                    p
                    for p in registration["policies"]
                    if p["macro"] == macro and p["family"] == family
                ]
                values = [
                    r
                    for r in rows
                    if r["panel"] == panel
                    and r["macro"] == macro
                    and r["family"] == family
                ]
                complete_cohort = len(values) == len(policies) * len(
                    panel_cells
                ) and all(row["complete_26h"] for row in values)
                result = {
                    "panel": panel,
                    "macro": macro,
                    "family": family,
                    "planned_models": len(policies),
                    "admitted_models": sum(p["available"] for p in policies),
                    "planned_policy_cases": len(policies) * len(panel_cells),
                    "executed_policy_cases": len(values),
                    "qualified_policy_cases": sum(r["qualified"] for r in values),
                    "complete_cohort": complete_cohort,
                    "no_survivor_only_mean": True,
                    "metric_scope": "beta0.5 cases; revenue paired to own beta0",
                    "dispersion": (
                        "SD across date/noise means after averaging training seeds"
                    ),
                    "metrics": {},
                }
                if complete_cohort:
                    biased = [row for row in values if row["bias"] == 0.5]
                    for metric in METRICS:
                        case_means = []
                        for cell in panel_cells:
                            if cell["regd_load_bias"] != 0.5:
                                continue
                            case_rows = [
                                r
                                for r in biased
                                if r["start_day"] == cell["start_day"]
                                and r["env_seed"] == cell["seed"]
                            ]
                            defined = [r.get(metric) for r in case_rows]
                            if len(defined) == len(policies) and all(
                                v is not None for v in defined
                            ):
                                case_means.append(float(np.mean(defined)))
                        result["metrics"][metric] = {
                            "mean": float(np.mean(case_means)) if case_means else None,
                            "sd": float(np.std(case_means, ddof=1))
                            if len(case_means) > 1
                            else None,
                            "defined_calendar_cases": len(case_means),
                            "planned_calendar_cases": len(panel_cells) // 2,
                        }
                summaries.append(result)
    return summaries


def main():
    output = ROOT / "reports/final_report.json"
    if output.exists():
        raise FileExistsError("Final report already exists; do not overwrite evidence")
    registration = read_json(ROOT / "final_registration.json")
    if (
        identity(
            {
                key: value
                for key, value in registration.items()
                if key != "final_registration_id"
            }
        )
        != registration["final_registration_id"]
    ):
        raise ValueError("Final registration identity changed")
    annual_median = float(
        pd.read_csv(SOURCE / "data/processed/energy/NYC.csv")["Load"].median()
    )
    rows = []
    for panel in registration["panels"]:
        rows.extend(collect(panel, registration, annual_median))
    paired_contrasts(rows)
    summaries = aggregate(rows, registration)
    pd.DataFrame(rows).drop(
        columns=["raw_trace_hashes", "cell"], errors="ignore"
    ).to_csv(ROOT / "reports/final_episode_metrics.csv", index=False)
    record = {
        "final_registration_id": registration["final_registration_id"],
        "training_registration_id": registration["registration_id"],
        "candidate_lock_sha256": file_hash(ROOT / "final_candidate_lock.json"),
        "verified_policy_episodes": len(rows),
        "scalar_or_array_checks": sum(row["checks"] for row in rows),
        "new_policy_rollouts_during_analysis": 0,
        "all_planned_slots_retained": True,
        "summary": summaries,
        "final_episode_metrics_sha256": file_hash(
            ROOT / "reports/final_episode_metrics.csv"
        ),
        "analysis_sha256": file_hash(__file__),
        "interpretation": [
            "Original peak dates are consumed descriptive regression cases.",
            "Six sealed dates are calendar combinations, not unseen workload phases.",
            "Both SAC variants and all training-seed slots are retained.",
            "BC is supervised imitation; eligibility alone is not an RL advantage.",
            "Facility relief excludes BESS; PCC relief includes actual BESS exchange.",
            "SPDR is a stress-average proxy, not a difference of maximum system loads.",
            "Case SD is descriptive, not an independent-sample confidence interval.",
            "Cashflow is benchmark revenue minus electricity, not business profit.",
        ],
    }
    write_json(output, record)
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "checks": record["scalar_or_array_checks"],
                "cohorts": len(summaries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
