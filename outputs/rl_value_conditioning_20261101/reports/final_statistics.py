"""Recompute both publication panels from immutable raw 26-hour traces."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# isort: off
from ctx import (
    ROOT,
    PRIOR,
    BASELINE,
    SOURCE,
    read_json,
    write_json,
    file_hash,
    cap_cpu,
    verify,
    identity,
)
from common import case_key
from finish_value import read_lock

sys.path.insert(0, str(BASELINE / "reports"))
from recompute_comparators import read_frame, verify_and_derive
from analyze_final import aggregate, paired_contrasts, capacity_floor, verify_prefix
import numpy as np
import pandas as pd
# isort: on


def check_episode(arguments):
    panel, directory, episode, median, service_gate, recovery_gate, service_floor = (
        arguments
    )
    directory = Path(directory)
    policy, cell = episode["policy"], episode["cell"]
    complete = episode["complete"] and episode["recovery"]["complete"]
    values = (
        verify_and_derive(directory, episode, median)
        if complete
        else verify_prefix(directory, episode)
    )
    result = {
        **values,
        "panel": panel,
        "policy_id": policy["id"],
        "family": policy["family"],
        "macro": policy["macro"],
        "training_seed": policy["training_seed"],
        "start_day": cell["start_day"],
        "env_seed": cell["seed"],
        "bias": cell["regd_load_bias"],
        "complete_26h": bool(complete),
        "physical_gate_qualified": episode["qualified"],
        "qualified": episode["qualified"],
        "thermal_warning_fraction": episode["service"]["thermal_warning_fraction"],
        "terminal_soc": episode["service_terminal"]["soc"],
        "terminal_backlog_kwh": episode["service_terminal"]["backlog_kwh"],
        "work_conservation_error_kwh": episode["service"][
            "work_conservation_error_kwh"
        ],
        "episode_path": str(directory),
        "source_summary": str(directory.parents[1] / "summary.json"),
        "source_episode_registration_id": episode.get("study_registration_id"),
        "floor_enforced": policy["macro"] == "full_residual"
        and policy["family"] != "nominal",
    }
    if complete:
        hardware = read_frame(directory / "hardware.csv.gz")
        recovery = read_frame(directory / "recovery.csv.gz")
        macros = read_frame(directory / "macro.csv.gz")
        np.testing.assert_allclose(
            hardware.bess_soc.iloc[-1], result["terminal_soc"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            hardware.thermal_warning.mean(),
            result["thermal_warning_fraction"],
            atol=1e-12,
            rtol=0,
        )
        for name, frame in (("service", hardware), ("recovery", recovery)):
            initial_backlog = (
                episode["initial"]["backlog_kwh"]
                if name == "service"
                else hardware.backlog_work_kwh.iloc[-1]
            )
            error = (
                initial_backlog
                + frame.work_arrived_kwh.sum()
                - frame.work_served_kwh.sum()
                - frame.backlog_work_kwh.iloc[-1]
            )
            if abs(error) >= 1e-4:
                raise AssertionError("Raw work conservation failed")
        raw_service_checks = {
            "complete": len(hardware) == 17280,
            "terminal_soc": hardware.bess_soc.iloc[-1]
            >= service_gate["min_terminal_soc"],
            "terminal_work": hardware.backlog_work_kwh.iloc[-1]
            <= service_gate["max_backlog_fraction"] * hardware.work_arrived_kwh.sum()
            + 1e-6,
            "thermal_warnings": hardware.thermal_warning.mean()
            <= service_gate["max_thermal_warning_fraction"],
        }
        last = recovery.iloc[-1]
        raw_closed = bool(
            abs(last.bess_soc - episode["initial"]["soc"])
            <= recovery_gate["soc_tolerance"]
            and abs(last.backlog_work_kwh) <= recovery_gate["backlog_tolerance_kwh"]
            and abs(last.temp_A - last.reference_temp_A)
            <= recovery_gate["temperature_tolerance_C"]
            and abs(last.temp_B - last.reference_temp_B)
            <= recovery_gate["temperature_tolerance_C"]
        )
        if (
            raw_service_checks != episode["service_checks"]
            or raw_closed != episode["recovery"]["closed"]
            or bool(all(raw_service_checks.values()) and raw_closed)
            != episode["qualified"]
        ):
            raise AssertionError(
                "Raw physical/recovery gates disagree with reported qualification"
            )
        result["checks"] += 7
        result["delivered_service_ratio"] = capacity_floor(macros)
        result["delivered_service_gate"] = (
            result["delivered_service_ratio"] >= service_floor
        )
        result["capacity_ratio_reference"] = (
            "full admissible residual under matching market draws"
        )
        if policy["macro"] == "full_residual" and policy["family"] != "nominal":
            result["qualified"] = bool(
                result["qualified"] and result["delivered_service_gate"]
            )
        result["exogenous_hashes"] = {}
        result["positive_load_weight"] = bool(
            np.maximum(0, hardware.grid_load_mw.to_numpy() - median).sum() > 0
        )
        for metric in ("LWCF_pct", "LWSR_pct"):
            if (result[metric] is not None) != result["positive_load_weight"]:
                raise AssertionError(
                    "Missing weighted metric has a non-exogenous cause"
                )
        for column in (
            "grid_load_mw",
            "regd_signal",
            "work_arrived_kwh",
            "p_reference_mw",
            "lmp",
        ):
            result["exogenous_hashes"][column] = hashlib.sha256(
                np.asarray(hardware[column], dtype="<f8").tobytes()
            ).hexdigest()
        duration = float(macros.duration_hours.sum())
        np.testing.assert_allclose(duration, 24.0, atol=1e-9, rtol=0)
        result["mean_committed_mw"] = float(
            np.dot(macros.committed_mw, macros.duration_hours) / duration
        )
        result["mean_paid_performance"] = float(
            np.dot(macros.perf_score, macros.duration_hours) / duration
        )
        if policy["macro"] == "full_residual":
            result["full_offer_schedule_sha256"] = hashlib.sha256(
                np.asarray(
                    macros[["committed_mw", "rate", "duration_hours", "bid_accepted"]],
                    dtype="<f8",
                ).tobytes()
            ).hexdigest()
        result["checks"] += 3
    return result


def init_worker(queue):
    cap_cpu(queue.get(timeout=60))


def add_new_bc_contrasts(rows, variants):
    lookup = {
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
        if row["family"] not in variants:
            continue
        comparison = lookup[
            (
                row["panel"],
                row["macro"],
                f"{row['macro']}__bc_only_{row['training_seed']}",
                row["start_day"],
                row["env_seed"],
                row["bias"],
            )
        ]
        if row["complete_26h"] and comparison["complete_26h"]:
            for suffix, metric in (
                ("rmse_kw", "rmse_kw"),
                ("relief_mw", "stress_facility_relief_mw"),
                ("cash_usd", "cashflow_26h_usd"),
                ("battery_mwh", "battery_throughput_26h_mwh"),
            ):
                row[f"paired_bc_delta_{suffix}"] = row[metric] - comparison[metric]


def main(cpus):
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    lock = read_lock(protocol)
    final_lock = read_json(ROOT / "final_lock.json")
    final_body = {key: value for key, value in final_lock.items() if key != "lock_id"}
    if (
        identity(final_body) != final_lock["lock_id"]
        or final_lock["registration_id"] != protocol["registration_id"]
        or read_json(PRIOR / "final_lock.json") != final_lock
    ):
        raise ValueError("Final lock provenance does not reconcile")
    final_summary = ROOT / "final_B/summary.json"
    if not final_summary.exists():
        raise ValueError("No complete locked final panel exists")
    report_path = ROOT / "reports/final_report.json"
    if any(
        (ROOT / "reports" / name).exists()
        for name in (
            "final_report.json",
            "final_episode_metrics.json",
            "final_episode_metrics.csv",
        )
    ):
        raise FileExistsError("Never overwrite an audited final report")
    old_final = read_json(BASELINE / "final_registration.json")
    regression_protocol = read_json(ROOT / "regression_registration.json")
    panels = {
        "regression": old_final["panels"]["regression"],
        "sealed": protocol["final_cells"],
    }
    locations = [
        ("regression", BASELINE / "final_regression", None),
        (
            "regression",
            ROOT / "peak_regression",
            regression_protocol["registration_id"],
        ),
        ("sealed", ROOT / "final_B", protocol["registration_id"]),
    ]
    tasks, source_summaries, authorizations, policy_locations = [], {}, {}, {}
    locked_policies = {policy["id"]: policy for policy in lock["inventory"]}
    median = float(
        pd.read_csv(SOURCE / "data/processed/energy/NYC.csv")["Load"].median()
    )
    for panel, directory, authorization in locations:
        path = directory / "summary.json"
        summary = read_json(path)
        if summary["runtime_failures"]:
            raise ValueError("Final report cannot silently filter runtime failures")
        source_summaries[str(path)] = file_hash(path)
        authorizations[str(path)] = {
            "episode_registration_id": authorization,
            "legacy_final_registration_id": old_final["final_registration_id"]
            if authorization is None
            else None,
            "reused_unchanged": authorization is None,
        }
        policy_ids = {episode["policy"]["id"] for episode in summary["episodes"]}
        previous_ids = policy_locations.setdefault(panel, set())
        if previous_ids & policy_ids:
            raise AssertionError("Historical and new directories overlap in policy IDs")
        previous_ids.update(policy_ids)
        for episode in summary["episodes"]:
            if episode.get("study_registration_id") != authorization:
                raise ValueError("An episode belongs to a different registration")
            policy = episode["policy"]
            expected_policy = locked_policies[policy["id"]]
            for key in ("family", "macro", "hardware", "training_seed"):
                if policy[key] != expected_policy[key]:
                    raise ValueError(
                        "Evaluated policy differs from its locked identity"
                    )
            if policy.get("checkpoint") != expected_policy.get("checkpoint"):
                raise ValueError(
                    "An evaluated checkpoint differs from its locked artifact"
                )
            tasks.append(
                (
                    panel,
                    str(
                        directory / episode["policy"]["id"] / case_key(episode["cell"])
                    ),
                    episode,
                    median,
                    protocol["service_gate"],
                    protocol["recovery"],
                    protocol["service_floor"],
                )
            )
    for panel, cells in panels.items():
        expected = {
            (slot["id"], case_key(cell))
            for slot in lock["inventory"]
            if slot["available"]
            for cell in cells
        }
        observed = [
            (episode["policy"]["id"], case_key(episode["cell"]))
            for name, _, episode, _, _, _, _ in tasks
            if name == panel
        ]
        if set(observed) != expected or len(observed) != len(expected):
            raise AssertionError("Publication cohort has missing or duplicate cases")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    with ProcessPoolExecutor(
        max_workers=len(cpus),
        mp_context=context,
        initializer=init_worker,
        initargs=(queue,),
    ) as pool:
        rows = list(pool.map(check_episode, tasks))
    queue.close()
    queue.join_thread()
    exogenous, full_offers, zero_weight_cases, qualification_counts = {}, {}, set(), {}
    matched_rows = 0
    for row in rows:
        definition = (
            "physical_recovery_and_capacity"
            if row["floor_enforced"]
            else "physical_and_recovery"
        )
        counter = qualification_counts.setdefault(
            f"{row['panel']}:{row['macro']}:{definition}",
            {"executed": 0, "qualified": 0},
        )
        counter["executed"] += 1
        counter["qualified"] += int(row["qualified"])
        if not row["complete_26h"]:
            continue
        key = (row["panel"], row["start_day"], row["env_seed"], row["bias"])
        previous = exogenous.setdefault(key, row["exogenous_hashes"])
        if previous != row["exogenous_hashes"]:
            raise AssertionError(
                "Policies did not share the same exogenous/nominal trajectories"
            )
        matched_rows += 1
        if not row["positive_load_weight"]:
            zero_weight_cases.add(key)
        if row["macro"] == "full_residual":
            reference_offer = full_offers.setdefault(
                key, row["full_offer_schedule_sha256"]
            )
            if reference_offer != row["full_offer_schedule_sha256"]:
                raise AssertionError(
                    "Full-residual offers differ across hardware policies"
                )
    paired_contrasts(rows)
    add_new_bc_contrasts(rows, protocol["variants"])
    by_case = {
        (
            row["panel"],
            row["policy_id"],
            row["start_day"],
            row["env_seed"],
            row["bias"],
        ): row
        for row in rows
    }
    macro_contrasts = []
    for row in rows:
        if row["family"] not in protocol["variants"] or row["macro"] != "full_residual":
            continue
        other = by_case[
            (
                row["panel"],
                row["policy_id"].replace("full_residual__", "rule__", 1),
                row["start_day"],
                row["env_seed"],
                row["bias"],
            )
        ]
        first_cash = row.get("cashflow_closed_recovery_usd")
        second_cash = other.get("cashflow_closed_recovery_usd")
        macro_contrasts.append(
            {
                "panel": row["panel"],
                "family": row["family"],
                "training_seed": row["training_seed"],
                "start_day": row["start_day"],
                "env_seed": row["env_seed"],
                "bias": row["bias"],
                "closed_cash_full_minus_rule_usd": first_cash - second_cash
                if first_cash is not None and second_cash is not None
                else None,
                "both_physical_recovery_qualified": bool(
                    row["physical_gate_qualified"] and other["physical_gate_qualified"]
                ),
                "full_service_floor_passed": row.get("delivered_service_gate", False),
            }
        )
    summary = aggregate(rows, {"panels": panels, "policies": lock["inventory"]})
    for group in summary:
        for name, value in group["metrics"].items():
            if value["defined_calendar_cases"] != value["planned_calendar_cases"]:
                if name in ("LWCF_pct", "LWSR_pct"):
                    positive_count = sum(
                        cell["regd_load_bias"] == 0.5
                        and (group["panel"], cell["start_day"], cell["seed"], 0.5)
                        not in zero_weight_cases
                        for cell in panels[group["panel"]]
                    )
                    if value["defined_calendar_cases"] != positive_count:
                        raise AssertionError(
                            "Weighted-metric gap is not a zero-weight case"
                        )
                    value["scope"] = "Only exogenous cases with positive load weight"
                    continue
                value["incomplete_metric_cohort"] = True
                value["mean"], value["sd"] = None, None
    report = {
        "training_registration_id": protocol["registration_id"],
        "candidate_lock_id": lock["candidate_lock_id"],
        "final_lock_id": final_lock["lock_id"],
        "headline_family_selected_before_final": lock["headline_family"],
        "panels": panels,
        "inventory": lock["inventory"],
        "aggregates": summary,
        "source_summary_hashes": source_summaries,
        "annual_load_median_mw": median,
        "load_source_sha256": file_hash(SOURCE / "data/processed/energy/NYC.csv"),
        "auditor_sha256": file_hash(__file__),
        "reused_raw_verifier_sha256": file_hash(
            BASELINE / "reports/recompute_comparators.py"
        ),
        "new_rollouts_by_this_analysis": 0,
        "executed_episodes": len(rows),
        "complete_26h_episodes": sum(row["complete_26h"] for row in rows),
        "qualified_episodes": sum(row["qualified"] for row in rows),
        "scalar_or_array_checks": sum(row["checks"] for row in rows),
        "exogenous_and_nominal_trajectories_match": matched_rows > 0,
        "complete_rows_with_exogenous_checks": matched_rows,
        "full_residual_offer_schedule_groups_checked": len(full_offers),
        "zero_load_weight_cases": [list(key) for key in sorted(zero_weight_cases)],
        "qualification_counts_by_definition": qualification_counts,
        "source_authorizations": authorizations,
        "high_level_incumbents": ["unchanged_rule_macro", "unchanged_full_residual"],
        "learned_high_level_policy_promoted": False,
        "fixed_hardware_macro_contrasts": macro_contrasts,
        "macro_contrast_scope": (
            "Paired unchanged full-residual versus unchanged rule macro"
        ),
        "high_level_development_disposition": {
            "loan_decision_path": str(ROOT / "high_loan/development/decision.json"),
            "loan_decision_sha256": file_hash(
                ROOT / "high_loan/development/decision.json"
            ),
            "scope": "Known development only; outside the final policy inventory",
        },
        "paired_delta_scope": (
            "Hardware-policy contrasts under the same macro algorithm; "
            "not learned high-level gains"
        ),
        "statistical_scope": (
            "Beta0.5 case means after averaging all training seeds; descriptive SD "
            "across date/noise means, not independent replicates "
            "or confidence intervals"
        ),
        "regression_controls_reused_unchanged": str(BASELINE / "final_regression"),
        "reused_regression_inventory": {
            "admitted_hardware_slots": 14,
            "policy_ids": 28,
            "calendar_noise_bias_cells": 30,
            "episodes": 840,
        },
    }
    write_json(report_path, report)
    write_json(ROOT / "reports/final_episode_metrics.json", rows)
    pd.DataFrame(rows).drop(
        columns=["raw_trace_hashes", "exogenous_hashes"], errors="ignore"
    ).to_csv(ROOT / "reports/final_episode_metrics.csv", index=False)
    print(
        {
            key: value
            for key, value in report.items()
            if key not in ("inventory", "aggregates", "panels", "source_summary_hashes")
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    main(args.cpus)
