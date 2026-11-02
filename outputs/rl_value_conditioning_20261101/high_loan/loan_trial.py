"""One adaptive high-level trial, with the original cash and physical bars intact."""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# isort: off
from loan_policy import ROOT, PRIOR
from loan_eval import evaluate
from study import (
    read_json,
    write_json,
    file_hash,
    identity,
    verify_registration,
    cap_cpu,
)
from common import case_key
import numpy as np
# isort: on


def verify(protocol):
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise ValueError("Loan trial registration changed")
    for group in ("source_hashes", "input_hashes"):
        for path, expected in protocol[group].items():
            if file_hash(path) != expected:
                raise ValueError(f"Registered high-level dependency changed: {path}")
    base = read_json(PRIOR / "registration.json")
    verify_registration(base)
    if base["registration_id"] != protocol["base_registration_id"]:
        raise ValueError(
            "High-level trial detached from its fixed hardware/market task"
        )
    if (
        protocol["cells"] != base["development_cells"]
        or len({case_key(cell) for cell in protocol["cells"]}) != 24
        or len({policy["id"] for policy in protocol["policies"]}) != 3
        or protocol["episode_count"]
        != len(protocol["cells"]) * len(protocol["policies"])
    ):
        raise ValueError("High-level policy/case budget differs from its registration")
    return base


def register():
    path = ROOT / "registration.json"
    if path.exists():
        raise FileExistsError("Never replace a prospective high-level registration")
    if (PRIOR / "final_opening.json").exists():
        raise RuntimeError("Do not redesign this trial after opening Final-B")
    base = read_json(PRIOR / "registration.json")
    forecast = read_json(PRIOR / "forecast_v1/report.json")
    prior_selection = read_json(PRIOR / "macro_development/selection.json")
    hardware = next(
        policy
        for policy in base["frozen_baseline_policies"]
        if policy["id"] == "bc_only_1100"
    )
    full = {**copy.deepcopy(hardware), "id": "full_control", "macro": "full_residual"}
    prefix = {
        **copy.deepcopy(hardware),
        "id": "prefix_control",
        "macro": "forecast_quantity",
        "forecaster": "ridge",
        "margin_threshold_usd_mw_h": 0.0,
        "forecast_checkpoint": forecast["model"],
    }
    loan = {
        **copy.deepcopy(prefix),
        "id": "expected_surplus_ridge",
        "macro": "expected_surplus_quantity",
    }
    if (
        base["high_level"]["planning_performance"] != 0.98
        or base["high_level"]["guard_ratio"] != 0.92
    ):
        raise ValueError("Do not silently retune the existing planning assumptions")
    inputs = [
        PRIOR / "registration.json",
        PRIOR / "macro_development/selection.json",
        PRIOR / "macro_development/summary.json",
        PRIOR / "forecast_v1/report.json",
        Path(forecast["model"]["path"]),
        Path(hardware["checkpoint"]["path"]),
    ]
    protocol = {
        "schema": "one-expected-terminal-surplus-quantity-trial-v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "base_registration_id": base["registration_id"],
        "cells": base["development_cells"],
        "policies": [full, prefix, loan],
        "episode_count": 72,
        "selected_after_prior_Dev_B_results": True,
        "final_dates_forbidden_to_this_runner": True,
        "single_new_candidate_no_sweep": True,
        "starting_forecaster_selected_from_prior_three_candidate_argmax": True,
        "prior_candidate_means_usd": {
            row["policy"]["id"]: row["mean_paired_cash_gain_usd"]
            for row in prior_selection["candidates"]
        },
        "prior_diagnostic_projections_not_results": {
            "ignoring_loan_gate_interactions_usd": 5.3518,
            "gate_replay_with_frozen_plant_usd": 4.2172,
            "neither_is_an_advancement_test_or_a_bound": True,
        },
        "loan_formula": (
            "measured_delivered - .92*measured_reference + "
            "(.98-.92)*5MW*hours_remaining_after_this_interval"
        ),
        "assumptions_not_guarantees": {
            "future_full_offer_performance": 0.98,
            "future_reference_quantity_at_least_mw": 5.0,
            "booked_delivered_service_includes_no_forecast_credit": True,
            "future_surplus_is_never_claimed_as_actual_delivery": True,
        },
        "advancement": {
            "all_24_physical_and_recovery_gates": True,
            "all_24_actual_service_ratios_at_least": 0.9,
            "minimum_mean_paired_cash_gain_usd": 5.0,
            "both_bias_strata_nonnegative_mean_gain": True,
            "no_new_learned_high_level_SAC_claim": True,
        },
        "cpus": base["secondary_evaluation_cpus"],
        "source_hashes": {
            **base["source_hashes"],
            **{str(source): file_hash(source) for source in ROOT.glob("*.py")},
        },
        "input_hashes": {str(source): file_hash(source) for source in inputs},
    }
    protocol["episode_count"] = len(protocol["cells"]) * len(protocol["policies"])
    protocol["registration_id"] = identity(protocol)
    verify(protocol)
    write_json(path, protocol)
    print(protocol["registration_id"], flush=True)


def init_worker(queue):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(queue.get(timeout=60))


def run():
    protocol = read_json(ROOT / "registration.json")
    base = verify(protocol)
    project = PRIOR.parents[1]
    paths = [
        str(Path(path).relative_to(project))
        for path in protocol["source_hashes"]
        if Path(path).is_relative_to(ROOT)
    ] + [str((ROOT / "registration.json").relative_to(project))]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=project,
        check=True,
    )
    attempt = ROOT / "attempt.json"
    if attempt.exists():
        raise RuntimeError("This high-level trial was already attempted")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_json(
        attempt,
        {
            "registration_id": protocol["registration_id"],
            "implementation_commit": commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    destination = ROOT / "development"
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "manifest.json", protocol)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
    results, failures = [], []
    with ProcessPoolExecutor(
        max_workers=len(protocol["cpus"]),
        mp_context=context,
        initializer=init_worker,
        initargs=(queue,),
    ) as pool:
        futures = {
            pool.submit(
                evaluate,
                policy,
                cell,
                base,
                destination / policy["id"] / case_key(cell),
                protocol["registration_id"],
            ): (policy["id"], case_key(cell))
            for policy in protocol["policies"]
            for cell in protocol["cells"]
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(futures[future], result["qualified"], flush=True)
            except Exception as error:
                failures.append({"case": futures[future], "error": repr(error)})
                print("LOAN EVALUATION FAILURE", failures[-1], flush=True)
    queue.close()
    queue.join_thread()
    write_json(
        destination / "summary.json",
        {
            "episodes": results,
            "runtime_failures": failures,
        },
    )
    verify(protocol)
    if failures or len(results) != protocol["episode_count"]:
        raise RuntimeError("The planned cohort did not complete; no filtered means")
    by_policy = {
        policy["id"]: {
            case_key(row["cell"]): row
            for row in results
            if row["policy"]["id"] == policy["id"]
        }
        for policy in protocol["policies"]
    }
    # Both unchanged controls must reproduce their previously recorded outcomes.
    previous = read_json(PRIOR / "macro_development/summary.json")["episodes"]
    for current, old in (
        ("full_control", "macro_full_reference"),
        ("prefix_control", "macro_ridge_threshold_0"),
    ):
        old_rows = {
            case_key(row["cell"]): row for row in previous if row["policy"]["id"] == old
        }
        for cell, row in by_policy[current].items():
            for key in (
                "grid",
                "service",
                "service_checks",
                "qualified",
                "recovery",
                "cashflow_closed_recovery_usd",
                "delivered_service_ratio",
                "delivered_service_gate",
            ):
                if row[key] != old_rows[cell][key]:
                    raise AssertionError(
                        "New runner changed an existing control outcome"
                    )
            np.testing.assert_allclose(
                row["cashflow_26h_usd"],
                old_rows[cell]["cashflow_26h_usd"],
                atol=1e-8,
                rtol=0,
            )
    rows = list(by_policy["expected_surplus_ridge"].values())
    pairs = []
    for row in rows:
        reference = by_policy["full_control"][case_key(row["cell"])]
        cash, base_cash = (
            row["cashflow_closed_recovery_usd"],
            reference["cashflow_closed_recovery_usd"],
        )
        pairs.append(
            {
                "cell": row["cell"],
                "gain_usd": cash - base_cash
                if cash is not None and base_cash is not None
                else None,
                "qualified": row["qualified"],
                "service_ratio": row["delivered_service_ratio"],
            }
        )
    defined = all(row["gain_usd"] is not None for row in pairs)
    mean = float(np.mean([row["gain_usd"] for row in pairs])) if defined else None
    by_bias = (
        {
            str(bias): float(
                np.mean(
                    [
                        row["gain_usd"]
                        for row in pairs
                        if row["cell"]["regd_load_bias"] == bias
                    ]
                )
            )
            for bias in (0.0, 0.5)
        }
        if defined
        else None
    )
    criterion = protocol["advancement"]
    eligible = bool(
        defined
        and all(
            row["qualified"]
            and row["service_ratio"]
            >= criterion["all_24_actual_service_ratios_at_least"]
            for row in pairs
        )
        and mean >= criterion["minimum_mean_paired_cash_gain_usd"]
        and all(value >= 0 for value in by_bias.values())
    )
    write_json(
        destination / "decision.json",
        {
            "registration_id": protocol["registration_id"],
            "unchanged_controls_reproduced": True,
            "pairs": pairs,
            "mean_paired_cash_gain_usd": mean,
            "by_bias_gain_usd": by_bias,
            "eligible": eligible,
            "criteria": protocol["advancement"],
            "not_a_fresh_panel_or_independent_replication": True,
            "starting_forecaster_selected_from_prior_three_candidate_argmax": True,
        },
    )
    print("MEAN GAIN", mean, "ELIGIBLE", eligible, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    args = parser.parse_args()
    register() if args.action == "register" else run()
