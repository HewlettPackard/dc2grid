"""Consumed original-peak comparison; never a fresh test or a selection surface."""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "track_b_fraction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "track_b"))

# isort: off
from train_macro import EconomicDispatchTask, MacroLatentEnv
from fraction_macro import FractionalDispatchTask
from evaluate_macro import scripted_action
from evaluate_low import audit_primary
from economy_context import (
    ROOT,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
    case_key,
)
from headroom import initialize_worker
from headroom_report import aggregate
from stable_baselines3 import PPO, SAC
import numpy as np
import pandas as pd
# isort: on

REGISTRATION = ROOT / "peak_regression_registration.json"
OUTPUT = ROOT / "original_peak_regression"


def prepare():
    if REGISTRATION.exists():
        raise FileExistsError("Do not replace the consumed-peak comparison plan")
    parent = read_json(ROOT / "track_b_fraction/registration.json")
    original_path = (
        ROOT.parent / "rl_value_conditioning_20261101/reports/final_report.json"
    )
    original = read_json(original_path)
    cells = original["panels"]["regression"]
    if len(cells) != 30 or {cell["start_day"] for cell in cells} != {207, 248, 249}:
        raise ValueError("The original Table 2 peak conditions changed")
    specs, source_protocols = [], {}
    inputs = dict(parent["input_hashes"])
    inputs[str(original_path)] = file_hash(original_path)
    for source in ("track_b", "track_b_fraction"):
        path = ROOT / source / "registration.json"
        source_protocols[source] = {"path": str(path), "sha256": file_hash(path)}
        inputs[str(path)] = file_hash(path)
        directory = "development_full" if source == "track_b" else "development"
        manifest_path = ROOT / source / directory / "manifest.json"
        manifest = read_json(manifest_path)
        inputs[str(manifest_path)] = file_hash(manifest_path)
        for policy in manifest["policies"]:
            artifact = policy["checkpoint"]
            inputs[artifact["path"]] = artifact["sha256"]
            specs.append(
                {
                    "id": f"{source}__{policy['id']}",
                    "source": source,
                    "policy": copy.deepcopy(policy),
                }
            )
    for policy in read_json(ROOT / "track_b/registration.json")["baseline_policies"]:
        specs.append(
            {
                "id": f"scripted__{policy['id']}",
                "source": "track_b",
                "policy": copy.deepcopy(policy),
            }
        )
    if len(specs) != 28:
        raise ValueError("Retain all 24 economic model checkpoints and four controls")
    plan = {
        "schema": "consumed-original-peak-economic-regression-v1",
        "registration_file": REGISTRATION.name,
        "source_hashes": {
            **parent["source_hashes"],
            str(Path(__file__).resolve()): file_hash(__file__),
        },
        "input_hashes": inputs,
        "source_protocols": source_protocols,
        "policies": specs,
        "cells": cells,
        "cpus": list(range(32)),
        "original_reference_report": str(original_path),
        "expected_episodes": len(specs) * len(cells),
        "role": "consumed_original_peak_regression_only",
        "macro": "existing_full_residual_zero_ask",
        "not_the_original_rule_macro_tracking_task": True,
        "no_new_training_or_model_selection": True,
        "no_advancement_from_regression_outcomes": True,
        "fresh_calendar_reservation_must_remain_unopened": True,
    }
    plan["registration_id"] = identity(plan)
    write_json(REGISTRATION, plan)
    verify(plan, require_commit=False)
    print(plan["registration_id"])


def evaluate(plan, entry, cell, destination):
    protocol = read_json(plan["source_protocols"][entry["source"]]["path"])
    policy = entry["policy"]
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    task_type = (
        FractionalDispatchTask
        if entry["source"] == "track_b_fraction"
        else EconomicDispatchTask
    )
    task = task_type(
        protocol,
        seed=0,
        fixed_cell=cell,
        record_trace=True,
        evaluation_mode=True,
        evaluation_cells=plan["cells"],
    )
    environment, model = task, None
    try:
        if "checkpoint" in policy:
            artifact = policy["checkpoint"]
            if file_hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError("A locked regression checkpoint changed")
            algorithm = policy["algorithm"]
            model = (PPO if algorithm == "ppo" else SAC).load(
                artifact["path"], device="cpu"
            )
            if model.c2g_context["registration_id"] != protocol["registration_id"]:
                raise ValueError("Wrong training identity in a regression model")
            if algorithm == "ppo":
                environment = MacroLatentEnv(task)
        observation, _ = environment.reset()
        while True:
            action = (
                model.predict(observation, deterministic=True)[0]
                if model is not None
                else scripted_action(policy, observation)
            )
            observation, _, terminated, truncated, _ = environment.step(action)
            if truncated:
                raise AssertionError("Regression task exposed a timeout")
            if terminated:
                break
        result = {
            **task.episode_result,
            "policy": policy,
            "evaluation_entry": entry,
            "evaluation_registration_id": plan["registration_id"],
            "source_training_registration_id": protocol["registration_id"],
            "role": plan["role"],
            "potential_used_at_deployment": False,
        }
        traces = {}
        for name, rows in task.terminal_traces().items():
            if rows:
                path = directory / name
                pd.DataFrame(rows).to_csv(
                    path, index=False, compression="gzip", mode="x"
                )
                traces[name] = file_hash(path)
        result["trace_hashes"] = traces
        write_json(directory / "episode.json", result)
        return result
    except Exception as error:
        write_json(
            directory / "failure.json",
            {
                "entry": entry,
                "cell": cell,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        environment.close()


def run():
    plan = read_json(REGISTRATION)
    commit = verify(plan)
    if (ROOT / "final_opening_grid_cash.json").exists():
        raise RuntimeError("This comparison is registered before any fresh opening")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(
        OUTPUT / "manifest.json",
        {
            **plan,
            "implementation_commit": commit,
        },
    )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in plan["cpus"]:
        queue.put(cpu)
    completed, failures = 0, []
    with ProcessPoolExecutor(
        max_workers=len(plan["cpus"]),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as pool:
        futures = {}
        for entry in plan["policies"]:
            for cell in plan["cells"]:
                destination = OUTPUT / entry["id"] / case_key(cell)
                future = pool.submit(evaluate, plan, entry, cell, str(destination))
                futures[future] = str(destination)
        for future in as_completed(futures):
            try:
                result = future.result()
                completed += 1
                print(
                    f"{completed}/{plan['expected_episodes']} "
                    f"{result['evaluation_entry']['id']} {case_key(result['cell'])} "
                    f"qualified={result['qualified']}",
                    flush=True,
                )
            except Exception as error:
                failure = {"directory": futures[future], "error": repr(error)}
                failures.append(failure)
                print(f"FAILED: {failure}", flush=True)
            write_json(
                OUTPUT / "status.json",
                {
                    "completed": completed,
                    "expected": plan["expected_episodes"],
                    "runtime_failures": failures,
                },
            )
    queue.close()
    queue.join_thread()
    verify(plan)
    if failures or completed != plan["expected_episodes"]:
        raise RuntimeError("Consumed peak regression has missing or failed cases")


def report():
    plan = read_json(REGISTRATION)
    verify(plan)
    status = read_json(OUTPUT / "status.json")
    if status["runtime_failures"] or status["completed"] != plan["expected_episodes"]:
        raise ValueError("Cannot summarize incomplete peak regression")
    episodes, checks = [], 0
    for entry in plan["policies"]:
        for cell in plan["cells"]:
            directory = OUTPUT / entry["id"] / case_key(cell)
            episode = read_json(directory / "episode.json")
            if (
                episode["evaluation_entry"] != entry
                or episode["cell"] != cell
                or episode["evaluation_registration_id"] != plan["registration_id"]
            ):
                raise ValueError("Regression identity or condition changed")
            checks += audit_primary(directory, episode)
            episodes.append(episode)
    families = aggregate(episodes, plan["cells"])
    original = read_json(plan["original_reference_report"])
    references = {
        row["family"]: row
        for row in original["aggregates"]
        if row["panel"] == "regression" and row["macro"] == "full_residual"
    }
    analytic = families["analytic"]["strata"]["0.5"]["metrics"]
    for key in (
        "stress_facility_relief_mw",
        "stress_pcc_relief_mw",
        "cashflow_26h_usd",
    ):
        np.testing.assert_allclose(
            analytic[key]["mean"],
            references["cooling_first"]["metrics"][key]["mean"],
            atol=1e-6,
            rtol=1e-10,
        )
    pid_cash = references["pid"]["metrics"]["cashflow_26h_usd"]["mean"]
    for row in families.values():
        cash = row["strata"]["0.5"]["metrics"]["cashflow_26h_usd"]["mean"]
        row["biased_cash_delta_vs_original_matched_pid"] = (
            cash - pid_cash if cash is not None else None
        )
    destination = ROOT / "reports/original_peak_regression.json"
    if destination.exists():
        raise FileExistsError("Never overwrite consumed peak regression results")
    write_json(
        destination,
        {
            "registration_id": plan["registration_id"],
            "role": plan["role"],
            "episodes": len(episodes),
            "raw_primary_checks": checks,
            "families": families,
            "frozen_full_residual_references": references,
            "analytic_reproduces_original_peak_panel": True,
            "fresh_reservation_unopened": not (
                ROOT / "final_opening_grid_cash.json"
            ).exists(),
            "no_model_selection_or_advancement": True,
            "not_original_rule_macro_tracking_benchmark": True,
        },
    )
    print(f"Audited {len(episodes)} original-peak episodes; {checks} primary checks.")
    for family, row in families.items():
        print(
            family,
            row["qualified_policy_cases"],
            "/",
            row["policy_case_count"],
            row["strata"]["0.5"]["metrics"],
            "cash_vs_original_PID",
            row["biased_cash_delta_vs_original_matched_pid"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.command]()
