"""Matched development evaluation of economic planners and scripted controls."""

from __future__ import annotations

import argparse
import multiprocessing
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# isort: off
from train_macro import EconomicDispatchTask, MacroLatentEnv
from economy_context import (
    case_key,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)
from evaluate_low import audit_primary
from headroom import initialize_worker
from headroom_report import aggregate
from stable_baselines3 import PPO, SAC
import numpy as np
import pandas as pd
# isort: on

ROOT = Path(__file__).resolve().parent


def scripted_action(policy, observation):
    mode = policy["mode"]
    if mode == "zero":
        value = 0.0
    elif mode == "one":
        value = 1 / 6
    elif mode == "greedy":
        value = 1.0
    elif mode == "margin":
        margin = float(observation[20]) * 200 - float(observation[17]) * 100
        value = float(margin >= policy["threshold_usd_mwh"])
    else:
        raise ValueError("Unknown registered scripted economic policy")
    return np.array([value], dtype=np.float32)


def evaluate_case(protocol, policy, cell, destination):
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    task = EconomicDispatchTask(
        protocol,
        seed=0,
        fixed_cell=cell,
        record_trace=True,
        evaluation_mode=True,
        evaluation_cells=protocol["development_cells"],
    )
    environment, model = task, None
    try:
        if policy["kind"] == "learned":
            artifact = policy["checkpoint"]
            if file_hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError("Economic checkpoint identity changed")
            algorithm = policy["algorithm"]
            model = (PPO if algorithm == "ppo" else SAC).load(
                artifact["path"],
                device="cpu",
            )
            if (
                model.c2g_context["registration_id"] != protocol["registration_id"]
                or model.gamma != 1.0
                or model.observation_space.shape != (23,)
                or model.action_space.shape != (1,)
            ):
                raise ValueError("Economic checkpoint task contract mismatch")
            model.policy.set_training_mode(False)
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
                raise AssertionError("Economic evaluation exposed a timeout terminal")
            if terminated:
                break
        episode = {
            **task.episode_result,
            "policy": policy,
            "study_registration_id": protocol["registration_id"],
            "source_hashes": protocol["source_hashes"],
        }
        hashes = {}
        for filename, rows in task.terminal_traces().items():
            if rows:
                path = directory / filename
                pd.DataFrame(rows).to_csv(
                    path, index=False, compression="gzip", mode="x"
                )
                hashes[filename] = file_hash(path)
        episode["trace_hashes"] = hashes
        write_json(directory / "episode.json", episode)
        return episode
    except Exception as error:
        write_json(
            directory / "failure.json",
            {
                "policy": policy,
                "cell": cell,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        environment.close()


def policy_inventory(protocol, group):
    if group == "baselines":
        return protocol["baseline_policies"], []
    policies, missing = [], []
    for algorithm in protocol["algorithms"]:
        for seed in protocol["seeds"]:
            directory = ROOT / f"runs/full/{algorithm}_{seed}"
            if not (directory / "completion.json").exists():
                if not (directory / "failure.json").exists():
                    raise RuntimeError("An economic training slot is not finished")
                missing.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "failure": read_json(directory / "failure.json"),
                    }
                )
                continue
            receipt = read_json(directory / "completion.json")
            if (
                receipt["status"] != "completed"
                or receipt["registration_id"] != protocol["registration_id"]
                or receipt["macro_transitions"] != protocol["budgets"]["full"]
            ):
                raise ValueError(
                    "Economic primary evaluation requires the final budget"
                )
            checkpoint = receipt["final_checkpoint"]
            policies.append(
                {
                    "id": f"economic_{algorithm}_{seed}",
                    "family": f"economic_{algorithm}",
                    "kind": "learned",
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "checkpoint": {
                        "path": checkpoint["path"],
                        "sha256": checkpoint["sha256"],
                    },
                }
            )
    return policies, missing


def run(group):
    protocol = read_json(ROOT / "registration.json")
    commit = verify(protocol)
    policies, missing = policy_inventory(protocol, group)
    output = ROOT / f"development_{group}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "registration_id": protocol["registration_id"],
        "implementation_commit": commit,
        "group": group,
        "role": "consumed_development_only",
        "policies": policies,
        "missing_training_slots": missing,
        "cells": protocol["development_cells"],
        "expected_episodes": len(policies) * len(protocol["development_cells"]),
    }
    manifest["manifest_id"] = identity(manifest)
    write_json(output / "manifest.json", manifest)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["evaluation_cpus"]:
        queue.put(cpu)
    completed, failures = 0, []
    write_json(
        output / "status.json",
        {
            "completed": 0,
            "expected": manifest["expected_episodes"],
            "runtime_failures": [],
            "missing_training_slots": missing,
        },
    )
    with ProcessPoolExecutor(
        max_workers=len(protocol["evaluation_cpus"]),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as pool:
        futures = {}
        for policy in policies:
            for cell in protocol["development_cells"]:
                destination = output / policy["id"] / case_key(cell)
                future = pool.submit(
                    evaluate_case,
                    protocol,
                    policy,
                    cell,
                    str(destination),
                )
                futures[future] = str(destination)
        for future in as_completed(futures):
            try:
                episode = future.result()
                completed += 1
                print(
                    f"{group} {completed}/{manifest['expected_episodes']} "
                    f"{episode['policy']['id']} {case_key(episode['cell'])} "
                    f"qualified={episode['qualified']}",
                    flush=True,
                )
            except Exception as error:
                failure = {"directory": futures[future], "error": repr(error)}
                failures.append(failure)
                print(f"FAILED: {failure}", flush=True)
            write_json(
                output / "status.json",
                {
                    "completed": completed,
                    "expected": manifest["expected_episodes"],
                    "runtime_failures": failures,
                    "missing_training_slots": missing,
                },
            )
    queue.close()
    queue.join_thread()
    verify(protocol)
    if failures or completed != manifest["expected_episodes"]:
        raise RuntimeError(
            "Economic development evaluation has failed or missing cases"
        )


def collect(protocol, group):
    directory = ROOT / f"development_{group}"
    manifest = read_json(directory / "manifest.json")
    status = read_json(directory / "status.json")
    if (
        status["runtime_failures"]
        or status["completed"] != manifest["expected_episodes"]
    ):
        raise ValueError("Cannot summarize incomplete economic evaluation")
    episodes, checks = [], 0
    for policy in manifest["policies"]:
        for cell in protocol["development_cells"]:
            path = directory / policy["id"] / case_key(cell)
            episode = read_json(path / "episode.json")
            if (
                episode["policy"] != policy
                or episode["cell"] != cell
                or episode["study_registration_id"] != protocol["registration_id"]
            ):
                raise ValueError("Economic episode identity changed")
            checks += audit_primary(path, episode)
            if episode["qualified"]:
                np.testing.assert_allclose(
                    episode["reward_sum"],
                    episode["excess_cash_usd"] / 100,
                    atol=1e-7,
                )
                if episode["delivered_service_ratio"] < 0.90:
                    raise AssertionError(
                        "A qualified economic episode misses the floor"
                    )
            elif episode["true_terminal"]:
                np.testing.assert_allclose(episode["reward_sum"], -65400, atol=1e-7)
            checks += 1
            episodes.append(episode)
    return episodes, manifest["missing_training_slots"], checks


def report():
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    controls, _, checks = collect(protocol, "baselines")
    learned, missing, count = collect(protocol, "full")
    checks += count
    frozen = [read_json(item["path"]) for item in protocol["frozen_control_episodes"]]
    baseline = aggregate(controls + frozen, protocol["development_cells"])
    by_family = defaultdict(list)
    for episode in learned:
        by_family[episode["policy"]["family"]].append(episode)
    families = {}
    for algorithm in protocol["algorithms"]:
        name = f"economic_{algorithm}"
        records = by_family[name]
        models = {row["policy"]["id"] for row in records}
        if len(models) != 4:
            families[name] = {
                "model_count": len(models),
                "complete_cohort": False,
                "all_qualified": False,
                "strata": {},
                "reason": "missing_training_seed_no_survivor_mean",
            }
        else:
            families[name] = aggregate(records, protocol["development_cells"])[name]
    eligible = {name: row for name, row in baseline.items() if row["all_qualified"]}
    if not eligible:
        raise ValueError("No qualified economic reference family")
    metrics = ("cashflow_26h_usd", "stress_facility_relief_mw", "stress_pcc_relief_mw")
    envelope = {
        bias: {
            key: max(
                row["strata"][bias]["metrics"][key]["mean"] for row in eligible.values()
            )
            for key in metrics
        }
        for bias in ("all", "0.0", "0.5")
    }
    decisions = {}
    for family, row in families.items():
        contrasts, qualified = {}, False
        if row["complete_cohort"]:
            contrasts = {
                bias: {
                    key: row["strata"][bias]["metrics"][key]["mean"] - value
                    for key, value in envelope[bias].items()
                }
                for bias in envelope
            }
            qualified = bool(
                row["all_qualified"]
                and contrasts["all"]["cashflow_26h_usd"]
                >= protocol["selection"]["cash_gain_threshold_usd"]
                and all(
                    contrasts[bias]["cashflow_26h_usd"] >= 0 for bias in ("0.0", "0.5")
                )
                and all(
                    contrasts["0.5"][key]
                    >= protocol["selection"]["relief_gain_threshold_mw"]
                    for key in ("stress_facility_relief_mw", "stress_pcc_relief_mw")
                )
            )
        decisions[family] = {
            "joint_improvement": qualified,
            "matched_envelope_contrasts": contrasts,
        }
    path = ROOT / "development_report.json"
    if path.exists():
        raise FileExistsError("Never overwrite an economic development report")
    write_json(
        path,
        {
            "registration_id": protocol["registration_id"],
            "role": "consumed_development_only",
            "raw_primary_checks": checks,
            "scripted_and_frozen_families": baseline,
            "learned_families": families,
            "decisions": decisions,
            "missing_training_slots": missing,
            "joint_reference_includes_new_scripted_economic_baselines": True,
            "not_the_original_tracking_benchmark": True,
            "final_data_not_opened": True,
        },
    )
    print(decisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("baselines", "full", "report"))
    args = parser.parse_args()
    if args.command == "report":
        report()
    else:
        run(args.command)
