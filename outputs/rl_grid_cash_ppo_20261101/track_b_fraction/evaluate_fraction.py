"""Audit all final fractional-budget policies on the same consumed development cells."""

from __future__ import annotations

import argparse
import multiprocessing
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# isort: off
from run_fraction import ROOT, MacroLatentEnv
from economy_context import case_key, file_hash, identity, read_json, verify, write_json
from evaluate_low import audit_primary
from headroom import initialize_worker
from headroom_report import aggregate
from stable_baselines3 import PPO, SAC
import numpy as np
import pandas as pd
# isort: on


def evaluate_case(protocol, policy, cell, destination):
    from fraction_macro import FractionalDispatchTask

    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    task = FractionalDispatchTask(
        protocol,
        seed=0,
        fixed_cell=cell,
        record_trace=True,
        evaluation_mode=True,
        evaluation_cells=protocol["development_cells"],
    )
    environment = task
    try:
        artifact = policy["checkpoint"]
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Fractional policy checkpoint changed")
        model = (PPO if policy["algorithm"] == "ppo" else SAC).load(
            artifact["path"],
            device="cpu",
        )
        if (
            model.c2g_context["registration_id"] != protocol["registration_id"]
            or model.c2g_context["variant"] != policy["variant"]
            or model.gamma != 1.0
            or model.observation_space.shape != (23,)
            or model.action_space.shape != (1,)
        ):
            raise ValueError("Fractional checkpoint metadata differs from its contract")
        model.policy.set_training_mode(False)
        if policy["algorithm"] == "ppo":
            environment = MacroLatentEnv(task)
        observation, _ = environment.reset()
        while True:
            action = model.predict(observation, deterministic=True)[0]
            observation, _, terminated, truncated, _ = environment.step(action)
            if truncated:
                raise AssertionError("Fractional evaluation exposed a timeout")
            if terminated:
                break
        episode = {
            **task.episode_result,
            "policy": policy,
            "study_registration_id": protocol["registration_id"],
            "source_hashes": protocol["source_hashes"],
            "potential_used_at_evaluation": False,
        }
        hashes = {}
        for name, rows in task.terminal_traces().items():
            if rows:
                path = directory / name
                pd.DataFrame(rows).to_csv(
                    path, index=False, compression="gzip", mode="x"
                )
                hashes[name] = file_hash(path)
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


def inventory(protocol):
    policies, missing = [], []
    for variant in protocol["variants"]:
        for algorithm in protocol["algorithms"]:
            for seed in protocol["seeds"]:
                run = ROOT / f"runs/full/{variant}_{algorithm}_{seed}"
                if not (run / "completion.json").exists():
                    if not (run / "failure.json").exists():
                        raise RuntimeError("A fractional training slot is not finished")
                    missing.append(
                        {
                            "variant": variant,
                            "algorithm": algorithm,
                            "seed": seed,
                            "failure": read_json(run / "failure.json"),
                        }
                    )
                    continue
                receipt = read_json(run / "completion.json")
                if (
                    receipt["status"] != "completed"
                    or receipt["registration_id"] != protocol["registration_id"]
                    or receipt["macro_transitions"] != protocol["budgets"]["full"]
                ):
                    raise ValueError(
                        "Only mandatory final-step fractional models qualify"
                    )
                checkpoint = receipt["final_checkpoint"]
                policies.append(
                    {
                        "id": f"{variant}_{algorithm}_{seed}",
                        "family": f"{variant}_{algorithm}",
                        "variant": variant,
                        "algorithm": algorithm,
                        "training_seed": seed,
                        "checkpoint": {
                            "path": checkpoint["path"],
                            "sha256": checkpoint["sha256"],
                        },
                    }
                )
    return policies, missing


def run():
    protocol = read_json(ROOT / "registration.json")
    commit = verify(protocol)
    policies, missing = inventory(protocol)
    output = ROOT / "development"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "registration_id": protocol["registration_id"],
        "implementation_commit": commit,
        "role": "consumed_development_only",
        "policies": policies,
        "missing_training_slots": missing,
        "cells": protocol["development_cells"],
        "expected_episodes": len(policies) * len(protocol["development_cells"]),
    }
    manifest["manifest_id"] = identity(manifest)
    write_json(output / "manifest.json", manifest)
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
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["evaluation_cpus"]:
        queue.put(cpu)
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
                    f"{completed}/{manifest['expected_episodes']} "
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
        raise RuntimeError("Fractional evaluation contains failed or missing cases")


def report():
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    output = ROOT / "development"
    manifest = read_json(output / "manifest.json")
    status = read_json(output / "status.json")
    if (
        status["runtime_failures"]
        or status["completed"] != manifest["expected_episodes"]
    ):
        raise ValueError("Cannot summarize incomplete fractional evaluation")
    episodes, checks = [], 0
    for policy in manifest["policies"]:
        for cell in protocol["development_cells"]:
            directory = output / policy["id"] / case_key(cell)
            episode = read_json(directory / "episode.json")
            if (
                episode["policy"] != policy
                or episode["cell"] != cell
                or episode["study_registration_id"] != protocol["registration_id"]
                or episode["potential_used_at_evaluation"]
            ):
                raise ValueError("Fractional evaluation identity or deployment changed")
            checks += audit_primary(directory, episode)
            expected = (
                episode["excess_cash_usd"] / 100 if episode["qualified"] else -65400
            )
            np.testing.assert_allclose(episode["reward_sum"], expected, atol=1e-7)
            checks += 1
            episodes.append(episode)
    frozen = [
        read_json(item["path"])
        for item in (
            protocol["frozen_control_episodes"]
            + protocol["frozen_v1_economic_episodes"]
        )
    ]
    references = aggregate(frozen, protocol["development_cells"])
    eligible = {name: row for name, row in references.items() if row["all_qualified"]}
    if not eligible:
        raise ValueError("No fully qualified frozen reference family")
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
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["policy"]["family"]].append(episode)
    families, decisions = {}, {}
    for variant in protocol["variants"]:
        for algorithm in protocol["algorithms"]:
            family = f"{variant}_{algorithm}"
            records = grouped[family]
            model_count = len({row["policy"]["id"] for row in records})
            if model_count != 4:
                families[family] = {
                    "model_count": model_count,
                    "complete_cohort": False,
                    "all_qualified": False,
                    "strata": {},
                }
                decisions[family] = {
                    "joint_improvement": False,
                    "missing_model_seed": True,
                }
                continue
            row = aggregate(records, protocol["development_cells"])[family]
            families[family] = row
            contrasts, joint = {}, False
            if row["complete_cohort"]:
                contrasts = {
                    bias: {
                        key: row["strata"][bias]["metrics"][key]["mean"] - value
                        for key, value in envelope[bias].items()
                    }
                    for bias in envelope
                }
                joint = bool(
                    row["all_qualified"]
                    and contrasts["all"]["cashflow_26h_usd"]
                    >= protocol["selection"]["cash_gain_threshold_usd"]
                    and all(
                        contrasts[bias]["cashflow_26h_usd"] >= 0
                        for bias in ("0.0", "0.5")
                    )
                    and all(
                        contrasts["0.5"][key]
                        >= protocol["selection"]["relief_gain_threshold_mw"]
                        for key in ("stress_facility_relief_mw", "stress_pcc_relief_mw")
                    )
                )
            decisions[family] = {
                "joint_improvement": joint,
                "matched_envelope_contrasts": contrasts,
            }
    path = ROOT / "development_report.json"
    if path.exists():
        raise FileExistsError("Never overwrite fractional development outcomes")
    write_json(
        path,
        {
            "registration_id": protocol["registration_id"],
            "role": "adaptive_consumed_development_only",
            "raw_primary_checks": checks,
            "reference_families": references,
            "learned_families": families,
            "decisions": decisions,
            "missing_training_slots": status["missing_training_slots"],
            "potential_not_used_at_deployment": True,
            "final_data_not_opened": True,
        },
    )
    print(decisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    run() if args.command == "run" else report()
