"""Evaluate locked final checkpoints on consumed Dev-B without opening final data."""

from __future__ import annotations

import argparse
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# isort: off
from ppo_adapter import HistoryPPOController
from economy_context import case_key, file_hash, identity, read_json, verify, write_json
from economy_eval import evaluate_episode
from evaluate import make_hardware
from headroom import initialize_worker
from low_learn import summarize
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
# isort: on

ROOT = Path(__file__).resolve().parent


def hardware(spec, protocol):
    if spec["hardware"] != "history_ppo":
        return make_hardware(spec, protocol)
    artifact = spec["checkpoint"]
    if file_hash(artifact["path"]) != artifact["sha256"]:
        raise ValueError("PPO checkpoint identity changed")
    model = PPO.load(artifact["path"], device="cpu")
    if model.c2g_context["registration_id"] != protocol["registration_id"]:
        raise ValueError("PPO checkpoint belongs to another experiment")
    model.policy.set_training_mode(False)
    return HistoryPPOController(model)


def evaluate_worker(protocol, policy, cell, destination):
    return evaluate_episode(
        protocol,
        policy,
        cell,
        Path(destination),
        hardware_factory=hardware,
    )


def policy_inventory(protocol, group):
    if group == "baselines":
        return protocol["baseline_policies"], []
    policies, missing = [], []
    for algorithm in protocol["algorithms"]:
        for seed in protocol["seeds"]:
            directory = ROOT / f"runs/full/{algorithm}_{seed}"
            completion = directory / "completion.json"
            if not completion.exists():
                failure = directory / "failure.json"
                if not failure.exists():
                    raise RuntimeError(
                        "Do not evaluate an unfinished training campaign"
                    )
                missing.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "failure": read_json(failure),
                    }
                )
                continue
            result = read_json(completion)
            if (
                result["registration_id"] != protocol["registration_id"]
                or result["num_timesteps"] != protocol["budgets"]["full"]
                or result["status"] != "completed"
            ):
                raise ValueError(
                    "Only the registered final training step is admissible"
                )
            checkpoint = result["final_checkpoint"]
            policies.append(
                {
                    "id": f"rule__native_{algorithm}_{seed}",
                    "family": f"native_{algorithm}",
                    "training_seed": seed,
                    "hardware": "history_ppo" if algorithm == "ppo" else "history_sac",
                    "macro": "rule",
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
        "role": "consumed_development_only",
        "group": group,
        "policies": policies,
        "missing_training_slots": missing,
        "cells": protocol["development_cells"],
        "expected_episodes": len(policies) * len(protocol["development_cells"]),
        "final_checkpoint_only": True,
    }
    manifest["manifest_id"] = identity(manifest)
    write_json(output / "manifest.json", manifest)
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
    completed, failures = 0, []
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
                    evaluate_worker,
                    protocol,
                    policy,
                    cell,
                    str(destination),
                )
                futures[future] = str(destination)
        for future in as_completed(futures):
            try:
                result = future.result()
                completed += 1
                print(
                    f"{group} {completed}/{manifest['expected_episodes']} "
                    f"{result['policy']['id']} {case_key(result['cell'])} "
                    f"qualified={result['qualified']}",
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
        raise RuntimeError("Development evaluation is incomplete; failures retained")


def audit_primary(directory, episode):
    frames = {}
    for name, digest in episode["trace_hashes"].items():
        if file_hash(directory / name) != digest:
            raise ValueError("Raw evaluation trace changed")
        frames[name] = pd.read_csv(directory / name)
    service = frames["hardware.csv.gz"]
    market = frames["macro.csv.gz"]
    checks, start = 0, 0
    for row in market.itertuples():
        count = int(row.sub_steps_run)
        window = service.iloc[start : start + count]
        performance = max(
            0.0,
            1
            - float(np.abs(window.tracking_err_kw).mean()) / (row.committed_mw * 1000),
        )
        np.testing.assert_allclose(performance, row.perf_score, atol=1e-8)
        np.testing.assert_allclose(
            row.rate * row.committed_mw * performance * row.duration_hours,
            row.regulation_revenue,
            atol=1e-7,
        )
        start += count
        checks += 2
    if start != len(service):
        raise AssertionError("Market and hardware durations differ")
    for frame in (frames[name] for name in frames if name != "macro.csv.gz"):
        np.testing.assert_allclose(
            frame.p_grid_import_mw,
            frame.p_facility_mw - frame.bess_actual_kw / 1000,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            frame.delta_p_actual_kw,
            (frame.p_reference_mw - frame.p_facility_mw) * 1000 + frame.bess_actual_kw,
            atol=1e-6,
        )
        checks += 2
    if not episode["complete"] or not episode["recovery"]["complete"]:
        if episode["qualified"] or episode["cashflow_26h_usd"] is not None:
            raise AssertionError("An incomplete episode was treated as complete")
        return checks
    recovery = frames["recovery.csv.gz"]
    if len(service) != 17280 or len(recovery) != 1440:
        raise AssertionError("The matched physical horizon changed")
    electricity = sum(
        float(np.dot(frame.p_grid_import_mw * frame.lmp, frame.duration_hours))
        for frame in (service, recovery)
    )
    np.testing.assert_allclose(
        market.regulation_revenue.sum() - electricity,
        episode["cashflow_26h_usd"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.sqrt(np.mean(service.tracking_err_kw.to_numpy() ** 2)),
        episode["service"]["tracking_rmse_kw"],
        atol=1e-6,
    )
    stress = service.grid_load_mw >= service.grid_load_mw.quantile(0.9)
    for column, key in (
        ("p_facility_mw", "stress_facility_relief_mw"),
        ("p_grid_import_mw", "stress_pcc_relief_mw"),
    ):
        np.testing.assert_allclose(
            (service.p_reference_mw - service[column])[stress].mean(),
            episode["grid"][key],
            atol=1e-8,
        )
    return checks + 4


def collect_group(protocol, group):
    directory = ROOT / f"development_{group}"
    manifest = read_json(directory / "manifest.json")
    status = read_json(directory / "status.json")
    if (
        status["runtime_failures"]
        or status["completed"] != manifest["expected_episodes"]
    ):
        raise ValueError("Cannot aggregate missing or runtime-failed evaluation cases")
    by_policy, raw_checks = {}, 0
    for policy in manifest["policies"]:
        episodes = []
        for cell in protocol["development_cells"]:
            path = directory / policy["id"] / case_key(cell)
            episode = read_json(path / "episode.json")
            if episode["policy"] != policy or episode["cell"] != cell:
                raise ValueError(
                    "Episode policy or condition differs from its manifest"
                )
            raw_checks += audit_primary(path, episode)
            episodes.append(episode)
        by_policy[policy["id"]] = episodes
    return by_policy, manifest["missing_training_slots"], raw_checks


def family_means(by_policy, expected_models=None):
    families = defaultdict(list)
    for episodes in by_policy.values():
        families[episodes[0]["policy"]["family"]].append(episodes)
    output = {}
    for family, models in families.items():
        complete = all(
            row["complete"] and row["recovery"]["complete"]
            for rows in models
            for row in rows
        )
        if expected_models is not None and len(models) != expected_models[family]:
            complete = False
        qualified = all(row["qualified"] for rows in models for row in rows)
        strata = {}
        if complete:
            for bias in ("all", "0.0", "0.5"):
                selected = [
                    row
                    for rows in models
                    for row in rows
                    if bias == "all" or row["cell"]["regd_load_bias"] == float(bias)
                ]
                strata[bias] = {
                    "rmse_kw": float(
                        np.mean(
                            [row["service"]["tracking_rmse_kw"] for row in selected]
                        )
                    ),
                    "cash_usd": float(
                        np.mean([row["cashflow_26h_usd"] for row in selected])
                    ),
                    "facility_relief_mw": float(
                        np.mean(
                            [
                                row["grid"]["stress_facility_relief_mw"]
                                for row in selected
                            ]
                        )
                    ),
                    "pcc_relief_mw": float(
                        np.mean(
                            [row["grid"]["stress_pcc_relief_mw"] for row in selected]
                        )
                    ),
                }
        output[family] = {
            "model_count": len(models),
            "all_qualified": qualified,
            "complete_cohort": complete,
            "strata": strata,
        }
    return output


def report():
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    baseline, _, checks = collect_group(protocol, "baselines")
    learned, missing, count = collect_group(protocol, "full")
    checks += count
    frozen_families = family_means(baseline)
    families = family_means(
        learned,
        {f"native_{name}": 4 for name in protocol["algorithms"]},
    )
    eligible = {
        name: values
        for name, values in frozen_families.items()
        if values["all_qualified"]
    }
    if not eligible:
        raise ValueError("No fully qualified matched reference family")
    envelope = {
        bias: {
            metric: max(row["strata"][bias][metric] for row in eligible.values())
            for metric in ("cash_usd", "facility_relief_mw", "pcc_relief_mw")
        }
        for bias in ("all", "0.0", "0.5")
    }
    per_model = {}
    tolerance = protocol["selection"]["mean_rmse_noninferiority_tolerance_kw"]
    for name, rows in learned.items():
        seed = rows[0]["policy"]["training_seed"]
        reference = baseline[f"rule__bc_only_{seed}"]
        result = summarize(rows)
        differences = {}
        if all(row["complete"] for row in rows):
            for bias in (0.0, 0.5):
                differences[str(bias)] = float(
                    np.mean(
                        [
                            row["service"]["tracking_rmse_kw"]
                            - other["service"]["tracking_rmse_kw"]
                            for row, other in zip(rows, reference, strict=True)
                            if row["cell"]["regd_load_bias"] == bias
                        ]
                    )
                )
        result["rmse_change_vs_own_bc_by_bias_kw"] = differences
        result["tracking_noninferior"] = bool(
            result["all_qualified"]
            and differences
            and all(value <= tolerance for value in differences.values())
        )
        result["individual_cases_worse_than_bc"] = [
            case_key(row["cell"])
            for row, other in zip(rows, reference, strict=True)
            if row["complete"]
            and row["service"]["tracking_rmse_kw"]
            > other["service"]["tracking_rmse_kw"] + tolerance
        ]
        per_model[name] = result
    decisions = {}
    for algorithm in protocol["algorithms"]:
        family = f"native_{algorithm}"
        slots = [f"rule__{family}_{seed}" for seed in protocol["seeds"]]
        complete = all(name in per_model for name in slots)
        tracking = complete and all(
            per_model[name]["tracking_noninferior"] for name in slots
        )
        row = families.get(family)
        joint = False
        contrasts = {}
        if complete and row["complete_cohort"]:
            contrasts = {
                bias: {
                    metric: row["strata"][bias][metric] - value
                    for metric, value in envelope[bias].items()
                }
                for bias in envelope
            }
            joint = bool(
                tracking
                and contrasts["all"]["cash_usd"]
                >= protocol["selection"]["cash_gain_threshold_usd"]
                and all(contrasts[bias]["cash_usd"] >= 0 for bias in ("0.0", "0.5"))
                and all(
                    contrasts["0.5"][metric]
                    >= protocol["selection"]["relief_gain_threshold_mw"]
                    for metric in ("facility_relief_mw", "pcc_relief_mw")
                )
            )
        decisions[algorithm] = {
            "all_four_final_models_evaluated": complete,
            "all_four_noninferior_to_own_bc": tracking,
            "joint_improvement": joint,
            "contrasts_vs_frozen_family_envelope": contrasts,
        }
    path = ROOT / "development_report.json"
    if path.exists():
        raise FileExistsError("Never overwrite a completed development report")
    write_json(
        path,
        {
            "registration_id": protocol["registration_id"],
            "role": "consumed_development_only",
            "per_model": per_model,
            "frozen_families": frozen_families,
            "learned_families": families,
            "missing_training_slots": missing,
            "decisions": decisions,
            "raw_primary_checks": checks,
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
