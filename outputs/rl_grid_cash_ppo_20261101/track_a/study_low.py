"""Register and launch a matched, full-budget low-level PPO/SAC comparison."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# isort: off
from ppo_adapter import ACTION_CONTRACT
from economy_context import (
    ROOT as EXPERIMENT,
    PRIOR,
    PROJECT,
    SNAPSHOT_COMMIT,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)
import numpy as np
import pandas as pd
# isort: on

ROOT = Path(__file__).resolve().parent
REGISTRATION = ROOT / "registration.json"
RESERVATION = EXPERIMENT / "final_reservation.json"


def reserve_calendar(prior: dict) -> dict:
    if RESERVATION.exists():
        existing = read_json(RESERVATION)
        body = {
            key: value for key, value in existing.items() if key != "reservation_id"
        }
        if identity(body) != existing["reservation_id"]:
            raise ValueError("Reserved calendar identity changed")
        return existing
    split = prior["split"]
    consumed = set(split["previously_consumed_days"])
    consumed.update(split["new_development_days"])
    consumed.update(split["new_final_days"])
    invalid = set(split["coverage_audit"]["load_coverage_invalid_days"])
    invalid.update(split["coverage_audit"]["weather_coverage_invalid_days"])
    load_path = PROJECT / "data/processed/energy/NYC.csv"
    load = pd.read_csv(load_path)
    dates = pd.to_datetime(load["Time Stamp"])
    days = (dates.dt.normalize() - pd.Timestamp("2023-01-01")).dt.days
    peaks = load.groupby(days)["Load"].max()
    q25, q75, q90 = (float(peaks.quantile(value)) for value in (0.25, 0.75, 0.9))
    eligible = [
        day
        for day in range(365)
        if day not in invalid and all(abs(day - old) >= 2 for old in consumed)
    ]
    previously_eligible = [
        day
        for day in range(365)
        if day not in invalid
        and all(abs(day - old) >= 2 for old in split["previously_consumed_days"])
    ]
    if (
        len(previously_eligible)
        != split["coverage_audit"]["eligible_unconsumed_starts"]
    ):
        raise ValueError("The original exogenous coverage mask does not reproduce")
    if not eligible or any(peaks[day] >= q90 for day in eligible):
        raise ValueError("Unexpected remaining-calendar or p90-tail availability")
    rng = np.random.default_rng(202611013)
    selected, tiers = [], {}
    for tier in ("p75_tail_not_p90", "middle", "low"):
        candidates = [
            day
            for day in eligible
            if (
                (tier == "p75_tail_not_p90" and q75 <= peaks[day] < q90)
                or (tier == "middle" and q25 <= peaks[day] < q75)
                or (tier == "low" and peaks[day] < q25)
            )
        ]
        added = 0
        for day in rng.permutation(candidates):
            day = int(day)
            if all(abs(day - other) >= 2 for other in selected):
                selected.append(day)
                tiers[str(day)] = tier
                added += 1
            if added == 2:
                break
        if added != 2:
            raise ValueError(f"Cannot reserve the predetermined calendar tier: {tier}")
    cells = []
    noise_seeds = set()
    for day in sorted(selected):
        for replicate in range(3):
            seed = int(
                np.random.SeedSequence(
                    [20261101, 24017, day, replicate],
                ).generate_state(1)[0]
            )
            if seed in noise_seeds or seed < 1_000_000:
                raise ValueError(
                    "Unexpected collision in date-specific final noise seeds"
                )
            noise_seeds.add(seed)
            for bias in (0.0, 0.5):
                cells.append({"start_day": day, "seed": seed, "regd_load_bias": bias})
    reservation = {
        "schema": "grid-cash-calendar-combination-reservation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "days": sorted(selected),
        "tiers": tiers,
        "cells": cells,
        "load_path": str(load_path),
        "load_sha256": file_hash(load_path),
        "consumed_days_before_reservation": sorted(consumed),
        "inherited_invalid_26h_starts": sorted(invalid),
        "eligible_starts_before_reservation": len(eligible),
        "daily_peak_quantiles_mw": {"p25": q25, "p75": q75, "p90": q90},
        "selection_seed": 202611013,
        "closed_26h_windows_separated_by_at_least_two_start_days": True,
        "distinct_noise_streams_across_dates": True,
        "both_tracks_must_lock_models_before_any_opening": True,
        "global_opening_file": str(EXPERIMENT / "final_opening_grid_cash.json"),
        "no_policy_outcomes_opened_at_reservation": True,
        "scope": "Audited dc2grid studies, not every outside study",
        "not_an_unseen_workload_phase_test": True,
        "not_a_p90_tail_holdout": True,
    }
    reservation["reservation_id"] = identity(reservation)
    write_json(RESERVATION, reservation)
    return reservation


def prepare() -> dict:
    if REGISTRATION.exists():
        raise FileExistsError("Never replace a committed training registration")
    prior = read_json(PRIOR / "registration.json")
    headroom = read_json(EXPERIMENT / "headroom_registration.json")
    report = read_json(EXPERIMENT / "headroom_report.json")
    if (
        report["registration_id"] != headroom["registration_id"]
        or report["qualified_new_episodes"] != 144
    ):
        raise ValueError("The audited pre-learning headroom prerequisite is missing")
    reservation = reserve_calendar(prior)
    keys = (
        "config_path",
        "service_gate",
        "recovery",
        "normalization",
        "hardware_reference",
        "train_days",
        "train_biases",
        "service_floor",
        "initializers",
        "online_seed_by_initializer",
        "architecture",
    )
    protocol = {key: copy.deepcopy(prior[key]) for key in keys}
    source = dict(prior["source_hashes"])
    source.update(
        {
            str(path): file_hash(path)
            for path in (
                *sorted(EXPERIMENT.glob("*.py")),
                *sorted(ROOT.glob("*.py")),
            )
        }
    )
    inputs = dict(prior["input_hashes"])
    inputs[str(RESERVATION)] = file_hash(RESERVATION)
    inputs[str(EXPERIMENT / "headroom_report.json")] = file_hash(
        EXPERIMENT / "headroom_report.json",
    )
    baselines = {}
    for entry in headroom["frozen_controls"]:
        spec = copy.deepcopy(entry["policy"])
        old_id = spec["id"]
        spec["id"] = "rule__" + old_id.removeprefix("full_residual__")
        spec["macro"] = "rule"
        baselines[spec["id"]] = spec
        if "checkpoint" in spec:
            inputs[spec["checkpoint"]["path"]] = spec["checkpoint"]["sha256"]
    if len(baselines) != 22:
        raise ValueError("Expected all 22 frozen comparator policy IDs")
    all_old_seeds = {
        int(cell["seed"])
        for name in ("development_cells", "final_cells", "regression_cells")
        for cell in prior[name]
    }
    all_old_seeds.update(
        {90001, 90002, 90003, 90004, 91001, 91002, 92001, 92002, 92003}
    )
    all_old_seeds.update(cell["seed"] for cell in reservation["cells"])
    protocol.update(
        schema="native-low-ppo-sac-matched-v1",
        registration_file="track_a/registration.json",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        baseline_commit=SNAPSHOT_COMMIT,
        algorithms=["ppo", "sac"],
        seeds=[1100, 1101, 1102, 1103],
        initial_sigma=0.01,
        n_envs=4,
        warmup_transitions=1024,
        critic_calibration_updates=256,
        sac=copy.deepcopy(prior["algorithm"]),
        ppo={
            "learning_rate": 0.0001,
            "n_steps": 128,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "normalize_advantage": True,
        },
        ppo_action_contract=copy.deepcopy(ACTION_CONTRACT),
        budgets={"smoke": 4096, "full": 552960},
        milestones={"smoke": [4096], "full": [4096, 69120, 552960]},
        development_cells=copy.deepcopy(prior["development_cells"]),
        regression_cells=copy.deepcopy(prior["regression_cells"]),
        final_cells=copy.deepcopy(reservation["cells"]),
        final_reservation_id=reservation["reservation_id"],
        forbidden_eval_seeds=sorted(all_old_seeds),
        baseline_policies=list(baselines.values()),
        source_hashes=source,
        input_hashes=inputs,
        gpus=list(range(8)),
        cpu_groups=[list(range(index * 5, index * 5 + 5)) for index in range(8)],
        evaluation_cpus=list(range(56, 64)),
        threads_per_process=1,
        training={
            "physical_action_and_observation_contract_unchanged": True,
            "original_reward_unchanged": True,
            "true_finite_terminals_no_timeout_bootstrap": True,
            "recovery_in_training": False,
            "recovery_in_evaluation": "unchanged physical two-hour continuation",
            "actor_initialization": "same four BC means and initial distributions",
            "no_frozen_bc_or_kl_training_regularizer_in_either_arm": True,
            "sac_and_ppo_critics_are_not_identical": True,
            "ppo_calibration": "2 x 512 transitions; 64 critic epochs each",
            "sac_calibration": "256 critic steps after 1024 policy steps",
            "ppo_entropy_is_latent_and_zero_weight": True,
            "sac_entropy_fixed_coefficient": prior["algorithm"]["ent_coef"],
            "matched_transitions_not_matched_optimizer_updates_or_compute": True,
        },
        selection={
            "final_checkpoint_only": True,
            "all_four_model_seeds_required": True,
            "all_24_physical_and_recovery_cells_required": True,
            "tracking_reference": "matched own-BC development mean",
            "mean_rmse_noninferiority_tolerance_kw": 1e-6,
            "require_noninferiority_in_each_bias_stratum": True,
            "case_level_regressions_must_also_be_reported": True,
            "overall_improvement_requires_joint_facility_pcc_and_cash_gain": True,
            "joint_reference": "best qualified matched family, per metric",
            "cash_gain_threshold_usd": 5.0,
            "relief_gain_threshold_mw": 0.001,
            "both_bias_cash_gains_must_be_nonnegative": True,
            "not_a_claim_of_noninferiority_on_every_individual_case": True,
        },
    )
    protocol["registration_id"] = identity(protocol)
    write_json(REGISTRATION, protocol)
    verify(protocol, require_commit=False)
    return protocol


def launch(stage: str) -> None:
    protocol = read_json(REGISTRATION)
    commit = verify(protocol)
    if stage == "full":
        for algorithm in protocol["algorithms"]:
            receipt = read_json(ROOT / f"runs/smoke/{algorithm}_1100/completion.json")
            if (
                receipt["status"] != "completed"
                or receipt["registration_id"] != protocol["registration_id"]
                or receipt["num_timesteps"] != protocol["budgets"]["smoke"]
            ):
                raise ValueError("Both native training-lifecycle smokes must finish")
    status_path = ROOT / f"{stage}_status.json"
    logs = ROOT / f"{stage}_logs"
    if status_path.exists() or logs.exists():
        raise FileExistsError("Never invisibly retry or overwrite a training campaign")
    resources = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpu_resources = {
        int(parts[0]): (int(parts[1]), int(parts[2]))
        for parts in (
            line.replace(" ", "").split(",") for line in resources.splitlines()
        )
    }
    if any(
        gpu_resources[gpu][0] < 8192 or gpu_resources[gpu][1] > 80
        for gpu in protocol["gpus"]
    ):
        raise RuntimeError("A registered GPU lacks memory or is already too busy")
    logs.mkdir()
    processes, jobs = [], []
    try:
        for algorithm_index, algorithm in enumerate(protocol["algorithms"]):
            for seed_index, seed in enumerate(protocol["seeds"]):
                if stage == "smoke" and seed_index:
                    continue
                index = algorithm_index * 4 + seed_index
                gpu, cpus = protocol["gpus"][index], protocol["cpu_groups"][index]
                command = [
                    sys.executable,
                    str(ROOT / "train_low.py"),
                    "--protocol",
                    str(REGISTRATION),
                    "--algorithm",
                    algorithm,
                    "--seed",
                    str(seed),
                    "--stage",
                    stage,
                    "--device",
                    "cuda",
                    "--cpus",
                    *map(str, cpus),
                ]
                environment = os.environ.copy()
                environment.update(
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                with (logs / f"{algorithm}_{seed}.log").open("w") as handle:
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                processes.append(process)
                jobs.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "gpu": gpu,
                        "cpus": cpus,
                        "pid": process.pid,
                        "state": "running",
                    }
                )
        while True:
            for process, job in zip(processes, jobs, strict=True):
                code = process.poll()
                if code is not None:
                    job.update(
                        state="completed" if code == 0 else "failed", exit_code=code
                    )
            running = any(process.poll() is None for process in processes)
            write_json(
                status_path,
                {
                    "registration_id": protocol["registration_id"],
                    "implementation_commit": commit,
                    "stage": stage,
                    "state": "running" if running else "finished",
                    "jobs": jobs,
                    "updated_unix": time.time(),
                },
            )
            if not running:
                break
            time.sleep(5)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(
                "A native learner failed; all failed evidence is retained"
            )
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    verify(protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "full"))
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare()["registration_id"])
    else:
        launch(args.command)
