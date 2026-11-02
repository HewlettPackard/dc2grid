"""Freeze and run the matched economic-dispatch PPO/SAC experiment."""

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
from train_macro import EconomicDispatchTask  # noqa: F401
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
# isort: on

ROOT = Path(__file__).resolve().parent
REGISTRATION = ROOT / "registration.json"


def prepare():
    if REGISTRATION.exists():
        raise FileExistsError("Never replace an economic training registration")
    prior = read_json(PRIOR / "registration.json")
    low = read_json(EXPERIMENT / "track_a/registration.json")
    headroom = read_json(EXPERIMENT / "headroom_registration.json")
    headroom_result = read_json(EXPERIMENT / "headroom_report.json")
    if headroom_result["qualified_new_episodes"] != 144:
        raise ValueError("Raw-verified physical headroom is required before training")
    keys = (
        "config_path",
        "service_gate",
        "recovery",
        "normalization",
        "train_days",
        "train_biases",
        "service_floor",
    )
    protocol = {key: copy.deepcopy(prior[key]) for key in keys}
    sources = dict(low["source_hashes"])
    sources.update({str(path): file_hash(path) for path in sorted(ROOT.glob("*.py"))})
    inputs = {**headroom["input_hashes"], **low["input_hashes"]}
    baselines = [
        {"id": "analytic", "family": "analytic", "kind": "scripted", "mode": "zero"},
        {
            "id": "constant_one",
            "family": "constant_one",
            "kind": "scripted",
            "mode": "one",
        },
        {
            "id": "greedy_budget",
            "family": "greedy_budget",
            "kind": "scripted",
            "mode": "greedy",
        },
        {
            "id": "margin_ten",
            "family": "margin_ten",
            "kind": "scripted",
            "mode": "margin",
            "threshold_usd_mwh": 10.0,
        },
    ]
    protocol.update(
        schema="economic-dispatch-native-ppo-sac-v1",
        registration_file="track_b/registration.json",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        baseline_commit=SNAPSHOT_COMMIT,
        algorithms=["ppo", "sac"],
        seeds=[1200, 1201, 1202, 1203],
        online_seed_by_seed={
            str(seed): 7200 + seed - 1200 for seed in range(1200, 1204)
        },
        n_envs=2,
        architecture=[128, 128],
        initial_sigma=0.05,
        warmup_transitions=256,
        critic_calibration_updates=256,
        sac={
            "learning_rate": 0.0001,
            "buffer_size": 100000,
            "batch_size": 64,
            "learning_starts": 256,
            "tau": 0.005,
            "gamma": 1.0,
            "ent_coef": 0.0001,
            "train_freq": 4,
            "gradient_steps": -1,
        },
        ppo={
            "learning_rate": 0.0001,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "normalize_advantage": True,
        },
        budgets={"smoke": 512, "full": 24576},
        milestones={"smoke": [512], "full": [512, 3072, 12288, 24576]},
        development_cells=copy.deepcopy(low["development_cells"]),
        regression_cells=copy.deepcopy(low["regression_cells"]),
        final_cells=copy.deepcopy(low["final_cells"]),
        final_reservation_id=low["final_reservation_id"],
        forbidden_eval_seeds=copy.deepcopy(low["forbidden_eval_seeds"]),
        baseline_policies=baselines,
        frozen_control_episodes=copy.deepcopy(headroom["frozen_controls"]),
        source_hashes=sources,
        input_hashes=inputs,
        cpu_groups=[list(range(40 + index * 3, 43 + index * 3)) for index in range(8)],
        evaluation_cpus=list(range(40, 56)),
        threads_per_process=1,
        device="cpu",
        full_training_waits_for_track_a_baselines_to_release_cpus=True,
        task={
            "observation_dimension": 23,
            "action_dimension": 1,
            "native_action_bounds": [-1.0, 1.0],
            "ppo_gaussian_latent_bounds": [-20.0, 20.0],
            "ppo_transform": "tanh and affine native-box map, once at the environment",
            "gamma": 1.0,
            "reward_scale_usd": 100.0,
            "internal_credit_target": 0.92,
            "credit_reserve_mwh": 0.02,
            "maximum_requested_extra_mw": 6.0,
            "market_policy": "unchanged full residual, zero ask",
            "full_work": True,
            "nominal_pump": 0.7,
            "soc_target": 0.5,
            "original_analytic_thermal_and_bess_compensation_retained": True,
            "terminal_inventory_accumulation_is_not_an_action": True,
            "two_hour_physical_recovery_in_training_and_evaluation": True,
            "true_terminal_without_timeout_bootstrap": True,
            "reference_subtraction_is_action_independent_at_fixed_bids": True,
            "different_objective_from_original_tracking_benchmark": True,
        },
        selection={
            "final_checkpoint_only": True,
            "all_four_seeds_and_all_24_cases_required": True,
            "require_actual_90pct_service_and_unchanged_physical_recovery_gates": True,
            "reference": "qualified scripted/frozen families, per-metric envelope",
            "cash_gain_threshold_usd": 5.0,
            "relief_gain_threshold_mw": 0.001,
            "primary_relief_bias": 0.5,
            "both_bias_cash_contrasts_must_be_nonnegative": True,
            "no_survivor_only_family_mean": True,
            "tracking_deviation_and_battery_reallocation_must_be_disclosed": True,
            "shared_final_opening_requires_both_tracks_locked": True,
        },
    )
    protocol["registration_id"] = identity(protocol)
    write_json(REGISTRATION, protocol)
    verify(protocol, require_commit=False)
    return protocol


def launch(stage):
    protocol = read_json(REGISTRATION)
    commit = verify(protocol)
    if stage == "full":
        baseline = read_json(EXPERIMENT / "track_a/development_baselines/status.json")
        if (
            baseline["runtime_failures"]
            or baseline["completed"] != baseline["expected"]
        ):
            raise RuntimeError(
                "Track A baseline workers must release the shared CPU range"
            )
        for algorithm in protocol["algorithms"]:
            receipt = read_json(ROOT / f"runs/smoke/{algorithm}_1200/completion.json")
            if (
                receipt["status"] != "completed"
                or receipt["registration_id"] != protocol["registration_id"]
                or receipt["macro_transitions"] != protocol["budgets"]["smoke"]
            ):
                raise ValueError("Both economic training-lifecycle smokes must finish")
    status = ROOT / f"{stage}_status.json"
    logs = ROOT / f"{stage}_logs"
    if status.exists() or logs.exists():
        raise FileExistsError("Never invisibly retry or overwrite an economic campaign")
    logs.mkdir()
    processes, jobs = [], []
    try:
        for algorithm_index, algorithm in enumerate(protocol["algorithms"]):
            for seed_index, seed in enumerate(protocol["seeds"]):
                if stage == "smoke" and seed_index:
                    continue
                index = algorithm_index * 4 + seed_index
                cpus = protocol["cpu_groups"][index]
                command = [
                    sys.executable,
                    str(ROOT / "train_macro.py"),
                    "--protocol",
                    str(REGISTRATION),
                    "--algorithm",
                    algorithm,
                    "--seed",
                    str(seed),
                    "--stage",
                    stage,
                    "--cpus",
                    *map(str, cpus),
                ]
                environment = os.environ.copy()
                environment.update(
                    CUDA_VISIBLE_DEVICES="",
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
                        "cpus": cpus,
                        "pid": process.pid,
                        "state": "running",
                        "device": "cpu",
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
                status,
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
            raise RuntimeError("Economic learner failures are preserved")
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
