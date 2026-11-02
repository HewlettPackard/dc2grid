"""Register a bounded action-coordinate and invariant-potential follow-up."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# isort: off
from run_fraction import ROOT
from economy_context import (
    ROOT as EXPERIMENT,
    PROJECT,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)
import numpy as np
# isort: on

REGISTRATION = ROOT / "registration.json"
V1_RESULT_COMMIT = "5e2bf2855a3da4ac8cfea1763df8e9d8b28e9f82"


def prepare():
    if REGISTRATION.exists():
        raise FileExistsError("Never replace a fractional-budget registration")
    prior = read_json(EXPERIMENT / "track_b/registration.json")
    report_path = EXPERIMENT / "track_b/development_report.json"
    report = read_json(report_path)
    if any(row["joint_improvement"] for row in report["decisions"].values()):
        raise ValueError(
            "This follow-up is registered against the observed V1 non-advancement"
        )
    protocol = copy.deepcopy(prior)
    del protocol["registration_id"]
    sources = dict(prior["source_hashes"])
    sources.update({str(path): file_hash(path) for path in sorted(ROOT.glob("*.py"))})
    inputs = dict(prior["input_hashes"])
    inputs[str(report_path)] = file_hash(report_path)
    coverage = EXPERIMENT / "reports/calendar_coverage_verification.json"
    inputs[str(coverage)] = file_hash(coverage)
    parent_sources = [
        EXPERIMENT.parent / "rl_state_local_20261101/registration.json",
        EXPERIMENT.parent / "rl_value_conditioning_20261101/registration.json",
        EXPERIMENT / "headroom_registration.json",
        EXPERIMENT / "track_a/registration.json",
        EXPERIMENT / "track_b/registration.json",
    ]
    parent_registry = []
    for path in parent_sources:
        record = read_json(path)
        digest = file_hash(path)
        inputs[str(path)] = digest
        parent_registry.append(
            {
                "path": str(path),
                "sha256": digest,
                "registration_id": record["registration_id"],
            }
        )
    frozen = []
    for group in ("baselines", "full"):
        directory = EXPERIMENT / "track_b" / f"development_{group}"
        for path in sorted(directory.glob("*/*/episode.json")):
            row = read_json(path)
            artifact = {
                "path": str(path),
                "sha256": file_hash(path),
                "policy": row["policy"],
                "cell": row["cell"],
            }
            frozen.append(artifact)
            inputs[str(path)] = artifact["sha256"]
    if len(frozen) != 288:
        raise ValueError("All V1 economic controls and learned outcomes are required")
    protocol.update(
        schema="economic-window-fraction-potential-factorial-v1",
        registration_file="track_b_fraction/registration.json",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        parent_registration_id=prior["registration_id"],
        preceding_result_commit=V1_RESULT_COMMIT,
        variants=["fraction", "potential"],
        potential_usd_per_mwh=10.0,
        source_hashes=sources,
        input_hashes=inputs,
        frozen_v1_economic_episodes=frozen,
        cpu_groups=[list(range(index * 3, index * 3 + 3)) for index in range(16)],
        evaluation_cpus=list(range(48, 64)),
        full_training_waits_for_track_a_baselines_to_release_cpus=False,
        adaptation_disclosure={
            "informed_by_consumed_development_v1_results": True,
            "observed_v1_models_near_greedy_but_below_margin10_cash": True,
            "potential_coefficient_from_registered_margin10_baseline": True,
            "no_claim_of_fresh_development_validation": True,
            "same_unopened_calendar_reservation": prior["final_reservation_id"],
        },
        adaptive_attempt_ledger={
            "scope": "audited Dev-B; checkpoint scans in parent manifests",
            "parent_registrations": parent_registry,
            "attempts": [
                {
                    "campaign": "state_local_fixed_and_replay_KL",
                    "families": 2,
                    "seeds_per_family": 4,
                    "pilot_transitions": 69120,
                    "extension_not_earned": True,
                },
                {
                    "campaign": "union_original_and_conditioned",
                    "families": 2,
                    "seeds_per_family": 4,
                    "pilot_transitions": 69120,
                    "final_transitions": 552960,
                    "repeated_prefixes_not_independent_replications": True,
                },
                {
                    "campaign": "causal_quantity_and_terminal_surplus",
                    "prefix_configurations": 3,
                    "new_terminal_surplus_candidate": 1,
                    "fixed_hardware": "bc_only_1100",
                    "not_promoted": True,
                },
                {
                    "campaign": "economic_headroom",
                    "new_nonlearned_probes": 5,
                    "repeated_analytic_control": 1,
                    "not_a_learned_policy_result": True,
                },
                {
                    "campaign": "native_low_PPO_SAC",
                    "families": 2,
                    "seeds_per_family": 4,
                    "transitions": 552960,
                    "not_promoted": True,
                },
                {
                    "campaign": "native_economic_PPO_SAC",
                    "families": 2,
                    "seeds_per_family": 4,
                    "macro_transitions": 24576,
                    "scripted_comparators": 4,
                    "not_promoted": True,
                },
                {
                    "campaign": "current_fraction_potential_factorial",
                    "families": 4,
                    "seeds_per_family": 4,
                    "macro_transitions": 24576,
                    "smoke_prefixes_not_independent_replications": True,
                },
            ],
            "maximum_additional_adaptive_campaigns_before_this_opening": 1,
            "this_factorial_consumes_that_one_campaign": True,
            "failure_keeps_the_new_final_reservation_unopened": True,
            "do_not_treat_repeated_Dev_B_as_fresh_validation": True,
        },
        coordinate={
            "fraction": "max(native_action,0)",
            "allowance": "first actual hardware tick after the bid handshake",
            "requested_extra": "fraction times frozen start-window allowance",
            "request_held_constant_for_current_window": True,
            "original_per_tick_guard_retained": True,
            "not_fraction_of_continually_recomputed_cap": True,
            "observation_dimension_unchanged": 23,
            "physical_initial_exploration_differs_from_absolute_coordinate": True,
        },
        shaping={
            "formula": "r+Phi(next)*(1-done)-Phi(current)",
            "phi": "10USD/MWh * observed_credit_MWh /100USD",
            "gamma": 1.0,
            "terminal_next_potential": 0.0,
            "initial_credit_and_potential": 0.0,
            "preserves_complete_and_invalid_finite_episode_returns": True,
            "raw_money_and_original_task_rewards_retained": True,
            "potential_not_used_at_deployment": True,
        },
    )
    protocol["task"]["action_coordinate"] = protocol["coordinate"]
    protocol["selection"]["reference"] = "all qualified V1 and scripted families"
    protocol["selection"]["headline_choice"] = (
        "largest development cash among joint-eligible whole four-seed families"
    )
    protocol["registration_id"] = identity(protocol)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", V1_RESULT_COMMIT, "HEAD"],
        cwd=PROJECT,
        check=True,
    )
    write_json(REGISTRATION, protocol)
    verify(protocol, require_commit=False)
    return protocol


def check_smokes(protocol):
    receipts = {}
    for variant in protocol["variants"]:
        for algorithm in protocol["algorithms"]:
            directory = ROOT / f"runs/smoke/{variant}_{algorithm}_1200"
            receipt = read_json(directory / "completion.json")
            if (
                receipt["status"] != "completed"
                or receipt["registration_id"] != protocol["registration_id"]
                or receipt["macro_transitions"] != protocol["budgets"]["smoke"]
            ):
                raise ValueError("All four factorial smoke slots must finish")
            artifact = receipt["all_macro_transitions"]
            if file_hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError("Recorded factorial warmup transitions changed")
            receipts[variant, algorithm] = receipt
    compared = []
    for algorithm in protocol["algorithms"]:
        with (
            np.load(
                receipts["fraction", algorithm]["all_macro_transitions"]["path"]
            ) as raw,
            np.load(
                receipts["potential", algorithm]["all_macro_transitions"]["path"]
            ) as shaped,
        ):
            steps = protocol["warmup_transitions"] // protocol["n_envs"]
            for name in ("observations", "next_observations", "actions", "dones"):
                np.testing.assert_array_equal(raw[name][:steps], shaped[name][:steps])
            coefficient = protocol["potential_usd_per_mwh"]
            current = (
                raw["observations"][:steps, :, 21].astype(np.float64)
                * 30
                * coefficient
                / 100
            )
            following = (
                raw["next_observations"][:steps, :, 21].astype(np.float64)
                * 30
                * coefficient
                / 100
            )
            expected = (
                raw["rewards"][:steps]
                + following * (1 - raw["dones"][:steps])
                - current
            )
            np.testing.assert_allclose(
                shaped["rewards"][:steps],
                expected,
                atol=2e-6,
                rtol=2e-6,
            )
            compared.append(algorithm)
    return {"matched_physical_warmup": compared, "only_sampled_reward_differs": True}


def launch(stage):
    protocol = read_json(REGISTRATION)
    commit = verify(protocol)
    smoke = check_smokes(protocol) if stage == "full" else None
    status, logs = ROOT / f"{stage}_status.json", ROOT / f"{stage}_logs"
    if status.exists() or logs.exists():
        raise FileExistsError("Never overwrite a fractional-budget campaign")
    logs.mkdir()
    processes, jobs = [], []
    try:
        for variant_index, variant in enumerate(protocol["variants"]):
            for algorithm_index, algorithm in enumerate(protocol["algorithms"]):
                for seed_index, seed in enumerate(protocol["seeds"]):
                    if stage == "smoke" and seed_index:
                        continue
                    index = variant_index * 8 + algorithm_index * 4 + seed_index
                    cpus = protocol["cpu_groups"][index]
                    command = [
                        sys.executable,
                        str(ROOT / "run_fraction.py"),
                        "--protocol",
                        str(REGISTRATION),
                        "--variant",
                        variant,
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
                    with (logs / f"{variant}_{algorithm}_{seed}.log").open(
                        "w"
                    ) as handle:
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
                            "variant": variant,
                            "algorithm": algorithm,
                            "seed": seed,
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
                status,
                {
                    "registration_id": protocol["registration_id"],
                    "implementation_commit": commit,
                    "stage": stage,
                    "state": "running" if running else "finished",
                    "jobs": jobs,
                    "warmup_parity_smokes": smoke,
                    "updated_unix": time.time(),
                },
            )
            if not running:
                break
            time.sleep(5)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("Fractional-budget learner failures are retained")
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
