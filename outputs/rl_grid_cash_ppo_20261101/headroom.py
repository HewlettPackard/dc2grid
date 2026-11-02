"""Register and execute development-only headroom probes before RL training."""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

# isort: off
from economy_context import (
    ROOT,
    PRIOR,
    VERIFIED,
    SNAPSHOT_COMMIT,
    cap_cpu,
    case_key,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)
from economy_eval import evaluate_episode
# isort: on

REGISTRATION = ROOT / "headroom_registration.json"


def prepare() -> dict:
    if REGISTRATION.exists():
        raise FileExistsError("Never overwrite a registered experiment")
    prior = read_json(PRIOR / "registration.json")
    keys = (
        "config_path",
        "service_gate",
        "recovery",
        "normalization",
        "hardware_reference",
        "train_days",
        "train_biases",
        "service_floor",
    )
    protocol = {key: copy.deepcopy(prior[key]) for key in keys}
    policies = [
        {
            "id": "analytic_control",
            "family": "cooling_first",
            "hardware": "cooling_first",
            "macro": "full_residual",
            "track": "control",
        }
    ]
    for name, command, gated, track in (
        ("offset_half_mw", [0, 0, 0.5], False, "economic_dispatch"),
        ("offset_one_mw", [0, 0, 1], False, "economic_dispatch"),
        ("inventory_probe", [0, 1, 0], False, "boundary_diagnostic"),
        ("offset_one_inventory", [0, 1, 1], False, "economic_dispatch"),
        ("positive_regd_offset", [0, 0, 1], True, "economic_dispatch"),
    ):
        policies.append(
            {
                "id": name,
                "family": name,
                "hardware": "constant_residual",
                "macro": "full_residual",
                "track": track,
                "residual": command,
                "positive_regd_only": gated,
                "learned": False,
            }
        )
    controls = []
    expected_cells = {case_key(cell) for cell in prior["development_cells"]}
    readiness = VERIFIED / "full_macro_readiness"
    for path in sorted(readiness.glob("*/*/episode.json")):
        episode = read_json(path)
        if episode["policy"]["macro"] != "full_residual":
            raise ValueError("Unexpected macro in the frozen readiness panel")
        if case_key(episode["cell"]) not in expected_cells:
            raise ValueError("Unexpected frozen development condition")
        controls.append(
            {
                "path": str(path),
                "sha256": file_hash(path),
                "policy": episode["policy"],
                "cell": episode["cell"],
            }
        )
    if len(controls) != 528:
        raise ValueError("Expected all 22 frozen policies on all 24 development cells")
    hashes = dict(prior["source_hashes"])
    hashes.update({str(path): file_hash(path) for path in sorted(ROOT.glob("*.py"))})
    inputs = dict(prior["input_hashes"])
    inputs.update({item["path"]: item["sha256"] for item in controls})
    protocol.update(
        schema="grid-cash-prelearning-headroom-v1",
        registration_file=REGISTRATION.name,
        registered_utc=datetime.now(timezone.utc).isoformat(),
        baseline_commit=SNAPSHOT_COMMIT,
        source_hashes=hashes,
        input_hashes=inputs,
        development_cells=copy.deepcopy(prior["development_cells"]),
        final_cells=[],
        policies=policies,
        frozen_controls=controls,
        cpus=list(range(48, 56)),
        threads_per_process=1,
        planned_episodes=len(policies) * len(expected_cells),
        user_approved_tracks=["benchmark_preserving", "economic_dispatch_separate"],
        full_work_fixed=True,
        new_model_training=False,
        no_new_final_or_regression_outcomes_opened=True,
        purpose="Measure headroom; not an RL result or a paper replacement.",
        comparison={
            "macro": "full_residual",
            "primary_relief_stratum": 0.5,
            "frozen_controls": "Complete frozen 22-policy development readiness panel",
            "family_aggregation": "Average model seeds within each case, then cases",
            "required_joint_metrics": [
                "stress_facility_relief_mw",
                "stress_pcc_relief_mw",
                "cashflow_26h_usd",
            ],
            "cash_headroom_threshold_usd": 5.0,
            "relief_headroom_threshold_mw": 0.001,
            "all_24_actual_physical_recovery_and_service_gates_required": True,
            "both_bias_cash_contrasts_must_be_nonnegative": True,
            "no_survivor_only_candidate_mean": True,
            "boundary_only_inventory_changes_cannot_establish_grid_relief": True,
            "constant_offsets_must_be_reported_as_efficiency_not_time_shifting": True,
        },
    )
    protocol["registration_id"] = identity(protocol)
    write_json(REGISTRATION, protocol)
    verify(protocol, require_commit=False)
    return protocol


def initialize_worker(queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(queue.get())


def run() -> None:
    protocol = read_json(REGISTRATION)
    commit = verify(protocol)
    output = ROOT / "headroom_development"
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        {
            "registration_id": protocol["registration_id"],
            "implementation_commit": commit,
            "role": "consumed_development_only",
            "planned_episodes": protocol["planned_episodes"],
            "policies": protocol["policies"],
            "cells": protocol["development_cells"],
            "cpus": protocol["cpus"],
            "model_training": False,
        },
    )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
    results, failures = [], []
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=len(protocol["cpus"]),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as executor:
        futures = {}
        for policy in protocol["policies"]:
            for cell in protocol["development_cells"]:
                destination = output / policy["id"] / case_key(cell)
                future = executor.submit(
                    evaluate_episode,
                    protocol,
                    policy,
                    cell,
                    destination,
                )
                futures[future] = str(destination)
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(futures[future])
                print(
                    f"{len(results) + len(failures)}/{protocol['planned_episodes']} "
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
                    "registration_id": protocol["registration_id"],
                    "completed": len(results),
                    "runtime_failures": failures,
                    "planned": protocol["planned_episodes"],
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    queue.close()
    queue.join_thread()
    verify(protocol)
    if failures:
        raise RuntimeError(
            f"{len(failures)} headroom episodes failed; evidence retained"
        )
    if len(results) != protocol["planned_episodes"]:
        raise AssertionError("Incomplete headroom cohort")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        print(prepare()["registration_id"])
    else:
        run()
