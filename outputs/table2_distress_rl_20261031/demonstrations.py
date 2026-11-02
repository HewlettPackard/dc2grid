"""Collect fixed demonstrations on training dates only, retaining every attempt."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# isort: off
from common import ROOT, cap_cpu, case_key, contracts, file_hash, write_json
from controllers import CoolingFirstController
from fast_task import StandaloneTask
from history_policy import ACTION_HIGH, ACTION_LOW, hardware_action

import numpy as np
# isort: on


def collect_case(protocol: dict, cell: dict, destination: str, cpu: int):
    cap_cpu(cpu)
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    task = StandaloneTask(protocol, seed=cell["seed"], fixed_cell=cell)
    teacher = CoolingFirstController()
    rng = np.random.default_rng(np.random.SeedSequence([cell["seed"], 139]))
    history_rows, observer_rows, target_rows, action_rows = [], [], [], []
    try:
        features, _ = task.reset()
        while True:
            observer_features = teacher.features(features[:18])
            target = teacher.action()[1:].copy()
            action = np.clip(
                target + rng.normal(0, protocol["demonstration_noise"], 3),
                ACTION_LOW,
                ACTION_HIGH,
            ).astype(np.float32)
            history_rows.append(features.copy())
            observer_rows.append(observer_features)
            target_rows.append(target)
            action_rows.append(action)
            teacher.last_action = hardware_action(action)
            features, _, term, trunc, _ = task.step(action)
            if term or trunc:
                break
        artifact = directory / "demonstration.npz"
        np.savez_compressed(
            artifact,
            history=np.asarray(history_rows, dtype=np.float32),
            observer=np.asarray(observer_rows, dtype=np.float32),
            targets=np.asarray(target_rows, dtype=np.float32),
            executed=np.asarray(action_rows, dtype=np.float32),
        )
        result = {
            **task.last_episode,
            "artifact": str(artifact),
            "artifact_sha256": file_hash(artifact),
            "rows": len(history_rows),
            "teacher": "causal_cooling_first",
            "labels_at_realized_dithered_states": True,
        }
        write_json(directory / "result.json", result)
        return result
    except Exception as error:
        write_json(
            directory / "failure.json",
            {
                "condition": cell,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "observed_steps": task.steps if hasattr(task, "steps") else 0,
            },
        )
        if history_rows:
            np.savez_compressed(
                directory / "partial_demonstration.npz",
                history=np.asarray(history_rows, dtype=np.float32),
                observer=np.asarray(observer_rows, dtype=np.float32),
                targets=np.asarray(target_rows, dtype=np.float32),
                executed=np.asarray(action_rows, dtype=np.float32),
            )
        raise
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = contracts()
    protocol.update(
        n_envs=1,
        demonstration_noise=[0.005, 0.003, 0.01],
        role="training_demonstrations",
        recovery_performed=False,
    )
    cells = [
        {
            "start_day": day,
            "regd_load_bias": bias,
            "seed": 710000 + day * 3 + bias_index,
        }
        for day in protocol["train_days"]
        for bias_index, bias in enumerate(protocol["train_biases"])
    ]
    if args.smoke:
        cells = cells[:3]
    forbidden = {
        cell["start_day"]
        for cell in protocol["development_cells"] + protocol["sealed_cells"]
    }
    if any(cell["start_day"] in forbidden for cell in cells):
        raise AssertionError("Demonstrations may never train on development/test dates")
    cpus = sorted(os.sched_getaffinity(0))[: args.workers]
    source = args.output / "source"
    source.mkdir()
    files = [
        "common.py",
        "controllers.py",
        "history_policy.py",
        "fast_task.py",
        "demonstrations.py",
    ]
    protocol["code_hashes"] = {name: file_hash(ROOT / name) for name in files}
    for name in files:
        shutil.copy2(ROOT / name, source / name)
    write_json(
        args.output / "manifest.json", {**protocol, "cells": cells, "cpus": cpus}
    )
    results, failures = [], []
    with ProcessPoolExecutor(
        max_workers=len(cpus), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                collect_case,
                protocol,
                cell,
                str(args.output / case_key(cell)),
                cpus[index % len(cpus)],
            ): cell
            for index, cell in enumerate(cells)
        }
        for future in as_completed(futures):
            cell = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    case_key(cell),
                    result["rows"],
                    result["service_qualified"],
                    result["service"]["tracking_rmse_kw"],
                    flush=True,
                )
            except Exception as error:
                failures.append({"cell": cell, "error": repr(error)})
                print("DEMONSTRATION FAILURE", failures[-1], flush=True)
    summary = {
        "planned_cases": len(cells),
        "completed_cases": len(results),
        "physical_transitions": sum(row["rows"] for row in results),
        "results": results,
        "runtime_failures": failures,
    }
    write_json(args.output / "summary.json", summary)
    if failures or any(not row["complete"] for row in results):
        raise RuntimeError("Incomplete demonstration corpus; all attempts retained")


if __name__ == "__main__":
    main()
