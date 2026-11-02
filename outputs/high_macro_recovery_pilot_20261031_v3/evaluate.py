"""One-shot development evaluation with explicit inactive-worker padding."""

from __future__ import annotations

import os
from pathlib import Path

# isort: off
from study_common import (
    EXTRA_FEATURES,
    OBSERVATION_DIM,
    TASK_SCHEMA,
    cell_key,
    file_hash,
    write_json,
)
from macro_task import MacroQuantityTask
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from baselines.rule_based_macro import RuleBasedMacroController
# isort: on


def cap_worker(cpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


class OneShotEvaluation(gym.Wrapper):
    def __init__(self, task: MacroQuantityTask):
        super().__init__(task)
        self.finished = False
        self.last_observation = None
        self.suppressed_steps = 0

    def assign_cell(self, cell: dict):
        permitted = {cell_key(item) for item in self.env.protocol["development_cells"]}
        if cell_key(cell) not in permitted:
            raise ValueError(
                "Evaluation may not execute an unregistered or sealed cell"
            )
        if not self.finished and not self.env.done:
            raise RuntimeError("Cannot replace an active evaluation case")
        self.env.fixed_cell = dict(cell)
        self.finished = False
        self.last_observation = None
        self.suppressed_steps = 0

    def reset(self, **kwargs):
        if self.finished:
            return self.last_observation.copy(), {"evaluation_padding_reset": True}
        observation, info = self.env.reset(**kwargs)
        self.last_observation = observation
        return observation, info

    def step(self, action):
        if self.finished:
            self.suppressed_steps += 1
            return (
                self.last_observation.copy(),
                0.0,
                True,
                False,
                {"evaluation_padding": True},
            )
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = observation
        self.finished = terminated or truncated
        return observation, reward, terminated, truncated, info

    def persist(self, directory: str, identity: dict) -> dict:
        if not self.finished or self.env.last_episode is None:
            raise RuntimeError("No completed evaluation to persist")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        traces = (
            ("hardware.csv.gz", self.env.service_rows),
            ("macro.csv.gz", self.env.macro_rows),
            ("recovery.csv.gz", self.env.recovery_rows),
        )
        fingerprints = {}
        for filename, rows in traces:
            if rows:
                pd.DataFrame(rows).to_csv(
                    destination / filename, index=False, compression="gzip"
                )
                fingerprints[filename] = file_hash(destination / filename)
        result = {
            **self.env.last_episode,
            **identity,
            "trace_hashes": fingerprints,
            "suppressed_padding_steps": self.suppressed_steps,
            "padding_is_not_a_new_rollout": True,
        }
        write_json(destination / "episode.json", result)
        return result


def make_eval_worker(protocol: dict, kappa: float, rank: int, cpus: list[int]):
    cap_worker(cpus[rank + 1])
    task = MacroQuantityTask(
        protocol,
        kappa,
        cell=protocol["development_cells"][rank],
        record_trace=True,
    )
    return OneShotEvaluation(task)


def evaluation_vector(protocol: dict, kappa: float, cpus: list[int]):
    from functools import partial

    return SubprocVecEnv(
        [
            partial(make_eval_worker, protocol, kappa, rank, cpus)
            for rank in range(protocol["n_envs"])
        ],
        start_method="spawn",
    )


class ConstantQuantity:
    def __init__(self, fraction: float):
        self.fraction = float(fraction)

    def predict(self, observations, deterministic=True):
        return np.full((len(observations), 1), self.fraction, dtype=np.float32), None


class CappedRuleQuantity:
    def __init__(self):
        self.rule = RuleBasedMacroController()

    def predict(self, observations, deterministic=True):
        output = []
        for observation in observations:
            native, _ = self.rule.predict(observation[:19], deterministic=True)
            maximum = min(30.0, float(observation[18]) * 30)
            if maximum <= 5:
                raise ValueError("Rule baseline has no feasible residual range")
            target = float(native[0]) * 30
            output.append([float(np.clip((target - 5) / (maximum - 5), 0, 1))])
        return np.asarray(output, dtype=np.float32), None


def load_macro(path: Path, protocol: dict, *, device: str = "cpu"):
    model = SAC.load(path, device=device)
    context = getattr(model, "c2g_context", {})
    if (
        context.get("task_schema") != TASK_SCHEMA
        or context.get("registration_id") != protocol["registration_id"]
        or context.get("extra_observation_features") != list(EXTRA_FEATURES)
        or context.get("hardware_checkpoint", {}).get("sha256")
        != protocol["hardware"]["sha256"]
        or model.observation_space.shape != (OBSERVATION_DIM,)
        or model.action_space.shape != (1,)
        or model.gamma != 1.0
    ):
        raise ValueError("Macro checkpoint/task/hardware identity mismatch")
    return model


def evaluate_suite(model, vector, protocol: dict, output: Path, identity: dict) -> dict:
    cells = protocol["development_cells"]
    width = vector.num_envs
    if len(cells) % width:
        raise ValueError("Development cells must form complete worker batches")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for offset in range(0, len(cells), width):
        batch = cells[offset : offset + width]
        for rank, cell in enumerate(batch):
            vector.env_method("assign_cell", cell, indices=rank)
        observation = vector.reset()
        active = np.ones(width, dtype=bool)
        for _ in range(96):
            actions, _ = model.predict(observation, deterministic=True)
            observation, _, done, infos = vector.step(actions)
            for rank in np.flatnonzero(active):
                if infos[rank].get("evaluation_padding"):
                    raise AssertionError("Padding was mistaken for an active case")
                if done[rank]:
                    active[rank] = False
                    if "c2g_episode" not in infos[rank]:
                        raise RuntimeError("Missing finite-task completion record")
            if not active.any():
                break
        if active.any():
            raise RuntimeError(
                "A development case exceeded its service decision budget"
            )
        for rank, cell in enumerate(batch):
            result = vector.env_method(
                "persist", str(output / cell_key(cell)), identity, indices=rank
            )[0]
            if cell_key(result["condition"]) != cell_key(cell):
                raise AssertionError("Evaluation returned the wrong registered case")
            results.append(result)
    complete_money = all(
        result["cashflow_closed_recovery_usd"] is not None for result in results
    )
    all_qualified = all(result["qualified"] for result in results)
    report = {
        **identity,
        "planned_cells": len(cells),
        "recorded_cells": len(results),
        "qualified_cells": sum(result["qualified"] for result in results),
        "service_complete_cells": sum(result["service_complete"] for result in results),
        "recovery_closed_cells": sum(
            result["recovery"]["closed"] for result in results
        ),
        "all_qualified": all_qualified,
        "mean_cashflow_closed_recovery_usd": float(
            np.mean([result["cashflow_closed_recovery_usd"] for result in results])
        )
        if complete_money
        else None,
        "mean_delivered_service_ratio": float(
            np.mean([result["delivered_service_ratio"] for result in results])
        ),
        "no_sealed_test_cases_executed": True,
        "cells": results,
    }
    write_json(output / "summary.json", report)
    return report
