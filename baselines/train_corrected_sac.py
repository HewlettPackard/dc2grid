"""Resource-bounded, multi-environment staged SAC with fixed validation cells."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from baselines.train_hierarchical_sac import _make_sac_inner_fn
from c2g_env import C2GFastEnv, C2GMacroEnv
from c2g_env.experiments.identity import (
    capture_source,
    configuration_id,
    file_fingerprint,
    source_id,
)
from c2g_env.train_envs import RuleMacroWrappedEnv

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cap_threads(count: int = 1, cpu: int | None = None) -> None:
    if count < 1:
        raise ValueError("Thread count must be positive")
    for key in THREAD_VARIABLES:
        os.environ[key] = str(count)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(count)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def validate_protocol(protocol: dict) -> None:
    if protocol["simulation_mode"] != "physical_v1":
        raise ValueError("This retraining protocol requires physical_v1")
    days = protocol["train_days"]
    validation = protocol["validation"]
    validation_days = {cell["start_day"] for cell in validation}
    held_out = set(protocol["held_out_days"])
    if not days or not protocol["train_biases"] or not validation:
        raise ValueError("Training and validation condition sets must be nonempty")
    if len(days) != len(set(days)):
        raise ValueError("Training days must be unique")
    if set(days) & validation_days or (set(days) | validation_days) & held_out:
        raise ValueError("Training, validation and held-out days must be disjoint")
    for day in set(days) | validation_days | held_out:
        if isinstance(day, bool) or not isinstance(day, int) or day < 0:
            raise ValueError("Calendar days must be nonnegative integers")
    biases = protocol["train_biases"] + [cell["regd_load_bias"] for cell in validation]
    if not np.isfinite(biases).all():
        raise ValueError("Biases must be finite")
    identities = [(c["start_day"], c["regd_load_bias"], c["seed"]) for c in validation]
    if len(identities) != len(set(identities)):
        raise ValueError("Validation cells must be unique")
    if protocol["n_envs"] < 1 or protocol["threads"] < 1:
        raise ValueError("Environment and thread counts must be positive")
    if protocol["algorithm"]["gradient_steps"] != -1:
        raise ValueError("Keep gradient_steps=-1 to preserve updates per transition")
    if protocol["algorithm"]["train_freq"] < 1:
        raise ValueError("train_freq must be positive")
    selection = protocol["selection"]
    if not selection["require_all_validation_cells"]:
        raise ValueError("Incomplete validation suites must not select checkpoints")
    if selection["score"] != "mean_step_reward":
        raise ValueError("Unsupported validation selection score")
    for stage in ("low", "high"):
        settings = protocol[stage]
        for key in ("timesteps", "eval_freq", "checkpoint_freq", "status_freq"):
            if settings[key] < 1:
                raise ValueError(f"{stage}.{key} must be positive")
        if settings["learning_starts"] < 0 or not 0 < settings["gamma"] < 1:
            raise ValueError(f"Invalid {stage} warmup or discount")


class EpisodeConditions(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        days: list[int],
        biases: list[float],
        *,
        seed: int,
        validation_cell: dict | None = None,
    ):
        super().__init__(env)
        self.days = days
        self.biases = biases
        self.seed_key = seed
        self.rng = np.random.default_rng(seed)
        self.validation_cell = validation_cell
        self.condition: dict = {}

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Episode conditions are fixed by the registered protocol")
        if self.validation_cell is not None:
            cell = self.validation_cell
        else:
            if seed is not None:
                self.rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed_key, seed])
                )
            cell = {
                "start_day": int(self.rng.choice(self.days)),
                "regd_load_bias": float(self.rng.choice(self.biases)),
                "seed": int(self.rng.integers(0, 2**31 - 1)),
            }
        self.condition = dict(cell)
        obs, info = self.env.reset(
            seed=cell["seed"],
            options={key: cell[key] for key in ("start_day", "regd_load_bias")},
        )
        return obs, {**info, "condition": self.condition}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {**info, "condition": self.condition}


def make_worker(
    protocol: dict,
    stage: str,
    seed: int,
    rank: int,
    cpus: list[int],
    inner_checkpoint: str | None,
    validation_cell: dict | None = None,
) -> gym.Env:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_threads(protocol["threads"], cpus[1 + rank % (len(cpus) - 1)] if cpus else None)
    kwargs = {"scenario": protocol["scenario"], "simulation_mode": "physical_v1"}
    if protocol.get("config_path"):
        kwargs["config_path"] = protocol["config_path"]
    if stage == "low":
        env = RuleMacroWrappedEnv(C2GFastEnv(**kwargs))
    else:
        if inner_checkpoint is None:
            raise ValueError(
                "Macro training requires a qualified frozen hardware checkpoint"
            )
        inner = _make_sac_inner_fn(inner_checkpoint, "physical_v1")
        env = C2GMacroEnv(**kwargs, inner_action_fn=inner)
    conditioned = EpisodeConditions(
        env,
        protocol["train_days"],
        protocol["train_biases"],
        seed=int(np.random.SeedSequence([seed, rank]).generate_state(1)[0]),
        validation_cell=validation_cell,
    )
    return Monitor(conditioned)


def make_vectors(
    protocol: dict,
    stage: str,
    seed: int,
    cpus: list[int],
    inner_checkpoint: str | None,
    *,
    validation: bool = False,
) -> VecEnv:
    cells = protocol["validation"] if validation else [None] * protocol["n_envs"]
    factories = [
        partial(make_worker, protocol, stage, seed, rank, cpus, inner_checkpoint, cell)
        for rank, cell in enumerate(cells)
    ]
    return SubprocVecEnv(factories, start_method="spawn")


def new_episode() -> dict:
    return {
        "steps": 0,
        "fast_steps": 0,
        "reward_sum": 0.0,
        "error_sq_sum": 0.0,
        "work_arrived_kwh": 0.0,
        "work_served_kwh": 0.0,
        "thermal_warning_ticks": 0,
        "regulation_revenue": 0.0,
        "electricity_cost": 0.0,
        "facility_relief_kwh": 0.0,
    }


def accumulate(episode: dict, reward: float, info: dict, stage: str) -> None:
    episode["steps"] += 1
    episode["reward_sum"] += float(reward)
    if stage == "low":
        episode["fast_steps"] += 1
        episode["error_sq_sum"] += info["tracking_err_kw"] ** 2
        episode["thermal_warning_ticks"] += int(info["thermal_warning"])
        episode["facility_relief_kwh"] += (
            info["facility_relief_kw"] * info["duration_hours"]
        )
        episode["electricity_cost"] += (
            info["p_grid_import_mw"] * info["lmp"] * info["duration_hours"]
        )
        last = info
    else:
        episode["fast_steps"] += info["sub_steps_run"]
        episode["error_sq_sum"] += info["tracking_error_sq_sum"]
        episode["thermal_warning_ticks"] += info["thermal_warning_ticks"]
        episode["regulation_revenue"] += info["regulation_revenue"]
        episode["electricity_cost"] += info["electricity_cost"]
        episode["facility_relief_kwh"] += info["facility_relief_kwh"]
        last = info["last_inner_info"]
    for key in ("work_arrived_kwh", "work_served_kwh"):
        episode[key] += info[key]
    episode.update(
        terminal_soc=float(last["bess_soc"]),
        terminal_backlog_kwh=float(last["backlog_work_kwh"]),
        terminated=bool(info["terminated"]),
        truncated=bool(info["truncated"]),
        expected_ticks=int(last["episode_ticks"]),
        condition=info["condition"],
    )


def finish_episode(episode: dict, selection: dict) -> dict:
    conservation_error = (
        episode["work_arrived_kwh"]
        - episode["work_served_kwh"]
        - episode["terminal_backlog_kwh"]
    )
    complete = (
        episode["fast_steps"] == episode["expected_ticks"]
        and episode["truncated"]
        and not episode["terminated"]
    )
    backlog_limit = (
        selection["max_backlog_fraction"] * episode["work_arrived_kwh"] + 1e-6
    )
    warning_fraction = episode["thermal_warning_ticks"] / episode["fast_steps"]
    qualified = (
        complete
        and abs(conservation_error) < 1e-4
        and episode["terminal_backlog_kwh"] <= backlog_limit
        and episode["terminal_soc"] >= selection["min_terminal_soc"]
        and warning_fraction <= selection["max_thermal_warning_fraction"]
    )
    return {
        **episode,
        "complete": complete,
        "qualified": qualified,
        "mean_step_reward": episode["reward_sum"] / episode["steps"],
        "tracking_rmse_kw": math.sqrt(episode["error_sq_sum"] / episode["fast_steps"]),
        "thermal_warning_fraction": warning_fraction,
        "work_conservation_error_kwh": conservation_error,
    }


def evaluate_suite(model: SAC, env: VecEnv, stage: str, selection: dict) -> list[dict]:
    observations = env.reset()
    active = np.ones(env.num_envs, dtype=bool)
    episodes = [new_episode() for _ in range(env.num_envs)]
    while active.any():
        actions, _ = model.predict(observations, deterministic=True)
        observations, rewards, dones, infos = env.step(actions)
        for index in np.flatnonzero(active):
            accumulate(episodes[index], rewards[index], infos[index], stage)
            if dones[index]:
                active[index] = False
    return [finish_episode(episode, selection) for episode in episodes]


class TrainingJournal(BaseCallback):
    def __init__(
        self,
        protocol: dict,
        stage: str,
        output: Path,
        validation: VecEnv,
        run_identity: dict,
    ):
        super().__init__()
        self.protocol, self.stage, self.output = protocol, stage, output
        self.validation, self.identity = validation, run_identity
        self.settings = protocol[stage]
        self.last_eval = self.last_checkpoint = self.last_status = 0
        self.best_score = -math.inf
        self.fast_steps = self.completed_episodes = 0
        self.episodes: dict[int, dict] = {}
        self.started = time.monotonic()
        self.validation_seconds = 0.0
        self.validation_calls = 0

    def status(self, state: str, **extra) -> None:
        elapsed = time.monotonic() - self.started
        losses = {}
        for key, value in self.model.logger.name_to_value.items():
            if key.startswith("train/") and np.isscalar(value):
                number = float(value)
                if not math.isfinite(number):
                    raise FloatingPointError(
                        f"Nonfinite training metric {key}: {value}"
                    )
                losses[key] = number
        write_json(
            self.output / "status.json",
            {
                **self.identity,
                "state": state,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "timesteps": self.num_timesteps,
                "optimizer_updates": self.model._n_updates,
                "training_fast_steps": self.fast_steps,
                "completed_training_episodes": self.completed_episodes,
                "elapsed_seconds": elapsed,
                "transitions_per_second": self.num_timesteps / max(elapsed, 1e-6),
                "validation_seconds": self.validation_seconds,
                "validation_calls": self.validation_calls,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "torch_threads": torch.get_num_threads(),
                "best_qualified_score": self.best_score
                if math.isfinite(self.best_score)
                else None,
                "losses": losses,
                **extra,
            },
        )

    def _on_training_start(self) -> None:
        self.status("training")

    def _on_step(self) -> bool:
        if not np.isfinite(self.locals["rewards"]).all():
            raise FloatingPointError("Nonfinite rollout reward")
        for index, (reward, info, done) in enumerate(
            zip(
                self.locals["rewards"],
                self.locals["infos"],
                self.locals["dones"],
            )
        ):
            self.fast_steps += info["sub_steps_run"] if self.stage == "high" else 1
            episode = self.episodes.setdefault(index, new_episode())
            accumulate(episode, reward, info, self.stage)
            if done:
                result = finish_episode(episode, self.protocol["selection"])
                with (self.output / "training_episodes.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "timesteps": self.num_timesteps,
                                "worker": index,
                                **result,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                self.completed_episodes += 1
                del self.episodes[index]
        if self.num_timesteps - self.last_status >= self.settings["status_freq"]:
            self.last_status = self.num_timesteps
            self.status("training")
        if (
            self.num_timesteps - self.last_checkpoint
            >= self.settings["checkpoint_freq"]
        ):
            self.last_checkpoint = self.num_timesteps
            self.save_checkpoint()
        if self.num_timesteps - self.last_eval >= self.settings["eval_freq"]:
            self.last_eval = self.num_timesteps
            self.evaluate()
        return True

    def save_checkpoint(self) -> None:
        directory = self.output / "checkpoints" / f"step_{self.num_timesteps}"
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save(directory / "model.zip")
        self.model.save_replay_buffer(directory / "replay_buffer.pkl")
        write_json(
            directory / "checkpoint.json",
            {
                **self.identity,
                "timesteps": self.num_timesteps,
                "optimizer_updates": self.model._n_updates,
                "model": file_fingerprint(directory / "model.zip"),
                "replay_buffer": file_fingerprint(directory / "replay_buffer.pkl"),
                "resume_scope": (
                    "optimizer/replay retained; "
                    "episode/RNG continuation is not bit-exact"
                ),
            },
        )

    def evaluate(self) -> None:
        self.status("validating")
        start = time.monotonic()
        results = evaluate_suite(
            self.model,
            self.validation,
            self.stage,
            self.protocol["selection"],
        )
        self.validation_seconds += time.monotonic() - start
        self.validation_calls += 1
        qualified = all(result["qualified"] for result in results)
        score = float(np.mean([result["mean_step_reward"] for result in results]))
        report = {
            "timesteps": self.num_timesteps,
            "optimizer_updates": self.model._n_updates,
            "all_cells_qualified": qualified,
            "selection_score": score,
            "attempts": len(results),
            "qualified_count": sum(r["qualified"] for r in results),
            "cells": results,
        }
        directory = self.output / "validation"
        directory.mkdir(exist_ok=True)
        write_json(directory / f"step_{self.num_timesteps}.json", report)
        if qualified and score > self.best_score:
            self.best_score = score
            best = self.output / "best_model"
            best.mkdir(exist_ok=True)
            self.model.save(best / "best_model.zip")
            write_json(
                best / "selection.json",
                {
                    **report,
                    "checkpoint": file_fingerprint(best / "best_model.zip"),
                    "protocol_id": self.identity["protocol_id"],
                },
            )
        self.status("training")
        print(
            f"{self.stage} step={self.num_timesteps} updates={self.model._n_updates} "
            f"validation_qualified={sum(r['qualified'] for r in results)}"
            f"/{len(results)} "
            f"mean_reward={score:.6f}",
            flush=True,
        )


def train(
    protocol: dict,
    stage: str,
    seed: int,
    output: Path,
    device: str,
    cpus: list[int],
    inner_checkpoint: str | None = None,
) -> None:
    validate_protocol(protocol)
    if cpus and len(cpus) < protocol["n_envs"] + 1:
        raise ValueError("Reserve one CPU for the learner and one per environment")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    output.mkdir(parents=True)
    cap_threads(protocol["threads"], cpus[0] if cpus else None)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_identity = {
        "stage": stage,
        "seed": seed,
        "device": device,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "n_envs": protocol["n_envs"],
        "protocol_id": configuration_id(protocol),
        "source_id": source_id(ROOT),
        "simulation_mode": "physical_v1",
        "inner_checkpoint": file_fingerprint(Path(inner_checkpoint))
        if inner_checkpoint
        else None,
        "budgets": protocol[stage],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "manifest.json", run_identity)
    env = validation = None
    journal = None
    succeeded = False
    try:
        env = make_vectors(protocol, stage, seed, cpus, inner_checkpoint)
        validation = make_vectors(
            protocol, stage, seed, cpus, inner_checkpoint, validation=True
        )
        env.seed(seed)
        parameters = {**protocol["algorithm"]}
        model = SAC(
            "MlpPolicy",
            env,
            device=device,
            seed=seed,
            policy_kwargs={"net_arch": parameters.pop("net_arch")},
            learning_starts=protocol[stage]["learning_starts"],
            gamma=protocol[stage]["gamma"],
            tensorboard_log=str(output / "tensorboard"),
            verbose=0,
            **parameters,
        )
        model.c2g_context = {
            **run_identity,
            "source_snapshot": capture_source(ROOT, output / "source"),
            "protocol": protocol,
            "training_seed": seed,
            "gradient_updates_per_transition": 1.0,
        }
        journal = TrainingJournal(protocol, stage, output, validation, run_identity)
        model.learn(protocol[stage]["timesteps"], callback=journal, log_interval=1)
        if journal.last_eval != model.num_timesteps:
            journal.evaluate()
        model.save(output / "final_model.zip")
        model.save_replay_buffer(output / "final_replay_buffer.pkl")
        journal.status(
            "completed",
            qualified_checkpoint=(output / "best_model/best_model.zip").exists(),
        )
        succeeded = True
    finally:
        if not succeeded:
            error = sys.exc_info()[1]
            write_json(
                output / "failure.json",
                {
                    **run_identity,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timesteps": journal.num_timesteps if journal else 0,
                },
            )
        for vector in (validation, env):
            if vector is not None:
                vector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", choices=["low", "high"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--inner-checkpoint")
    parser.add_argument("--pilot-steps", type=int)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if args.pilot_steps is not None:
        if args.pilot_steps < 2 * protocol["n_envs"]:
            parser.error("--pilot-steps must allow rollout and optimizer updates")
        protocol[args.stage].update(
            timesteps=args.pilot_steps,
            learning_starts=args.pilot_steps // 8,
            eval_freq=args.pilot_steps + 1,
            checkpoint_freq=args.pilot_steps // 2,
            status_freq=max(protocol["n_envs"], args.pilot_steps // 8),
        )
        protocol["pilot_only"] = True
    train(
        protocol,
        args.stage,
        args.seed,
        args.output,
        args.device,
        args.cpus,
        args.inner_checkpoint,
    )


if __name__ == "__main__":
    main()
