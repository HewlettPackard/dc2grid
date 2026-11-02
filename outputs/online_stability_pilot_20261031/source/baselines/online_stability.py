"""Isolated, registered online continuation of the corrected SAC checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import deque
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import SubprocVecEnv

from baselines.train_corrected_sac import (
    TrainingJournal,
    accumulate,
    cap_threads,
    finish_episode,
    make_worker,
    new_episode,
    write_json,
)
from c2g_env import C2GFastEnv
from c2g_env.experiments.identity import (
    check_policy_semantics,
    configuration_id,
    file_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
REWARD_KEYS = (
    "reward_throughput",
    "reward_tracking",
    "reward_thermal",
    "reward_soc",
    "reward_freq",
    "reward_volt",
    "reward_backlog",
)


def fingerprint_state(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def actor_identity(model: SAC) -> dict:
    return {
        "actor": fingerprint_state(model.actor.state_dict()),
        "actor_optimizer": fingerprint_state(model.actor.optimizer.state_dict()),
        "log_alpha": fingerprint_state(model.log_ent_coef),
        "alpha_optimizer": fingerprint_state(
            model.ent_coef_optimizer.state_dict()
            if model.ent_coef_optimizer is not None
            else None
        ),
    }


def apply_treatment(model: SAC, arm: str, donor: SAC | None = None) -> dict:
    before = actor_identity(model)
    if arm not in ("sham", "optimizer_reset", "critic_restore"):
        raise ValueError(f"Unknown treatment {arm}")
    if arm == "critic_restore":
        if donor is None:
            raise ValueError("Critic restoration requires the registered donor")
        check_policy_semantics(donor, "physical_v1")
        model.critic.load_state_dict(donor.critic.state_dict(), strict=True)
        model.critic_target.load_state_dict(
            donor.critic_target.state_dict(), strict=True
        )
    if arm != "sham":
        if type(model.critic.optimizer) is not torch.optim.Adam:
            raise TypeError("This registered intervention requires Adam")
        model.critic.optimizer = torch.optim.Adam(
            model.critic.parameters(), **copy.deepcopy(model.critic.optimizer.defaults)
        )
    if actor_identity(model) != before:
        raise AssertionError("Treatment changed actor or entropy initialization")
    return {
        "actor_and_entropy": before,
        "critic": fingerprint_state(model.critic.state_dict()),
        "target": fingerprint_state(model.critic_target.state_dict()),
        "critic_optimizer": fingerprint_state(model.critic.optimizer.state_dict()),
    }


def stream_seed(base_seed: int, stream: int, phase: int) -> int:
    return int(
        np.random.SeedSequence([20261031, base_seed, stream, phase]).generate_state(1)[
            0
        ]
    )


def set_update_rng(base_seed: int, stream: int, phase: int) -> dict:
    seeds = {
        "numpy": stream_seed(base_seed, stream, phase),
        "torch": stream_seed(base_seed, stream, phase + 1),
    }
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch"])
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seeds["torch"])
    return seeds


def calibrate_critic(model: SAC, steps: int, batch_size: int) -> list[dict]:
    before = actor_identity(model)
    model.policy.set_training_mode(True)
    milestones = []
    for index in range(steps):
        batch = model.replay_buffer.sample(batch_size)
        with torch.no_grad():
            action, log_probability = model.actor.action_log_prob(
                batch.next_observations
            )
            alpha = model.log_ent_coef.detach().exp()
            target = torch.cat(
                model.critic_target(batch.next_observations, action), dim=1
            )
            soft_value = target.min(
                dim=1, keepdim=True
            ).values - alpha * log_probability.reshape(-1, 1)
            discount = batch.discounts if batch.discounts is not None else model.gamma
            target = batch.rewards + (1 - batch.dones) * discount * soft_value
        values = model.critic(batch.observations, batch.actions)
        loss = 0.5 * sum(F.mse_loss(value, target) for value in values)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite critic calibration loss")
        model.critic.optimizer.zero_grad()
        loss.backward()
        model.critic.optimizer.step()
        polyak_update(
            model.critic.parameters(), model.critic_target.parameters(), model.tau
        )
        polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.0)
        if index + 1 in (1, 100, 250, steps):
            milestones.append(
                {
                    "critic_update": index + 1,
                    "batch_loss": float(loss.detach()),
                    "batch_q_mean": float(torch.cat(values, dim=1).detach().mean()),
                }
            )
    if actor_identity(model) != before:
        raise AssertionError("Critic calibration changed actor or entropy state")
    return milestones


class Diagnostics(gym.Wrapper):
    """Add measurements without changing actions, observations or rewards."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(self.unwrapped, C2GFastEnv):
            raise TypeError("Online pilot only supports hardware control")
        self.fast = self.unwrapped

    def step(self, action):
        commitment = self.fast.committed_mw
        denominator = max(commitment * 1000, self.fast._tracking_floor_kw)
        observation, reward, term, trunc, info = self.env.step(action)
        queue_norm = self.fast._workload._backlog_kw / self.fast._workload.p_flex_max_kw
        info.update(
            pilot_committed_mw=commitment,
            pilot_normalized_error_sq=(info["tracking_err_kw"] / denominator) ** 2,
            pilot_queue_norm=queue_norm,
            pilot_queue_saturated=queue_norm >= 2,
            pilot_reference_temp_A=self.fast._reference_thermal.temp_A,
        )
        return observation, reward, term, trunc, info


def diagnostic_worker(protocol, seed, rank, cpus, cell=None):
    return Diagnostics(make_worker(protocol, "low", seed, rank, cpus, None, cell))


def vectors(protocol, seed, cpus, *, validation=False):
    cells = protocol["validation"] if validation else [None] * protocol["n_envs"]
    return SubprocVecEnv(
        [
            partial(diagnostic_worker, protocol, seed, rank, cpus, cell)
            for rank, cell in enumerate(cells)
        ],
        start_method="spawn",
    )


def evaluate(model, vector, selection):
    observations = vector.reset()
    active = np.ones(vector.num_envs, dtype=bool)
    episodes = [new_episode() for _ in range(vector.num_envs)]
    extra = [
        {
            "normalized_error_sq_sum": 0.0,
            "committed_mw_sum": 0.0,
            "queue_saturated_ticks": 0,
            "max_queue_norm": 0.0,
            **dict.fromkeys(REWARD_KEYS, 0.0),
        }
        for _ in range(vector.num_envs)
    ]
    while active.any():
        actions = model.predict(observations, deterministic=True)[0]
        observations, rewards, dones, infos = vector.step(actions)
        for index in np.flatnonzero(active):
            info = infos[index]
            accumulate(episodes[index], rewards[index], info, "low")
            record = extra[index]
            record["normalized_error_sq_sum"] += info["pilot_normalized_error_sq"]
            record["committed_mw_sum"] += info["pilot_committed_mw"]
            record["queue_saturated_ticks"] += int(info["pilot_queue_saturated"])
            record["max_queue_norm"] = max(
                record["max_queue_norm"], info["pilot_queue_norm"]
            )
            for key in REWARD_KEYS:
                record[key] += info[key]
            if dones[index]:
                active[index] = False
    result = []
    for episode, record in zip(episodes, extra):
        count = episode["fast_steps"]
        result.append(
            {
                **finish_episode(episode, selection),
                "normalized_rmse": math.sqrt(record["normalized_error_sq_sum"] / count),
                "mean_committed_mw": record["committed_mw_sum"] / count,
                "queue_saturation_fraction": record["queue_saturated_ticks"] / count,
                "max_queue_norm": record["max_queue_norm"],
                **{key: record[key] / count for key in REWARD_KEYS},
            }
        )
    return result


def atomic_model(model, path):
    temporary = path.with_name(path.stem + ".pending.zip")
    model.save(temporary)
    temporary.replace(path)


class PilotJournal(TrainingJournal):
    def __init__(self, protocol, output, validation, identity, base_step, base_updates):
        super().__init__(protocol, "low", output, validation, identity)
        self.base_step = base_step
        self.base_updates = base_updates
        self.last_status = self.last_eval = self.last_checkpoint = base_step

    def status(self, state, **extra):
        import time

        new_steps = self.num_timesteps - self.base_step
        super().status(
            state,
            new_online_transitions=new_steps,
            transitions_per_second=new_steps
            / max(time.monotonic() - self.started, 1e-6),
            new_normal_sac_updates=self.model._n_updates - self.base_updates,
            **extra,
        )

    def _on_step(self):
        for index, info in enumerate(self.locals["infos"]):
            episode = self.episodes.setdefault(index, new_episode())
            diagnostic = episode.setdefault(
                "diagnostics",
                {
                    "normalized_error_sq_sum": 0.0,
                    "commitment_mw_sum": 0.0,
                    "queue_saturated_ticks": 0,
                    "max_raw_queue_norm": 0.0,
                    "minimum_step_reward": math.inf,
                    **dict.fromkeys(REWARD_KEYS, 0.0),
                },
            )
            diagnostic["normalized_error_sq_sum"] += info["pilot_normalized_error_sq"]
            diagnostic["commitment_mw_sum"] += info["pilot_committed_mw"]
            diagnostic["queue_saturated_ticks"] += int(info["pilot_queue_saturated"])
            diagnostic["max_raw_queue_norm"] = max(
                diagnostic["max_raw_queue_norm"],
                info["pilot_queue_norm"],
            )
            diagnostic["minimum_step_reward"] = min(
                diagnostic["minimum_step_reward"],
                float(self.locals["rewards"][index]),
            )
            for key in REWARD_KEYS:
                diagnostic[key] += info[key]
            if self.locals["dones"][index]:
                count = episode["fast_steps"] + 1
                diagnostic.update(
                    normalized_rmse=math.sqrt(
                        diagnostic["normalized_error_sq_sum"] / count
                    ),
                    mean_commitment_mw=diagnostic["commitment_mw_sum"] / count,
                    queue_saturation_fraction=diagnostic["queue_saturated_ticks"]
                    / count,
                    mean_components={
                        key: diagnostic[key] / count for key in REWARD_KEYS
                    },
                )
        if self.num_timesteps - self.base_step == self.settings["timesteps"]:
            self.last_eval = self.last_checkpoint = self.num_timesteps
        return super()._on_step()

    def save_checkpoint(self):
        new_steps = self.num_timesteps - self.base_step
        directory = self.output / "checkpoints" / f"online_{new_steps}"
        directory.mkdir(parents=True, exist_ok=False)
        atomic_model(self.model, directory / "model.zip")
        temporary = directory / "replay.pending.pkl"
        self.model.save_replay_buffer(temporary)
        temporary.replace(directory / "replay_buffer.pkl")
        write_json(
            directory / "checkpoint.json",
            {
                **self.identity,
                "base_timesteps": self.base_step,
                "new_online_transitions": new_steps,
                "timesteps": self.num_timesteps,
                "new_normal_sac_updates": self.model._n_updates - self.base_updates,
                "model": file_fingerprint(directory / "model.zip"),
                "replay_buffer": file_fingerprint(directory / "replay_buffer.pkl"),
                "replay_position": self.model.replay_buffer.pos,
                "snapshot_scope": (
                    "Mid-run callback snapshots precede current transition insertion; "
                    "the final snapshot is post-update. No bit-exact simulator resume."
                ),
            },
        )

    def evaluate(self):
        import time

        self.status("validating")
        started = time.monotonic()
        cells = evaluate(self.model, self.validation, self.protocol["selection"])
        self.validation_seconds += time.monotonic() - started
        self.validation_calls += 1
        complete = all(cell["complete"] for cell in cells)
        qualified = all(cell["qualified"] for cell in cells)
        score = (
            float(np.mean([cell["mean_step_reward"] for cell in cells]))
            if complete
            else None
        )
        new_steps = self.num_timesteps - self.base_step
        report = {
            "new_online_transitions": new_steps,
            "timesteps": self.num_timesteps,
            "new_normal_sac_updates": self.model._n_updates - self.base_updates,
            "all_cells_complete": complete,
            "all_cells_qualified": qualified,
            "attempts": len(cells),
            "qualified_count": sum(c["qualified"] for c in cells),
            "selection_score": score,
            "mean_rmse_kw": float(np.mean([c["tracking_rmse_kw"] for c in cells]))
            if complete
            else None,
            "mean_normalized_rmse": float(
                np.mean([c["normalized_rmse"] for c in cells])
            )
            if complete
            else None,
            "cells": cells,
        }
        directory = self.output / "validation"
        directory.mkdir(exist_ok=True)
        write_json(directory / f"online_{new_steps}.json", report)
        if qualified and score > self.best_score:
            self.best_score = score
            best = self.output / "best_model"
            best.mkdir(exist_ok=True)
            atomic_model(self.model, best / "best_model.zip")
            write_json(
                best / "selection.json",
                {
                    **report,
                    "model": file_fingerprint(best / "best_model.zip"),
                },
            )
        self.status("training")
        print(
            f"{self.identity['job_id']} online={new_steps} "
            f"qualified={report['qualified_count']}/{len(cells)} "
            f"mean_reward={score} rmse={report['mean_rmse_kw']}",
            flush=True,
        )


def verify_registration(registration):
    body = {
        key: value for key, value in registration.items() if key != "registration_id"
    }
    if configuration_id(body) != registration["registration_id"]:
        raise RuntimeError("Pilot registration identity no longer matches its contents")
    for relative, expected in registration["source_files"].items():
        if file_fingerprint(ROOT / relative)["sha256"] != expected:
            raise RuntimeError(f"Registered pilot source changed: {relative}")
    path = Path(registration["protocol_file"]["path"])
    if file_fingerprint(path)["sha256"] != registration["protocol_file"]["sha256"]:
        raise RuntimeError("Registered protocol changed")


def run_job(registration, job, output, cpus, device="cuda:0"):
    verify_registration(registration)
    for artifact in job["inputs"].values():
        if file_fingerprint(Path(artifact["path"]))["sha256"] != artifact["sha256"]:
            raise RuntimeError(f"Checkpoint/replay input changed: {artifact['path']}")
    output.mkdir(parents=True, exist_ok=False)
    cap_threads(1, cpus[0] if cpus else None)
    protocol = copy.deepcopy(registration["environment_protocol"])
    settings = registration["pilot_protocol"]
    protocol["low"].update(
        timesteps=settings["online_transitions"],
        eval_freq=settings["evaluation_interval"],
        checkpoint_freq=settings["checkpoint_interval"],
        status_freq=settings["status_interval"],
    )
    base_seed, stream = job["base_seed"], job["stream"]
    env_seed = stream_seed(base_seed, stream, 10)
    identity = {
        "job_id": job["id"],
        "base_seed": base_seed,
        "continuation_stream": stream,
        "arm": job["arm"],
        "registration_id": registration["registration_id"],
        "simulation_mode": "physical_v1",
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
        "n_envs": protocol["n_envs"],
        "critic_only_calibration_updates": settings["critic_calibration_steps"],
        "condition_stream_seed": env_seed,
    }
    write_json(output / "manifest.json", {**identity, "inputs": job["inputs"]})
    train_env = validation = model = journal = None
    completed = False
    try:
        train_env = vectors(protocol, env_seed, cpus)
        validation = vectors(protocol, env_seed, cpus, validation=True)
        model = SAC.load(
            job["inputs"]["late_model"]["path"],
            env=train_env,
            device=device,
            force_reset=True,
            tensorboard_log=str(output / "tensorboard"),
        )
        model.set_logger(configure(str(output / "tensorboard"), ["tensorboard"]))
        check_policy_semantics(model, "physical_v1")
        model.load_replay_buffer(job["inputs"]["late_replay"]["path"])
        if model.replay_buffer.n_envs != protocol["n_envs"]:
            raise ValueError("Replay and online environment counts must match")
        if (
            model.gradient_steps != -1
            or model.train_freq.frequency != protocol["algorithm"]["train_freq"]
        ):
            raise ValueError(
                "Loaded checkpoint update cadence differs from the protocol"
            )
        for attribute in ("gamma", "tau", "batch_size"):
            expected = (
                protocol["low"]["gamma"]
                if attribute == "gamma"
                else protocol["algorithm"][attribute]
            )
            if getattr(model, attribute) != expected:
                raise ValueError(f"Loaded {attribute} differs from the protocol")
        expected_lr = protocol["algorithm"]["learning_rate"]
        if any(model.lr_schedule(progress) != expected_lr for progress in (0, 0.5, 1)):
            raise ValueError("Loaded learning-rate schedule differs from the protocol")
        base_step, base_updates = model.num_timesteps, model._n_updates
        initial_position = model.replay_buffer.pos
        unchanged = actor_identity(model)
        donor = None
        if job["arm"] == "critic_restore":
            donor = SAC.load(job["inputs"]["donor_model"]["path"], device=device)
        treatment = apply_treatment(model, job["arm"], donor)
        del donor
        calibration_rng = set_update_rng(base_seed, stream, 20)
        calibration = calibrate_critic(
            model,
            settings["critic_calibration_steps"],
            model.batch_size,
        )
        if actor_identity(model) != unchanged:
            raise AssertionError(
                "Calibration failed the actor/entropy preservation contract"
            )
        update_rng = set_update_rng(base_seed, stream, 30)
        train_env.seed(env_seed)
        model.action_space.seed(env_seed)
        model._last_obs = None
        model.ep_info_buffer = deque(maxlen=model._stats_window_size)
        model.ep_success_buffer = deque(maxlen=model._stats_window_size)
        write_json(
            output / "initialization.json",
            {
                "unchanged_actor_entropy": unchanged,
                "treatment_initial_state": treatment,
                "calibration": calibration,
                "calibration_rng": calibration_rng,
                "online_rng": update_rng,
                "base_timesteps": base_step,
                "base_sac_updates": base_updates,
                "initial_replay_position": initial_position,
            },
        )
        model.c2g_context = {
            "simulation_mode": "physical_v1",
            "experiment": "online-value-state-pilot",
            "registration_id": registration["registration_id"],
            **identity,
            "inputs": job["inputs"],
            "parent_context": model.c2g_context,
        }
        journal = PilotJournal(
            protocol,
            output,
            validation,
            identity,
            base_step,
            base_updates,
        )
        journal.init_callback(model)
        journal.num_timesteps = base_step
        journal.evaluate()
        model.learn(
            total_timesteps=settings["online_transitions"],
            callback=journal,
            reset_num_timesteps=False,
            tb_log_name="online",
            log_interval=1,
        )
        new_steps = model.num_timesteps - base_step
        expected = settings["online_transitions"]
        if new_steps != expected or model._n_updates - base_updates != expected:
            raise AssertionError(
                "Online transition/update budget did not match registration"
            )
        final_position = (
            initial_position + new_steps // protocol["n_envs"]
        ) % model.replay_buffer.buffer_size
        if model.replay_buffer.pos != final_position:
            raise AssertionError(
                "Replay position does not reflect all new online transitions"
            )
        journal.save_checkpoint()
        journal.evaluate()
        atomic_model(model, output / "final_model.zip")
        model.save_replay_buffer(output / "final_replay_buffer.pkl")
        journal.status(
            "completed",
            actual_replay_position=model.replay_buffer.pos,
            expected_replay_position=final_position,
            qualified_checkpoint=(output / "best_model/best_model.zip").exists(),
        )
        completed = True
    finally:
        if not completed:
            error = sys.exc_info()[1]
            write_json(
                output / "failure.json",
                {
                    **identity,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timesteps": model.num_timesteps if model is not None else None,
                },
            )
        for vector in (validation, train_env):
            if vector is not None:
                vector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    job = next(j for j in registration["jobs"] if j["id"] == args.job)
    run_job(
        registration,
        job,
        Path(registration["output_root"]) / "runs" / job["id"],
        args.cpus,
        args.device,
    )


if __name__ == "__main__":
    main()
