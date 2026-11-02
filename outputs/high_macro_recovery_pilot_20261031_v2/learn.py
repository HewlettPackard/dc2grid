"""Matched finite-task SAC learners with explicit warmup and post-update evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from functools import partial
from pathlib import Path

# isort: off
from study_common import (
    TASK_SCHEMA,
    EXTRA_FEATURES,
    file_hash,
    load_json,
    tensor_digest,
    write_json,
)
from macro_task import MacroQuantityTask
from evaluate import cap_worker, evaluate_suite, evaluation_vector
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.vec_env import SubprocVecEnv
# isort: on


class QuantitySAC(SAC):
    def __init__(
        self,
        *args,
        teacher_fraction: float = 0.98,
        teacher_explore_every: int = 20,
        **kwargs,
    ):
        self.teacher_fraction = float(teacher_fraction)
        self.teacher_explore_every = int(teacher_explore_every)
        if self.teacher_explore_every < 2:
            raise ValueError("Teacher exploration interval must be at least two")
        super().__init__(*args, **kwargs)
        self.teacher_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed if self.seed is not None else 0, 619])
        )
        self.warm_exploration_count = 0

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            action = np.full((n_envs, 1), self.teacher_fraction, dtype=np.float32)
            local_step = self.num_timesteps // n_envs
            for rank in range(n_envs):
                offset = rank * max(1, self.teacher_explore_every // n_envs)
                if (local_step + offset) % self.teacher_explore_every == 0:
                    action[rank, 0] = self.teacher_rng.uniform(0.0, 1.0)
                    self.warm_exploration_count += 1
            return action, self.policy.scale_action(action)
        return super()._sample_action(learning_starts, action_noise, n_envs)


def initialize_from_teacher(model: SAC, protocol: dict) -> dict:
    fraction = float(protocol["teacher_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("Imitation initializer must be inside the quantity interval")
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(math.log(protocol["initial_actor_std"]))
    target = torch.full_like(model.actor.mu.bias, math.atanh(2 * fraction - 1))
    optimizer = torch.optim.SGD([model.actor.mu.bias], lr=0.1)
    losses = []
    for _ in range(protocol["supervised_bias_updates"]):
        loss = F.mse_loss(model.actor.mu.bias, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.actor.optimizer.zero_grad()
    learned = float(((torch.tanh(model.actor.mu.bias) + 1) / 2).detach().cpu()[0])
    if abs(learned - fraction) > 1e-5:
        raise AssertionError("Teacher initializer did not learn its constant action")
    if model.actor.optimizer.state:
        raise AssertionError("Supervised initialization polluted the SAC Adam state")
    return {
        "kind": "constant_fraction_supervised_bias_initialization",
        "not_a_state_dependent_selector_or_RL_result": True,
        "updates": protocol["supervised_bias_updates"],
        "target_fraction": fraction,
        "learned_fraction": learned,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "actor_tensor_digest": tensor_digest(model.actor.state_dict()),
        "critic_tensor_digest": tensor_digest(model.critic.state_dict()),
        "critic_target_tensor_digest": tensor_digest(model.critic_target.state_dict()),
        "sac_actor_optimizer_updates": 0,
    }


def calibrate_critic(model: SAC, steps: int) -> dict:
    if model.ent_coef_optimizer is not None or model.target_update_interval != 1:
        raise ValueError("Calibration requires fixed entropy and per-update targets")
    actor_before = tensor_digest(model.actor.state_dict())
    model.critic.set_training_mode(True)
    if model.actor.optimizer.state:
        raise AssertionError("Calibration must precede actor Adam updates")
    first_updates = model._n_updates
    losses = []
    batch_norm = get_parameters_by_name(model.critic, ["running_"])
    target_norm = get_parameters_by_name(model.critic_target, ["running_"])
    for _ in range(steps):
        data = model.replay_buffer.sample(
            model.batch_size, env=model._vec_normalize_env
        )
        with torch.no_grad():
            next_actions, next_log_prob = model.actor.action_log_prob(
                data.next_observations
            )
            next_values = torch.cat(
                model.critic_target(data.next_observations, next_actions), dim=1
            )
            next_values = next_values.min(dim=1, keepdim=True).values
            target = data.rewards + (1 - data.dones) * model.gamma * (
                next_values - model.ent_coef_tensor * next_log_prob.reshape(-1, 1)
            )
        values = model.critic(data.observations, data.actions)
        loss = 0.5 * sum(F.mse_loss(value, target) for value in values)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite critic calibration loss")
        model.critic.optimizer.zero_grad()
        loss.backward()
        model.critic.optimizer.step()
        polyak_update(
            model.critic.parameters(), model.critic_target.parameters(), model.tau
        )
        polyak_update(batch_norm, target_norm, 1.0)
        losses.append(float(loss.detach().cpu()))
    model._n_updates += steps
    if (
        tensor_digest(model.actor.state_dict()) != actor_before
        or model.actor.optimizer.state
    ):
        raise AssertionError("Critic calibration changed the actor or its optimizer")
    return {
        "critic_only_updates": model._n_updates - first_updates,
        "actor_unchanged": True,
        "actor_optimizer_unchanged": True,
        "mean_loss": float(np.mean(losses)),
        "last_loss": losses[-1],
    }


def make_training_worker(protocol: dict, kappa: float, seed: int, rank: int, cpus):
    cap_worker(cpus[rank + 1])
    return Monitor(MacroQuantityTask(protocol, kappa, run_seed=seed, rank=rank))


class Journal(BaseCallback):
    def __init__(self, output: Path, context: dict):
        super().__init__()
        self.output, self.context = output, context
        self.service_steps = self.recovery_steps = self.completed_episodes = 0
        self.started = time.monotonic()
        self.last_status = 0
        self.calibration_updates = 0
        self.action_count = 0
        self.action_sum = self.action_squared_sum = 0.0
        self.action_min = math.inf
        self.action_max = -math.inf

    def publish(self, state: str, **extra):
        losses = {}
        for key, value in self.model.logger.name_to_value.items():
            if key.startswith("train/") and np.isscalar(value):
                value = float(value)
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite training metric: {key}")
                losses[key] = value
        write_json(
            self.output / "status.json",
            {
                **self.context,
                "state": state,
                "pid": os.getpid(),
                "updated_unix": time.time(),
                "elapsed_seconds": time.monotonic() - self.started,
                "macro_transitions": self.model.num_timesteps,
                "critic_updates": self.model._n_updates,
                "calibration_updates": self.calibration_updates,
                "normal_sac_updates": self.model._n_updates - self.calibration_updates,
                "service_fast_steps": self.service_steps,
                "recovery_fast_steps": self.recovery_steps,
                "completed_training_episodes": self.completed_episodes,
                "torch_threads": torch.get_num_threads(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "losses": losses,
                "action_statistics": {
                    "count": self.action_count,
                    "min": self.action_min if self.action_count else None,
                    "max": self.action_max if self.action_count else None,
                    "mean": self.action_sum / self.action_count
                    if self.action_count
                    else None,
                    "std": math.sqrt(
                        max(
                            0.0,
                            self.action_squared_sum / self.action_count
                            - (self.action_sum / self.action_count) ** 2,
                        )
                    )
                    if self.action_count
                    else None,
                },
                **extra,
            },
        )

    def _on_step(self):
        if not np.isfinite(self.locals["rewards"]).all():
            raise FloatingPointError("Nonfinite macro training reward")
        for rank, info in enumerate(self.locals["infos"]):
            action = info["quantity_fraction"]
            self.action_count += 1
            self.action_sum += action
            self.action_squared_sum += action**2
            self.action_min = min(self.action_min, action)
            self.action_max = max(self.action_max, action)
            self.service_steps += info["service_fast_steps"]
            self.recovery_steps += info["recovery_fast_steps"]
            if "c2g_episode" in info:
                self.completed_episodes += 1
                row = {
                    "macro_transitions": self.num_timesteps,
                    "worker": rank,
                    **info["c2g_episode"],
                }
                with (self.output / "training_episodes.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
        if self.num_timesteps - self.last_status >= 256:
            self.last_status = self.num_timesteps
            self.publish("collecting", status_position="before_this_rollout_update")
        return True


def train(
    protocol: dict, variant: str, seed: int, output: Path, cpus: list[int], device: str
):
    from register import verify_registration, verify_training_gate

    verify_training_gate(protocol)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    os.sched_setaffinity(0, {cpus[0]})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested learner GPU is unavailable")
    kappa = protocol["variants"][variant]
    context = {
        "task_schema": TASK_SCHEMA,
        "simulation_mode": "physical_v1",
        "variant": variant,
        "kappa": kappa,
        "training_seed": seed,
        "registration_id": protocol["registration_id"],
        "hardware_checkpoint": protocol["hardware"],
        "extra_observation_features": list(EXTRA_FEATURES),
        "device": device,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "macro_decision_seconds": 900,
        "hardware_decision_seconds": 5,
        "recovery_seconds": protocol["recovery"]["duration_seconds"],
        "finite_task_gamma": 1.0,
    }
    write_json(output / "manifest.json", context)
    training = validation = model = journal = None
    successful = False
    best_score = -math.inf
    try:
        training = SubprocVecEnv(
            [
                partial(make_training_worker, protocol, kappa, seed, rank, cpus)
                for rank in range(protocol["n_envs"])
            ],
            start_method="spawn",
        )
        write_json(
            output / "initial_worker_runtime.json", training.env_method("worker_status")
        )
        validation = evaluation_vector(protocol, kappa, cpus)
        model = QuantitySAC(
            "MlpPolicy",
            training,
            teacher_fraction=protocol["teacher_fraction"],
            teacher_explore_every=protocol["teacher_exploration"]["every_local_steps"],
            seed=seed,
            device=device,
            learning_starts=protocol["learning_starts"],
            **protocol["algorithm"],
        )
        model.c2g_context = {**context, "protocol": protocol}
        model.set_logger(configure(str(output / "metrics"), ["csv", "tensorboard"]))
        initialization = initialize_from_teacher(model, protocol)
        write_json(output / "initialization.json", initialization)
        model.save(output / "imitation_model.zip")
        journal = Journal(output, context)
        journal.init_callback(model)
        journal.publish("validating_initialization")
        initial_report = evaluate_suite(
            model,
            validation,
            protocol,
            output / "validation/step_0",
            {**context, "macro_transitions": 0, "kind": "imitation_only"},
        )
        write_json(output / "initial_validation.json", initial_report)

        model.learn(
            protocol["learning_starts"],
            callback=journal,
            reset_num_timesteps=False,
            log_interval=1,
        )
        if model.num_timesteps != protocol["learning_starts"] or model._n_updates != 0:
            raise AssertionError("Teacher warmup did not stop before optimizer updates")
        warm_actions = model.policy.unscale_action(
            model.replay_buffer.actions[: model.replay_buffer.pos].reshape(-1, 1)
        )
        write_json(
            output / "warmup_actions.json",
            {
                "macro_transitions": model.num_timesteps,
                "teacher_fraction": protocol["teacher_fraction"],
                "exploration": protocol["teacher_exploration"],
                "exploration_actions": model.warm_exploration_count,
                "actions": warm_actions.flatten().tolist(),
                "is_teacher_guided_not_constant_only": True,
            },
        )
        write_json(output / "worker_runtime.json", training.env_method("worker_status"))
        calibration = calibrate_critic(model, protocol["critic_calibration_updates"])
        journal.calibration_updates = protocol["critic_calibration_updates"]
        write_json(output / "critic_calibration.json", calibration)
        for target in protocol["milestones"]:
            model.learn(
                target - model.num_timesteps,
                callback=journal,
                reset_num_timesteps=False,
                log_interval=1,
            )
            if model.num_timesteps != target:
                raise AssertionError(
                    "Macro budget overshot a registered update boundary"
                )
            expected = (
                target - protocol["learning_starts"] + journal.calibration_updates
            )
            if model._n_updates != expected:
                raise AssertionError("Gradient-update budget differs from registration")
            checkpoint = output / "checkpoints" / f"step_{target}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            model.save(checkpoint / "model.zip")
            model.save_replay_buffer(checkpoint / "replay.pkl")
            write_json(
                checkpoint / "metadata.json",
                {
                    **context,
                    "macro_transitions": target,
                    "critic_updates": model._n_updates,
                    "normal_sac_updates": model._n_updates
                    - journal.calibration_updates,
                    "after_all_optimizer_updates": True,
                    "model_sha256": file_hash(checkpoint / "model.zip"),
                    "replay_sha256": file_hash(checkpoint / "replay.pkl"),
                    "resume_is_not_bit_exact_without_live_environment_state": True,
                },
            )
            journal.publish("validating", checkpoint_macro_transitions=target)
            report = evaluate_suite(
                model,
                validation,
                protocol,
                output / f"validation/step_{target}",
                {
                    **context,
                    "macro_transitions": target,
                    "critic_updates": model._n_updates,
                    "kind": "SAC",
                },
            )
            score = report["mean_cashflow_closed_recovery_usd"]
            if report["all_qualified"] and score is not None and score > best_score:
                best_score = score
                best = output / "best_model"
                best.mkdir(exist_ok=True)
                model.save(best / "model.zip")
                write_json(
                    best / "selection.json",
                    {
                        "macro_transitions": target,
                        "score": score,
                        "qualified_cells": report["qualified_cells"],
                        "checkpoint_sha256": file_hash(best / "model.zip"),
                        "source_validation": str(
                            output / f"validation/step_{target}/summary.json"
                        ),
                        "selection_uses_sealed_test": False,
                    },
                )
            journal.publish(
                "training",
                best_qualified_cash=best_score if math.isfinite(best_score) else None,
            )
        model.save(output / "final_model.zip")
        write_json(
            output / "inflight_training_episodes.json", training.env_method("progress")
        )
        write_json(
            output / "final_worker_runtime.json", training.env_method("worker_status")
        )
        verify_registration(protocol)
        journal.publish(
            "completed",
            qualified_checkpoint=(output / "best_model/model.zip").exists(),
            final_model_sha256=file_hash(output / "final_model.zip"),
        )
        successful = True
    finally:
        if not successful:
            error = sys.exc_info()[1]
            write_json(
                output / "failure.json",
                {
                    **context,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "macro_transitions": model.num_timesteps
                    if model is not None
                    else 0,
                },
            )
        for vector in (validation, training):
            if vector is not None:
                vector.close()
        if model is not None:
            model.logger.close()


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Learner interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train(
        load_json(args.protocol),
        args.variant,
        args.seed,
        args.output,
        args.cpus,
        args.device,
    )
