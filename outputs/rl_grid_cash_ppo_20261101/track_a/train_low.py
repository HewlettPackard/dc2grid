"""Matched native PPO/SAC fits, with exact budgets and BC actor initialization."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
import traceback
from functools import partial
from pathlib import Path

# isort: off
from ppo_adapter import LatentActionEnv, initialize_from_bc
from economy_context import (
    cap_cpu,
    file_hash,
    read_json,
    verify,
    write_json,
)
from fit_bc import parameter_digest
from low_task import DiagnosticTask
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
# isort: on

ROOT = Path(__file__).resolve().parent


class GuardedTask(DiagnosticTask):
    def condition(self):
        cell = super().condition()
        if cell["seed"] in self.protocol["forbidden_eval_seeds"]:
            raise ValueError("Training seed collided with a reserved evaluation stream")
        return cell


def worker(protocol, seed, rank, cpu, algorithm):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(cpu)
    task = GuardedTask(protocol, seed=seed, rank=rank)
    return LatentActionEnv(task) if algorithm == "ppo" else task


def actor_parameters(model, algorithm):
    if algorithm == "sac":
        return tuple(model.actor.parameters())
    return (
        tuple(model.policy.mlp_extractor.policy_net.parameters())
        + tuple(model.policy.action_net.parameters())
        + (model.policy.log_std,)
    )


def actor_identity(model, algorithm):
    if algorithm == "sac":
        return {"actor": parameter_digest(model.actor)}
    return {
        "body": parameter_digest(model.policy.mlp_extractor.policy_net),
        "mean": parameter_digest(model.policy.action_net),
        "log_std": model.policy.log_std.detach().cpu().tolist(),
    }


def attach_update_counters(model, algorithm):
    model.c2g_update_counts = {"actor": 0, "critic": 0, "critic_only": 0}
    model.c2g_calibrating = False

    def actor_step(optimizer, args, kwargs):
        if not model.c2g_calibrating:
            model.c2g_update_counts["actor"] += 1

    def critic_step(optimizer, args, kwargs):
        model.c2g_update_counts["critic"] += 1
        if model.c2g_calibrating:
            model.c2g_update_counts["critic_only"] += 1

    def joint_step(optimizer, args, kwargs):
        actor_step(optimizer, args, kwargs)
        critic_step(optimizer, args, kwargs)

    if algorithm == "sac":
        return [
            model.actor.optimizer.register_step_post_hook(actor_step),
            model.critic.optimizer.register_step_post_hook(critic_step),
        ]
    return [model.policy.optimizer.register_step_post_hook(joint_step)]


def warmup_and_calibrate(
    model,
    algorithm,
    *,
    warmup_steps,
    critic_updates,
    callback=None,
):
    if algorithm not in ("ppo", "sac") or warmup_steps <= 0 or critic_updates <= 0:
        raise ValueError("Invalid native calibration configuration")
    if model.num_timesteps or any(model.c2g_update_counts.values()):
        raise ValueError("Warmup and calibration require a fresh initialized model")
    before = actor_identity(model, algorithm)
    parameters = actor_parameters(model, algorithm)
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("Expected a trainable initialized actor before calibration")
    if algorithm == "sac":
        starts, gradients = model.learning_starts, model.gradient_steps
        model.learning_starts, model.gradient_steps = 0, 0
        try:
            model.learn(warmup_steps, callback=callback, reset_num_timesteps=False)
        finally:
            model.learning_starts, model.gradient_steps = starts, gradients
        if model._n_updates or any(model.c2g_update_counts.values()):
            raise AssertionError("SAC collection-only warmup performed an update")
    else:
        rollout_size = model.n_steps * model.n_envs
        if warmup_steps % rollout_size or rollout_size % model.batch_size:
            raise ValueError("PPO warmup must contain complete, divisible rollouts")
        epochs = critic_updates // (warmup_steps // model.batch_size)
        if epochs * (warmup_steps // model.batch_size) != critic_updates:
            raise ValueError("PPO calibration cannot realize the exact update budget")
        original_epochs = model.n_epochs
        model.n_epochs = epochs
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.c2g_calibrating = True
    try:
        if algorithm == "sac":
            model.train(critic_updates, model.batch_size)
        else:
            model.learn(warmup_steps, callback=callback, reset_num_timesteps=False)
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)
        model.c2g_calibrating = False
        if algorithm == "ppo":
            model.n_epochs = original_epochs
    if actor_identity(model, algorithm) != before:
        raise AssertionError("Critic calibration changed the initialized actor")
    optimizer = model.actor.optimizer if algorithm == "sac" else model.policy.optimizer
    if any(optimizer.state.get(parameter) for parameter in parameters):
        raise AssertionError("Critic calibration created actor optimizer state")
    if model.num_timesteps != warmup_steps or model.c2g_update_counts != {
        "actor": 0,
        "critic": critic_updates,
        "critic_only": critic_updates,
    }:
        raise AssertionError("Warmup or critic-update accounting differs from the plan")
    return {
        "warmup_transitions": model.num_timesteps,
        "critic_only_updates": critic_updates,
        "actor_unchanged": before,
        "actor_optimizer_state_empty": True,
        "ppo_calibration_uses_each_on_policy_warmup_rollout": algorithm == "ppo",
        "sac_calibration_uses_the_complete_warmup_replay": algorithm == "sac",
    }


class Journal(BaseCallback):
    def __init__(self, output: Path, algorithm: str):
        super().__init__()
        self.output, self.algorithm = output, algorithm
        self.last_written, self.episodes = 0, 0
        self.started = time.monotonic()

    def _on_step(self):
        for info in self.locals["infos"]:
            if info.get("TimeLimit.truncated", False):
                raise AssertionError("A finite-task terminal was marked for bootstrap")
            episode = info.get("episode_result")
            if episode is not None:
                with (self.output / "training_episodes.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {**episode, "aggregate_timesteps": self.num_timesteps},
                            allow_nan=False,
                        )
                        + "\n"
                    )
                self.episodes += 1
        if self.num_timesteps - self.last_written >= 4096:
            write_json(
                self.output / "progress.json",
                {
                    "timesteps": self.num_timesteps,
                    "optimizer_updates": self.model.c2g_update_counts,
                    "sb3_update_counter": self.model._n_updates,
                    "training_episodes": self.episodes,
                    "elapsed_seconds": time.monotonic() - self.started,
                    "state": "running",
                },
            )
            self.last_written = self.num_timesteps
        return True

    def _on_rollout_end(self):
        if self.num_timesteps % 4096:
            return
        if self.algorithm == "sac":
            buffer = self.model.replay_buffer
            count = min(128, buffer.size())
            indices = np.arange(buffer.pos - count, buffer.pos) % buffer.buffer_size
            observations = buffer.observations[indices].reshape(-1, 24)
            actions = buffer.actions[indices].reshape(-1, 3)
            with torch.no_grad():
                mean, log_std, _ = self.model.actor.get_action_dist_params(
                    torch.as_tensor(observations, device=self.model.device),
                )
            rail = (np.abs(actions) >= 1).mean(axis=0)
        else:
            buffer = self.model.rollout_buffer
            observations = buffer.observations.reshape(-1, 24)
            actions = buffer.actions.reshape(-1, 3)
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(
                    torch.as_tensor(observations, device=self.model.device),
                ).distribution
                mean, log_std = distribution.mean, distribution.stddev.log()
                rail = (
                    (torch.tanh(torch.as_tensor(actions)).abs() >= 1)
                    .double()
                    .mean(dim=0)
                    .numpy()
                )
        record = {
            "timesteps": self.num_timesteps,
            "scope": "collected states, before the next optimizer block",
            "sample_size": len(observations),
            "max_abs_pre_tanh_mean": mean.abs().max(dim=0).values.cpu().tolist(),
            "mean_latent_std": log_std.exp().mean(dim=0).cpu().tolist(),
            "saturated_action_fraction": rail.tolist(),
        }
        with (self.output / "policy_diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def fit(args):
    protocol = read_json(args.protocol)
    commit = verify(protocol)
    if (
        args.algorithm not in protocol["algorithms"]
        or args.seed not in protocol["seeds"]
    ):
        raise ValueError("Unregistered algorithm or initializer")
    if len(args.cpus) != 5 or len(set(args.cpus)) != 5:
        raise ValueError("Use one learner and four distinct environment cores")
    output = ROOT / "runs" / args.stage / f"{args.algorithm}_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    cap_cpu(args.cpus[0])
    source = output / "source"
    source.mkdir()
    for name, digest in protocol["source_hashes"].items():
        path = Path(name)
        if path.is_relative_to(ROOT):
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if file_hash(target) != digest:
                raise AssertionError("Training-source copy changed")
    initializer = protocol["initializers"][str(args.seed)]
    write_json(
        output / "manifest.json",
        {
            "registration_id": protocol["registration_id"],
            "implementation_commit": commit,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "stage": args.stage,
            "cpus": args.cpus,
            "source_hashes": protocol["source_hashes"],
            "initializer": initializer,
            "budget": protocol["budgets"][args.stage],
        },
    )
    online_seed = protocol["online_seed_by_initializer"][str(args.seed)]
    vector = SubprocVecEnv(
        [
            partial(
                worker, protocol, online_seed, rank, args.cpus[rank + 1], args.algorithm
            )
            for rank in range(4)
        ],
        start_method="spawn",
    )
    handles, model = [], None
    try:
        if args.algorithm == "sac":
            model = SAC.load(
                initializer["path"],
                env=vector,
                device=args.device,
                seed=online_seed,
                **copy.deepcopy(protocol["sac"]),
            )
            initialization = {
                "actor": parameter_digest(model.actor),
                "critic": parameter_digest(model.critic),
                "critic_target": parameter_digest(model.critic_target),
            }
        else:
            teacher = SAC.load(initializer["path"], device=args.device)
            model = PPO(
                "MlpPolicy",
                vector,
                device=args.device,
                seed=online_seed,
                policy_kwargs={
                    "net_arch": {
                        "pi": list(protocol["architecture"]),
                        "vf": list(protocol["architecture"]),
                    },
                    "activation_fn": torch.nn.ReLU,
                    "log_std_init": math.log(protocol["initial_sigma"]),
                },
                **copy.deepcopy(protocol["ppo"]),
            )
            initialization = initialize_from_bc(model, teacher)
            del teacher
        model.c2g_context = copy.deepcopy(
            {
                **model.c2g_context,
                "kind": f"native_{args.algorithm}_bc_initialized_track_a",
                "registration_id": protocol["registration_id"],
                "initializer": initializer,
                "no_bc_or_kl_training_prior": True,
                "reward": "unchanged original low-level reward",
            }
        )
        model.set_logger(configure(str(output / "metrics"), ["csv"]))
        handles = attach_update_counters(model, args.algorithm)
        journal = Journal(output, args.algorithm)
        calibration = warmup_and_calibrate(
            model,
            args.algorithm,
            warmup_steps=protocol["warmup_transitions"],
            critic_updates=protocol["critic_calibration_updates"],
            callback=journal,
        )
        write_json(output / "initialization.json", initialization)
        write_json(output / "calibration.json", calibration)
        checkpoints = []
        for target in protocol["milestones"][args.stage]:
            model.learn(
                target - model.num_timesteps,
                callback=journal,
                reset_num_timesteps=False,
            )
            if model.num_timesteps != target:
                raise AssertionError(
                    "The native learner overshot its transition budget"
                )
            if args.algorithm == "sac" and model.c2g_update_counts != {
                "actor": target - protocol["warmup_transitions"],
                "critic": target
                - protocol["warmup_transitions"]
                + protocol["critic_calibration_updates"],
                "critic_only": protocol["critic_calibration_updates"],
            }:
                raise AssertionError("Native SAC optimizer accounting changed")
            if any(
                not torch.isfinite(parameter).all()
                for parameter in model.policy.parameters()
            ):
                raise FloatingPointError("Nonfinite learned policy parameters")
            model.c2g_context.update(
                online_transitions=target,
                optimizer_updates=copy.deepcopy(model.c2g_update_counts),
                true_finite_terminal=True,
            )
            checkpoint = output / f"step_{target}.zip"
            model.save(checkpoint)
            checkpoints.append(
                {
                    "timesteps": target,
                    "path": str(checkpoint),
                    "sha256": file_hash(checkpoint),
                    "optimizer_updates": copy.deepcopy(model.c2g_update_counts),
                    "selected_for_primary_evaluation": target
                    == protocol["budgets"][args.stage],
                }
            )
            write_json(output / "checkpoints.json", checkpoints)
            model.logger.dump(step=target)
        prefixes = vector.env_method("training_prefix")
        diagnostics = []
        for rank in range(4):
            diagnostics.extend(
                vector.env_method(
                    "save_diagnostics",
                    str(output / f"worker_{rank}_transitions.npz"),
                    indices=rank,
                )
            )
        if sum(row["transitions"] for row in diagnostics) != model.num_timesteps:
            raise AssertionError("Full physical transition diagnostics are incomplete")
        if args.algorithm == "sac":
            model.save_replay_buffer(output / "final_replay.pkl")
        else:
            buffer = model.rollout_buffer
            np.savez_compressed(
                output / "last_rollout_only.npz",
                observations=buffer.observations,
                actions=buffer.actions,
                rewards=buffer.rewards,
                returns=buffer.returns,
                advantages=buffer.advantages,
                episode_starts=buffer.episode_starts,
                values=buffer.values,
                log_probs=buffer.log_probs,
            )
        verify(protocol)
        write_json(
            output / "completion.json",
            {
                "status": "completed",
                "registration_id": protocol["registration_id"],
                "algorithm": args.algorithm,
                "seed": args.seed,
                "stage": args.stage,
                "num_timesteps": model.num_timesteps,
                "optimizer_updates": model.c2g_update_counts,
                "sb3_update_counter_not_optimizer_count_for_ppo": model._n_updates,
                "training_episodes": journal.episodes,
                "final_checkpoint": checkpoints[-1],
                "training_prefixes": prefixes,
                "full_transition_diagnostics": diagnostics,
                "elapsed_seconds": time.monotonic() - journal.started,
                "no_recovery_in_original_reward_training": True,
                "recovery_required_in_every_evaluation": True,
            },
        )
        write_json(
            output / "progress.json",
            {
                "timesteps": model.num_timesteps,
                "optimizer_updates": model.c2g_update_counts,
                "training_episodes": journal.episodes,
                "elapsed_seconds": time.monotonic() - journal.started,
                "state": "completed",
            },
        )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "algorithm": args.algorithm,
                "seed": args.seed,
                "stage": args.stage,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "timesteps": model.num_timesteps if model is not None else None,
            },
        )
        raise
    finally:
        for handle in handles:
            handle.remove()
        vector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "registration.json")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpus", type=int, nargs=5, required=True)
    fit(parser.parse_args())
