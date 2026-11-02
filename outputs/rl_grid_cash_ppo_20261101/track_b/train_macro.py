"""Native PPO/SAC training for the separately disclosed economic-dispatch task."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
import traceback
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "track_a"))

# isort: off
from economy_macro import EconomicDispatchTask
from economy_context import cap_cpu, file_hash, read_json, verify, write_json
from train_low import attach_update_counters, warmup_and_calibrate
from fit_bc import parameter_digest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.vec_env import SubprocVecEnv
# isort: on

ROOT = Path(__file__).resolve().parent


class MacroLatentEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        if (
            env.observation_space.shape != (23,)
            or env.action_space.shape != (1,)
            or not np.all(env.action_space.low == -1)
            or not np.all(env.action_space.high == 1)
        ):
            raise ValueError("Wrong economic-dispatch observation/action contract")
        self.action_space = gym.spaces.Box(-20.0, 20.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        latent = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(latent):
            raise ValueError("PPO macro action is outside its declared latent box")
        scaled = torch.tanh(torch.from_numpy(latent)).numpy()
        return -1.0 + 0.5 * (scaled + 1.0) * 2.0


def worker(protocol, seed, rank, cpu, algorithm):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(cpu)
    task = EconomicDispatchTask(protocol, seed=seed, rank=rank, record_trace=False)
    return MacroLatentEnv(task) if algorithm == "ppo" else task


def initialize_actor(model, algorithm, seed, sigma, architecture):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        body = torch.nn.Sequential(*create_mlp(23, -1, architecture, torch.nn.ReLU))
    target = (
        model.actor.latent_pi
        if algorithm == "sac"
        else model.policy.mlp_extractor.policy_net
    )
    mean = model.actor.mu if algorithm == "sac" else model.policy.action_net
    with torch.no_grad():
        target.load_state_dict(body.state_dict(), strict=True)
        mean.weight.zero_()
        mean.bias.zero_()
        if algorithm == "sac":
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(math.log(sigma))
            log_std = model.actor.log_std.bias
        else:
            model.policy.log_std.fill_(math.log(sigma))
            log_std = model.policy.log_std
    return {
        "actor_body_sha256": parameter_digest(target),
        "mean_head_sha256": parameter_digest(mean),
        "initial_log_std": log_std.detach().cpu().tolist(),
        "initial_deterministic_overlay_mw": 0.0,
        "initial_stochastic_distribution_shared": True,
        "no_bc_initializer_or_imitation_prior": True,
    }


class MacroJournal(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output, self.episodes = output, 0
        self.started = time.monotonic()
        self.observations, self.following, self.actions = [], [], []
        self.rewards, self.dones = [], []

    def _on_step(self):
        following = self.locals["new_obs"].copy()
        for rank, info in enumerate(self.locals["infos"]):
            if info.get("TimeLimit.truncated", False):
                raise AssertionError(
                    "The economic task must not bootstrap a time limit"
                )
            if self.locals["dones"][rank]:
                following[rank] = info["terminal_observation"]
                episode = info["episode_result"]
                if episode is None:
                    raise ValueError("An economic terminal has no recorded episode")
                with (self.output / "training_episodes.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                **episode,
                                "aggregate_macro_steps": self.num_timesteps,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                self.episodes += 1
        self.observations.append(self.model._last_obs.copy())
        self.following.append(following)
        self.actions.append(self.locals["actions"].copy())
        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"].copy())
        if self.num_timesteps % 256 == 0:
            write_json(
                self.output / "progress.json",
                {
                    "macro_steps": self.num_timesteps,
                    "optimizer_updates": self.model.c2g_update_counts,
                    "episodes": self.episodes,
                    "elapsed_seconds": time.monotonic() - self.started,
                    "state": "running",
                },
            )
        return True

    def save_transitions(self, model, *, partial=False):
        observations = np.stack(self.observations)
        count = observations.shape[0] * observations.shape[1]
        if count != model.num_timesteps and not partial:
            raise AssertionError("Macro transition accounting is incomplete")
        name = (
            "partial_macro_transitions.npz" if partial else "all_macro_transitions.npz"
        )
        path = self.output / name
        np.savez_compressed(
            path,
            observations=observations,
            next_observations=np.stack(self.following),
            actions=np.stack(self.actions),
            rewards=np.stack(self.rewards),
            dones=np.stack(self.dones),
        )
        return {
            "path": str(path),
            "sha256": file_hash(path),
            "transitions": count,
            "partial": partial,
            "model_timesteps": model.num_timesteps,
        }


def fit(args):
    protocol = read_json(args.protocol)
    commit = verify(protocol)
    if (
        args.algorithm not in protocol["algorithms"]
        or args.seed not in protocol["seeds"]
    ):
        raise ValueError("Unregistered economic learner")
    if len(args.cpus) != 3 or len(set(args.cpus)) != 3:
        raise ValueError("Use one learner and two distinct simulator cores")
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
                raise AssertionError("Economic training source changed")
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
            "budget_macro_transitions": protocol["budgets"][args.stage],
            "device": "cpu",
            "task": "economic_dispatch_not_original_tracking",
        },
    )
    online_seed = protocol["online_seed_by_seed"][str(args.seed)]
    vector = SubprocVecEnv(
        [
            partial(
                worker, protocol, online_seed, rank, args.cpus[rank + 1], args.algorithm
            )
            for rank in range(2)
        ],
        start_method="spawn",
    )
    handles, model, journal = [], None, None
    try:
        if any(gamma != 1 for gamma in vector.get_attr("required_gamma")):
            raise ValueError("Economic task requires undiscounted finite-horizon cash")
        common = dict(env=vector, device="cpu", seed=online_seed)
        if args.algorithm == "sac":
            model = SAC(
                "MlpPolicy",
                policy_kwargs={"net_arch": list(protocol["architecture"])},
                **common,
                **copy.deepcopy(protocol["sac"]),
            )
        else:
            model = PPO(
                "MlpPolicy",
                policy_kwargs={
                    "net_arch": {
                        "pi": list(protocol["architecture"]),
                        "vf": list(protocol["architecture"]),
                    },
                    "activation_fn": torch.nn.ReLU,
                    "ortho_init": False,
                },
                **common,
                **copy.deepcopy(protocol["ppo"]),
            )
        initialization = initialize_actor(
            model,
            args.algorithm,
            args.seed,
            protocol["initial_sigma"],
            protocol["architecture"],
        )
        model.c2g_context = {
            "registration_id": protocol["registration_id"],
            "algorithm": args.algorithm,
            "track": "economic_dispatch",
            "simulation_mode": "physical_v1",
            "observation_dimension": 23,
            "action_dimension": 1,
            "ppo_latent_transform_applied_by_environment_only": args.algorithm == "ppo",
            "market_policy": "unchanged_full_residual_zero_ask",
            "gamma": 1.0,
            "full_work_fixed": True,
            "soc_target": 0.5,
        }
        model.set_logger(configure(str(output / "metrics"), ["csv"]))
        handles = attach_update_counters(model, args.algorithm)
        journal = MacroJournal(output)
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
                raise AssertionError("Economic learner overshot its exact macro budget")
            if args.algorithm == "sac" and model.c2g_update_counts != {
                "actor": target - protocol["warmup_transitions"],
                "critic": target
                - protocol["warmup_transitions"]
                + protocol["critic_calibration_updates"],
                "critic_only": protocol["critic_calibration_updates"],
            }:
                raise AssertionError("Economic SAC optimizer accounting changed")
            if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
                raise FloatingPointError("Economic policy parameters became nonfinite")
            model.c2g_context.update(
                macro_transitions=target,
                optimizer_updates=copy.deepcopy(model.c2g_update_counts),
            )
            path = output / f"step_{target}.zip"
            model.save(path)
            checkpoints.append(
                {
                    "macro_steps": target,
                    "path": str(path),
                    "sha256": file_hash(path),
                    "optimizer_updates": copy.deepcopy(model.c2g_update_counts),
                }
            )
            write_json(output / "checkpoints.json", checkpoints)
            model.logger.dump(step=target)
        transitions = journal.save_transitions(model)
        if args.algorithm == "sac":
            model.save_replay_buffer(output / "final_macro_replay.pkl")
        prefixes = vector.env_method("progress")
        verify(protocol)
        completion = {
            "status": "completed",
            "registration_id": protocol["registration_id"],
            "algorithm": args.algorithm,
            "seed": args.seed,
            "stage": args.stage,
            "macro_transitions": model.num_timesteps,
            "optimizer_updates": model.c2g_update_counts,
            "training_episodes": journal.episodes,
            "final_checkpoint": checkpoints[-1],
            "all_macro_transitions": transitions,
            "training_prefixes": prefixes,
            "elapsed_seconds": time.monotonic() - journal.started,
            "physical_service_and_recovery_are_in_the_task": True,
        }
        write_json(output / "completion.json", completion)
        write_json(
            output / "progress.json",
            {
                "macro_steps": model.num_timesteps,
                "optimizer_updates": model.c2g_update_counts,
                "episodes": journal.episodes,
                "state": "completed",
            },
        )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "algorithm": args.algorithm,
                "seed": args.seed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "macro_steps": model.num_timesteps if model is not None else None,
            },
        )
        if journal is not None and journal.observations:
            journal.save_transitions(model, partial=True)
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
    parser.add_argument("--cpus", type=int, nargs=3, required=True)
    fit(parser.parse_args())
