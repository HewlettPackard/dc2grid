"""Native learner for the registered fractional-budget/potential factorial."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import time
import traceback
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "track_b"))

# isort: off
from train_macro import (
    MacroJournal,
    MacroLatentEnv,
    initialize_actor,
    attach_update_counters,
    warmup_and_calibrate,
)
from economy_context import cap_cpu, file_hash, read_json, verify, write_json
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
# isort: on

ROOT = Path(__file__).resolve().parent


def worker(protocol, seed, rank, cpu, algorithm, variant):
    from fraction_macro import CreditPotentialEnv, FractionalDispatchTask

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(cpu)
    task = FractionalDispatchTask(protocol, seed=seed, rank=rank, record_trace=False)
    if variant == "potential":
        task = CreditPotentialEnv(
            task,
            coefficient_usd_per_mwh=protocol["potential_usd_per_mwh"],
        )
    elif variant != "fraction":
        raise ValueError("Unknown fractional-budget ablation")
    return MacroLatentEnv(task) if algorithm == "ppo" else task


def fit(args):
    protocol = read_json(args.protocol)
    commit = verify(protocol)
    if (
        args.algorithm not in protocol["algorithms"]
        or args.variant not in protocol["variants"]
        or args.seed not in protocol["seeds"]
        or len(args.cpus) != 3
        or len(set(args.cpus)) != 3
    ):
        raise ValueError("Unregistered fractional-budget training slot")
    output = ROOT / "runs" / args.stage / f"{args.variant}_{args.algorithm}_{args.seed}"
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
                raise AssertionError("Fractional training source changed")
    write_json(
        output / "manifest.json",
        {
            "registration_id": protocol["registration_id"],
            "implementation_commit": commit,
            "variant": args.variant,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "stage": args.stage,
            "cpus": args.cpus,
            "source_hashes": protocol["source_hashes"],
            "budget_macro_transitions": protocol["budgets"][args.stage],
            "device": "cpu",
        },
    )
    online_seed = protocol["online_seed_by_seed"][str(args.seed)]
    vector = SubprocVecEnv(
        [
            partial(
                worker,
                protocol,
                online_seed,
                rank,
                args.cpus[rank + 1],
                args.algorithm,
                args.variant,
            )
            for rank in range(2)
        ],
        start_method="spawn",
    )
    handles, model, journal = [], None, None
    try:
        if any(gamma != 1 for gamma in vector.get_attr("required_gamma")):
            raise ValueError("The cash potential requires gamma=1 and true terminals")
        if args.algorithm == "sac":
            model = SAC(
                "MlpPolicy",
                vector,
                device="cpu",
                seed=online_seed,
                policy_kwargs={"net_arch": list(protocol["architecture"])},
                **copy.deepcopy(protocol["sac"]),
            )
        else:
            model = PPO(
                "MlpPolicy",
                vector,
                device="cpu",
                seed=online_seed,
                policy_kwargs={
                    "net_arch": {
                        "pi": list(protocol["architecture"]),
                        "vf": list(protocol["architecture"]),
                    },
                    "activation_fn": torch.nn.ReLU,
                    "ortho_init": False,
                },
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
            "variant": args.variant,
            "track": "economic_fraction",
            "simulation_mode": "physical_v1",
            "observation_dimension": 23,
            "action_dimension": 1,
            "ppo_latent_transform_applied_by_environment_only": args.algorithm == "ppo",
            "action_parameterization": "fraction_of_frozen_start_window_allowance",
            "potential_used_at_deployment": False,
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
                raise AssertionError("Fractional learner overshot its macro budget")
            if args.algorithm == "sac" and model.c2g_update_counts != {
                "actor": target - protocol["warmup_transitions"],
                "critic": target
                - protocol["warmup_transitions"]
                + protocol["critic_calibration_updates"],
                "critic_only": protocol["critic_calibration_updates"],
            }:
                raise AssertionError("Fractional SAC optimizer accounting changed")
            if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
                raise FloatingPointError("Fractional policy became nonfinite")
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
        write_json(
            output / "completion.json",
            {
                "status": "completed",
                "registration_id": protocol["registration_id"],
                "variant": args.variant,
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
                "potential_is_training_only": True,
            },
        )
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
                "variant": args.variant,
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
    parser.add_argument("--variant", choices=("fraction", "potential"), required=True)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--cpus", type=int, nargs=3, required=True)
    fit(parser.parse_args())
