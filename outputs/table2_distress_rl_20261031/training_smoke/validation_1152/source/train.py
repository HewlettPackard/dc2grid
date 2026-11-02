"""Bounded standalone SAC fit with post-update validation and immutable receipts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import signal
import time
import traceback
from functools import partial
from pathlib import Path

# isort: off
from common import cap_cpu, file_hash, identity, read_json, write_json
from evaluate import run_panel
from fast_task import make_training_worker
from fit_bc import corpus, parameter_digest
from history_policy import FEATURE_CONTRACT
from prepare_study import verify_protocol
from sac_core import GuidedSAC

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
# isort: on


class Journal(BaseCallback):
    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory
        self.last_written = 0
        self.episodes = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            episode = info.get("episode_result")
            if episode is not None:
                with (self.directory / "training_episodes.jsonl").open("a") as handle:
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
                self.directory / "progress.json",
                {
                    "timesteps": self.num_timesteps,
                    "critic_updates": self.model._n_updates,
                    "actor_updates": self.model.actor_updates,
                    "completed_training_episodes": self.episodes,
                    "updated_unix": time.time(),
                },
            )
            self.last_written = self.num_timesteps
        return True


def replay_digest(buffer) -> dict:
    if buffer.full:
        raise ValueError("The initial warmup buffer should not be full")
    return {
        key: hashlib.sha256(
            np.ascontiguousarray(getattr(buffer, key)[: buffer.pos]).tobytes()
        ).hexdigest()
        for key in ("observations", "next_observations", "actions", "rewards", "dones")
    }


def candidate_summary(episodes, selection):
    if len(episodes) != 24:
        raise ValueError("Selection requires every one of the 24 development cases")
    qualified = all(row["qualified"] for row in episodes)
    if not all(row["complete"] for row in episodes):
        return {"eligible": False, "qualified": False, "reason": "incomplete"}
    rmse = float(np.mean([row["service"]["tracking_rmse_kw"] for row in episodes]))
    relief = float(
        np.mean(
            [
                row["grid"]["stress_facility_relief_mw"]
                for row in episodes
                if row["cell"]["regd_load_bias"] == 0.5
            ]
        )
    )
    cash_values = [row["cashflow_26h_usd"] for row in episodes]
    cash = (
        float(np.mean(cash_values)) if all(v is not None for v in cash_values) else None
    )
    return {
        "qualified": qualified,
        "qualified_cells": sum(row["qualified"] for row in episodes),
        "rmse_kw": rmse,
        "biased_facility_relief_mw": relief,
        "mean_cashflow_26h_usd": cash,
        "eligible": bool(
            qualified
            and cash is not None
            and rmse < selection["rmse_must_be_below_old_sac_kw"]
            and relief >= selection["biased_relief_must_be_at_least_old_sac_mw"]
            and cash >= selection["closed_cash_must_be_at_least_pid_usd"]
        ),
    }


def fit(args):
    protocol = read_json(args.protocol)
    verify_protocol(protocol)
    expected_identity = identity(protocol)
    if args.variant not in protocol["variants"] or args.seed not in protocol["seeds"]:
        raise ValueError("Unregistered fit")
    cpus = args.cpus
    if len(cpus) != protocol["n_envs"] + 1:
        raise ValueError("Need distinct learner and environment cores")
    cap_cpu(cpus[0])
    args.output.mkdir(parents=True, exist_ok=False)
    callback = Journal(args.output)
    vector = SubprocVecEnv(
        [
            partial(make_training_worker, protocol, args.seed, rank, cpus[rank + 1])
            for rank in range(protocol["n_envs"])
        ],
        start_method="spawn",
    )
    succeeded = False
    try:
        artifact = protocol["initializers"][str(args.seed)]
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Initializer checkpoint changed")
        model = GuidedSAC.load(
            artifact["path"],
            env=vector,
            device=args.device,
            **copy.deepcopy(protocol["algorithm"]),
        )
        if model.c2g_context.get("feature_contract") != FEATURE_CONTRACT:
            raise ValueError("Wrong observation semantics in the initializer")
        if model.policy_kwargs.get("net_arch") != protocol["architecture"]:
            raise ValueError("Network architecture changed")
        if identity(protocol) != expected_identity:
            raise ValueError("Model construction mutated the registration")
        model.set_logger(configure(str(args.output / "metrics"), ["csv"]))
        train_x, train_y, _, _, corpus_info = corpus(
            Path(protocol["corpus_path"]), "history"
        )
        model.attach_demonstrations(train_x, train_y, args.seed)
        model.c2g_context = copy.deepcopy(
            {
                **model.c2g_context,
                "kind": "demonstration_initialized_standalone_SAC",
                "variant": args.variant,
                "registration_id": protocol["registration_id"],
                "feature_contract": FEATURE_CONTRACT,
                "initialization": artifact,
                "training_protocol": protocol,
            }
        )
        initial = {
            "actor": parameter_digest(model.actor),
            "critic": parameter_digest(model.critic),
            "critic_target": parameter_digest(model.critic_target),
            "initialization": artifact,
            "corpus": corpus_info,
        }
        model.learn(
            total_timesteps=protocol["learning_starts"],
            callback=callback,
            reset_num_timesteps=False,
        )
        if (
            model._n_updates
            or model.actor_updates
            or parameter_digest(model.actor) != initial["actor"]
        ):
            raise AssertionError("Collection-only warmup changed policy weights")
        initial["warmup_replay"] = replay_digest(model.replay_buffer)
        initial["warmup_transitions"] = model.num_timesteps
        write_json(args.output / "initialization.json", initial)
        calibration = model.calibrate_critics(protocol["critic_calibration_updates"])
        balance = model.calibrate_regularization(protocol["bc_initial_gradient_ratio"])
        model.bc_weight = balance["bc_weight"] if args.variant == "guided_sac" else 0.0
        write_json(
            args.output / "calibration.json",
            {**calibration, **balance, "applied_bc_weight": model.bc_weight},
        )
        milestones = (
            [protocol["learning_starts"] + 128]
            if args.smoke
            else protocol["milestones"]
        )
        records, selected = [], None
        for target in milestones:
            model.learn(
                total_timesteps=target - model.num_timesteps,
                callback=callback,
                reset_num_timesteps=False,
            )
            if model.num_timesteps != target:
                raise AssertionError("Training did not stop at the exact milestone")
            expected_actor = target - protocol["learning_starts"]
            if model.actor_updates != expected_actor or model._n_updates != (
                expected_actor + protocol["critic_calibration_updates"]
            ):
                raise AssertionError("SAC update/transition accounting mismatch")
            model.logger.dump(step=target)
            model.c2g_context.update(
                online_transitions=target,
                sac_actor_updates=model.actor_updates,
                critic_updates=model._n_updates,
                applied_bc_weight=model.bc_weight,
            )
            checkpoint = args.output / f"step_{target}.zip"
            model.save(checkpoint)
            model.save_replay_buffer(args.output / "latest_replay.pkl")
            policy = {
                "id": f"{args.variant}_{args.seed}_step_{target}",
                "hardware": "history_sac",
                "macro": "rule",
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": file_hash(checkpoint),
                },
            }
            evaluation_protocol = copy.deepcopy(protocol)
            evaluation_protocol.update(
                workers=protocol["n_envs"], evaluation_cpus=cpus[1:]
            )
            cells = protocol["development_cells"]
            if args.smoke:
                cells = [
                    {
                        "start_day": protocol["train_days"][0],
                        "seed": 710048,
                        "regd_load_bias": 0.0,
                    }
                ]
            result = run_panel(
                evaluation_protocol,
                [policy],
                cells,
                args.output / f"validation_{target}",
            )
            summary = (
                candidate_summary(result["episodes"], protocol["selection"])
                if not args.smoke
                else {"smoke_cases": len(result["episodes"])}
            )
            row = {"timesteps": target, "checkpoint": policy["checkpoint"], **summary}
            records.append(row)
            if row.get("eligible") and (
                selected is None
                or (row["biased_facility_relief_mw"], -row["rmse_kw"])
                > (selected["biased_facility_relief_mw"], -selected["rmse_kw"])
            ):
                selected = row
            write_json(args.output / "validation_summary.json", records)
            write_json(
                args.output / "selection.json",
                {
                    "selected": selected,
                    "rule": protocol["selection"],
                    "all_milestones_retained": True,
                },
            )
            print(args.variant, args.seed, row, flush=True)
        if identity(protocol) != expected_identity:
            raise ValueError("Training mutated the registration")
        verify_protocol(protocol)
        prefix = vector.env_method("training_prefix")
        write_json(args.output / "training_prefixes.json", prefix)
        write_json(
            args.output / "completed.json",
            {
                "status": "completed",
                "variant": args.variant,
                "seed": args.seed,
                "online_transitions": model.num_timesteps,
                "sac_actor_updates": model.actor_updates,
                "critic_updates": model._n_updates,
                "selected": selected,
                "smoke_only": args.smoke,
                "registration_unchanged": True,
                "feature_contract": FEATURE_CONTRACT,
                "code_hashes": protocol["code_hashes"],
            },
        )
        succeeded = True
    finally:
        vector.close()
        if not succeeded:
            write_json(
                args.output / "failure.json",
                {"status": "runtime_failure", "traceback": traceback.format_exc()},
            )


def main():
    def interrupted(signum, frame):
        raise InterruptedError(f"Learner interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
