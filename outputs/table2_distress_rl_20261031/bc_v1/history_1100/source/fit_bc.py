"""Fit explicitly supervised warmstarts; these checkpoints are not RL results."""

from __future__ import annotations

import argparse
import math
import shutil
import time
import traceback
from pathlib import Path

from common import ROOT, cap_cpu, file_hash, read_json, write_json
from history_policy import ACTION_HIGH, ACTION_LOW, FEATURE_CONTRACT

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC

class SpaceOnlyEnv(gym.Env):
    def __init__(self, dimension: int):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(dimension,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("The offline initializer must not simulate transitions")

    def step(self, action):
        raise RuntimeError("The offline initializer must not simulate transitions")


def corpus(directory: Path, mode: str):
    summary = read_json(directory / "summary.json")
    if summary["runtime_failures"] or not all(
        row["complete"] for row in summary["results"]
    ):
        raise ValueError("An incomplete corpus cannot be silently filtered")
    days = sorted({row["condition"]["start_day"] for row in summary["results"]})
    validation_days = days[4::5]
    train_x, train_y, val_x, val_y = [], [], [], []
    for row in sorted(summary["results"], key=lambda item: item["artifact"]):
        path = Path(row["artifact"])
        if file_hash(path) != row["artifact_sha256"]:
            raise ValueError(f"Demonstration artifact changed: {path}")
        with np.load(path) as data:
            features = data["observer"] if mode == "observer" else data["history"]
            if mode == "original":
                features = features[:, :18]
            targets = data["targets"]
            if row["condition"]["start_day"] in validation_days:
                val_x.append(features.copy())
                val_y.append(targets.copy())
            else:
                train_x.append(features.copy())
                train_y.append(targets.copy())
    if not train_x or not val_x:
        raise ValueError("Need nonempty training and internal-validation date groups")
    return (
        np.concatenate(train_x), np.concatenate(train_y),
        np.concatenate(val_x), np.concatenate(val_y),
        {
            "corpus_sha256": file_hash(directory / "summary.json"),
            "internal_validation_days": validation_days,
            "internal_validation_is_training_support_not_the_external_panel": True,
            "physical_teacher_transitions": summary["physical_transitions"],
        },
    )


def parameter_digest(module):
    import hashlib

    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def fit(args):
    cap_cpu(args.cpu)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    code_hashes = {
        name: file_hash(ROOT / name)
        for name in ("fit_bc.py", "common.py", "history_policy.py", "controllers.py")
    }
    for name in code_hashes:
        shutil.copy2(ROOT / name, source / name)
    write_json(
        args.output / "manifest.json",
        {**vars(args), "code_hashes": code_hashes, "role": "supervised_pretraining"},
    )
    x, y, vx, vy, evidence = corpus(args.corpus, args.mode)
    model = SAC(
        "MlpPolicy", SpaceOnlyEnv(x.shape[1]),
        learning_rate=1e-4, buffer_size=100000, learning_starts=1024,
        batch_size=256, tau=0.005, gamma=0.99, ent_coef=1e-5,
        train_freq=16, gradient_steps=-1,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed, device=args.device, verbose=0,
    )
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(math.log(0.01))
    critic_before = parameter_digest(model.critic)
    target_before = parameter_digest(model.critic_target)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    tx = torch.as_tensor(x, device=model.device)
    ty = torch.as_tensor(model.policy.scale_action(y), device=model.device)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 617]))
    validation_indices = rng.choice(
        len(vx), size=min(65536, len(vx)), replace=False
    )
    validation_x = torch.as_tensor(vx[validation_indices], device=model.device)
    validation_y = torch.as_tensor(vy[validation_indices], device=model.device)
    action_low = torch.as_tensor(ACTION_LOW, device=model.device)
    action_span = torch.as_tensor(ACTION_HIGH - ACTION_LOW, device=model.device)
    validation, best = [], float("inf")
    started = time.monotonic()
    model.c2g_context = {
        "simulation_mode": "physical_v1",
        "feature_contract": FEATURE_CONTRACT
        if args.mode == "history" else {"id": args.mode, "dimension": x.shape[1]},
        "kind": "supervised_behavior_cloning_only",
        "seed": args.seed,
        "online_transitions": 0,
        "sac_updates": 0,
        "corpus": evidence,
        "code_hashes": code_hashes,
    }
    for update in range(1, args.updates + 1):
        index = torch.randint(len(tx), (args.batch_size,), device=model.device)
        prediction = model.actor(tx[index], deterministic=True)
        loss = F.mse_loss(prediction, ty[index])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 1000 == 0 or update == args.updates:
            with torch.no_grad():
                predictions = torch.cat(
                    [
                        model.actor(batch, deterministic=True)
                        for batch in validation_x.split(8192)
                    ]
                )
                native = action_low + (predictions + 1) * action_span / 2
                residual = native - validation_y
                mse = residual.square().mean(dim=0)
                proxy = (
                    50 * residual[:, 1] - 5 * residual[:, 2]
                ).square().mean().sqrt() * 1000
            row = {
                "supervised_updates": update,
                "training_loss": float(loss.detach().cpu()),
                "action_rmse": mse.sqrt().cpu().tolist(),
                "linearized_tracking_proxy_kw_not_a_rollout": float(proxy.cpu()),
            }
            validation.append(row)
            score = float(mse.sum().cpu())
            if score < best:
                best = score
                model.c2g_context["supervised_updates"] = update
                model.save(args.output / "best_bc.zip")
                write_json(args.output / "selection.json", row)
            print(args.seed, args.mode, row, flush=True)
    if (
        model.actor.optimizer.state or model.critic.optimizer.state
        or parameter_digest(model.critic) != critic_before
        or parameter_digest(model.critic_target) != target_before
        or model._n_updates != 0 or model.num_timesteps != 0
    ):
        raise AssertionError("Supervised pretraining changed SAC optimizer/value state")
    model.c2g_context["supervised_updates"] = args.updates
    model.save(args.output / "final_bc.zip")
    result = {
        "kind": "supervised_only_not_RL",
        "seed": args.seed,
        "mode": args.mode,
        "updates": args.updates,
        "train_rows": len(x),
        "internal_validation_rows": len(vx),
        "validation_subset_rows": len(validation_x),
        "sac_optimizer_states_empty": True,
        "critic_digest_unchanged": critic_before,
        "target_digest_unchanged": target_before,
        "elapsed_seconds": time.monotonic() - started,
        "validation": validation,
        "selected_checkpoint": {
            "path": str(args.output / "best_bc.zip"),
            "sha256": file_hash(args.output / "best_bc.zip"),
        },
        **evidence,
    }
    write_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["history", "original", "observer"], default="history")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--updates", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    try:
        fit(args)
    except Exception as error:
        if args.output.exists():
            write_json(
                args.output / "failure.json",
                {"error": repr(error), "traceback": traceback.format_exc()},
            )
        raise


if __name__ == "__main__":
    main()
