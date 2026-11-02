"""Frozen neural policies and prespecified scripts in the same fraction coordinates."""

from __future__ import annotations

import math

# isort: off
from confirmation_context import file_hash
from fraction_macro import CreditPotentialEnv, FractionalDispatchTask
from train_macro import MacroLatentEnv
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
# isort: on

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
MARGINS = (None, 0.0, 5.0, 10.0, 15.0, 20.0)


def scripted_family() -> list[dict]:
    policies = [
        {
            "id": "script_zero",
            "family": "script_zero",
            "kind": "scripted",
            "mode": "constant",
            "fraction": 0.0,
            "training_seed": None,
        }
    ]
    for fraction in FRACTIONS:
        for margin in MARGINS:
            label = "always" if margin is None else f"margin_{int(margin):02d}"
            name = f"script_f{round(fraction * 100):03d}_{label}"
            policies.append(
                {
                    "id": name,
                    "family": name,
                    "kind": "scripted",
                    "mode": "constant" if margin is None else "margin",
                    "fraction": fraction,
                    "threshold_usd_mwh": margin,
                    "training_seed": None,
                }
            )
    policies.append(
        {
            "id": "script_deadline_margin10",
            "family": "script_deadline_margin10",
            "kind": "scripted",
            "mode": "deadline_margin",
            "fraction": 1.0,
            "threshold_usd_mwh": 10.0,
            "training_seed": None,
        }
    )
    policies.append(
        {
            "id": "script_credit_one",
            "family": "script_credit_one",
            "kind": "scripted",
            "mode": "credit",
            "fraction": 1.0,
            "threshold_credit_mwh": 1.0,
            "training_seed": None,
        }
    )
    if len(policies) != 27 or len({row["id"] for row in policies}) != 27:
        raise AssertionError("The complete fixed scripted family changed")
    return policies


def scripted_action(policy: dict, observation) -> np.ndarray:
    observation = np.asarray(observation)
    if observation.shape != (23,) or not np.isfinite(observation).all():
        raise ValueError("Scripted controls require the same finite 23D observation")
    fraction = policy["fraction"]
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid registered fraction")
    mode = policy["mode"]
    if mode == "constant":
        action = fraction
    elif mode == "credit":
        threshold = policy["threshold_credit_mwh"]
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid registered credit threshold")
        action = fraction if float(observation[21]) * 30 >= threshold else 0.0
    elif mode in ("margin", "deadline_margin"):
        threshold = policy["threshold_usd_mwh"]
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid registered price threshold")
        if mode == "deadline_margin":
            remaining = float(observation[22])
            if not 0 <= remaining <= 1:
                raise ValueError("The causal remaining-time feature is invalid")
            threshold *= remaining
        margin = float(observation[20]) * 200 - float(observation[17]) * 100
        action = fraction if margin >= threshold else 0.0
    else:
        raise ValueError("Unknown registered scripted policy")
    return np.array([action], dtype=np.float32)


def load_model(policy: dict):
    artifact = policy["checkpoint"]
    if file_hash(artifact["path"]) != artifact["sha256"]:
        raise ValueError("Frozen model weights changed")
    algorithm = policy["algorithm"]
    if algorithm not in ("ppo", "sac"):
        raise ValueError("Unknown frozen algorithm")
    model = (PPO if algorithm == "ppo" else SAC).load(artifact["path"], device="cpu")
    context = getattr(model, "c2g_context", None)
    bound = 20.0 if algorithm == "ppo" else 1.0
    if (
        context is None
        or context["registration_id"] != policy["training_registration_id"]
        or context["variant"] != policy["variant"]
        or model.gamma != 1.0
        or model.observation_space.shape != (23,)
        or model.action_space.shape != (1,)
        or not np.all(model.action_space.low == -bound)
        or not np.all(model.action_space.high == bound)
    ):
        raise ValueError("Frozen model is incompatible with the registered task")
    model.policy.set_training_mode(False)
    return model


def make_environment(task_protocol, cell, cells, policy, *, record_trace=True):
    task = FractionalDispatchTask(
        task_protocol,
        seed=0,
        fixed_cell=cell,
        record_trace=record_trace,
        evaluation_mode=True,
        evaluation_cells=cells,
    )
    environment = MacroLatentEnv(task) if policy.get("algorithm") == "ppo" else task
    current = environment
    while isinstance(current, gym.Wrapper):
        if isinstance(current, CreditPotentialEnv):
            raise AssertionError(
                "Training potential must not be active during evaluation"
            )
        current = current.env
    if current is not task:
        raise AssertionError("Unexpected confirmation wrapper stack")
    return task, environment
