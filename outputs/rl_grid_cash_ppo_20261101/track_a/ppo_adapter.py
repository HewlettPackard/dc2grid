"""Stock PPO in Gaussian coordinates; apply tanh only at the actuator boundary."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# isort: off
import economy_context  # noqa: E402,F401
from fit_bc import parameter_digest
from history_policy import (
    ACTION_HIGH,
    ACTION_LOW,
    FEATURE_CONTRACT,
    OwnHistory,
    hardware_action,
)
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.distributions import DiagGaussianDistribution
# isort: on

LATENT_LIMIT = 20.0
ACTION_CONTRACT = {
    "id": "ppo-gaussian-latent-tanh-physical-v1",
    "dimension": 3,
    "latent_low": [-20.0, -20.0, -20.0],
    "latent_high": [20.0, 20.0, 20.0],
    "physical_low": ACTION_LOW.tolist(),
    "physical_high": ACTION_HIGH.tolist(),
    "rollout_actions": "original Gaussian samples, before clipping or tanh",
    "training_transform": "outside StandaloneTask, before physical action history",
    "deployment_transform": "HistoryPPOController only",
    "entropy_coefficient": 0.0,
}


def native_from_latent(action) -> np.ndarray:
    latent = np.asarray(action, dtype=np.float32)
    if latent.ndim not in (1, 2) or latent.shape[-1] != 3:
        raise ValueError("Expected one or a batch of three latent actuator coordinates")
    if not np.isfinite(latent).all():
        raise ValueError("Latent actuator coordinates must be finite")
    bounded = np.clip(latent, -LATENT_LIMIT, LATENT_LIMIT)
    scaled = torch.tanh(torch.from_numpy(bounded)).numpy()
    return ACTION_LOW + 0.5 * (scaled + 1.0) * (ACTION_HIGH - ACTION_LOW)


class LatentActionEnv(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if (
            not isinstance(env.action_space, gym.spaces.Box)
            or not np.array_equal(env.action_space.low, ACTION_LOW)
            or not np.array_equal(env.action_space.high, ACTION_HIGH)
            or env.observation_space.shape != (24,)
        ):
            raise ValueError(
                "PPO wrapper requires the existing 24D physical-action task"
            )
        self.action_space = gym.spaces.Box(
            -LATENT_LIMIT,
            LATENT_LIMIT,
            shape=(3,),
            dtype=np.float32,
        )

    def action(self, action):
        latent = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(latent):
            raise ValueError(
                "PPO environment action is outside the declared latent box"
            )
        return native_from_latent(latent)


def validate_ppo(model: PPO, *, require_context: bool = True) -> None:
    if (
        model.observation_space.shape != (24,)
        or model.action_space.shape != (3,)
        or not np.all(model.action_space.low == -LATENT_LIMIT)
        or not np.all(model.action_space.high == LATENT_LIMIT)
        or model.use_sde
        or model.policy.squash_output
        or type(model.policy.action_dist) is not DiagGaussianDistribution
        or model.ent_coef != 0
    ):
        raise ValueError(
            "PPO feature, distribution, entropy or action contract mismatch"
        )
    if require_context:
        context = getattr(model, "c2g_context", None)
        if (
            context is None
            or context.get("feature_contract") != FEATURE_CONTRACT
            or context.get("action_contract") != ACTION_CONTRACT
            or context.get("simulation_mode") != "physical_v1"
        ):
            raise ValueError(
                "PPO deployment metadata does not match its action transform"
            )


def initialize_from_bc(model: PPO, teacher: SAC) -> dict:
    validate_ppo(model, require_context=False)
    if model.num_timesteps or model._n_updates:
        raise ValueError("BC transfer is allowed only into an untrained PPO model")
    if (
        teacher.observation_space.shape != (24,)
        or not np.array_equal(teacher.action_space.low, ACTION_LOW)
        or not np.array_equal(teacher.action_space.high, ACTION_HIGH)
        or teacher.use_sde
        or not isinstance(teacher.actor.log_std, torch.nn.Linear)
    ):
        raise ValueError(
            "BC initializer has a different feature or action distribution"
        )
    context = getattr(teacher, "c2g_context", None)
    if (
        context is None
        or context.get("feature_contract") != FEATURE_CONTRACT
        or context.get("simulation_mode") != "physical_v1"
    ):
        raise ValueError("BC initializer lacks the frozen physical feature contract")
    log_std = teacher.actor.log_std
    if (
        torch.count_nonzero(log_std.weight).item()
        or torch.any(log_std.bias < -20)
        or torch.any(log_std.bias > 2)
    ):
        raise ValueError("Exact transfer requires the initializer's constant log_std")
    with torch.no_grad():
        model.policy.mlp_extractor.policy_net.load_state_dict(
            teacher.actor.latent_pi.state_dict(),
            strict=True,
        )
        model.policy.action_net.load_state_dict(
            teacher.actor.mu.state_dict(), strict=True
        )
        model.policy.log_std.copy_(log_std.bias)
    model.c2g_context = {
        "feature_contract": copy.deepcopy(FEATURE_CONTRACT),
        "action_contract": copy.deepcopy(ACTION_CONTRACT),
        "simulation_mode": "physical_v1",
        "full_work_fixed": True,
        "analytic_controller_or_projection_at_deployment": False,
    }
    validate_ppo(model)
    return {
        "actor_body_sha256": parameter_digest(model.policy.mlp_extractor.policy_net),
        "actor_mean_sha256": parameter_digest(model.policy.action_net),
        "teacher_body_sha256": parameter_digest(teacher.actor.latent_pi),
        "teacher_mean_sha256": parameter_digest(teacher.actor.mu),
        "initial_log_std": model.policy.log_std.detach().cpu().tolist(),
        "physical_distribution_matches_only_at_initialization": True,
        "ppo_critic_not_copied_from_sac_q_functions": True,
    }


class HistoryPPOController:
    def __init__(self, model: PPO):
        validate_ppo(model)
        self.model = model
        self.observer = OwnHistory()

    def reset(self):
        self.observer.reset()

    def predict(self, observation, deterministic=True):
        features = self.observer.features(observation)
        latent = self.model.predict(features, deterministic=deterministic)[0]
        native = hardware_action(native_from_latent(latent))
        self.observer.last_action = native.copy()
        return native, None
