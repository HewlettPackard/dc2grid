"""
baselines/safety/train_ppo_lagrangian.py  —  PPO-Lagrangian (Constrained RL)
======================================================================
Augments standard SB3 PPO with Lagrange multipliers for hard-constraint
satisfaction.  Three constraint costs are tracked:

  Cost 1 (thermal):   1 if max(T_A, T_B) > T_warn   per step
  Cost 2 (SOC):       1 if SOC ∉ [0.12, 0.90]       per step
  Cost 3 (frequency): 1 if |Δf| > 0.3 Hz            per step

The Lagrangian objective is:
    L = E[Σ γ^t r_t] - Σ_j λ_j (E[Σ c_j,t] - d_j)

where d_j is the cost budget (max allowable fraction of violating steps)
and λ_j ≥ 0 are dual variables updated via gradient ascent:
    λ_j ← max(0, λ_j + lr_λ · (mean_cost_j - d_j))

Implementation wraps a standard SB3 PPO with a custom callback that
adjusts the reward at each step:  r' = r - Σ λ_j · c_j.

Usage
-----
  uv run python baselines/safety/train_ppo_lagrangian.py algo=ppo_lagrangian
  uv run python baselines/safety/train_ppo_lagrangian.py algo=ppo_lagrangian scenario=scenario_b
"""
from __future__ import annotations

import logging
from pathlib import Path

import baselines._hydra_compat  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from c2g_env.thermal_limits import T_SAFE as _T_SAFE, T_WARN as _T_WARN
from baselines.metrics_callback import C2GMetricsCallback

log = logging.getLogger(__name__)

# ── Observation indices ────────────────────────────────
_I_TEMP_A = 0
_I_TEMP_B = 1
_I_SOC    = 2
_I_FREQ   = 14

_T_WARN_NORM = _T_WARN / _T_SAFE
_SOC_SAFE_LO = 0.12
_SOC_SAFE_HI = 0.90
_FREQ_THRESH = 0.6   # normalised: 0.3 Hz / 0.5 Hz


class LagrangianRewardWrapper(gym.Wrapper):
    """
    Modifies the step reward by subtracting Lagrange-weighted constraint costs.

    r' = r - λ_thermal · c_thermal - λ_soc · c_soc - λ_freq · c_freq
    """

    def __init__(self, env: gym.Env, lambdas: np.ndarray) -> None:
        super().__init__(env)
        self.lambdas = lambdas  # shared mutable reference
        self._episode_costs = np.zeros(3)
        self._episode_steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute constraint violations
        c_thermal = float(max(obs[_I_TEMP_A], obs[_I_TEMP_B]) > _T_WARN_NORM)
        c_soc     = float(obs[_I_SOC] < _SOC_SAFE_LO or obs[_I_SOC] > _SOC_SAFE_HI)
        c_freq    = float(abs(obs[_I_FREQ]) > _FREQ_THRESH)

        costs = np.array([c_thermal, c_soc, c_freq])
        self._episode_costs += costs
        self._episode_steps += 1

        # Lagrangian penalty
        penalty = float(self.lambdas @ costs)
        modified_reward = reward - penalty

        info["constraint_costs"] = costs
        info["lagrangian_penalty"] = penalty

        if terminated or truncated:
            info["episode_cost_rates"] = self._episode_costs / max(1, self._episode_steps)
            self._episode_costs = np.zeros(3)
            self._episode_steps = 0

        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._episode_costs = np.zeros(3)
        self._episode_steps = 0
        return self.env.reset(**kwargs)


class LagrangianUpdateCallback(BaseCallback):
    """
    Updates Lagrange multipliers after each rollout based on mean constraint
    cost rates.
    """

    def __init__(
        self,
        lambdas: np.ndarray,
        cost_budgets: np.ndarray,
        lr_lambda: float = 0.01,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.lambdas = lambdas
        self.cost_budgets = cost_budgets
        self.lr_lambda = lr_lambda
        self._cost_buffer: list[np.ndarray] = []

    def _on_step(self) -> bool:
        # Collect cost rates from info dicts
        for info in self.locals.get("infos", []):
            if "episode_cost_rates" in info:
                self._cost_buffer.append(info["episode_cost_rates"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._cost_buffer:
            return
        mean_costs = np.mean(self._cost_buffer, axis=0)
        self._cost_buffer.clear()

        # Dual gradient ascent
        for j in range(len(self.lambdas)):
            self.lambdas[j] = max(
                0.0,
                self.lambdas[j] + self.lr_lambda * (mean_costs[j] - self.cost_budgets[j]),
            )

        if self.verbose:
            log.info(
                f"Lagrangian update: λ={self.lambdas}, "
                f"mean_costs={mean_costs}, budgets={self.cost_budgets}"
            )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    scenario = cfg.scenario.name
    algo_cfg = cfg.algo
    seed     = cfg.experiment.seed
    out_dir  = Path(HydraConfig.get().runtime.output_dir)

    log.info(f"PPO-Lagrangian: scenario={scenario}, seed={seed}")

    # Shared mutable Lagrange multipliers (3 constraints)
    lambdas = np.array([
        float(getattr(algo_cfg, "lambda_thermal_init", 0.1)),
        float(getattr(algo_cfg, "lambda_soc_init", 0.1)),
        float(getattr(algo_cfg, "lambda_freq_init", 0.1)),
    ])

    # Cost budgets (max allowable fraction of violating steps)
    cost_budgets = np.array([
        float(getattr(algo_cfg, "budget_thermal", 0.05)),
        float(getattr(algo_cfg, "budget_soc", 0.10)),
        float(getattr(algo_cfg, "budget_freq", 0.05)),
    ])

    lr_lambda = float(getattr(algo_cfg, "lr_lambda", 0.01))

    # Create wrapped environments
    def make_env():
        env = C2GFastEnv(scenario=scenario)
        return LagrangianRewardWrapper(env, lambdas)

    n_envs = int(getattr(algo_cfg, "n_envs", 4))
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Eval env (standard, no Lagrangian penalty)
    eval_env = make_vec_env(lambda: C2GFastEnv(scenario=scenario), n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    timesteps = int(getattr(algo_cfg, "timesteps", 300_000))

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(getattr(algo_cfg, "learning_rate", 3e-4)),
        n_steps=int(getattr(algo_cfg, "n_steps", 512)),
        batch_size=int(getattr(algo_cfg, "batch_size", 128)),
        n_epochs=int(getattr(algo_cfg, "n_epochs", 10)),
        gamma=float(getattr(algo_cfg, "gamma", 0.99)),
        gae_lambda=float(getattr(algo_cfg, "gae_lambda", 0.95)),
        clip_range=float(getattr(algo_cfg, "clip_range", 0.2)),
        ent_coef=float(getattr(algo_cfg, "ent_coef", 0.005)),
        vf_coef=float(getattr(algo_cfg, "vf_coef", 0.5)),
        max_grad_norm=float(getattr(algo_cfg, "max_grad_norm", 0.5)),
        seed=seed,
        verbose=1,
    )

    callbacks = [
        LagrangianUpdateCallback(lambdas, cost_budgets, lr_lambda, verbose=1),
        CheckpointCallback(save_freq=10_000, save_path=str(out_dir / "checkpoints")),
        EvalCallback(
            eval_env,
            eval_freq=int(getattr(algo_cfg, "eval_freq", 10_000)),
            n_eval_episodes=int(getattr(algo_cfg, "n_eval_episodes", 5)),
            best_model_save_path=str(out_dir / "best_model"),
            deterministic=True,
        ),
        C2GMetricsCallback(csv_path=out_dir / "metrics.csv"),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)

    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    log.info(f"PPO-Lagrangian training complete. Final λ={lambdas}")


if __name__ == "__main__":
    train()
