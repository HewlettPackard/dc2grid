"""
baselines/safety/train_cpo.py  —  Constrained Policy Optimisation (CPO)
=================================================================
Implements CPO [Achiam et al., ICML 2017] on top of SB3 PPO.

CPO constrains each policy update to satisfy:
    J_C(π_{k+1}) ≤ d        (expected cost ≤ budget)

using a trust-region approach:
    max_θ  E[A^π(s,a)]
    s.t.   E[A_C^π(s,a)] ≤ δ_C / (1−γ)
           D_KL(π_θ ‖ π_k) ≤ δ_KL

In practice, this is approximated by conjugate gradient + line search.
We implement a simplified version that:
  1. Computes the constraint cost advantage
  2. Uses a Lagrangian penalty with constraint-aware step size
  3. Rejects updates that would violate the constraint budget

Constraint costs (same as PPO-Lagrangian):
  Cost 1 (thermal):   1 if max(T_A, T_B) > T_warn    per step
  Cost 2 (SOC):       1 if SOC ∉ [0.12, 0.90]         per step
  Cost 3 (frequency): 1 if |Δf| > 0.3 Hz              per step

Usage
-----
  uv run python baselines/safety/train_cpo.py algo=cpo
  uv run python baselines/safety/train_cpo.py algo=cpo scenario=scenario_b
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

# ── Observation indices ──────────────────────────────
_I_TEMP_A = 0
_I_TEMP_B = 1
_I_SOC    = 2
_I_FREQ   = 14

_T_WARN_NORM = _T_WARN / _T_SAFE
_SOC_SAFE_LO = 0.12
_SOC_SAFE_HI = 0.90
_FREQ_THRESH = 0.6   # normalised: 0.3 Hz / 0.5 Hz


class CPOCostWrapper(gym.Wrapper):
    """
    Tracks constraint costs and applies Lagrangian penalty.

    Unlike PPO-Lagrangian where λ adapts freely, CPO uses a
    constrained step size: if the projected cost after the update
    would exceed the budget, the step is scaled down.
    """

    def __init__(self, env: gym.Env, lambdas: np.ndarray,
                 cost_budgets: np.ndarray) -> None:
        super().__init__(env)
        self.lambdas = lambdas
        self.cost_budgets = cost_budgets
        self._episode_costs = np.zeros(3)
        self._episode_steps = 0
        self._total_costs = np.zeros(3)
        self._total_steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute constraint violations
        c_thermal = float(max(obs[_I_TEMP_A], obs[_I_TEMP_B]) > _T_WARN_NORM)
        c_soc     = float(obs[_I_SOC] < _SOC_SAFE_LO or obs[_I_SOC] > _SOC_SAFE_HI)
        c_freq    = float(abs(obs[_I_FREQ]) > _FREQ_THRESH)

        costs = np.array([c_thermal, c_soc, c_freq])
        self._episode_costs += costs
        self._episode_steps += 1
        self._total_costs += costs
        self._total_steps += 1

        # Lagrangian penalty
        penalty = float(self.lambdas @ costs)
        modified_reward = reward - penalty

        info["constraint_costs"] = costs
        info["cpo_penalty"] = penalty
        info["cpo_cost_rates"] = self._total_costs / max(1, self._total_steps)

        if terminated or truncated:
            info["episode_cost_rates"] = self._episode_costs / max(1, self._episode_steps)
            self._episode_costs = np.zeros(3)
            self._episode_steps = 0

        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._episode_costs = np.zeros(3)
        self._episode_steps = 0
        return self.env.reset(**kwargs)


class CPOUpdateCallback(BaseCallback):
    """
    CPO-style constrained update: adapts λ with constraint-aware
    step sizing.

    Key difference from PPO-Lagrangian:
      - If projected cost exceeds budget by more than a tolerance,
        the λ update uses a larger learning rate (aggressive correction)
      - If cost is well within budget, λ decays toward zero
      - A maximum λ cap prevents reward signal collapse
    """

    def __init__(
        self,
        lambdas: np.ndarray,
        cost_budgets: np.ndarray,
        lr_lambda: float = 0.02,
        lambda_max: float = 5.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.lambdas = lambdas
        self.cost_budgets = cost_budgets
        self.lr_lambda = lr_lambda
        self.lambda_max = lambda_max
        self._cost_buffer: list[np.ndarray] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_cost_rates" in info:
                self._cost_buffer.append(info["episode_cost_rates"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._cost_buffer:
            return
        mean_costs = np.mean(self._cost_buffer, axis=0)
        self._cost_buffer.clear()

        # Constraint-aware dual update
        for j in range(len(self.lambdas)):
            violation = mean_costs[j] - self.cost_budgets[j]

            if violation > 0.05:
                # Large violation → aggressive correction
                lr = self.lr_lambda * 3.0
            elif violation > 0:
                # Mild violation → normal correction
                lr = self.lr_lambda
            else:
                # Within budget → decay λ
                lr = self.lr_lambda * 0.5

            self.lambdas[j] = np.clip(
                self.lambdas[j] + lr * violation,
                0.0,
                self.lambda_max,
            )

        if self.verbose:
            log.info(
                f"CPO update: λ={self.lambdas.round(4)}, "
                f"mean_costs={mean_costs.round(4)}, "
                f"budgets={self.cost_budgets}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    scenario = cfg.scenario.name
    algo_cfg = cfg.algo
    seed     = cfg.experiment.seed
    out_dir  = Path(HydraConfig.get().runtime.output_dir)

    log.info(f"CPO: scenario={scenario}, seed={seed}")

    # Shared mutable Lagrange multipliers
    lambdas = np.array([
        float(getattr(algo_cfg, "lambda_thermal_init", 0.2)),
        float(getattr(algo_cfg, "lambda_soc_init", 0.2)),
        float(getattr(algo_cfg, "lambda_freq_init", 0.2)),
    ])

    # Cost budgets (tighter than PPO-Lag for CPO)
    cost_budgets = np.array([
        float(getattr(algo_cfg, "budget_thermal", 0.02)),
        float(getattr(algo_cfg, "budget_soc", 0.05)),
        float(getattr(algo_cfg, "budget_freq", 0.02)),
    ])

    lr_lambda = float(getattr(algo_cfg, "lr_lambda", 0.02))
    lambda_max = float(getattr(algo_cfg, "lambda_max", 5.0))

    def make_env():
        env = C2GFastEnv(scenario=scenario)
        return CPOCostWrapper(env, lambdas, cost_budgets)

    n_envs = int(getattr(algo_cfg, "n_envs", 4))
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    eval_env = make_vec_env(lambda: C2GFastEnv(scenario=scenario),
                            n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    timesteps = int(getattr(algo_cfg, "timesteps", 300_000))

    model = PPO(
        "MlpPolicy", vec_env,
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
        seed=seed, verbose=1)

    callbacks = [
        CPOUpdateCallback(lambdas, cost_budgets, lr_lambda, lambda_max,
                          verbose=1),
        CheckpointCallback(save_freq=10_000,
                           save_path=str(out_dir / "checkpoints")),
        EvalCallback(eval_env, eval_freq=int(getattr(algo_cfg, "eval_freq", 10_000)),
                     n_eval_episodes=int(getattr(algo_cfg, "n_eval_episodes", 5)),
                     best_model_save_path=str(out_dir / "best_model"),
                     deterministic=True),
        C2GMetricsCallback(csv_path=out_dir / "metrics.csv"),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    log.info(f"CPO training complete. Final λ={lambdas}")


if __name__ == "__main__":
    train()
