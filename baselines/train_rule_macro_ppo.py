"""
baselines/train_rule_macro_ppo.py  —  PPO Low-Level + Rule-Based Macro
=======================================================================
Trains a PPO low-level controller (4-D: throttle, pump, hvac, bess) on
``C2GFastEnv`` while a frozen ``RuleBasedMacroController`` drives the
15-minute market bidding decisions that determine ``committed_mw``.

This gives the low-level agent a realistic training distribution: the
grid commitment varies over time (as the macro controller bids), so the
agent learns to track *dynamic* regulation signals — not just the fixed
``dr_baseline_mw`` default.

Architecture
------------
  ┌──────────────────────────────────────────────┐
  │  RuleMacroWrappedEnv  (gymnasium.Wrapper)    │
  │                                              │
  │  every 180 steps (15 min):                   │
  │    ① aggregate sub-obs → macro obs (19-D)    │
  │    ② RuleBasedMacroController.predict(obs)   │
  │    ③ grid.step_rmcp() → clear_bid()          │
  │    ④ env.committed_mw = cleared MW           │
  │                                              │
  │  every step (5 s):                           │
  │    PPO action (4-D) → C2GFastEnv.step()      │
  └──────────────────────────────────────────────┘

Usage
-----
  # Single run with defaults
  python baselines/train_rule_macro_ppo.py

  # Override scenario
  python baselines/train_rule_macro_ppo.py scenario=scenario_a

  # Sweep
  python baselines/train_rule_macro_ppo.py --multirun \\
      scenario=default,scenario_a,scenario_b,scenario_c \\
      experiment.seed=1,2,3

Outputs (Hydra-managed)
-----------------------
  outputs/<algo>_<scenario>/seed_<N>/<timestamp>/
      .hydra/               — config snapshot
      episode_metrics.csv
      checkpoints/
      best_model/
      tensorboard/
"""
from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401  # Hydra 1.3.x + Python ≥3.14 fix

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from c2g_env.train_envs import RuleMacroWrappedEnv, FixedActionWrapper
from baselines.metrics_callback import (
    C2GMetricsCallback, C2GEvalMetricsWrapper, SyncNormEvalCallback,
)
from baselines.resume import maybe_load_vecnormalize, maybe_load_model


# ======================================================================
# Env factory
# ======================================================================

def make_env_fn(scenario: str, seed: int,
                fixed_actions: dict[int, float] | None = None):
    def _init():
        fast_env = C2GFastEnv(scenario=scenario)
        env = RuleMacroWrappedEnv(fast_env)
        if fixed_actions:
            env = FixedActionWrapper(env, fixed_actions=fixed_actions)
        env.reset(seed=seed)
        return env
    return _init


# ======================================================================
# Hydra entry-point
# ======================================================================

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario = cfg.scenario.env_id
    seed = cfg.experiment.seed
    algo_cfg = cfg.algo
    log_cfg = cfg.logging

    print(f"[RuleMacro+PPO] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}  n_envs={algo_cfg.n_envs}")

    fixed_actions: dict[int, float] | None = None
    raw_fixed = algo_cfg.get("fixed_actions", None)
    if raw_fixed is not None:
        fixed_actions = {int(k): float(v) for k, v in raw_fixed.items()}
        print(f"[RuleMacro+PPO] fixed_actions={fixed_actions}")

    # ── Environments ──────────────────────────────────────────────────
    raw_vec_env = make_vec_env(make_env_fn(scenario, seed, fixed_actions),
                               n_envs=algo_cfg.n_envs, seed=seed)
    vec_env, resume_from = maybe_load_vecnormalize(
        raw_vec_env, algo_cfg, label="RuleMacro+PPO")

    _eval_metric_wrappers = []

    def _make_eval_env():
        fast_env = C2GFastEnv(scenario=scenario)
        env = RuleMacroWrappedEnv(fast_env)
        if fixed_actions:
            env = FixedActionWrapper(env, fixed_actions=fixed_actions)
        wrapper = C2GEvalMetricsWrapper(env)
        _eval_metric_wrappers.append(wrapper)
        wrapper.reset(seed=seed)
        return wrapper

    eval_env = make_vec_env(_make_eval_env, n_envs=1, seed=seed)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=algo_cfg.clip_obs, training=False)

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(algo_cfg.eval_freq, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ckpt",
    )
    eval_cb = SyncNormEvalCallback(
        eval_env,
        eval_metrics_wrappers=_eval_metric_wrappers,
        csv_path=out_dir / "eval_metrics.csv" if log_cfg.csv else None,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "tensorboard"),
        eval_freq=algo_cfg.eval_freq,
        n_eval_episodes=algo_cfg.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )
    metrics_cb = C2GMetricsCallback(
        print_freq=log_cfg.console_freq,
        csv_path=out_dir / "episode_metrics.csv" if log_cfg.csv else None,
        verbose=1,
    )

    # ── Model ─────────────────────────────────────────────────────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)
    tb_log = str(out_dir / "tensorboard") if log_cfg.tensorboard else None

    model, reset_timesteps = maybe_load_model(
        PPO, resume_from, vec_env,
        device=cfg.device,
        tensorboard_log=tb_log,
        label="RuleMacro+PPO",
        # fresh-model kwargs:
        policy="MlpPolicy",
        learning_rate=algo_cfg.learning_rate,
        n_steps=algo_cfg.n_steps,
        batch_size=algo_cfg.batch_size,
        n_epochs=algo_cfg.n_epochs,
        gamma=algo_cfg.gamma,
        gae_lambda=algo_cfg.gae_lambda,
        clip_range=algo_cfg.clip_range,
        ent_coef=algo_cfg.ent_coef,
        vf_coef=algo_cfg.vf_coef,
        max_grad_norm=algo_cfg.max_grad_norm,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0,
        seed=seed,
    )

    # ── Train ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=algo_cfg.timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name=cfg.experiment.name,
        reset_num_timesteps=reset_timesteps,
    )

    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\n[RuleMacro+PPO] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
