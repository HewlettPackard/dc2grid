"""
baselines/train_hierarchical.py  —  Sequential Hierarchical RL Baseline
========================================================================
Two-phase training pipeline for the C2G-Bench hierarchical architecture:

  Phase 1  Train a low-level PPO agent on C2GFastEnv    (5-second control)
  Phase 2  Freeze the low-level policy and train a
           high-level PPO agent on C2GMacroEnv           (15-minute bidding)

This is the recommended starting-point for hierarchical RL research on
C2G-Bench.  Researchers may improve upon it by:
  - Joint fine-tuning (unfreeze low-level during Phase 2)
  - Option-Critic / HIRO / HAM architectures
  - Lagrangian safety layers around the low-level policy
  - Communication channels between levels

Usage
-----
  # Full two-phase pipeline
  python baselines/train_hierarchical.py

  # Skip Phase 1 (reuse an existing low-level model)
  python baselines/train_hierarchical.py \
      +hrl.skip_phase1=true \
      +hrl.low_level_path=outputs/ppo_default/seed_42/.../final_model

  # Override scenario or seed
  python baselines/train_hierarchical.py scenario=scenario_a experiment.seed=2

Outputs
-------
  outputs/hrl_<scenario>/seed_<N>/<timestamp>/
      phase1/          — low-level PPO artifacts
      phase2/          — macro-level PPO artifacts
      final_model.zip  — the macro-level policy (the one you deploy)
"""
from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401  # Hydra 1.3.x + Python ≥3.14 fix

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv, C2GMacroEnv
from baselines.metrics_callback import C2GMetricsCallback


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------

def _make_fast_env(scenario: str, seed: int):
    def _init():
        env = C2GFastEnv(scenario=scenario)
        env.reset(seed=seed)
        return env
    return _init


def train_low_level(scenario: str, seed: int, timesteps: int,
                    out_dir: Path, log_cfg) -> Path:
    """Train a PPO agent on C2GFastEnv and return the model path."""
    phase_dir = out_dir / "phase1"
    phase_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(_make_fast_env(scenario, seed),
                           n_envs=4, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    eval_env = make_vec_env(_make_fast_env(scenario, seed + 999),
                            n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=512, batch_size=128,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(phase_dir / "tensorboard"),
        verbose=0, seed=seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(phase_dir / "best_model"),
        log_path=str(phase_dir / "tensorboard"),
        eval_freq=10_000, n_eval_episodes=3,
        deterministic=True, verbose=0,
    )
    metrics_cb = C2GMetricsCallback(
        print_freq=getattr(log_cfg, "console_freq", 20),
        csv_path=phase_dir / "episode_metrics.csv",
        verbose=1,
    )

    print(f"[Phase 1] Training low-level PPO: {timesteps:,} steps")
    model.learn(total_timesteps=timesteps,
                callback=[eval_cb, metrics_cb],
                tb_log_name="phase1_low")

    model_path = phase_dir / "final_model"
    model.save(str(model_path))
    vec_env.save(str(phase_dir / "vec_normalize.pkl"))
    print(f"[Phase 1] Complete -> {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def _make_inner_fn(model_path: Path):
    """Load frozen low-level policy as inner_action_fn."""
    inner_model = PPO.load(str(model_path))

    def inner_fn(inner_obs: np.ndarray, macro_action: np.ndarray) -> np.ndarray:
        action, _ = inner_model.predict(inner_obs, deterministic=True)
        return action.astype(np.float32)

    return inner_fn


def _make_macro_env(scenario: str, seed: int, inner_fn):
    def _init():
        env = C2GMacroEnv(scenario=scenario, inner_action_fn=inner_fn)
        env.reset(seed=seed)
        return env
    return _init


def train_macro(scenario: str, seed: int, timesteps: int,
                inner_fn, out_dir: Path, log_cfg) -> Path:
    """Train a PPO agent on C2GMacroEnv with a frozen inner policy."""
    phase_dir = out_dir / "phase2"
    phase_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(_make_macro_env(scenario, seed, inner_fn),
                           n_envs=4, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    eval_env = make_vec_env(_make_macro_env(scenario, seed + 999, inner_fn),
                            n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=1e-4, n_steps=256, batch_size=64,
        n_epochs=10, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        tensorboard_log=str(phase_dir / "tensorboard"),
        verbose=0, seed=seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(phase_dir / "best_model"),
        log_path=str(phase_dir / "tensorboard"),
        eval_freq=5_000, n_eval_episodes=3,
        deterministic=True, verbose=0,
    )
    metrics_cb = C2GMetricsCallback(
        print_freq=getattr(log_cfg, "console_freq", 20),
        csv_path=phase_dir / "episode_metrics.csv",
        verbose=1,
    )

    print(f"[Phase 2] Training macro PPO (with frozen inner): {timesteps:,} steps")
    model.learn(total_timesteps=timesteps,
                callback=[eval_cb, metrics_cb],
                tb_log_name="phase2_macro")

    model_path = phase_dir / "final_model"
    model.save(str(model_path))
    vec_env.save(str(phase_dir / "vec_normalize.pkl"))
    print(f"[Phase 2] Complete -> {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir  = Path(HydraConfig.get().runtime.output_dir)
    scenario = cfg.scenario.env_id
    seed     = cfg.experiment.seed
    log_cfg  = cfg.logging

    # HRL-specific overrides (optional, via +hrl.xxx on CLI)
    hrl_cfg = cfg.get("hrl")
    if hrl_cfg is None:
        hrl = {}
    elif OmegaConf.is_config(hrl_cfg):
        hrl = OmegaConf.to_container(hrl_cfg, resolve=True)
    else:
        hrl = dict(hrl_cfg)
    skip_phase1    = hrl.get("skip_phase1", False)
    low_level_path = hrl.get("low_level_path", None)
    phase1_steps   = hrl.get("phase1_steps", 300_000)
    phase2_steps   = hrl.get("phase2_steps", 100_000)

    print("=" * 60)
    print("  C2G-Bench: Sequential Hierarchical RL Training")
    print(f"  scenario={scenario}  seed={seed}")
    print(f"  Phase 1: {'SKIP' if skip_phase1 else f'{phase1_steps:,} steps'}")
    print(f"  Phase 2: {phase2_steps:,} steps")
    print("=" * 60)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if skip_phase1 and low_level_path:
        model_path = Path(low_level_path)
        print(f"[Phase 1] Skipped - reusing {model_path}")
    else:
        model_path = train_low_level(scenario, seed, phase1_steps,
                                     out_dir, log_cfg)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    inner_fn = _make_inner_fn(model_path)
    macro_model_path = train_macro(scenario, seed, phase2_steps,
                                   inner_fn, out_dir, log_cfg)

    # Copy final macro model to top-level for easy discovery
    import shutil
    shutil.copy2(str(macro_model_path) + ".zip",
                 str(out_dir / "final_model.zip"))

    print(f"\n{'=' * 60}")
    print(f"  HRL training complete -> {out_dir.resolve()}")
    print(f"  Low-level model:  {model_path}")
    print(f"  Macro model:      {macro_model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
