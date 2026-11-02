"""
baselines/train_ppo_macro.py  —  PPO on C2GMacroEnv (High-Level Agent)
=======================================================================
Trains a PPO agent on the 15-minute macro environment. The macro agent
decides MW commitment and BESS dispatch target; the inner FastEnv uses
fixed safe defaults (or a pre-trained low-level policy if specified).

This is the simplest hierarchical baseline: the outer agent learns
grid-level bidding while the inner control is static.

Usage
-----
  # Macro-only training (inner uses fixed defaults)
  python baselines/train_ppo_macro.py algo=ppo_macro

  # Hierarchical: plug in a pre-trained low-level PPO
  python baselines/train_ppo_macro.py algo=ppo_macro \
      algo.inner_policy_path=outputs/ppo_default/seed_42/.../final_model

  # Multi-run sweep
  python baselines/train_ppo_macro.py algo=ppo_macro --multirun \
      scenario=default,scenario_a experiment.seed=1,2,3
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

from c2g_env import C2GMacroEnv
from baselines.metrics_callback import C2GMetricsCallback


# ---------------------------------------------------------------------------
# Build inner_action_fn from a pre-trained low-level SB3 policy
# ---------------------------------------------------------------------------

def _make_inner_action_fn(model_path: str):
    """
    Load a trained SB3 model and return a callable compatible with
    ``C2GMacroEnv(inner_action_fn=fn)``.

    The wrapper maps ``(inner_obs, macro_action) → low_action``
    where ``low_action`` has shape (4,): [throttle, pump, hvac, bess].
    """
    from stable_baselines3 import PPO as _PPO, SAC as _SAC

    p = Path(model_path)
    # Try PPO first, fall back to SAC
    try:
        inner_model = _PPO.load(str(p))
    except Exception:
        inner_model = _SAC.load(str(p))

    def inner_fn(inner_obs: np.ndarray, macro_action: np.ndarray) -> np.ndarray:
        action, _ = inner_model.predict(inner_obs, deterministic=True)
        return action.astype(np.float32)

    return inner_fn


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_macro_env_fn(scenario: str, seed: int, inner_fn=None):
    def _init():
        env = C2GMacroEnv(scenario=scenario, inner_action_fn=inner_fn)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario  = cfg.scenario.env_id
    seed      = cfg.experiment.seed
    algo_cfg  = cfg.algo
    log_cfg   = cfg.logging

    # Optional: load pre-trained low-level policy for hierarchical RL
    inner_fn = None
    inner_path = algo_cfg.get("inner_policy_path", None)
    if inner_path is not None and str(inner_path) != "null":
        inner_fn = _make_inner_action_fn(str(inner_path))
        print(f"[HRL] Loaded inner policy from {inner_path}")
    else:
        print("[Macro] Using fixed inner defaults (throttle=1.0, pump=0.7, hvac=0.7)")

    print(f"[PPO-Macro] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}  n_envs={algo_cfg.n_envs}")

    # ── Environments ──────────────────────────────────────────────────────
    vec_env = make_vec_env(
        make_macro_env_fn(scenario, seed, inner_fn),
        n_envs=algo_cfg.n_envs, seed=seed,
    )
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = algo_cfg.norm_obs,
        norm_reward = algo_cfg.norm_reward,
        clip_obs    = algo_cfg.clip_obs,
        clip_reward = algo_cfg.clip_reward,
    )

    eval_env = make_vec_env(
        make_macro_env_fn(scenario, seed + 999, inner_fn),
        n_envs=1, seed=seed + 999,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=algo_cfg.clip_obs, training=False)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(algo_cfg.eval_freq, 1),
        save_path   = str(out_dir / "checkpoints"),
        name_prefix = "ckpt",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(out_dir / "best_model"),
        log_path             = str(out_dir / "tensorboard"),
        eval_freq            = algo_cfg.eval_freq,
        n_eval_episodes      = algo_cfg.n_eval_episodes,
        deterministic        = True,
        verbose              = 0,
    )
    metrics_cb = C2GMetricsCallback(
        print_freq = log_cfg.console_freq,
        csv_path   = out_dir / "episode_metrics.csv" if log_cfg.csv else None,
        verbose    = 1,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = algo_cfg.learning_rate,
        n_steps         = algo_cfg.n_steps,
        batch_size      = algo_cfg.batch_size,
        n_epochs        = algo_cfg.n_epochs,
        gamma           = algo_cfg.gamma,
        gae_lambda      = algo_cfg.gae_lambda,
        clip_range      = algo_cfg.clip_range,
        ent_coef        = algo_cfg.ent_coef,
        vf_coef         = algo_cfg.vf_coef,
        max_grad_norm   = algo_cfg.max_grad_norm,
        policy_kwargs   = dict(net_arch=net_arch),
        tensorboard_log = str(out_dir / "tensorboard") if log_cfg.tensorboard else None,
        verbose         = 0,
        seed            = seed,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = algo_cfg.timesteps,
        callback            = [checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name         = cfg.experiment.name,
        reset_num_timesteps = True,
    )

    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\n[PPO-Macro] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
