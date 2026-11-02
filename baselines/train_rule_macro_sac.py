"""
baselines/train_rule_macro_sac.py  —  SAC Low-Level + Rule-Based Macro
=======================================================================
Trains a SAC low-level controller (4-D: throttle, pump, hvac, bess) on
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
  │    SAC action (4-D) → C2GFastEnv.step()      │
  └──────────────────────────────────────────────┘

Usage
-----
  # Single run with defaults
  python baselines/train_rule_macro_sac.py algo=sac

  # Override scenario
  python baselines/train_rule_macro_sac.py algo=sac scenario=scenario_a

  # Sweep
  python baselines/train_rule_macro_sac.py algo=sac --multirun \\
      scenario=default,scenario_a,scenario_b,scenario_c \\
      experiment.seed=1,2,3

  # Reward-weight sensitivity (conf/reward/*.yaml)
  python baselines/train_rule_macro_sac.py algo=sac --multirun \\
      reward=beta8,beta8_alpha05 experiment.seed=1,2,3

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
from baselines._hydra_compat import plant_overrides_from_cfg, reward_overrides_from_cfg

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from c2g_env import C2GFastEnv
from c2g_env.train_envs import RuleMacroWrappedEnv
from baselines.metrics_callback import C2GMetricsCallback


# ======================================================================
# Env factory
# ======================================================================

def make_env_fn(scenario: str, seed: int, plant_overrides: dict | None = None,
                reward_overrides: dict | None = None):
    def _init():
        fast_env = C2GFastEnv(scenario=scenario, plant_overrides=plant_overrides,
                              reward_overrides=reward_overrides)
        env = RuleMacroWrappedEnv(fast_env)
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
    seed     = cfg.experiment.seed
    algo_cfg = cfg.algo
    log_cfg  = cfg.logging
    plant_overrides = plant_overrides_from_cfg(cfg)
    reward_overrides = reward_overrides_from_cfg(cfg)

    print(f"[RuleMacro+SAC] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}  reward_overrides={reward_overrides}")

    # ── Environments ──────────────────────────────────────────────────
    # SAC is off-policy: single env is fine; no VecNormalize needed
    env = make_env_fn(scenario, seed, plant_overrides, reward_overrides)()

    eval_env = make_vec_env(make_env_fn(scenario, seed + 999, plant_overrides, reward_overrides),
                            n_envs=1, seed=seed + 999)

    # ── Callbacks ─────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)

    model = SAC(
        policy                  = "MlpPolicy",
        env                     = env,
        learning_rate           = algo_cfg.learning_rate,
        buffer_size             = algo_cfg.buffer_size,
        learning_starts         = algo_cfg.learning_starts,
        batch_size              = algo_cfg.batch_size,
        tau                     = algo_cfg.tau,
        gamma                   = algo_cfg.gamma,
        train_freq              = algo_cfg.train_freq,
        gradient_steps          = algo_cfg.gradient_steps,
        ent_coef                = algo_cfg.ent_coef,
        target_update_interval  = algo_cfg.target_update_interval,
        policy_kwargs           = dict(net_arch=net_arch),
        tensorboard_log         = str(out_dir / "tensorboard") if log_cfg.tensorboard else None,
        verbose                 = 0,
        seed                    = seed,
    )

    # ── Train ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = algo_cfg.timesteps,
        callback            = [checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name         = cfg.experiment.name,
        reset_num_timesteps = True,
    )

    model.save(str(out_dir / "final_model"))
    print(f"\n[RuleMacro+SAC] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
