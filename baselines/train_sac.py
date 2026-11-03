"""
baselines/train_sac.py  —  SAC Training Script (SB3 + Hydra)
=============================================================
Usage
-----
  python baselines/train_sac.py algo=sac
  python baselines/train_sac.py algo=sac scenario=scenario_a
  python baselines/train_sac.py algo=sac --multirun \\
      scenario=default,scenario_a,scenario_b,scenario_c \\
      experiment.seed=1,2,3

Outputs (managed by Hydra)
--------------------------
  outputs/<algo>_<scenario>/seed_<N>/<timestamp>/
      .hydra/           — config snapshot
      episode_metrics.csv
      checkpoints/
      best_model/
      tensorboard/
"""

from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401  # Hydra 1.3.x + Python ≥3.14 fix
from baselines._hydra_compat import plant_overrides_from_cfg, simulation_options_from_cfg

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from c2g_env import C2GFastEnv
from baselines.metrics_callback import C2GMetricsCallback
from baselines.sac_provenance import bind_training_context


def make_env_fn(scenario: str, seed: int, plant_overrides: dict | None = None, **simulation_options):
    def _init():
        env = C2GFastEnv(scenario=scenario, plant_overrides=plant_overrides, **simulation_options)
        env.reset(seed=seed)
        return env
    return _init


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario = cfg.scenario.env_id
    seed     = cfg.experiment.seed
    algo_cfg = cfg.algo
    log_cfg  = cfg.logging
    plant_overrides = plant_overrides_from_cfg(cfg)
    if algo_cfg.name != "sac":
        raise ValueError("This trainer requires algo=sac")

    print(f"[SAC] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}")

    # ── Environments ──────────────────────────────────────────────────────
    # SAC is off-policy: single env is fine; no VecNormalize needed
    env = make_env_fn(scenario, seed, plant_overrides, **simulation_options_from_cfg(cfg))()

    eval_env = make_vec_env(make_env_fn(scenario, seed + 999, plant_overrides,
                                       **simulation_options_from_cfg(cfg, validation=True)),
                            n_envs=1, seed=seed + 999)

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
        device                  = cfg.device,
    )
    bind_training_context(model, env, out_dir)

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = algo_cfg.timesteps,
        callback            = [checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name         = cfg.experiment.name,
        reset_num_timesteps = True,
    )

    model.save(str(out_dir / "final_model"))
    print(f"\n[SAC] Training complete -> {out_dir.resolve()}")

if __name__ == "__main__":
    train()
