"""
baselines/train_ppo.py  —  PPO Training Script (SB3 + Hydra)
=============================================================
All hyperparameters are declared in conf/algo/ppo.yaml and
conf/scenario/*.yaml.  Hydra handles output-dir creation,
config snapshotting, and multi-run sweeps automatically.

Usage
-----
  # Single run with defaults (default scenario, ppo algo)
  python baselines/train_ppo.py

  # Override scenario
  python baselines/train_ppo.py scenario=scenario_a

  # Override multiple values inline
  python baselines/train_ppo.py scenario=scenario_b algo.timesteps=500000 experiment.seed=7

  # Grid sweep over all scenarios × 3 seeds
  python baselines/train_ppo.py --multirun \\
      scenario=default,scenario_a,scenario_b,scenario_c \\
      experiment.seed=1,2,3

Outputs (managed by Hydra)
--------------------------
  outputs/<algo>_<scenario>/seed_<N>/<timestamp>/
      .hydra/           — config snapshot (config.yaml, overrides.yaml)
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from c2g_env import C2GFastEnv
from baselines.metrics_callback import C2GMetricsCallback


class SyncNormEvalCallback(EvalCallback):
    """
    EvalCallback that syncs VecNormalize obs/reward running statistics
    from the training environment to the eval environment before each
    evaluation round.

    Without this sync, eval_env has cold normalization stats (mean=0, var=1)
    while the agent was trained on obs scaled by the training env's accumulated
    stats — making eval rewards incomparable to training rewards.
    """

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)
        return super()._on_step()


def make_env_fn(scenario: str, seed: int):
    def _init():
        env = C2GFastEnv(scenario=scenario)
        env.reset(seed=seed)
        return env
    return _init


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario  = cfg.scenario.env_id
    seed      = cfg.experiment.seed
    algo_cfg  = cfg.algo
    log_cfg   = cfg.logging

    print(f"[PPO] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}  n_envs={algo_cfg.n_envs}")

    # ── Environments ──────────────────────────────────────────────────────
    vec_env = make_vec_env(make_env_fn(scenario, seed),
                           n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = algo_cfg.norm_obs,
        norm_reward = algo_cfg.norm_reward,
        clip_obs    = algo_cfg.clip_obs,
        clip_reward = algo_cfg.clip_reward,
    )

    eval_env = make_vec_env(make_env_fn(scenario, seed + 999),
                            n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=algo_cfg.clip_obs, training=False)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(algo_cfg.eval_freq, 1),
        save_path   = str(out_dir / "checkpoints"),
        name_prefix = "ckpt",
    )
    eval_cb = SyncNormEvalCallback(
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
    print(f"\n[PPO] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
