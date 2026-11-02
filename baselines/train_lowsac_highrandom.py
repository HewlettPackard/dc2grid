"""
baselines/train_lowsac_highrandom.py  —  SAC + Random High-Level Controller
=============================================================================
Trains SAC as a low-level controller (4-D: throttle, pump, hvac, bess)
while a random high-level controller periodically changes committed_mw,
simulating varying grid regulation commitments.

Every 180 sub-steps (= one 15-min macro tick), a new committed_mw is
sampled uniformly from [0, committed_mw_max].  This exposes the SAC
agent to a range of tracking demands during training, making it robust
to different high-level decisions.

Usage
-----
  python baselines/train_lowsac_highrandom.py algo=sac
  python baselines/train_lowsac_highrandom.py algo=sac scenario=scenario_b
  python baselines/train_lowsac_highrandom.py algo=sac --multirun \\
      scenario=default,scenario_a,scenario_b,scenario_c \\
      experiment.seed=1,2,3
"""
from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401

import gymnasium as gym
import numpy as np
import yaml

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from c2g_env import C2GFastEnv
from baselines.metrics_callback import C2GMetricsCallback

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "c2g_env" / "config.yaml"
_MACRO_SUBSTEPS = 180  # 180 × 5s = 15 min


class RandomHighLevelWrapper(gym.Wrapper):
    """Wraps C2GFastEnv with a random high-level controller.

    Every ``macro_period`` sub-steps, samples a new ``committed_mw``
    uniformly from [0, committed_mw_max], simulating random macro bids.
    The observation and action spaces are unchanged (low-level SAC).
    """

    def __init__(self, env: C2GFastEnv, macro_period: int = _MACRO_SUBSTEPS):
        super().__init__(env)
        self._macro_period = macro_period
        self._sub_tick = 0

        # Read committed_mw_max from the env's scenario config
        cfg_path = _CONFIG_PATH
        with open(cfg_path, encoding="utf-8") as fh:
            full_cfg = yaml.safe_load(fh)
        scenario = env._scenario
        self._committed_mw_max = float(full_cfg[scenario]["committed_mw_max"])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sub_tick = 0
        # Sample initial committed_mw
        self.env.committed_mw = self.np_random.uniform(0.0, self._committed_mw_max)
        return obs, info

    def step(self, action):
        # Randomise committed_mw every macro period
        if self._sub_tick > 0 and self._sub_tick % self._macro_period == 0:
            self.env.committed_mw = self.np_random.uniform(0.0, self._committed_mw_max)
        self._sub_tick += 1
        return self.env.step(action)


def make_env_fn(scenario: str, seed: int, wrap: bool = True):
    def _init():
        env = C2GFastEnv(scenario=scenario)
        if wrap:
            env = RandomHighLevelWrapper(env)
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

    print(f"[SAC+RandomHigh] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}")

    # ── Environments ──────────────────────────────────────────────────────
    env = make_env_fn(scenario, seed, wrap=True)()

    eval_env = make_vec_env(make_env_fn(scenario, seed + 999, wrap=True),
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
        device                  = "cuda",  # uses GPU if available, else CPU
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = algo_cfg.timesteps,
        callback            = [checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name         = cfg.experiment.name,
        reset_num_timesteps = True,
    )

    model.save(str(out_dir / "final_model"))
    print(f"\n[SAC+RandomHigh] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
