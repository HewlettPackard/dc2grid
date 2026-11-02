"""
baselines/train_hierarchical_sac.py  —  Frozen Hardware + SAC Macro
================================================================================
Two-phase hierarchical pipeline:

  Phase 1  Select a separately trained hardware SAC checkpoint
  Phase 2  Train SAC high-level agent on C2GMacroEnv (15-minute bidding)
           with that exact hardware policy frozen

The rule-based hardware controller is an explicitly selected alternative,
not an implicit replacement for the learned hardware policy.

Usage
-----
  # Staged SAC (hardware checkpoint must match the environment semantics)
  python -m baselines.train_hierarchical_sac algo=sac \\
      +hrl.inner_checkpoint=/path/to/hardware/final_model.zip

  # Explicit classical-inner baseline
  python -m baselines.train_hierarchical_sac algo=sac +hrl.inner_controller=rule_based

  # Add experiment.seed=N, +hrl.phase2_steps=N, or scenario=scenario_a as needed.

Outputs
-------
  outputs/hrl_sac_<scenario>/seed_<N>/<timestamp>/
      phase2/              — SAC macro-level artifacts
      final_model.zip      — the macro-level SAC policy (the one you deploy)
      episode_metrics.csv  — per-episode metrics
"""
from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401  # Hydra 1.3.x + Python ≥3.14 fix

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from c2g_env import C2GMacroEnv
from baselines.rule_based_mpc import RuleBasedController
from baselines.macro_metrics_callback import C2GMacroMetricsCallback
from baselines._hydra_compat import plant_overrides_from_cfg, simulation_options_from_cfg
from baselines.sac_provenance import bind_training_context
from c2g_env.experiments.identity import check_policy_semantics, file_fingerprint


# ---------------------------------------------------------------------------
# Inner policy: rule-based low-level controller
# ---------------------------------------------------------------------------

def _make_rule_based_inner_fn():
    """Create a rule-based inner_action_fn for C2GMacroEnv.

    Uses RuleBasedController (low-level, 4D actions for C2GFastEnv).
    """
    controller = RuleBasedController()

    def inner_fn(inner_obs: np.ndarray, macro_action: np.ndarray) -> np.ndarray:
        action, _ = controller.predict(inner_obs, deterministic=True)
        return action.astype(np.float32)

    return inner_fn


def _make_sac_inner_fn(checkpoint: str, simulation_mode: str):
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "final_model.zip"
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    model = SAC.load(str(path), device="cpu")
    check_policy_semantics(model, simulation_mode)
    if model.observation_space.shape != (18,) or model.action_space.shape != (4,):
        raise ValueError("The frozen hardware SAC must have 18 observations and 4 actions")

    def inner_fn(inner_obs: np.ndarray, macro_action: np.ndarray) -> np.ndarray:
        return model.predict(inner_obs, deterministic=True)[0]

    inner_fn.checkpoint = file_fingerprint(path)
    return inner_fn


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_macro_env(scenario: str, seed: int, inner_fn, **env_options):
    def _init():
        env = C2GMacroEnv(scenario=scenario, inner_action_fn=inner_fn, **env_options)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Phase 2: SAC high-level training
# ---------------------------------------------------------------------------

def train_macro_sac(scenario: str, seed: int, timesteps: int,
                    inner_fn, out_dir: Path, log_cfg, algo_cfg: dict,
                    env_options: dict | None = None, eval_options: dict | None = None,
                    device: str = "cpu") -> Path:
    """Train macro SAC against the explicitly selected frozen inner policy."""
    phase_dir = out_dir / "phase2"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # SAC is off-policy — single env, no VecNormalize needed
    env = _make_macro_env(scenario, seed, inner_fn, **(env_options or {}))()

    eval_env = make_vec_env(_make_macro_env(scenario, seed + 999, inner_fn, **(eval_options or env_options or {})),
                            n_envs=1, seed=seed + 999)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=algo_cfg.get("eval_freq", 5_000),
        save_path=str(phase_dir / "checkpoints"),
        name_prefix="ckpt",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(phase_dir / "best_model"),
        log_path=str(phase_dir / "tensorboard"),
        eval_freq=algo_cfg.get("eval_freq", 5_000),
        n_eval_episodes=algo_cfg.get("n_eval_episodes", 5),
        deterministic=True, verbose=0,
    )
    metrics_cb = C2GMacroMetricsCallback(
        print_freq=getattr(log_cfg, "console_freq", 20),
        csv_path=phase_dir / "episode_metrics.csv",
        verbose=1,
    )

    # ── SAC Model ─────────────────────────────────────────────────────────
    net_arch = algo_cfg.get("net_arch", [256, 256])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=algo_cfg.get("learning_rate", 3e-4),
        buffer_size=algo_cfg.get("buffer_size", 100_000),
        learning_starts=algo_cfg.get("learning_starts", 2_000),
        batch_size=algo_cfg.get("batch_size", 256),
        tau=algo_cfg.get("tau", 0.005),
        gamma=algo_cfg.get("gamma", 0.99),
        train_freq=algo_cfg.get("train_freq", 1),
        gradient_steps=algo_cfg.get("gradient_steps", 1),
        ent_coef=algo_cfg.get("ent_coef", "auto"),
        target_update_interval=algo_cfg.get("target_update_interval", 1),
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=str(phase_dir / "tensorboard"),
        verbose=0,
        seed=seed,
        device=device,
    )
    bind_training_context(model, env, phase_dir, inner_checkpoint=getattr(inner_fn, "checkpoint", None))

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[Phase 2] Training macro SAC with frozen inner policy: {timesteps:,} steps")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name="phase2_sac_macro",
        reset_num_timesteps=True,
    )

    model_path = phase_dir / "final_model"
    model.save(str(model_path))
    print(f"[Phase 2] Complete → {model_path}")
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

    # Get algo config (use sac.yaml if available)
    algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True) if "algo" in cfg else {}
    if algo_cfg.get("name") != "sac":
        raise ValueError("This trainer requires algo=sac")

    # HRL-specific overrides (optional, via +hrl.xxx on CLI)
    hrl_cfg = cfg.get("hrl")
    if hrl_cfg is None:
        hrl = {}
    elif OmegaConf.is_config(hrl_cfg):
        hrl = OmegaConf.to_container(hrl_cfg, resolve=True)
    else:
        hrl = dict(hrl_cfg)
    phase2_steps = hrl.get("phase2_steps", algo_cfg.get("timesteps", 400_000))

    print("═" * 60)
    print("  C2G-Bench: Frozen Hardware + SAC Macro")
    print(f"  scenario={scenario}  seed={seed}")
    print("  Phase 1: Explicit frozen hardware policy")
    print(f"  Phase 2: SAC — {phase2_steps:,} steps")
    print("═" * 60)

    # ── Phase 1: Rule-based (instant) ────────────────────────────────────
    env_options = simulation_options_from_cfg(cfg)
    eval_options = simulation_options_from_cfg(cfg, validation=True)
    env_options["plant_overrides"] = plant_overrides_from_cfg(cfg)
    eval_options["plant_overrides"] = plant_overrides_from_cfg(cfg)
    checkpoint = hrl.get("inner_checkpoint")
    if checkpoint:
        inner_fn = _make_sac_inner_fn(checkpoint, env_options.get("simulation_mode", "physical_v1"))
        print(f"[Phase 1] Frozen hardware SAC: {inner_fn.checkpoint}")
    elif hrl.get("inner_controller") == "rule_based":
        inner_fn = _make_rule_based_inner_fn()
        print("[Phase 1] Explicit rule-based hardware baseline")
    else:
        raise ValueError(
            "Specify +hrl.inner_checkpoint=/path/to/hardware.zip for staged SAC, "
            "or +hrl.inner_controller=rule_based for the explicit classical-inner baseline"
        )

    # ── Phase 2: SAC high-level ──────────────────────────────────────────
    macro_model_path = train_macro_sac(
        scenario, seed, phase2_steps,
        inner_fn, out_dir, log_cfg, algo_cfg,
        env_options=env_options, eval_options=eval_options, device=cfg.device,
    )

    # Copy final model to top-level for easy discovery
    import shutil
    shutil.copy2(str(macro_model_path) + ".zip",
                 str(out_dir / "final_model.zip"))

    print(f"\n{'═' * 60}")
    print(f"  HRL-SAC training complete → {out_dir.resolve()}")
    print(f"  Inner policy:  {checkpoint or 'rule_based hardware'} (frozen)")
    print(f"  Macro model:   {macro_model_path}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
