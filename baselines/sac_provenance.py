from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import SAC

from c2g_env.experiments.identity import capture_source


def bind_training_context(
    model: SAC,
    env: Any,
    output_dir: Path,
    *,
    inner_checkpoint: dict | None = None,
) -> None:
    unwrapped = env.unwrapped
    fast = getattr(unwrapped, "_fast_env", unwrapped)
    model.c2g_context = {
        **capture_source(Path(__file__).resolve().parents[1], output_dir),
        "simulation_mode": fast.simulation_mode,
        "training_seed": model.seed,
        "global": fast._gcfg,
        "scenario": fast._scfg,
        "reward": fast._rcfg,
        "plant_overrides": fast._plant_overrides,
        "thermal_overrides": fast._thermal_overrides,
        "start_day": fast._episode_start_day,
        "regd_load_bias": fast._episode_regd_bias,
        "inner_checkpoint": inner_checkpoint,
    }
