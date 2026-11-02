"""
c2g_env/thermal_limits.py  —  Single Source of Truth for the thermal anchors
============================================================================
The silicon thermal limit ``T_safe`` (``global.T_safe``) and the per-zone
warning thresholds ``T_warn_A`` / ``T_warn_B`` (``reward.T_warn_A`` and
``reward.T_warn_B``) are defined once in ``c2g_env/config.yaml`` and read here
so every consumer — the environment, baselines, safety filters, evaluation
scripts, and tests — uses identical values. Changing a config value therefore
propagates everywhere without editing code.

Usage::

    from c2g_env.thermal_limits import T_SAFE, T_WARN_A, T_WARN_B, T_WARN
    from c2g_env.thermal_limits import load_t_safe, load_t_warn
"""
from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_cfg(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_t_safe(config_path: str | Path | None = None) -> float:
    """Read ``global.T_safe`` [°C] from ``config.yaml``.

    Parameters
    ----------
    config_path : str or Path, optional
        Override path to ``config.yaml``. Defaults to the packaged config.
    """
    return float(_load_cfg(config_path)["global"]["T_safe"])


def load_t_warn(config_path: str | Path | None = None) -> tuple[float, float]:
    """Read ``reward.T_warn_A`` / ``reward.T_warn_B`` [°C] as ``(A, B)``.

    Parameters
    ----------
    config_path : str or Path, optional
        Override path to ``config.yaml``. Defaults to the packaged config.
    """
    rcfg = _load_cfg(config_path)["reward"]
    return float(rcfg["T_warn_A"]), float(rcfg["T_warn_B"])


#: Silicon thermal limit [°C], loaded once from the packaged config at import.
T_SAFE: float = load_t_safe()

#: Per-zone warning thresholds [°C], loaded once from the packaged config.
T_WARN_A, T_WARN_B = load_t_warn()

#: Zone-agnostic warning threshold for consumers that apply one bound to the
#: hotter of the two zones; the stricter of the two keeps such checks conservative.
T_WARN: float = min(T_WARN_A, T_WARN_B)
