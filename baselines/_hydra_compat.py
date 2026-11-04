"""
Monkey-patch for Hydra 1.3.x on Python >= 3.14.

Python 3.14 added ``argparse.ArgumentParser._check_help`` which calls
``'%' in help_string`` on every argument's *help* value.  Hydra 1.3.2's
``LazyCompletionHelp`` is a local class that may not implement
``__contains__``, causing::

    TypeError: argument of type 'LazyCompletionHelp' is not a container
               or iterable

Upstream issue: https://github.com/facebookresearch/hydra/issues/3121

Import this module **before** any ``@hydra.main()`` call to apply the
fix automatically.  It is a no-op on Python < 3.14 or when
``_check_help`` does not exist.
"""
from __future__ import annotations

import argparse
import sys

if sys.version_info >= (3, 14) and hasattr(argparse.ArgumentParser, "_check_help"):
    _original_check_help = argparse.ArgumentParser._check_help

    def _safe_check_help(self, action):  # type: ignore[no-untyped-def]
        try:
            _original_check_help(self, action)
        except (TypeError, ValueError):
            # Hydra's LazyCompletionHelp does not satisfy the string
            # protocol that Python 3.14's argparse expects — silently
            # skip the validation for this argument.
            pass

    argparse.ArgumentParser._check_help = _safe_check_help  # type: ignore[attr-defined]


def plant_overrides_from_cfg(cfg) -> dict | None:
    """Return the plant-capacity override dict from a Hydra cfg, or None.

    Reads the selected ``plant_profiles`` group's ``plant`` block and converts
    it to a plain dict for the env's ``plant_overrides`` kwarg. Returns None for
    the default ``none`` profile (built-in 250 MW facility).
    """
    from omegaconf import OmegaConf

    pp = cfg.get("plant_profiles") if hasattr(cfg, "get") else None
    if not pp:
        return None
    plant = pp.get("plant")
    if not plant:
        return None
    return OmegaConf.to_container(plant, resolve=True)


def simulation_options_from_cfg(cfg, *, validation: bool = False) -> dict:
    from omegaconf import OmegaConf

    options = OmegaConf.to_container(cfg.environment, resolve=True) if "environment" in cfg else {}
    if validation and "validation" in cfg and options.get("simulation_mode") != "legacy_v0":
        options.update(OmegaConf.to_container(cfg.validation, resolve=True))
    if "env_overrides" in cfg and options.get("simulation_mode") != "legacy_v0":
        for key in ("grid_market", "weather_market"):
            if key in cfg.env_overrides:
                options[key] = cfg.env_overrides[key]
    return options
