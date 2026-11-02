"""Access to the facility capacity profiles under ``conf/plant_profiles/``.

Shared by the benchmark runner and the experiment sweeps so the profile
directory is read in one place.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

_PROFILE_DIR = Path(__file__).resolve().parent.parent / "conf" / "plant_profiles"


def available_plant_profiles() -> list[str]:
    """Profile names under conf/plant_profiles/, ordered by nameplate capacity."""

    def _capacity_mw(name: str) -> int:
        match = re.search(r"(\d+)mw", name)
        return int(match.group(1)) if match else 250

    return sorted((p.stem for p in _PROFILE_DIR.glob("*.yaml")), key=_capacity_mw)


def load_plant_profile(name: str | None) -> dict | None:
    """Load the ``plant`` override block from conf/plant_profiles/<name>.yaml.

    Returns None for the default ``none`` profile (built-in 250 MW facility).
    """
    if not name or name == "none":
        return None
    with open(_PROFILE_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        profile = yaml.safe_load(fh)
    return profile.get("plant")
