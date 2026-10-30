from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[2]


def load_config(path: Path = MODULE_ROOT / "config.yaml") -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("The experiment configuration must be a mapping.")
    for section in ("c2g", "three_l", "simulation", "controllers", "reward"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    validate_numbers(cfg)
    validate_paths(cfg)
    if set(cfg["controllers"]) != {"inner", "model_dir"}:
        raise ValueError(
            "controllers requires inner and model_dir; native 3L policies "
            "use their upstream parameter defaults."
        )
    return cfg


def validate_paths(cfg: dict) -> None:
    paths = {
        "c2g.config": cfg["c2g"]["config"],
        "three_l.root": cfg["three_l"]["root"],
        "three_l.config": cfg["three_l"]["config"],
    }
    for key, value in paths.items():
        if not isinstance(value, str) or "\\" in value:
            raise ValueError(
                f"{key} must be a POSIX-style relative path (forward slashes): "
                f"{value!r}"
            )


def validate_numbers(cfg: dict) -> None:
    values = {
        **cfg["reward"],
        "warmup_hours": cfg["simulation"]["warmup_hours"],
        "scoring_hours": cfg["simulation"]["scoring_hours"],
    }
    for key, value in values.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if cfg["reward"]["cost_scale_eur"] <= 0 or cfg["simulation"]["scoring_hours"] <= 0:
        raise ValueError("cost_scale_eur and scoring_hours must be positive.")
    seed = cfg["simulation"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("simulation.seed must be a nonnegative integer.")


def three_l_config(cfg: dict) -> tuple[Path, dict, Path, dict]:
    root = (REPO_ROOT / cfg["three_l"]["root"]).resolve()
    with (root / cfg["three_l"]["config"]).open(encoding="utf-8") as handle:
        backend = json.load(handle)
    inputs = (root / backend["sim"]["inputs_path"]).resolve()
    backend["sim"]["inputs_path"] = str(inputs)
    scenario = inputs / backend["sim"]["scenario"]
    with (scenario / "config.yaml").open(encoding="utf-8") as handle:
        calendar = yaml.safe_load(handle)[backend["sim"]["study_case"]]
    if backend.get("dc"):
        raise ValueError("Use a market-only 3L configuration, without a dc overlay.")
    return root, backend, scenario, calendar


def contained_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(MODULE_ROOT):
        raise ValueError(f"Output must remain inside {MODULE_ROOT}: {resolved}")
    return resolved
