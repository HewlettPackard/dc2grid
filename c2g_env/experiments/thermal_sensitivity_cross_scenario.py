"""Full-environment thermal sensitivity across scenarios and controllers.

The rule-based macro controller is fixed while bang-bang, PID, rule-based,
and frozen SAC hardware controllers are evaluated over the physically
grounded plant configurations in ``conf/experiments.yaml``.

Runs append one row at a time and resume by default. Use ``--overwrite`` to
discard an existing output file.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.rule_based_macro import RuleBasedMacroController
from baselines.rule_based_mpc import RuleBasedController
from c2g_env import C2GMacroEnv
from c2g_env.experiments.thermal_sensitivity import build_configurations, load_config
from c2g_env.plant_profiles import available_plant_profiles, load_plant_profile
from c2g_env.thermal_limits import T_WARN

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _REPO_ROOT / "copilot" / "artifacts" / "thermal_sensitivity_cross_scenario.csv"
_FAULT_KEYS = ("thermal_fault", "freq_fault", "voltage_fault", "soc_fault", "sla_fault")
# Thermal terms proportional to facility size; the rest are intensive and
# stay fixed when the nameplate changes.
_EXTENSIVE_THERMAL_KEYS = ("C_A", "C_B", "K_liq", "K_air", "K_env_A", "K_env_B")
_FIELDS = (
    "config_id", "config_aliases", "param", "value", "thermal_overrides",
    "scenario", "seed", "macro_controller", "hardware_controller",
    "completed", "terminated", "truncated", "termination_reason",
    "fast_steps", "macro_steps", "thermal_warning_steps", "thermal_warning_fraction",
    "temp_A_max", "temp_B_max", "temp_max", "return", "elapsed_seconds",
    *_FAULT_KEYS,
)


def _explicit_configurations(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unique plant changes relative to nominal scenario behavior."""
    nominal = {key: float(value) for key, value in cfg["nominal"].items()}
    unique: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}
    for config in build_configurations(cfg):
        explicit = {
            key: float(value)
            for key, value in config["overrides"].items()
            if not np.isclose(float(value), nominal[key])
        }
        signature = tuple(sorted(explicit.items()))
        if signature in unique:
            unique[signature]["aliases"].append(config["id"])
            continue
        unique[signature] = {
            **config,
            "overrides": explicit,
            "aliases": [config["id"]],
        }
    return list(unique.values())


def _rescale_thermal_grid(cfg: dict[str, Any], plant: dict[str, Any]) -> dict[str, Any]:
    """Return ``cfg`` with the extensive thermal grid scaled to the profile.

    The capacity ratio is read off the profile's own ``C_A`` so the sweep keeps
    the time constants and cooling approaches that the 250 MW grid encodes.
    """
    scale = float(plant["C_A"]) / float(cfg["nominal"]["C_A"])

    def _scaled(key: str, value: float) -> float:
        return float(value) * scale if key in _EXTENSIVE_THERMAL_KEYS else float(value)

    scaled = dict(cfg)
    scaled["nominal"] = {k: _scaled(k, v) for k, v in cfg["nominal"].items()}
    scaled["sweep"] = {
        k: [_scaled(k, v) for v in values] for k, values in cfg.get("sweep", {}).items()
    }
    scaled["coupled_cases"] = {
        case: {k: _scaled(k, v) for k, v in overrides.items()}
        for case, overrides in cfg.get("coupled_cases", {}).items()
    }
    return scaled


def _make_hardware_controller(name: str, env: C2GMacroEnv, sac_model: Any) -> Any:
    if name == "bang_bang":
        return BangBangController()
    if name == "pid":
        return PIDController()
    if name == "rule_based":
        fast_env = env._fast_env
        return RuleBasedController(
            committed_mw_max=float(fast_env._committed_mw_max),
            bess_p_max_mw=float(fast_env._bess.P_MAX_MW),
        )
    if name == "sac":
        return sac_model
    raise ValueError(f"Unknown hardware controller: {name}")


def _predict_fn(controller: Any) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def predict(obs: np.ndarray, _macro_action: np.ndarray) -> np.ndarray:
        action, _ = controller.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return predict


def _run_episode(
    config: dict[str, Any], scenario: str, seed: int, hardware: str, sac_model: Any,
    macro_controller: Any, macro_name: str,
    plant_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "fast_steps": 0,
        "thermal_warning_steps": 0,
        "temp_A_max": float("-inf"),
        "temp_B_max": float("-inf"),
        **{key: False for key in _FAULT_KEYS},
    }

    def collect(_pre, _action, _obs, _reward, _done, info):
        stats["fast_steps"] += 1
        temp_a = float(info["temp_A"])
        temp_b = float(info["temp_B"])
        stats["temp_A_max"] = max(stats["temp_A_max"], temp_a)
        stats["temp_B_max"] = max(stats["temp_B_max"], temp_b)
        stats["thermal_warning_steps"] += int(max(temp_a, temp_b) > T_WARN)
        for key in _FAULT_KEYS:
            stats[key] = stats[key] or bool(info.get(key, False))

    env = C2GMacroEnv(
        scenario=scenario,
        thermal_overrides=config["overrides"],
        plant_overrides=plant_overrides,
        sub_step_callback=collect,
    )
    obs, _ = env.reset(seed=seed)
    controller = _make_hardware_controller(hardware, env, sac_model)
    env._inner_action_fn = _predict_fn(controller)
    started = time.perf_counter()
    total_reward = 0.0
    macro_steps = 0
    terminated = truncated = False
    last_info: dict[str, Any] = {}

    while not (terminated or truncated):
        action, _ = macro_controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        macro_steps += 1
    env.close()

    fast_steps = int(stats["fast_steps"])
    last_inner = last_info.get("last_inner_info", {})
    fault_reason = next(
        (key.removesuffix("_fault") for key in _FAULT_KEYS if stats[key]),
        None,
    )
    return {
        "config_id": config["id"],
        "config_aliases": "|".join(config["aliases"]),
        "param": config["param"],
        "value": config["value"],
        "thermal_overrides": json.dumps(config["overrides"], sort_keys=True),
        "scenario": scenario,
        "seed": seed,
        "macro_controller": macro_name,
        "hardware_controller": hardware,
        "completed": True,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": fault_reason or last_inner.get("truncation_reason") or last_info.get("truncation_reason") or "horizon",
        "fast_steps": fast_steps,
        "macro_steps": macro_steps,
        "thermal_warning_steps": stats["thermal_warning_steps"],
        "thermal_warning_fraction": stats["thermal_warning_steps"] / max(fast_steps, 1),
        "temp_A_max": stats["temp_A_max"],
        "temp_B_max": stats["temp_B_max"],
        "temp_max": max(stats["temp_A_max"], stats["temp_B_max"]),
        "return": total_reward,
        "elapsed_seconds": time.perf_counter() - started,
        **{key: int(stats[key]) for key in _FAULT_KEYS},
    }


def _completed_keys(path: Path) -> set[tuple[str, str, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["config_id"], row["scenario"], int(row["seed"]), row["hardware_controller"])
            for row in csv.DictReader(handle)
            if row.get("completed", "").lower() == "true"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--controllers", nargs="+")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--plant-profile",
        default="none",
        choices=available_plant_profiles(),
        help="Facility capacity profile from conf/plant_profiles/. Default "
             "'none' = 250 MW. Extensive thermal sweep values are rescaled to match.",
    )
    parser.add_argument(
        "--sac-model",
        help="Override the SAC low-level checkpoint from conf/experiments.yaml.",
    )
    parser.add_argument(
        "--macro-controller",
        choices=["rule_macro", "sac_macro"],
        help="High-level controller. Default from conf/experiments.yaml.",
    )
    parser.add_argument(
        "--sac-macro-model",
        help="Override the SAC macro checkpoint from conf/experiments.yaml.",
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_cfg = cfg["cross_scenario"]
    plant_overrides = load_plant_profile(args.plant_profile)
    if plant_overrides:
        cfg = _rescale_thermal_grid(cfg, plant_overrides)
    configs = _explicit_configurations(cfg)
    if args.configs:
        selected = set(args.configs)
        configs = [config for config in configs if config["id"] in selected]
    scenarios = args.scenarios or run_cfg["scenarios"]
    seeds = args.seeds or run_cfg["seeds"]
    controllers = args.controllers or run_cfg["hardware_controllers"]

    sac_model = None
    if "sac" in controllers:
        from stable_baselines3 import SAC
        sac_model = SAC.load(str(_REPO_ROOT / (args.sac_model or run_cfg["sac_model"])))

    macro_name = args.macro_controller or run_cfg["macro_controller"]
    if macro_name == "sac_macro":
        from stable_baselines3 import SAC
        macro_controller = SAC.load(
            str(_REPO_ROOT / (args.sac_macro_model or run_cfg["sac_macro_model"]))
        )
    else:
        macro_controller = RuleBasedMacroController()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.out.exists():
        args.out.unlink()
    completed = _completed_keys(args.out)
    pending = [
        (config, scenario, seed, controller)
        for config in configs
        for scenario in scenarios
        for seed in seeds
        for controller in controllers
        if (config["id"], scenario, seed, controller) not in completed
    ]
    if args.max_runs is not None:
        pending = pending[:args.max_runs]
    print(
        f"Running {len(pending)} episodes ({len(configs)} unique plant configs, "
        f"{len(scenarios)} scenarios, {len(seeds)} seeds, {len(controllers)} controllers) "
        f"on plant profile '{args.plant_profile}' with macro '{macro_name}'"
    )

    write_header = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        for index, (config, scenario, seed, controller) in enumerate(pending, 1):
            row = _run_episode(config, scenario, seed, controller, sac_model, macro_controller, macro_name, plant_overrides)
            writer.writerow(row)
            handle.flush()
            print(
                f"[{index}/{len(pending)}] {config['id']} {scenario} s{seed} {controller}: "
                f"Tmax={row['temp_max']:.2f}, reason={row['termination_reason']}, "
                f"{row['elapsed_seconds']:.1f}s"
            )


if __name__ == "__main__":
    main()