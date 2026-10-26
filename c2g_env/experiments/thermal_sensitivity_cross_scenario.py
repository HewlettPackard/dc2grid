"""Full-environment thermal sensitivity across scenarios and controllers.

The rule-based macro controller is fixed while bang-bang, PID, rule-based,
and frozen SAC hardware controllers are evaluated over the physically
grounded plant configurations in ``conf/experiments.yaml``.

Runs append one row at a time and resume by default. Use ``--overwrite`` to
discard an existing output file.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from importlib.metadata import version
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import torch
import yaml

from baselines.bang_bang import BangBangController
from baselines.metrics_callback import (
    C2GTransitionLoggerCallback,
    _accumulate_step,
    _compute_episode_metrics,
    _make_episode_buffer,
)
from baselines.pid_controller import PIDController
from baselines.rule_based_macro import RuleBasedMacroController
from baselines.rule_based_mpc import RuleBasedController
from c2g_env import C2GMacroEnv
from c2g_env.experiments.identity import capture_source, configuration_id, file_fingerprint
from c2g_env.env_low_level import _THERMAL_OVERRIDE_KEYS, _PLANT_THERMAL_KEYS, _PLANT_BESS_KEYS
from c2g_env.experiments.thermal_sensitivity import build_configurations, load_config
from c2g_env.plant_profiles import available_plant_profiles, load_plant_profile
from c2g_env.thermal_limits import T_WARN

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _REPO_ROOT / "copilot" / "artifacts" / "thermal_sensitivity_cross_scenario.csv"
_FAULT_KEYS = ("thermal_fault", "freq_fault", "voltage_fault", "soc_fault", "sla_fault")
# Thermal terms proportional to facility size; the rest are intensive and
# stay fixed when the nameplate changes.
_EXTENSIVE_THERMAL_KEYS = ("C_A", "C_B", "K_liq", "K_air", "K_env_A", "K_env_B")
_CALLBACK_METRIC_FIELDS = tuple(_compute_episode_metrics(_make_episode_buffer()))
_FIELDS = (
    "run_id", "episode_id", "plant_profile", "facility_capacity_mw", "run_config",
    "recording_dir", "runtime_summary",
    "config_id", "config_aliases", "param", "value", "thermal_overrides",
    "scenario", "seed", "macro_controller", "hardware_controller",
    "completed", "terminated", "truncated", "termination_reason",
    "fast_steps", "macro_steps", "thermal_warning_steps", "thermal_warning_fraction",
    "temp_A_max", "temp_B_max", "temp_max", "return", "elapsed_seconds",
    *_FAULT_KEYS,
    *_CALLBACK_METRIC_FIELDS,
    "workload/mean_throughput_ratio", "workload/served_work_ratio",
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


def _initial_runtime_state(fast_env):
    return {
        "temp_A": fast_env._thermal.temp_A, "temp_B": fast_env._thermal.temp_B,
        "bess_soc": fast_env._bess.soc_fraction, "bess_age_frac": fast_env._bess._age_frac,
        "thermal_parameters": {key: getattr(fast_env._thermal, key)
                               for key in sorted(_THERMAL_OVERRIDE_KEYS | _PLANT_THERMAL_KEYS | {"T_safe", "COP_base_liq", "PUMP_MIN"})},
        "bess_parameters": {key: getattr(fast_env._bess, key) for key in sorted(_PLANT_BESS_KEYS)},
        "committed_mw_max": fast_env._committed_mw_max, "dr_baseline_mw": fast_env._dr_baseline_mw,
        "bess_backend": type(fast_env._bess).__name__,
    }


def _derived_runtime_metrics(hardware, macro, initial):
    hardware_stats = hardware["scalars"]
    macro_stats = macro["scalars"]
    arrived = hardware_stats["info.p_flex_nom_kw"]["sum"]
    revenue = macro_stats["info.regulation_revenue"]["sum"]
    cost = macro_stats["info.electricity_cost"]["sum"]
    return {
        "facility_energy_mwh": hardware_stats["info.p_facility_mw"]["integral_seconds"] / 3600.0,
        "it_energy_mwh": hardware_stats["info.p_total_it_mw"]["integral_seconds"] / 3600.0,
        "cooling_energy_mwh": hardware_stats["info.p_cooling_mw"]["integral_seconds"] / 3600.0,
        "peak_facility_mw": hardware_stats["info.p_facility_mw"]["maximum"],
        "tracking_mae_kw": hardware_stats["info.tracking_err_kw"]["mean"],
        "tracking_rmse_kw": hardware_stats["info.tracking_err_kw"]["rms"],
        "bess_signed_energy_mwh": hardware_stats["info.bess_actual_kw"]["integral_seconds"] / 3_600_000.0,
        "bess_throughput_mwh": hardware_stats["info.bess_actual_kw"]["absolute_integral_seconds"] / 3_600_000.0,
        "bess_initial_soc": initial["bess_soc"],
        "bess_final_soc": hardware_stats["info.bess_soc"]["final"],
        "bess_min_soc": min(initial["bess_soc"], hardware_stats["info.bess_soc"]["minimum"]),
        "bess_max_soc": max(initial["bess_soc"], hardware_stats["info.bess_soc"]["maximum"]),
        "bess_age_increment": hardware_stats["info.bess_age_frac"]["final"] - initial["bess_age_frac"],
        "served_work_ratio": hardware_stats["info.p_flex_served_kw"]["sum"] / arrived if arrived > 0 else None,
        "regulation_revenue": revenue, "electricity_cost": cost, "net_revenue": revenue - cost,
    }


def _episode_summary(config, scenario, seed, hardware, macro_name, env, hardware_summary, macro_summary, last_info, elapsed):
    scalars = hardware_summary["scalars"]
    faults = {key: bool(scalars[f"info.{key}"]["maximum"]) if f"info.{key}" in scalars else None for key in _FAULT_KEYS}
    reason = next((key.removesuffix("_fault") for key, active in faults.items() if active), None)
    return {
        "config_id": config["id"], "config_aliases": "|".join(config["aliases"]),
        "param": config["param"], "value": config["value"],
        "thermal_overrides": json.dumps(config["overrides"], sort_keys=True),
        "scenario": scenario, "seed": seed, "macro_controller": macro_name, "hardware_controller": hardware,
        "facility_capacity_mw": env._fast_env._facility_cap_kw / 1_000.0,
        "completed": True, "terminated": bool(macro_summary["scalars"]["terminated"]["final"]),
        "truncated": bool(macro_summary["scalars"]["truncated"]["final"]),
        "termination_reason": reason or last_info.get("last_inner_info", {}).get("truncation_reason") or "horizon",
        "fast_steps": hardware_summary["steps"], "macro_steps": macro_summary["steps"],
        "thermal_warning_steps": scalars["info.thermal_warning"]["true_count"],
        "thermal_warning_fraction": scalars["info.thermal_warning"]["mean"],
        "temp_A_max": scalars["info.temp_A"]["maximum"], "temp_B_max": scalars["info.temp_B"]["maximum"],
        "temp_max": max(scalars["info.temp_A"]["maximum"], scalars["info.temp_B"]["maximum"]),
        "return": macro_summary["scalars"]["reward"]["sum"], "elapsed_seconds": elapsed,
        **{key: int(value) if value is not None else None for key, value in faults.items()},
    }


def _finalize_recording(recording_dir, context, run_config, initial, hardware_summary, macro_summary, row):
    runtime = {"hardware": hardware_summary, "macro": macro_summary,
               "derived": _derived_runtime_metrics(hardware_summary, macro_summary, initial)}
    row.update(hardware_summary["episode_metrics"])
    row["workload/mean_throughput_ratio"] = hardware_summary["scalars"]["info.throughput_ratio"]["mean"]
    row["workload/served_work_ratio"] = runtime["derived"]["served_work_ratio"]
    (recording_dir / "summary.json").write_text(json.dumps({"episode": row, **runtime}, allow_nan=False), encoding="utf-8")
    files = {name: {"sha256": file_fingerprint(recording_dir / name)["sha256"], "bytes": (recording_dir / name).stat().st_size}
             for name in ("hardware.jsonl.gz", "macro.jsonl.gz", "summary.json")}
    manifest = {
        "schema_version": 3, "completed": True, "identity": context, "run_config": run_config,
        "initial_state": initial, "files": files,
        "observation_columns": {"hardware": C2GTransitionLoggerCallback.STATE_NAMES, "macro": C2GTransitionLoggerCallback.STATE_NAMES_MACRO},
        "action_columns": {"hardware": C2GTransitionLoggerCallback.ACTION_NAMES, "macro": C2GTransitionLoggerCallback.ACTION_NAMES_MACRO},
        "units": {"*_mw": "MW", "*_kw": "kW", "*_mwh": "MWh", "temp_*": "degC", "*_hz": "Hz", "*_pu": "per unit", "duration_seconds": "s"},
        "definitions": {
            "scalar_statistics": "Numeric info fields, actions, reward and end flags; nested last_inner_info excluded from macro statistics.",
            "episode_metrics": "Hardware metrics from C2GMetricsCallback; ep/length counts recorded steps and ep/survived requires the configured hardware horizon without termination.",
            "integral_seconds": "Sum of scalar value times actual step duration; MW integrals divided by 3600 give MWh.",
            "duration_weighted_mean": "Integral divided by the total duration on which the field was available.",
            "flag_fraction": "Mean is the fraction of observed samples; duration_weighted_mean is the exposure-time fraction.",
            "first_true_step": "One-based step within the corresponding hardware or macro stream; null means never true.",
            "throughput_ratio": "Reported throttle action, not completed-work fraction.",
            "served_work_ratio": "Total served flexible work divided by total arrivals in the workload model's kW-equivalent tick units.",
            "net_revenue": "Reported regulation revenue minus reported electricity cost; existing environment accounting is unchanged.",
            "missing_values": "Absent scalar fields are omitted; unavailable fault summaries are null, not zero.",
        },
    }
    (recording_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return runtime


def _run_episode(
    config: dict[str, Any], scenario: str, seed: int, hardware: str, sac_model: Any,
    macro_controller: Any, macro_name: str,
    plant_overrides: dict[str, Any] | None = None,
    *, record_dir: Path, episode_context: dict, run_config: dict,
) -> dict[str, Any]:
    env = C2GMacroEnv(scenario=scenario, thermal_overrides=config["overrides"], plant_overrides=plant_overrides)
    recording_dir = record_dir / episode_context["episode_id"][:16] / uuid4().hex
    started = time.perf_counter()
    with ExitStack() as stack:
        stack.callback(env.close)
        obs, reset_info = env.reset(seed=seed)
        fast_env = env._fast_env
        context = {**episode_context, "facility_capacity_mw": fast_env._facility_cap_kw / 1_000.0}
        initial = _initial_runtime_state(fast_env)
        hardware_log = C2GTransitionLoggerCallback(runtime_path=recording_dir / "hardware.jsonl.gz", experiment_context=context)
        stack.callback(hardware_log.close)
        macro_log = C2GTransitionLoggerCallback(runtime_path=recording_dir / "macro.jsonl.gz", agent_type="macro", experiment_context=context)
        stack.callback(macro_log.close)
        hardware_log.record_runtime_reset(env._last_hw_obs, env._last_hw_reset_info, initial)
        macro_log.record_runtime_reset(obs, reset_info, initial)
        controller = _make_hardware_controller(hardware, env, sac_model)
        env._inner_action_fn = _predict_fn(controller)
        macro_steps = 0
        episode_buffer = _make_episode_buffer()

        def collect(pre, action, post, reward, _done, info):
            _accumulate_step(episode_buffer, reward, info)
            hardware_log.record_runtime_transition(
                pre, action, post, reward, info["terminated"], info["truncated"], info,
                hardware_step=info["tick"], macro_step=macro_steps + 1, duration_seconds=fast_env._dt,
            )

        env._sub_step_callback = collect
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = macro_controller.predict(obs, deterministic=True)
            pre = obs
            obs, reward, terminated, truncated, last_info = env.step(action)
            macro_steps += 1
            macro_log.record_runtime_transition(
                pre, action, obs, reward, terminated, truncated, last_info,
                hardware_step=fast_env._tick, macro_step=macro_steps,
                duration_seconds=last_info["sub_steps_run"] * fast_env._dt,
            )
        hardware_summary = hardware_log.runtime_summary()
        macro_summary = macro_log.runtime_summary()
        hardware_summary["episode_metrics"] = _compute_episode_metrics(episode_buffer)
        hardware_summary["episode_metrics"]["ep/length"] = float(hardware_summary["steps"])
        hardware_summary["episode_metrics"]["ep/survived"] = float(
            truncated and not terminated and hardware_summary["steps"] >= fast_env._episode_ticks
        )
    row = _episode_summary(config, scenario, seed, hardware, macro_name, env, hardware_summary, macro_summary,
                           last_info, time.perf_counter() - started)
    runtime = _finalize_recording(recording_dir, context, run_config, initial, hardware_summary, macro_summary, row)
    row["recording_dir"] = str(recording_dir.resolve())
    row["runtime_summary"] = json.dumps(runtime, allow_nan=False)
    return row


def _episode_id(run_id: str, config: dict[str, Any], scenario: str, seed: int, hardware: str) -> str:
    return configuration_id({
        "run_id": run_id,
        "config_id": config["id"],
        "thermal_overrides": config["overrides"],
        "scenario": scenario,
        "seed": seed,
        "hardware_controller": hardware,
    })


def _recording_complete(row) -> bool:
    try:
        recording_dir = Path(row["recording_dir"])
        manifest = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("completed") is not True or manifest["identity"]["episode_id"] != row["episode_id"]:
            return False
        for name in ("hardware.jsonl.gz", "macro.jsonl.gz", "summary.json"):
            path = recording_dir / name
            expected = manifest["files"][name]
            if path.stat().st_size != expected["bytes"] or file_fingerprint(path)["sha256"] != expected["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(_FIELDS):
            raise ValueError(f"Incompatible result schema in {path}; use a new output file.")
        return {
            row["episode_id"]
            for row in reader
            if row.get("completed", "").lower() == "true" and _recording_complete(row)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--record-dir", type=Path, default=_REPO_ROOT / "copilot" / "artifacts" / "thermal_runtime")
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
    torch.set_num_threads(int(run_cfg["torch_num_threads"]))
    device = yaml.safe_load((_REPO_ROOT / "conf" / "config.yaml").read_text(encoding="utf-8"))["device"]
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
    hardware_checkpoint = None
    if "sac" in controllers:
        from stable_baselines3 import SAC
        checkpoint_path = _REPO_ROOT / (args.sac_model or run_cfg["sac_model"])
        hardware_checkpoint = file_fingerprint(checkpoint_path)
        sac_model = SAC.load(str(checkpoint_path), device=device)

    macro_name = args.macro_controller or run_cfg["macro_controller"]
    macro_checkpoint = None
    if macro_name == "sac_macro":
        from stable_baselines3 import SAC
        checkpoint_path = _REPO_ROOT / (args.sac_macro_model or run_cfg["sac_macro_model"])
        macro_checkpoint = file_fingerprint(checkpoint_path)
        macro_controller = SAC.load(str(checkpoint_path), device=device)
    else:
        macro_controller = RuleBasedMacroController()

    run_config = {
        "schema_version": 3,
        "plant_profile": args.plant_profile,
        "plant_overrides": plant_overrides,
        "macro_controller": macro_name,
        "hardware_checkpoint": hardware_checkpoint,
        "macro_checkpoint": macro_checkpoint,
        "environment": yaml.safe_load((_REPO_ROOT / "c2g_env" / "config.yaml").read_text(encoding="utf-8")),
        "device": device,
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "torch", "stable-baselines3", "gymnasium")},
        **capture_source(_REPO_ROOT, args.record_dir / "provenance"),
    }
    run_id = configuration_id(run_config)
    run_config_json = json.dumps(run_config, sort_keys=True, allow_nan=False)

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
        if _episode_id(run_id, config, scenario, seed, controller) not in completed
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
            episode_context = {
                "run_id": run_id, "episode_id": _episode_id(run_id, config, scenario, seed, controller),
                "plant_profile": args.plant_profile, "config_id": config["id"], "thermal_overrides": config["overrides"],
                "scenario": scenario, "seed": seed, "macro_controller": macro_name, "hardware_controller": controller,
            }
            try:
                row = _run_episode(config, scenario, seed, controller, sac_model, macro_controller, macro_name, plant_overrides,
                                   record_dir=args.record_dir, episode_context=episode_context, run_config=run_config)
            except Exception:
                with (args.record_dir / "runtime_issues.jsonl").open("a", encoding="utf-8") as issues:
                    issues.write(json.dumps({**episode_context, "traceback": traceback.format_exc()}) + "\n")
                raise
            row.update(
                run_id=run_id,
                episode_id=_episode_id(run_id, config, scenario, seed, controller),
                plant_profile=args.plant_profile,
                run_config=run_config_json,
            )
            writer.writerow(row)
            handle.flush()
            print(
                f"[{index}/{len(pending)}] {config['id']} {scenario} s{seed} {controller}: "
                f"Tmax={row['temp_max']:.2f}, reason={row['termination_reason']}, "
                f"{row['elapsed_seconds']:.1f}s"
            )


if __name__ == "__main__":
    main()