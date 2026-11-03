"""Isolated, service-first evaluation of the existing corrected Table 1 policies."""

from __future__ import annotations

import ast
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SCHEMA = "table1-corrected-existing-case-v1"
PANEL_SPEC = {
    "scenario": "default",
    "simulation_mode": "physical_v1",
    "start_day": 0,
    "regd_load_bias": 0.0,
    "seeds": [100, 101, 102, 103, 104],
    "dt_seconds": 5,
    "macro_seconds": 900,
    "service_seconds": 86400,
    "service_fast_ticks": 17280,
    "service_macro_calls": 96,
    "thermal_warning_threshold_C": 33.0,
    "thermal_warning_comparison": "post-step temp_A >= 33 or temp_B >= 33",
    "conditions_are_source_default_reconstruction": True,
    "original_producing_run_conditions_proven": False,
    "start_day_was_in_training": True,
    "independent_training_seeds": False,
}
ROW_SPECS = [
    {"id": "rule_macro+none", "macro": "rule_macro", "hardware": "none"},
    {"id": "rule_macro+random", "macro": "rule_macro", "hardware": "random"},
    {"id": "rule_macro+bang_bang", "macro": "rule_macro", "hardware": "bang_bang"},
    {"id": "rule_macro+rule_based", "macro": "rule_macro", "hardware": "rule_based"},
    {"id": "rule_macro+pid", "macro": "rule_macro", "hardware": "pid"},
    {"id": "rule_macro+sac", "macro": "rule_macro", "hardware": "sac"},
    {"id": "sac_macro+sac", "macro": "sac_macro", "hardware": "sac"},
]
MODEL_SPECS = {
    "low": {
        "path": str(
            PROJECT
            / "outputs/corrected_sac_8gpu_20261030/seed_1007"
            / "low/best_model/best_model.zip"
        ),
        "sha256": "5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024",
        "training_seed": 1007,
        "selected_timesteps": 250002,
    },
    "high": {
        "path": str(
            PROJECT
            / "outputs/corrected_sac_8gpu_20261030/seed_1007"
            / "high/best_model/best_model.zip"
        ),
        "sha256": "a1b5e5752d68086565c7c4fec545c80b2c702b50f07fe90ab283379898bdb388",
        "training_seed": 1007,
        "selected_timesteps": 8004,
    },
}
RECOVERY_OPS_SPEC = {
    "path": str(
        PROJECT / "outputs/high_macro_recovery_pilot_20261031_v3/recovery_ops.py"
    ),
    "sha256": "45395520fd57814fac7551c0c0e19382f921f25b3e6add2592cbc1762fc8a069",
}
RECOVERY_SPEC = {
    "duration_seconds": 7200,
    "throttle": 1.0,
    "pump": 0.7,
    "hvac": 0.7,
    "committed_mw": 0.0,
    "regulation_revenue_usd": 0.0,
    "soc_tolerance": 1e-6,
    "backlog_tolerance_kwh": 1e-6,
    "temperature_tolerance_C": 0.1,
    "work_conservation_tolerance_kwh": 1e-4,
    "soc_target": "episode_initial_soc",
    "temperature_target": "continuing_nominal_reference_trajectory",
    "fixed_duration_even_if_closed_early": True,
    "terminate_on_existing_physical_faults": True,
    "wear_price_model": "USD per MWh of absolute AC battery throughput",
    "wear_prices_usd_per_mwh": [0.0, 5.0, 10.0, 25.0, 50.0],
}
SERVICE_GATE_SPEC = {
    "min_service_end_soc": 0.20,
    "max_backlog_fraction_of_arrived_work": 0.01,
    "max_thermal_warning_fraction": 0.01,
    "work_conservation_tolerance_kwh": 1e-4,
    "require_physical_completion": True,
    "require_existing_two_hour_recovery_closure": True,
    "capacity_admission_floor": None,
    "credit_guard": None,
    "service_hardware_action_override": None,
}
DATA_PATHS = (
    "data/processed/energy/NYC.csv",
    "data/processed/weather/NYC.csv",
    "data/processed/workload_traces/batch_v2023.csv",
    "data/processed/workload_traces/dlrm_v2025.csv",
    "data/processed/workload_traces/genai_v2026.csv",
)
SERVICE_METRIC_KEYS = (
    "tracking_rmse_kw",
    "regulation_revenue_usd",
    "macro_mean_reward",
    "low_mean_reward",
    "thermal_warning_fraction",
    "bess_aging_x1e4",
)
FAULT_KEYS = ("thermal_fault", "freq_fault", "voltage_fault")


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def identity(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_value,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_receipt(path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def runtime_source_paths() -> list[str]:
    return sorted(
        str(path.relative_to(PROJECT))
        for directory in ("c2g_env", "baselines", "evaluation", "scripts")
        for path in (PROJECT / directory).rglob("*.py")
    )


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=json_value)
        handle.write("\n")


def _verify_receipt(receipt: dict) -> Path:
    path = Path(receipt["path"])
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError(f"Input is outside the registered repository: {path}")
    if not path.is_file() or file_hash(path) != receipt["sha256"]:
        raise ValueError(f"Missing or changed registered input: {path}")
    return path


def _row_spec(policy) -> dict:
    name = policy if isinstance(policy, str) else policy["id"]
    row = next((row for row in ROW_SPECS if row["id"] == name), None)
    if row is None:
        raise ValueError(f"Policy is outside the seven-row roster: {name}")
    if not isinstance(policy, str):
        for key, value in row.items():
            if policy.get(key) != value:
                raise ValueError(f"Policy role differs from the fixed row: {key}")
    return dict(row)


def verify_protocol(protocol: dict, policy, seed: int) -> dict:
    """Validate identities and the closed panel; never create a registration."""
    row = _row_spec(policy)
    if not protocol.get("registration_id"):
        raise ValueError("A parent-created registration_id is required")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in PANEL_SPEC["seeds"]
    ):
        raise ValueError("Seed is outside the registered five-episode panel")
    if protocol["panel"] != PANEL_SPEC or protocol["recovery"] != RECOVERY_SPEC:
        raise ValueError("The fixed service/recovery contract was changed")
    if protocol["service_gate"] != SERVICE_GATE_SPEC:
        raise ValueError("The supplemental physical gates were changed")
    if protocol["policies"] != ROW_SPECS:
        raise ValueError("The seven-row roster or its order was changed")
    if Path.cwd().resolve() != PROJECT:
        raise ValueError("Run from the repository root; native data paths are relative")
    if file_hash(__file__) != protocol["evaluator_sha256"]:
        raise ValueError("Evaluator changed after registration")
    if set(runtime_source_paths()) != set(protocol["source_files"]):
        raise ValueError("Native source inventory differs from the registration")
    for path, digest in protocol["source_files"].items():
        _verify_receipt({"path": path, "sha256": digest})
    if set(protocol["data_files"]) != set(DATA_PATHS):
        raise ValueError("All five native input files must be registered")
    for path, digest in protocol["data_files"].items():
        _verify_receipt({"path": path, "sha256": digest})
    config_path = _verify_receipt(protocol["config"])
    if config_path != PROJECT / "c2g_env/config.yaml":
        raise ValueError(
            "Production evaluation must use the unchanged native configuration"
        )
    for role, expected in MODEL_SPECS.items():
        supplied = protocol["models"][role]
        if supplied != expected:
            raise ValueError(f"Wrong selected {role} model; no final-model fallback")
        _verify_receipt(supplied)
    if protocol["recovery_ops"] != RECOVERY_OPS_SPEC:
        raise ValueError("The inherited recovery implementation was changed")
    _verify_receipt(protocol["recovery_ops"])
    return {
        "config": protocol["config"],
        "source_files": protocol["source_files"],
        "source_id": identity(protocol["source_files"]),
        "data_files": protocol["data_files"],
        "models": protocol["models"],
        "recovery_ops": protocol["recovery_ops"],
        "evaluator_sha256": protocol["evaluator_sha256"],
        "row": row,
    }


def _verify_import_origins(protocol: dict) -> None:
    for name, module in list(sys.modules.items()):
        if name in ("c2g_env", "baselines") or name.startswith(
            ("c2g_env.", "baselines.")
        ):
            filename = getattr(module, "__file__", None)
            if filename is None:
                continue
            path = Path(filename).resolve()
            if not path.is_relative_to(PROJECT):
                raise RuntimeError(f"Foreign simulator import: {name}: {path}")
            relative = str(path.relative_to(PROJECT))
            if relative not in protocol["source_files"]:
                raise RuntimeError(f"Unregistered simulator import: {name}: {path}")
            if file_hash(path) != protocol["source_files"][relative]:
                raise RuntimeError(f"Changed simulator import: {name}: {path}")


def load_recovery_ops(receipt: dict = RECOVERY_OPS_SPEC):
    """Reuse pinned functions without executing their historical import bootstrap."""
    if receipt != RECOVERY_OPS_SPEC:
        raise ValueError("Unexpected recovery implementation")
    path = _verify_receipt(receipt)
    from c2g_env.physics.bess import _SimpleBESSModel
    from c2g_env.physics.macro_grid import MacroGridSignal
    from c2g_env.physics.weather import WeatherLoader

    names = {
        "plant_state",
        "continuation_identity",
        "begin_recovery",
        "battery_dispatch",
        "closure",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names or len(functions) != len(names):
        raise ValueError("Inherited recovery function inventory changed")
    namespace = {
        "identity": identity,
        "copy": copy,
        "math": math,
        "np": np,
        "_SimpleBESSModel": _SimpleBESSModel,
        "MacroGridSignal": MacroGridSignal,
        "WeatherLoader": WeatherLoader,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return SimpleNamespace(**{name: namespace[name] for name in names})


def _load_model(role: str, receipt: dict):
    from stable_baselines3 import SAC

    from c2g_env.experiments.identity import check_policy_semantics

    path = _verify_receipt(receipt)
    expected = MODEL_SPECS[role]
    if receipt != expected:
        raise ValueError(f"Unexpected {role} model")
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("data"))
    context = metadata.get("c2g_context", {})
    if (
        metadata["num_timesteps"] != expected["selected_timesteps"]
        or metadata["seed"] != expected["training_seed"]
        or context.get("simulation_mode") != "physical_v1"
        or context.get("stage") != role
    ):
        raise ValueError(f"Incorrect selected {role} checkpoint metadata")
    if (
        role == "high"
        and context.get("inner_checkpoint", {}).get("sha256")
        != MODEL_SPECS["low"]["sha256"]
    ):
        raise ValueError("Macro model was not trained with this exact frozen low model")
    model = SAC.load(str(path), device="cpu")
    check_policy_semantics(model, "physical_v1", allow_mismatch=False)
    model.policy.set_training_mode(False)
    return model


class _NominalController:
    def predict(self, observation, deterministic=True):
        return np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32), None


class _RandomController:
    def __init__(self, action_space, seed: int):
        self.action_space = copy.deepcopy(action_space)
        self.action_space.seed(seed)

    def predict(self, observation, deterministic=True):
        return self.action_space.sample(), None


def _hardware_controller(name: str, fast, seed: int, models: dict):
    from baselines.bang_bang import BangBangController
    from baselines.pid_controller import PIDController
    from baselines.rule_based_mpc import RuleBasedController

    if name == "none":
        controller = _NominalController()
    elif name == "random":
        controller = _RandomController(fast.action_space, seed)
    elif name == "bang_bang":
        controller = BangBangController()
    elif name == "pid":
        controller = PIDController()
    elif name == "rule_based":
        controller = RuleBasedController(
            committed_mw_max=fast._committed_mw_max, bess_p_max_mw=fast._bess.P_MAX_MW
        )
    elif name == "sac":
        controller = _load_model("low", models["low"])
    else:
        raise ValueError(f"Unsupported hardware controller: {name}")
    if hasattr(controller, "reset"):
        controller.reset()
    return controller


class _CsvTrace:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.writer = None
        self.count = 0

    def write(self, row: dict) -> None:
        if self.handle is None:
            self.handle = gzip.open(self.path, "xt", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(self.handle, fieldnames=sorted(row))
            self.writer.writeheader()
        encoded = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                value = json.dumps(value, separators=(",", ":"), default=json_value)
            elif isinstance(value, np.generic):
                value = value.item()
            encoded[key] = value
        self.writer.writerow(encoded)
        self.count += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def hardware_row(
    pre, action, post, reward, info, fast, phase: str, macro_call: int
) -> dict:
    metered_response = 1000.0 * (
        float(info["p_reference_mw"]) - float(info["p_grid_import_mw"])
    )
    signed_error = float(info["delta_p_demanded_kw"]) - metered_response
    return {
        **info,
        "phase": phase,
        "macro_call": macro_call,
        "observation_before": np.asarray(pre).tolist(),
        "observation_after": np.asarray(post).tolist(),
        "requested_action": np.asarray(action).tolist(),
        "low_reward": float(reward),
        "metered_response_kw": metered_response,
        "metered_signed_error_kw": signed_error,
        "table1_thermal_warning": bool(
            info["temp_A"] >= 33.0 or info["temp_B"] >= 33.0
        ),
        "committed_actual_mw": float(fast.committed_mw),
        "regd_observed": float(pre[6]),
        "committed_observed_mw": float(pre[17] * fast._committed_mw_max),
        "reference_temp_A": float(fast._reference_thermal.temp_A),
        "reference_temp_B": float(fast._reference_thermal.temp_B),
    }


class HardwareTotals:
    def __init__(self):
        self.ticks = 0
        self.reward_sum = 0.0
        self.error_sq_sum = 0.0
        self.warning_ticks = 0
        self.duration_hours = 0.0
        self.electricity_cost_usd = 0.0
        self.grid_import_mwh = 0.0
        self.battery_throughput_mwh = 0.0
        self.work_arrived_kwh = 0.0
        self.work_served_kwh = 0.0
        self.backlog_final_kwh = 0.0
        self.fault_counts = dict.fromkeys(FAULT_KEYS, 0)
        self.first_fault_tick = dict.fromkeys(FAULT_KEYS)
        self.max_abs_freq_dev_hz = 0.0
        self.min_voltage_pu = None
        self.max_temp_A = None
        self.max_temp_B = None

    def add(self, row: dict) -> None:
        numeric = (
            "metered_signed_error_kw",
            "metered_response_kw",
            "tracking_err_kw",
            "delta_p_actual_kw",
            "duration_hours",
            "low_reward",
            "lmp",
            "p_grid_import_mw",
            "bess_actual_kw",
            "work_arrived_kwh",
            "work_served_kwh",
            "backlog_work_kwh",
            "temp_A",
            "temp_B",
            "freq_dev_hz",
            "v_pcc_pu",
        )
        if not all(math.isfinite(float(row[key])) for key in numeric):
            raise ValueError("Nonfinite hardware trace value")
        if not math.isclose(
            abs(row["metered_signed_error_kw"]),
            row["tracking_err_kw"],
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                "Reported tracking differs from physical metered error"
            )
        if not math.isclose(
            row["metered_response_kw"],
            row["delta_p_actual_kw"],
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                "Reported response differs from physical metered response"
            )
        if row["duration_hours"] <= 0:
            raise ValueError("Nonpositive physical interval")
        self.ticks += 1
        self.reward_sum += row["low_reward"]
        self.error_sq_sum += row["metered_signed_error_kw"] ** 2
        self.warning_ticks += int(row["table1_thermal_warning"])
        self.duration_hours += row["duration_hours"]
        self.electricity_cost_usd += (
            row["lmp"] * row["p_grid_import_mw"] * row["duration_hours"]
        )
        self.grid_import_mwh += row["p_grid_import_mw"] * row["duration_hours"]
        self.battery_throughput_mwh += (
            abs(row["bess_actual_kw"]) / 1000.0 * row["duration_hours"]
        )
        self.work_arrived_kwh += row["work_arrived_kwh"]
        self.work_served_kwh += row["work_served_kwh"]
        self.backlog_final_kwh = row["backlog_work_kwh"]
        self.max_abs_freq_dev_hz = max(
            self.max_abs_freq_dev_hz, abs(row["freq_dev_hz"])
        )
        self.min_voltage_pu = (
            row["v_pcc_pu"]
            if self.min_voltage_pu is None
            else min(self.min_voltage_pu, row["v_pcc_pu"])
        )
        for zone in ("A", "B"):
            previous = getattr(self, f"max_temp_{zone}")
            setattr(
                self,
                f"max_temp_{zone}",
                row[f"temp_{zone}"]
                if previous is None
                else max(previous, row[f"temp_{zone}"]),
            )
        for key in FAULT_KEYS:
            if row[key]:
                self.fault_counts[key] += 1
                if self.first_fault_tick[key] is None:
                    self.first_fault_tick[key] = int(row["tick"])

    def energy(self, initial_backlog: float = 0.0) -> dict:
        available = self.work_arrived_kwh + initial_backlog
        return {
            "duration_hours": self.duration_hours,
            "electricity_cost_usd": self.electricity_cost_usd,
            "grid_import_mwh": self.grid_import_mwh,
            "battery_throughput_mwh": self.battery_throughput_mwh,
            "work_arrived_kwh": self.work_arrived_kwh,
            "work_served_kwh": self.work_served_kwh,
            "backlog_final_kwh": self.backlog_final_kwh,
            "work_served_fraction_of_available": self.work_served_kwh / available
            if available > 0
            else None,
            "work_conservation_error_kwh": available
            - self.work_served_kwh
            - self.backlog_final_kwh,
        }

    def faults(self) -> dict:
        return {
            "counts": self.fault_counts,
            "first_tick": self.first_fault_tick,
            "max_abs_frequency_deviation_hz": self.max_abs_freq_dev_hz,
            "minimum_voltage_pu": self.min_voltage_pu,
            "maximum_temp_A_C": self.max_temp_A,
            "maximum_temp_B_C": self.max_temp_B,
        }


def service_summary(totals, macros, initial, terminal, panel, terminated, truncated):
    expected_substeps = panel["macro_seconds"] // panel["dt_seconds"]
    full_windows = sum(int(row["sub_steps_run"] == expected_substeps) for row in macros)
    complete = bool(
        totals.ticks == panel["service_fast_ticks"]
        and len(macros) == panel["service_macro_calls"]
        and full_windows == panel["service_macro_calls"]
        and truncated
        and not terminated
        and not any(totals.fault_counts.values())
    )
    metrics = {
        "tracking_rmse_kw": math.sqrt(totals.error_sq_sum / totals.ticks)
        if totals.ticks
        else None,
        "regulation_revenue_usd": float(
            sum(row["regulation_revenue"] for row in macros)
        ),
        "macro_mean_reward": float(
            sum(row["macro_reward"] for row in macros) / len(macros)
        )
        if macros
        else None,
        "low_mean_reward": totals.reward_sum / totals.ticks if totals.ticks else None,
        "thermal_warning_fraction": totals.warning_ticks / totals.ticks
        if totals.ticks
        else None,
        "bess_aging_x1e4": (
            terminal["bess_age_fraction"] - initial["bess_age_fraction"]
        )
        * 1e4,
        "survived": int(complete),
    }
    energy = totals.energy(initial["backlog_kwh"])
    gate = {
        "physical_completion": complete,
        "service_end_soc": terminal["soc"] >= SERVICE_GATE_SPEC["min_service_end_soc"],
        "service_end_backlog": (
            terminal["backlog_kwh"]
            <= SERVICE_GATE_SPEC["max_backlog_fraction_of_arrived_work"]
            * totals.work_arrived_kwh
        ),
        "thermal_warning_fraction": (
            totals.ticks > 0
            and metrics["thermal_warning_fraction"]
            <= SERVICE_GATE_SPEC["max_thermal_warning_fraction"]
        ),
        "work_conservation": (
            abs(energy["work_conservation_error_kwh"])
            <= SERVICE_GATE_SPEC["work_conservation_tolerance_kwh"]
        ),
    }
    committed_mwh = sum(row["committed_mw"] * row["duration_hours"] for row in macros)
    return {
        "complete": complete,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "expected_fast_ticks": panel["service_fast_ticks"],
        "observed_fast_ticks": totals.ticks,
        "expected_macro_calls": panel["service_macro_calls"],
        "observed_macro_calls": len(macros),
        "full_macro_windows": full_windows,
        "observed_service_metrics": metrics,
        "table1_service_metrics": {
            **{key: metrics[key] if complete else None for key in SERVICE_METRIC_KEYS},
            "survived": int(complete),
        },
        "initial_state": initial,
        "service_end_state": terminal,
        "thermal_warning_ticks": totals.warning_ticks,
        "energy": energy,
        "faults": totals.faults(),
        "supplemental_service_gate": {
            "checks": gate,
            "qualified": all(gate.values()),
            "contract": SERVICE_GATE_SPEC,
        },
        "descriptive_market_metrics_not_gates": {
            "committed_mwh": committed_mwh,
            "mean_committed_mw": committed_mwh / totals.duration_hours
            if totals.duration_hours
            else None,
            "bid_acceptance_fraction": (
                sum(float(row["bid_accepted"]) for row in macros) / len(macros)
                if macros
                else None
            ),
        },
    }


def _validate_environment(env, protocol) -> dict:
    fast = env._fast_env
    panel = protocol["panel"]
    if (
        fast.simulation_mode != "physical_v1"
        or fast._dt != panel["dt_seconds"]
        or fast._episode_ticks != panel["service_fast_ticks"]
        or env._substeps * fast._dt != panel["macro_seconds"]
        or env._episode_macro_ticks != panel["service_macro_calls"]
        or fast._plant_overrides
        or fast._thermal_overrides
        or fast._facility_cap_kw != 250000
        or not fast._weather_driven
        or not fast._weather.source.startswith("real:NYC.csv:calendar_climatology_")
        or fast._grid._market.zone_csv != "NYC"
        or fast._rcfg["T_warn_A"] != 33.0
        or fast._rcfg["T_warn_B"] != 33.0
        or type(fast._bess).__name__ != "_SimpleBESSModel"
        or type(fast._elec).__name__ != "DatacenterElectrical"
    ):
        raise ValueError("Runtime differs from the native default physical contract")
    return {
        "bess_backend": type(fast._bess).__name__,
        "electrical_backend": type(fast._elec).__name__,
        "weather_source": fast._weather.source,
        "noise_policy": (
            "Unmodified native stochastic RegD, market, frequency and workload "
            "behavior; environment reset seed recorded."
        ),
        "known_regd_and_voltage_limits": (
            "Inherited unchanged; no correction or noise disabling in this evaluator."
        ),
        "global_config": fast._gcfg,
        "scenario_config": fast._scfg,
        "reward_config": fast._rcfg,
    }


def _runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "gymnasium", "stable-baselines3")
        },
        "inference_device": "cpu",
        "torch_threads": 1,
    }


def run_case(protocol: dict, policy, seed: int, destination) -> dict:
    """Run one registered case, never overwrite a case or retry a failed attempt."""
    directory = Path(destination).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError(
            "Case outputs must remain inside this separate study directory"
        )
    directory.mkdir(parents=True, exist_ok=False)
    traces = {
        name: _CsvTrace(directory / name)
        for name in (
            "service_hardware.csv.gz",
            "service_macro.csv.gz",
            "recovery_hardware.csv.gz",
        )
    }
    env = None
    service = None
    totals, recovery_totals = HardwareTotals(), HardwareTotals()
    macro_rows = []
    phase = "preflight"
    record = {
        "schema": SCHEMA,
        "registration_id": protocol.get("registration_id"),
        "policy": policy,
        "seed": seed,
        "training_performed": False,
        "llm_calls_performed": False,
    }
    try:
        provenance = verify_protocol(protocol, policy, seed)
        row = provenance["row"]
        record.update(
            policy=row,
            provenance=provenance,
            panel=protocol["panel"],
            runtime=_runtime_versions(),
        )
        _write_json(directory / "attempt.json", record)
        import torch

        from baselines.rule_based_macro import RuleBasedMacroController
        from c2g_env import C2GMacroEnv

        torch.set_num_threads(1)
        ops = load_recovery_ops(protocol["recovery_ops"])
        env = C2GMacroEnv(
            scenario="default",
            config_path=protocol["config"]["path"],
            simulation_mode="physical_v1",
            start_day=0,
            regd_load_bias=0.0,
        )
        observation, reset_info = env.reset(seed=seed)
        fast = env._fast_env
        record["environment"] = _validate_environment(env, protocol)
        record["reset_info"] = reset_info
        record["hardware_reset_info"] = env._last_hw_reset_info
        record["initial_macro_observation"] = observation.tolist()
        record["initial_hardware_observation"] = env._last_hw_obs.tolist()
        initial = ops.plant_state(fast)
        record["reset_plant_state"] = initial
        low = _hardware_controller(row["hardware"], fast, seed, protocol["models"])
        macro = (
            _load_model("high", protocol["models"]["high"])
            if row["macro"] == "sac_macro"
            else RuleBasedMacroController()
        )
        _verify_import_origins(protocol)
        macro_calls = 0

        def inner_action(obs, _macro_action):
            action, _ = low.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise ValueError("Low policy produced an invalid action")
            return action

        def collect(pre, action, post, reward, done, info):
            result = hardware_row(
                pre, action, post, reward, info, fast, "service", macro_calls
            )
            traces["service_hardware.csv.gz"].write(result)
            totals.add(result)

        env._inner_action_fn = inner_action
        env._sub_step_callback = collect
        phase = "service"
        terminated = truncated = False
        last_macro_info = None
        while not (terminated or truncated):
            if macro_calls >= protocol["panel"]["service_macro_calls"]:
                raise AssertionError(
                    "Native service exceeded the registered macro horizon"
                )
            pre = observation.copy()
            action, _ = macro.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("Macro policy produced an invalid action")
            macro_calls += 1
            observation, reward, terminated, truncated, info = env.step(action)
            last_macro_info = info
            result = {
                **{
                    key: value
                    for key, value in info.items()
                    if key != "last_inner_info"
                },
                "phase": "service",
                "macro_call": macro_calls,
                "observation_before": pre.tolist(),
                "observation_after": observation.tolist(),
                "requested_action": action.tolist(),
                "macro_reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            traces["service_macro.csv.gz"].write(result)
            if not all(
                math.isfinite(float(result[key]))
                for key in (
                    "macro_reward",
                    "regulation_revenue",
                    "electricity_cost",
                    "duration_hours",
                    "tracking_error_sq_sum",
                )
            ):
                raise ValueError("Nonfinite macro settlement")
            macro_rows.append(result)
        terminal = ops.plant_state(fast)
        service = service_summary(
            totals,
            macro_rows,
            initial,
            terminal,
            protocol["panel"],
            terminated,
            truncated,
        )
        record["service"] = service
        if not math.isclose(
            totals.electricity_cost_usd,
            sum(item["electricity_cost"] for item in macro_rows),
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise AssertionError("Hardware energy cost and macro settlement disagree")
        if not math.isclose(
            totals.error_sq_sum,
            sum(item["tracking_error_sq_sum"] for item in macro_rows),
            rel_tol=1e-10,
            abs_tol=1e-5,
        ):
            raise AssertionError("Macro and metered fast tracking totals disagree")
        if (
            abs(service["energy"]["work_conservation_error_kwh"])
            > RECOVERY_SPEC["work_conservation_tolerance_kwh"]
        ):
            raise AssertionError("Service work is not conserved")
        traces["service_hardware.csv.gz"].close()
        traces["service_macro.csv.gz"].close()
        recovery = {
            "performed": False,
            "complete": False,
            "closed": False,
            "reason": "Service did not finish without physical termination",
        }
        if service["complete"]:
            phase = "recovery"
            cell = {"start_day": 0, "regd_load_bias": 0.0, "seed": seed}
            extension = ops.begin_recovery(
                fast, cell, last_macro_info["last_inner_info"], protocol["recovery"]
            )
            pre = env._last_hw_obs.copy()
            pre[17] = 0.0
            terminated = truncated = False
            for _ in range(extension["recovery_ticks"]):
                action = np.array(
                    [
                        RECOVERY_SPEC["throttle"],
                        RECOVERY_SPEC["pump"],
                        RECOVERY_SPEC["hvac"],
                        ops.battery_dispatch(fast._bess, initial["soc"]),
                    ],
                    dtype=np.float32,
                )
                post, reward, terminated, truncated, info = fast.step(action)
                result = hardware_row(
                    pre, action, post, reward, info, fast, "recovery", 0
                )
                traces["recovery_hardware.csv.gz"].write(result)
                recovery_totals.add(result)
                if (
                    result["committed_actual_mw"] != 0.0
                    or result["delta_p_demanded_kw"] != 0.0
                ):
                    raise AssertionError("Recovery inherited a regulation commitment")
                pre = post
                if terminated or truncated:
                    break
            complete = bool(
                recovery_totals.ticks == extension["recovery_ticks"]
                and truncated
                and not terminated
                and not any(recovery_totals.fault_counts.values())
            )
            final = ops.plant_state(fast)
            energy = recovery_totals.energy(terminal["backlog_kwh"])
            if (
                abs(energy["work_conservation_error_kwh"])
                > RECOVERY_SPEC["work_conservation_tolerance_kwh"]
            ):
                raise AssertionError("Recovery work is not conserved")
            recovery = {
                "performed": True,
                "complete": complete,
                "extension": extension,
                "expected_ticks": extension["recovery_ticks"],
                "observed_ticks": recovery_totals.ticks,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "terminal_state": final,
                **ops.closure(final, initial, protocol["recovery"], complete=complete),
                "energy": energy,
                "faults": recovery_totals.faults(),
                "regulation_revenue_usd": 0.0,
                "total_bess_age_fraction_delta": final["bess_age_fraction"]
                - initial["bess_age_fraction"],
            }
        record["supplemental_recovery"] = recovery
        recovery_cost = recovery_totals.electricity_cost_usd
        observed_cash = (
            service["observed_service_metrics"]["regulation_revenue_usd"]
            - totals.electricity_cost_usd
            - recovery_cost
        )
        arrived = totals.work_arrived_kwh + recovery_totals.work_arrived_kwh
        served = totals.work_served_kwh + recovery_totals.work_served_kwh
        record["supplemental_accounting"] = {
            "not_part_of_table1_service_columns": True,
            "observed_net_grid_cash_usd": observed_cash,
            "closed_recovery_net_grid_cash_usd": observed_cash
            if recovery["closed"]
            else None,
            "net_grid_cash_definition": (
                "service regulation revenue minus metered service and recovery "
                "electricity cost; not total business profit"
            ),
            "observed_battery_throughput_mwh": totals.battery_throughput_mwh
            + recovery_totals.battery_throughput_mwh,
            "observed_work_arrived_kwh": arrived,
            "observed_work_served_kwh": served,
            "observed_work_served_fraction": served / arrived if arrived > 0 else None,
            "service_and_recovery_closed": bool(
                service["complete"] and recovery["closed"]
            ),
            "all_supplemental_gates_pass": bool(
                service["supplemental_service_gate"]["qualified"] and recovery["closed"]
            ),
        }
        record["status"] = (
            "complete"
            if record["supplemental_accounting"]["all_supplemental_gates_pass"]
            else "service_complete_supplemental_gate_failed"
            if service["complete"]
            else "service_incomplete"
        )
        for trace in traces.values():
            trace.close()
        record["trace_hashes"] = {
            name: file_hash(trace.path)
            for name, trace in traces.items()
            if trace.path.exists()
        }
        _write_json(directory / "episode.json", record)
        return record
    except Exception as error:
        for trace in traces.values():
            trace.close()
        failure = {
            **record,
            "status": "execution_failure",
            "failed_phase": phase,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "observed_service_ticks": totals.ticks,
            "observed_macro_calls_returned": len(macro_rows),
            "observed_recovery_ticks": recovery_totals.ticks,
            "observed_service_faults": totals.faults(),
            "observed_recovery_faults": recovery_totals.faults(),
            "service": service,
            "trace_hashes": {
                name: file_hash(trace.path)
                for name, trace in traces.items()
                if trace.path.exists()
            },
            "no_retry_or_full_horizon_imputation": True,
        }
        _write_json(directory / "failure.json", failure)
        raise
    finally:
        for trace in traces.values():
            trace.close()
        if env is not None:
            env.close()
