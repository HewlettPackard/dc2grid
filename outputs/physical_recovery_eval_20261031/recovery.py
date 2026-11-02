"""Append a physical recovery phase to the frozen 24-hour service simulation."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
MATCHED = PROJECT / "outputs/matched_validation_20261031"
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_spec = importlib.util.spec_from_file_location(
    "frozen_matched_matrix", MATCHED / "run_matrix.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load the frozen matched-evaluation helpers")
matched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matched)
os.environ["MPLCONFIGDIR"] = str(ROOT / "plot-cache")
SOURCE = matched.SOURCE

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines.train_corrected_sac import accumulate, finish_episode, new_episode  # noqa: E402
from c2g_env import C2GMacroEnv  # noqa: E402
from c2g_env.experiments.identity import configuration_id  # noqa: E402
from c2g_env.physics.bess import _SimpleBESSModel  # noqa: E402
from c2g_env.physics.macro_grid import MacroGridSignal  # noqa: E402
from c2g_env.physics.weather import WeatherLoader  # noqa: E402


def verify_frozen_imports() -> None:
    for name, module in list(sys.modules.items()):
        if name in ("c2g_env", "baselines") or name.startswith(
            ("c2g_env.", "baselines.")
        ):
            filename = getattr(module, "__file__", None)
            if filename is not None and not Path(filename).resolve().is_relative_to(
                SOURCE
            ):
                raise RuntimeError(f"Module imported outside the frozen source: {name}")


verify_frozen_imports()

RECOVERY = {
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
ACTION_COLUMNS = ("throttle_batch", "pump_speed_A", "hvac_effort", "bess_dispatch")


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def plant_state(fast: Any) -> dict:
    return {
        "tick": fast._tick,
        "soc": fast._bess.soc_fraction,
        "nominal_stored_mwh": fast._bess.soc_fraction * fast._bess.E_NOM_MWH,
        "usable_energy_property_mwh": fast._bess.soc_mwh,
        "bess_capacity_mwh": fast._bess.E_NOM_MWH,
        "bess_age_fraction": fast._bess._age_frac,
        "backlog_kwh": fast._workload._backlog_kw * fast._dt / 3600,
        "temp_A": fast._thermal.temp_A,
        "temp_B": fast._thermal.temp_B,
        "reference_temp_A": fast._reference_thermal.temp_A,
        "reference_temp_B": fast._reference_thermal.temp_B,
    }


def continuation_identity(fast: Any) -> str:
    """Exclude only the time limit, replay arrays and deliberately expired contract."""
    return configuration_id(
        {
            "plant": plant_state(fast),
            "thermal": vars(fast._thermal),
            "reference_thermal": vars(fast._reference_thermal),
            "electrical": fast._elec.get_diagnostics(),
            "reference_electrical": fast._reference_elec.get_diagnostics(),
            "workload": {
                key: getattr(fast._workload, key)
                for key in (
                    "_tick",
                    "_backlog_kw",
                    "_delay_accum_kw_steps",
                    "_total_served_kw",
                )
            },
            "grid_tick": fast._grid._tick,
            "regd_state": fast._grid._regd_state,
            "regd_buffer": fast._grid._regd_buffer,
            "grid_frequency": fast._grid.f_grid,
            "rngs": {
                name: getattr(fast._grid, name).bit_generator.state
                for name in (
                    "_rng",
                    "_regd_rng",
                    "_market_rng",
                    "_bid_rng",
                    "_frequency_rng",
                )
            },
            "workload_rng": fast._workload._rng.bit_generator.state,
            "environment_rng": fast.np_random.bit_generator.state,
            "previous_controls": [fast._prev_throttle, fast._prev_pump_speed],
            "previous_regd": fast._prev_regd_signal,
        }
    )


def continuation_inputs(fast: Any, cell: dict, total_ticks: int) -> tuple[Any, Any]:
    grid = MacroGridSignal(
        energy_dir=fast._gcfg["energy_dir"],
        dt_seconds=fast._dt,
        market="nyiso_nyc",
        timestamp_replay=True,
        start_day=cell["start_day"],
        episode_ticks=total_ticks,
        regd_load_bias=cell["regd_load_bias"],
        seed=cell["seed"],
    )
    weather = WeatherLoader(
        weather_dir=fast._gcfg["weather_dir"],
        market="nyiso_nyc",
        dt_seconds=fast._dt,
        replay_timestamps=grid.calendar_timestamps,
    )
    if not fast._weather_driven or fast._grid._market.zone_csv != "NYC":
        raise ValueError(
            "This registered continuation supports only default NYC replay"
        )
    return grid, weather


def begin_recovery(fast: Any, cell: dict, last_info: dict, protocol: dict) -> dict:
    if (
        fast.simulation_mode != "physical_v1"
        or not last_info["truncated"]
        or last_info["terminated"]
        or not fast._episode_done
        or fast._tick != fast._episode_ticks
    ):
        raise ValueError(
            "Recovery requires a healthy, time-limit-truncated service episode"
        )
    recovery_ticks = protocol["duration_seconds"] / fast._dt
    if recovery_ticks < 1 or not float(recovery_ticks).is_integer():
        raise ValueError(
            "Recovery duration must be a positive integral number of ticks"
        )
    service_ticks = fast._episode_ticks
    total_ticks = service_ticks + int(recovery_ticks)
    before = continuation_identity(fast)
    future_grid, future_weather = continuation_inputs(fast, cell, total_ticks)
    np.testing.assert_array_equal(
        future_grid.calendar_timestamps[: service_ticks + 1],
        fast._grid.calendar_timestamps,
    )
    np.testing.assert_array_equal(
        future_grid._calendar_load_mw[: service_ticks + 1], fast._grid._calendar_load_mw
    )
    np.testing.assert_array_equal(
        future_weather._replay_temps[: service_ticks + 1], fast._weather._replay_temps
    )
    service_load = fast._grid._calendar_load_mw[:service_ticks]
    mean, peak = service_load.mean(), service_load.max()
    anomaly = np.clip(
        (future_grid._calendar_load_mw - mean) / max(peak - mean, 1e-9), -1, 1
    )
    np.testing.assert_array_equal(
        anomaly[: service_ticks + 1], fast._grid._load_anomaly
    )

    # Only append exogenous data: never replace the live grid/RNG or physical state.
    fast._grid.calendar_timestamps = future_grid.calendar_timestamps
    fast._grid._calendar_load_mw = future_grid._calendar_load_mw
    fast._grid._load_anomaly = anomaly
    fast._weather = future_weather
    fast.committed_mw = protocol["committed_mw"]
    fast._grid.set_committed(protocol["committed_mw"])
    fast._current_grid_state = {
        **fast._current_grid_state,
        "committed_mw": protocol["committed_mw"],
        "delta_p_kw": 0.0,
    }
    fast._episode_ticks = total_ticks
    fast._episode_done = False
    after = continuation_identity(fast)
    if before != after:
        raise AssertionError(
            "Appending recovery changed live plant or stochastic state"
        )
    return {
        "service_ticks": service_ticks,
        "recovery_ticks": int(recovery_ticks),
        "state_digest_before": before,
        "state_digest_after": after,
        "state_preserved": True,
        "service_replay_prefix_preserved": True,
        "bias_normalization_uses_service_period_only": True,
    }


def battery_dispatch(bess: _SimpleBESSModel, target_soc: float) -> float:
    if type(bess) is not _SimpleBESSModel:
        raise TypeError(
            "Recovery dispatch requires the registered simple battery backend"
        )
    if not math.isfinite(target_soc) or not bess.SOC_MIN <= target_soc <= bess.SOC_MAX:
        raise ValueError("Invalid recovery SOC target")
    if abs(bess.soc_fraction - target_soc) <= 1e-10:
        return 0.0

    def next_soc(dispatch: float) -> float:
        trial = copy.copy(bess)
        return trial.step(float(np.float32(dispatch)) * bess.P_MAX_MW)["soc_fraction"]

    low, high = -1.0, 1.0
    if next_soc(low) < target_soc:
        return low
    if next_soc(high) > target_soc:
        return high
    for _ in range(40):
        middle = (low + high) / 2
        if next_soc(middle) > target_soc:
            low = middle
        else:
            high = middle
    return float(np.float32((low + high) / 2))


def closure(state: dict, initial: dict, protocol: dict, *, complete: bool) -> dict:
    residuals = {
        "soc_error": state["soc"] - initial["soc"],
        "nominal_stored_energy_error_mwh": (
            state["nominal_stored_mwh"] - initial["nominal_stored_mwh"]
        ),
        "usable_energy_property_change_mwh": (
            state["usable_energy_property_mwh"] - initial["usable_energy_property_mwh"]
        ),
        "backlog_kwh": state["backlog_kwh"],
        "temperature_A_minus_reference_C": state["temp_A"] - state["reference_temp_A"],
        "temperature_B_minus_reference_C": state["temp_B"] - state["reference_temp_B"],
    }
    if not all(math.isfinite(value) for value in residuals.values()):
        raise ValueError("Non-finite recovery inventories")
    checks = {
        "soc_closed": abs(residuals["soc_error"]) <= protocol["soc_tolerance"],
        "work_closed": abs(residuals["backlog_kwh"])
        <= protocol["backlog_tolerance_kwh"],
        "thermal_A_closed": abs(residuals["temperature_A_minus_reference_C"])
        <= protocol["temperature_tolerance_C"],
        "thermal_B_closed": abs(residuals["temperature_B_minus_reference_C"])
        <= protocol["temperature_tolerance_C"],
    }
    return {
        **residuals,
        **checks,
        "closed": complete and all(checks.values()),
        "irreversible_age_not_reset_or_monetized_as_stored_energy": True,
    }


def energy_metrics(frame: pd.DataFrame) -> dict:
    hours = frame.duration_hours
    battery = frame.bess_actual_kw / 1000
    charge_mwh = float((-battery.clip(upper=0) * hours).sum())
    discharge_mwh = float((battery.clip(lower=0) * hours).sum())
    charge_cost = float((-battery.clip(upper=0) * hours * frame.lmp).sum())
    discharge_value = float((battery.clip(lower=0) * hours * frame.lmp).sum())
    return {
        "duration_hours": float(hours.sum()),
        "electricity_cost_usd": float(
            (frame.lmp * frame.p_grid_import_mw * hours).sum()
        ),
        "grid_import_mwh": float((frame.p_grid_import_mw * hours).sum()),
        "facility_energy_mwh": float((frame.p_facility_mw * hours).sum()),
        "battery_net_discharge_mwh": float((battery * hours).sum()),
        "battery_discharge_mwh": float((battery.clip(lower=0) * hours).sum()),
        "battery_charge_mwh": float((-battery.clip(upper=0) * hours).sum()),
        "battery_throughput_mwh": float((battery.abs() * hours).sum()),
        "backlog_exposure_kwh_hours": float((frame.backlog_work_kwh * hours).sum()),
        "max_backlog_kwh": float(frame.backlog_work_kwh.max()),
        "work_arrived_kwh": float(frame.work_arrived_kwh.sum()),
        "work_served_kwh": float(frame.work_served_kwh.sum()),
        "thermal_warning_ticks": int(frame.thermal_warning.sum()),
        "lmp_min_usd_mwh": float(frame.lmp.min()),
        "lmp_max_usd_mwh": float(frame.lmp.max()),
        "lmp_time_weighted_mean_usd_mwh": float(np.average(frame.lmp, weights=hours)),
        "battery_charge_electricity_cost_usd": charge_cost,
        "battery_discharge_avoided_electricity_cost_usd": discharge_value,
        "battery_charge_energy_weighted_price_usd_mwh": (
            charge_cost / charge_mwh if charge_mwh > 0 else None
        ),
        "battery_discharge_energy_weighted_price_usd_mwh": (
            discharge_value / discharge_mwh if discharge_mwh > 0 else None
        ),
    }


def service_summary(
    fast: Any,
    frame: pd.DataFrame,
    macros: pd.DataFrame,
    episode: dict,
    policy: dict,
    cell: dict,
    initial: dict,
    gate: dict,
) -> dict:
    matched.verify_physical_rows(frame)
    energy = energy_metrics(frame)
    np.testing.assert_allclose(
        energy["electricity_cost_usd"],
        macros.electricity_cost.sum(),
        atol=1e-6,
        rtol=1e-12,
    )
    episode["regulation_revenue"] = float(macros.regulation_revenue.sum())
    stats = finish_episode(episode, gate)
    gross = float((macros.rate * macros.committed_mw * macros.settlement_hours).sum())
    complete = stats["complete"]
    return {
        "policy_id": policy["id"],
        "cell": cell,
        "status": "complete" if complete else "terminated_or_incomplete",
        "complete": complete,
        "constraint_qualified": stats["qualified"],
        "prior_selection_qualified": policy["prior_selection_qualified"],
        "diagnostic_only": policy["diagnostic_only"],
        "expected_ticks": fast._episode_ticks,
        "observed_ticks": len(frame),
        "dt_seconds": fast._dt,
        "committed_max_mw": fast._committed_mw_max,
        "tracking_rmse_kw": stats["tracking_rmse_kw"] if complete else None,
        "normalized_rmse": float(np.sqrt(frame.normalized_error_sq.mean()))
        if complete
        else None,
        "hardware_mean_reward": stats["mean_step_reward"] if complete else None,
        "observed_hardware_metrics": stats,
        "observed_energy_metrics": energy,
        "revenue_usd": stats["regulation_revenue"] if complete else None,
        "electricity_cost_usd": energy["electricity_cost_usd"] if complete else None,
        "revenue_before_tracking_discount_usd": gross if complete else None,
        "tracking_discount_usd": gross - stats["regulation_revenue"]
        if complete
        else None,
        "mean_committed_mw": float(
            np.average(macros.committed_mw, weights=macros.duration_hours)
        ),
        "mean_paid_performance": float(
            np.average(macros.perf_score, weights=macros.duration_hours)
        ),
        "stress_threshold_mw": float(
            np.quantile(fast._grid._calendar_load_mw[:-1], 0.9)
        ),
        "annual_load_median_mw": float(fast._grid._load_median),
        "battery_throughput_mwh": energy["battery_throughput_mwh"],
        "battery_age_fraction_delta": fast._bess._age_frac
        - initial["bess_age_fraction"],
        "initial": initial,
        "terminal": plant_state(fast),
        "reward_components": {key: float(frame[key].mean()) for key in matched.REWARDS},
    }


def verify_service_replay(
    frame: pd.DataFrame, macros: pd.DataFrame, directory: Path
) -> dict:
    results = {}
    for name, produced in (("hardware.csv.gz", frame), ("macro.csv.gz", macros)):
        with gzip.open(directory / name, "rt") as handle:
            expected_csv = handle.read()
        columns = expected_csv.splitlines()[0].split(",")
        if produced[columns].to_csv(index=False) != expected_csv:
            raise AssertionError(
                f"Service trajectory differs from frozen validation: {name}"
            )
        results[name] = matched.file_hash(directory / name)
    return {
        "verified": True,
        "uncompressed_csv_exact_match": True,
        "original_trace_hashes": results,
    }


def run_episode(
    policy: dict, cell: dict, registration: dict, *, directory: Path
) -> str:
    from baselines.bang_bang import BangBangController
    from baselines.pid_controller import PIDController
    from baselines.rule_based_macro import RuleBasedMacroController
    from baselines.rule_based_mpc import RuleBasedController

    verify_frozen_imports()
    directory.mkdir(parents=True, exist_ok=False)
    hardware_rows: list[dict] = []
    recovery_rows: list[dict] = []
    macro_rows: list[dict] = []
    episode = new_episode()
    protocol = registration["recovery"]
    env = None
    succeeded = False

    def row(pre, post, info):
        fast = env._fast_env
        result = {
            **info,
            **fast._elec.get_diagnostics(),
            "committed_actual_mw": fast.committed_mw,
            "committed_observed_mw": float(pre[17]) * fast._committed_mw_max,
            "regd_observed": float(pre[6]),
            "p_base_norm": float(post[3]),
            "p_flex_norm": float(post[4]),
            "load_anomaly": float(fast._grid._load_anomaly[info["tick"] - 1]),
            "macro_index": len(macro_rows),
        }
        result["normalized_error_sq"] = (
            info["tracking_err_kw"]
            / max(fast.committed_mw * 1000, fast._tracking_floor_kw)
        ) ** 2
        return result

    def callback(pre, action, post, reward, done, info):
        hardware_rows.append(row(pre, post, info))
        accumulate(episode, reward, {**info, "condition": cell}, "low")

    try:
        env = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=registration["config_path"],
            sub_step_callback=callback,
            start_day=cell["start_day"],
            regd_load_bias=cell["regd_load_bias"],
        )
        observation, reset_info = env.reset(seed=cell["seed"])
        fast = env._fast_env
        initial = plant_state(fast)
        hardware = None
        if policy["hardware"] == "sac":
            hardware = matched.load_sac(policy["hardware_checkpoint"], 18)
        elif policy["hardware"] == "pid":
            hardware = PIDController()
        elif policy["hardware"] == "bang_bang":
            hardware = BangBangController()
        elif policy["hardware"] == "rule_based":
            hardware = RuleBasedController(
                committed_mw_max=fast._committed_mw_max,
                bess_p_max_mw=fast._bess.P_MAX_MW,
            )
        elif policy["hardware"] != "none":
            raise ValueError(f"Unknown hardware controller {policy['hardware']}")
        if hardware is not None:
            env._inner_action_fn = lambda obs, macro: hardware.predict(
                obs, deterministic=True
            )[0]
        if policy["macro"] == "sac_macro":
            macro = matched.load_sac(policy["macro_checkpoint"], 19)
            if (
                macro.c2g_context["inner_checkpoint"]["sha256"]
                != policy["hardware_checkpoint"]["sha256"]
            ):
                raise ValueError(
                    "Learned macro does not match its frozen hardware checkpoint"
                )
        elif policy["macro"] == "rule_macro":
            macro = RuleBasedMacroController()
        else:
            raise ValueError(f"Unknown macro controller {policy['macro']}")

        replay_directory = None
        if registration["stage"] == "development":
            replay_directory = MATCHED / "traces" / policy["id"] / cell["id"]
            original = json.loads((replay_directory / "episode.json").read_text())
            for name, digest in original["trace_hashes"].items():
                if matched.file_hash(replay_directory / name) != digest:
                    raise ValueError(
                        f"Frozen development trace changed: {replay_directory / name}"
                    )
            actions = read_frame(replay_directory / "hardware.csv.gz")[
                list(ACTION_COLUMNS)
            ].to_numpy(np.float32)
            env._inner_action_fn = lambda obs, macro: actions[fast._tick]

        while True:
            action = macro.predict(observation, deterministic=True)[0]
            observation, reward, terminated, truncated, info = env.step(action)
            macro_rows.append(
                {
                    **{
                        key: value
                        for key, value in info.items()
                        if key != "last_inner_info"
                    },
                    "macro_reward": reward,
                }
            )
            if terminated or truncated:
                break
        service_frame = pd.DataFrame(hardware_rows)
        macros = pd.DataFrame(macro_rows)
        service = service_summary(
            fast,
            service_frame,
            macros,
            episode,
            policy,
            cell,
            initial,
            registration["selection_gate"],
        )
        replay_check = (
            verify_service_replay(service_frame, macros, replay_directory)
            if replay_directory
            else {
                "verified": False,
                "reason": "Fresh policy inference, not validation replay",
            }
        )
        recovery = {
            "performed": False,
            "complete": False,
            "closed": False,
            "reason": "Service did not finish without a physical termination",
        }
        if service["complete"]:
            extension = begin_recovery(fast, cell, info["last_inner_info"], protocol)
            pre = env._last_hw_obs.copy()
            pre[17] = 0
            for _ in range(extension["recovery_ticks"]):
                action = np.array(
                    [
                        protocol["throttle"],
                        protocol["pump"],
                        protocol["hvac"],
                        battery_dispatch(fast._bess, initial["soc"]),
                    ],
                    dtype=np.float32,
                )
                post, reward, term, trunc, inner_info = fast.step(action)
                recovery_rows.append(
                    {
                        **row(pre, post, inner_info),
                        "reference_temp_A": fast._reference_thermal.temp_A,
                        "reference_temp_B": fast._reference_thermal.temp_B,
                    }
                )
                pre = post
                if term or trunc:
                    break
            recovery_frame = pd.DataFrame(recovery_rows)
            total = pd.concat([service_frame, recovery_frame], ignore_index=True)
            matched.verify_physical_rows(total)
            if not (recovery_frame.committed_actual_mw == 0).all():
                raise AssertionError(
                    "Recovery must not inherit a standing regulation contract"
                )
            if not (recovery_frame.delta_p_demanded_kw == 0).all():
                raise AssertionError(
                    "Recovery must have zero regulation dispatch demand"
                )
            complete = (
                len(recovery_rows) == extension["recovery_ticks"] and trunc and not term
            )
            final = plant_state(fast)
            recovery = {
                "performed": True,
                "complete": complete,
                "extension": extension,
                "expected_ticks": extension["recovery_ticks"],
                "observed_ticks": len(recovery_rows),
                "terminated": bool(term),
                "truncated": bool(trunc),
                "terminal": final,
                **closure(final, initial, protocol, complete=complete),
                "observed_energy_metrics": energy_metrics(recovery_frame),
                "total_observed_energy_metrics": energy_metrics(total),
                "regulation_revenue_usd": 0.0,
                "total_battery_age_fraction_delta": fast._bess._age_frac
                - initial["bess_age_fraction"],
                "work_conservation_error_kwh": float(
                    total.work_arrived_kwh.sum()
                    - total.work_served_kwh.sum()
                    - final["backlog_kwh"]
                ),
            }
            if (
                abs(recovery["work_conservation_error_kwh"])
                >= protocol["work_conservation_tolerance_kwh"]
            ):
                raise AssertionError(
                    "Work conservation failed across the service/recovery boundary"
                )
            recovery_frame.to_csv(
                directory / "recovery.csv.gz", index=False, compression="gzip"
            )
        service_frame.to_csv(
            directory / "hardware.csv.gz", index=False, compression="gzip"
        )
        macros.to_csv(directory / "macro.csv.gz", index=False, compression="gzip")
        result = {
            **service,
            "recovery": recovery,
            "overall_qualified": service["constraint_qualified"] and recovery["closed"],
            "service_replay_check": replay_check,
            "registration_id": registration["registration_id"],
            "reset_info": reset_info,
            "policy": policy,
            "trace_hashes": {
                name: matched.file_hash(directory / name)
                for name in ("hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz")
                if (directory / name).exists()
            },
        }
        matched.write_json(directory / "episode.json", result)
        succeeded = True
        return str(directory / "episode.json")
    finally:
        if not succeeded:
            error = sys.exc_info()[1]
            for name, rows in (("service", hardware_rows), ("recovery", recovery_rows)):
                if rows:
                    pd.DataFrame(rows).to_csv(
                        directory / f"partial_{name}.csv.gz",
                        index=False,
                        compression="gzip",
                    )
            matched.write_json(
                directory / "failure.json",
                {
                    "policy_id": policy["id"],
                    "cell": cell,
                    "status": "runtime_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        if env is not None:
            env.close()


def recovery_pair(a: dict, b: dict, protocol: dict) -> dict:
    ra, rb = a["recovery"], b["recovery"]
    full = a["complete"] and b["complete"] and ra["complete"] and rb["complete"]
    closed = full and ra["closed"] and rb["closed"]
    result = {
        "recovery_pair_complete": full,
        "recovery_pair_closed": closed,
        "a_overall_qualified": a["overall_qualified"],
        "b_overall_qualified": b["overall_qualified"],
        "cashflow_delta_after_closed_recovery_usd": None,
    }
    if not full:
        return {
            **result,
            "recovery_unavailable_reason": "No full 26-hour paired trajectory",
        }
    ea, eb = ra["observed_energy_metrics"], rb["observed_energy_metrics"]
    ta, tb = ra["total_observed_energy_metrics"], rb["total_observed_energy_metrics"]
    delta_cost = ta["electricity_cost_usd"] - tb["electricity_cost_usd"]
    cash = a["revenue_usd"] - b["revenue_usd"] - delta_cost
    throughput_delta = ta["battery_throughput_mwh"] - tb["battery_throughput_mwh"]
    result.update(
        {
            "recovery_electricity_cost_delta_usd": ea["electricity_cost_usd"]
            - eb["electricity_cost_usd"],
            "total_electricity_cost_delta_usd": delta_cost,
            "cashflow_delta_after_fixed_recovery_usd": cash,
            "cashflow_delta_after_closed_recovery_usd": cash if closed else None,
            "total_battery_throughput_delta_mwh": throughput_delta,
            "total_battery_age_fraction_delta": ra["total_battery_age_fraction_delta"]
            - rb["total_battery_age_fraction_delta"],
            "total_backlog_exposure_delta_kwh_hours": ta["backlog_exposure_kwh_hours"]
            - tb["backlog_exposure_kwh_hours"],
            "total_work_served_delta_kwh": ta["work_served_kwh"]
            - tb["work_served_kwh"],
            "terminal_usable_energy_property_delta_mwh": ra["terminal"][
                "usable_energy_property_mwh"
            ]
            - rb["terminal"]["usable_energy_property_mwh"],
            "wear_adjusted_cashflow_usd": {
                str(price): cash - price * throughput_delta if closed else None
                for price in protocol["wear_prices_usd_per_mwh"]
            },
            "wear_break_even_usd_per_mwh": cash / throughput_delta
            if closed and abs(throughput_delta) > 1e-12 and cash / throughput_delta >= 0
            else None,
            "wear_sensitivity_slope_mwh": -throughput_delta if closed else None,
            "business_profit_claim": False,
        }
    )
    return result
