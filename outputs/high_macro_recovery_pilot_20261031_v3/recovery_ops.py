"""Training-safe implementation of the previously approved physical recovery."""

from __future__ import annotations

import copy
import math

# Bootstrap the frozen physics path before simulator imports.
# isort: off
from study_common import identity
import numpy as np

from c2g_env.physics.bess import _SimpleBESSModel
from c2g_env.physics.macro_grid import MacroGridSignal
from c2g_env.physics.weather import WeatherLoader
# isort: on


def plant_state(fast) -> dict:
    return {
        "tick": fast._tick,
        "soc": fast._bess.soc_fraction,
        "nominal_stored_mwh": fast._bess.soc_fraction * fast._bess.E_NOM_MWH,
        "usable_energy_property_mwh": fast._bess.soc_mwh,
        "bess_age_fraction": fast._bess._age_frac,
        "backlog_kwh": fast._workload._backlog_kw * fast._dt / 3600,
        "temp_A": fast._thermal.temp_A,
        "temp_B": fast._thermal.temp_B,
        "reference_temp_A": fast._reference_thermal.temp_A,
        "reference_temp_B": fast._reference_thermal.temp_B,
    }


def continuation_identity(fast) -> str:
    return identity(
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


def begin_recovery(fast, cell: dict, last_info: dict, protocol: dict) -> dict:
    if (
        fast.simulation_mode != "physical_v1"
        or not last_info["truncated"]
        or last_info["terminated"]
        or not fast._episode_done
        or fast._tick != fast._episode_ticks
    ):
        raise ValueError("Recovery may continue only a healthy service time limit")
    recovery_ticks = protocol["duration_seconds"] / fast._dt
    if recovery_ticks < 1 or not float(recovery_ticks).is_integer():
        raise ValueError("Recovery must be an integral positive number of steps")
    service_ticks = fast._episode_ticks
    total_ticks = service_ticks + int(recovery_ticks)
    before = continuation_identity(fast)
    if not fast._weather_driven or fast._grid._market.zone_csv != "NYC":
        raise ValueError("This recovery contract requires default NYC replay")
    future_grid = MacroGridSignal(
        energy_dir=fast._gcfg["energy_dir"],
        dt_seconds=fast._dt,
        market="nyiso_nyc",
        timestamp_replay=True,
        start_day=cell["start_day"],
        episode_ticks=total_ticks,
        regd_load_bias=cell["regd_load_bias"],
        seed=cell["seed"],
    )
    future_weather = WeatherLoader(
        weather_dir=fast._gcfg["weather_dir"],
        market="nyiso_nyc",
        dt_seconds=fast._dt,
        replay_timestamps=future_grid.calendar_timestamps,
    )
    for future, original in (
        (future_grid.calendar_timestamps, fast._grid.calendar_timestamps),
        (future_grid._calendar_load_mw, fast._grid._calendar_load_mw),
        (future_weather._replay_temps, fast._weather._replay_temps),
    ):
        np.testing.assert_array_equal(future[: service_ticks + 1], original)
    service_load = fast._grid._calendar_load_mw[:service_ticks]
    mean, peak = service_load.mean(), service_load.max()
    anomaly = np.clip(
        (future_grid._calendar_load_mw - mean) / max(peak - mean, 1e-9), -1, 1
    )
    np.testing.assert_array_equal(
        anomaly[: service_ticks + 1], fast._grid._load_anomaly
    )
    fast._grid.calendar_timestamps = future_grid.calendar_timestamps
    fast._grid._calendar_load_mw = future_grid._calendar_load_mw
    fast._grid._load_anomaly = anomaly
    fast._weather = future_weather
    fast.committed_mw = 0.0
    fast._grid.set_committed(0.0)
    fast._current_grid_state = {
        **fast._current_grid_state,
        "committed_mw": 0.0,
        "delta_p_kw": 0.0,
    }
    fast._episode_ticks = total_ticks
    fast._episode_done = False
    after = continuation_identity(fast)
    if before != after:
        raise AssertionError("Recovery extension changed physical or stochastic state")
    return {
        "service_ticks": service_ticks,
        "recovery_ticks": int(recovery_ticks),
        "state_digest_before": before,
        "state_digest_after": after,
        "state_preserved": True,
        "service_replay_prefix_preserved": True,
    }


def battery_dispatch(bess: _SimpleBESSModel, target_soc: float) -> float:
    if type(bess) is not _SimpleBESSModel:
        raise TypeError("Recovery requires the registered simple BESS backend")
    if not math.isfinite(target_soc) or not bess.SOC_MIN <= target_soc <= bess.SOC_MAX:
        raise ValueError("Invalid recovery SOC target")
    if abs(bess.soc_fraction - target_soc) <= 1e-10:
        return 0.0

    def next_soc(dispatch):
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
        "backlog_kwh": state["backlog_kwh"],
        "temperature_A_minus_reference_C": state["temp_A"] - state["reference_temp_A"],
        "temperature_B_minus_reference_C": state["temp_B"] - state["reference_temp_B"],
        "usable_energy_property_change_mwh": (
            state["usable_energy_property_mwh"] - initial["usable_energy_property_mwh"]
        ),
    }
    residuals = {key: float(value) for key, value in residuals.items()}
    if not all(math.isfinite(value) for value in residuals.values()):
        raise ValueError("Nonfinite recovery inventory")
    checks = {
        "soc_closed": abs(residuals["soc_error"]) <= protocol["soc_tolerance"],
        "work_closed": abs(residuals["backlog_kwh"])
        <= protocol["backlog_tolerance_kwh"],
        "thermal_A_closed": abs(residuals["temperature_A_minus_reference_C"])
        <= protocol["temperature_tolerance_C"],
        "thermal_B_closed": abs(residuals["temperature_B_minus_reference_C"])
        <= protocol["temperature_tolerance_C"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {**residuals, **checks, "closed": bool(complete and all(checks.values()))}
