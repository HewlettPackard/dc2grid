"""Matched physical episodes for explicitly labelled nonlearned headroom probes."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

# isort: off
from economy_context import assert_default_plant, file_hash, write_json
from controllers import ResidualController
from evaluate import grid_metrics, make_hardware
from macro_task import Ledger
from market_policy import make_macro, maximum_native_quantity
from recovery_ops import begin_recovery, battery_dispatch, closure, plant_state
from c2g_env import C2GMacroEnv
from c2g_env.obs_indices import Fast
import numpy as np
import pandas as pd
# isort: on


class ConstantResidual:
    def __init__(self, command, *, positive_regd_only: bool = False):
        self.command = np.asarray(command, dtype=np.float32)
        if (
            self.command.shape != (3,)
            or not np.isfinite(self.command).all()
            or np.any(np.abs(self.command) > 1)
        ):
            raise ValueError("Expected three finite residuals within [-1, 1]")
        self.positive_regd_only = positive_regd_only

    def predict(self, observation, deterministic=True):
        if np.shape(observation) != (24,) or not np.isfinite(observation).all():
            raise ValueError("Expected the analytic controller's 24 causal features")
        command = self.command.copy()
        if self.positive_regd_only and observation[Fast.REGD] < 0:
            command[2] = 0
        return command, None


def hardware_for(spec: dict, protocol: dict):
    if spec["hardware"] == "constant_residual":
        return ResidualController(
            ConstantResidual(
                spec["residual"],
                positive_regd_only=spec["positive_regd_only"],
            )
        )
    return make_hardware(spec, protocol)


def decomposition(frame: pd.DataFrame) -> dict:
    hours = frame.duration_hours.to_numpy()
    facility = (frame.p_reference_mw - frame.p_facility_mw).to_numpy()
    pcc = (frame.p_reference_mw - frame.p_grid_import_mw).to_numpy()
    battery = frame.bess_actual_kw.to_numpy() / 1000
    requested = frame.delta_p_demanded_kw.to_numpy() / 1000
    error = pcc - requested
    stress = frame.grid_load_mw.to_numpy() >= np.quantile(frame.grid_load_mw, 0.9)
    np.testing.assert_allclose(pcc, facility + battery, atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(
        error * 1000,
        frame.delta_p_actual_kw - frame.delta_p_demanded_kw,
        atol=1e-6,
        rtol=1e-10,
    )
    return {
        "stress_requested_mw": float(requested[stress].mean()),
        "stress_signed_overdelivery_mw": float(error[stress].mean()),
        "stress_battery_reallocation_mw": float(-battery[stress].mean()),
        "service_overdelivery_mwh": float(np.dot(np.maximum(error, 0), hours)),
        "service_underdelivery_mwh": float(np.dot(np.maximum(-error, 0), hours)),
        "nonstress_facility_relief_mw": float(facility[~stress].mean()),
        "nonstress_pcc_relief_mw": float(pcc[~stress].mean()),
        "service_electricity_saving_usd": float(
            np.dot(pcc * frame.lmp.to_numpy(), hours)
        ),
        "facility_equals_request_plus_error_minus_battery": True,
        "constant_efficiency_offsets_are_not_time_shifting": True,
    }


def evaluate_episode(
    protocol: dict,
    policy: dict,
    cell: dict,
    directory: Path,
    *,
    hardware_factory=hardware_for,
    macro_factory=make_macro,
) -> dict:
    if policy["macro"] not in ("rule", "full_residual"):
        raise ValueError("Expected an explicitly registered existing macro")
    directory.mkdir(parents=True, exist_ok=False)
    service, recovered_ledger = Ledger(), Ledger()
    hardware_rows, macro_rows, recovery_rows = [], [], []
    core = None
    pre_regd_sum = 0.0
    started = time.monotonic()

    def row(pre, post, info):
        fast = core._fast_env
        return {
            **info,
            **fast._elec.get_diagnostics(),
            "observed_committed_mw": float(pre[Fast.COMMITTED]) * 30,
            "committed_mw": fast.committed_mw,
            "observed_temp_B": float(pre[Fast.TEMP_B]) * 35,
            "observed_regd_signal": float(pre[Fast.REGD]),
            "p_base_norm": float(post[Fast.P_BASE]),
            "p_flex_norm": float(post[Fast.P_FLEX]),
            "reference_temp_A": fast._reference_thermal.temp_A,
            "reference_temp_B": fast._reference_thermal.temp_B,
            "load_anomaly": float(fast._grid._load_anomaly[info["tick"] - 1]),
        }

    def callback(pre, action, post, reward, done, info):
        nonlocal pre_regd_sum
        if (
            policy["hardware"] in ("cooling_first", "constant_residual")
            and action[0] != 1
        ):
            raise AssertionError("The new probes must retain full work service")
        pre_regd_sum += float(pre[Fast.REGD])
        service.update(core._fast_env, info, protocol["normalization"])
        hardware_rows.append(row(pre, post, info))

    try:
        hardware = hardware_factory(policy, protocol)
        macro = macro_factory(policy, protocol)
        core = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=protocol["config_path"],
            start_day=cell["start_day"],
            regd_load_bias=cell["regd_load_bias"],
            sub_step_callback=callback,
        )
        observation, reset_info = core.reset(seed=cell["seed"])
        fast = core._fast_env
        assert_default_plant(fast)
        if fast._episode_ticks != 17280 or fast._dt != 5:
            raise ValueError("Expected the unchanged 24-hour, five-second task")
        initial = plant_state(fast)
        if hardware is not None:
            if hasattr(hardware, "reset"):
                hardware.reset()
            core._inner_action_fn = lambda obs, _: hardware.predict(
                obs, deterministic=True
            )[0]
        maximum_capacity = delivered_capacity = 0.0
        while True:
            before_steps, before_signed = service.steps, pre_regd_sum
            action = macro.choose(observation.copy(), core._last_hw_obs.copy())
            observation, _, term, trunc, info = core.step(action)
            if policy["macro"] == "full_residual" and (
                info["bid_mw"] > info["residual_mw"]
                or info["bid_mw"] < 5
                or info["bid_price"] != 0
            ):
                raise AssertionError("The unchanged full-residual contract failed")
            signed_mean = (pre_regd_sum - before_signed) / (
                service.steps - before_steps
            )
            maximum = maximum_native_quantity(info["residual_mw"])
            reference = maximum if info["bid_accepted"] else 5.0
            maximum_capacity += reference * info["duration_hours"]
            delivered_capacity += (
                info["committed_mw"] * info["perf_score"] * info["duration_hours"]
            )
            macro_rows.append(
                {
                    **{
                        key: value
                        for key, value in info.items()
                        if key != "last_inner_info"
                    },
                    **macro.last_diagnostics,
                    "signed_regd_mean": signed_mean,
                    "maximum_reference_offer_mw": maximum,
                }
            )
            macro.observe(
                {
                    key: info[key]
                    for key in (
                        "residual_mw",
                        "bid_accepted",
                        "duration_hours",
                        "committed_mw",
                        "perf_score",
                    )
                },
                signed_mean,
            )
            if term or trunc:
                break
        terminal = plant_state(fast)
        ledger = service.report()
        frame = pd.DataFrame(hardware_rows)
        macros = pd.DataFrame(macro_rows)
        np.testing.assert_allclose(
            macros.electricity_cost.sum(),
            ledger["electricity_usd"],
            atol=1e-6,
            rtol=1e-12,
        )
        complete = bool(trunc and not term and service.steps == 17280)
        gate = protocol["service_gate"]
        checks = {
            "complete": complete,
            "terminal_soc": terminal["soc"] >= gate["min_terminal_soc"],
            "terminal_work": ledger["backlog_kwh"]
            <= gate["max_backlog_fraction"] * ledger["work_arrived_kwh"] + 1e-6,
            "thermal_warnings": ledger["thermal_warning_fraction"]
            <= gate["max_thermal_warning_fraction"],
        }
        recovered = {"performed": False, "complete": False, "closed": False}
        if complete:
            recovered_ledger.initial_backlog_kwh = ledger["backlog_kwh"]
            config = protocol["recovery"]
            extension = begin_recovery(fast, cell, info["last_inner_info"], config)
            if extension["recovery_ticks"] != 1440:
                raise ValueError("Expected the unchanged two-hour recovery")
            pre = core._last_hw_obs.copy()
            pre[Fast.COMMITTED] = 0
            for _ in range(extension["recovery_ticks"]):
                action = np.array(
                    [
                        config["throttle"],
                        config["pump"],
                        config["hvac"],
                        battery_dispatch(fast._bess, initial["soc"]),
                    ],
                    dtype=np.float32,
                )
                post, _, rterm, rtrunc, step_info = fast.step(action)
                recovered_ledger.update(fast, step_info, protocol["normalization"])
                recovery_rows.append(row(pre, post, step_info))
                pre = post
                if fast.committed_mw != 0 or step_info["delta_p_demanded_kw"] != 0:
                    raise AssertionError("Recovery inherited a regulation obligation")
                if rterm or rtrunc:
                    break
            recovery_complete = bool(
                not rterm and recovered_ledger.steps == extension["recovery_ticks"]
            )
            recovered = {
                "performed": True,
                "complete": recovery_complete,
                "extension": extension,
                "ledger": recovered_ledger.report(),
                **closure(
                    plant_state(fast), initial, config, complete=recovery_complete
                ),
            }
        if maximum_capacity <= 0:
            raise ValueError("Capacity-service denominator must be positive")
        revenue = float(macros.regulation_revenue.sum())
        cash = (
            revenue - service.electricity_usd - recovered_ledger.electricity_usd
            if complete and recovered["complete"]
            else None
        )
        exposure = delivered_capacity / maximum_capacity
        physical_qualified = bool(all(checks.values()) and recovered["closed"])
        floor_required = policy["macro"] == "full_residual"
        result = {
            "study_registration_id": protocol["registration_id"],
            "policy": policy,
            "cell": cell,
            "reset_info": reset_info,
            "complete": complete,
            "service_checks": checks,
            "physical_qualified": physical_qualified,
            "qualified": bool(
                physical_qualified
                and (not floor_required or exposure >= protocol["service_floor"])
            ),
            "delivered_service_floor_required": floor_required,
            "initial": initial,
            "service_terminal": terminal,
            "service": ledger,
            "recovery": recovered,
            "grid": grid_metrics(frame, annual_median=fast._grid._load_median)
            if complete
            else None,
            "decomposition": decomposition(frame) if complete else None,
            "regulation_revenue_usd": revenue,
            "cashflow_26h_usd": cash,
            "cashflow_closed_recovery_usd": cash if recovered["closed"] else None,
            "cash_vs_nominal_26h_usd": (
                cash
                + service.nominal_electricity_usd
                + recovered_ledger.nominal_electricity_usd
            )
            if cash is not None
            else None,
            "committed_mw_hours": float(
                np.dot(macros.committed_mw, macros.duration_hours)
            ),
            "performance_weighted_mw_hours": delivered_capacity,
            "maximum_reference_mw_hours": maximum_capacity,
            "delivered_service_ratio": exposure,
            "delivered_service_gate": bool(exposure >= protocol["service_floor"]),
            "elapsed_seconds": time.monotonic() - started,
            "source_hashes": protocol["source_hashes"],
        }
        fingerprints = {}
        for name, rows in (
            ("hardware.csv.gz", hardware_rows),
            ("macro.csv.gz", macro_rows),
            ("recovery.csv.gz", recovery_rows),
        ):
            if rows:
                path = directory / name
                pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
                fingerprints[name] = file_hash(path)
        result["trace_hashes"] = fingerprints
        write_json(directory / "episode.json", result)
        return result
    except Exception as error:
        write_json(
            directory / "failure.json",
            {
                "policy": policy,
                "cell": cell,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "observed_service_steps": len(hardware_rows),
                "observed_recovery_steps": len(recovery_rows),
            },
        )
        for name, rows in (
            ("partial_hardware.csv.gz", hardware_rows),
            ("partial_recovery.csv.gz", recovery_rows),
        ):
            if rows:
                pd.DataFrame(rows).to_csv(
                    directory / name, index=False, compression="gzip"
                )
        raise
    finally:
        if core is not None:
            core.close()
