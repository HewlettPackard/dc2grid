"""Versioned generic macro runner, checked against the immutable V1 evaluator."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

# isort: off
from loan_policy import ExpectedSurplusQuantity
from market_policy import make_macro, maximum_native_quantity
from study import write_json, file_hash, identity
from common import assert_default_plant
from evaluate import make_hardware, grid_metrics
from macro_task import Ledger
from recovery_ops import begin_recovery, battery_dispatch, closure, plant_state
from c2g_env import C2GMacroEnv
import numpy as np
import pandas as pd
# isort: on


def evaluate(policy, cell, base, destination, registration_id):
    if (
        cell not in base["development_cells"]
        and cell["start_day"] not in base["train_days"]
    ):
        raise ValueError("This development-only evaluator cannot open final dates")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    initial_protocol_identity = identity(base)
    hardware_rows, macro_rows, recovery_rows = [], [], []
    service, restoration = Ledger(), Ledger()
    core, signed_sum = None, 0.0
    start = time.monotonic()

    def row(pre, post, info):
        fast = core._fast_env
        return {
            **info,
            **fast._elec.get_diagnostics(),
            "observed_committed_mw": float(pre[17]) * 30,
            "committed_mw": fast.committed_mw,
            "observed_temp_B": float(pre[1]) * 35,
            "observed_regd_signal": float(pre[6]),
            "p_base_norm": float(post[3]),
            "p_flex_norm": float(post[4]),
            "reference_temp_A": fast._reference_thermal.temp_A,
            "reference_temp_B": fast._reference_thermal.temp_B,
            "load_anomaly": float(fast._grid._load_anomaly[info["tick"] - 1]),
        }

    def callback(pre, action, post, reward, done, info):
        nonlocal signed_sum
        signed_sum += float(pre[6])
        service.update(core._fast_env, info, base["normalization"])
        hardware_rows.append(row(pre, post, info))

    try:
        hardware = make_hardware(policy, base)
        macro = (
            ExpectedSurplusQuantity(policy, base)
            if policy["macro"] == "expected_surplus_quantity"
            else make_macro(policy, base)
        )
        core = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=base["config_path"],
            start_day=cell["start_day"],
            regd_load_bias=cell["regd_load_bias"],
            sub_step_callback=callback,
        )
        observation, reset_info = core.reset(seed=cell["seed"])
        fast = core._fast_env
        assert_default_plant(fast)
        if fast._episode_ticks != 17280 or fast._dt != 5:
            raise ValueError("The frozen service horizon changed")
        if core._rmcp_max != base["high_level"]["rmcp_normalizer"]:
            raise ValueError("Posted-price normalization changed")
        initial = plant_state(fast)
        if hardware is not None:
            if hasattr(hardware, "reset"):
                hardware.reset()
            core._inner_action_fn = lambda obs, _: hardware.predict(
                obs, deterministic=True
            )[0]
        maximum, delivered = 0.0, 0.0
        while True:
            before, before_count = signed_sum, service.steps
            action = macro.choose(observation.copy(), core._last_hw_obs.copy())
            observation, _, term, trunc, info = core.step(action)
            if policy["macro"] != "rule" and (
                info["bid_price"] != 0
                or info["bid_mw"] < 5
                or info["bid_mw"] > info["residual_mw"]
            ):
                raise AssertionError(
                    "The unchanged zero-ask capacity band was violated"
                )
            mean_signal = (signed_sum - before) / (service.steps - before_count)
            reference = maximum_native_quantity(info["residual_mw"])
            maximum += (reference if info["bid_accepted"] else 5) * info[
                "duration_hours"
            ]
            delivered += (
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
                    "signed_regd_mean": mean_signal,
                    "maximum_reference_offer_mw": reference,
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
                mean_signal,
            )
            if term or trunc:
                break
        terminal, ledger = plant_state(fast), service.report()
        complete = bool(trunc and not term and service.steps == 17280)
        if hasattr(macro, "delivered"):
            np.testing.assert_allclose(
                [macro.delivered, macro.maximum],
                [delivered, maximum],
                atol=1e-9,
                rtol=0,
            )
            if complete and macro.window != 96:
                raise AssertionError("Policy and physical macro clocks diverged")
        gate = base["service_gate"]
        service_checks = {
            "complete": complete,
            "terminal_soc": terminal["soc"] >= gate["min_terminal_soc"],
            "terminal_work": ledger["backlog_kwh"]
            <= gate["max_backlog_fraction"] * ledger["work_arrived_kwh"] + 1e-6,
            "thermal_warnings": ledger["thermal_warning_fraction"]
            <= gate["max_thermal_warning_fraction"],
        }
        recovered = {"performed": False, "complete": False, "closed": False}
        if complete:
            config = base["recovery"]
            restoration.initial_backlog_kwh = ledger["backlog_kwh"]
            extension = begin_recovery(fast, cell, info["last_inner_info"], config)
            if extension["recovery_ticks"] != 1440:
                raise ValueError("Recovery duration differs from the original protocol")
            pre = core._last_hw_obs.copy()
            pre[17] = 0
            for _ in range(1440):
                action = np.array(
                    [
                        config["throttle"],
                        config["pump"],
                        config["hvac"],
                        battery_dispatch(fast._bess, initial["soc"]),
                    ],
                    dtype=np.float32,
                )
                post, _, rterm, rtrunc, detail = fast.step(action)
                restoration.update(fast, detail, base["normalization"])
                recovery_rows.append(row(pre, post, detail))
                pre = post
                if fast.committed_mw != 0 or detail["delta_p_demanded_kw"] != 0:
                    raise AssertionError("Recovery contains an undisclosed obligation")
                if rterm or rtrunc:
                    break
            finished = bool(rtrunc and not rterm and restoration.steps == 1440)
            recovered = {
                "performed": True,
                "complete": finished,
                "extension": extension,
                "ledger": restoration.report(),
                **closure(plant_state(fast), initial, config, complete=finished),
            }
        frames, macros = pd.DataFrame(hardware_rows), pd.DataFrame(macro_rows)
        np.testing.assert_allclose(
            macros.electricity_cost.sum(),
            service.electricity_usd,
            atol=1e-6,
            rtol=1e-12,
        )
        revenue = float(macros.regulation_revenue.sum())
        cash = (
            revenue - service.electricity_usd - restoration.electricity_usd
            if (complete and recovered["complete"])
            else None
        )
        ratio = delivered / maximum
        result = {
            "study_registration_id": registration_id,
            "policy": policy,
            "cell": cell,
            "reset_info": reset_info,
            "initial": initial,
            "service_terminal": terminal,
            "service": ledger,
            "complete": complete,
            "recovery": recovered,
            "service_checks": service_checks,
            "qualified": bool(all(service_checks.values()) and recovered["closed"]),
            "grid": grid_metrics(frames, annual_median=fast._grid._load_median)
            if complete
            else None,
            "regulation_revenue_usd": revenue,
            "cashflow_26h_usd": cash,
            "cashflow_closed_recovery_usd": cash if recovered["closed"] else None,
            "delivered_service_ratio": ratio,
            "delivered_service_gate": ratio >= base["service_floor"],
            "performance_weighted_mw_hours": delivered,
            "maximum_reference_mw_hours": maximum,
            "committed_mw_hours": float(
                np.dot(macros.committed_mw, macros.duration_hours)
            ),
            "cash_vs_nominal_26h_usd": (
                cash
                + service.nominal_electricity_usd
                + restoration.nominal_electricity_usd
            )
            if cash is not None
            else None,
            "base_registration_id": base["registration_id"],
            "base_source_hashes": base["source_hashes"],
            "runner_sha256": file_hash(__file__),
            "elapsed_seconds": time.monotonic() - start,
        }
        hashes = {}
        for name, data in (
            ("hardware.csv.gz", hardware_rows),
            ("macro.csv.gz", macro_rows),
            ("recovery.csv.gz", recovery_rows),
        ):
            if data:
                path = destination / name
                pd.DataFrame(data).to_csv(path, index=False, compression="gzip")
                hashes[name] = file_hash(path)
        result["trace_hashes"] = hashes
        if identity(base) != initial_protocol_identity:
            raise AssertionError(
                "Episode construction mutated the frozen base protocol"
            )
        write_json(destination / "episode.json", result)
        return result
    except Exception as error:
        write_json(
            destination / "failure.json",
            {
                "policy": policy,
                "cell": cell,
                "error": repr(error),
                "observed_steps": len(hardware_rows),
                "traceback": traceback.format_exc(),
            },
        )
        if hardware_rows:
            pd.DataFrame(hardware_rows).to_csv(
                destination / "partial_hardware.csv.gz", index=False, compression="gzip"
            )
        if recovery_rows:
            pd.DataFrame(recovery_rows).to_csv(
                destination / "partial_recovery.csv.gz",
                index=False,
                compression="gzip",
            )
        raise
    finally:
        if core is not None:
            core.close()
