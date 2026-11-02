"""Shared 26-hour evaluation with frozen physics and causal macro hooks."""

from __future__ import annotations

import copy
import multiprocessing
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# isort: off
from study import (
    ROOT,
    cap_cpu,
    file_hash,
    identity,
    read_json,
    write_json,
    verify_registration,
    implementation_commit,
)
from market_policy import make_macro, maximum_native_quantity
from common import assert_default_plant, case_key
from evaluate import make_hardware, grid_metrics
from macro_task import Ledger
from recovery_ops import begin_recovery, battery_dispatch, closure, plant_state
from c2g_env import C2GMacroEnv
import numpy as np
import pandas as pd
# isort: on


def final_lock(protocol):
    path = ROOT / "final_lock.json"
    if not path.exists():
        raise RuntimeError(
            "Final outcomes remain sealed until an explicit candidate lock exists"
        )
    lock = read_json(path)
    body = {key: value for key, value in lock.items() if key != "lock_id"}
    if (
        identity(body) != lock["lock_id"]
        or lock["registration_id"] != protocol["registration_id"]
        or lock["cells"] != protocol["final_cells"]
    ):
        raise ValueError("Final lock identity, registration or conditions changed")
    return lock


def evaluate_case(protocol, policy, cell, directory):
    if any(
        cell["start_day"] == item["start_day"]
        for item in protocol.get("final_cells", [])
    ):
        lock = final_lock(protocol)
        if cell not in lock["cells"] or policy not in lock["policies"]:
            raise ValueError("This final policy or condition is not locked")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    hardware_rows, recovery_rows, macro_rows = [], [], []
    service, recovered_ledger = Ledger(), Ledger()
    core = None
    pre_regd_sum = 0.0
    started = time.monotonic()

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
        nonlocal pre_regd_sum
        pre_regd_sum += float(pre[6])
        service.update(core._fast_env, info, protocol["normalization"])
        hardware_rows.append(row(pre, post, info))

    try:
        hardware = make_hardware(policy, protocol)
        macro = make_macro(policy, protocol)
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
        if (
            "high_level" in protocol
            and protocol["high_level"]["rmcp_normalizer"] != core._rmcp_max
        ):
            raise ValueError(
                "Macro RMCP normalization differs from the frozen environment"
            )
        assert_default_plant(fast)
        if fast._episode_ticks != 17280 or fast._dt != 5:
            raise ValueError("Evaluation service horizon is not 24 hours")
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
            if policy["macro"] != "rule" and (
                info["bid_mw"] > info["residual_mw"]
                or info["bid_mw"] < 5
                or info["bid_price"] != 0
            ):
                raise AssertionError("Registered zero-ask quantity contract failed")
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
                    **{k: v for k, v in info.items() if k != "last_inner_info"},
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
        frame, macros = pd.DataFrame(hardware_rows), pd.DataFrame(macro_rows)
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
                raise ValueError(
                    "Recovery horizon differs from the audited 26-hour window"
                )
            pre = core._last_hw_obs.copy()
            pre[17] = 0
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
                rtrunc
                and not rterm
                and recovered_ledger.steps == extension["recovery_ticks"]
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
        revenue = float(macros.regulation_revenue.sum())
        cash = (
            revenue - service.electricity_usd - recovered_ledger.electricity_usd
            if complete and recovered["complete"]
            else None
        )
        exposure = delivered_capacity / maximum_capacity if maximum_capacity else None
        result = {
            "study_registration_id": protocol["registration_id"],
            "policy": policy,
            "cell": cell,
            "reset_info": reset_info,
            "complete": complete,
            "service_checks": checks,
            "qualified": bool(all(checks.values()) and recovered["closed"]),
            "initial": initial,
            "service_terminal": terminal,
            "service": ledger,
            "recovery": recovered,
            "grid": grid_metrics(frame, annual_median=fast._grid._load_median)
            if complete
            else None,
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
            "delivered_service_gate": bool(
                exposure is not None and exposure >= protocol["service_floor"]
            ),
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
                "observed_steps": len(hardware_rows),
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


def initialize_worker(queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(queue.get())


def run_panel(protocol, policies, cells, output, cpus, *, role):
    verify_registration(protocol)
    commit = implementation_commit(protocol)
    if not cpus or len(cpus) != len(set(cpus)):
        raise ValueError("Evaluation requires a nonempty set of distinct CPU cores")
    if len(policies) != len({policy["id"] for policy in policies}) or not policies:
        raise ValueError("Evaluation requires unique nonempty policy slots")
    if role not in ("training_smoke", "regression", "development", "final"):
        raise ValueError("Unknown panel role")
    if role != "training_smoke":
        allowed = {case_key(cell) for cell in protocol[f"{role}_cells"]}
        if len(cells) != len(allowed) or {case_key(cell) for cell in cells} != allowed:
            raise ValueError(
                "Panel does not contain the exact registered condition set"
            )
    elif any(cell["start_day"] not in protocol["train_days"] for cell in cells):
        raise ValueError("Smoke cells must use training support")
    output = Path(output)
    if role == "final":
        lock = final_lock(protocol)
        if policies != lock["policies"]:
            raise ValueError(
                "Final panel must execute the full locked admitted inventory"
            )
        opening = ROOT / "final_opening.json"
        if opening.exists():
            raise RuntimeError(
                "Final panel was already opened; explicit adjudication is required"
            )
        write_json(
            opening,
            {
                "lock_id": lock["lock_id"],
                "output": str(output),
                "opened_unix": time.time(),
                "policies": [policy["id"] for policy in policies],
            },
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        {
            "registration_id": protocol["registration_id"],
            "role": role,
            "policies": policies,
            "cells": cells,
            "cpus": cpus,
            "source_hashes": protocol["source_hashes"],
            "implementation_commit": commit,
        },
    )
    source = output / "source"
    source.mkdir()
    for path, digest in protocol["source_hashes"].items():
        path = Path(path)
        if path.is_relative_to(ROOT):
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if file_hash(target) != digest:
                raise AssertionError("Source snapshot changed during copy")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    results, failures = [], []
    with ProcessPoolExecutor(
        max_workers=len(cpus),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as pool:
        futures = {
            pool.submit(
                evaluate_case,
                copy.deepcopy(protocol),
                policy,
                cell,
                str(output / policy["id"] / case_key(cell)),
            ): (policy["id"], case_key(cell))
            for policy in policies
            for cell in cells
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(
                    futures[future],
                    result["qualified"],
                    result["service"]["tracking_rmse_kw"],
                    flush=True,
                )
            except Exception as error:
                failure = {"case": futures[future], "error": repr(error)}
                failures.append(failure)
                print("EVALUATION ERROR", failure, flush=True)
    queue.close()
    queue.join_thread()
    summary = {"episodes": results, "runtime_failures": failures}
    write_json(output / "summary.json", summary)
    verify_registration(protocol)
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluation runtime failures; receipts retained"
        )
    return summary
