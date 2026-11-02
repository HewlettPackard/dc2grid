"""Matched full-day development and locked evaluation with physical recovery."""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# isort: off
from common import (
    ROOT,
    assert_default_plant,
    cap_cpu,
    case_key,
    contracts,
    file_hash,
    write_json,
)
from controllers import CoolingFirstController, FullResidualMacro, ResidualController
from history_policy import (
    ACTION_HIGH,
    ACTION_LOW,
    FEATURE_CONTRACT,
    HistorySACController,
)

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.rule_based_macro import RuleBasedMacroController
from baselines.rule_based_mpc import RuleBasedController
from c2g_env import C2GMacroEnv
from macro_task import Ledger
from recovery_ops import battery_dispatch, begin_recovery, closure, plant_state
# isort: on


def make_hardware(policy: dict, protocol: dict):
    kind = policy["hardware"]
    if kind == "cooling_first":
        return CoolingFirstController(
            transformer_compensation=policy.get("transformer_compensation", True)
        )
    if kind == "pid":
        return PIDController()
    if kind == "rule":
        return RuleBasedController(committed_mw_max=30, bess_p_max_mw=5)
    if kind == "bang_bang":
        return BangBangController()
    if kind == "none":
        return None
    if kind not in ("old_sac", "residual_sac", "history_sac"):
        raise ValueError(f"Unknown hardware controller: {kind}")
    artifact = (
        protocol["hardware_reference"] if kind == "old_sac" else policy["checkpoint"]
    )
    if file_hash(artifact["path"]) != artifact["sha256"]:
        raise ValueError("Changed hardware checkpoint")
    model = SAC.load(artifact["path"], device="cpu")
    model.policy.set_training_mode(False)
    expected = (18,) if kind == "old_sac" else (24,)
    if model.observation_space.shape != expected:
        raise ValueError("Checkpoint observation contract mismatch")
    if kind == "old_sac":
        return model
    if kind == "history_sac":
        context = getattr(model, "c2g_context", {})
        if (
            context.get("feature_contract") != FEATURE_CONTRACT
            or model.action_space.shape != (3,)
            or not np.array_equal(model.action_space.low, ACTION_LOW)
            or not np.array_equal(model.action_space.high, ACTION_HIGH)
            or context.get("simulation_mode") != "physical_v1"
        ):
            raise ValueError("Checkpoint feature/action/physics identity mismatch")
        return HistorySACController(model)
    return ResidualController(
        model, transformer_compensation=policy.get("transformer_compensation", True)
    )


def grid_metrics(frame: pd.DataFrame, *, annual_median: float) -> dict:
    load = frame.grid_load_mw.to_numpy()
    reference = frame.p_reference_mw.to_numpy()
    actual = frame.p_facility_mw.to_numpy()
    relief = reference - actual
    threshold = float(np.quantile(load, 0.9))
    stress = load >= threshold
    weights = np.maximum(0, load - annual_median)
    weight_sum = float(weights.sum())
    stress_relief = float(relief[stress].mean())
    return {
        "stress_threshold_mw": threshold,
        "stress_facility_relief_mw": stress_relief,
        "LFR_pct": float(100 * relief[stress].sum() / reference[stress].sum()),
        "SPDR_pct": float(100 * stress_relief / (load + reference).max()),
        "LWCF_pct": float(100 * np.dot(weights, relief) / weight_sum / 30)
        if weight_sum > 0
        else None,
        "LWSR_pct": float(
            100 * np.dot(weights, relief) / np.dot(weights, load + reference)
        )
        if weight_sum > 0
        else None,
        "actual_facility_boundary_peak_reduction_mw": float(
            (load + reference).max() - (load + actual).max()
        ),
        "stress_pcc_relief_mw": float(
            (reference - frame.p_grid_import_mw.to_numpy())[stress].mean()
        ),
        "actual_pcc_boundary_peak_reduction_mw": float(
            (load + reference).max() - (load + frame.p_grid_import_mw.to_numpy()).max()
        ),
        "max_temperature_A_C": float(frame.temp_A.max()),
        "max_temperature_B_C": float(frame.temp_B.max()),
        "max_A_reference_heat_excess_gj": float(
            np.maximum(frame.temp_A - frame.reference_temp_A, 0).max() * 27
        ),
        "max_B_reference_heat_excess_gj": float(
            np.maximum(frame.temp_B - frame.reference_temp_B, 0).max() * 10
        ),
        "fraction_temp_B_above_32C": float((frame.temp_B > 32).mean()),
        "hvac_rail_fraction": float(
            ((frame.hvac_effort <= 1e-6) | (frame.hvac_effort >= 1 - 1e-6)).mean()
        ),
        "stress_requested_mw": float(frame.delta_p_demanded_kw[stress].mean() / 1000),
        "stress_battery_discharge_mw": float(
            frame.bess_actual_kw[stress].mean() / 1000
        ),
        "service_facility_energy_relief_mwh": float(
            np.dot(relief, frame.duration_hours)
        ),
        "negative_relief_outside_stress_mwh": float(
            np.dot(np.maximum(-relief[~stress], 0), frame.duration_hours[~stress])
        ),
    }


def run_case(policy: dict, cell: dict, protocol: dict, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    hardware_rows, recovery_rows, macro_rows = [], [], []
    service_ledger, recovery_ledger = Ledger(), Ledger()
    env = None
    started = time.monotonic()

    def record(pre, action, post, reward, done, info):
        fast = env._fast_env
        service_ledger.update(fast, info, protocol["normalization"])
        hardware_rows.append(
            {
                **info,
                **fast._elec.get_diagnostics(),
                "observed_committed_mw": float(pre[17]) * 30,
                "committed_mw": fast.committed_mw,
                "p_base_norm": float(post[3]),
                "p_flex_norm": float(post[4]),
                "observed_temp_B": float(pre[1]) * 35,
                "reference_temp_A": fast._reference_thermal.temp_A,
                "reference_temp_B": fast._reference_thermal.temp_B,
                "load_anomaly": float(fast._grid._load_anomaly[info["tick"] - 1]),
            }
        )

    try:
        hardware = make_hardware(policy, protocol)
        macro = (
            RuleBasedMacroController()
            if policy["macro"] == "rule"
            else FullResidualMacro()
        )
        if policy["macro"] not in ("rule", "full_residual"):
            raise ValueError("Unknown macro controller")
        env = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=protocol["config_path"],
            start_day=cell["start_day"],
            regd_load_bias=cell["regd_load_bias"],
            sub_step_callback=record,
        )
        observation, reset_info = env.reset(seed=cell["seed"])
        fast = env._fast_env
        assert_default_plant(fast)
        if fast._episode_ticks != 17280 or fast._dt != 5:
            raise ValueError("Expected an unchanged 24-hour, 5-second service episode")
        initial = plant_state(fast)
        if hardware is not None:
            if hasattr(hardware, "reset"):
                hardware.reset()
            env._inner_action_fn = lambda obs, _: hardware.predict(
                obs, deterministic=True
            )[0]
        while True:
            action, _ = macro.predict(observation, deterministic=True)
            observation, _, term, trunc, macro_info = env.step(action)
            macro_rows.append(
                {k: v for k, v in macro_info.items() if k != "last_inner_info"}
            )
            if policy["macro"] == "full_residual" and (
                macro_info["bid_mw"] > macro_info["residual_mw"]
                or macro_info["bid_price"] != 0
            ):
                raise AssertionError("Price-taking quantity bound failed")
            if term or trunc:
                break
        service_terminal = plant_state(fast)
        service = service_ledger.report()
        frame = pd.DataFrame(hardware_rows)
        macros = pd.DataFrame(macro_rows)
        np.testing.assert_allclose(
            macros.electricity_cost.sum(),
            service["electricity_usd"],
            atol=1e-6,
            rtol=1e-12,
        )
        complete = bool(trunc and not term and service["steps"] == 17280)
        gate = protocol["service_gate"]
        service_checks = {
            "complete": complete,
            "terminal_soc": service_terminal["soc"] >= gate["min_terminal_soc"],
            "terminal_work": service["backlog_kwh"]
            <= gate["max_backlog_fraction"] * service["work_arrived_kwh"] + 1e-6,
            "thermal_warnings": service["thermal_warning_fraction"]
            <= gate["max_thermal_warning_fraction"],
        }
        recovered = {"performed": False, "complete": False, "closed": False}
        if complete:
            recovery_ledger.initial_backlog_kwh = service["backlog_kwh"]
            extension = begin_recovery(
                fast, cell, macro_info["last_inner_info"], protocol["recovery"]
            )
            for _ in range(extension["recovery_ticks"]):
                action = np.array(
                    [1, 0.7, 0.7, battery_dispatch(fast._bess, initial["soc"])],
                    dtype=np.float32,
                )
                _, _, rterm, rtrunc, info = fast.step(action)
                recovery_ledger.update(fast, info, protocol["normalization"])
                recovery_rows.append(
                    {
                        **info,
                        **fast._elec.get_diagnostics(),
                        "reference_temp_A": fast._reference_thermal.temp_A,
                        "reference_temp_B": fast._reference_thermal.temp_B,
                    }
                )
                if info["delta_p_demanded_kw"] != 0 or fast.committed_mw != 0:
                    raise AssertionError("Recovery inherited a regulation obligation")
                if rterm or rtrunc:
                    break
            recovery_complete = bool(
                rtrunc
                and not rterm
                and recovery_ledger.steps == extension["recovery_ticks"]
            )
            recovered = {
                "performed": True,
                "complete": recovery_complete,
                "ledger": recovery_ledger.report(),
                "extension": extension,
                **closure(
                    plant_state(fast),
                    initial,
                    protocol["recovery"],
                    complete=recovery_complete,
                ),
            }
        revenue = float(macros.regulation_revenue.sum())
        result = {
            "policy": policy,
            "cell": cell,
            "reset_info": reset_info,
            "complete": complete,
            "service_checks": service_checks,
            "qualified": bool(all(service_checks.values()) and recovered["closed"]),
            "initial": initial,
            "service_terminal": service_terminal,
            "service": service,
            "recovery": recovered,
            "grid": grid_metrics(frame, annual_median=fast._grid._load_median)
            if complete
            else None,
            "regulation_revenue_usd": revenue,
            "committed_mw_hours": float(
                np.dot(macros.committed_mw, macros.duration_hours)
            ),
            "performance_weighted_mw_hours": float(
                np.dot(macros.committed_mw * macros.perf_score, macros.duration_hours)
            ),
            "cashflow_26h_usd": revenue
            - service_ledger.electricity_usd
            - recovery_ledger.electricity_usd
            if complete and recovered["complete"]
            else None,
            "cash_vs_nominal_26h_usd": revenue
            + service_ledger.nominal_electricity_usd
            + recovery_ledger.nominal_electricity_usd
            - service_ledger.electricity_usd
            - recovery_ledger.electricity_usd
            if complete and recovered["complete"]
            else None,
            "elapsed_seconds": time.monotonic() - started,
            "code_hashes": protocol["code_hashes"],
        }
        result["cashflow_closed_recovery_usd"] = (
            result["cashflow_26h_usd"] if recovered["closed"] else None
        )
        fingerprints = {}
        for name, rows in (
            ("hardware.csv.gz", hardware_rows),
            ("macro.csv.gz", macro_rows),
            ("recovery.csv.gz", recovery_rows),
        ):
            if rows:
                path = destination / name
                pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
                fingerprints[name] = file_hash(path)
        result["trace_hashes"] = fingerprints
        write_json(destination / "episode.json", result)
        return result
    except Exception as error:
        write_json(
            destination / "failure.json",
            {
                "policy": policy,
                "cell": cell,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "observed_service_steps": len(hardware_rows),
            },
        )
        for name, rows in (
            ("partial_hardware.csv.gz", hardware_rows),
            ("partial_recovery.csv.gz", recovery_rows),
        ):
            if rows:
                pd.DataFrame(rows).to_csv(
                    destination / name, index=False, compression="gzip"
                )
        raise
    finally:
        if env is not None:
            env.close()


def initialize_worker(cpu_queue):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(cpu_queue.get())


def run_worker(policy, cell, protocol, destination):
    return run_case(policy, cell, protocol, Path(destination))


def run_panel(protocol: dict, policies: list[dict], cells: list[dict], output: Path):
    if any(policy["hardware"] == "history_sac" for policy in policies) and (
        "history_policy.py" not in protocol["code_hashes"]
    ):
        raise ValueError("Learned history policy source must be fingerprinted")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name, digest in protocol["code_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Execution source changed before panel start: {name}")
        shutil.copy2(ROOT / name, source / name)
    write_json(
        output / "manifest.json", {**protocol, "policies": policies, "cells": cells}
    )
    results, failures = [], []
    import os

    cpus = protocol.get(
        "evaluation_cpus", sorted(os.sched_getaffinity(0))[: protocol["workers"]]
    )
    if len(cpus) != protocol["workers"] or len(set(cpus)) != len(cpus):
        raise ValueError("Insufficient assigned CPU cores")
    context = multiprocessing.get_context("spawn")
    cpu_queue = context.Queue()
    for cpu in cpus:
        cpu_queue.put(cpu)
    with ProcessPoolExecutor(
        max_workers=len(cpus),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(cpu_queue,),
    ) as pool:
        jobs = {}
        for policy, cell in ((policy, cell) for policy in policies for cell in cells):
            destination = output / policy["id"] / case_key(cell)
            future = pool.submit(
                run_worker,
                policy,
                cell,
                protocol,
                str(destination),
            )
            jobs[future] = (policy["id"], case_key(cell))
        for future in as_completed(jobs):
            try:
                result = future.result()
                results.append(result)
                print(
                    jobs[future],
                    "qualified",
                    result["qualified"],
                    "RMSE",
                    result["service"]["tracking_rmse_kw"],
                    flush=True,
                )
            except Exception as error:
                failures.append({"case": jobs[future], "error": repr(error)})
                print("RUNTIME FAILURE", failures[-1], flush=True)
    summary = {"episodes": results, "runtime_failures": failures}
    cpu_queue.close()
    cpu_queue.join_thread()
    write_json(output / "summary.json", summary)
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluation cases failed; receipts retained"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = contracts()
    protocol["code_hashes"] = {
        name: file_hash(ROOT / name)
        for name in ("common.py", "controllers.py", "history_policy.py", "evaluate.py")
    }
    protocol["workers"] = args.workers
    policies = [
        {"id": name, "hardware": name, "macro": "rule"}
        for name in ("pid", "old_sac", "cooling_first")
    ]
    cells = protocol["development_cells"]
    if args.smoke:
        cells = [
            cell
            for cell in cells
            if cell["start_day"] in (38, 192) and cell["seed"] == 91001
        ]
    run_panel(protocol, policies, cells, args.output)


if __name__ == "__main__":
    main()
