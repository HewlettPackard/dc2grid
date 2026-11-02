"""Validation-only matched physical and economic controller evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]
SOURCE = PROJECT / "outputs/online_stability_pilot_20261031/source"
for key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLCONFIGDIR"] = str(OUTPUT / "plot-cache")
sys.path.insert(0, str(SOURCE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COMPONENTS = {
    "it_A": "p_it_A_mw",
    "it_B": "p_it_B_mw",
    "hvac": "p_hvac_mw",
    "pump": "p_pump_mw",
    "chiller": "p_chiller_mw",
    "ups_loss": "p_ups_loss_mw",
    "pdu_loss": "p_pdu_loss_mw",
    "transformer_loss": "p_xfmr_loss_mw",
    "auxiliary": "p_aux_mw",
}
EXOGENOUS = (
    "timestamp",
    "grid_load_mw",
    "regd_signal",
    "T_amb",
    "lmp",
    "load_anomaly",
    "work_arrived_kwh",
    "p_base_norm",
    "p_flex_norm",
    "is_spike",
)
BIAS_SHARED = tuple(key for key in EXOGENOUS if key != "regd_signal")
REWARDS = (
    "reward_throughput",
    "reward_tracking",
    "reward_thermal",
    "reward_soc",
    "reward_freq",
    "reward_volt",
    "reward_backlog",
)


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n"
    )
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_hashes():
    return {
        str(path.relative_to(SOURCE)): file_hash(path)
        for folder in ("c2g_env", "baselines", "evaluation", "scripts", "conf")
        for path in sorted((SOURCE / folder).rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml")
    }


def verify_registration(registration):
    from c2g_env.experiments.identity import configuration_id

    body = {k: v for k, v in registration.items() if k != "registration_id"}
    if configuration_id(body) != registration["registration_id"]:
        raise RuntimeError("Matrix registration content changed")
    if file_hash(__file__) != registration["runner_sha256"]:
        raise RuntimeError("Registered evaluation runner changed")
    if file_hash(OUTPUT / "policy_inventory.json") != registration["inventory_sha256"]:
        raise RuntimeError("Registered policy inventory changed")
    if (
        file_hash(OUTPUT / "accounting_review.json")
        != registration["accounting_review_sha256"]
    ):
        raise RuntimeError("Registered accounting review changed")
    if source_hashes() != registration["source_files"]:
        raise RuntimeError("Frozen evaluation source changed")
    for name, expected in registration["data_files"].items():
        if file_hash(SOURCE / name) != expected:
            raise RuntimeError(f"Data changed: {name}")


def register():
    import gymnasium
    import stable_baselines3
    import torch

    from c2g_env.experiments.identity import configuration_id

    target = OUTPUT / "registration.json"
    if target.exists():
        raise FileExistsError("Matrix is already registered")
    inventory_path = OUTPUT / "policy_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    audit = json.loads((OUTPUT / "policy_inventory_audit.json").read_text())
    review = json.loads((OUTPUT / "accounting_review.json").read_text())
    if audit["status"] != "passed" or not review["no_blocking_issues_remaining"]:
        raise RuntimeError("Independent inventory/accounting review is not cleared")
    policies, cells = inventory["policies"], inventory["cells"]
    if len({p["id"] for p in policies}) != len(policies):
        raise ValueError("Duplicate policy IDs")
    if len({c["id"] for c in cells}) != len(cells):
        raise ValueError("Duplicate cell IDs")
    expected = {
        (day, seed, bias)
        for day, seeds in [(196, [90001, 90002]), (256, [90003, 90004])]
        for seed in seeds
        for bias in (0.0, 0.5)
    }
    if {(c["start_day"], c["seed"], c["regd_load_bias"]) for c in cells} != expected:
        raise ValueError(
            "Matrix must contain the eight prespecified matched-bias cells"
        )
    if any(c["start_day"] in (207, 248, 249) for c in cells):
        raise ValueError("Held-out peak dates are prohibited")
    for policy in policies:
        if Path(policy["id"]).name != policy["id"]:
            raise ValueError("Invalid policy ID")
        if policy["available"]:
            for key in ("hardware_checkpoint", "macro_checkpoint"):
                artifact = policy[key]
                if artifact and file_hash(artifact["path"]) != artifact["sha256"]:
                    raise RuntimeError(f"Checkpoint changed: {policy['id']}")
    data_paths = [
        "data/processed/energy/NYC.csv",
        "data/processed/weather/NYC.csv",
        "data/processed/workload_traces/batch_v2023.csv",
        "data/processed/workload_traces/dlrm_v2025.csv",
        "data/processed/workload_traces/genai_v2026.csv",
    ]
    body = {
        "schema": "matched-controller-validation-v1",
        "inventory": inventory,
        "inventory_sha256": file_hash(inventory_path),
        "inventory_audit_sha256": file_hash(OUTPUT / "policy_inventory_audit.json"),
        "runner_sha256": file_hash(__file__),
        "accounting_review_sha256": file_hash(OUTPUT / "accounting_review.json"),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "source_root": str(SOURCE),
        "source_files": source_hashes(),
        "data_files": {name: file_hash(SOURCE / name) for name in data_paths},
        "planned_policy_cells": len(policies) * len(cells),
        "executable_policy_cells": sum(p["available"] for p in policies) * len(cells),
        "workers": 4,
        "worker_cpus": [47, 59, 5, 7],
        "native_threads": 1,
        "primary_contrasts": [
            "Every available policy versus rule macro + nominal fixed hardware",
            "Learned macro versus rule macro with the exact same hardware checkpoint",
            "Restored final hardware versus its parent's selected healthy hardware, "
            "when available",
            "Each policy beta=.5 versus its own beta=0 on the same day and seed",
        ],
        "economic_definition": (
            "Delta benchmark settlement revenue minus delta metered grid-import cost; "
            "before recovery, degradation valuation and workload-service valuation. "
            "Terminal battery, queue and thermal obligations reported separately."
        ),
        "recovery_rollout_performed": False,
        "gate_policy": (
            "Prior checkpoint eligibility, runtime completeness, and current cell "
            "qualification are separate axes"
        ),
        "missingness_policy": (
            "Unavailable policies and failed/terminated cells remain explicit; "
            "no prefix or survivor-only score"
        ),
        "scope": (
            "Selected-checkpoint validation inventory, not an independent "
            "algorithm ranking or held-out paper result"
        ),
        "settlement_scope": (
            "Benchmark price/quantity/performance settlement proxy; "
            "not a claim about actual ISO payments or system-wide avoided costs"
        ),
    }
    registration = {**body, "registration_id": configuration_id(body)}
    write_json(target, registration)
    print(
        "Registered:",
        registration["registration_id"],
        registration["executable_policy_cells"],
        "rollouts",
    )


def worker_init(cpus):
    import torch

    index = multiprocessing.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {cpus[index % len(cpus)]})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def load_sac(artifact, dimension):
    from stable_baselines3 import SAC

    from c2g_env.experiments.identity import check_policy_semantics

    if file_hash(artifact["path"]) != artifact["sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    model = SAC.load(artifact["path"], device="cpu")
    check_policy_semantics(model, "physical_v1")
    if model.observation_space.shape != (dimension,):
        raise ValueError(
            f"Unexpected observation dimension {model.observation_space.shape}"
        )
    if model.action_space.shape != ((4,) if dimension == 18 else (2,)):
        raise ValueError(f"Unexpected action dimension {model.action_space.shape}")
    return model


def run_episode(policy, cell, registration, *, config_path=None, directory=None):
    from baselines.bang_bang import BangBangController
    from baselines.pid_controller import PIDController
    from baselines.rule_based_macro import RuleBasedMacroController
    from baselines.rule_based_mpc import RuleBasedController
    from baselines.train_corrected_sac import accumulate, finish_episode, new_episode
    from c2g_env import C2GMacroEnv

    directory = directory or OUTPUT / "traces" / policy["id"] / cell["id"]
    directory.mkdir(parents=True, exist_ok=False)
    hardware_rows, macro_rows = [], []
    episode = new_episode()
    env = None
    succeeded = False

    def callback(pre, action, post, reward, done, info):
        fast = env._fast_env
        electrical = fast._elec.get_diagnostics()
        row = {
            **info,
            **electrical,
            "committed_actual_mw": fast.committed_mw,
            "committed_observed_mw": float(pre[17]) * fast._committed_mw_max,
            "regd_observed": float(pre[6]),
            "p_base_norm": float(post[3]),
            "p_flex_norm": float(post[4]),
            "load_anomaly": float(fast._grid._load_anomaly[info["tick"] - 1]),
            "macro_index": len(macro_rows),
        }
        row["normalized_error_sq"] = (
            info["tracking_err_kw"]
            / max(fast.committed_mw * 1000, fast._tracking_floor_kw)
        ) ** 2
        hardware_rows.append(row)
        accumulate(episode, reward, {**info, "condition": cell}, "low")

    try:
        env = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=config_path,
            sub_step_callback=callback,
            start_day=cell["start_day"],
            regd_load_bias=cell["regd_load_bias"],
        )
        observation, reset_info = env.reset(seed=cell["seed"])
        fast = env._fast_env
        initial = {
            "soc": fast._bess.soc_fraction,
            "bess_capacity_mwh": fast._bess.E_NOM_MWH,
            "bess_age_fraction": fast._bess._age_frac,
            "nominal_stored_mwh": fast._bess.soc_fraction * fast._bess.E_NOM_MWH,
            "usable_energy_property_mwh": fast._bess.soc_mwh,
            "temp_A": fast._thermal.temp_A,
            "temp_B": fast._thermal.temp_B,
        }
        hardware = None
        if policy["hardware"] == "sac":
            hardware = load_sac(policy["hardware_checkpoint"], 18)
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
            macro = load_sac(policy["macro_checkpoint"], 19)
            expected_inner = macro.c2g_context["inner_checkpoint"]["sha256"]
            if expected_inner != policy["hardware_checkpoint"]["sha256"]:
                raise ValueError(
                    "Learned macro is paired with a different hardware policy"
                )
        elif policy["macro"] == "rule_macro":
            macro = RuleBasedMacroController()
        else:
            raise ValueError(f"Unknown macro controller {policy['macro']}")
        done = False
        while not done:
            action = macro.predict(observation, deterministic=True)[0]
            observation, reward, terminated, truncated, info = env.step(action)
            macro_rows.append(
                {
                    **{k: v for k, v in info.items() if k != "last_inner_info"},
                    "macro_reward": reward,
                }
            )
            done = terminated or truncated
        hardware_frame = pd.DataFrame(hardware_rows)
        macro_frame = pd.DataFrame(macro_rows)
        verify_physical_rows(hardware_frame)
        observed_cost = float(
            (
                hardware_frame.lmp
                * hardware_frame.p_grid_import_mw
                * hardware_frame.duration_hours
            ).sum()
        )
        reported_cost = float(macro_frame.electricity_cost.sum())
        if not math.isclose(observed_cost, reported_cost, rel_tol=1e-12, abs_tol=1e-6):
            raise AssertionError(
                "Fast-interval and macro electricity settlement disagree"
            )
        episode["regulation_revenue"] = float(macro_frame.regulation_revenue.sum())
        stats = finish_episode(episode, registration["inventory"]["selection_gate"])
        complete = stats["complete"]
        hours = macro_frame.duration_hours
        gross = float(
            (
                macro_frame.rate
                * macro_frame.committed_mw
                * macro_frame.settlement_hours
            ).sum()
        )
        energy = float(
            (hardware_frame.bess_actual_kw * hardware_frame.duration_hours / 1000).sum()
        )
        summary = {
            "policy_id": policy["id"],
            "cell": cell,
            "status": "complete" if complete else "terminated_or_incomplete",
            "prior_selection_qualified": policy["prior_selection_qualified"],
            "diagnostic_only": policy["diagnostic_only"],
            "complete": complete,
            "constraint_qualified": stats["qualified"],
            "expected_ticks": fast._episode_ticks,
            "observed_ticks": len(hardware_frame),
            "dt_seconds": fast._dt,
            "committed_max_mw": fast._committed_mw_max,
            "tracking_rmse_kw": stats["tracking_rmse_kw"] if complete else None,
            "normalized_rmse": float(np.sqrt(hardware_frame.normalized_error_sq.mean()))
            if complete
            else None,
            "hardware_mean_reward": stats["mean_step_reward"] if complete else None,
            "macro_mean_reward": float(macro_frame.macro_reward.mean())
            if complete
            else None,
            "observed_hardware_metrics": stats,
            "observed_regulation_revenue_usd": stats["regulation_revenue"],
            "observed_electricity_cost_usd": observed_cost,
            "revenue_usd": stats["regulation_revenue"] if complete else None,
            "electricity_cost_usd": observed_cost if complete else None,
            "revenue_before_tracking_discount_usd": gross if complete else None,
            "tracking_discount_usd": gross - stats["regulation_revenue"]
            if complete
            else None,
            "mean_committed_mw": float(
                np.average(macro_frame.committed_mw, weights=hours)
            ),
            "mean_paid_performance": float(
                np.average(macro_frame.perf_score, weights=hours)
            ),
            "regd_clipping_fraction": float(
                hardware_frame.regd_signal.abs().ge(1).mean()
            ),
            "bid_acceptance_fraction": float(macro_frame.bid_accepted.mean()),
            "commitment_observation_mae_mw": float(
                (
                    hardware_frame.committed_actual_mw
                    - hardware_frame.committed_observed_mw
                )
                .abs()
                .mean()
            ),
            "stress_threshold_mw": float(
                np.quantile(fast._grid._calendar_load_mw[: fast._episode_ticks], 0.90)
            ),
            "annual_load_median_mw": float(fast._grid._load_median),
            "battery_net_discharge_mwh": energy,
            "battery_discharge_mwh": float(
                (
                    hardware_frame.bess_actual_kw.clip(lower=0)
                    * hardware_frame.duration_hours
                    / 1000
                ).sum()
            ),
            "battery_charge_mwh": float(
                (
                    -hardware_frame.bess_actual_kw.clip(upper=0)
                    * hardware_frame.duration_hours
                    / 1000
                ).sum()
            ),
            "battery_throughput_mwh": float(
                (
                    hardware_frame.bess_actual_kw.abs()
                    * hardware_frame.duration_hours
                    / 1000
                ).sum()
            ),
            "battery_age_fraction_delta": fast._bess._age_frac
            - initial["bess_age_fraction"],
            "terminal_battery_drawdown_mwh": (
                initial["nominal_stored_mwh"]
                - fast._bess.soc_fraction * fast._bess.E_NOM_MWH
            ),
            "max_backlog_kwh": float(hardware_frame.backlog_work_kwh.max()),
            "backlog_exposure_kwh_hours": float(
                (hardware_frame.backlog_work_kwh * hardware_frame.duration_hours).sum()
            ),
            "observed_lmp_min_usd_mwh": float(hardware_frame.lmp.min()),
            "observed_lmp_max_usd_mwh": float(hardware_frame.lmp.max()),
            "observed_lmp_mean_usd_mwh": float(hardware_frame.lmp.mean()),
            "initial": initial,
            "terminal": {
                "soc": fast._bess.soc_fraction,
                "nominal_stored_mwh": fast._bess.soc_fraction * fast._bess.E_NOM_MWH,
                "usable_energy_property_mwh": fast._bess.soc_mwh,
                "backlog_kwh": float(hardware_frame.backlog_work_kwh.iloc[-1]),
                "temp_A": fast._thermal.temp_A,
                "temp_B": fast._thermal.temp_B,
                "reference_temp_A": fast._reference_thermal.temp_A,
                "reference_temp_B": fast._reference_thermal.temp_B,
            },
            "reward_components": {
                key: float(hardware_frame[key].mean()) for key in REWARDS
            },
            "weather_source": fast._weather.source,
            "reset_info": reset_info,
            "policy": policy,
            "registration_id": registration["registration_id"],
            "recovery_rollout_performed": False,
        }
        hardware_frame.to_csv(
            directory / "hardware.csv.gz", index=False, compression="gzip"
        )
        macro_frame.to_csv(directory / "macro.csv.gz", index=False, compression="gzip")
        summary["trace_hashes"] = {
            name: file_hash(directory / name)
            for name in ("hardware.csv.gz", "macro.csv.gz")
        }
        write_json(directory / "episode.json", summary)
        succeeded = True
        return str(directory / "episode.json")
    finally:
        if not succeeded:
            error = sys.exc_info()[1]
            if hardware_rows:
                pd.DataFrame(hardware_rows).to_csv(
                    directory / "partial_hardware.csv.gz",
                    index=False,
                    compression="gzip",
                )
            write_json(
                directory / "failure.json",
                {
                    "policy_id": policy["id"],
                    "cell": cell,
                    "status": "runtime_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "observed_ticks": len(hardware_rows),
                },
            )
        if env is not None:
            env.close()


def verify_physical_rows(frame):
    components = frame[list(COMPONENTS.values())].sum(axis=1)
    np.testing.assert_allclose(components, frame.p_facility_mw, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(
        frame.p_grid_import_mw,
        frame.p_facility_mw - frame.bess_actual_kw / 1000,
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        frame[list(REWARDS)].sum(axis=1),
        frame.reward,
        rtol=1e-10,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        frame.work_arrived_kwh.cumsum() - frame.work_served_kwh.cumsum(),
        frame.backlog_work_kwh,
        rtol=1e-10,
        atol=1e-5,
    )


def pair_metrics(a, b, sa, sb, *, bias_pair=False, include_relief_ratios=True):
    keys = BIAS_SHARED if bias_pair else EXOGENOUS
    common = min(len(a), len(b))
    for key in keys:
        if not np.array_equal(a[key].to_numpy()[:common], b[key].to_numpy()[:common]):
            raise ValueError(f"Exogenous mismatch: {key}")
    if (
        sa["expected_ticks"] != sb["expected_ticks"]
        or sa["dt_seconds"] != sb["dt_seconds"]
    ):
        raise ValueError("Paired planned horizons differ")
    complete = (
        sa["complete"] and sb["complete"] and len(a) == len(b) == sa["expected_ticks"]
    )
    result = {
        "pair_complete": complete,
        "a_constraint_qualified": sa["constraint_qualified"],
        "b_constraint_qualified": sb["constraint_qualified"],
    }
    if not complete:
        return {
            **result,
            "unavailable_reason": (
                "No full-horizon comparison for incomplete or terminated episodes"
            ),
        }
    if bias_pair:
        np.testing.assert_allclose(
            a.regd_signal,
            np.clip(b.regd_signal + 0.5 * a.load_anomaly, -1, 1),
            rtol=0,
            atol=1e-12,
        )
    elif sb.get("policy_id") == "nominal":
        np.testing.assert_allclose(
            a.p_reference_mw,
            b.p_facility_mw,
            rtol=1e-12,
            atol=1e-8,
        )
    dt = sa["dt_seconds"] / 3600
    load = a.grid_load_mw.to_numpy()
    if sa["stress_threshold_mw"] != sb["stress_threshold_mw"]:
        raise ValueError("Different planned stress windows")
    stress = load >= sa["stress_threshold_mw"]
    weights = np.maximum(0, load - sa["annual_load_median_mw"])
    relief = b.p_facility_mw.to_numpy() - a.p_facility_mw.to_numpy()
    pcc_relief = b.p_grid_import_mw.to_numpy() - a.p_grid_import_mw.to_numpy()
    component_relief = {
        key: float((b[column].to_numpy() - a[column].to_numpy())[stress].mean())
        for key, column in COMPONENTS.items()
    }
    np.testing.assert_allclose(
        sum(component_relief.values()), relief[stress].mean(), atol=1e-8
    )
    battery_support = float(
        ((a.bess_actual_kw - b.bess_actual_kw) / 1000).to_numpy()[stress].mean()
    )
    np.testing.assert_allclose(
        relief[stress].mean() + battery_support, pcc_relief[stress].mean(), atol=1e-8
    )
    cost_components = {
        key: float((a.lmp * (a[column] - b[column])).sum() * dt)
        for key, column in COMPONENTS.items()
    }
    cost_components["battery_exchange"] = float(
        (a.lmp * (b.bess_actual_kw - a.bess_actual_kw) / 1000).sum() * dt
    )
    cost_delta = sa["electricity_cost_usd"] - sb["electricity_cost_usd"]
    np.testing.assert_allclose(sum(cost_components.values()), cost_delta, atol=1e-6)
    revenue_delta = sa["revenue_usd"] - sb["revenue_usd"]
    gross_delta = (
        sa["revenue_before_tracking_discount_usd"]
        - sb["revenue_before_tracking_discount_usd"]
    )
    discount_delta = sa["tracking_discount_usd"] - sb["tracking_discount_usd"]
    np.testing.assert_allclose(gross_delta - discount_delta, revenue_delta, atol=1e-6)
    stress_positive = np.maximum(relief[stress], 0).sum()
    result.update(
        stress_facility_relief_mw=float(relief[stress].mean()),
        stress_pcc_relief_mw=float(pcc_relief[stress].mean()),
        LFR_pct=float(
            100 * relief[stress].sum() / b.p_facility_mw.to_numpy()[stress].sum()
        ),
        SPDR_proxy_pct=float(
            100 * relief[stress].mean() / np.max(load + b.p_facility_mw)
        ),
        actual_system_facility_peak_reduction_mw=float(
            np.max(load + b.p_facility_mw) - np.max(load + a.p_facility_mw)
        ),
        actual_system_pcc_peak_reduction_mw=float(
            np.max(load + b.p_grid_import_mw) - np.max(load + a.p_grid_import_mw)
        ),
        LWSR_pct=float(
            100 * (weights * relief).sum() / (weights * (load + b.p_facility_mw)).sum()
        )
        if weights.sum() > 0
        else None,
        LWCF_pct=float(
            100 * (weights * relief).sum() / (sa["committed_max_mw"] * weights.sum())
        )
        if weights.sum() > 0
        else None,
        stress_weight_defined=bool(weights.sum() > 0),
        facility_relief_mwh=float(relief.sum() * dt),
        pcc_relief_mwh=float(pcc_relief.sum() * dt),
        stress_work_shift_kwh=float(
            (b.work_served_kwh - a.work_served_kwh).to_numpy()[stress].sum()
        ),
        nonstress_work_service_delta_kwh=float(
            (a.work_served_kwh - b.work_served_kwh).to_numpy()[~stress].sum()
        ),
        daily_work_service_delta_kwh=float(
            (a.work_served_kwh - b.work_served_kwh).sum()
        ),
        rebound_pct=float(100 * np.maximum(-relief[~stress], 0).sum() / stress_positive)
        if stress_positive > 0
        else None,
        rebound_defined=bool(stress_positive > 0),
        component_stress_relief_mw=component_relief,
        stress_battery_support_mw=battery_support,
        component_electricity_cost_delta_usd=cost_components,
        regulation_revenue_delta_usd=revenue_delta,
        electricity_cost_delta_usd=cost_delta,
        cashflow_delta_before_recovery_usd=revenue_delta - cost_delta,
        gross_contract_revenue_delta_usd=gross_delta,
        tracking_discount_delta_usd=discount_delta,
        mean_committed_mw_delta=sa["mean_committed_mw"] - sb["mean_committed_mw"],
        terminal_stored_energy_delta_mwh=sa["terminal"]["nominal_stored_mwh"]
        - sb["terminal"]["nominal_stored_mwh"],
        terminal_backlog_delta_kwh=sa["terminal"]["backlog_kwh"]
        - sb["terminal"]["backlog_kwh"],
        terminal_temp_A_delta_C=sa["terminal"]["temp_A"] - sb["terminal"]["temp_A"],
        terminal_temp_B_delta_C=sa["terminal"]["temp_B"] - sb["terminal"]["temp_B"],
        battery_throughput_delta_mwh=sa["battery_throughput_mwh"]
        - sb["battery_throughput_mwh"],
        battery_age_fraction_delta=sa["battery_age_fraction_delta"]
        - sb["battery_age_fraction_delta"],
        recovery_and_degradation_monetized=False,
    )
    if bias_pair:
        result["regulation_revenue_change_pct"] = (
            100 * (sa["revenue_usd"] / sb["revenue_usd"] - 1)
            if sb["revenue_usd"] != 0
            else None
        )
        result["revenue_ratio_defined"] = sb["revenue_usd"] != 0
    if bias_pair or not include_relief_ratios:
        for key in ("LFR_pct", "SPDR_proxy_pct", "LWSR_pct", "LWCF_pct"):
            result.pop(key)
        result["relief_ratio_reference"] = "not_applicable_to_this_contrast"
    else:
        result["relief_ratio_reference"] = "nominal_fixed_hardware"
    return result


@lru_cache(maxsize=64)
def load_episode(policy_id, cell_id):
    directory = OUTPUT / "traces" / policy_id / cell_id
    summary = json.loads((directory / "episode.json").read_text())
    frame = pd.read_csv(directory / "hardware.csv.gz", float_precision="round_trip")
    return frame, summary


def summarize(registration):
    inventory = registration["inventory"]
    policies = {p["id"]: p for p in inventory["policies"]}
    by_condition = {
        (c["start_day"], c["seed"], c["regd_load_bias"]): c for c in inventory["cells"]
    }
    summaries, comparisons = [], []
    for cell in sorted(
        inventory["cells"],
        key=lambda c: (c["start_day"], c["seed"], c["regd_load_bias"]),
    ):
        baseline, baseline_summary = load_episode("nominal", cell["id"])
        for policy in inventory["policies"]:
            directory = OUTPUT / "traces" / policy["id"] / cell["id"]
            if not policy["available"] or not (directory / "episode.json").exists():
                summaries.append(
                    {
                        "policy_id": policy["id"],
                        "cell_id": cell["id"],
                        "available": policy["available"],
                        "status": "unavailable"
                        if not policy["available"]
                        else "runtime_failure",
                        "reason": policy["unavailable_reason"]
                        if not policy["available"]
                        else "See retained failure record",
                    }
                )
                continue
            frame, summary = load_episode(policy["id"], cell["id"])
            summaries.append(summary)
            metrics = pair_metrics(frame, baseline, summary, baseline_summary)
            comparisons.append(
                {
                    "contrast": "vs_nominal",
                    "a": policy["id"],
                    "b": "nominal",
                    "cell": cell,
                    **metrics,
                }
            )
            reference = (
                policy["matched_rule_id"]
                if policy["macro"] == "sac_macro"
                else policy["parent_selected_low_id"]
            )
            if reference:
                if policies[reference]["available"]:
                    other, other_summary = load_episode(reference, cell["id"])
                    comparison = pair_metrics(
                        frame,
                        other,
                        summary,
                        other_summary,
                        include_relief_ratios=False,
                    )
                else:
                    comparison = {
                        "pair_complete": False,
                        "unavailable_reason": policies[reference]["unavailable_reason"],
                    }
                comparisons.append(
                    {
                        "contrast": "macro_vs_matched_rule"
                        if policy["macro"] == "sac_macro"
                        else "restored_vs_selected_parent",
                        "a": policy["id"],
                        "b": reference,
                        "cell": cell,
                        **comparison,
                    }
                )
            if cell["regd_load_bias"] == 0.5:
                unbiased = by_condition[cell["start_day"], cell["seed"], 0.0]
                other, other_summary = load_episode(policy["id"], unbiased["id"])
                comparisons.append(
                    {
                        "contrast": "matched_bias_effect",
                        "a": policy["id"],
                        "b": policy["id"],
                        "cell": cell,
                        "unbiased_cell": unbiased,
                        **pair_metrics(
                            frame, other, summary, other_summary, bias_pair=True
                        ),
                    }
                )
    reports = OUTPUT / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "episodes.json", summaries)
    write_json(reports / "comparisons.json", comparisons)
    pd.json_normalize(summaries).to_csv(reports / "episodes.csv", index=False)
    pd.json_normalize(comparisons).to_csv(reports / "comparisons.csv", index=False)
    print(
        "Saved",
        len(summaries),
        "planned episode rows and",
        len(comparisons),
        "paired comparisons",
    )


def run(registration):
    verify_registration(registration)
    target = OUTPUT / "status.json"
    if target.exists():
        raise FileExistsError("Refusing to rerun a registered matrix")
    inventory = registration["inventory"]
    jobs = [
        (p, c)
        for p in inventory["policies"]
        if p["available"]
        for c in inventory["cells"]
    ]
    outcomes = {}

    def publish(state):
        write_json(
            target,
            {
                "state": state,
                "updated_unix": time.time(),
                "registration_id": registration["registration_id"],
                "planned_policy_cells": registration["planned_policy_cells"],
                "executable_policy_cells": len(jobs),
                "completed_tasks": len(outcomes),
                "runtime_failures": sum(
                    r["status"] == "runtime_failure" for r in outcomes.values()
                ),
                "outcomes": outcomes,
            },
        )

    publish("running")
    with ProcessPoolExecutor(
        max_workers=registration["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=worker_init,
        initargs=(registration["worker_cpus"],),
    ) as executor:
        futures = {
            executor.submit(run_episode, p, c, registration): f"{p['id']}__{c['id']}"
            for p, c in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            error = future.exception()
            if error is not None:
                outcomes[key] = {
                    "status": "runtime_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                print("FAILED", key, repr(error), flush=True)
            else:
                outcomes[key] = {"status": "recorded", "episode": future.result()}
            publish("running")
            if len(outcomes) % 8 == 0:
                print(f"Recorded {len(outcomes)}/{len(jobs)} rollouts", flush=True)
    failures = [
        key for key, value in outcomes.items() if value["status"] == "runtime_failure"
    ]
    if failures:
        publish("completed_with_runtime_failures")
        raise RuntimeError(
            f"Runtime failures retained; do not summarize incomplete matrix: {failures}"
        )
    verify_registration(registration)
    summarize(registration)
    publish("completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
    else:
        registration = json.loads((OUTPUT / "registration.json").read_text())
        if args.summarize_only:
            verify_registration(registration)
            summarize(registration)
        else:
            run(registration)


if __name__ == "__main__":
    main()
