"""Register, execute and summarize development recovery and locked peak-day tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

# Bootstrap frozen imports and thread caps before loading numerical frameworks.
# isort: off
from recovery import (
    MATCHED,
    RECOVERY,
    ROOT,
    SOURCE,
    continuation_inputs,
    matched,
    read_frame,
    recovery_pair,
    run_episode,
)

import numpy as np
import pandas as pd
import yaml

from c2g_env import C2GMacroEnv
from c2g_env.experiments.identity import configuration_id
# isort: on

CPUS = [3, 7, 18, 21, 25, 34, 41, 44]
HELDOUT_DAYS = [207, 248, 249]
HELDOUT_SEEDS = [100, 101, 102, 103, 104]
SELECTION_RULES = {
    "hardware": (
        "Among original selected low checkpoints passing all eight development "
        "service gates and recovery closures, maximize mean beta=.5 facility "
        "stress-window relief versus nominal; break ties by policy ID."
    ),
    "macro": (
        "Among original selected exact macro/hardware pairs passing all eight "
        "development service gates and recovery closures, maximize mean incremental "
        "cashflow after closed recovery versus rule macro with the identical hardware. "
        "The matched rule baseline must also qualify and close on all eight cells. "
        "Break ties by policy ID. This is not total business profit."
    ),
    "test_scope": (
        "Four classical comparators plus every original selected low/hierarchy slot. "
        "Execute only checkpoints that passed all eight existing matched-validation "
        "service gates; retain unavailable and non-admitted slots explicitly. "
        "Recovery failures cannot be repaired by selecting on held-out outcomes."
    ),
    "restored_policies": (
        "Stability/recovery ablations only; not candidates in this locked test."
    ),
    "heldout_selection_or_retraining": False,
}


def load_json(path: Path):
    return json.loads(path.read_text())


def code_hashes() -> dict:
    names = ["recovery.py", "run_evaluation.py", "test_recovery.py", "README.md"]
    return {name: matched.file_hash(ROOT / name) for name in names}


def runtime_versions() -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }


def policy_inventory() -> tuple[list[dict], dict, list[dict]]:
    original = load_json(MATCHED / "policy_inventory.json")
    episodes = load_json(MATCHED / "reports/episodes.json")
    policies = []
    for source in original["policies"]:
        if source["group"] == "restored":
            continue
        policy = copy.deepcopy(source)
        policy["checkpoint_available"] = source["available"]
        evidence = [row for row in episodes if row["policy_id"] == source["id"]]
        passing = sum(bool(row.get("constraint_qualified")) for row in evidence)
        policy["matched_validation_qualified_cells"] = passing
        if source["available"] and passing != 8:
            policy["available"] = False
            policy["unavailable_reason"] = (
                f"Not admitted: only {passing}/8 existing matched-validation service "
                "cells qualified. Checkpoint retained; no survivor-only promotion."
            )
        policies.append(policy)
    if len(policies) != 20 or sum(p["available"] for p in policies) != 16:
        raise ValueError(
            "Original selected-checkpoint inventory differs from the reviewed scope"
        )
    by_id = {p["id"]: p for p in policies}
    for policy in policies:
        if policy["available"] and policy["macro"] == "sac_macro":
            low = by_id[policy["matched_rule_id"]]
            if (
                not low["available"]
                or low["hardware_checkpoint"] != policy["hardware_checkpoint"]
            ):
                raise ValueError("Missing or mismatched exact-hardware macro reference")
    return policies, original["selection_gate"], original["cells"]


def create_config() -> Path:
    path = ROOT / "environment.yaml"
    config = yaml.safe_load((SOURCE / "c2g_env/config.yaml").read_text())
    if (
        config["global"]["episode_ticks"] != 17280
        or config["global"]["dt_seconds"] != 5
    ):
        raise ValueError("Expected the frozen 24-hour, five-second simulator")
    for key in ("trace_dir", "energy_dir", "weather_dir"):
        config["global"][key] = str((SOURCE / config["global"][key]).resolve())
    content = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(
                "Recovery configuration exists with different content"
            )
        return path
    path.write_text(content)
    return path


def preflight(config: Path) -> list[dict]:
    rows = []
    for day in [196, 256, *HELDOUT_DAYS]:
        cell = {"start_day": day, "seed": 100, "regd_load_bias": 0.5}
        env = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=config,
            start_day=day,
            regd_load_bias=0.5,
        )
        try:
            _, info = env.reset(seed=100)
            fast = env._fast_env
            ticks = fast._episode_ticks + int(RECOVERY["duration_seconds"] / fast._dt)
            grid, weather = continuation_inputs(fast, cell, ticks)
            rows.append(
                {
                    "start_day": day,
                    "ticks": ticks,
                    "first_timestamp": str(grid.calendar_timestamps[0]),
                    "last_observation_timestamp": str(grid.calendar_timestamps[-1]),
                    "load_sha256": hashlib.sha256(
                        grid._calendar_load_mw.tobytes()
                    ).hexdigest(),
                    "temperature_sha256": hashlib.sha256(
                        weather._replay_temps.tobytes()
                    ).hexdigest(),
                    "weather_source": info["weather_source"],
                    "workload_phase_day_mod_30": day % 30,
                    "policy_actions_executed": 0,
                    "purpose": (
                        "Structural replay coverage only, "
                        "not held-out performance inspection"
                    ),
                }
            )
        finally:
            env.close()
    return rows


def sign_registration(body: dict) -> dict:
    return {**body, "registration_id": configuration_id(body)}


def verify_registration(registration: dict) -> None:
    body = {
        key: value for key, value in registration.items() if key != "registration_id"
    }
    if configuration_id(body) != registration["registration_id"]:
        raise RuntimeError("Recovery registration content changed")
    if registration["code_files"] != code_hashes():
        raise RuntimeError(
            "Registered recovery implementation or documentation changed"
        )
    if runtime_versions() != registration["runtime"]:
        raise RuntimeError("Registered runtime changed")
    if matched.file_hash(registration["config_path"]) != registration["config_sha256"]:
        raise RuntimeError("Registered recovery configuration changed")
    if (
        registration["recovery"] != RECOVERY
        or registration["selection_rules"] != SELECTION_RULES
    ):
        raise RuntimeError("Registered recovery or selection rules changed")
    original = load_json(MATCHED / "registration.json")
    matched.verify_registration(original)
    for name, digest in registration["protected_inputs"].items():
        if matched.file_hash(name) != digest:
            raise RuntimeError(f"Protected evidence changed: {name}")
    for policy in registration["policies"]:
        for key in ("hardware_checkpoint", "macro_checkpoint"):
            artifact = policy[key]
            if artifact and matched.file_hash(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"Checkpoint changed: {policy['id']}")
    if registration["stage"] == "heldout":
        if (
            matched.file_hash(ROOT / "selection.json")
            != registration["selection_sha256"]
        ):
            raise RuntimeError("Locked selection changed")


def register_development() -> dict:
    target = ROOT / "development_registration.json"
    if target.exists() or (ROOT / "heldout").exists():
        raise FileExistsError(
            "Registration exists or held-out evaluation has already started"
        )
    review = load_json(ROOT / "recovery_review.json")
    if not review["no_blocking_issues_remaining"]:
        raise RuntimeError("Recovery review has unresolved blocking issues")
    checks = load_json(ROOT / "implementation_checks.json")
    if not checks["passed"] or checks["code_files"] != code_hashes():
        raise RuntimeError("Implementation tests do not match the current evaluator")
    original = load_json(MATCHED / "registration.json")
    matched.verify_registration(original)
    policies, gate, cells = policy_inventory()
    config = create_config()
    coverage = preflight(config)
    protected = [
        MATCHED / "registration.json",
        MATCHED / "policy_inventory.json",
        MATCHED / "policy_inventory_audit.json",
        MATCHED / "accounting_review.json",
        MATCHED / "reports/episodes.json",
        MATCHED / "reports/comparisons.json",
        SOURCE / "c2g_env/config.yaml",
        ROOT / "macro1002_diagnosis.json",
        ROOT / "implementation_checks.json",
        ROOT / "test_results.xml",
    ]
    body = {
        "schema": "physical-recovery-evaluation-v1",
        "stage": "development",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user_approved_physical_recovery": True,
        "code_files": code_hashes(),
        "runtime": runtime_versions(),
        "config_path": str(config),
        "config_sha256": matched.file_hash(config),
        "protected_inputs": {str(path): matched.file_hash(path) for path in protected},
        "frozen_matched_registration_id": original["registration_id"],
        "policies": policies,
        "selection_gate": gate,
        "cells": cells,
        "recovery": RECOVERY,
        "selection_rules": SELECTION_RULES,
        "replay_coverage": coverage,
        "workers": len(CPUS),
        "worker_cpus": CPUS,
        "native_threads": 1,
        "gpus_used": 0,
        "planned_policy_cells": len(policies) * len(cells),
        "executable_policy_cells": sum(p["available"] for p in policies) * len(cells),
        "development_service_reconstruction": (
            "Replay existing hardware actions; infer original macro actions; "
            "require exact "
            "uncompressed hardware and macro CSV equality before appending recovery."
        ),
        "heldout_service_execution": (
            "Fresh deterministic inference from locked checkpoints"
        ),
        "planned_heldout_days": HELDOUT_DAYS,
        "planned_heldout_seeds": HELDOUT_SEEDS,
        "planned_biases": [0.0, 0.5],
        "economic_scope": (
            "Benchmark settlement minus metered electricity over 24h service "
            "plus fixed 2h recovery. Closed-inventory comparisons require both "
            "trajectories to complete and close. Wear valuation is an assumed "
            "linear AC-throughput sensitivity, not calibrated wear cost. Latency/SLA "
            "valuation and total business profit are not claimed."
        ),
        "missingness": (
            "Retain unavailable, non-admitted, terminated, nonqualified and "
            "unclosed cases; "
            "never average a full cohort using only successful cases."
        ),
        "registration_review": review,
        "implementation_checks": checks,
    }
    if not body["registration_review"]["no_blocking_issues_remaining"]:
        raise RuntimeError("Recovery review has unresolved blocking issues")
    body["protected_inputs"][str(ROOT / "recovery_review.json")] = matched.file_hash(
        ROOT / "recovery_review.json"
    )
    registration = sign_registration(body)
    matched.write_json(target, registration)
    verify_registration(registration)
    print("Registered development:", registration["registration_id"], flush=True)
    return registration


def mean_sd(
    rows: list[dict], key: str, expected: int
) -> tuple[float | None, float | None]:
    if len(rows) != expected or any(row.get(key) is None for row in rows):
        return None, None
    values = np.array([row[key] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite cohort metric: {key}")
    return float(values.mean()), float(values.std(ddof=1)) if expected > 1 else 0.0


def worker_init(cpus: list[int]) -> None:
    index = multiprocessing.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {cpus[index % len(cpus)]})
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


@lru_cache(maxsize=48)
def load_traces(stage: str, policy_id: str, cell_id: str):
    directory = ROOT / stage / "traces" / policy_id / cell_id
    summary = load_json(directory / "episode.json")
    for name, digest in summary["trace_hashes"].items():
        if matched.file_hash(directory / name) != digest:
            raise RuntimeError(f"Trace changed: {directory / name}")
    service = read_frame(directory / "hardware.csv.gz")
    recovery = (
        read_frame(directory / "recovery.csv.gz")
        if summary["recovery"]["performed"]
        else None
    )
    return service, recovery, summary


def compare(stage: str, a: dict, b: dict, protocol: dict, *, contrast: str) -> dict:
    if "recovery" not in a or "recovery" not in b:
        return {
            "pair_complete": False,
            "recovery_pair_complete": False,
            "recovery_pair_closed": False,
            "unavailable_reason": (
                "A paired policy-cell is unavailable, non-admitted or runtime-failed"
            ),
            "cashflow_delta_after_closed_recovery_usd": None,
        }
    af, ar, _ = load_traces(stage, a["policy_id"], a["cell"]["id"])
    bf, br, _ = load_traces(stage, b["policy_id"], b["cell"]["id"])
    bias_pair = contrast == "matched_bias_effect"
    service = matched.pair_metrics(
        af,
        bf,
        a,
        b,
        bias_pair=bias_pair,
        include_relief_ratios=contrast == "vs_nominal",
    )
    if ar is not None and br is not None:
        keys = matched.BIAS_SHARED if bias_pair else matched.EXOGENOUS
        n = min(len(ar), len(br))
        for key in keys:
            np.testing.assert_array_equal(
                ar[key].to_numpy()[:n], br[key].to_numpy()[:n]
            )
        if bias_pair:
            np.testing.assert_allclose(
                ar.regd_signal.to_numpy()[:n],
                np.clip(
                    br.regd_signal.to_numpy()[:n]
                    + 0.5 * ar.load_anomaly.to_numpy()[:n],
                    -1,
                    1,
                ),
                rtol=0,
                atol=1e-12,
            )
        elif b["policy_id"] == "nominal":
            np.testing.assert_allclose(
                ar.p_reference_mw.to_numpy()[:n],
                br.p_facility_mw.to_numpy()[:n],
                rtol=1e-12,
                atol=1e-8,
            )
    if service["pair_complete"]:
        service["reward_component_delta"] = {
            key: a["reward_components"][key] - b["reward_components"][key]
            for key in matched.REWARDS
        }
    return {**service, **recovery_pair(a, b, protocol)}


def summarize(registration: dict) -> dict:
    stage = registration["stage"]
    output = ROOT / stage
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    policies, cells = registration["policies"], registration["cells"]
    records = {}
    for policy in policies:
        for cell in cells:
            directory = output / "traces" / policy["id"] / cell["id"]
            if not policy["available"]:
                summary = {
                    "policy_id": policy["id"],
                    "cell": cell,
                    "status": "unavailable",
                    "checkpoint_available": policy["checkpoint_available"],
                    "reason": policy["unavailable_reason"],
                }
            elif (directory / "episode.json").exists():
                summary = load_json(directory / "episode.json")
            elif (directory / "failure.json").exists():
                summary = load_json(directory / "failure.json")
            else:
                raise RuntimeError(
                    f"Expected an explicit completion/failure record: {directory}"
                )
            records[policy["id"], cell["id"]] = summary
    by_condition = {(c["start_day"], c["seed"], c["regd_load_bias"]): c for c in cells}
    comparisons = []
    for policy in policies:
        for cell in cells:
            a = records[policy["id"], cell["id"]]
            contrasts = [("vs_nominal", "nominal", cell)]
            if policy["group"] == "old_low":
                contrasts.append(("low_vs_pid", "pid", cell))
            if policy["macro"] == "sac_macro":
                contrasts.append(
                    ("macro_vs_matched_rule", policy["matched_rule_id"], cell)
                )
            if cell["regd_load_bias"] == 0.5:
                contrasts.append(
                    (
                        "matched_bias_effect",
                        policy["id"],
                        by_condition[cell["start_day"], cell["seed"], 0.0],
                    )
                )
            for contrast, reference, other_cell in contrasts:
                b = records[reference, other_cell["id"]]
                comparisons.append(
                    {
                        "contrast": contrast,
                        "a": policy["id"],
                        "b": reference,
                        "cell": cell,
                        "reference_cell": other_cell,
                        **compare(
                            stage, a, b, registration["recovery"], contrast=contrast
                        ),
                    }
                )
    episodes = list(records.values())
    matched.write_json(reports / "episodes.json", episodes)
    matched.write_json(reports / "comparisons.json", comparisons)
    summary_rows = []
    n = len(cells)
    nb = sum(c["regd_load_bias"] == 0.5 for c in cells)
    for policy in policies:
        group = [r for r in episodes if r["policy_id"] == policy["id"]]
        biased = [
            c
            for c in comparisons
            if c["a"] == policy["id"]
            and c["contrast"] == "vs_nominal"
            and c["cell"]["regd_load_bias"] == 0.5
        ]
        bias_effect = [
            c
            for c in comparisons
            if c["a"] == policy["id"] and c["contrast"] == "matched_bias_effect"
        ]
        macro = [
            c
            for c in comparisons
            if c["a"] == policy["id"] and c["contrast"] == "macro_vs_matched_rule"
        ]
        nominal = [
            c
            for c in comparisons
            if c["a"] == policy["id"] and c["contrast"] == "vs_nominal"
        ]
        row = {
            "policy_id": policy["id"],
            "group": policy["group"],
            "available": policy["available"],
            "checkpoint_available": policy["checkpoint_available"],
            "planned_cells": n,
            "service_complete_cells": sum(bool(r.get("complete")) for r in group),
            "service_qualified_cells": sum(
                bool(r.get("constraint_qualified")) for r in group
            ),
            "recovery_complete_cells": sum(
                bool(r.get("recovery", {}).get("complete")) for r in group
            ),
            "recovery_closed_cells": sum(
                bool(r.get("recovery", {}).get("closed")) for r in group
            ),
            "overall_qualified_cells": sum(
                bool(r.get("overall_qualified")) for r in group
            ),
            "unavailable_reason": policy["unavailable_reason"],
        }
        for target, source, subset, count in (
            ("rmse_kw", "tracking_rmse_kw", group, n),
            ("normalized_rmse", "normalized_rmse", group, n),
            ("biased_facility_relief_mw", "stress_facility_relief_mw", biased, nb),
            (
                "biased_actual_system_peak_relief_mw",
                "actual_system_facility_peak_reduction_mw",
                biased,
                nb,
            ),
            (
                "matched_bias_revenue_change_pct",
                "regulation_revenue_change_pct",
                bias_effect,
                nb,
            ),
            (
                "cash_vs_nominal_before_recovery_usd",
                "cashflow_delta_before_recovery_usd",
                nominal,
                n,
            ),
            (
                "cash_vs_nominal_after_closed_recovery_usd",
                "cashflow_delta_after_closed_recovery_usd",
                nominal,
                n,
            ),
            (
                "macro_cash_before_recovery_usd",
                "cashflow_delta_before_recovery_usd",
                macro,
                n,
            ),
            (
                "macro_cash_after_closed_recovery_usd",
                "cashflow_delta_after_closed_recovery_usd",
                macro,
                n,
            ),
        ):
            row[target], row[target + "_sd"] = mean_sd(subset, source, count)
        summary_rows.append(row)
    matched.write_json(reports / "policy_summary.json", summary_rows)
    pd.DataFrame(summary_rows).to_csv(reports / "policy_summary.csv", index=False)

    condition_rows = []
    for policy in policies:
        if policy["macro"] != "sac_macro":
            continue
        for day in sorted({cell["start_day"] for cell in cells}):
            for bias in (0.0, 0.5):
                expected = sum(
                    c["start_day"] == day and c["regd_load_bias"] == bias for c in cells
                )
                group = [
                    c
                    for c in comparisons
                    if c["a"] == policy["id"]
                    and c["contrast"] == "macro_vs_matched_rule"
                    and c["cell"]["start_day"] == day
                    and c["cell"]["regd_load_bias"] == bias
                ]
                row = {
                    "policy_id": policy["id"],
                    "start_day": day,
                    "bias": bias,
                    "planned_pairs": expected,
                    "complete_pairs": sum(bool(c["pair_complete"]) for c in group),
                    "closed_recovery_pairs": sum(
                        bool(c["recovery_pair_closed"]) for c in group
                    ),
                }
                for key in (
                    "cashflow_delta_before_recovery_usd",
                    "cashflow_delta_after_closed_recovery_usd",
                    "recovery_electricity_cost_delta_usd",
                    "total_battery_throughput_delta_mwh",
                    "stress_facility_relief_mw",
                ):
                    row[key], row[key + "_sd"] = mean_sd(group, key, expected)
                condition_rows.append(row)
    pd.DataFrame(condition_rows).to_csv(
        reports / "macro_condition_summary.csv", index=False
    )
    matched.write_json(reports / "macro_condition_summary.json", condition_rows)

    table2 = []
    primary = (
        load_json(ROOT / "selection.json")["primary_hardware"]
        if stage == "heldout"
        else None
    )
    for policy in policies:
        if policy["id"] not in {"rule_hardware", "pid", "bang_bang", primary}:
            continue
        relief = [
            c
            for c in comparisons
            if c["a"] == policy["id"]
            and c["contrast"] == "vs_nominal"
            and c["cell"]["regd_load_bias"] == 0.5
        ]
        bias = [
            c
            for c in comparisons
            if c["a"] == policy["id"] and c["contrast"] == "matched_bias_effect"
        ]
        rows = [
            ("stress_facility_relief_mw", relief),
            ("LFR_pct", relief),
            ("SPDR_proxy_pct", relief),
            ("LWSR_pct", relief),
            ("LWCF_pct", relief),
            ("regulation_revenue_change_pct", bias),
        ]
        for metric in ("reward_throughput", "reward_soc", "reward_thermal"):
            rows.append(
                (
                    metric,
                    [
                        {metric: r.get("reward_component_delta", {}).get(metric)}
                        for r in relief
                    ],
                )
            )
        for metric, subset in rows:
            mean, sd = mean_sd(subset, metric, nb)
            table2.append(
                {
                    "policy_id": policy["id"],
                    "metric": metric,
                    "mean": mean,
                    "sample_sd": sd,
                    "planned_biased_cells": nb,
                    "defined_metric_cells": sum(
                        item.get(metric) is not None for item in subset
                    ),
                    "complete_biased_cells": sum(
                        bool(r["pair_complete"]) for r in relief
                    ),
                }
            )
    pd.DataFrame(table2).to_csv(reports / "table2_shaped_metrics.csv", index=False)
    inventory_rows = []
    for episode in episodes:
        row = {
            "policy_id": episode["policy_id"],
            "cell_id": episode["cell"]["id"],
            "status": episode["status"],
        }
        if "recovery" in episode:
            recovery = episode["recovery"]
            row.update(
                {
                    "service_complete": episode["complete"],
                    "service_qualified": episode["constraint_qualified"],
                    "service_terminal_soc": episode["terminal"]["soc"],
                    "service_terminal_backlog_kwh": episode["terminal"]["backlog_kwh"],
                    "service_time_weighted_lmp_usd_mwh": episode[
                        "observed_energy_metrics"
                    ]["lmp_time_weighted_mean_usd_mwh"],
                    "recovery_complete": recovery["complete"],
                    "recovery_closed": recovery["closed"],
                }
            )
            if recovery["performed"]:
                row.update(
                    {
                        key: recovery[key]
                        for key in (
                            "soc_error",
                            "backlog_kwh",
                            "temperature_A_minus_reference_C",
                            "temperature_B_minus_reference_C",
                            "usable_energy_property_change_mwh",
                        )
                    }
                )
                row.update(
                    {
                        f"recovery_{key}": value
                        for key, value in recovery["observed_energy_metrics"].items()
                    }
                )
        inventory_rows.append(row)
    pd.DataFrame(inventory_rows).to_csv(
        reports / "recovery_inventory_and_prices.csv", index=False
    )
    observed_recovery = [
        row["recovery"] for row in episodes if row.get("recovery", {}).get("performed")
    ]
    readback = {
        "registration_id": registration["registration_id"],
        "stage": stage,
        "planned_policy_cells": len(episodes),
        "recorded_rollouts": sum("recovery" in r for r in episodes),
        "unavailable_policy_cells": sum(r["status"] == "unavailable" for r in episodes),
        "runtime_failures": sum(r["status"] == "runtime_failure" for r in episodes),
        "service_complete": sum(bool(r.get("complete")) for r in episodes),
        "service_qualified": sum(bool(r.get("constraint_qualified")) for r in episodes),
        "recovery_complete": sum(
            bool(r.get("recovery", {}).get("complete")) for r in episodes
        ),
        "recovery_closed": sum(
            bool(r.get("recovery", {}).get("closed")) for r in episodes
        ),
        "overall_qualified": sum(bool(r.get("overall_qualified")) for r in episodes),
        "exact_development_service_replays": sum(
            bool(r.get("service_replay_check", {}).get("verified")) for r in episodes
        ),
        "comparison_records": len(comparisons),
        "complete_service_pairs": sum(c["pair_complete"] for c in comparisons),
        "closed_recovery_pairs": sum(c["recovery_pair_closed"] for c in comparisons),
        "observed_recovery_residual_cases": len(observed_recovery),
        "max_abs_observed_recovery_soc_error": max(
            (abs(row["soc_error"]) for row in observed_recovery), default=None
        ),
        "max_abs_observed_recovery_temperature_A_error_C": max(
            (abs(row["temperature_A_minus_reference_C"]) for row in observed_recovery),
            default=None,
        ),
        "max_abs_observed_recovery_temperature_B_error_C": max(
            (abs(row["temperature_B_minus_reference_C"]) for row in observed_recovery),
            default=None,
        ),
        "new_training_steps": 0,
        "interpretation": [
            "Selected frozen checkpoints, not from-scratch algorithm comparisons.",
            "Unavailable, nonqualified, incomplete and unclosed cells are retained.",
            "SPDR is a stress-average proxy, not actual maximum-load reduction.",
            "Weather is calendar-matched 2024 climatology with 2023 grid load.",
            "Recovery uses continuing proxy LMP, not real realized ISO prices.",
            "Recovery is next-day 00:00-02:00; prices are measured, not assumed flat.",
            "Finite-tolerance closure does not erase degradation or workload latency.",
            "Wear sensitivity assumes an AC-throughput price, not calibrated wear.",
            "Standard deviations describe evaluation cases, not training uncertainty.",
        ],
    }
    matched.write_json(reports / "readback.json", readback)
    load_traces.cache_clear()
    return readback


def lock_heldout() -> dict:
    if (ROOT / "heldout_registration.json").exists() or (ROOT / "heldout").exists():
        raise FileExistsError("Held-out evaluation is already locked or started")
    development = load_json(ROOT / "development_registration.json")
    verify_registration(development)
    if load_json(ROOT / "development/status.json")["state"] != "completed":
        raise RuntimeError(
            "Development recovery must complete before held-out selection"
        )
    summaries = load_json(ROOT / "development/reports/policy_summary.json")
    by_id = {row["policy_id"]: row for row in summaries}
    n = len(development["cells"])
    low = [
        row
        for row in summaries
        if row["group"] == "old_low"
        and row["overall_qualified_cells"] == n
        and row["biased_facility_relief_mw"] is not None
    ]
    macro = [
        row
        for row in summaries
        if row["group"] == "old_pair"
        and row["overall_qualified_cells"] == n
        and row["macro_cash_after_closed_recovery_usd"] is not None
        and by_id[row["policy_id"].replace("old_pair_", "old_low_")][
            "overall_qualified_cells"
        ]
        == n
    ]
    if not low or not macro:
        raise RuntimeError(
            "No fully qualified closed-recovery candidate for one hierarchy level"
        )
    selected_low = sorted(
        low, key=lambda r: (-r["biased_facility_relief_mw"], r["policy_id"])
    )[0]
    selected_macro = sorted(
        macro,
        key=lambda r: (-r["macro_cash_after_closed_recovery_usd"], r["policy_id"]),
    )[0]
    selection = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_registration_id": development["registration_id"],
        "selection_rules": SELECTION_RULES,
        "primary_hardware": selected_low["policy_id"],
        "primary_macro_pair": selected_macro["policy_id"],
        "macro_exact_hardware_reference": selected_macro["policy_id"].replace(
            "old_pair_", "old_low_"
        ),
        "development_scores": {"hardware": selected_low, "macro": selected_macro},
        "all_development_candidates": summaries,
        "development_episodes_sha256": matched.file_hash(
            ROOT / "development/reports/episodes.json"
        ),
        "development_comparisons_sha256": matched.file_hash(
            ROOT / "development/reports/comparisons.json"
        ),
        "heldout_results_seen": False,
        "retraining_performed": False,
    }
    matched.write_json(ROOT / "selection.json", selection)
    cells = [
        {
            "id": f"d{day}_s{seed}_b{'0p5' if bias else '0'}",
            "start_day": day,
            "seed": seed,
            "regd_load_bias": bias,
        }
        for day in HELDOUT_DAYS
        for seed in HELDOUT_SEEDS
        for bias in (0.0, 0.5)
    ]
    body = {
        key: value for key, value in development.items() if key != "registration_id"
    }
    body.update(
        {
            "stage": "heldout",
            "cells": cells,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "development_registration_id": development["registration_id"],
            "selection_sha256": matched.file_hash(ROOT / "selection.json"),
            "primary_hardware": selection["primary_hardware"],
            "primary_macro_pair": selection["primary_macro_pair"],
            "planned_policy_cells": len(development["policies"]) * len(cells),
            "executable_policy_cells": sum(
                p["available"] for p in development["policies"]
            )
            * len(cells),
        }
    )
    body["protected_inputs"] = {
        **body["protected_inputs"],
        str(ROOT / "development_registration.json"): matched.file_hash(
            ROOT / "development_registration.json"
        ),
        str(ROOT / "development/reports/episodes.json"): selection[
            "development_episodes_sha256"
        ],
        str(ROOT / "development/reports/comparisons.json"): selection[
            "development_comparisons_sha256"
        ],
    }
    registration = sign_registration(body)
    matched.write_json(ROOT / "heldout_registration.json", registration)
    verify_registration(registration)
    print(
        "Locked selection:",
        selection["primary_hardware"],
        selection["primary_macro_pair"],
        flush=True,
    )
    print("Held-out registration:", registration["registration_id"], flush=True)
    return registration


def run(registration: dict) -> None:
    verify_registration(registration)
    stage = registration["stage"]
    output = ROOT / stage
    output.mkdir(parents=True, exist_ok=True)
    target = output / "status.json"
    if target.exists():
        raise FileExistsError("Refusing to rerun or overwrite a registered evaluation")
    jobs = [
        (p, c)
        for p in registration["policies"]
        if p["available"]
        for c in registration["cells"]
    ]
    outcomes = {}

    def publish(state: str):
        matched.write_json(
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
            executor.submit(
                run_episode,
                policy,
                cell,
                registration,
                directory=output / "traces" / policy["id"] / cell["id"],
            ): f"{policy['id']}__{cell['id']}"
            for policy, cell in jobs
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
                print(
                    f"{stage}: recorded {len(outcomes)}/{len(jobs)} rollouts",
                    flush=True,
                )
    verify_registration(registration)
    readback = summarize(registration)
    state = (
        "completed_with_runtime_failures"
        if readback["runtime_failures"]
        else "completed"
    )
    publish(state)
    print(json.dumps(readback, indent=2), flush=True)
    if readback["runtime_failures"]:
        raise RuntimeError(
            "Evaluation retained runtime failures; do not promote incomplete results"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "register-development",
            "run-development",
            "lock-heldout",
            "run-heldout",
            "summarize",
        ),
    )
    parser.add_argument("--stage", choices=("development", "heldout"))
    args = parser.parse_args()
    os.sched_setaffinity(0, {59})
    if args.command == "register-development":
        register_development()
    elif args.command == "lock-heldout":
        lock_heldout()
    elif args.command == "summarize":
        if args.stage is None:
            parser.error("--stage is required for summarize")
        registration = load_json(ROOT / f"{args.stage}_registration.json")
        verify_registration(registration)
        print(json.dumps(summarize(registration), indent=2))
    else:
        stage = "development" if args.command == "run-development" else "heldout"
        run(load_json(ROOT / f"{stage}_registration.json"))


if __name__ == "__main__":
    main()
