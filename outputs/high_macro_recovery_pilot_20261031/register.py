"""Immutable study registration and pre-SAC feasibility gates."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import platform
import time
import zipfile

# isort: off
from study_common import (
    ROOT,
    TASK_SCHEMA,
    VARIANTS,
    file_hash,
    identity,
    load_json,
    verify_previous_evidence,
    write_json,
)
import numpy as np
# isort: on


def program_hashes() -> dict:
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "README.md"]
    return {path.name: file_hash(path) for path in paths}


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "stable_baselines3", "gymnasium")
        },
    }


def entropy_calibration(protocol: dict) -> dict:
    bound = protocol["normalization"]["cash_prefix_bound_usd"]
    price, residual, performance, hours = 27.51, 15.0, 0.9, 0.25
    acceptance = 1 / (1 + math.exp(-0.15 * price))
    gross_gradient = price * performance * hours * acceptance * (residual - 5) / bound
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    weights = weights / np.sqrt(np.pi)
    std = protocol["initial_actor_std"]
    rows = []
    for fraction in (0.9, 0.95, 0.98, 0.995, 0.999):
        mu = math.atanh(2 * fraction - 1)
        z = mu + np.sqrt(2) * std * nodes
        action = np.tanh(z)
        logp = (
            -0.5 * ((z - mu) / std) ** 2
            - math.log(std * math.sqrt(2 * math.pi))
            - np.log(1 - action**2 + 1e-6)
        )
        mean_gradient = float(
            np.dot(weights, 2 * action * (1 - action**2) / (1 - action**2 + 1e-6))
        ) / (2 * fraction * (1 - fraction))
        rows.append(
            {
                "quantity_fraction": fraction,
                "mean_log_density": float(np.dot(weights, logp)),
                "old_alpha_gradient_ratio": 1e-5 * abs(mean_gradient) / gross_gradient,
                "registered_alpha_gradient_ratio": (
                    protocol["algorithm"]["ent_coef"]
                    * abs(mean_gradient)
                    / gross_gradient
                ),
            }
        )
    maximum = max(row["registered_alpha_gradient_ratio"] for row in rows)
    if maximum > 0.05:
        raise ValueError(
            "Entropy is too large for the registered reference signal scale"
        )
    return {
        "fixed_entropy_coefficient": protocol["algorithm"]["ent_coef"],
        "cash_normalization_usd": bound,
        "reference_price": price,
        "reference_residual_mw": residual,
        "reference_performance": performance,
        "reference_gross_revenue_gradient": gross_gradient,
        "rows": rows,
        "maximum_registered_gradient_ratio": maximum,
        "limit": 0.05,
        "interpretation": (
            "Calibration against a representative gross-revenue derivative, not a "
            "proof about the full cash gradient near a stationary point. No claim "
            "that entropy is negligible for every possible policy/distribution."
        ),
    }


def validate_protocol(protocol: dict):
    if (
        protocol["schema"] != TASK_SCHEMA
        or protocol["algorithm"]["gamma"] != 1
        or protocol["algorithm"]["gradient_steps"] != -1
        or protocol["variants"] != VARIANTS
        or protocol["minimum_service_fraction"] != 0.9
        or protocol["sealed_test_execution_in_this_pilot"]
    ):
        raise ValueError("Protocol differs from the approved finite-task study")
    quantum = protocol["n_envs"] * protocol["algorithm"]["train_freq"]
    budgets = [
        protocol["learning_starts"],
        protocol["total_timesteps"],
        *protocol["milestones"],
    ]
    if any(value % quantum for value in budgets):
        raise ValueError("Budgets must stop after complete rollout/update blocks")
    if protocol["milestones"][-1] != protocol["total_timesteps"]:
        raise ValueError("The final post-update evaluation must match the budget")
    if set(protocol["train_days"]) & {
        item["start_day"]
        for item in protocol["development_cells"] + protocol["sealed_test_cells"]
    }:
        raise ValueError("Calendar split overlap")
    development_days = {row["start_day"] for row in protocol["development_cells"]}
    sealed_days = {row["start_day"] for row in protocol["sealed_test_cells"]}
    if development_days & sealed_days:
        raise ValueError("Development and sealed calendars overlap")
    all_days = sorted(set(protocol["train_days"]) | development_days | sealed_days)
    if any((b - a) * 86400 <= 93600 for a, b in zip(all_days, all_days[1:])):
        raise ValueError("Reserved 26-hour calendar windows overlap")
    cpus = [cpu for group in protocol["cpu_groups"] for cpu in group]
    if len(cpus) != 40 or len(set(cpus)) != 40:
        raise ValueError("Exactly forty distinct assigned CPU threads are required")
    sibling_sets = set()
    from pathlib import Path

    for cpu in cpus:
        siblings = (
            Path(f"/sys/devices/system/cpu/cpu{cpu}/topology") / "thread_siblings_list"
        ).read_text()
        if siblings in sibling_sets:
            raise ValueError("Assigned CPUs share a physical core")
        sibling_sets.add(siblings)
    if protocol["threads"] != 1 or protocol["n_envs"] != 4:
        raise ValueError("Resource caps differ from approval")


def verify_registration(protocol: dict):
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise RuntimeError("Registered macro protocol changed")
    if program_hashes() != protocol["code_files"]:
        raise RuntimeError("Registered macro implementation changed")
    if runtime_versions() != protocol["runtime"]:
        raise RuntimeError("Registered runtime changed")
    if file_hash(protocol["config_path"]) != protocol["config_sha256"]:
        raise RuntimeError("Registered environment configuration changed")
    for path, expected in protocol["protected_files"].items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Registered evidence changed: {path}")
    if (
        file_hash(ROOT / "implementation.zip")
        != protocol["implementation_archive_sha256"]
    ):
        raise RuntimeError("Registered implementation archive changed")
    verify_previous_evidence()
    validate_protocol(protocol)


def verify_training_gate(protocol: dict):
    verify_registration(protocol)
    gate = load_json(ROOT / "training_gate.json")
    if gate["registration_id"] != protocol["registration_id"] or not gate["allowed"]:
        raise RuntimeError("SAC is blocked by the registered baseline feasibility gate")
    if file_hash(ROOT / "baselines/summary.json") != gate["baseline_summary_sha256"]:
        raise RuntimeError("Baseline feasibility evidence changed")
    baseline = load_json(ROOT / "baselines/summary.json")
    qualifying = {
        row["policy_id"]
        for row in baseline["policies"]
        if row["all_qualified"] and row["policy_id"].startswith("fraction_")
    }
    if (
        baseline["registration_id"] != protocol["registration_id"]
        or not qualifying
        or qualifying != set(gate["fully_qualified_constant_baselines"])
    ):
        raise RuntimeError(
            "Baseline evidence does not satisfy the feasibility predicate"
        )


def register():
    path = ROOT / "registration.json"
    if path.exists():
        raise FileExistsError("This macro study is already registered")
    protocol = load_json(ROOT / "protocol_draft.json")
    validate_protocol(protocol)
    checks = load_json(ROOT / "checks.json")
    if not checks["passed"] or checks["code_files"] != program_hashes():
        raise RuntimeError("Tests/GPU smoke do not match this implementation")
    review = load_json(ROOT / "audits/task_contract_resolution.json")
    if not review["no_code_blockers_remaining"]:
        raise RuntimeError("Adversarial task review has unresolved code blockers")
    verify_previous_evidence()
    scale = entropy_calibration(protocol)
    write_json(ROOT / "entropy_scale.json", scale)
    protected = [
        ROOT / "audits/data_split.json",
        ROOT / "audits/market_objective.json",
        ROOT / "audits/prior_results_adversary.json",
        ROOT / "audits/task_contract_resolution.json",
        ROOT / "audits/other_study_insights.json",
        ROOT / "entropy_scale.json",
        ROOT / "checks.json",
        ROOT / "test_results.xml",
        ROOT / "smoke/gpu_smoke.json",
    ]
    protocol.update(
        {
            "created_unix": time.time(),
            "code_files": program_hashes(),
            "runtime": runtime_versions(),
            "config_sha256": file_hash(protocol["config_path"]),
            "protected_files": {str(item): file_hash(item) for item in protected},
            "training_feasibility_gate": (
                "Before any full SAC fit, evaluate all registered constant-fraction "
                "baselines on all24 development cells. At least one constant baseline "
                "must qualify on every cell. Otherwise stop and retain all results; "
                "no relaxation, substitution or automatic relaunch."
            ),
            "budget_definition": {
                "macro_transitions_per_fit": 20480,
                "included_teacher_guided_warmup_macro_transitions": 1024,
                "uniform_exploration_transitions_in_warmup": 52,
                "normal_sac_updates": 19456,
                "additional_critic_only_updates": 512,
                "additional_supervised_bias_updates": 256,
                "supervised_and_calibration_updates_collect_no_environment_steps": True,
                "no_fault_complete_training_episodes": 212,
                "no_fault_inflight_training_episodes": 4,
            },
            "pairing_limit": (
                "Same seed/curriculum ordinal, not identical total training exposure "
                "after policies terminate at different times. Compare on the fixed "
                "development matrix; retain every training attempt and inflight prefix."
            ),
        }
    )
    archive = ROOT / "implementation.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for name in protocol["code_files"]:
            handle.write(ROOT / name, name)
    protocol["implementation_archive_sha256"] = file_hash(archive)
    protocol = {**protocol, "registration_id": identity(protocol)}
    write_json(path, protocol)
    verify_registration(protocol)
    print("Registered high-level pilot:", protocol["registration_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    register()
