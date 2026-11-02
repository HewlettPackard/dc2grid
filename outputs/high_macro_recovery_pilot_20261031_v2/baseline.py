"""Run the preregistered strong baseline family before any full SAC learner."""

from __future__ import annotations

import os

# isort: off
from study_common import ROOT, file_hash, load_json, write_json
from evaluate import (
    CappedRuleQuantity,
    ConstantQuantity,
    evaluate_suite,
    evaluation_vector,
)
from register import verify_registration
import torch
# isort: on


def main():
    protocol = load_json(ROOT / "registration.json")
    verify_registration(protocol)
    directory = ROOT / "baselines"
    directory.mkdir(exist_ok=False)
    cpus = protocol["cpu_groups"][0]
    os.sched_setaffinity(0, {cpus[0]})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    models = [
        (f"fraction_{fraction:g}", ConstantQuantity(fraction))
        for fraction in protocol["baseline_fractions"]
    ]
    models.append(("zero_price_capped_rule", CappedRuleQuantity()))
    vector = None
    reports = []
    try:
        vector = evaluation_vector(protocol, 0.0, cpus)
        for name, model in models:
            print("Evaluating", name, flush=True)
            result = evaluate_suite(
                model,
                vector,
                protocol,
                directory / name,
                {
                    "policy_id": name,
                    "kind": "non_learned_macro_baseline",
                    "registration_id": protocol["registration_id"],
                },
            )
            reports.append(
                {
                    "policy_id": name,
                    "qualified_cells": result["qualified_cells"],
                    "all_qualified": result["all_qualified"],
                    "cashflow": result["mean_cashflow_closed_recovery_usd"],
                    "delivered_service_ratio": result["mean_delivered_service_ratio"],
                    "summary": str(directory / name / "summary.json"),
                }
            )
            print(name, reports[-1], flush=True)
    finally:
        if vector is not None:
            vector.close()
    verify_registration(protocol)
    passing = [
        row
        for row in reports
        if row["all_qualified"] and row["policy_id"].startswith("fraction_")
    ]
    write_json(
        directory / "summary.json",
        {
            "registration_id": protocol["registration_id"],
            "policies": reports,
            "all_planned_baseline_cases": len(models)
            * len(protocol["development_cells"]),
            "new_training_steps": 0,
            "sealed_policy_evaluations": 0,
        },
    )
    gate = {
        "registration_id": protocol["registration_id"],
        "allowed": bool(passing),
        "fully_qualified_constant_baselines": [row["policy_id"] for row in passing],
        "baseline_summary_sha256": file_hash(directory / "summary.json"),
        "reason": "Full-development joint feasibility demonstrated"
        if passing
        else (
            "No constant baseline meets every approved joint gate. Stop before SAC; "
            "do not relax constraints or hide failed cases."
        ),
    }
    write_json(ROOT / "training_gate.json", gate)
    print("Training feasibility gate:", gate, flush=True)


if __name__ == "__main__":
    main()
