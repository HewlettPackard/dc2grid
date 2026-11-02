"""Freeze the standalone SAC comparison before any reinforcement updates."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# isort: off
from common import (
    MACRO_STUDY,
    ROOT,
    SOURCE,
    contracts,
    file_hash,
    identity,
    read_json,
    write_json,
)
from history_policy import FEATURE_CONTRACT

from prepare import choose_cpu_groups
from study_common import verify_previous_evidence
# isort: on

CODE_FILES = (
    "common.py",
    "controllers.py",
    "history_policy.py",
    "fast_task.py",
    "fit_bc.py",
    "sac_core.py",
    "evaluate.py",
    "evaluate_models.py",
    "train.py",
    "launch_training.py",
    "prepare_study.py",
    "test_controllers.py",
    "test_learning_contract.py",
    "test_sac_core.py",
)


def verify_protocol(protocol: dict) -> None:
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if identity(body) != protocol["registration_id"]:
        raise ValueError("Study registration was mutated")
    for name, expected in protocol["code_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Registered source changed: {name}")
    for name, expected in protocol["input_hashes"].items():
        if file_hash(name) != expected:
            raise ValueError(f"Registered input changed: {name}")
    frozen = protocol["frozen_physics"]
    for group in ("physics_source_files", "data_files"):
        for name, expected in frozen[group].items():
            if file_hash(SOURCE / name) != expected:
                raise ValueError(f"Frozen physical dependency changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "registration.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing study registration")
    inherited = verify_previous_evidence()
    protocol = contracts()
    initializers = {
        str(seed): read_json(ROOT / f"bc_v1/history_{seed}/result.json")[
            "selected_checkpoint"
        ]
        for seed in range(1100, 1104)
    }
    baseline_path = ROOT / "development_baselines_v1/summary.json"
    baselines = read_json(baseline_path)["episodes"]
    analytical = [row for row in baselines if row["policy"]["id"] == "cooling_first"]
    if len(analytical) != 24 or not all(row["qualified"] for row in analytical):
        raise ValueError("The full analytic baseline panel must qualify before SAC")
    prior = [row for row in baselines if row["policy"]["id"] == "old_sac"]
    pid = [row for row in baselines if row["policy"]["id"] == "pid"]
    expected_cells = {
        (cell["start_day"], cell["seed"], cell["regd_load_bias"])
        for cell in protocol["development_cells"]
    }
    for name, rows in (("old_sac", prior), ("pid", pid)):
        observed_cells = {
            (
                row["cell"]["start_day"],
                row["cell"]["seed"],
                row["cell"]["regd_load_bias"],
            )
            for row in rows
        }
        if (
            len(rows) != 24
            or observed_cells != expected_cells
            or not all(row["complete"] for row in rows)
        ):
            raise ValueError(f"Incomplete or mismatched reference cohort: {name}")
    prior_relief = [
        row["grid"]["stress_facility_relief_mw"]
        for row in prior
        if row["cell"]["regd_load_bias"] == 0.5
    ]
    if len(prior_relief) != 12:
        raise ValueError("Expected exactly twelve biased reference cases")
    input_paths = [
        baseline_path,
        ROOT / "bc_development_v1/summary.json",
        ROOT / "reports/comparator_metrics.json",
        ROOT / "demonstrations_v1/summary.json",
        ROOT / "demonstrations_v1/manifest.json",
        *[
            MACRO_STUDY / name
            for name in (
                "macro_task.py",
                "recovery_ops.py",
                "study_common.py",
                "prepare.py",
            )
        ],
        Path(protocol["config_path"]),
        *[Path(item["path"]) for item in initializers.values()],
        *[
            ROOT / f"bc_v1/history_{seed}/{name}.json"
            for seed in range(1100, 1104)
            for name in ("result", "selection", "manifest")
        ],
    ]
    protocol.update(
        schema="table2-standalone-demonstration-initialized-sac-v1",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        phase="development_training_only",
        final_evaluation_requires_separate_locked_registration=True,
        implementation_revision=1,
        feature_contract=FEATURE_CONTRACT,
        variants=["warm_sac", "guided_sac"],
        seeds=list(range(1100, 1104)),
        initializers=initializers,
        corpus_path=str(ROOT / "demonstrations_v1"),
        n_envs=4,
        threads_per_process=1,
        cpu_groups=choose_cpu_groups(),
        gpus=list(range(8)),
        total_timesteps=153600,
        milestones=[4096, 8192, 16384, 32768, 65536, 153600],
        learning_starts=1024,
        critic_calibration_updates=256,
        bc_initial_gradient_ratio=1.0,
        reward="unchanged physical_v1 low-level seven-component reward",
        finite_endpoint="24-hour task is a true terminal; no timeout bootstrap",
        training_recovery_scope="evaluation only; physical 24h+2h closure retained",
        selection={
            "all_development_cells_must_qualify": True,
            "rmse_must_be_below_old_sac_kw": sum(
                row["service"]["tracking_rmse_kw"] for row in prior
            )
            / len(prior),
            "biased_relief_must_be_at_least_old_sac_mw": sum(prior_relief)
            / len(prior_relief),
            "closed_cash_must_be_at_least_pid_usd": sum(
                row["cashflow_26h_usd"] for row in pid
            )
            / len(pid),
            "score": "maximize biased mean facility relief; tie-break by lower RMSE",
            "per_training_seed": True,
            "bc_only_is_not_an_RL_checkpoint": True,
            "retain_all_failed_runs_and_cells": True,
        },
        algorithm={
            "learning_rate": 1e-4,
            "buffer_size": 200000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": 1e-5,
            "train_freq": 16,
            "gradient_steps": -1,
        },
        architecture=[128, 128],
        demonstration_teacher="causal_cooling_first; not the historical SAC reference",
        initializer_selection={
            "training_loss": "MSE of SB3-scaled actions",
            "selection_loss": "sum of native-unit action MSEs",
            "validation_subset": "fixed 65536 training-support rows, seeded per BC fit",
            "paired_arms": "same exact selected initializer for a given training seed",
            "no_posthoc_reselection": True,
            "producer_code": (
                "Each bc_v1 fit has its own immutable source/manifest snapshot; "
                "the new loader's label-bound assertions are a forward guard."
            ),
        },
        online_exposure={
            "if_no_early_termination": (
                "8 complete service episodes plus 4 prefixes; 12 dates, cycle zero only"
            ),
            "early_physical_failures": "can produce additional resets and cycles",
            "actual_support": "report episode journals and final prefixes per fit",
            "shared_external_data": "the fixed training-only demonstration corpus",
        },
        supplemental_metrics=[
            "24h modeled system maximum reduction at facility and PCC boundaries",
            "26h closed cashflow and battery throughput",
            "thermal warnings, peak temperatures and reference-relative heat debt",
            "per-date and per-training-seed paired differences",
        ],
        input_hashes={str(path): file_hash(path) for path in input_paths},
        code_hashes={name: file_hash(ROOT / name) for name in CODE_FILES},
        frozen_physics=inherited,
        python=sys.version.split()[0],
        reporting={
            "primary_boundary": "facility, excluding battery exchange",
            "revenue_contrast": "same controller: beta0.5 versus beta0",
            "all_four_training_seeds": True,
            "uncertainty": "descriptive; crossed date/noise/training seeds disclosed",
            "old_sac": (
                "frozen historical selected checkpoint; "
                "not a data-matched algorithm ablation"
            ),
            "analytic_inverter": "required non-RL comparator",
            "bc_only": "required supervised comparator",
            "no_automatic_RL_superiority_claim": True,
            "both_variants_reported": True,
            "no_posthoc_variant_selection": True,
            "development_selection": "best eligible of six post-update milestones",
            "gate_scopes": {
                "relief": "12 biased development cases",
                "rmse_and_cash": "24 development cases, both biases",
                "pid_cash_reference": "24 complete cases, including one gate failure",
            },
            "rl_increment_reference": "each seed's exact BC-only initializer",
            "dominance_checks": [
                "versus analytic controller",
                "versus paired BC-only initializer",
            ],
            "regression_panel": (
                "original peak dates are consumed, descriptive regression evidence"
            ),
            "sealed_panel": (
                "six unseen calendar combinations, balanced low/medium/high load; "
                "not six peak days or unseen workload phases"
            ),
            "disposition": {
                "no_eligible_fit": (
                    "No qualifying SAC improvement; retain all failures and "
                    "never relabel BC as RL or relax gates after evaluation."
                ),
                "eligible_but_dominated": (
                    "Report improvement relative to the named legacy references; "
                    "no RL advantage over the dominating analytic/BC comparator."
                ),
                "eligible_nondominated": (
                    "Report the measured tradeoff and paired BC increment; "
                    "no generic RL superiority claim."
                ),
            },
        },
    )
    protocol["registration_id"] = identity(protocol)
    verify_protocol(protocol)
    source = args.output.parent / (args.output.stem + "_source")
    source.mkdir(exist_ok=False)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, source / name)
    write_json(args.output, protocol)
    verify_protocol(protocol)
    print(protocol["registration_id"], flush=True)


if __name__ == "__main__":
    main()
