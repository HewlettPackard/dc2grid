"""Execute registered development controls and evaluate explicit continuation rules."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

# isort: off
from study import ROOT, read_json, write_json, file_hash, identity, verify_registration
from panel import run_panel
from low_learn import summarize
import numpy as np
# isort: on


def reference_baselines(protocol, cpus):
    if (ROOT / "pilot_status.json").exists() or (ROOT / "runs/pilot").exists():
        raise RuntimeError(
            "Comparator constants must be frozen before production RL begins"
        )
    output = ROOT / "development_baselines"
    result = run_panel(
        protocol,
        protocol["frozen_baseline_policies"],
        protocol["development_cells"],
        output,
        cpus,
        role="development",
    )
    per_initializer = {}
    for seed in protocol["seeds"]:
        reference = {}
        for name, family in (("bc", "bc_only"), ("guided", "guided_sac")):
            rows = [
                row
                for row in result["episodes"]
                if row["policy"]["family"] == family
                and row["policy"]["training_seed"] == seed
            ]
            if len(rows) != 24:
                raise ValueError("Missing a matched reference condition")
            summary = summarize(rows)
            if summary["complete_cells"] != 24 or summary["cash_usd"] is None:
                raise ValueError(
                    "Reference failed to complete; cannot fabricate full-case means"
                )
            reference[name] = summary
        bc_bound = (
            protocol["extension"]["maximum_rmse_ratio_to_own_BC"]
            * reference["bc"]["rmse_kw"]
        )
        guided_bound = (
            protocol["extension"][
                "maximum_rmse_ratio_to_locked_Guided_same_initializer"
            ]
            * reference["guided"]["rmse_kw"]
        )
        reference["literal_extension_rmse_bound_kw"] = min(bc_bound, guided_bound)
        reference["binding_comparator"] = "bc" if bc_bound <= guided_bound else "guided"
        per_initializer[str(seed)] = reference
    reference = {
        "registration_id": protocol["registration_id"],
        "panel_summary": {
            "path": str(output / "summary.json"),
            "sha256": file_hash(output / "summary.json"),
        },
        "per_initializer": per_initializer,
        "frozen_before_reinforcement_training": True,
    }
    reference["reference_id"] = identity(reference)
    destination = ROOT / "development_references.json"
    if destination.exists():
        raise FileExistsError("Never overwrite a resolved comparator lock")
    write_json(destination, reference)
    print(per_initializer, flush=True)


def extension_decision(protocol):
    destination = ROOT / "extension_decision.json"
    if destination.exists():
        raise FileExistsError("The extension decision is immutable")
    expected = {
        (variant, seed)
        for variant in protocol["variants"]
        for seed in protocol["seeds"]
    }
    status = read_json(ROOT / "pilot_status.json")
    if (
        status["registration_id"] != protocol["registration_id"]
        or status["state"] != "finished"
        or len(status["jobs"]) != len(expected)
        or {(job["variant"], job["seed"]) for job in status["jobs"]} != expected
        or any(job["status"] != "completed" for job in status["jobs"])
        or any(
            Path(job["output"]) != ROOT / (f"runs/pilot/{job['variant']}_{job['seed']}")
            for job in status["jobs"]
        )
    ):
        raise ValueError("Extension needs all eight exact completed supervisor slots")
    expected_directories = {f"{variant}_{seed}" for variant, seed in expected}
    if {path.name for path in (ROOT / "runs/pilot").iterdir()} != expected_directories:
        raise ValueError(
            "Additional pilot attempts exist and require explicit adjudication"
        )
    failures = sorted(str(path) for path in (ROOT / "runs").rglob("failure.json"))
    if failures:
        write_json(ROOT / "extension_blocked_failures.json", {"failures": failures})
        raise RuntimeError("Unadjudicated failed production attempts block extension")
    arms, pairs = {}, {}
    for variant in protocol["variants"]:
        records = []
        for seed in protocol["seeds"]:
            path = ROOT / f"runs/pilot/{variant}_{seed}"
            complete = read_json(path / "completed.json")
            if (
                complete["status"] != "completed"
                or complete["online_transitions"] != 69120
                or complete["registration_id"] != protocol["registration_id"]
                or complete["stage"] != "pilot"
                or complete["smoke_only"]
                or complete["variant"] != variant
                or complete["seed"] != seed
                or complete["source_hashes"] != protocol["source_hashes"]
            ):
                raise ValueError(
                    "A full, registered pilot receipt is required per slot"
                )
            records.append(
                {
                    "seed": seed,
                    "qualified": complete["final"]["extension_qualified"],
                    "summary": {
                        key: complete["final"][key]
                        for key in (
                            "rmse_kw",
                            "all_qualified",
                            "rmse_ratio_to_bc",
                            "rmse_ratio_to_guided",
                            "increment_vs_bc",
                            "disqualification_reasons",
                        )
                    },
                    "receipt_sha256": file_hash(path / "completed.json"),
                }
            )
            initial = read_json(path / "initialization.json")
            calibration = read_json(path / "calibration.json")
            pairs.setdefault(seed, []).append(
                {
                    "actor": initial["actor"],
                    "critic": initial["critic"],
                    "critic_target": initial["critic_target"],
                    "reference": initial["reference"],
                    "warmup_replay": initial["warmup_replay"],
                    "calibrated_actor": calibration["actor_digest_unchanged"],
                    "calibrated_critic": calibration["critic_digest"],
                    "calibrated_critic_target": calibration["critic_target_digest"],
                }
            )
        arms[variant] = {
            "all_four_pass": all(record["qualified"] for record in records),
            "fits": records,
        }
    if any(pair[0] != pair[1] for pair in pairs.values()):
        raise AssertionError(
            "Paired initializations, warmup data or critic calibration differ"
        )
    decision = {
        "registration_id": protocol["registration_id"],
        "extend_both_arms": any(arm["all_four_pass"] for arm in arms.values()),
        "arms": arms,
        "paired_initialization_and_warmup_exact": True,
        "rule": protocol["extension"],
        "failure_inventory": failures,
        "supervisor_status_sha256": file_hash(ROOT / "pilot_status.json"),
    }
    decision["decision_id"] = identity(decision)
    write_json(destination, decision)
    print(decision, flush=True)


def macro_development(protocol, cpus):
    forecast = read_json(ROOT / "forecast_v1/report.json")
    hardware = next(
        row
        for row in protocol["frozen_baseline_policies"]
        if row["id"] == protocol["high_level"]["fixed_hardware_for_selection"]
    )
    policies = [
        {
            **copy.deepcopy(hardware),
            "id": "macro_full_reference",
            "macro": "full_residual",
        }
    ]
    for forecaster, threshold in protocol["high_level"]["candidate_grid"]:
        policies.append(
            {
                **copy.deepcopy(hardware),
                "id": f"macro_{forecaster}_threshold_{threshold:g}",
                "macro": "forecast_quantity",
                "forecaster": forecaster,
                "margin_threshold_usd_mw_h": threshold,
                "forecast_checkpoint": forecast["model"],
            }
        )
    output = ROOT / "macro_development"
    results = run_panel(
        protocol,
        policies,
        protocol["development_cells"],
        output,
        cpus,
        role="development",
    )

    def key(row):
        return tuple(
            row["cell"][name] for name in ("start_day", "seed", "regd_load_bias")
        )

    reference = {
        key(row): row
        for row in results["episodes"]
        if row["policy"]["id"] == "macro_full_reference"
    }
    records = []
    for policy in policies[1:]:
        rows = [
            row for row in results["episodes"] if row["policy"]["id"] == policy["id"]
        ]
        if len(rows) != len(reference):
            raise ValueError("Macro comparison lost a matched cell")
        qualified = all(
            row["qualified"] and row["delivered_service_gate"] for row in rows
        )
        gains = (
            [
                row["cashflow_closed_recovery_usd"]
                - reference[key(row)]["cashflow_closed_recovery_usd"]
                for row in rows
            ]
            if all(
                row["cashflow_closed_recovery_usd"] is not None
                and reference[key(row)]["cashflow_closed_recovery_usd"] is not None
                for row in rows
            )
            else None
        )
        by_bias = (
            {
                str(bias): float(
                    np.mean(
                        [
                            gain
                            for row, gain in zip(rows, gains, strict=True)
                            if row["cell"]["regd_load_bias"] == bias
                        ]
                    )
                )
                for bias in (0.0, 0.5)
            }
            if gains is not None
            else None
        )
        mean = float(np.mean(gains)) if gains is not None else None
        records.append(
            {
                "policy": policy,
                "all_qualified_and_service_floor": qualified,
                "mean_paired_cash_gain_usd": mean,
                "by_bias_cash_gain_usd": by_bias,
                "advanced": bool(
                    qualified
                    and mean is not None
                    and mean
                    >= protocol["high_level"][
                        "minimum_mean_paired_cash_gain_usd_for_advancement"
                    ]
                    and all(gain >= 0 for gain in by_bias.values())
                ),
                "paired_gains": [
                    {"cell": row["cell"], "cash_gain_usd": gain}
                    for row, gain in zip(rows, gains, strict=True)
                ]
                if gains is not None
                else None,
            }
        )
    eligible = [row for row in records if row["advanced"]]
    selected = (
        max(eligible, key=lambda row: row["mean_paired_cash_gain_usd"])
        if eligible
        else None
    )
    write_json(
        output / "selection.json",
        {
            "candidates": records,
            "selected": selected,
            "no_gain_is_a_valid_result": True,
            "forecast_is_not_a_learned_high_level_SAC": True,
        },
    )
    print(records, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baselines", "extension", "macro"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "registration.json")
    parser.add_argument("--cpus", nargs="+", type=int)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    verify_registration(protocol)
    if args.action == "baselines":
        reference_baselines(protocol, args.cpus)
    elif args.action == "extension":
        extension_decision(protocol)
    else:
        macro_development(protocol, args.cpus or protocol["secondary_evaluation_cpus"])
