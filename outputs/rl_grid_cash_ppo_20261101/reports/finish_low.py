"""Correct baseline lookup by policy metadata; keep registered evaluation immutable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "track_a"))

# isort: off
from evaluate_low import collect_group, family_means, summarize
from economy_context import case_key, file_hash, read_json, verify, write_json
import numpy as np
# isort: on

ROOT = Path(__file__).resolve().parents[1] / "track_a"


def bc_reference_index(by_policy):
    references = {}
    for rows in by_policy.values():
        policy = rows[0]["policy"]
        if policy["family"] != "bc_only" or policy["macro"] != "rule":
            continue
        seed = policy["training_seed"]
        if seed in references:
            raise ValueError(f"Ambiguous BC reference for initializer {seed}")
        references[seed] = rows
    return references


def main():
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    baseline, _, checks = collect_group(protocol, "baselines")
    learned, missing, count = collect_group(protocol, "full")
    checks += count
    references = bc_reference_index(baseline)
    if set(references) != set(protocol["seeds"]):
        raise ValueError("The complete matched BC initializer set is required")
    frozen = family_means(baseline)
    families = family_means(
        learned,
        {f"native_{name}": 4 for name in protocol["algorithms"]},
    )
    eligible = {name: row for name, row in frozen.items() if row["all_qualified"]}
    if not eligible:
        raise ValueError("No fully qualified reference family")
    envelope = {
        bias: {
            metric: max(row["strata"][bias][metric] for row in eligible.values())
            for metric in ("cash_usd", "facility_relief_mw", "pcc_relief_mw")
        }
        for bias in ("all", "0.0", "0.5")
    }
    per_model = {}
    tolerance = protocol["selection"]["mean_rmse_noninferiority_tolerance_kw"]
    for name, rows in learned.items():
        reference = references[rows[0]["policy"]["training_seed"]]
        if [case_key(row["cell"]) for row in rows] != [
            case_key(row["cell"]) for row in reference
        ]:
            raise ValueError("Candidate and BC cases are not paired")
        result = summarize(rows)
        differences = {}
        if all(row["complete"] for row in rows):
            for bias in (0.0, 0.5):
                differences[str(bias)] = float(
                    np.mean(
                        [
                            row["service"]["tracking_rmse_kw"]
                            - other["service"]["tracking_rmse_kw"]
                            for row, other in zip(rows, reference, strict=True)
                            if row["cell"]["regd_load_bias"] == bias
                        ]
                    )
                )
        result["rmse_change_vs_own_bc_by_bias_kw"] = differences
        result["tracking_noninferior"] = bool(
            result["all_qualified"]
            and differences
            and all(value <= tolerance for value in differences.values())
        )
        result["individual_cases_worse_than_bc"] = [
            case_key(row["cell"])
            for row, other in zip(rows, reference, strict=True)
            if row["complete"]
            and row["service"]["tracking_rmse_kw"]
            > other["service"]["tracking_rmse_kw"] + tolerance
        ]
        per_model[name] = result
    decisions = {}
    for algorithm in protocol["algorithms"]:
        family = f"native_{algorithm}"
        slots = [f"rule__{family}_{seed}" for seed in protocol["seeds"]]
        complete = all(name in per_model for name in slots)
        tracking = complete and all(
            per_model[name]["tracking_noninferior"] for name in slots
        )
        row, contrasts, joint = families.get(family), {}, False
        if complete and row["complete_cohort"]:
            contrasts = {
                bias: {
                    metric: row["strata"][bias][metric] - value
                    for metric, value in envelope[bias].items()
                }
                for bias in envelope
            }
            joint = bool(
                tracking
                and contrasts["all"]["cash_usd"]
                >= protocol["selection"]["cash_gain_threshold_usd"]
                and all(contrasts[bias]["cash_usd"] >= 0 for bias in ("0.0", "0.5"))
                and all(
                    contrasts["0.5"][metric]
                    >= protocol["selection"]["relief_gain_threshold_mw"]
                    for metric in ("facility_relief_mw", "pcc_relief_mw")
                )
            )
        decisions[algorithm] = {
            "all_four_final_models_evaluated": complete,
            "all_four_noninferior_to_own_bc": tracking,
            "joint_improvement": joint,
            "contrasts_vs_frozen_family_envelope": contrasts,
        }
    path = ROOT / "development_report.json"
    if path.exists():
        raise FileExistsError("Never overwrite an audited development report")
    write_json(
        path,
        {
            "registration_id": protocol["registration_id"],
            "role": "consumed_development_only",
            "per_model": per_model,
            "frozen_families": frozen,
            "learned_families": families,
            "missing_training_slots": missing,
            "decisions": decisions,
            "raw_primary_checks": checks,
            "final_data_not_opened": True,
            "analysis_correction": {
                "reason": "Match BC by family/macro/seed, not the full__ prefix",
                "registered_source_unchanged": True,
                "original_error": "KeyError: rule__bc_only_1100",
                "producer": str(Path(__file__).resolve()),
                "producer_sha256": file_hash(__file__),
            },
        },
    )
    print(decisions)


if __name__ == "__main__":
    main()
