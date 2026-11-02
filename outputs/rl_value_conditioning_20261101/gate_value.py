"""Apply the unchanged whole-arm advancement rule to the complete union pilot."""

from __future__ import annotations

import json
from pathlib import Path

from ctx import ROOT, file_hash, identity, read_json, verify, write_json


def main():
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    destination = ROOT / "extension_decision.json"
    if destination.exists():
        raise FileExistsError("Never overwrite an advancement decision")
    status = read_json(ROOT / "pilot_status.json")
    expected = {
        (variant, seed)
        for variant in protocol["variants"]
        for seed in protocol["seeds"]
    }
    if (
        status["state"] != "finished"
        or status["registration_id"] != protocol["registration_id"]
        or len(status["jobs"]) != len(expected)
        or {(job["variant"], job["seed"]) for job in status["jobs"]} != expected
        or any(job["status"] != "completed" for job in status["jobs"])
        or any(
            Path(job["output"]) != ROOT / (f"runs/pilot/{job['variant']}_{job['seed']}")
            for job in status["jobs"]
        )
    ):
        raise ValueError("All eight exact native jobs are required before advancement")
    directories = {f"{variant}_{seed}" for variant, seed in expected}
    if {path.name for path in (ROOT / "runs/pilot").iterdir()} != directories:
        raise ValueError("Additional attempts need explicit adjudication")
    failures = sorted(str(path) for path in (ROOT / "runs").rglob("failure.json"))
    if failures:
        write_json(ROOT / "advancement_blocked.json", {"failures": failures})
        raise ValueError("A failed production attempt cannot disappear")
    arms, paired, exposure = {}, {}, {}
    for variant in protocol["variants"]:
        rows = []
        for seed in protocol["seeds"]:
            directory = ROOT / f"runs/pilot/{variant}_{seed}"
            complete = read_json(directory / "completed.json")
            if (
                complete["stage"] != "pilot"
                or complete["smoke_only"]
                or complete["seed"] != seed
                or complete["variant"] != variant
                or complete["status"] != "completed"
                or complete["online_transitions"]
                != protocol["stage_milestones"]["pilot"][-1]
                or complete["registration_id"] != protocol["registration_id"]
                or complete["source_hashes"] != protocol["source_hashes"]
            ):
                raise ValueError("Receipt does not match its registered slot")
            initial = read_json(directory / "initialization.json")
            calibration = read_json(directory / "calibration.json")
            if not initial["historical_raw_prefix_exact"]:
                raise AssertionError("A fit did not preserve the historical raw prefix")
            if calibration["historical_critic_equality_expected"] != (
                variant == "union_original"
            ):
                raise AssertionError("Historical critic comparison was misclassified")
            paired.setdefault(seed, []).append(
                {
                    **{
                        key: initial[key]
                        for key in (
                            "actor",
                            "critic",
                            "critic_target",
                            "reference",
                            "warmup_replay",
                            "warmup_transitions",
                        )
                    },
                    "calibrated_actor": calibration["actor_digest_unchanged"],
                }
            )
            journal = [
                json.loads(line)
                for line in (directory / "training_episodes.jsonl")
                .read_text()
                .splitlines()
            ]
            prefixes = read_json(directory / "training_prefixes.json")
            exposure.setdefault(seed, []).append(
                {
                    "episodes": [
                        {
                            "condition": row["condition"],
                            "steps": row["steps"],
                            "complete": row["complete"],
                        }
                        for row in journal
                    ],
                    "prefixes": [
                        {"condition": row["condition"], "steps": row["steps"]}
                        for row in prefixes
                    ],
                }
            )
            final = complete["final"]
            rows.append(
                {
                    "seed": seed,
                    "extension_qualified": final["extension_qualified"],
                    "rmse_kw": final["rmse_kw"],
                    "qualified_cells": final["qualified_cells"],
                    "rmse_ratio_to_bc": final["rmse_ratio_to_bc"],
                    "rmse_ratio_to_guided": final["rmse_ratio_to_guided"],
                    "increment_vs_bc": final["increment_vs_bc"],
                    "disqualification_reasons": final["disqualification_reasons"],
                    "receipt_sha256": file_hash(directory / "completed.json"),
                }
            )
        arms[variant] = {
            "fits": rows,
            "all_four_pass": all(row["extension_qualified"] for row in rows),
        }
    if any(pair[0] != pair[1] for pair in paired.values()):
        raise AssertionError(
            "Union arms did not share initial parameters and raw warmup"
        )
    decision = {
        "registration_id": protocol["registration_id"],
        "extend_both_arms": any(arm["all_four_pass"] for arm in arms.values()),
        "arms": arms,
        "paired_raw_prefix_exact": True,
        "conditioned_critic_calibration_intentionally_allowed_to_differ": True,
        "training_conditions_match_by_seed": {
            str(seed): pair[0] == pair[1] for seed, pair in exposure.items()
        },
        "condition_sequences": {str(seed): pair for seed, pair in exposure.items()},
        "source_status_sha256": file_hash(ROOT / "pilot_status.json"),
        "failure_inventory": failures,
        "rule": protocol["extension"],
    }
    decision["decision_id"] = identity(decision)
    write_json(destination, decision)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
