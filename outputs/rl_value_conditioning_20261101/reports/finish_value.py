"""Lock final-step models, check full-macro readiness, then open Final-B once."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# isort: off
from ctx import (
    ROOT,
    PRIOR,
    BASELINE,
    PROJECT,
    read_json,
    write_json,
    file_hash,
    identity,
    verify,
)
from panel import run_panel
from study import verify_registration
from common import case_key
# isort: on


def require_unblocked():
    for name in ("final_blocked.json", "final_blocked_regression.json"):
        if (ROOT / name).exists():
            raise RuntimeError(
                f"Finalization was blocked; explicit adjudication is required: {name}"
            )


def mark_attempt(name, protocol):
    path = ROOT / f"{name}_attempt.json"
    if path.exists():
        raise FileExistsError(f"The {name} stage was already attempted")
    write_json(
        path,
        {
            "registration_id": protocol["registration_id"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "producer_sha256": file_hash(__file__),
        },
    )


def primary_inventory(protocol):
    decision = read_json(ROOT / "extension_decision.json")
    if not decision["extend_both_arms"]:
        raise ValueError("No extension was earned; Final-B stays sealed")
    status = read_json(ROOT / "extended_status.json")
    expected = {
        (variant, seed)
        for variant in protocol["variants"]
        for seed in protocol["seeds"]
    }
    if (
        status["state"] != "finished"
        or len(status["jobs"]) != 8
        or {(job["variant"], job["seed"]) for job in status["jobs"]} != expected
        or any(job["status"] != "completed" for job in status["jobs"])
    ):
        raise ValueError("All eight complete extended fits are required")
    inventory = []
    for variant in protocol["variants"]:
        completed = []
        for seed in protocol["seeds"]:
            path = ROOT / f"runs/extended/{variant}_{seed}/completed.json"
            result = read_json(path)
            if (
                result["status"] != "completed"
                or result["stage"] != "extended"
                or result["seed"] != seed
                or result["variant"] != variant
                or result["registration_id"] != protocol["registration_id"]
                or result["source_hashes"] != protocol["source_hashes"]
                or result["online_transitions"] != 552960
                or result["final"]["timesteps"] != 552960
            ):
                raise ValueError("Extended receipt mismatches its final slot")
            checkpoint = result["final"]["checkpoint"]
            if Path(checkpoint["path"]) != path.parent / "step_552960.zip":
                raise ValueError("The locked model is not the final-step checkpoint")
            if file_hash(checkpoint["path"]) != checkpoint["sha256"]:
                raise ValueError("Final-step checkpoint changed")
            completed.append((seed, result, path))
        admitted = all(item["final"]["extension_qualified"] for _, item, _ in completed)
        for seed, result, path in completed:
            inventory.append(
                {
                    "id": f"{variant}_{seed}",
                    "family": variant,
                    "training_seed": seed,
                    "hardware": "history_sac",
                    "checkpoint": result["final"]["checkpoint"],
                    "available": admitted,
                    "training_registration_id": protocol["registration_id"],
                    "training_transitions": result["online_transitions"],
                    "completion_sha256": file_hash(path),
                    "role": "candidate",
                    "selection_provenance": (
                        "Final 552960 step, whole-arm four-seed gate"
                    ),
                    "unavailable_reason": None
                    if admitted
                    else (
                        "Whole arm failed unchanged final-step, "
                        "four-seed admission criteria"
                    ),
                }
            )
    return inventory


def full_readiness(protocol, cpus):
    require_unblocked()
    candidates = primary_inventory(protocol)
    if not any(policy["available"] for policy in candidates):
        write_json(
            ROOT / "final_blocked.json",
            {
                "reason": "Neither full-budget arm qualifies",
                "inventory": candidates,
            },
        )
        raise ValueError("No complete arm is ready; no final data will be opened")
    controls = [
        {
            key: policy[key]
            for key in ("id", "family", "hardware", "training_seed", "checkpoint")
            if key in policy
        }
        for policy in read_json(BASELINE / "final_candidate_lock.json")["inventory"]
        if policy["available"]
    ]
    policies = [
        {
            **copy.deepcopy(policy),
            "id": f"full__{policy['id']}",
            "macro": "full_residual",
        }
        for policy in [*controls, *candidates]
        if policy.get("available", True)
    ]
    mark_attempt("full_macro_readiness", protocol)
    return run_panel(
        protocol,
        policies,
        protocol["development_cells"],
        ROOT / "full_macro_readiness",
        cpus,
        role="development",
    )


def make_lock(protocol):
    require_unblocked()
    destination = ROOT / "candidate_lock.json"
    if destination.exists():
        raise FileExistsError("The candidate lock is immutable")
    candidates = primary_inventory(protocol)
    readiness = read_json(ROOT / "full_macro_readiness/summary.json")
    if readiness["runtime_failures"]:
        raise ValueError("Full-macro readiness has unresolved runtime failures")
    candidate_by_id = {item["id"]: item for item in candidates}
    for row in readiness["episodes"]:
        policy = row["policy"]
        if policy["family"] in protocol["variants"]:
            candidate = candidate_by_id[policy["id"].removeprefix("full__")]
            if policy["checkpoint"] != candidate["checkpoint"]:
                raise ValueError(
                    "Full-macro readiness evaluated a different checkpoint"
                )
    full_eligible = {}
    for variant in protocol["variants"]:
        rows = [
            row for row in readiness["episodes"] if row["policy"]["family"] == variant
        ]
        full_eligible[variant] = bool(
            len(rows) == 4 * len(protocol["development_cells"])
            and all(row["qualified"] and row["delivered_service_gate"] for row in rows)
        )
    original = read_json(BASELINE / "final_candidate_lock.json")["inventory"]
    slots = []
    for policy in original:
        slots.append(
            {
                key: policy[key]
                for key in (
                    "id",
                    "family",
                    "hardware",
                    "available",
                    "training_seed",
                    "checkpoint",
                    "unavailable_reason",
                )
                if key in policy
            }
        )
        slots[-1]["role"] = (
            "frozen_comparator" if policy["available"] else "historical_non_admitted"
        )
        slots[-1]["selection_provenance"] = (
            "Previously locked slot; not reselected on new development outcomes"
        )
    prior_protocol = read_json(PRIOR / "registration.json")
    prior_decision = read_json(PRIOR / "extension_decision.json")
    if prior_decision["extend_both_arms"]:
        raise ValueError("First-pilot disposition changed")
    for variant in prior_protocol["variants"]:
        for seed in prior_protocol["seeds"]:
            result = read_json(PRIOR / f"runs/pilot/{variant}_{seed}/completed.json")
            slots.append(
                {
                    "id": f"{variant}_{seed}",
                    "family": variant,
                    "training_seed": seed,
                    "hardware": "history_sac",
                    "available": False,
                    "role": "historical_non_admitted",
                    "checkpoint": result["final"]["checkpoint"],
                    "unavailable_reason": (
                        "First pilot failed the registered whole-arm rule"
                    ),
                }
            )
    slots.extend(candidates)
    headline_scores = {}
    for variant in protocol["variants"]:
        if all(item["available"] for item in candidates if item["family"] == variant):
            headline_scores[variant] = sum(
                read_json(ROOT / f"runs/extended/{variant}_{seed}/completed.json")[
                    "final"
                ]["rmse_kw"]
                for seed in protocol["seeds"]
            ) / len(protocol["seeds"])
    inventory = []
    for macro in ("rule", "full_residual"):
        for slot in slots:
            value = copy.deepcopy(slot)
            value.update(id=f"{macro}__{slot['id']}", macro=macro)
            if macro == "full_residual" and slot["family"] in full_eligible:
                value["available"] = slot["available"] and full_eligible[slot["family"]]
                if not value["available"]:
                    value["unavailable_reason"] = (
                        "Rule-macro admission rejected; "
                        "full-macro readiness not evaluated"
                        if not slot["available"]
                        else "Full-macro physical/service readiness failed"
                    )
            inventory.append(value)
    if not any(
        slot["available"] and slot["family"] in protocol["variants"]
        for slot in inventory
    ):
        raise ValueError("No new method qualifies for final evaluation")
    lock = {
        "registration_id": protocol["registration_id"],
        "inventory": inventory,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "final_cells": protocol["final_cells"],
        "final_step_only_new_families": protocol["variants"],
        "frozen_comparators_not_filtered_by_new_readiness_outcomes": True,
        "no_survivor_only_family_mean": True,
        "headline_family": min(
            headline_scores, key=lambda name: (headline_scores[name], name)
        ),
        "headline_selection": (
            "Lowest final-step mean Dev-B RMSE among complete admitted arms"
        ),
        "headline_development_scores_kw": headline_scores,
        "readiness_summary_sha256": file_hash(
            ROOT / "full_macro_readiness/summary.json"
        ),
        "first_pilot_decision_sha256": file_hash(PRIOR / "extension_decision.json"),
        "producer_sha256": file_hash(__file__),
    }
    lock["candidate_lock_id"] = identity(lock)
    write_json(destination, lock)
    old_final = read_json(BASELINE / "final_registration.json")
    evaluation = copy.deepcopy(protocol)
    evaluation.pop("registration_id")
    evaluation.update(
        schema="locked-consumed-peak-regression-before-Final-B-v1",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        training_registration_id=protocol["registration_id"],
        candidate_lock_id=lock["candidate_lock_id"],
        regression_cells=old_final["panels"]["regression"],
        no_checkpoint_selection_on_regression=True,
        regression_claim="Consumed peak dates, not held out",
        required_regression_readiness=(
            "Complete physical/recovery gates; same high service floor"
        ),
        physical_readiness_uses_consumed_regression=True,
    )
    evaluation["input_hashes"][str(destination)] = file_hash(destination)
    evaluation["source_hashes"][str(Path(__file__).resolve())] = file_hash(__file__)
    evaluation["registration_id"] = identity(evaluation)
    verify_registration(evaluation)
    write_json(ROOT / "regression_registration.json", evaluation)
    print(lock["candidate_lock_id"], flush=True)


def read_lock(protocol):
    lock = read_json(ROOT / "candidate_lock.json")
    body = {key: value for key, value in lock.items() if key != "candidate_lock_id"}
    if (
        identity(body) != lock["candidate_lock_id"]
        or lock["registration_id"] != protocol["registration_id"]
        or lock["producer_sha256"] != file_hash(__file__)
    ):
        raise ValueError("Candidate lock or producer changed")
    return lock


def regression(protocol, cpus):
    require_unblocked()
    lock = read_lock(protocol)
    evaluation = read_json(ROOT / "regression_registration.json")
    policies = [
        slot
        for slot in lock["inventory"]
        if slot["available"] and slot["family"] in protocol["variants"]
    ]
    mark_attempt("peak_regression", evaluation)
    return run_panel(
        evaluation,
        policies,
        evaluation["regression_cells"],
        ROOT / "peak_regression",
        cpus,
        role="regression",
    )


def open_final(protocol, cpus):
    require_unblocked()
    lock = read_lock(protocol)
    regression_summary = read_json(ROOT / "peak_regression/summary.json")
    if regression_summary["runtime_failures"]:
        raise ValueError("Unresolved regression runtime errors")
    evaluation = read_json(ROOT / "regression_registration.json")
    expected = {
        (slot["id"], case_key(cell))
        for slot in lock["inventory"]
        if slot["available"] and slot["family"] in protocol["variants"]
        for cell in evaluation["regression_cells"]
    }
    observed = {
        (row["policy"]["id"], case_key(row["cell"]))
        for row in regression_summary["episodes"]
    }
    if observed != expected or len(regression_summary["episodes"]) != len(expected):
        raise ValueError("Peak-date readiness lost a locked policy or condition")
    bad = [
        row
        for row in regression_summary["episodes"]
        if not row["qualified"]
        or (
            row["policy"]["macro"] == "full_residual"
            and not row["delivered_service_gate"]
        )
    ]
    if bad:
        write_json(
            ROOT / "final_blocked_regression.json",
            {
                "failed_cells": [
                    {"policy": row["policy"]["id"], "cell": row["cell"]} for row in bad
                ],
                "no_final_outcomes_observed": True,
            },
        )
        raise ValueError(
            "Known peak-date physical readiness failed; Final-B stays sealed"
        )
    shared = PRIOR / "final_lock.json"
    if shared.exists() or (PRIOR / "final_opening.json").exists():
        raise FileExistsError(
            "The shared one-look final panel is already locked or opened"
        )
    final = {
        "registration_id": protocol["registration_id"],
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_lock_id": lock["candidate_lock_id"],
        "cells": protocol["final_cells"],
        "policies": [slot for slot in lock["inventory"] if slot["available"]],
        "all_method_slots_including_non_admitted": lock["inventory"],
        "regression_summary_sha256": file_hash(ROOT / "peak_regression/summary.json"),
        "producer_sha256": file_hash(__file__),
    }
    final["lock_id"] = identity(final)
    write_json(shared, final)
    write_json(ROOT / "final_lock.json", final)
    print(
        "Final lock written; commit both lock files before evaluate-final", flush=True
    )


def evaluate_final(protocol, cpus):
    require_unblocked()
    final = read_json(ROOT / "final_lock.json")
    paths = [
        str((ROOT / "final_lock.json").relative_to(PROJECT)),
        str((PRIOR / "final_lock.json").relative_to(PROJECT)),
        str((ROOT / "candidate_lock.json").relative_to(PROJECT)),
        str((ROOT / "regression_registration.json").relative_to(PROJECT)),
        str((ROOT / "peak_regression/summary.json").relative_to(PROJECT)),
        str((ROOT / "full_macro_readiness/summary.json").relative_to(PROJECT)),
    ]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        cwd=PROJECT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=PROJECT,
        check=True,
    )
    if read_json(PRIOR / "final_lock.json") != final:
        raise ValueError("Shared and study final locks differ")
    return run_panel(
        protocol,
        final["policies"],
        protocol["final_cells"],
        ROOT / "final_B",
        cpus,
        role="final",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "full-development",
            "lock",
            "regression",
            "lock-final",
            "evaluate-final",
        ),
    )
    parser.add_argument("--cpus", type=int, nargs="+")
    args = parser.parse_args()
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    cpus = args.cpus or [cpu for group in protocol["cpu_groups"] for cpu in group]
    if args.action == "full-development":
        full_readiness(protocol, cpus)
    elif args.action == "lock":
        make_lock(protocol)
    elif args.action == "regression":
        regression(protocol, cpus)
    elif args.action == "lock-final":
        open_final(protocol, cpus)
    else:
        evaluate_final(protocol, cpus)
