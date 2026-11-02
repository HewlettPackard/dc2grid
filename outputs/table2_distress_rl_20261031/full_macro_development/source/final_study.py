"""Lock and evaluate Table 2 candidates without reselection on final conditions."""

from __future__ import annotations

import argparse
import copy

# isort: off
from common import ROOT, file_hash, identity, read_json, write_json
from evaluate import run_panel
from prepare import choose_cpu_groups
from prepare_study import verify_protocol
# isort: on


def model_inventory(protocol):
    inventory = [
        {
            "id": name,
            "family": name,
            "hardware": hardware,
            "available": True,
            "training_seed": None,
        }
        for name, hardware in (
            ("nominal", "none"),
            ("pid", "pid"),
            ("rule_hardware", "rule"),
            ("bang_bang", "bang_bang"),
            ("old_sac", "old_sac"),
            ("cooling_first", "cooling_first"),
        )
    ]
    for seed in protocol["seeds"]:
        inventory.append(
            {
                "id": f"bc_only_{seed}",
                "family": "bc_only",
                "hardware": "history_sac",
                "available": True,
                "training_seed": seed,
                "checkpoint": protocol["initializers"][str(seed)],
            }
        )
    admitted = {}
    for variant in protocol["variants"]:
        admitted[variant] = 0
        for seed in protocol["seeds"]:
            directory = ROOT / "runs" / f"{variant}_{seed}"
            selection_path = directory / "selection.json"
            complete_path = directory / "completed.json"
            selected = (
                read_json(selection_path)["selected"]
                if selection_path.exists()
                else None
            )
            completed = read_json(complete_path) if complete_path.exists() else None
            available = bool(
                selected is not None
                and completed is not None
                and completed["status"] == "completed"
                and not completed["smoke_only"]
            )
            policy = {
                "id": f"{variant}_{seed}",
                "family": variant,
                "hardware": "history_sac",
                "available": available,
                "training_seed": seed,
                "selection": selected,
                "native_completion": completed,
                "selection_sha256": file_hash(selection_path)
                if selection_path.exists()
                else None,
            }
            if available:
                policy["checkpoint"] = selected["checkpoint"]
                admitted[variant] += 1
            else:
                policy["unavailable_reason"] = (
                    "No eligible checkpoint with a completed registered fit; "
                    "slot retained, never averaged as a successful seed."
                )
            inventory.append(policy)
    if max(admitted.values(), default=0) != len(protocol["seeds"]):
        raise RuntimeError(
            "No SAC variant has all four eligible seeds; final panel remains sealed"
        )
    return inventory, admitted


def prepare():
    output = ROOT / "final_candidate_lock.json"
    if output.exists():
        raise FileExistsError("Candidate lock already exists")
    protocol = read_json(ROOT / "registration.json")
    verify_protocol(protocol)
    status = read_json(ROOT / "training_status.json")
    if status["state"] != "finished" or len(status["jobs"]) != 8:
        raise RuntimeError("Wait for all eight fits before locking candidates")
    inventory, admitted = model_inventory(protocol)
    body = {
        "training_registration_id": protocol["registration_id"],
        "training_status_sha256": file_hash(ROOT / "training_status.json"),
        "inventory": inventory,
        "admitted_per_variant": admitted,
        "both_variants_and_all_seed_slots_retained": True,
        "selection_uses_development_only": True,
        "no_selection_from_final_panels": True,
    }
    body["lock_id"] = identity(body)
    write_json(output, body)
    print("Locked candidates:", body["lock_id"], admitted, flush=True)


def conditions(protocol, role):
    if role == "sealed":
        return copy.deepcopy(protocol["sealed_cells"])
    if role == "regression":
        return [
            {"start_day": day, "seed": seed, "regd_load_bias": bias}
            for day in (207, 248, 249)
            for seed in range(100, 105)
            for bias in (0.0, 0.5)
        ]
    if role == "development":
        return copy.deepcopy(protocol["development_cells"])
    raise ValueError("Unknown evaluation role")


def policies_for(lock, macros):
    return [
        {**copy.deepcopy(item), "id": f"{macro}__{item['id']}", "macro": macro}
        for macro in macros
        for item in lock["inventory"]
    ]


def load_lock():
    lock = read_json(ROOT / "final_candidate_lock.json")
    if identity({k: v for k, v in lock.items() if k != "lock_id"}) != lock["lock_id"]:
        raise ValueError("Candidate lock changed")
    for policy in lock["inventory"]:
        if policy["available"] and "checkpoint" in policy:
            artifact = policy["checkpoint"]
            if file_hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Selected checkpoint changed: {policy['id']}")
    return lock


def panel_protocol(protocol, workers=32):
    result = copy.deepcopy(protocol)
    cpus = [cpu for group in choose_cpu_groups() for cpu in group][:workers]
    result.update(workers=workers, evaluation_cpus=cpus)
    result["code_hashes"]["final_study.py"] = file_hash(__file__)
    return result


def development():
    protocol = read_json(ROOT / "registration.json")
    verify_protocol(protocol)
    lock = load_lock()
    policies = policies_for(lock, ["full_residual"])
    settings = panel_protocol(protocol)
    settings["role"] = "secondary_macro_development_diagnostic_no_reselection"
    settings["all_policy_slots"] = policies
    result = run_panel(
        settings,
        [p for p in policies if p["available"]],
        conditions(protocol, "development"),
        ROOT / "full_macro_development",
    )
    write_json(
        ROOT / "full_macro_development_receipt.json",
        {
            "candidate_lock_id": lock["lock_id"],
            "summary_sha256": file_hash(ROOT / "full_macro_development/summary.json"),
            "episodes": len(result["episodes"]),
            "qualified": sum(row["qualified"] for row in result["episodes"]),
            "no_checkpoint_reselection": True,
        },
    )


def seal():
    target = ROOT / "final_registration.json"
    if target.exists():
        raise FileExistsError("Final registration already exists")
    protocol = read_json(ROOT / "registration.json")
    verify_protocol(protocol)
    lock = load_lock()
    receipt = read_json(ROOT / "full_macro_development_receipt.json")
    if receipt["candidate_lock_id"] != lock["lock_id"]:
        raise ValueError("Secondary macro diagnostic used a different candidate lock")
    settings = panel_protocol(protocol)
    settings.update(
        candidate_lock_id=lock["lock_id"],
        candidate_lock_sha256=file_hash(ROOT / "final_candidate_lock.json"),
        secondary_diagnostic_sha256=file_hash(
            ROOT / "full_macro_development_receipt.json"
        ),
        policies=policies_for(lock, ["rule", "full_residual"]),
        panels={
            "regression": conditions(protocol, "regression"),
            "sealed": conditions(protocol, "sealed"),
        },
        final_claims={
            "regression": "Previously consumed three peak dates; descriptive only.",
            "sealed": (
                "Six previously unexecuted calendar combinations; "
                "not unseen workload phases."
            ),
            "primary_macro": "rule",
            "secondary_macro": "full_residual, zero asking price",
            "secondary_service_floor": 0.90,
            "secondary_floor_definition": (
                "Performance-weighted capacity-hours / maximum-residual accepted "
                "capacity-hours, same market draws. Nominal is reference-only."
            ),
            "means": (
                "All complete cases retained, including gate failures. "
                "No full-cohort mean if a planned learned seed is unavailable."
            ),
            "uncertainty": (
                "Descriptive SD across date/noise case means after averaging "
                "training seeds; separate training-seed results. No iid inference."
            ),
            "learning_claim": "Requires paired increment over exact BC initializer.",
        },
    )
    settings["final_registration_id"] = identity(settings)
    write_json(target, settings)
    print(
        "Final panels registered; no policy run yet:", settings["final_registration_id"]
    )


def execute(role):
    settings = read_json(ROOT / "final_registration.json")
    body = {k: v for k, v in settings.items() if k != "final_registration_id"}
    if identity(body) != settings["final_registration_id"]:
        raise ValueError("Final registration changed")
    protocol = read_json(ROOT / "registration.json")
    verify_protocol(protocol)
    lock = load_lock()
    if (
        file_hash(ROOT / "final_candidate_lock.json")
        != settings["candidate_lock_sha256"]
    ):
        raise ValueError("Final candidate lock receipt changed")
    if lock["lock_id"] != settings["candidate_lock_id"]:
        raise ValueError("Final candidate identities differ")
    if role not in settings["panels"]:
        raise ValueError("Unregistered final panel")
    settings["role"] = role
    settings["all_policy_slots"] = settings["policies"]
    run_panel(
        settings,
        [policy for policy in settings["policies"] if policy["available"]],
        settings["panels"][role],
        ROOT / f"final_{role}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "development", "seal", "regression", "sealed"]
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "development":
        development()
    elif args.stage == "seal":
        seal()
    else:
        execute(args.stage)


if __name__ == "__main__":
    main()
