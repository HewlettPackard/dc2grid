"""Declare descriptive PID additions before any new PID reference rollout."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from pid_context import (
    BASELINE,
    CONFIRMATION,
    ECONOMIC,
    PROJECT,
    ROOT,
    effective_source_hashes,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)


def prepare():
    destination = ROOT / "registration.json"
    if destination.exists():
        raise FileExistsError("Never overwrite the PID reference registration")
    prior = read_json(CONFIRMATION / "registration.json")
    task_path = ECONOMIC / "track_b_fraction/registration.json"
    task = read_json(task_path)
    sources = effective_source_hashes(prior)
    sources.update({str(path): file_hash(path) for path in ROOT.glob("*.py")})
    inputs = {
        str(task_path): file_hash(task_path),
        str(CONFIRMATION / "registration.json"): file_hash(
            CONFIRMATION / "registration.json"
        ),
        str(CONFIRMATION / "reports/final_report.json"): file_hash(
            CONFIRMATION / "reports/final_report.json"
        ),
        str(CONFIRMATION / "reports/final_cases.json"): file_hash(
            CONFIRMATION / "reports/final_cases.json"
        ),
        str(PROJECT / "NIPS26_Data_Center_Energy_Grid_Benchmark.zip"): file_hash(
            PROJECT / "NIPS26_Data_Center_Energy_Grid_Benchmark.zip"
        ),
        task["config_path"]: file_hash(task["config_path"]),
    }
    inputs.update(
        {
            path: digest
            for path, digest in prior["input_hashes"].items()
            if "/data/processed/" in path
        }
    )
    protocol = {
        "schema": "post-confirmation-matched-pid-reference-v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": BASELINE,
        "role": "post_confirmation_descriptive_reference_not_a_new_blind_test",
        "confirmation_registration_id": prior["registration_id"],
        "policies": [
            {
                "id": "pid_native",
                "family": "pid_native",
                "hardware": "pid",
                "macro": "full_residual",
                "full_work": False,
                "label": "PID (native four-actuator controller)",
            },
            {
                "id": "pid_full_work",
                "family": "pid_full_work",
                "hardware": "pid",
                "macro": "full_residual",
                "full_work": True,
                "label": "PID (full work; throttle fixed at 1)",
            },
        ],
        "panels": {
            name: {
                "cells": copy.deepcopy(prior["panels"][name]["cells"]),
                "source_confirmation_panel": name,
                "scope": "post_confirmation_matched_PID_reference",
                "all_cells_previously_consumed": True,
                "fresh": False,
                "primary": False,
            }
            for name in ("calendar", "peak_noise")
        },
        **{
            key: copy.deepcopy(task[key])
            for key in (
                "config_path",
                "normalization",
                "service_gate",
                "recovery",
                "service_floor",
            )
        },
        "source_hashes": sources,
        "input_hashes": inputs,
        "cpus": list(range(32)),
        "threads_per_process": 1,
        "expected_new_episodes": 440,
        "training_performed": False,
        "gain_tuning_performed": False,
        "native_pid_may_defer_work": True,
        "full_work_variant_changes_only_executed_throttle": True,
        "no_credit_allocation_guard_added_to_pid": True,
        "physical_work_closure_and_actual_service_floor_measured_for_both": True,
        "failed_or_incomplete_episodes_retained_without_survivor_means": True,
        "complete_but_disqualified_means_diagnostic_only": True,
        "existing_confirmation_decision_unchanged": True,
        "no_new_population_or_preregistered_superiority_claim": True,
    }
    protocol["registration_id"] = identity(protocol)
    verify(protocol, committed=False)
    write_json(destination, protocol)
    print(protocol["registration_id"])
    print("PID references: 2; matched cells per policy: 220; no tuning")


if __name__ == "__main__":
    prepare()
