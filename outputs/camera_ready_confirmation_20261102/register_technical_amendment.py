"""Record narrowly scoped audit/resume fixes before either confirmation opens."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from confirmation_context import (
    PROJECT,
    ROOT,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)

ORIGINAL_COMMIT = "b74ebbf4c72698c53a8d625c2150226551a6b12b"
CHANGED = (
    "confirmation_context.py",
    "confirmation_audit.py",
    "confirmation_analysis.py",
    "run_confirmation.py",
    "test_confirmation.py",
)
UNCHANGED_STATISTICS = (
    "noise_indices",
    "paired_estimate",
    "point_frontier",
    "summarize",
    "contrast",
    "confirmation_inference",
    "zero_parity",
)


def function_ast(text, name):
    matches = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous frozen analysis function: {name}")
    return ast.dump(matches[0], include_attributes=False)


def prepare():
    destination = ROOT / "technical_amendment.json"
    if destination.exists():
        raise FileExistsError("Never overwrite an amendment")
    if (ROOT / "final_opening.json").exists() or any(
        (ROOT / "evaluation" / panel).exists() for panel in ("calendar", "peak_noise")
    ):
        raise RuntimeError(
            "This correction must precede every new confirmation outcome"
        )
    protocol = read_json(ROOT / "registration.json")
    original_registration = subprocess.check_output(
        [
            "git",
            "show",
            f"{ORIGINAL_COMMIT}:{(ROOT / 'registration.json').relative_to(PROJECT)}",
        ],
        cwd=PROJECT,
    )
    if hashlib.sha256(original_registration).hexdigest() != file_hash(
        ROOT / "registration.json"
    ):
        raise ValueError("The original scientific registration was changed")
    original_analysis = subprocess.check_output(
        [
            "git",
            "show",
            f"{ORIGINAL_COMMIT}:"
            f"{(ROOT / 'confirmation_analysis.py').relative_to(PROJECT)}",
        ],
        cwd=PROJECT,
        text=True,
    )
    updated_analysis = (ROOT / "confirmation_analysis.py").read_text()
    function_hashes = {}
    for name in UNCHANGED_STATISTICS:
        before = function_ast(original_analysis, name)
        after = function_ast(updated_analysis, name)
        if before != after:
            raise ValueError(f"Scientific analysis function changed: {name}")
        function_hashes[name] = hashlib.sha256(after.encode()).hexdigest()
    retained = [
        {"path": str(path), "sha256": file_hash(path)}
        for path in sorted((ROOT / "evaluation").glob("*/*/*/episode.json"))
    ]
    inventory_path = ROOT / "pre_amendment_completed_cases.json"
    if inventory_path.exists():
        raise FileExistsError("Never overwrite the pre-amendment evidence inventory")
    write_json(inventory_path, retained)
    amendment = {
        "schema": "frozen-confirmation-technical-amendment-v1",
        "registration_id": protocol["registration_id"],
        "original_registration_commit": ORIGINAL_COMMIT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "no_confirmation_opening_or_outcomes": True,
        "original_registration_byte_identical": True,
        "source_hash_overrides": {
            str(ROOT / name): file_hash(ROOT / name) for name in CHANGED
        },
        "new_source_files": {str(Path(__file__).resolve()): file_hash(__file__)},
        "unchanged_scientific_spec_hashes": {
            name: identity(protocol[name])
            for name in (
                "policies",
                "panels",
                "analysis",
                "task_protocol",
            )
        },
        "unchanged_analysis_function_ast_sha256": function_hashes,
        "retained_known_result_inventory": {
            "path": str(inventory_path),
            "sha256": file_hash(inventory_path),
            "episodes": len(retained),
            "no_result_deleted_or_replaced": True,
        },
        "scope": [
            "Exclude endogenous frequency from pairing equality, not raw traces",
            "Mirror the frozen ledger's existing 1e-6kWh terminal-work slack",
            "Require the declared new-vs-historical record schema in the auditor",
            "Reuse completed slots, including physical failures, without re-rolling",
            "Retain interrupted attempts; publish a new full receipt view on resume",
            "Reuse but never rewrite the single unchanged confirmation opening",
            "Verify this amendment and the original source Git objects on every run",
        ],
        "rollout_generator_and_controller_source_unchanged": {
            str(ROOT / name): protocol["source_hashes"][str(ROOT / name)]
            for name in (
                "controls.py",
                "confirmation_eval.py",
                "prepare_confirmation.py",
            )
        },
        "physics_models_cases_thresholds_and_statistics_unchanged": True,
        "evidence_basis": "source inspection and already-consumed original-peak traces",
        "new_candidate_or_checkpoint_selection": False,
        "clarifications": {
            "secondary_bias": "zero-bias stratum, not a panel restriction",
            "secondary_panel_bias_strata": [0.0, 0.5],
            "primary_calendar_scope": "below p90, with two dates between p75 and p90",
            "checkpoint_means_are_not_action_averaging_ensembles": True,
        },
    }
    amendment["amendment_id"] = identity(amendment)
    write_json(destination, amendment)
    verify(protocol, committed=False)
    print(amendment["amendment_id"])
    print("Preserved known-panel outcomes:", len(retained))
    print("Frozen statistical functions unchanged:", len(function_hashes))


if __name__ == "__main__":
    prepare()
