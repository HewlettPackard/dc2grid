"""Extract release metadata from immutable records; never import the simulator."""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from adapter import (
    HERE,
    ROLE,
    SCHEMA,
    case_key,
    file_hash,
    identity,
    read_json,
    scientific_snapshot,
)

ECONOMIC = "outputs/rl_grid_cash_ppo_20261101"
ABSOLUTE = f"{ECONOMIC}/track_b"
FRACTIONAL = f"{ECONOMIC}/track_b_fraction"
CONFIRMATION = "outputs/camera_ready_confirmation_20261102"
PID = "outputs/pid_comparison_extract_20261102"
PHYSICS = "outputs/online_stability_pilot_20261031/source"
IMPORT_ROOTS = [
    PID,
    CONFIRMATION,
    FRACTIONAL,
    ABSOLUTE,
    f"{ECONOMIC}/track_a",
    ECONOMIC,
    "outputs/rl_state_local_20261101",
    "outputs/table2_distress_rl_20261031",
    "outputs/high_macro_recovery_pilot_20261031_v3",
    PHYSICS,
]
ENTRYPOINTS = [
    "confirmation_context",
    "controls",
    "confirmation_eval",
    "evaluate_macro",
    "evaluate_fraction",
    "pid_context",
    "pid_reference",
    "confirmation_audit",
    "audit_pid",
]
PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "PyYAML",
    "torch",
    "stable-baselines3",
    "gymnasium",
    "cloudpickle",
    "matplotlib",
)


def save(name: str, data) -> None:
    # This maintainer-only extractor writes solely beside itself, never to root.
    (HERE / name).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def dependency_closure(root: Path) -> tuple[dict, list[str]]:
    search = [root / relative for relative in IMPORT_ROOTS]
    origins, external, pending = {}, set(), list(ENTRYPOINTS)

    def find(name: str):
        relative = Path(*name.split("."))
        for directory in search:
            for path in (directory / relative / "__init__.py",
                         (directory / relative).with_suffix(".py")):
                if path.is_file():
                    return path
        return None

    while pending:
        name = pending.pop()
        if name in origins:
            continue
        path = find(name)
        if path is None:
            if name:
                external.add(name.split(".")[0])
            continue
        origins[name] = path.relative_to(root).as_posix()
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        if parent and find(parent):
            pending.append(parent)
        package = name if path.name == "__init__.py" else parent
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parts = package.split(".")
                    prefix = ".".join(parts[:len(parts) - node.level + 1])
                    module = ".".join(x for x in (prefix, module) if x)
                if module:
                    pending.append(module)
                    for alias in node.names:
                        child = f"{module}.{alias.name}"
                        if find(child):
                            pending.append(child)
    external -= set(sys.stdlib_module_names)
    external -= {"track_b", "track_b_fraction"}
    return dict(sorted(origins.items())), sorted(external)


def artifact(root: Path, path: Path, expected: str | None = None) -> dict:
    digest = file_hash(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Recorded frozen artifact changed: {path}")
    return {
        "recorded_path": str(path),
        "checkout_relative": path.relative_to(root).as_posix(),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def audit_expectation(row: dict) -> dict:
    keys = (
        "complete",
        "qualified",
        "physical_qualified",
        "metrics",
        "inputs_sha256",
        "stress_mask_sha256",
        "peak_phases",
        "failure_reasons",
        "work_conservation_error_kwh",
    )
    return {key: copy.deepcopy(row[key]) for key in keys if key in row}


def make_case(
    root: Path,
    policy_key: str,
    panel: str,
    cell: dict,
    path: Path,
    digest: str,
    expected: dict,
    *,
    historical_flat: bool,
) -> dict:
    case_id = f"{policy_key}:{panel}:{case_key(cell)}"
    return {
        "case_id": case_id,
        "policy_key": policy_key,
        "panel": panel,
        "cell": cell,
        "record": {
            "recorded_path": str(path),
            "checkout_relative": path.relative_to(root).as_posix(),
            "sha256": digest,
            "historical_flat_schema": historical_flat,
        },
        "expected": expected,
    }


def build(root: Path) -> dict:
    root = root.resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "--no-pager", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if not commit.startswith("34e01546"):
        raise ValueError("This extraction is bound to the approved read-only snapshot")
    protocol_paths = {
        "absolute": root / ABSOLUTE / "registration.json",
        "fractional": root / FRACTIONAL / "registration.json",
        "confirmation": root / CONFIRMATION / "registration.json",
        "pid": root / PID / "registration.json",
    }
    protocols = {key: read_json(path) for key, path in protocol_paths.items()}
    for key, protocol in protocols.items():
        body = {k: v for k, v in protocol.items() if k != "registration_id"}
        if identity(body) != protocol["registration_id"]:
            raise ValueError(f"Original registration identity changed: {key}")
    policies, cases, model_rows, extraction_inputs = {}, {}, {}, {}
    floor = protocols["confirmation"]["macro_tracking_floor_kw"]

    def add_policy(backend: str, policy: dict) -> str:
        key = f"{backend}/{policy['id']}"
        entry = {
            "policy_key": key,
            "backend": backend,
            "policy": copy.deepcopy(policy),
            "protocol_recorded_path": str(protocol_paths[backend]),
            "protocol_registration_id": protocols[backend]["registration_id"],
            "action_space_comparison": (
                "end_to_end_four_actuator_reference_not_fraction_action_space"
                if backend == "pid"
                else "absolute_0_to_6MW_not_fraction_coordinates"
                if backend == "absolute"
                else "same_native_fraction_action_space"
            ),
            "potential_active_at_deployment": False,
            "macro_tracking_floor_kw": floor,
        }
        if key in policies and policies[key] != entry:
            raise ValueError(f"Policy slot collision: {key}")
        policies[key] = entry
        if "checkpoint" in policy:
            checkpoint = policy["checkpoint"]
            path = Path(checkpoint["path"])
            row = artifact(root, path, checkpoint["sha256"])
            with zipfile.ZipFile(path) as archive:
                model_data = json.loads(archive.read("data"))
                row["checkpoint_context"] = model_data["c2g_context"]
                row["system_info"] = archive.read("system_info.txt").decode()
                row["sb3_version"] = archive.read(
                    "_stable_baselines3_version"
                ).decode().strip()
            row["algorithm"] = policy["algorithm"]
            row["training_seed"] = policy["training_seed"]
            row["final_macro_transitions"] = 24576
            row["observation_dimension"] = 23
            row["action_dimension"] = 1
            row["deployment_device"] = "cpu"
            row["deterministic_prediction"] = True
            row["gamma"] = model_data["gamma"]
            if row["gamma"] != 1.0 or row["checkpoint_context"].get(
                "macro_transitions"
            ) != 24576:
                raise ValueError("Only frozen full-budget gamma-one models qualify")
            model_rows[str(path)] = row
        return key

    def source_receipt(path: Path):
        extraction_inputs[str(path)] = artifact(root, path)

    def add_case(row: dict):
        if row["case_id"] in cases:
            raise ValueError(f"Duplicate consumed slot: {row['case_id']}")
        cases[row["case_id"]] = row

    for backend, relative, directories in (
        ("absolute", ABSOLUTE, ("development_full", "development_baselines")),
        ("fractional", FRACTIONAL, ("development",)),
    ):
        for directory in directories:
            manifest_path = root / relative / directory / "manifest.json"
            source_receipt(manifest_path)
            original_manifest = read_json(manifest_path)
            for policy in original_manifest["policies"]:
                add_policy(backend, policy)
            for path in sorted((manifest_path.parent).glob("*/*/episode.json")):
                record = read_json(path)
                key = add_policy(backend, record["policy"])
                add_case(
                    make_case(
                        root,
                        key,
                        "development",
                        record["cell"],
                        path,
                        file_hash(path),
                        {"snapshot": scientific_snapshot(record)},
                        historical_flat=True,
                    )
                )

    for policy in protocols["confirmation"]["policies"]:
        add_policy("confirmation", policy)
    for filename in ("development_cases.json", "final_cases.json"):
        path = root / CONFIRMATION / "reports" / filename
        source_receipt(path)
        for row in read_json(path):
            add_case(
                make_case(
                    root,
                    f"confirmation/{row['policy_id']}",
                    row["panel"],
                    row["cell"],
                    Path(row["episode_path"]),
                    row["episode_sha256"],
                    {
                        "snapshot": {
                            "qualified": row["qualified"],
                            "closed_26h_complete": row["complete"],
                        },
                        "audit": audit_expectation(row),
                    },
                    historical_flat=row["reused"],
                )
            )

    for policy in protocols["pid"]["policies"]:
        add_policy("pid", policy)
    pid_cases_path = root / PID / "reports/pid_cases.json"
    source_receipt(pid_cases_path)
    for row in read_json(pid_cases_path):
        path = Path(row["path"])
        artifact(root, path, row["sha256"])
        record = read_json(path)
        add_case(
            make_case(
                root,
                f"pid/{row['policy_id']}",
                row["panel"],
                row["cell"],
                path,
                row["sha256"],
                {
                    "snapshot": scientific_snapshot(record),
                    "audit": audit_expectation(row),
                },
                historical_flat=True,
            )
        )

    representatives = []
    calendar = {"start_day": 7, "seed": 207829280, "regd_load_bias": 0.5}
    development = {"start_day": 24, "seed": 93001, "regd_load_bias": 0.5}
    failed_pid = {"start_day": 107, "seed": 3002708602, "regd_load_bias": 0.0}
    representative_slots = [
        ("confirmation/script_zero", "calendar", calendar, "zero_fraction"),
        ("confirmation/script_f100_always", "calendar", calendar, "full_fraction"),
        ("absolute/economic_sac_1200", "development", development, "absolute_MW_SAC"),
        (
            "fractional/fraction_sac_1200",
            "development",
            development,
            "fraction_SAC_dev",
        ),
        ("fractional/potential_sac_1200", "development", development, "creditPhi_dev"),
        ("confirmation/model__fraction_sac_1200", "calendar", calendar, "fraction_SAC"),
        (
            "confirmation/model__potential_sac_1200",
            "calendar",
            calendar,
            "SAC_creditPhi",
        ),
        ("pid/pid_native", "calendar", calendar, "native_PID"),
        ("pid/pid_full_work", "calendar", calendar, "full_work_PID"),
        ("pid/pid_native", "calendar", failed_pid, "native_PID_failed_service_SOC"),
        (
            "pid/pid_full_work",
            "calendar",
            failed_pid,
            "full_work_PID_failed_service_SOC",
        ),
    ]
    for key, panel, cell, coverage in representative_slots:
        case_id = f"{key}:{panel}:{case_key(cell)}"
        row = cases[case_id]
        path = Path(row["record"]["recorded_path"])
        artifact(root, path, row["record"]["sha256"])
        row["expected"]["snapshot"] = scientific_snapshot(read_json(path))
        representatives.append({"coverage": coverage, **copy.deepcopy(row)})

    failed_cases = [
        copy.deepcopy(row)
        for row in cases.values()
        if not row["expected"]["snapshot"]["qualified"]
    ]
    runtime_failures = []
    evaluation_directories = [
        root / ABSOLUTE / "development_full",
        root / ABSOLUTE / "development_baselines",
        root / FRACTIONAL / "development",
        root / CONFIRMATION / "evaluation",
        root / PID / "evaluation",
    ]
    for directory in evaluation_directories:
        for path in sorted(directory.rglob("failure.json")):
            runtime_failures.append({
                **artifact(root, path), "failure": read_json(path)
            })

    origins, external = dependency_closure(root)
    expected_hashes = {}
    for protocol in protocols.values():
        expected_hashes.update(protocol["source_hashes"])
    baseline = read_json(
        root / "outputs/table2_distress_rl_20261031/registration.json"
    )
    for relative, digest in baseline["frozen_physics"]["physics_source_files"].items():
        expected_hashes[str(root / PHYSICS / relative)] = digest
    sources = []
    for relative in sorted(set(origins.values())):
        path = root / relative
        row = artifact(root, path, expected_hashes.get(str(path)))
        row["hash_also_bound_by_original_registration"] = str(path) in expected_hashes
        sources.append(row)

    environment = Path(protocols["fractional"]["config_path"])
    configs = [
        artifact(
            root,
            environment,
            protocols["confirmation"]["input_hashes"][str(environment)],
        ),
        artifact(
            root,
            root / PHYSICS / "c2g_env/config.yaml",
            expected_hashes[str(root / PHYSICS / "c2g_env/config.yaml")],
        ),
    ]
    inputs = protocols["confirmation"]["input_hashes"]
    data = [
        artifact(root, Path(path), digest)
        for path, digest in inputs.items()
        if "/data/processed/" in path
    ]
    if len(data) != 5:
        raise ValueError("Expected NYC energy/weather and all three workload traces")
    protocol_rows = [
        artifact(root, path) for path in protocol_paths.values()
    ]
    models = list(sorted(model_rows.values(), key=lambda row: row["recorded_path"]))
    path_map = {
        row["recorded_path"]: row["checkout_relative"]
        for group in (sources, configs, data, protocol_rows, models)
        for row in group
    }
    for relative in (
        "data/processed/workload_traces",
        "data/processed/energy",
        "data/processed/weather",
    ):
        path_map[str(root / relative)] = relative

    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    save("policies.json", {
        "schema": "dc2grid-economic-frozen-policy-catalog-v1",
        "role": ROLE,
        "policies": dict(sorted(policies.items())),
    })
    save("cases.json", {
        "schema": "dc2grid-economic-consumed-case-catalog-v1",
        "role": ROLE,
        "all_recorded_policy_cell_outcomes_included": True,
        "cases": dict(sorted(cases.items())),
    })
    save("representative_cases.json", {
        "schema": "dc2grid-economic-portability-goldens-v1",
        "role": ROLE,
        "selection_rule": (
            "Fixed first registered development/calendar cells with bias 0.5, "
            "plus the first archived PID service-SOC failure; not performance ranking."
        ),
        "representatives_are_not_a_statistical_cohort": True,
        "cases": representatives,
    })
    counts = collections.Counter()
    for row in cases.values():
        backend = policies[row["policy_key"]]["backend"]
        qualified = row["expected"]["snapshot"]["qualified"]
        counts[(backend, row["panel"], qualified)] += 1
    save("failures.json", {
        "schema": "dc2grid-economic-retained-failures-v1",
        "role": ROLE,
        "completed_disqualified_cases": failed_cases,
        "runtime_failure_attempts": runtime_failures,
        "no_survivor_only_means": True,
        "no_rerolling_or_cherry_picking_successful_attempts": True,
        "counts": [
            {"backend": key[0], "panel": key[1], "qualified": key[2], "count": value}
            for key, value in sorted(counts.items())
        ],
    })
    save("extraction_provenance.json", {
        "role": ROLE,
        "source_commit": commit,
        "read_only_extraction": True,
        "simulator_episodes_run": 0,
        "source_reports_and_manifests": list(extraction_inputs.values()),
        "sources_not_individually_registered": [
            row["checkout_relative"] for row in sources
            if not row["hash_also_bound_by_original_registration"]
        ],
        "these_historical_reports_are_not_replay_runtime_dependencies": True,
    })
    save("comparison_scopes.json", {
        "role": ROLE,
        "same_action_space": {
            "confirmation_roster": [
                key for key in policies if key.startswith("confirmation/")
            ],
            "action": "fraction_of_frozen_start_of_window_allowance",
            "observation_dimension": 23,
            "same_frozen_analytic_executor": True,
            "PPO_pre_tanh_mapped_once_by_original_MacroLatentEnv": True,
            "creditPhi_is_training_only": True,
            "registered_confirmation_analysis_preserved": protocols[
                "confirmation"
            ]["analysis"],
        },
        "absolute_coordinate_development": {
            "policies": [
                key for key in policies if key.startswith("absolute/")
            ],
            "action": "nonnegative_absolute_request_up_to_6MW",
            "never_reinterpret_absolute_models_as_fraction_models": True,
            "not_a_fresh_confirmation_panel": True,
        },
        "PID_end_to_end_reference": {
            "policies": ["pid/pid_native", "pid/pid_full_work"],
            "same_cells_as_confirmation_calendar_and_peak_noise": True,
            "cells_per_policy": 220,
            "not_same_fraction_action_space": True,
            "no_credit_allocation_guard_added_to_PID": True,
            "full_work_changes_only_executed_throttle_to_one": True,
            "native_PID_may_defer_work": True,
            "disqualified_cases_retained": len(failed_cases),
            "complete_disqualified_cash_is_diagnostic_not_admissible": True,
        },
        "scientific_contract": {
            key: copy.deepcopy(protocols["fractional"][key])
            for key in ("service_gate", "recovery", "service_floor", "normalization")
        },
        "horizon": {
            "service_hours": 24,
            "recovery_hours": 2,
            "step_seconds": 5,
            "service_steps": 17280,
            "recovery_steps": 1440,
            "macro_decisions": 96,
            "service_terminal_SOC_not_recovery_terminal_SOC": True,
            "stress_mean_relief_not_24h_or_26h_maximum_peak_reduction": True,
            "no_survivor_only_aggregate_or_replacement_of_failed_cases": True,
        },
    })
    requirements = [
        f"{name}=={version.split('+', 1)[0] if name == 'torch' else version}"
        for name, version in packages.items()
    ]
    (HERE / "requirements-runtime.txt").write_text(
        "# Python 3.11; CPU inference. Install only in a clean release environment.\n"
        + "\n".join(requirements) + "\n", encoding="utf-8"
    )
    metadata_names = (
        "policies.json",
        "cases.json",
        "representative_cases.json",
        "failures.json",
        "extraction_provenance.json",
        "comparison_scopes.json",
        "requirements-runtime.txt",
    )
    manifest = {
        "schema": SCHEMA,
        "role": ROLE,
        "recorded_root": str(root),
        "source_commit": commit,
        "recorded_path_map": dict(sorted(path_map.items())),
        "sources": sources,
        "configs": configs,
        "protocols": protocol_rows,
        "data": data,
        "models": models,
        "environment_config": str(environment),
        "config_directory_keys": ["trace_dir", "energy_dir", "weather_dir"],
        "import_roots": IMPORT_ROOTS,
        "entrypoint_modules": ENTRYPOINTS,
        "module_origins": origins,
        "external_import_roots": external,
        "runtime": {
            "python": sys.version.split()[0],
            "required_python_major_minor": "3.11",
            "packages": packages,
            "torch_local_build_suffix_may_differ_for_CPU_wheels": True,
            "cuda_required": False,
            "threads_per_process": 1,
            "battery_backend": "unchanged_frozen_pure_python_equivalent_circuit",
            "physical_v1_explicitly_selects_simple_BESS_even_if_PySAM_installed": True,
        },
        "verification_scope": {
            "frozen_scientific_source_bytes": True,
            "frozen_complete_original_protocol_bytes": True,
            "all_model_bytes_before_deserialization": True,
            "all_replay_data_bytes": True,
            "original_git_ancestry_required": False,
            "author_snapshot_or_paper_required": False,
            "benchmark_zip_required": False,
            "pre_amendment_raw_inventory_required": False,
            "legacy_verify_functions_invoked": False,
            "scientific_admission_or_closure_gates_changed": False,
        },
        "metadata_sha256": {
            name: file_hash(HERE / name) for name in metadata_names
        },
        "adapter_sha256": {
            name: file_hash(HERE / name)
            for name in (
                "adapter.py",
                "build_metadata.py",
                "test_adapter.py",
                "test_runtime_imports.py",
            )
            if (HERE / name).exists()
        },
    }
    manifest["manifest_id"] = identity(manifest)
    save("runtime_manifest.json", manifest)
    return {
        "manifest_id": manifest["manifest_id"],
        "source_modules": len(sources),
        "models": len(models),
        "policies": len(policies),
        "consumed_cases": len(cases),
        "representatives": len(representatives),
        "retained_disqualified_cases": len(failed_cases),
        "runtime_failure_attempts": len(runtime_failures),
        "external_import_roots": external,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.root), indent=2))
