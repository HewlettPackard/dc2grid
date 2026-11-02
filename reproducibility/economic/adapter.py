"""Path-relocated, inference-only replay of already consumed economic/PID cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
SCHEMA = "dc2grid-economic-release-runtime-v1"
ROLE = "release_reevaluation_of_consumed_cases_not_preregistered_opening"
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def checkout_path(root: Path, relative: str) -> Path:
    """Resolve an explicitly listed relative target, never an old-host fallback."""
    relative_path = PurePosixPath(relative)
    if (
        not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or "\\" in relative
        or ":" in relative
        or str(relative_path) != relative
    ):
        raise ValueError(f"Not a canonical checkout-relative path: {relative!r}")
    root = root.resolve(strict=True)
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Path escapes the checkout: {relative}")
    return target


def mapped_path(root: Path, recorded: str, manifest: dict) -> Path:
    try:
        relative = manifest["recorded_path_map"][recorded]
    except KeyError as error:
        raise ValueError(f"Unmapped recorded runtime path: {recorded}") from error
    return checkout_path(root, relative)


def case_key(cell: dict) -> str:
    if set(cell) != {"start_day", "seed", "regd_load_bias"}:
        raise ValueError("A consumed cell has exactly day, seed and bias")
    for key in ("start_day", "seed"):
        if type(cell[key]) is not int or cell[key] < 0:
            raise ValueError(f"Invalid cell {key}")
    bias = cell["regd_load_bias"]
    if (
        type(bias) not in (int, float)
        or not math.isfinite(bias)
        or bias not in (0.0, 0.5)
    ):
        raise ValueError("Invalid recorded bias")
    return f"d{cell['start_day']}_s{cell['seed']}_b{bias:g}".replace(".", "p")


def scientific_snapshot(record: dict) -> dict:
    """Keep 24h service admission separate from complete/closed 26h recovery."""
    outcome = record.get("outcome", record)
    keys = (
        "task_schema",
        "complete",
        "service_complete",
        "service_fast_steps",
        "recovery_fast_steps",
        "decisions",
        "service_physical_terminated",
        "physical_terminated",
        "termination_reason",
        "true_terminal",
        "truncated",
        "timeout_bootstrap",
        "required_gamma",
        "service_checks",
        "physical_qualified",
        "qualified",
        "delivered_service_floor_required",
        "delivered_service_floor",
        "delivered_service_gate",
        "delivered_service_ratio",
        "committed_mw_hours",
        "performance_weighted_mw_hours",
        "maximum_reference_mw_hours",
        "initial",
        "service_terminal",
        "service",
        "grid",
        "decomposition",
        "regulation_revenue_usd",
        "cashflow_26h_usd",
        "cashflow_closed_recovery_usd",
        "cash_vs_nominal_26h_usd",
        "ideal_regulation_reference_usd",
        "contractual_demand_electricity_usd",
        "exogenous_reference_cash_usd",
        "credit",
    )
    result = {key: copy.deepcopy(outcome[key]) for key in keys if key in outcome}
    recovery = outcome.get("recovery", {})
    # Continuation hashes contain no policy score, but are not numeric metrics.
    result["recovery"] = {
        key: copy.deepcopy(value)
        for key, value in recovery.items()
        if key != "extension"
    }
    result["closed_26h_complete"] = bool(
        outcome.get("complete", False) and recovery.get("complete", False)
    )
    result["cash_is_admissible"] = bool(
        result["closed_26h_complete"] and outcome.get("qualified", False)
    )
    return result


def compare_expected(actual, expected, location="") -> list[dict]:
    """Golden-value tolerances do not alter any frozen qualification gate."""
    differences = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{"field": location, "expected": expected, "actual": actual}]
        for key, value in expected.items():
            path = f"{location}.{key}" if location else key
            if key not in actual:
                differences.append({"field": path, "error": "missing"})
            else:
                differences.extend(compare_expected(actual[key], value, path))
        return differences
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [{"field": location, "expected": expected, "actual": actual}]
        for index, value in enumerate(expected):
            differences.extend(
                compare_expected(actual[index], value, f"{location}[{index}]")
            )
        return differences
    if type(expected) is float:
        tolerance = 1e-6 if "usd" in location.lower() else 1e-8
        equal = (
            type(actual) in (float, int)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
        )
    else:
        equal = type(actual) is type(expected) and actual == expected
    if not equal:
        differences.append(
            {"field": location, "expected": expected, "actual": actual}
        )
    return differences


def load_metadata() -> tuple[dict, dict, dict]:
    manifest = read_json(HERE / "runtime_manifest.json")
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("schema") != SCHEMA or identity(body) != manifest["manifest_id"]:
        raise ValueError("Release runtime manifest identity changed")
    for name, digest in manifest["metadata_sha256"].items():
        path = checkout_path(HERE, name)
        if file_hash(path) != digest:
            raise ValueError(f"Release metadata changed: {name}")
    for name, digest in manifest["adapter_sha256"].items():
        if file_hash(checkout_path(HERE, name)) != digest:
            raise ValueError(f"Release adapter changed: {name}")
    return (
        manifest,
        read_json(HERE / "policies.json")["policies"],
        read_json(HERE / "cases.json")["cases"],
    )


def verify_files(root: Path, manifest: dict) -> dict:
    counts = {}
    for group in ("sources", "configs", "protocols", "data", "models"):
        rows = manifest[group]
        for row in rows:
            path = mapped_path(root, row["recorded_path"], manifest)
            if path != checkout_path(root, row["checkout_relative"]):
                raise ValueError("Inconsistent explicit path mapping")
            if not path.is_file() or file_hash(path) != row["sha256"]:
                raise ValueError(f"Frozen {group} dependency changed/missing: {path}")
        counts[group] = len(rows)
    return counts


def runtime_versions(manifest: dict) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The frozen scientific runtime requires Python 3.11")
    observed = {"python": sys.version.split()[0]}
    for name, expected in manifest["runtime"]["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Missing frozen runtime dependency: {name}") from error
        # A CPU wheel is allowed; CUDA is never used for frozen inference.
        compare = actual.split("+", 1)[0] if name == "torch" else actual
        wanted = expected.split("+", 1)[0] if name == "torch" else expected
        if compare != wanted:
            raise RuntimeError(
                f"Runtime version drift: {name}={actual}, frozen={expected}"
            )
        observed[name] = actual
    return observed


def verify(root: Path, *, check_runtime: bool = True) -> dict:
    manifest, policies, cases = load_metadata()
    result = {
        "role": ROLE,
        "manifest_id": manifest["manifest_id"],
        "checkout": str(root.resolve(strict=True)),
        "verified_dependencies": verify_files(root, manifest),
        "policies": len(policies),
        "consumed_cases": len(cases),
        "original_git_history_required": False,
        "unpublished_inputs_required": False,
        "historical_raw_inventory_required": False,
    }
    if check_runtime:
        result["runtime"] = runtime_versions(manifest)
    return result


def resolve_request(
    policy_spec: dict, cell: dict, policies: dict, cases: dict
) -> tuple[dict, dict]:
    if set(policy_spec) != {"policy_key", "panel"}:
        raise ValueError("Use an immutable policy_key and consumed panel only")
    key = policy_spec["policy_key"]
    if key not in policies:
        raise ValueError("Unknown frozen release policy")
    candidate = f"{key}:{policy_spec['panel']}:{case_key(cell)}"
    if candidate not in cases:
        raise ValueError("No consumed outcome exists for this exact policy/cell")
    case = cases[candidate]
    if case["cell"] != cell or case["policy_key"] != key:
        raise ValueError("Consumed case identity differs")
    return policies[key], case


def relocate_config(original: dict, root: Path, manifest: dict) -> dict:
    result = copy.deepcopy(original)
    allowed = manifest["config_directory_keys"]
    for key in allowed:
        recorded = original["global"][key]
        path = mapped_path(root, recorded, manifest)
        if not path.is_dir():
            raise ValueError(f"Missing verified data directory: {path}")
        result["global"][key] = str(path)
    restored = copy.deepcopy(result)
    for key in allowed:
        restored["global"][key] = original["global"][key]
    if restored != original:
        raise AssertionError("Configuration relocation changed scientific fields")
    return result


def relocate_policy(original: dict, root: Path, manifest: dict) -> dict:
    result = copy.deepcopy(original)
    if "checkpoint" in result:
        result["checkpoint"]["path"] = str(
            mapped_path(root, original["checkpoint"]["path"], manifest)
        )
    return result


def relocate_protocol(original: dict, config: Path) -> dict:
    result = copy.deepcopy(original)
    if "config_path" not in result:
        raise ValueError("Task protocol has no explicit environment config")
    result["config_path"] = str(config)
    return result


def _environment(output: Path) -> dict:
    result = dict(os.environ)
    cache = output / "runtime_cache"
    for key, suffix in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TMPDIR", "python"),
        ("TEMP", "python"),
        ("TMP", "python"),
    ):
        path = cache / suffix
        path.mkdir(parents=True, exist_ok=True)
        result[key] = str(path)
    result.update({key: "1" for key in THREAD_KEYS})
    result.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
        PYTHONNOUSERSITE="1",
    )
    result.pop("PYTHONPATH", None)
    return result


def _check_import_origins(root: Path, manifest: dict) -> None:
    for name, relative in manifest["module_origins"].items():
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if filename is not None and Path(filename).resolve() != checkout_path(
            root, relative
        ):
            raise RuntimeError(f"Wrong frozen module origin: {name}: {filename}")
    allowed = {row["checkout_relative"] for row in manifest["sources"]}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(root):
            relative = path.relative_to(root).as_posix()
            if relative.startswith("outputs/") and relative not in allowed:
                raise RuntimeError(f"Unverified checkout module imported: {name}")


def _bootstrap(root: Path, manifest: dict, output: Path) -> None:
    sys.dont_write_bytecode = True
    os.environ.update(_environment(output))
    _check_import_origins(root, manifest)
    # SB3 imports plotting support. Initialize its cache before historical
    # context modules assign their obsolete experiment-local MPLCONFIGDIR.
    import matplotlib

    matplotlib.use("Agg")
    importlib.import_module("matplotlib.pyplot")
    sys.path[:0] = [
        str(checkout_path(root, relative))
        for relative in manifest["import_roots"]
    ]
    for name in manifest["entrypoint_modules"]:
        importlib.import_module(name)
    os.environ.update(_environment(output))
    _check_import_origins(root, manifest)
    importlib.import_module("common").cap_cpu()


def _prepare_inputs(
    root: Path, manifest: dict, policy_entry: dict, output: Path
) -> tuple[dict, dict]:
    import yaml

    original_config = mapped_path(root, manifest["environment_config"], manifest)
    config = yaml.safe_load(original_config.read_text(encoding="utf-8"))
    relocated = relocate_config(config, root, manifest)
    inputs = output / "runtime_inputs"
    inputs.mkdir(exist_ok=False)
    config_path = inputs / "environment.yaml"
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(relocated, handle, sort_keys=False)
    original_path = mapped_path(
        root, policy_entry["protocol_recorded_path"], manifest
    )
    original_protocol = read_json(original_path)
    policy = relocate_policy(policy_entry["policy"], root, manifest)
    if policy_entry["backend"] == "confirmation":
        protocol = copy.deepcopy(original_protocol)
        protocol["policies"] = [
            relocate_policy(item, root, manifest)
            for item in original_protocol["policies"]
        ]
        task_path = mapped_path(
            root, original_protocol["task_protocol"]["path"], manifest
        )
        task = relocate_protocol(read_json(task_path), config_path)
        relocated_task_path = inputs / "fractional_task_protocol.json"
        write_json(relocated_task_path, task)
        protocol["task_protocol"] = {
            "path": str(relocated_task_path),
            "sha256": file_hash(relocated_task_path),
        }
    else:
        protocol = relocate_protocol(original_protocol, config_path)
    write_json(inputs / "runtime_protocol.json", protocol)
    write_json(
        inputs / "relocation_receipt.json",
        {
            "role": ROLE,
            "original_protocol_sha256": file_hash(original_path),
            "original_config_sha256": file_hash(original_config),
            "runtime_protocol_sha256": file_hash(inputs / "runtime_protocol.json"),
            "runtime_config_sha256": file_hash(config_path),
            "allowed_semantic_changes": [],
            "allowed_path_changes": [
                "config_path",
                "global.trace_dir",
                "global.energy_dir",
                "global.weather_dir",
                "policy.checkpoint.path",
                "confirmation.task_protocol.path_and_derived_file_sha256",
            ],
            "recorded_path_to_checkout_path": {
                old: str(checkout_path(root, relative))
                for old, relative in manifest["recorded_path_map"].items()
            },
            "historical_registration_ids_are_provenance_not_new_registration": True,
        },
    )
    return protocol, policy


def _audit(entry: dict, protocol: dict, case: dict, record: dict, directory: Path):
    backend = entry["backend"]
    if backend == "pid":
        return importlib.import_module("audit_pid").audit_episode(
            directory,
            record,
            protocol,
            expected_inputs=case["expected"].get("audit", {}).get("inputs_sha256"),
        )
    if backend in ("confirmation", "fractional"):
        return importlib.import_module("confirmation_audit").audit_record(
            directory,
            record,
            tracking_floor_kw=entry["macro_tracking_floor_kw"],
            historical=backend == "fractional",
        )
    checks = importlib.import_module("evaluate_low").audit_primary(directory, record)
    return {
        "complete": bool(record["complete"] and record["recovery"]["complete"]),
        "qualified": record["qualified"],
        "raw_checks": checks,
        "metrics": importlib.import_module("headroom_report").episode_metrics(record),
    }


def _worker(root: Path, policy_spec: dict, cell: dict, output: Path) -> dict:
    manifest, policies, cases = load_metadata()
    validate_locations(root, output, manifest)
    verification = verify(root)
    entry, case = resolve_request(policy_spec, cell, policies, cases)
    _bootstrap(root, manifest, output)
    protocol, policy = _prepare_inputs(root, manifest, entry, output)
    directory = output / "frozen_evaluator"
    backend = entry["backend"]
    if backend == "confirmation":
        record = importlib.import_module("confirmation_eval").evaluate_case(
            protocol, policy_spec["panel"], policy, cell, directory
        )
    elif backend in ("absolute", "fractional"):
        module = "evaluate_macro" if backend == "absolute" else "evaluate_fraction"
        record = importlib.import_module(module).evaluate_case(
            protocol, policy, cell, directory
        )
    elif backend == "pid":
        record = importlib.import_module("economy_eval").evaluate_episode(
            protocol,
            policy,
            cell,
            directory,
            hardware_factory=importlib.import_module("pid_reference").hardware_factory,
        )
    else:
        raise ValueError("Unsupported frozen evaluator")
    record = read_json(directory / "episode.json")
    audit = _audit(entry, protocol, case, record, directory)
    importlib.import_module("common").write_json(output / "frozen_audit.json", audit)
    audit = read_json(output / "frozen_audit.json")
    _check_import_origins(root, manifest)
    actual = {"snapshot": scientific_snapshot(record), "audit": audit}
    write_json(output / "release_metrics.json", actual)
    differences = compare_expected(actual, case["expected"])
    write_json(
        output / "golden_comparison.json",
        {
            "role": ROLE,
            "case_id": case["case_id"],
            "matches_saved_case": not differences,
            "differences": differences,
            "numeric_atol": {"currency_usd": 1e-6, "other_float": 1e-8},
            "numeric_rtol": 0.0,
            "all_gate_booleans_exact": True,
            "gzip_bytes_and_elapsed_times_not_comparison_targets": True,
        },
    )
    receipt = {
        **verification,
        "case_id": case["case_id"],
        "original_consumed_record": case["record"],
        "original_policy": entry["policy"],
        "cell": cell,
        "training_performed": False,
        "potential_used_at_evaluation": False,
        "new_panel_opened": False,
        "new_preregistered_claim": False,
        "qualified": actual["snapshot"]["qualified"],
        "closed_26h_complete": actual["snapshot"]["closed_26h_complete"],
        "matches_saved_case": not differences,
        "episode_sha256": file_hash(directory / "episode.json"),
        "trace_hashes": record["trace_hashes"],
        "scientific_sources_and_inputs_verified_after_execution": verify_files(
            root, manifest
        ),
    }
    write_json(output / "release_receipt.json", receipt)
    if differences:
        raise AssertionError("Replay differs from saved metrics; all evidence retained")
    return receipt


def validate_locations(root: Path, output: Path, manifest: dict) -> None:
    root = root.resolve(strict=True)
    output = output.resolve()
    old_root = Path(manifest["recorded_root"]).resolve()
    if root == old_root or root.is_relative_to(old_root):
        raise ValueError("Replay requires a new checkout, never the frozen source tree")
    if output.is_relative_to(old_root):
        raise ValueError("Cannot write into the original read-only tree")
    if output == root or output.is_relative_to(root / "outputs"):
        raise ValueError("Use a new release-result folder outside frozen outputs/")
    if output == HERE or output.is_relative_to(HERE):
        raise ValueError("Keep replay results separate from the release adapter")


def evaluate(root: Path, policy_spec: dict, cell: dict, output: Path) -> dict:
    """Execute exactly one consumed slot in a fresh process and new output folder."""
    root = root.resolve(strict=True)
    output = output.resolve()
    manifest, policies, cases = load_metadata()
    resolve_request(policy_spec, cell, policies, cases)
    validate_locations(root, output, manifest)
    verify(root)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "release_request.json",
        {
            "role": ROLE,
            "root": str(root),
            "policy_spec": policy_spec,
            "cell": cell,
            "manifest_id": manifest["manifest_id"],
        },
    )
    environment = _environment(output)
    with (output / "worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(HERE / "adapter.py"),
                "_worker",
                "--request",
                str(output / "release_request.json"),
            ],
            cwd=output,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"Release replay failed ({process.returncode}); "
            f"retain all files in {output}"
        )
    return read_json(output / "release_receipt.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--files-only", action="store_true")
    run = subparsers.add_parser("evaluate")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--case-id", required=True)
    run.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("representatives")
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.root, check_runtime=not args.files_only)
    elif args.command == "representatives":
        load_metadata()
        result = read_json(HERE / "representative_cases.json")
    elif args.command == "evaluate":
        _, _, cases = load_metadata()
        case = cases[args.case_id]
        result = evaluate(
            args.root,
            {"policy_key": case["policy_key"], "panel": case["panel"]},
            case["cell"],
            args.output,
        )
    else:
        request = read_json(args.request)
        output = args.request.resolve().parent
        try:
            result = _worker(
                Path(request["root"]),
                request["policy_spec"],
                request["cell"],
                output,
            )
        except BaseException as error:
            write_json(
                output / "release_failure.json",
                {
                    "role": ROLE,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "all_completed_and_partial_evidence_retained": True,
                    "no_retry_or_winner_selection_performed": True,
                },
            )
            raise
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
