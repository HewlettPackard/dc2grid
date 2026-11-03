"""Hash-checked Table 2 replay on the explicitly bound, main-based shared core.

Public API: evaluate(root: Path, policy_spec: dict, cell: dict, output: Path).
policy_spec contains ONLY {"snapshot": inventory key, "policy_id": archived ID}.
No training, panel opening, selection, or scientific gate changes.
"""

from __future__ import annotations

import argparse
import copy
import csv
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

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from reproducibility.assets import verify as verify_integration  # noqa: E402
from reproducibility.runtime import CORE_COMMIT  # noqa: E402

TABLE = "outputs/table2_distress_rl_20261031"
STATE = "outputs/rl_state_local_20261101"
VALUE = "outputs/rl_value_conditioning_20261101"
FULL_WORK = frozenset(
    {
        "nominal",
        "cooling_first",
        "bc_only",
        "guided_sac",
        "fixed_reference_kl",
        "replay_reference_kl",
        "union_original",
        "union_conditioned",
    }
)
HARDWARE = frozenset(
    {"none", "rule", "pid", "bang_bang", "old_sac", "cooling_first", "history_sac"}
)


class ReleaseError(RuntimeError):
    """A missing or changed scientific input is a blocker, never a fallback."""


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def inventory() -> dict:
    value = json.loads(Path(__file__).with_name("inventory.json").read_text())
    if value.get("schema") != "dc2grid-tracking-portable-replay-v1":
        raise ReleaseError("Unsupported tracking inventory")
    if value.get("integration_core_commit") != CORE_COMMIT:
        raise ReleaseError("Tracking inventory is not bound to the shared core")
    return value


def contained_path(root: Path, relative: str, *, is_file: bool = True) -> Path:
    value = PurePosixPath(relative)
    if (
        not relative
        or value.is_absolute()
        or ".." in value.parts
        or "\\" in relative
        or relative == "."
    ):
        raise ReleaseError(f"Expected a checkout-relative input path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ReleaseError(f"Input symlink escapes the new checkout: {relative}")
    if is_file and not path.is_file():
        raise ReleaseError(f"Missing required scientific input: {relative}")
    return path


def archive_relative(path: str, index: dict) -> str:
    value = PurePosixPath(path)
    prefix = PurePosixPath(index["archive_root_label"])
    if not value.is_absolute() or not value.is_relative_to(prefix):
        raise ReleaseError(f"Unmapped archival path; no original-path fallback: {path}")
    result = str(value.relative_to(prefix))
    if ".." in PurePosixPath(result).parts or "\\" in result or result == ".":
        raise ReleaseError(f"Unsafe archival path: {path}")
    return result


def normalize_cell(cell: dict) -> dict:
    if not isinstance(cell, dict) or set(cell) != {
        "start_day",
        "seed",
        "regd_load_bias",
    }:
        raise ReleaseError("Cell must have exactly start_day, seed, regd_load_bias")
    for key in ("start_day", "seed"):
        if type(cell[key]) is not int or cell[key] < 0:
            raise ReleaseError(f"{key} must be a nonnegative integer")
    bias = cell["regd_load_bias"]
    if type(bias) not in (int, float) or not math.isfinite(bias):
        raise ReleaseError("regd_load_bias must be a finite number")
    return {**cell, "regd_load_bias": float(bias)}


def _check_identity(document: dict, key: str, name: str) -> None:
    if (
        key not in document
        or identity({field: value for field, value in document.items() if field != key})
        != document[key]
    ):
        raise ReleaseError(f"Original {key} no longer verifies: {name}")


def _check_membership(value, collection, label):
    if value not in collection:
        raise ReleaseError(f"Not authorized by the original consumed {label}")


def _authorize(panel, protocol, manifest, policy, cell, read):
    authorization = panel["authorization"]
    if manifest["role"] != panel["role"]:
        raise ReleaseError("Consumed manifest role differs from the release index")
    if authorization == "table2_final":
        _check_identity(protocol, "final_registration_id", panel["protocol"])
        baseline = read(f"{TABLE}/registration.json")
        _check_identity(baseline, "registration_id", f"{TABLE}/registration.json")
        lock = read(f"{TABLE}/final_candidate_lock.json")
        _check_identity(lock, "lock_id", f"{TABLE}/final_candidate_lock.json")
        if not (
            protocol["registration_id"]
            == baseline["registration_id"]
            == lock["training_registration_id"]
            == manifest["registration_id"]
            and protocol["candidate_lock_id"] == lock["lock_id"]
            and protocol["final_registration_id"] == manifest["final_registration_id"]
        ):
            raise ReleaseError("Table2 original registration/selection links differ")
        _check_membership(policy, protocol["policies"], "Table2 policy lock")
        _check_membership(cell, protocol["panels"][panel["role"]], "Table2 panel")
        if protocol["final_claims"]["secondary_service_floor"] != 0.9:
            raise ReleaseError("Original capacity-service floor changed")
        return

    _check_identity(protocol, "registration_id", panel["protocol"])
    if protocol["registration_id"] != manifest["registration_id"]:
        raise ReleaseError("Consumed manifest and original registration differ")
    if authorization == "development":
        _check_membership(cell, protocol["development_cells"], "development panel")
        return
    if authorization not in ("value_regression", "value_final"):
        raise ReleaseError("Unsupported authorization contract")

    training = read(f"{VALUE}/registration.json")
    _check_identity(training, "registration_id", f"{VALUE}/registration.json")
    candidate = read(f"{VALUE}/candidate_lock.json")
    _check_identity(candidate, "candidate_lock_id", f"{VALUE}/candidate_lock.json")
    if candidate["registration_id"] != training["registration_id"]:
        raise ReleaseError("Union candidate lock is detached from its registration")
    _check_membership(policy, candidate["inventory"], "union candidate lock")
    if authorization == "value_regression":
        if not (
            protocol["training_registration_id"] == training["registration_id"]
            and protocol["candidate_lock_id"] == candidate["candidate_lock_id"]
        ):
            raise ReleaseError("Consumed peak regression has different selection links")
        _check_membership(cell, protocol["regression_cells"], "peak regression panel")
        return

    final = read(f"{VALUE}/final_lock.json")
    _check_identity(final, "lock_id", f"{VALUE}/final_lock.json")
    shared = read(f"{STATE}/final_lock.json")
    if not (
        final == shared
        and final["registration_id"] == protocol["registration_id"]
        and final["candidate_lock_id"] == candidate["candidate_lock_id"]
        and final["cells"] == protocol["final_cells"]
        and final["policies"] == manifest["policies"]
    ):
        raise ReleaseError("Original shared Final-B lock/cell/policy gate failed")
    _check_membership(cell, final["cells"], "Final-B cell lock")
    _check_membership(policy, final["policies"], "Final-B policy lock")


def extract_metrics(episode: dict, *, ratio: float | None = None) -> dict:
    service = episode["service"]
    recovery = episode["recovery"]
    grid = episode.get("grid") or {}
    complete_26h = bool(episode["complete"] and recovery["complete"])
    floor_enforced = (
        episode["policy"]["macro"] == "full_residual"
        and episode["policy"]["family"] != "nominal"
    )
    if ratio is None:
        ratio = episode.get("delivered_service_ratio")
    if complete_26h and floor_enforced and ratio is None:
        raise ReleaseError(
            "Cannot qualify full-residual service without its actual floor"
        )
    delivered_gate = None if ratio is None else ratio >= 0.9
    return {
        "complete": episode["complete"],
        "complete_26h": complete_26h,
        "service_steps": service["steps"],
        "recovery_steps": recovery.get("ledger", {}).get("steps", 0),
        "service_checks": episode["service_checks"],
        "recovery_closed": recovery["closed"],
        "physical_gate_qualified": episode["qualified"],
        "floor_enforced": floor_enforced,
        "delivered_service_ratio": ratio,
        "delivered_service_gate": delivered_gate,
        "publication_qualified": bool(
            episode["qualified"] and (not floor_enforced or delivered_gate)
        ),
        "tracking_rmse_kw": service["tracking_rmse_kw"]
        if episode["complete"]
        else None,
        "service_prefix_rmse_kw": service["tracking_rmse_kw"],
        "stress_facility_relief_mw": grid.get("stress_facility_relief_mw"),
        "LFR_pct": grid.get("LFR_pct"),
        "SPDR_pct": grid.get("SPDR_pct"),
        "LWSR_pct": grid.get("LWSR_pct"),
        "LWCF_pct": grid.get("LWCF_pct"),
        "cashflow_26h_usd": episode["cashflow_26h_usd"],
        "cashflow_closed_recovery_usd": episode["cashflow_closed_recovery_usd"],
        "regulation_revenue_usd": episode["regulation_revenue_usd"],
        "battery_throughput_26h_mwh": (
            service["battery_throughput_mwh"]
            + recovery["ledger"]["battery_throughput_mwh"]
        )
        if complete_26h
        else None,
        "service_battery_throughput_mwh": service["battery_throughput_mwh"],
        "thermal_warning_fraction": service["thermal_warning_fraction"],
        "service_terminal_soc": episode["service_terminal"]["soc"],
        "service_terminal_temp_A_C": episode["service_terminal"]["temp_A"],
        "service_terminal_temp_B_C": episode["service_terminal"]["temp_B"],
        "service_work_arrived_kwh": service["work_arrived_kwh"],
        "service_work_served_kwh": service["work_served_kwh"],
        "service_backlog_kwh": service["backlog_kwh"],
        "service_work_conservation_error_kwh": service["work_conservation_error_kwh"],
    }


def compare_metrics(expected: dict, actual: dict) -> list[dict]:
    differences = []

    def compare(left, right, key):
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                differences.append({"metric": key, "expected": left, "actual": right})
                return
            for name in left:
                compare(left[name], right[name], f"{key}.{name}" if key else name)
        elif left is None or isinstance(left, (bool, str, int)):
            if type(left) is not type(right) or left != right:
                differences.append({"metric": key, "expected": left, "actual": right})
        elif isinstance(left, float):
            absolute = 1e-5 if key.endswith("_usd") else 1e-6
            if (
                type(right) not in (float, int)
                or not math.isfinite(right)
                or not math.isclose(left, right, rel_tol=1e-9, abs_tol=absolute)
            ):
                differences.append(
                    {
                        "metric": key,
                        "expected": left,
                        "actual": right,
                        "absolute_tolerance": absolute,
                        "relative_tolerance": 1e-9,
                    }
                )
        else:
            raise ReleaseError(f"Unsupported metric type at {key}")

    compare(expected, actual, "")
    return differences


def _prepare(root: Path, policy_spec: dict, cell: dict, index: dict) -> dict:
    root = root.resolve(strict=True)
    if not isinstance(policy_spec, dict) or set(policy_spec) != {
        "snapshot",
        "policy_id",
    }:
        raise ReleaseError(
            "Only snapshot and policy_id are accepted; no policy overrides"
        )
    if any(not isinstance(value, str) for value in policy_spec.values()):
        raise ReleaseError("Snapshot and policy_id must be strings")
    if policy_spec["snapshot"] not in index["panels"]:
        raise ReleaseError("Unknown consumed snapshot; new panels are not permitted")
    cell = normalize_cell(cell)
    panel = index["panels"][policy_spec["snapshot"]]
    _check_membership(
        policy_spec["policy_id"], panel["policy_ids"], "Table 2 policy roster"
    )
    verify_integration(root, "table2")
    verified = {}
    documents = {}

    def verify(name):
        if name not in index["files"]:
            raise ReleaseError(
                f"Input not present in the frozen release inventory: {name}"
            )
        path = contained_path(root, name)
        if name not in verified:
            actual = file_hash(path)
            if actual != index["files"][name]["sha256"]:
                raise ReleaseError(f"Original SHA-256 mismatch: {name}")
            verified[name] = actual
        return path

    def read(name):
        if name not in documents:
            documents[name] = json.loads(verify(name).read_text())
        return documents[name]

    protocol = read(panel["protocol"])
    manifest = read(panel["manifest"])
    summary = read(panel["summary"])
    matching = [
        row
        for row in summary["episodes"]
        if row["policy"]["id"] == policy_spec["policy_id"] and row["cell"] == cell
    ]
    if len(matching) != 1:
        raise ReleaseError("The exact policy/cell must occur once in a consumed result")
    episode = matching[0]
    archived_policy = episode["policy"]
    if archived_policy.get("available", True) is not True:
        raise ReleaseError(
            "An unavailable/non-admitted slot cannot be replayed as selected"
        )
    if archived_policy["hardware"] not in HARDWARE or archived_policy["macro"] not in (
        "rule",
        "full_residual",
    ):
        raise ReleaseError(
            "Only the registered tracking hardware and macro families are supported"
        )
    _check_membership(
        archived_policy, manifest["policies"], "manifest policy inventory"
    )
    _check_membership(cell, manifest["cells"], "manifest cell inventory")
    _authorize(panel, protocol, manifest, archived_policy, cell, read)

    if panel["authorization"] == "table2_final":
        name = f"{TABLE}/final_candidate_lock.json"
        if verified[name] != protocol["candidate_lock_sha256"]:
            raise ReleaseError("Table2 candidate lock hash link changed")
    for name in (
        index["runtime"]["source_files"]
        + index["runtime"]["config_files"]
        + index["runtime"]["data_files"]
    ):
        verify(name)

    configuration = archive_relative(protocol["config_path"], index)
    verify(configuration)
    policy = copy.deepcopy(archived_policy)
    artifact = (
        protocol["hardware_reference"]
        if policy["hardware"] == "old_sac"
        else policy.get("checkpoint")
    )
    model_path = None
    if artifact is not None:
        model_path = archive_relative(artifact["path"], index)
        if model_path not in index["models"]:
            raise ReleaseError(
                "The selected checkpoint is not in the original identity map"
            )
        model = index["models"][model_path]
        if model["sha256"] != artifact["sha256"] or model["family"] != policy["family"]:
            raise ReleaseError("Checkpoint family or original selected hash changed")
        verify(model_path)
        if policy["hardware"] != "old_sac":
            policy["checkpoint"]["path"] = str(contained_path(root, model_path))

    # These are explicit single-case inputs, not a rewritten registration.
    # Panel authorization above is complete before absolute paths are translated.
    runtime = {
        key: copy.deepcopy(protocol[key])
        for key in ("registration_id", "normalization", "service_gate", "recovery")
    }
    runtime["config_path"] = str(contained_path(root, configuration))
    reference = copy.deepcopy(protocol["hardware_reference"])
    reference["path"] = str(
        contained_path(root, archive_relative(reference["path"], index), is_file=False)
    )
    runtime["hardware_reference"] = reference
    if panel["evaluator"] == "table2":
        runtime["code_hashes"] = {
            name: index["files"][name]["sha256"]
            for name in index["runtime"]["source_files"]
        }
    else:
        runtime["service_floor"] = protocol["service_floor"]
        if runtime["service_floor"] != 0.9:
            raise ReleaseError("Unchanged physical evaluation requires the 90% floor")
        if "high_level" in protocol:
            runtime["high_level"] = copy.deepcopy(protocol["high_level"])
        runtime["source_hashes"] = {
            str(contained_path(root, name)): index["files"][name]["sha256"]
            for name in index["runtime"]["source_files"]
        }

    ratio = episode.get("delivered_service_ratio")
    if panel["evaluator"] == "table2" and policy["macro"] == "full_residual":
        with verify(panel["metrics_csv"]).open() as stream:
            reports = [
                row
                for row in csv.DictReader(stream)
                if row["panel"] == panel["metrics_panel"]
                and row["policy_id"] == policy["id"]
                and int(row["start_day"]) == cell["start_day"]
                and int(row["env_seed"]) == cell["seed"]
                and float(row["bias"]) == cell["regd_load_bias"]
            ]
        if len(reports) != 1 or not reports[0]["delivered_service_ratio"]:
            raise ReleaseError(
                "Missing original full-residual publication gate evidence"
            )
        ratio = float(reports[0]["delivered_service_ratio"])
    expected = extract_metrics(episode, ratio=ratio)
    return {
        "root": str(root),
        "policy_spec": dict(policy_spec),
        "cell": cell,
        "evaluator": panel["evaluator"],
        "policy": policy,
        "archived_policy": archived_policy,
        "runtime_protocol": runtime,
        "original_config": configuration,
        "original_protocol_path": panel["protocol"],
        "expected_metrics": expected,
        "expected_reset_info": episode["reset_info"],
        "selected_model": model_path,
        "verified_inputs": dict(sorted(verified.items())),
        "original_registration_id": protocol["registration_id"],
        "original_final_registration_id": protocol.get("final_registration_id"),
        "authorization": panel["authorization"],
    }


def prepare(root: Path, policy_spec: dict, cell: dict) -> dict:
    """Read/hash/authorize only: no import of scientific code and no rollout."""
    return _prepare(Path(root), policy_spec, cell, inventory())


def verify_unchanged_inputs(root: Path, verified: dict) -> None:
    for relative, expected in verified.items():
        if file_hash(contained_path(root, relative)) != expected:
            raise ReleaseError(f"Scientific input changed during replay: {relative}")


def rebase_configuration(config: dict, root: Path, index: dict) -> dict:
    result = copy.deepcopy(config)
    for key, relative in index["runtime"]["data_directories"].items():
        if archive_relative(config["global"][key], index) != relative:
            raise ReleaseError(f"Unexpected original scientific data directory: {key}")
        path = contained_path(root, relative, is_file=False)
        if not path.is_dir():
            raise ReleaseError(f"Missing scientific data directory: {relative}")
        result["global"][key] = str(path)
    return result


def _output_path(root: Path, output: Path, index: dict) -> Path:
    if str(root) == index["archive_root_label"]:
        raise ReleaseError(
            "Physics replay must run from a NEW checkout, never the original"
        )
    output = output if output.is_absolute() else root / output
    output = output.resolve()
    if output == root or not output.is_relative_to(root):
        raise ReleaseError(
            "Replay output must be a new directory inside the new checkout"
        )
    return output


def evaluate(root: Path, policy_spec: dict, cell: dict, output: Path) -> dict:
    """Replay one consumed cell in an isolated CPU process and verify its metrics.

    The output bundle contains original-evaluator traces, derived path-only YAML,
    a worker log, an input-hash receipt, and an exact-gate/tolerant-number comparison.
    A failed scientific gate is a valid reproduced outcome; a mismatch is an error.
    """
    index = inventory()
    root = Path(root).resolve(strict=True)
    output = _output_path(root, Path(output), index)
    _prepare(root, policy_spec, cell, index)
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    request = {
        "root": str(root),
        "policy_spec": policy_spec,
        "cell": cell,
        "output": str(output),
    }
    (runtime / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(runtime / "matplotlib"),
            "XDG_CACHE_HOME": str(runtime / "cache"),
            "TMPDIR": str(runtime),
            "TEMP": str(runtime),
            "TMP": str(runtime),
        }
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    with (runtime / "worker.log").open("w") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "_worker",
                "--request",
                str(runtime / "request.json"),
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    receipt = output / "release_receipt.json"
    if completed.returncode or not receipt.is_file():
        raise ReleaseError(
            f"Tracking replay failed (exit {completed.returncode}); inspect "
            f"{runtime / 'worker.log'} and {output / 'release_failure.json'}"
        )
    return json.loads(receipt.read_text())


def _runtime_versions(index):
    if sys.version_info[:2] != (3, 11):
        raise ReleaseError("Frozen sources require Python 3.11")
    observed = {"python": sys.version.split()[0]}
    for package, expected in index["runtime"]["model_recorded_versions"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ReleaseError(
                f"Missing runtime package: {package}=={expected}"
            ) from error
        observed[package] = actual
        comparable = actual.split("+")[0] if package == "torch" else actual
        if comparable != expected:
            raise ReleaseError(
                f"Frozen-model runtime mismatch: {package}={actual}, "
                f"expected {expected}"
            )
    for package in ("pandas", "PyYAML", "matplotlib"):
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ReleaseError(f"Missing evaluator dependency: {package}") from error
    return observed


def _verify_import_origins(root, index):
    origins = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name in index["runtime"]["module_files"]:
            expected = contained_path(root, index["runtime"]["module_files"][name])
            if not filename or Path(filename).resolve() != expected:
                raise ReleaseError(f"Wrong frozen module origin: {name}: {filename}")
            origins[name] = str(expected.relative_to(root))
        elif (
            name == "c2g_env"
            or name == "baselines"
            or name.startswith(("c2g_env.", "baselines."))
        ):
            raise ReleaseError(f"Uninventoried simulator module was imported: {name}")
    return origins


def _trace_contracts(directory: Path, family: str) -> dict:
    import pandas as pd

    hardware = pd.read_csv(directory / "hardware.csv.gz", float_precision="round_trip")
    full_work = family in FULL_WORK
    if full_work and not (hardware.throttle_batch == 1).all():
        raise ReleaseError(
            "A full-work family changed its frozen throttle=1 restriction"
        )
    recovery_path = directory / "recovery.csv.gz"
    recovery_rows = 0
    if recovery_path.exists():
        recovery = pd.read_csv(recovery_path, float_precision="round_trip")
        recovery_rows = len(recovery)
        if not (
            (recovery.throttle_batch == 1).all()
            and (recovery.delta_p_demanded_kw == 0).all()
        ):
            raise ReleaseError("Recovery work or zero-obligation contract changed")
    return {
        "full_work_enforced_for_this_family": full_work,
        "service_rows": len(hardware),
        "recovery_rows": recovery_rows,
        "native_classical_or_old_sac_actions_not_overridden": family not in FULL_WORK,
    }


def _worker(request_path: Path):
    request = json.loads(request_path.read_text())
    index = inventory()
    root = Path(request["root"]).resolve(strict=True)
    output = _output_path(root, Path(request["output"]), index)
    runtime_dir = output / "runtime"
    try:
        plan = _prepare(root, request["policy_spec"], request["cell"], index)
        versions = _runtime_versions(index)
        import yaml

        config = yaml.safe_load(
            contained_path(root, plan["original_config"]).read_text()
        )
        relocated = rebase_configuration(config, root, index)
        config_path = runtime_dir / "environment.yaml"
        config_path.write_text(yaml.safe_dump(relocated, sort_keys=False))
        protocol = copy.deepcopy(plan["runtime_protocol"])
        protocol["config_path"] = str(config_path)
        (runtime_dir / "execution_protocol.json").write_text(
            json.dumps(protocol, indent=2, allow_nan=False) + "\n"
        )

        # Initialize plotting cache before study.py sets its historical cache env var.
        import matplotlib

        matplotlib.get_cachedir()
        sys.path[:0] = [
            str(contained_path(root, directory, is_file=False))
            for directory in index["runtime"]["import_directories"]
        ]
        importlib.import_module("study")
        os.environ["MPLCONFIGDIR"] = str(runtime_dir / "matplotlib")
        common = importlib.import_module("common")
        table_evaluator = importlib.import_module("evaluate")
        state_evaluator = importlib.import_module("panel")
        analysis = importlib.import_module("analyze_final")
        origins = _verify_import_origins(root, index)
        final_lock_validated = False
        if plan["authorization"] == "value_final":
            original_protocol = json.loads(
                contained_path(root, plan["original_protocol_path"]).read_text()
            )
            original_lock = state_evaluator.final_lock(original_protocol)
            _check_membership(plan["cell"], original_lock["cells"], "Final-B cell lock")
            _check_membership(
                plan["archived_policy"],
                original_lock["policies"],
                "Final-B policy lock",
            )
            final_lock_validated = True
        common.cap_cpu()
        directory = output / "episode"
        if plan["evaluator"] == "table2":
            result = table_evaluator.run_case(
                plan["policy"], plan["cell"], protocol, directory
            )
        elif plan["evaluator"] == "state":
            result = state_evaluator.evaluate_case(
                protocol, plan["policy"], plan["cell"], directory
            )
        else:
            raise ReleaseError("Unknown frozen single-case evaluator")
        result = json.loads((directory / "episode.json").read_text())
        if result["reset_info"] != plan["expected_reset_info"]:
            raise ReleaseError("Replay source/physical reset contract differs")
        if result["policy"] != plan["policy"] or result["cell"] != plan["cell"]:
            raise ReleaseError("Frozen evaluator returned a different policy or cell")

        ratio = result.get("delivered_service_ratio")
        if result["complete"] and plan["policy"]["macro"] == "full_residual":
            ratio = float(
                analysis.capacity_floor(analysis.read_frame(directory / "macro.csv.gz"))
            )
            if "delivered_service_ratio" in result and not math.isclose(
                ratio, result["delivered_service_ratio"], rel_tol=0, abs_tol=1e-12
            ):
                raise ReleaseError(
                    "Original evaluator and publication capacity gates differ"
                )
        metrics = extract_metrics(result, ratio=ratio)
        contracts = _trace_contracts(directory, plan["policy"]["family"])
        comparison = compare_metrics(plan["expected_metrics"], metrics)
        verify_unchanged_inputs(root, plan["verified_inputs"])
        receipt = {
            "schema": "dc2grid-table2-integrated-consumed-case-replay-v1",
            "verified": not comparison,
            "new_checkout": str(root),
            "original_head": index["original_head"],
            "policy_spec": plan["policy_spec"],
            "cell": plan["cell"],
            "archived_policy": plan["archived_policy"],
            "resolved_policy": plan["policy"],
            "original_registration_id": plan["original_registration_id"],
            "original_final_registration_id": plan["original_final_registration_id"],
            "authorization": plan["authorization"],
            "unchanged_final_lock_api_validated": final_lock_validated,
            "selected_model": plan["selected_model"],
            "verified_inputs": plan["verified_inputs"],
            "runtime_config_sha256": file_hash(config_path),
            "runtime_config_only_changed_fields": [
                "global.trace_dir",
                "global.energy_dir",
                "global.weather_dir",
            ],
            "runtime_versions": versions,
            "integration_import_origins": origins,
            "integration_core_commit": CORE_COMMIT,
            "source_binding_changes": index["source_binding_changes"],
            "evaluator_api": (
                "evaluate.run_case"
                if plan["evaluator"] == "table2"
                else "panel.evaluate_case"
            ),
            "source_episode": str(directory / "episode.json"),
            "expected_metrics": plan["expected_metrics"],
            "actual_metrics": metrics,
            "metric_differences": comparison,
            "trace_contract_checks": contracts,
            "no_training_or_new_panel": True,
            "original_registrations_unchanged": True,
            "integrated_runtime_sources_verified": True,
            "scientific_inputs_rehashed_after_replay": True,
            "qualification_failure_is_not_a_replay_failure": True,
        }
        (output / "release_receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
        if comparison:
            raise ReleaseError(f"Replay differs from consumed metrics: {comparison}")
        _verify_import_origins(root, index)
    except Exception as error:
        (output / "release_failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "no_fallback_attempted": True,
                },
                indent=2,
            )
            + "\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List representative consumed requests; no rollout")
    for command in ("inspect", "evaluate"):
        cli = sub.add_parser(command)
        cli.add_argument("--root", required=True, type=Path)
        cli.add_argument("--snapshot", required=True)
        cli.add_argument("--policy", required=True)
        cli.add_argument("--day", required=True, type=int)
        cli.add_argument("--seed", required=True, type=int)
        cli.add_argument("--bias", required=True, type=float)
        if command == "evaluate":
            cli.add_argument("--output", required=True, type=Path)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        _worker(args.request)
        return
    if args.command == "list":
        print(
            json.dumps(
                [
                    {key: row[key] for key in ("id", "policy_spec", "cell", "reason")}
                    for row in inventory()["representatives"]
                ],
                indent=2,
            )
        )
        return
    spec = {"snapshot": args.snapshot, "policy_id": args.policy}
    cell = {"start_day": args.day, "seed": args.seed, "regd_load_bias": args.bias}
    if args.command == "inspect":
        plan = prepare(args.root, spec, cell)
        print(
            json.dumps(
                {
                    "authorized": True,
                    "no_rollout": True,
                    "evaluator": plan["evaluator"],
                    "selected_model": plan["selected_model"],
                    "verified_inputs": len(plan["verified_inputs"]),
                    "expected_metrics": plan["expected_metrics"],
                },
                indent=2,
            )
        )
    else:
        receipt = evaluate(args.root, spec, cell, args.output)
        print(
            json.dumps(
                {
                    "verified": receipt["verified"],
                    "source_episode": receipt["source_episode"],
                    "actual_metrics": receipt["actual_metrics"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
