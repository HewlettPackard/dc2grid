"""Optional single-pair replay of the consumed Q2/Q3 PID semantics diagnostic.

The archived probe is fingerprinted and parsed, NEVER imported or executed.
Only unchanged evaluation.grid_relief native APIs run the two trajectories.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

if __package__:
    from . import tracking_release as release
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tracking_release as release

REPORT = (
    "outputs/pid_comparison_extract_20261102/reports/pid_semantics_reproduction.json"
)
PROBE = (
    "outputs/pid_comparison_extract_20261102/diagnostics/probe_pid_relief_semantics.py"
)
SOURCE_DIRECTORIES = ("c2g_env", "baselines", "evaluation", "scripts")
SUMMARY_ASSIGNMENTS = (
    "load",
    "threshold",
    "stress",
    "facility",
    "cooling",
    "actual_pcc",
    "demanded",
    "result",
)


def inventory() -> dict:
    result = json.loads(
        Path(__file__).with_name("pid_semantics_inventory.json").read_text()
    )
    if result.get("schema") != "dc2grid-pid-semantics-replay-v1":
        raise release.ReleaseError("Unsupported PID diagnostic inventory")
    return result


def normalize_request(request: dict) -> dict:
    if not isinstance(request, dict) or set(request) != {"mode", "day", "seed"}:
        raise release.ReleaseError(
            "PID request has exactly mode, day, seed; no overrides"
        )
    if request["mode"] not in ("legacy_v0", "physical_v1"):
        raise release.ReleaseError(
            "Only the two original PID semantics modes are allowed"
        )
    if type(request["day"]) is not int or request["day"] not in (207, 248, 249):
        raise release.ReleaseError("No new dates: only original days 207, 248, 249")
    if type(request["seed"]) is not int or request["seed"] not in range(100, 105):
        raise release.ReleaseError("Only the original noise seeds 100 through 104")
    return dict(request)


def trace_frame(directory):
    import pandas as pd

    data = pd.read_csv(directory / "hardware.csv.gz")
    context = json.loads((directory / "episode.json").read_text())
    data.attrs.update(context["metric_context"])
    return data


def summarize_pair(mode, day, seed, agent, baseline, metrics):
    import numpy as np

    load = agent.load_mw.to_numpy()
    threshold = agent.attrs.get("stress_threshold_mw", np.quantile(load, 0.9))
    stress = load >= threshold
    facility = baseline.p_fac_mw.to_numpy() - agent.p_fac_mw.to_numpy()
    cooling = baseline.p_cooling_mw.to_numpy() - agent.p_cooling_mw.to_numpy()
    actual_pcc = facility + agent.bess_actual_kw.to_numpy() / 1000
    demanded = agent.dp_dem_kw.to_numpy() / 1000
    result = {
        "mode": mode,
        "day": day,
        "seed": seed,
        "agent_ticks": len(agent),
        "baseline_ticks": len(baseline),
        "reported_formula_relief_mw": float(metrics["dP_stress_mw"]),
        "stress_cooling_relief_mw": float(cooling[stress].mean()),
        "stress_other_facility_relief_mw": float((facility - cooling)[stress].mean()),
        "stress_nominal_facility_mw": float(
            baseline.p_fac_mw.to_numpy()[stress].mean()
        ),
        "stress_actual_facility_mw": float(agent.p_fac_mw.to_numpy()[stress].mean()),
        "stress_demanded_response_mw": float(demanded[stress].mean()),
        "stress_true_metered_error_mw": float((actual_pcc - demanded)[stress].mean()),
        "stress_battery_discharge_mw": float(
            agent.bess_actual_kw.to_numpy()[stress].mean() / 1000
        ),
        "stress_observed_regd": float(agent.regd_observed.to_numpy()[stress].mean()),
        "stress_actual_commitment_mw": float(
            agent.committed_actual_mw.to_numpy()[stress].mean()
        ),
        "stress_observed_commitment_mw": float(
            agent.committed_observed_mw.to_numpy()[stress].mean()
        ),
        "stress_hvac_effort": float(agent.hvac_effort.to_numpy()[stress].mean()),
        "stress_pump_speed": float(agent.pump_speed_A.to_numpy()[stress].mean()),
        "stress_temp_A": float(agent.temp_A.to_numpy()[stress].mean()),
        "stress_temp_B": float(agent.temp_B.to_numpy()[stress].mean()),
        "stress_mean_grid_load_mw": float(load[stress].mean()),
        "stress_tick_count": int(stress.sum()),
    }
    return result


def verify_formula_parity(probe_source: str, adapter_source: str) -> dict:
    original = ast.parse(probe_source)
    adapter = ast.parse(adapter_source)
    main = next(
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    summary = next(
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef) and node.name == "summarize_pair"
    )

    def assignments(function):
        selected = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in SUMMARY_ASSIGNMENTS
        ]
        selected.sort(key=lambda node: node.lineno)
        if tuple(node.targets[0].id for node in selected) != SUMMARY_ASSIGNMENTS:
            raise release.ReleaseError(
                "Unexpected archived PID component formula layout"
            )
        return [ast.dump(node, include_attributes=False) for node in selected]

    if assignments(main) != assignments(summary):
        raise release.ReleaseError(
            "Portable component formulas differ from archived probe"
        )
    original_frame = next(
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == "frame"
    )
    adapter_frame = next(
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef) and node.name == "trace_frame"
    )
    body = lambda function: [  # noqa: E731
        ast.dump(node, include_attributes=False)
        for node in function.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if body(original_frame) != body(adapter_frame):
        raise release.ReleaseError("PID trace reader differs from the original probe")
    return {
        "component_assignment_ASTs_identical": True,
        "trace_reader_AST_identical": True,
        "archived_probe_imported_or_executed": False,
    }


def source_paths(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for directory in SOURCE_DIRECTORIES
        for path in (root / directory).rglob("*.py")
    }


def _prepare(root: Path, request: dict, index: dict) -> dict:
    root = root.resolve(strict=True)
    request = normalize_request(request)
    verified = {}
    for relative, item in index["files"].items():
        if not item["required_for_replay"]:
            continue
        path = release.contained_path(root, relative)
        actual = release.file_hash(path)
        if actual != item["sha256"]:
            raise release.ReleaseError(
                f"Original PID input SHA-256 mismatch: {relative}"
            )
        verified[relative] = actual
    if source_paths(root) != set(index["source_identity_files"]):
        raise release.ReleaseError(
            "Native source_id requires the exact original four Python source trees; "
            "do not omit originals or add release helpers within those trees"
        )
    actual_source_id = release.identity(
        {path: verified[path] for path in index["source_identity_files"]}
    )
    if actual_source_id != index["source_id"]:
        raise release.ReleaseError("Native PID scientific source identity changed")
    report = json.loads(release.contained_path(root, REPORT).read_text())
    if (
        report["source_id"] != actual_source_id
        or report["recorded_probe_source_sha256"] != verified[PROBE]
        or report["case_pair_count"] != 30
        or report["diagnostic_run_count"] != 60
        or report["per_mode_case_pairs"] != 15
        or report["role"] != "diagnostic_reproduction_not_new_publication_evidence"
        or report["all_60_trajectories_complete_24h_service"] is not True
        or report[
            "recovery_not_run_or_needed_for_this_service_only_metric_reproduction"
        ]
        is not True
    ):
        raise release.ReleaseError(
            "Archived 60-trajectory PID diagnostic contract changed"
        )
    keys = [(row["mode"], row["day"], row["seed"]) for row in report["cases"]]
    allowed = {
        (mode, day, seed)
        for mode in ("legacy_v0", "physical_v1")
        for day in (207, 248, 249)
        for seed in range(100, 105)
    }
    if len(keys) != 30 or set(keys) != allowed:
        raise release.ReleaseError("PID replay is restricted to the original 30 pairs")
    expected = next(
        row
        for row in report["cases"]
        if (row["mode"], row["day"], row["seed"])
        == (request["mode"], request["day"], request["seed"])
    )
    formula_check = verify_formula_parity(
        release.contained_path(root, PROBE).read_text(), Path(__file__).read_text()
    )
    return {
        "root": str(root),
        "request": request,
        "expected": expected,
        "verified_inputs": verified,
        "formula_check": formula_check,
        "source_id": actual_source_id,
        "expected_native_guards": {
            "qualified": True,
            "comparison_status": "complete",
            "agent_ticks": 17280,
            "baseline_ticks": 17280,
            "paired_ticks": 17280,
            "expected_ticks": 17280,
        },
    }


def prepare_pid_pair(root: Path, request: dict) -> dict:
    """Verify source/data/report hashes and exact pair membership; no simulation."""
    return _prepare(Path(root), request, inventory())


def rebase_config(configuration: dict, root: Path, index: dict) -> dict:
    result = copy.deepcopy(configuration)
    for key, relative in index["data_directories"].items():
        original = configuration["global"][key]
        recorded = (
            release.archive_relative(original, index)
            if Path(original).is_absolute()
            else original
        )
        if recorded != relative:
            raise release.ReleaseError(f"Unexpected native data directory: {key}")
        path = release.contained_path(root, relative, is_file=False)
        if not path.is_dir():
            raise release.ReleaseError(f"Missing native data directory: {relative}")
        result["global"][key] = str(path)
    return result


def compare_values(expected: dict, actual: dict) -> list[dict]:
    if set(expected) != set(actual):
        return [
            {"field": "<keys>", "expected": sorted(expected), "actual": sorted(actual)}
        ]
    differences = []
    for key, left in expected.items():
        right = actual[key]
        if isinstance(left, float):
            valid = (
                type(right) in (float, int)
                and math.isfinite(right)
                and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
            )
        else:
            valid = type(left) is type(right) and left == right
        if not valid:
            differences.append({"field": key, "expected": left, "actual": right})
    return differences


def check_power_balance(values: dict) -> None:
    relief = values["reported_formula_relief_mw"]
    identities = {
        "cooling_plus_other": (
            values["stress_cooling_relief_mw"]
            + values["stress_other_facility_relief_mw"]
        ),
        "nominal_minus_actual": (
            values["stress_nominal_facility_mw"] - values["stress_actual_facility_mw"]
        ),
        "demand_minus_battery_plus_metered_error": (
            values["stress_demanded_response_mw"]
            - values["stress_battery_discharge_mw"]
            + values["stress_true_metered_error_mw"]
        ),
    }
    for name, actual in identities.items():
        if not math.isclose(relief, actual, rel_tol=0, abs_tol=1e-8):
            raise release.ReleaseError(
                f"Original PID power-component balance failed: {name}"
            )


def evaluate_pid_pair(root: Path, request: dict, output: Path) -> dict:
    """Replay PID + fixed-setpoint reference for ONE already-consumed mode/day/seed."""
    index = inventory()
    root = Path(root).resolve(strict=True)
    output = release._output_path(root, Path(output), index)
    _prepare(root, request, index)
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    (runtime / "request.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "request": request,
                "output": str(output),
            },
            indent=2,
        )
        + "\n"
    )
    environment = os.environ.copy()
    environment.update(
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
        environment[key] = "1"
    with (runtime / "worker.log").open("w") as log:
        finished = subprocess.run(
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
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    receipt = output / "pid_release_receipt.json"
    if finished.returncode or not receipt.is_file():
        raise release.ReleaseError(
            f"PID replay failed; inspect {runtime / 'worker.log'} "
            f"and {output / 'pid_release_failure.json'}"
        )
    return json.loads(receipt.read_text())


def _verify_origins(root, index):
    origins = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name in index["runtime_modules"]:
            expected = release.contained_path(root, index["runtime_modules"][name])
            if not filename or Path(filename).resolve() != expected:
                raise release.ReleaseError(f"Wrong native PID import origin: {name}")
            origins[name] = str(expected.relative_to(root))
        elif name == "run_benchmark" or name.startswith(
            ("c2g_env.", "baselines.", "evaluation.")
        ):
            raise release.ReleaseError(f"Uninventoried native module: {name}")
    return origins


def _worker(request_path: Path):
    specification = json.loads(request_path.read_text())
    index = inventory()
    root = Path(specification["root"]).resolve(strict=True)
    output = release._output_path(root, Path(specification["output"]), index)
    runtime = output / "runtime"
    try:
        plan = _prepare(root, specification["request"], index)
        if sys.version_info[:2] != (3, 11):
            raise release.ReleaseError(
                "The original native evaluator requires Python 3.11"
            )
        versions = {"python": sys.version.split()[0]}
        for package in (
            "numpy",
            "pandas",
            "torch",
            "gymnasium",
            "stable-baselines3",
            "PyYAML",
            "scipy",
            "tqdm",
        ):
            versions[package] = importlib.metadata.version(package)
        import torch
        import yaml

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        sys.path[:0] = [str(root), str(root / "evaluation")]
        native = importlib.import_module("evaluation.grid_relief")
        benchmark = importlib.import_module("run_benchmark")
        origins = _verify_origins(root, index)
        if (
            benchmark._PLANT_OVERRIDES is not None
            or benchmark._PLANT_PROFILE != "none"
            or benchmark._SIMULATION_OPTIONS
            or benchmark._ALLOW_SEMANTIC_MISMATCH
        ):
            raise release.ReleaseError("Native benchmark defaults were altered")
        if native.source_id(root) != plan["source_id"]:
            raise release.ReleaseError(
                "Unchanged native source_id disagrees with the archive"
            )
        request = plan["request"]
        mode, day, seed = request["mode"], request["day"], request["seed"]
        native_directory = runtime / "native"
        native_directory.mkdir()
        native_config_path = Path(
            native._market_config("nyiso_nyc", str(native_directory), mode)
        )
        native_config = yaml.safe_load(native_config_path.read_text())
        runtime_config = rebase_config(native_config, root, index)
        config_path = runtime / "environment.yaml"
        config_path.write_text(yaml.safe_dump(runtime_config))
        for hardware in ("pid", "none"):
            native._run_episode(
                hw=hardware,
                scenario="default",
                seed=seed,
                model_dir=None,
                macro="rule_macro",
                config_path=str(config_path),
                start_day=day,
                regd_bias=0.5,
                trace_dir=str(output / hardware),
                simulation_mode=mode,
            )
        contexts = {}
        for hardware in ("pid", "none"):
            context = json.loads((output / hardware / "episode.json").read_text())
            expected_context = {
                "scenario": "default",
                "seed": seed,
                "start_day": day,
                "regd_bias": 0.5,
                "macro": "rule_macro",
                "hardware": hardware,
                "hardware_checkpoint": None,
                "macro_checkpoint": None,
                "hardware_steps": 17280,
                "macro_steps": 96,
                "terminated": False,
                "truncated": True,
                "simulation_mode": mode,
                "source_id": plan["source_id"],
            }
            observed = {key: context[key] for key in expected_context}
            if compare_values(expected_context, observed):
                raise release.ReleaseError(
                    "Native PID trajectory metadata/gates changed"
                )
            contexts[hardware] = observed
        agent, baseline = trace_frame(output / "pid"), trace_frame(output / "none")
        native_metrics = native._metrics(agent, baseline)
        guards = {key: native_metrics[key] for key in plan["expected_native_guards"]}
        guard_differences = compare_values(plan["expected_native_guards"], guards)
        actual = summarize_pair(mode, day, seed, agent, baseline, native_metrics)
        check_power_balance(actual)
        differences = compare_values(plan["expected"], actual)
        release.verify_unchanged_inputs(root, plan["verified_inputs"])
        if source_paths(root) != set(index["source_identity_files"]):
            raise release.ReleaseError("Native source tree changed during PID replay")
        _verify_origins(root, index)
        receipt = {
            "schema": "dc2grid-pid-semantics-single-pair-receipt-v1",
            "verified": not differences and not guard_differences,
            "request": request,
            "new_checkout": str(root),
            "role": "diagnostic_reproduction_not_new_publication_evidence",
            "original_source_id": plan["source_id"],
            "verified_original_inputs": plan["verified_inputs"],
            "expected_full_components": plan["expected"],
            "actual_full_components": actual,
            "component_differences": differences,
            "expected_native_guards": plan["expected_native_guards"],
            "actual_native_guards": guards,
            "guard_differences": guard_differences,
            "trajectory_context_checks": contexts,
            "formula_parity": plan["formula_check"],
            "native_config_sha256": release.file_hash(native_config_path),
            "runtime_config_sha256": release.file_hash(config_path),
            "only_rebased_fields": [
                "global.trace_dir",
                "global.energy_dir",
                "global.weather_dir",
            ],
            "native_import_origins": origins,
            "runtime_versions": versions,
            "native_apis": ["_market_config", "_run_episode", "_metrics"],
            "archived_probe_executed": False,
            "global_OUT_or_PROJECT_overridden": False,
            "adapter_monkeypatching": False,
            "native_legacy_bias_hook_retained_unchanged": mode == "legacy_v0",
            "recovery_performed": False,
            "pair_guard_is_not_26h_tracking_qualification": True,
            "original_component_balances_verified": True,
            "same_labels_do_not_mean_identical_cross_mode_exogenous_inputs": True,
            "not_individual_change_causal_attribution": True,
        }
        (output / "pid_release_receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
        if not receipt["verified"]:
            raise release.ReleaseError(
                f"PID reproduction differs: {differences + guard_differences}"
            )
    except Exception as error:
        (output / "pid_release_failure.json").write_text(
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
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for command in ("inspect", "evaluate"):
        child = commands.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument(
            "--mode", choices=("legacy_v0", "physical_v1"), required=True
        )
        child.add_argument("--day", type=int, required=True)
        child.add_argument("--seed", type=int, required=True)
        if command == "evaluate":
            child.add_argument("--output", type=Path, required=True)
    child = commands.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "_worker":
        _worker(arguments.request)
    elif arguments.command == "list":
        print(json.dumps(inventory()["representatives"], indent=2))
    else:
        request = {
            "mode": arguments.mode,
            "day": arguments.day,
            "seed": arguments.seed,
        }
        if arguments.command == "inspect":
            plan = prepare_pid_pair(arguments.root, request)
            print(
                json.dumps(
                    {
                        "authorized": True,
                        "no_trajectories_executed": True,
                        "source_id": plan["source_id"],
                        "expected": plan["expected"],
                        "native_guards": plan["expected_native_guards"],
                    },
                    indent=2,
                )
            )
        else:
            result = evaluate_pid_pair(arguments.root, request, arguments.output)
            print(
                json.dumps(
                    {
                        "verified": result["verified"],
                        "actual_full_components": result["actual_full_components"],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
