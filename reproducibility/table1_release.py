"""Reproduce the existing-checkpoint Table 1 on the main-based shared core."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The isolated worker has no checkout on sys.path until this explicit bootstrap.
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from reproducibility import assets, inputs  # noqa: E402
from reproducibility.runtime import (  # noqa: E402
    CORE_COMMIT,
    CORE_SOURCE_AMENDMENTS,
    file_hash,
    identity,
    inside,
    integration_sources,
    limit_threads,
    read_json,
    resolve_recorded,
)

STUDY_RELATIVE = "outputs/table1_corrected_existing_20261102"
STUDY = ROOT / STUDY_RELATIVE
SOURCE_COMMIT = "f477cdea7a1a7461d5d67fa27832c36d45d82388"
SOURCE_REGISTRATION = "ed5313c9b6566c816769176f0a99eac2bd9566af1f402e4cfd087437005a1971"
POLICIES = (
    "rule_macro+none",
    "rule_macro+random",
    "rule_macro+bang_bang",
    "rule_macro+rule_based",
    "rule_macro+pid",
    "rule_macro+sac",
    "sac_macro+sac",
)
SEEDS = tuple(range(100, 105))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def study_module(name: str):
    allowed = {
        "table1_eval",
        "prepare_table1",
        "analyze_table1",
        "render_table1",
        "verify_table1_publication",
    }
    require(name in allowed, "Unexpected frozen study module")
    sys.path.insert(0, str(STUDY))
    module = importlib.import_module(name)
    for imported in allowed.intersection(sys.modules):
        require(
            Path(sys.modules[imported].__file__).resolve() == STUDY / f"{imported}.py",
            f"Foreign study import: {imported}",
        )
    return module


def producer_protocol() -> dict:
    protocol = read_json(STUDY / "registration.json")
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    require(
        protocol["registration_id"] == SOURCE_REGISTRATION
        and identity(body) == SOURCE_REGISTRATION,
        "The original producer registration identity changed",
    )
    return protocol


def verify_export(*, with_inputs: bool = False) -> dict:
    verification = assets.verify(ROOT, "table1")
    original = read_json(STUDY / "original_table1.numeric.json")
    require("table_tex" not in original, "Embedded original manuscript TeX is excluded")
    require(len(original["rows"]) == 9, "The nine historical numeric rows changed")
    protocol = producer_protocol()
    for relative, expected in integration_sources(protocol["source_files"]).items():
        require(
            file_hash(inside(ROOT, relative)) == expected,
            f"Bound integration source changed: {relative}",
        )
    actual_sources = {
        str(path.relative_to(ROOT))
        for folder in ("c2g_env", "baselines", "evaluation", "scripts")
        for path in (ROOT / folder).rglob("*.py")
    }
    require(
        actual_sources == set(protocol["source_files"]),
        "The native source inventory differs from the declared shared core",
    )
    for descriptor in (
        protocol["config"],
        protocol["recovery_ops"],
        *protocol["models"].values(),
    ):
        path = resolve_recorded(ROOT, descriptor["path"])
        require(
            path.is_file() and file_hash(path) == descriptor["sha256"],
            f"Frozen configuration/recovery/model changed: {path}",
        )
    report = read_json(STUDY / "reports/table1_report.json")
    require(
        report["registration_id"] == SOURCE_REGISTRATION
        and report["planned"] == report["recorded"] == 35,
        "The source result cohort differs",
    )
    require(
        {(case["policy"], case["seed"]) for case in report["cases"]}
        == {(policy, seed) for policy in POLICIES for seed in SEEDS},
        "Missing or unregistered source case",
    )
    for case in report["cases"]:
        path = inside(ROOT, f"{STUDY_RELATIVE}/{case['episode_path']}")
        require(
            file_hash(path) == case["episode_sha256"],
            f"Original episode receipt changed: {case['episode_path']}",
        )
    if with_inputs:
        for relative, expected in protocol["data_files"].items():
            path = inside(ROOT, relative)
            require(
                path.is_file() and file_hash(path) == expected,
                f"Supply the exact authorized input dataset: {relative}",
            )
        require(
            inputs.status(ROOT)["ready_for_frozen_replay"],
            "The public input-data inventory is not ready",
        )
    return {
        **verification,
        "source_commit": SOURCE_COMMIT,
        "producer_registration_id": SOURCE_REGISTRATION,
        "integration_core_commit": CORE_COMMIT,
        "core_source_amendments": CORE_SOURCE_AMENDMENTS,
        "numeric_only_derived_file": 1,
        "episode_receipts": 35,
        "native_source_files": len(protocol["source_files"]),
        "complete_service": report["complete_service"],
        "qualified_supplemental": report["qualified_supplemental"],
        "checkpoints": {
            role: {
                **{key: value for key, value in spec.items() if key != "path"},
                "path": str(resolve_recorded(ROOT, spec["path"]).relative_to(ROOT)),
            }
            for role, spec in protocol["models"].items()
        },
        "input_datasets_verified": with_inputs,
        "input_datasets_checked": with_inputs,
        "public_adapter_sha256": file_hash(Path(__file__)),
    }


def numerical_views() -> tuple[dict[str, str], dict]:
    verification = verify_export()
    render = study_module("render_table1")
    report = read_json(STUDY / "reports/table1_report.json")
    protocol = producer_protocol()
    render.validate_report(report, protocol)
    corrected = render.display_panel(report["rows"], "corrected")
    archived = render.display_panel(
        read_json(STUDY / "original_table1.numeric.json")["rows"], "archived"
    )
    supplemental = render.supplemental_rows(report)
    stream = io.StringIO(newline="")
    fields = (
        "panel",
        "policy",
        "metric",
        "mean",
        "sd",
        "display",
        "complete_service",
        "qualified",
        "diagnostic",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for panel, rows in (("corrected", corrected), ("archived", archived)):
        for row in rows:
            for key in render.METRICS:
                metric = row["metrics"][key]
                writer.writerow(
                    {
                        "panel": panel,
                        "policy": row["policy"],
                        "metric": key,
                        "mean": metric["mean"],
                        "sd": metric["sd"],
                        "display": metric["display"],
                        "complete_service": row.get("complete_service"),
                        "qualified": row.get("qualified"),
                        "diagnostic": row.get("diagnostic_not_admissible"),
                    }
                )
    csv_text = stream.getvalue()
    require(
        csv_text.encode() == (STUDY / "publication/table1_cells.csv").read_bytes(),
        "Recomputed public CSV differs from the frozen 96-cell producer CSV",
    )
    values = {
        **verification,
        "schema": "table1-public-numeric-reconstruction-v1",
        "panel": protocol["panel"],
        "statistic": report["statistic"],
        "corrected": corrected,
        "archived": archived,
        "supplemental": supplemental,
        "not_rerun": protocol["missing_rows"],
        "nonmissing_mean_sd_cells": 93,
        "mean_sd_statistics": 186,
        "metric_cells_in_csv": 96,
        "original_archive_or_embedded_tex_read": False,
        "training_or_new_evaluation_performed": False,
    }
    tex = (
        r"\providecommand{\numcell}[2]{\shortstack[r]{$#1$\\$\pm #2$}}"
        + "\n"
        + "% Corrected service-only Table 1\n"
        + render.tex_table(corrected, corrected=True)
        + "\n\n"
        + "% Archived numeric Table 1; separate historical panel\n"
        + render.tex_table(archived, corrected=False)
        + "\n\n"
        + "% Supplemental 26-hour accounting, not primary Table 1 metrics\n"
        + render.supplemental_tex(supplemental)
        + "\n"
    )
    return {
        "table1_cells.csv": csv_text,
        "table1_numeric.tex": tex,
        "table1_values.json": json.dumps(values, indent=2, allow_nan=False) + "\n",
    }, values


def check_pdf(values: dict) -> dict:
    checker = study_module("verify_table1_publication")
    pdf = STUDY / "publication/table1_corrected_comparison.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    pages = [page for page in text.split("\f") if page.strip()]
    require(len(pages) == 4, "Expected the frozen four-page Table 1 PDF")
    count = checker.verify_panel(pages[0], values["corrected"], corrected=True)
    count += checker.verify_panel(pages[1], values["archived"], corrected=False)
    checker.verify_supplemental(pages[2], values["supplemental"])
    require(count == 93, "Not all PDF mean/SD cells were checked")
    return {
        "pages": 4,
        "mean_sd_cells": count,
        "word_bounds_checked": checker.verify_pdf_bounds(pdf),
        "sha256": file_hash(pdf),
    }


def validate_request(policy: str, seed: int) -> None:
    require(policy in POLICIES, "Policy is outside the seven-row Table 1 roster")
    require(type(seed) is int and seed in SEEDS, "Seed must be one of 100--104")


def execution_protocol() -> dict:
    original = producer_protocol()
    protocol = copy.deepcopy(original)
    for descriptor in (
        protocol["config"],
        protocol["recovery_ops"],
        *protocol["models"].values(),
    ):
        descriptor["path"] = str(resolve_recorded(ROOT, descriptor["path"]))
    protocol["source_files"] = integration_sources(original["source_files"])
    restored = copy.deepcopy(protocol)
    restored["source_files"] = original["source_files"]
    restored["config"]["path"] = original["config"]["path"]
    restored["recovery_ops"]["path"] = original["recovery_ops"]["path"]
    for role in original["models"]:
        restored["models"][role]["path"] = original["models"][role]["path"]
    require(
        restored == original,
        "Protocol changed beyond paths and the explicit Box-metadata source amendment",
    )
    protocol.pop("registration_id")
    protocol["public_replay"] = {
        "schema": "table1-integrated-runtime-rebase-v1",
        "producer_registration_id": SOURCE_REGISTRATION,
        "source_commit": SOURCE_COMMIT,
        "integration_core_commit": CORE_COMMIT,
        "source_amendments": CORE_SOURCE_AMENDMENTS,
        "changed_panel_model_gate_values": [],
    }
    protocol["registration_id"] = identity(protocol)
    return protocol


def replay_output(value: Path) -> Path:
    path = (value if value.is_absolute() else ROOT / value).resolve()
    replay_root = STUDY / "replays"
    require(
        path.is_relative_to(replay_root) and path != replay_root,
        f"Replay output must be a new directory below {replay_root}",
    )
    require(not path.exists(), "A replay output already exists; no overwrite/retry")
    return path


def compare_values(actual, expected, label: str = "case") -> int:
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(actual) == set(expected),
            f"Dictionary shape differs: {label}",
        )
        return sum(
            compare_values(actual[key], value, f"{label}.{key}")
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"List shape differs: {label}",
        )
        return sum(
            compare_values(value, reference, f"{label}[{index}]")
            for index, (value, reference) in enumerate(
                zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, float):
        tolerance = (
            1e-5
            if any(word in label.lower() for word in ("usd", "cost", "revenue"))
            else 1e-6
        )
        require(
            type(actual) in (float, int)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance),
            f"Numeric result differs: {label}: {actual!r} != {expected!r}",
        )
    else:
        require(
            type(actual) is type(expected) and actual == expected,
            f"Discrete result differs: {label}: {actual!r} != {expected!r}",
        )
    return 1


def compare_episode(actual: dict, expected: dict) -> dict:
    count = 0
    for key in (
        "schema",
        "policy",
        "seed",
        "panel",
        "status",
        "training_performed",
        "llm_calls_performed",
        "reset_plant_state",
        "initial_macro_observation",
        "initial_hardware_observation",
        "service",
        "supplemental_accounting",
    ):
        count += compare_values(actual[key], expected[key], key)
    observed_recovery = copy.deepcopy(actual["supplemental_recovery"])
    recorded_recovery = copy.deepcopy(expected["supplemental_recovery"])
    for recovery in (observed_recovery, recorded_recovery):
        extension = recovery["extension"]
        require(
            extension["state_preserved"]
            and extension["service_replay_prefix_preserved"]
            and extension["state_digest_before"] == extension["state_digest_after"],
            "Recovery extension did not preserve the continuing physical state",
        )
        extension.pop("state_digest_before")
        extension.pop("state_digest_after")
    count += compare_values(
        observed_recovery, recorded_recovery, "supplemental_recovery"
    )
    return {
        "verified": True,
        "scalar_checks": count,
        "service_complete": actual["service"]["complete"],
        "qualified_supplemental": actual["supplemental_accounting"][
            "all_supplemental_gates_pass"
        ],
        "source_failure_outcome_preserved": actual["status"] == expected["status"],
        "float_tolerance": {"rtol": 1e-9, "atol": 1e-6, "USD_atol": 1e-5},
        "not_compared": [
            "Checkout-specific paths, registration IDs and host/runtime metadata",
            "Cross-run state-digest strings; before/after preservation checked per run",
            "Compressed trace hashes and wall-clock timing",
        ],
    }


def evaluate(policy: str, seed: int, output: Path) -> dict:
    validate_request(policy, seed)
    verification = verify_export(with_inputs=True)
    protocol = execution_protocol()
    destination = replay_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    runtime = destination / "runtime"
    runtime.mkdir()
    write_json(runtime / "execution_protocol.json", protocol)
    write_json(runtime / "request.json", {"policy": policy, "seed": seed})
    log = runtime / "worker.log"
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "_worker",
        "--policy",
        policy,
        "--seed",
        str(seed),
        "--output",
        str(destination),
    ]
    with log.open("x") as stream:
        process = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if process.returncode:
        write_json(
            destination / "replay_failure.json",
            {
                "status": "worker_failed",
                "returncode": process.returncode,
                "worker_log": str(log),
                "source_registration_id": SOURCE_REGISTRATION,
            },
        )
        raise RuntimeError(f"Frozen Table 1 replay failed; inspect {log}")
    observed_path = destination / "episode/episode.json"
    expected_path = (
        STUDY / "cases" / policy.replace("+", "__") / f"seed_{seed}" / "episode.json"
    )
    observed, expected = read_json(observed_path), read_json(expected_path)
    require(
        observed["registration_id"] == protocol["registration_id"],
        "The replay episode is not bound to the derived execution protocol",
    )
    for key in (
        "config",
        "source_files",
        "data_files",
        "models",
        "recovery_ops",
        "evaluator_sha256",
    ):
        require(
            observed["provenance"][key] == protocol[key],
            f"The replay used different scientific inputs: {key}",
        )
    require(
        set(observed["trace_hashes"]) == set(expected["trace_hashes"]),
        "The replay trace inventory differs from the original case",
    )
    for name, digest in observed["trace_hashes"].items():
        require(
            file_hash(observed_path.parent / name) == digest,
            f"A replay trace differs from its episode receipt: {name}",
        )
    try:
        comparison = compare_episode(observed, expected)
    except (ValueError, KeyError, TypeError) as error:
        write_json(
            destination / "replay_failure.json",
            {
                "status": "comparison_failed",
                "error": str(error),
                "observed_episode": str(observed_path),
                "source_episode": str(expected_path),
            },
        )
        raise
    verify_export(with_inputs=True)
    receipt = {
        "schema": "table1-integrated-replay-receipt-v1",
        **comparison,
        "policy": policy,
        "seed": seed,
        "source_commit": SOURCE_COMMIT,
        "producer_registration_id": SOURCE_REGISTRATION,
        "execution_registration_id": protocol["registration_id"],
        "integration_core_commit": CORE_COMMIT,
        "core_source_amendments": CORE_SOURCE_AMENDMENTS,
        "source_binding_sha256": verification["source_binding_sha256"],
        "public_adapter_sha256": verification["public_adapter_sha256"],
        "original_episode_sha256": file_hash(expected_path),
        "replayed_episode_sha256": file_hash(observed_path),
        "training_performed": False,
        "new_case_selection_performed": False,
        "output": str(destination),
    }
    write_json(destination / "replay_receipt.json", receipt)
    return receipt


def worker(policy: str, seed: int, output: Path) -> None:
    validate_request(policy, seed)
    require(
        output.resolve().is_relative_to(STUDY / "replays"),
        "Worker output is outside the public replay namespace",
    )
    verify_export(with_inputs=True)
    protocol = read_json(output / "runtime/execution_protocol.json")
    require(
        protocol == execution_protocol(), "Execution protocol was changed after mapping"
    )
    require(
        read_json(output / "runtime/request.json") == {"policy": policy, "seed": seed},
        "Execution request changed",
    )
    evaluator = study_module("table1_eval")
    evaluator.run_case(protocol, policy, seed, output / "episode")


def main() -> None:
    limit_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--with-inputs", action="store_true")
    verify.add_argument("--check-pdf", action="store_true")
    tables = commands.add_parser("tables")
    tables.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "_worker"):
        sub = commands.add_parser(name)
        sub.add_argument("--policy", choices=POLICIES, required=True)
        sub.add_argument("--seed", type=int, choices=SEEDS, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_export(with_inputs=args.with_inputs)
        _, values = numerical_views()
        result["numeric_metric_cells"] = values["metric_cells_in_csv"]
        result["nonmissing_mean_sd_cells"] = values["nonmissing_mean_sd_cells"]
        if args.check_pdf:
            result["pdf"] = check_pdf(values)
    elif args.command == "tables":
        outputs, values = numerical_views()
        destination = (
            args.output if args.output.is_absolute() else ROOT / args.output
        ).resolve()
        require(destination.is_relative_to(ROOT), "Output must remain inside checkout")
        require(not destination.exists(), "Numeric output already exists")
        destination.mkdir(parents=True)
        for name, text in outputs.items():
            (destination / name).write_bytes(text.encode())
        result = {
            "verified": True,
            "output": str(destination),
            "metric_cells": values["metric_cells_in_csv"],
            "nonmissing_mean_sd_cells": values["nonmissing_mean_sd_cells"],
        }
    elif args.command == "evaluate":
        result = evaluate(args.policy, args.seed, args.output)
    else:
        worker(args.policy, args.seed, args.output)
        return
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
