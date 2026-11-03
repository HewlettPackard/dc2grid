"""Freeze the existing-model Table 1 protocol before any production episode."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import subprocess
import zipfile
from pathlib import Path

import table1_eval as evaluator

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASELINE = "34e015461a4c7e6a4fc1df8b61a72af92fc2db7a"
REGISTRATION = ROOT / "registration.json"
IMPLEMENTATION = (
    "table1_eval.py",
    "test_table1_eval.py",
    "test_table1_registration.py",
    "prepare_table1.py",
    "run_table1.py",
)


def original_table1() -> dict:
    archive_path = PROJECT / "NIPS26_Data_Center_Energy_Grid_Benchmark.zip"
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read("camera_ready.tex")
    text = raw.decode("utf-8")
    active = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )
    marker = active.index(r"\label{tab:results_default}")
    start = active.rfind(r"\begin{table}", 0, marker)
    end = active.index(r"\end{table}", marker) + len(r"\end{table}")
    table = active[start:end]
    identifiers = (
        "rule_macro+none",
        "rule_macro+random",
        "rule_macro+bang_bang",
        "rule_macro+rule_based",
        "rule_macro+pid",
        "rule_macro+sac",
        "sac_macro+sac",
        "ppo_macro+sac",
        "llm_macro+sac",
    )
    metrics = (
        "tracking_rmse_kw",
        "regulation_revenue_usd",
        "macro_mean_reward",
        "low_mean_reward",
        "thermal_warning_fraction",
        "bess_aging_x1e4",
    )
    rows = []
    for line in table.splitlines():
        if not line.lstrip().startswith(r"\quad"):
            continue
        columns = line.split("&")
        if len(columns) != 7:
            raise ValueError("Original Table 1 does not have the expected six metrics")
        values = {}
        for key, cell in zip(metrics, columns[1:], strict=True):
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", cell.replace("{,}", ""))
            if len(numbers) not in (0, 2):
                raise ValueError("Cannot parse an archived Table 1 mean/SD pair")
            values[key] = (
                {"mean": float(numbers[0]), "sd": float(numbers[1])}
                if numbers
                else {"mean": None, "sd": None}
            )
        rows.append({"id": identifiers[len(rows)], "metrics": values})
    if len(rows) != 9:
        raise ValueError("Original Table 1 must contain exactly nine rows")
    return {
        "source_archive_sha256": evaluator.file_hash(archive_path),
        "camera_ready_member_sha256": hashlib.sha256(raw).hexdigest(),
        "member_hash_kind": "SHA256 of the exact archived camera_ready.tex bytes",
        "member": "camera_ready.tex",
        "table_label": "tab:results_default",
        "source_lines": [614, 648],
        "exact_producing_cases_or_checkpoint_mapping_recovered": False,
        "reported_legacy_values_not_new_results": True,
        "rows": rows,
        "table_tex": table,
    }


def build() -> dict:
    if REGISTRATION.exists():
        raise FileExistsError("Refusing to overwrite an existing registration")
    sources = {}
    for relative in evaluator.runtime_source_paths():
        path = Path(relative)
        name = str(path.relative_to(PROJECT) if path.is_absolute() else path)
        sources[name] = evaluator.file_hash(PROJECT / name)
    implementation = {
        str((ROOT / name).relative_to(PROJECT)): evaluator.file_hash(ROOT / name)
        for name in IMPLEMENTATION
    }
    original = original_table1()
    original_path = ROOT / "original_table1.json"
    if original_path.exists():
        raise FileExistsError("Refusing to replace original Table 1 source evidence")
    original_path.write_text(json.dumps(original, indent=2, allow_nan=False) + "\n")
    protocol = {
        "schema": "table1-corrected-existing-models-v1",
        "role": "new_corrected_default_panel_not_exact_archival_reproduction",
        "source_baseline_commit": BASELINE,
        "panel": copy.deepcopy(evaluator.PANEL_SPEC),
        "policies": copy.deepcopy(evaluator.ROW_SPECS),
        "models": copy.deepcopy(evaluator.MODEL_SPECS),
        "config": evaluator.file_receipt(PROJECT / "c2g_env/config.yaml"),
        "source_files": sources,
        "data_files": {
            str(path): evaluator.file_hash(PROJECT / path)
            for path in evaluator.DATA_PATHS
        },
        "evaluator_sha256": evaluator.file_hash(ROOT / "table1_eval.py"),
        "implementation_files": implementation,
        "recovery_ops": copy.deepcopy(evaluator.RECOVERY_OPS_SPEC),
        "recovery": copy.deepcopy(evaluator.RECOVERY_SPEC),
        "service_gate": copy.deepcopy(evaluator.SERVICE_GATE_SPEC),
        "original_table1": evaluator.file_receipt(original_path),
        "planned_cases": 35,
        "training_performed": False,
        "checkpoint_selection_using_new_Table1_outcomes": False,
        "missing_rows": {
            "ppo_macro+sac": (
                "No compatible corrected original-bidding PPO checkpoint; "
                "no new training authorized."
            ),
            "llm_macro+sac": (
                "Pinned paper-conformant LLM execution not established; "
                "no LLM calls authorized."
            ),
        },
        "aggregation": {
            "expected_cases_per_row": 5,
            "sample_standard_deviation_ddof": 1,
            "primary_scope": "24-hour service only",
            "all_cases_must_complete_for_primary_mean": True,
            "supplemental_failures_remain_in_complete_cohort_means": True,
            "incomplete_primary_values": None,
            "prefix_diagnostics_not_ranked_against_full_service": True,
            "capacity_admission_floor_or_credit_guard": None,
            "no_training_seed_ensemble": True,
        },
        "supplemental_scope": (
            "Existing two-hour physical recovery, metered 26-hour cash and "
            "battery throughput; not Table1 service metrics."
        ),
        "known_frozen_limits": [
            "RegD is not guaranteed energy-neutral per window.",
            "Voltage interface does not reflect signed battery PCC exchange.",
            "Work is an energy-equivalent proxy, not measured job/FLOP completion.",
            (
                "Remaining queue/clock/reference-state observation limitations "
                "are not repaired by this evaluation."
            ),
        ],
        "execution": {
            "workers": 8,
            "worker_cpus": list(range(32, 40)),
            "numerical_threads": 1,
            "inference_device": "cpu",
            "retry_existing_attempts": False,
            "physical_failure_is_a_recorded_outcome": True,
        },
        "python_version_at_registration": platform.python_version(),
        "original_archive_and_previous_studies_must_remain_unchanged": True,
        "automatic_publishing_authorized": False,
    }
    protocol["registration_id"] = evaluator.identity(protocol)
    REGISTRATION.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return protocol


def verify(*, require_committed: bool = True) -> dict:
    protocol = json.loads(REGISTRATION.read_text())
    body = {key: value for key, value in protocol.items() if key != "registration_id"}
    if evaluator.identity(body) != protocol["registration_id"]:
        raise ValueError("The fixed evaluation registration changed")
    for group in ("source_files", "data_files", "implementation_files"):
        for relative, expected in protocol[group].items():
            if evaluator.file_hash(PROJECT / relative) != expected:
                raise ValueError(f"Registered dependency changed: {relative}")
    for record in (
        protocol["config"],
        protocol["original_table1"],
        protocol["recovery_ops"],
        *protocol["models"].values(),
    ):
        if evaluator.file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Registered artifact changed: {record['path']}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"],
        cwd=PROJECT,
        check=True,
    )
    if require_committed:
        paths = list(protocol["implementation_files"])
        paths += [
            str(REGISTRATION.relative_to(PROJECT)),
            str((ROOT / "original_table1.json").relative_to(PROJECT)),
        ]
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *paths],
            cwd=PROJECT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *paths],
            cwd=PROJECT,
            check=True,
        )
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    protocol = build() if args.command == "prepare" else verify()
    print(
        json.dumps(
            {
                "registration_id": protocol["registration_id"],
                "planned": protocol["planned_cases"],
                "rows": len(protocol["policies"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
