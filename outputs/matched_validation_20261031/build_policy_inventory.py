"""Audit the fixed candidate universe; never load policies or run environments.

Run from the repository with .venv/bin/python -B. The default writes only
policy_inventory.json and policy_inventory_audit.json beside this script.
--check repeats the audit, checks reproducibility, and writes nothing.
Checkpoint inspection reads only the ZIP's JSON `data` member. Replay files
are hashed as opaque bytes, never deserialized. No source module is imported.
"""

from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:40]))

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
BASELINE = ROOT / "outputs/corrected_sac_8gpu_20261030"
PILOT = ROOT / "outputs/online_stability_pilot_20261031"
INVENTORY = OUTPUT / "policy_inventory.json"
REPORT = OUTPUT / "policy_inventory_audit.json"
SCHEMA = "matched-validation-policy-inventory-v1"
PARENTS = (1000, 1001, 1006, 1007)
ORIGINAL_CELLS = (
    (196, 0.0, 90001),
    (196, 0.5, 90002),
    (256, 0.0, 90003),
    (256, 0.5, 90004),
)
EXPECTED_MISSING = {"old_low_1000", "old_pair_1000", "old_pair_1005"}
EXPECTED_DIAGNOSTIC = {
    "restored_1000_0",
    "restored_1000_1",
    "restored_1000_2",
    "restored_1001_0",
    "restored_1001_2",
}
GATE = {
    "min_terminal_soc": 0.2,
    "max_backlog_fraction": 0.01,
    "max_thermal_warning_fraction": 0.01,
    "score": "mean_step_reward",
    "require_all_validation_cells": True,
}
CLASSICAL = {
    "nominal": ("none", "c2g_env/env_high_level.py", "C2GMacroEnv(inner_action_fn=None)"),
    "pid": ("pid", "baselines/pid_controller.py", "PIDController"),
    "rule_hardware": (
        "rule_based",
        "baselines/rule_based_mpc.py",
        "RuleBasedController",
    ),
    "bang_bang": ("bang_bang", "baselines/bang_bang.py", "BangBangController"),
}
CAVEATS = [
    "Original candidates are the best qualified checkpoints selected across training; "
    "restored candidates are all registered critic_restore final-budget checkpoints. "
    "This is a candidate inventory, not evidence of algorithmic superiority.",
    "The twelve restored candidates are three continuation streams nested within each "
    "of four selected parent seeds, not twelve independent from-scratch training seeds.",
    "Re-evaluations of the four original validation cells are not independent new "
    "samples. The four bias counterparts reuse the same day's two seeds; matched "
    "comparisons remain dependent within day, seed, parent, and continuation stream.",
    "Prior qualification is based on the four original cells, not a claim of "
    "qualification on the eight-cell matched validation matrix.",
    "parent_selected_low_id on restored policies identifies a same-seed comparison "
    "candidate, not the continuation's actual late-step initial weights.",
    "Macro and hardware selection_score values are mean reward per their respective "
    "decision step and must not be compared directly across control levels.",
]


def canonical_id(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def condition_key(value):
    return value["start_day"], value["regd_load_bias"], value["seed"]


def cell_id(day, bias, seed):
    return f"d{day}_s{seed}_b{format(bias, 'g').replace('.', 'p')}"


def make_cells():
    return [
        {
            "id": cell_id(day, bias, seed),
            "start_day": day,
            "regd_load_bias": bias,
            "seed": seed,
            "original_validation_cell": (day, bias, seed) in ORIGINAL_CELLS,
        }
        for day, seeds in ((196, (90001, 90002)), (256, (90003, 90004)))
        for seed in seeds
        for bias in (0.0, 0.5)
    ]


class Audit:
    def __init__(self):
        self.discrepancies = []
        self.files = {}
        self.stats = {}
        self.absences = set()
        self.snapshots = {}
        self.model_data = {}
        self.hash_bytes = 0
        self.checks = 0
        self.validation_suites = Counter()
        self.manifest_pairings = 0

    def expect(self, condition, check, actual=None, expected=None):
        self.checks += 1
        if not condition:
            self.discrepancies.append(
                {"check": check, "actual": actual, "expected": expected}
            )
        return bool(condition)

    def path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Read outside the designated repository: {path}")
        return path

    def fingerprint(self, path, role, expected=None, required=True):
        path = self.path(path)
        key = str(path)
        if not path.is_file():
            self.absences.add(key)
            if required:
                self.expect(False, f"required_file:{key}", "absent", "regular file")
            return None
        if key not in self.files:
            stat = path.stat()
            if len(self.files) >= 2048 or self.hash_bytes + stat.st_size > 2**31:
                raise ValueError("Inventory audit exceeded its 2048-file/2-GiB hash bound")
            before = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            after = path.stat()
            self.expect(
                before == (after.st_ino, after.st_size, after.st_mtime_ns),
                f"stable_during_hash:{key}",
            )
            self.hash_bytes += stat.st_size
            self.files[key] = {
                "path": key,
                "sha256": digest,
                "size_bytes": stat.st_size,
                "roles": [],
            }
            self.stats[key] = before
        receipt = self.files[key]
        if role not in receipt["roles"]:
            receipt["roles"].append(role)
        if expected is not None:
            self.expect(
                receipt["sha256"] == expected,
                f"sha256:{key}",
                receipt["sha256"],
                expected,
            )
        return {"path": key, "sha256": receipt["sha256"]}

    def text(self, path, role, expected=None):
        receipt = self.fingerprint(path, role, expected)
        if receipt is None:
            raise FileNotFoundError(path)
        path = self.path(path)
        if path.stat().st_size > 8 * 2**20:
            raise ValueError(f"Metadata exceeds 8 MiB: {path}")
        return path.read_text()

    def read_json(self, path, role, expected=None):
        return json.loads(self.text(path, role, expected))

    def snapshot(self, saved):
        path = self.path(saved["source_archive"])
        receipt = self.fingerprint(path, "original_source_snapshot")
        if receipt is None:
            return None
        if str(path) not in self.snapshots:
            with zipfile.ZipFile(path) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if len(members) > 2048 or sum(i.file_size for i in members) > 64 * 2**20:
                    raise ValueError(f"Source snapshot exceeds bounded inspection: {path}")
                names = [item.filename for item in members]
                self.expect(len(names) == len(set(names)), f"unique_zip_members:{path}")
                hashes = {
                    item.filename: hashlib.sha256(archive.read(item)).hexdigest()
                    for item in members
                }
            self.snapshots[str(path)] = {
                "archive": receipt,
                "source_id": canonical_id(hashes),
                "python_source_id": canonical_id(
                    {name: digest for name, digest in hashes.items() if name.endswith(".py")}
                ),
                "member_count": len(hashes),
                "member_hashes": hashes,
            }
        result = self.snapshots[str(path)]
        self.expect(
            result["source_id"] == saved["source_id"],
            f"snapshot_content_identity:{path}",
            result["source_id"],
            saved["source_id"],
        )
        return {key: value for key, value in result.items() if key != "member_hashes"}

    def model(self, path, role, expected=None):
        path = self.path(path)
        receipt = self.fingerprint(path, role, expected)
        if receipt is None:
            return None, None, None
        if str(path) not in self.model_data:
            with zipfile.ZipFile(path) as archive:
                members = [item for item in archive.infolist() if item.filename == "data"]
                if len(members) != 1 or members[0].file_size > 8 * 2**20:
                    raise ValueError(f"Missing, duplicate, or oversized model JSON: {path}")
                raw = archive.read(members[0])
            data = json.loads(raw)
            context = data.get("c2g_context")
            if not isinstance(context, dict) or ":serialized:" in context:
                raise ValueError(f"c2g_context is not plain JSON: {path}")
            self.model_data[str(path)] = (data, hashlib.sha256(raw).hexdigest())
        data, data_sha = self.model_data[str(path)]
        self.expect(
            data["c2g_context"].get("simulation_mode") == "physical_v1",
            f"model_physical_v1:{path}",
        )
        metadata = {
            "zip_member": "data",
            "zip_data_sha256": data_sha,
            "num_timesteps": data.get("num_timesteps"),
            "optimizer_updates": data.get("_n_updates"),
            "saved_seed": data.get("seed"),
            "simulation_mode": data["c2g_context"].get("simulation_mode"),
            "decoded_serialized_objects": False,
        }
        return receipt, data, metadata

    def quality(self, report, label, family):
        cells = report["cells"]
        self.validation_suites[family] += 1
        identities = [condition_key(cell["condition"]) for cell in cells]
        self.expect(
            len(identities) == 4 and set(identities) == set(ORIGINAL_CELLS),
            f"four_original_validation_cells:{label}",
            identities,
            ORIGINAL_CELLS,
        )
        summaries = []
        for index, cell in enumerate(cells):
            tag = f"{label}:cell_{index}"
            complete = (
                cell["fast_steps"] == cell["expected_ticks"]
                and cell["truncated"]
                and not cell["terminated"]
            )
            conservation = (
                cell["work_arrived_kwh"]
                - cell["work_served_kwh"]
                - cell["terminal_backlog_kwh"]
            )
            warning = cell["thermal_warning_ticks"] / cell["fast_steps"]
            backlog_limit = GATE["max_backlog_fraction"] * cell["work_arrived_kwh"] + 1e-6
            failures = []
            for name, passed in (
                ("incomplete_episode", complete),
                ("work_conservation", abs(conservation) < 1e-4),
                ("terminal_backlog", cell["terminal_backlog_kwh"] <= backlog_limit),
                ("terminal_soc", cell["terminal_soc"] >= GATE["min_terminal_soc"]),
                ("thermal_warning", warning <= GATE["max_thermal_warning_fraction"]),
            ):
                if not passed:
                    failures.append(name)
            qualified = not failures
            self.expect(cell["expected_ticks"] == 17280, f"full_day_ticks:{tag}")
            self.expect(cell["complete"] == complete, f"cell_complete:{tag}")
            self.expect(cell["qualified"] == qualified, f"cell_qualified:{tag}")
            for name, calculated in (
                ("work_conservation_error_kwh", conservation),
                ("thermal_warning_fraction", warning),
                ("mean_step_reward", cell["reward_sum"] / cell["steps"]),
                ("tracking_rmse_kw", math.sqrt(cell["error_sq_sum"] / cell["fast_steps"])),
            ):
                self.expect(
                    math.isclose(cell[name], calculated, rel_tol=1e-12, abs_tol=1e-12),
                    f"recomputed_{name}:{tag}",
                )
            summaries.append(
                {
                    "condition": cell["condition"],
                    "complete": complete,
                    "qualified": qualified,
                    "failure_reasons": failures,
                    "terminal_soc": cell["terminal_soc"],
                    "terminal_backlog_kwh": cell["terminal_backlog_kwh"],
                    "thermal_warning_fraction": warning,
                }
            )
        all_qualified = bool(cells) and all(cell["qualified"] for cell in summaries)
        all_complete = bool(cells) and all(cell["complete"] for cell in summaries)
        score = sum(cell["mean_step_reward"] for cell in cells) / len(cells)
        self.expect(report["attempts"] == len(cells), f"attempt_count:{label}")
        self.expect(
            report["qualified_count"] == sum(cell["qualified"] for cell in summaries),
            f"qualified_count:{label}",
        )
        self.expect(
            report["all_cells_qualified"] == all_qualified, f"all_qualified:{label}"
        )
        self.expect(
            math.isclose(report["selection_score"], score, rel_tol=1e-12, abs_tol=1e-12),
            f"selection_score:{label}",
        )
        if "all_cells_complete" in report:
            self.expect(report["all_cells_complete"] == all_complete, f"all_complete:{label}")
        return summaries

    def check_unchanged(self):
        for name, before in self.stats.items():
            path = Path(name)
            after = path.stat() if path.is_file() else None
            signature = (
                (after.st_ino, after.st_size, after.st_mtime_ns) if after else None
            )
            self.expect(signature == before, f"unchanged_through_audit:{name}")
        for name in self.absences:
            self.expect(not Path(name).exists(), f"absence_unchanged:{name}")


def original_context(audit, data, seed, stage, baseline, inner, label):
    context = data["c2g_context"]
    expected = {
        "stage": stage,
        "seed": seed,
        "training_seed": seed,
        "protocol_id": baseline["protocol_id"],
        "source_id": baseline["source_id"],
        "simulation_mode": "physical_v1",
        "inner_checkpoint": inner,
        "protocol": baseline["protocol"],
    }
    for key, value in expected.items():
        audit.expect(context.get(key) == value, f"saved_context_{key}:{label}")
    audit.expect(data.get("seed") == seed, f"saved_algorithm_seed:{label}")
    snapshot = audit.snapshot(context["source_snapshot"])
    if snapshot:
        audit.expect(
            snapshot["source_id"] == baseline["source_snapshot"]["source_id"],
            f"saved_snapshot_matches_baseline:{label}",
        )
        audit.expect(
            snapshot["python_source_id"] == baseline["source_id"],
            f"saved_python_source_identity:{label}",
        )
    return snapshot


def old_stage(audit, baseline, campaign_status, seed, stage, inner):
    directory = BASELINE / f"seed_{seed}" / stage
    model_path = directory / "best_model/best_model.zip"
    selection_path = directory / "best_model/selection.json"
    issue_start = len(audit.discrepancies)
    provenance = {
        "selection_rule": "highest_mean_step_reward_among_fully_qualified_original_checkpoints",
        "qualification_basis": "four_original_validation_cells",
        "simulation_mode": "physical_v1",
        "protocol_id": baseline["protocol_id"],
        "source_id": baseline["source_id"],
        "expected_checkpoint_path": str(model_path),
        "expected_selection_receipt_path": str(selection_path),
        "fallback_to_final_weights": False,
        "manifest_inner_checkpoint": None,
        "saved_c2g_context_inner_checkpoint": None,
    }
    if not directory.exists():
        job = next(job for job in campaign_status["jobs"] if job["seed"] == seed)
        audit.expect(
            seed == 1000 and stage == "high" and job["state"] == "blocked" and inner is None,
            f"untrained_stage_explained:{directory}",
        )
        audit.absences.update((str(directory), str(model_path), str(selection_path)))
        provenance.update(
            training_state="not_trained",
            blocking_campaign_job=job,
            selection_status="not_trained_no_qualified_frozen_hardware",
            validation_history=[],
        )
        return None, False, False, "Macro not trained: no qualified frozen seed-1000 hardware checkpoint.", provenance

    manifest = audit.read_json(directory / "manifest.json", "original_stage_manifest")
    status = audit.read_json(directory / "status.json", "original_stage_status")
    protocol = audit.read_json(directory / "protocol.json", "original_stage_protocol")
    audit.expect(protocol == baseline["protocol"], f"stage_protocol:{directory}")
    for record_name, record in (("manifest", manifest), ("status", status)):
        for key, expected in (
            ("stage", stage),
            ("seed", seed),
            ("protocol_id", baseline["protocol_id"]),
            ("source_id", baseline["source_id"]),
            ("simulation_mode", "physical_v1"),
            ("inner_checkpoint", inner),
        ):
            audit.expect(record.get(key) == expected, f"{record_name}_{key}:{directory}")
    audit.expect(status["state"] == "completed", f"completed_stage:{directory}")
    if stage == "high":
        audit.manifest_pairings += 1
        audit.expect(inner is not None, f"trained_macro_has_selected_inner:{directory}")
    history = []
    validation_paths = sorted(
        (directory / "validation").glob("step_*.json"),
        key=lambda path: int(path.stem.removeprefix("step_")),
    )
    if len(validation_paths) > 16:
        raise ValueError(f"Unexpectedly large original validation history: {directory}")
    for path in validation_paths:
        report = audit.read_json(path, "original_validation")
        audit.quality(report, str(path), "original_history")
        audit.expect(
            report["timesteps"] == int(path.stem.removeprefix("step_")),
            f"validation_step_filename:{path}",
        )
        history.append((path, report))
    audit.expect(
        len(history) == status["validation_calls"] == (8 if stage == "low" else 6),
        f"complete_selection_history:{directory}",
    )
    qualified = [item for item in history if item[1]["all_cells_qualified"]]
    best = max(qualified, key=lambda item: item[1]["selection_score"]) if qualified else None
    checkpoint = audit.fingerprint(model_path, "selected_original_model", required=False)
    selection_receipt = audit.fingerprint(
        selection_path, "original_selection_receipt", required=False
    )
    audit.expect(
        (checkpoint is not None) == (selection_receipt is not None) == bool(best),
        f"selected_artifact_presence_matches_qualified_history:{directory}",
    )
    audit.expect(
        status["qualified_checkpoint"] == bool(best),
        f"stage_any_qualified_status:{directory}",
    )
    audit.expect(
        status["best_qualified_score"] == (best[1]["selection_score"] if best else None),
        f"stage_best_score:{directory}",
    )
    provenance.update(
        training_state=status["state"],
        manifest=audit.fingerprint(directory / "manifest.json", "original_stage_manifest"),
        status=audit.fingerprint(directory / "status.json", "original_stage_status"),
        manifest_inner_checkpoint=manifest["inner_checkpoint"],
        selection_receipt=selection_receipt,
        qualified_validation_checkpoints=len(qualified),
        validation_history=[
            {
                "receipt": audit.fingerprint(path, "original_validation"),
                "timesteps": report["timesteps"],
                "all_cells_qualified": report["all_cells_qualified"],
                "qualified_count": report["qualified_count"],
                "selection_score": report["selection_score"],
            }
            for path, report in history
        ],
    )
    if not checkpoint or not selection_receipt:
        provenance["selection_status"] = "no_fully_qualified_checkpoint"
        reason = (
            f"No fully qualified {'hardware' if stage == 'low' else 'macro'} checkpoint "
            f"selected across {len(history)} validation suites; no final-weight fallback."
        )
        return checkpoint, False, False, reason, provenance
    selection = audit.read_json(selection_path, "original_selection_receipt")
    audit.quality(selection, str(selection_path), "original_selection")
    audit.expect(selection["checkpoint"] == checkpoint, f"selection_hash:{selection_path}")
    audit.expect(
        selection["protocol_id"] == baseline["protocol_id"],
        f"selection_protocol:{selection_path}",
    )
    if best:
        audit.expect(
            {key: value for key, value in selection.items() if key not in ("checkpoint", "protocol_id")}
            == best[1],
            f"exact_best_qualified_validation_receipt:{selection_path}",
        )
    checkpoint, data, metadata = audit.model(
        model_path, "selected_original_model", selection["checkpoint"]["sha256"]
    )
    snapshot = original_context(audit, data, seed, stage, baseline, inner, str(model_path))
    audit.expect(
        data["num_timesteps"] == selection["timesteps"], f"selected_model_step:{model_path}"
    )
    audit.expect(
        data["_n_updates"] == selection["optimizer_updates"],
        f"selected_model_updates:{model_path}",
    )
    provenance.update(
        selection_status="best_fully_qualified_checkpoint",
        selected_timesteps=selection["timesteps"],
        selected_optimizer_updates=selection["optimizer_updates"],
        selected_score=selection["selection_score"],
        selected_validation=audit.fingerprint(best[0], "original_validation") if best else None,
        selected_cell_qualification=audit.quality(
            selection, f"selected:{model_path}", "selected_policy_summary"
        ),
        checkpoint_metadata=metadata,
        saved_source_snapshot=snapshot,
        saved_c2g_context_inner_checkpoint=data["c2g_context"]["inner_checkpoint"],
        frozen_inner_hash_verified=stage == "high" and data["c2g_context"]["inner_checkpoint"] == inner,
    )
    available = len(audit.discrepancies) == issue_start
    reason = None if available else "Checkpoint/selection provenance discrepancy; see audit."
    return checkpoint, bool(selection["all_cells_qualified"]), available, reason, provenance


def policy(identifier, group, seed=None, stream=None, hardware="sac", macro="rule_macro"):
    return {
        "id": identifier,
        "group": group,
        "base_seed": seed,
        "stream": stream,
        "hardware": hardware,
        "macro": macro,
        "hardware_checkpoint": None,
        "macro_checkpoint": None,
        "available": True,
        "prior_selection_qualified": None if group == "classical" else False,
        "diagnostic_only": False,
        "unavailable_reason": None,
        "matched_rule_id": None,
        "parent_selected_low_id": None,
        "selection_provenance": {},
    }


def registered_inputs(audit, baseline, registration, seed, inputs):
    start = len(audit.discrepancies)
    late_step = registration["pilot_protocol"]["late_step"]
    donor_step = registration["pilot_protocol"]["donor_step"]
    receipts = {}
    models = {}
    for name, step in (("late_model", late_step), ("donor_model", donor_step)):
        directory = BASELINE / f"seed_{seed}/low/checkpoints/step_{step}"
        checkpoint = audit.read_json(directory / "checkpoint.json", "registered_input_receipt")
        audit.expect(checkpoint["model"] == inputs[name], f"registered_{name}_receipt:{seed}")
        audit.expect(
            inputs[name]["path"] == str(directory / "model.zip"),
            f"registered_{name}_path:{seed}",
        )
        audit.expect(checkpoint["timesteps"] == step, f"registered_input_step:{seed}:{name}")
        actual, data, metadata = audit.model(
            directory / "model.zip", "registered_parent_model", inputs[name]["sha256"]
        )
        original_context(audit, data, seed, "low", baseline, None, actual["path"])
        audit.expect(data["num_timesteps"] == step, f"registered_model_step:{seed}:{name}")
        audit.expect(
            data["_n_updates"] == checkpoint["optimizer_updates"],
            f"registered_model_updates:{seed}:{name}",
        )
        receipts[name] = {
            "artifact": actual,
            "checkpoint_receipt": audit.fingerprint(
                directory / "checkpoint.json", "registered_input_receipt"
            ),
            "metadata": metadata,
        }
        models[name] = data
        if name == "late_model":
            audit.expect(
                checkpoint["replay_buffer"] == inputs["late_replay"],
                f"registered_replay_receipt:{seed}",
            )
            audit.expect(
                inputs["late_replay"]["path"] == str(directory / "replay_buffer.pkl"),
                f"registered_replay_path:{seed}",
            )
            receipts["late_replay"] = {
                "artifact": audit.fingerprint(
                    directory / "replay_buffer.pkl",
                    "opaque_registered_replay_bytes",
                    inputs["late_replay"]["sha256"],
                ),
                "deserialized": False,
            }
    return {
        "verified": len(audit.discrepancies) == start,
        "receipts": receipts,
        "late_context": models["late_model"]["c2g_context"],
        "late_updates": models["late_model"]["_n_updates"],
    }


def restored_policy(audit, registration, readback, final_rows, pilot_status, job, inputs):
    seed, stream = job["base_seed"], job["stream"]
    identifier = f"restored_{seed}_{stream}"
    result = policy(identifier, "restored", seed, stream)
    result["parent_selected_low_id"] = f"old_low_{seed}"
    start = len(audit.discrepancies)
    directory = PILOT / "runs" / job["id"]
    budget = registration["pilot_protocol"]["online_transitions"]
    model_path = directory / "final_model.zip"
    validation_path = directory / "validation" / f"online_{budget}.json"
    expected = readback["final_artifact_hashes"][job["id"]]
    final = audit.read_json(
        validation_path, "restored_final_validation", expected["final_validation_sha256"]
    )
    checkpoint, data, metadata = audit.model(
        model_path, "restored_final_model", expected["final_model_sha256"]
    )
    manifest = audit.read_json(directory / "manifest.json", "restored_manifest")
    status = audit.read_json(directory / "status.json", "restored_status")
    initialization = audit.read_json(directory / "initialization.json", "restored_initialization")
    context = data["c2g_context"]
    identity = {
        "job_id": job["id"],
        "base_seed": seed,
        "continuation_stream": stream,
        "arm": "critic_restore",
        "registration_id": registration["registration_id"],
        "simulation_mode": "physical_v1",
    }
    for name, record in (("context", context), ("manifest", manifest), ("status", status)):
        for key, value in identity.items():
            audit.expect(record.get(key) == value, f"restored_{name}_{key}:{identifier}")
    for key, value in manifest.items():
        audit.expect(context.get(key) == value, f"restored_context_manifest_{key}:{identifier}")
    audit.expect(manifest["inputs"] == job["inputs"], f"registered_inputs:{identifier}")
    audit.expect(context["parent_context"] == inputs["late_context"], f"exact_late_parent_context:{identifier}")
    audit.expect(status["state"] == "completed", f"restored_completed:{identifier}")
    campaign_job = next(record for record in pilot_status["jobs"] if record["id"] == job["id"])
    audit.expect(
        campaign_job["state"] == "completed" and campaign_job["returncode"] == 0,
        f"registered_job_completed:{identifier}",
    )
    for record_name, record in (("status", status), ("validation", final)):
        audit.expect(
            record["new_online_transitions"] == record["new_normal_sac_updates"] == budget,
            f"final_budget_accounting_{record_name}:{identifier}",
        )
    audit.expect(
        initialization["base_timesteps"] == registration["pilot_protocol"]["late_step"],
        f"late_initialization_step:{identifier}",
    )
    audit.expect(
        initialization["base_sac_updates"] == inputs["late_updates"],
        f"late_initialization_updates:{identifier}",
    )
    audit.expect(
        data["num_timesteps"]
        == final["timesteps"]
        == status["timesteps"]
        == initialization["base_timesteps"] + budget,
        f"final_model_timestep:{identifier}",
    )
    audit.expect(
        data["_n_updates"]
        == status["optimizer_updates"]
        == initialization["base_sac_updates"] + budget,
        f"final_model_updates:{identifier}",
    )
    audit.expect(
        status["actual_replay_position"] == status["expected_replay_position"],
        f"final_replay_accounting:{identifier}",
    )
    audit.expect(
        context["critic_only_calibration_updates"] == registration["pilot_protocol"]["critic_calibration_steps"],
        f"registered_calibration_budget:{identifier}",
    )
    cell_summary = audit.quality(final, str(validation_path), "restored_final")
    qualified = final["all_cells_qualified"]
    audit.expect(final["all_cells_complete"], f"restored_full_final_validation:{identifier}")
    row = final_rows[job["id"]]
    audit.expect(
        int(row["base_seed"]) == seed
        and int(row["stream"]) == stream
        and row["arm"] == job["arm"]
        and (row["all_qualified"] == "True") == qualified
        and int(row["qualified_cells"]) == final["qualified_count"]
        and int(row["new_transitions"]) == int(row["normal_sac_updates"]) == budget,
        f"readback_final_run_row:{identifier}",
    )
    for csv_key, json_key in (
        ("reward", "selection_score"),
        ("rmse_kw", "mean_rmse_kw"),
        ("nrmse", "mean_normalized_rmse"),
    ):
        audit.expect(
            math.isclose(float(row[csv_key]), final[json_key], rel_tol=1e-12, abs_tol=1e-12),
            f"readback_{csv_key}:{identifier}",
        )
    result.update(
        hardware_checkpoint=checkpoint,
        prior_selection_qualified=qualified,
        diagnostic_only=not qualified,
        available=inputs["verified"] and len(audit.discrepancies) == start,
        selection_provenance={
            "selection_rule": "all_registered_critic_restore_final_budget_hardware_checkpoints",
            "selection_status": "final_budget_qualified" if qualified else "final_budget_unqualified_diagnostic",
            "qualification_basis": "four_original_validation_cells_at_final_budget",
            "simulation_mode": "physical_v1",
            "registered_job": job,
            "registration_id": registration["registration_id"],
            "registration": audit.fingerprint(PILOT / "registration.json", "pilot_registration"),
            "manifest": audit.fingerprint(directory / "manifest.json", "restored_manifest"),
            "status": audit.fingerprint(directory / "status.json", "restored_status"),
            "initialization": audit.fingerprint(directory / "initialization.json", "restored_initialization"),
            "final_budget_validation": audit.fingerprint(validation_path, "restored_final_validation"),
            "final_readback": audit.fingerprint(PILOT / "reports/readback.json", "pilot_final_readback"),
            "final_readback_expected_hashes": expected,
            "final_budget_cell_qualification": cell_summary,
            "final_budget_qualified_cells": final["qualified_count"],
            "final_budget_score": final["selection_score"],
            "new_online_transitions": budget,
            "selected_timesteps": final["timesteps"],
            "checkpoint_metadata": metadata,
            "registered_inputs_verified": inputs["verified"],
            "registered_input_receipts": inputs["receipts"],
            "late_parent_context_sha256": canonical_id(context["parent_context"]),
            "parent_selected_low_link_is_comparator_not_initial_weights": True,
            "any_earlier_qualified_checkpoint_reported": status["qualified_checkpoint"],
            "secondary_best_qualified_checkpoint_used": False,
            "learned_macro_for_these_final_weights": "not_attempted",
        },
    )
    if not result["available"]:
        result["unavailable_reason"] = "Registered final checkpoint provenance discrepancy; see audit."
    return result


def inventory_errors(inventory):
    errors = []

    def check(condition, label):
        if not condition:
            errors.append(label)

    check(
        set(inventory)
        == {
            "schema", "created_at", "policies", "cells", "selection_gate",
            "held_out_days", "not_attempted_surfaces", "source_receipts",
        },
        "root_fields",
    )
    check(inventory["schema"] == SCHEMA, "schema")
    policies = inventory["policies"]
    expected_ids = (
        set(CLASSICAL)
        | {f"old_low_{seed}" for seed in range(1000, 1008)}
        | {f"old_pair_{seed}" for seed in range(1000, 1008)}
        | {f"restored_{seed}_{stream}" for seed in PARENTS for stream in range(3)}
    )
    ids = [entry["id"] for entry in policies]
    check(len(ids) == len(set(ids)) == 32 and set(ids) == expected_ids, "exact_unique_universe")
    check(inventory["cells"] == make_cells(), "exact_eight_matched_validation_cells")
    check(len({cell["id"] for cell in inventory["cells"]}) == 8, "unique_cell_ids")
    check(sum(cell["original_validation_cell"] for cell in inventory["cells"]) == 4, "four_original_cells")
    check(inventory["held_out_days"] == [207, 248, 249], "held_out_days")
    check(
        not {cell["start_day"] for cell in inventory["cells"]} & set(inventory["held_out_days"]),
        "no_held_out_cells",
    )
    check({key: inventory["selection_gate"].get(key) for key in GATE} == GATE, "selection_gate")
    by_id = {entry["id"]: entry for entry in policies}
    check({p["id"] for p in policies if not p["available"]} == EXPECTED_MISSING, "expected_unavailable")
    check({p["id"] for p in policies if p["diagnostic_only"]} == EXPECTED_DIAGNOSTIC, "five_diagnostic_finals")
    check(sum(p["available"] for p in policies) == 29, "29_available")
    fields = set(policy("example", "old_low"))
    for entry in policies:
        identifier = entry["id"]
        check(set(entry) == fields, f"policy_fields:{identifier}")
        check(entry["hardware"] in {"none", "pid", "rule_based", "bang_bang", "sac"}, f"hardware:{identifier}")
        check(entry["macro"] in {"rule_macro", "sac_macro"}, f"macro:{identifier}")
        check(entry["selection_provenance"].get("simulation_mode") == "physical_v1", f"physical_v1:{identifier}")
        check(type(entry["available"]) is bool and type(entry["diagnostic_only"]) is bool, f"boolean_flags:{identifier}")
        check((entry["unavailable_reason"] is None) == entry["available"], f"availability_reason:{identifier}")
        for key in ("hardware_checkpoint", "macro_checkpoint"):
            artifact = entry[key]
            if artifact is not None:
                check(
                    set(artifact) == {"path", "sha256"}
                    and Path(artifact["path"]).is_absolute()
                    and bool(re.fullmatch("[0-9a-f]{64}", artifact["sha256"])),
                    f"checkpoint_fingerprint:{identifier}:{key}",
                )
        group = entry["group"]
        if group == "classical":
            check(
                identifier in CLASSICAL
                and entry["hardware"] == CLASSICAL[identifier][0]
                and entry["base_seed"] is None
                and entry["stream"] is None
                and entry["prior_selection_qualified"] is None
                and entry["parent_selected_low_id"] is None,
                f"classical_identity:{identifier}",
            )
        else:
            check(type(entry["prior_selection_qualified"]) is bool, f"prior_qualification_boolean:{identifier}")
            check(entry["hardware"] == "sac", f"learned_hardware:{identifier}")
            if entry["available"]:
                check(entry["hardware_checkpoint"] is not None, f"available_hardware:{identifier}")
        if group in ("classical", "old_low", "restored"):
            check(entry["macro"] == "rule_macro" and entry["macro_checkpoint"] is None, f"rule_macro:{identifier}")
            check(entry["matched_rule_id"] is None, f"not_learned_pair:{identifier}")
        if group in ("old_low", "old_pair"):
            seed = entry["base_seed"]
            check(identifier == f"{group}_{seed}" and seed in range(1000, 1008), f"original_identity:{identifier}")
            check(entry["stream"] is None, f"original_stream:{identifier}")
            if entry["available"]:
                check(entry["prior_selection_qualified"] is True and not entry["diagnostic_only"], f"qualified_original:{identifier}")
            key = "macro_checkpoint" if group == "old_pair" else "hardware_checkpoint"
            artifact = entry[key]
            if artifact is not None:
                check(artifact["path"].endswith("/best_model/best_model.zip"), f"no_original_final_fallback:{identifier}")
        if group == "old_low":
            check(entry["parent_selected_low_id"] is None, f"original_low_has_no_parent:{identifier}")
        if group == "old_pair":
            low_id = f"old_low_{entry['base_seed']}"
            low = by_id.get(low_id)
            check(entry["macro"] == "sac_macro", f"learned_macro:{identifier}")
            check(entry["matched_rule_id"] == low_id and entry["parent_selected_low_id"] == low_id, f"matched_low_id:{identifier}")
            check(low is not None and entry["hardware_checkpoint"] == low["hardware_checkpoint"], f"pair_exact_low_hash:{identifier}")
            if entry["available"]:
                check(
                    entry["macro_checkpoint"] is not None
                    and entry["selection_provenance"]["saved_c2g_context_inner_checkpoint"] == entry["hardware_checkpoint"]
                    and entry["selection_provenance"]["frozen_inner_hash_verified"] is True,
                    f"saved_frozen_inner_pairing:{identifier}",
                )
        if group == "restored":
            seed, stream = entry["base_seed"], entry["stream"]
            check(identifier == f"restored_{seed}_{stream}" and seed in PARENTS and stream in range(3), f"restored_identity:{identifier}")
            check(entry["parent_selected_low_id"] == f"old_low_{seed}", f"restored_comparison_parent:{identifier}")
            check(entry["diagnostic_only"] == (not entry["prior_selection_qualified"]), f"restored_final_qualification:{identifier}")
            artifact = entry["hardware_checkpoint"]
            if artifact:
                check(artifact["path"] == str(PILOT / f"runs/base_{seed}_stream_{stream}_critic_restore/final_model.zip"), f"registered_final_path:{identifier}")
            check(entry["selection_provenance"]["secondary_best_qualified_checkpoint_used"] is False, f"no_restored_best_substitution:{identifier}")
        check(group in {"classical", "old_low", "old_pair", "restored"}, f"policy_group:{identifier}")
    surfaces = inventory["not_attempted_surfaces"]
    check(
        any(
            surface.get("id") == "learned_macro_on_restored_final_hardware"
            and surface.get("status") == "not_attempted"
            and set(surface.get("hardware_policy_ids", []))
            == {f"restored_{seed}_{stream}" for seed in PARENTS for stream in range(3)}
            for surface in surfaces
        ),
        "restored_learned_macro_not_attempted",
    )
    return errors


def negative_self_tests(inventory):
    tests = []

    def rejected(name, change):
        altered = copy.deepcopy(inventory)
        change(altered)
        if not inventory_errors(altered):
            raise AssertionError(f"Inventory validator accepted corruption: {name}")
        tests.append(name)

    rejected("duplicate_policy_id", lambda x: x["policies"][1].update(id="nominal"))
    rejected("drop_unavailable_candidate", lambda x: x["policies"].pop(4))
    rejected("held_out_cell", lambda x: x["cells"][0].update(start_day=207))
    rejected("duplicate_cell_id", lambda x: x["cells"][1].update(id=x["cells"][0]["id"]))
    rejected("wrong_semantics", lambda x: x["policies"][0]["selection_provenance"].update(simulation_mode="legacy"))
    pair_index = next(i for i, p in enumerate(inventory["policies"]) if p["id"] == "old_pair_1001")
    restored_index = next(i for i, p in enumerate(inventory["policies"]) if p["id"] == "restored_1001_0")
    rejected(
        "incorrect_frozen_inner_hash",
        lambda x: x["policies"][pair_index]["selection_provenance"].update(
            saved_c2g_context_inner_checkpoint={"path": "invalid", "sha256": "0" * 64}
        ),
    )
    rejected(
        "restored_best_instead_of_final",
        lambda x: x["policies"][restored_index]["hardware_checkpoint"].update(
            path=str(PILOT / "runs/base_1001_stream_0_critic_restore/best_model/best_model.zip")
        ),
    )
    rejected(
        "exclude_unqualified_restored_final",
        lambda x: x["policies"][restored_index].update(available=False, unavailable_reason="excluded"),
    )
    rejected(
        "borrow_earlier_qualification",
        lambda x: x["policies"][restored_index].update(prior_selection_qualified=True),
    )
    return tests


def build_inventory():
    audit = Audit()
    audit.expect(OUTPUT == ROOT / "outputs/matched_validation_20261031", "fixed_output_directory")
    audit.fingerprint(Path(__file__), "inventory_builder")
    baseline = audit.read_json(BASELINE / "manifest.json", "baseline_manifest")
    campaign_status = audit.read_json(BASELINE / "status.json", "baseline_campaign_status")
    protocol = yaml.safe_load(audit.text(BASELINE / "protocol.yaml", "baseline_protocol"))
    registration = audit.read_json(PILOT / "registration.json", "pilot_registration")
    pilot_protocol = yaml.safe_load(
        audit.text(
            PILOT / "protocol.yaml",
            "pilot_protocol",
            registration["protocol_file"]["sha256"],
        )
    )
    readback = audit.read_json(PILOT / "reports/readback.json", "pilot_final_readback")
    pilot_status = audit.read_json(PILOT / "status.json", "pilot_campaign_status")
    audit.fingerprint(PILOT / "reports/summarize.py", "prior_final_readback_implementation")
    audit.expect(protocol == baseline["protocol"] == registration["environment_protocol"], "identical_environment_protocol")
    audit.expect(canonical_id(protocol) == baseline["protocol_id"], "baseline_protocol_identity")
    audit.expect(protocol["simulation_mode"] == "physical_v1", "protocol_physical_v1")
    audit.expect(protocol["selection"] == GATE, "registered_selection_gate")
    audit.expect(tuple(map(condition_key, protocol["validation"])) == ORIGINAL_CELLS, "original_validation_registration")
    audit.expect(protocol["held_out_days"] == [207, 248, 249], "registered_held_out_days")
    audit.expect(not set(protocol["train_days"]) & {196, 256, 207, 248, 249}, "train_validation_test_disjoint")
    audit.expect(pilot_protocol == registration["pilot_protocol"], "pilot_protocol_content")
    audit.expect(
        canonical_id({key: value for key, value in registration.items() if key != "registration_id"})
        == registration["registration_id"]
        == readback["registration_id"]
        == pilot_status["registration_id"],
        "pilot_registration_identity",
    )
    audit.expect(registration["baseline_source_id"] == baseline["source_id"], "registered_baseline_source")
    audit.expect(registration["baseline_source_snapshot"] == baseline["source_snapshot"], "registered_baseline_snapshot")
    snapshot = audit.snapshot(baseline["source_snapshot"])
    audit.expect(snapshot["python_source_id"] == baseline["source_id"], "baseline_python_source_identity")
    snapshot_members = audit.snapshots[baseline["source_snapshot"]["source_archive"]]["member_hashes"]
    for name, expected in snapshot_members.items():
        audit.fingerprint(ROOT / name, "primary_source_matches_original_snapshot", expected)
    for name, expected in registration["source_files"].items():
        audit.fingerprint(PILOT / "source" / name, "registered_pilot_source", expected)
    for name, expected in baseline["data"].items():
        audit.expect(expected["path"] == str(ROOT / name), f"baseline_data_path:{name}")
        audit.fingerprint(ROOT / name, "baseline_data", expected["sha256"])
    jobs = registration["jobs"]
    audit.expect(
        len(jobs) == len({job["id"] for job in jobs}) == registration["registered_job_count"] == 36,
        "registered_job_universe",
    )
    restore_jobs = [job for job in jobs if job["arm"] == "critic_restore"]
    audit.expect(
        {(job["base_seed"], job["stream"]) for job in restore_jobs}
        == {(seed, stream) for seed in PARENTS for stream in range(3)}
        and len(restore_jobs) == 12,
        "all_twelve_registered_restore_finals",
    )
    audit.expect(
        pilot_status["counts"] == {"pending": 0, "running": 0, "completed": 36, "failed": 0},
        "pilot_final_status",
    )
    rows = list(csv.DictReader(io.StringIO(audit.text(PILOT / "reports/final_runs.csv", "pilot_final_runs"))))
    audit.expect(len(rows) == len({row["id"] for row in rows}) == 36, "final_csv_unique_registered_rows")
    final_rows = {row["id"]: row for row in rows}
    audit.expect(set(final_rows) == {job["id"] for job in jobs}, "final_csv_registration_coverage")
    policies = []
    for identifier, (hardware, source, implementation) in CLASSICAL.items():
        entry = policy(identifier, "classical", hardware=hardware)
        entry["selection_provenance"] = {
            "selection_rule": "prespecified_classical_reference_no_learned_checkpoint_selection",
            "simulation_mode": "physical_v1",
            "hardware_implementation": implementation,
            "hardware_source": audit.fingerprint(ROOT / source, "classical_hardware_source"),
            "macro_implementation": "baselines.rule_based_macro.RuleBasedMacroController",
            "macro_source": audit.fingerprint(ROOT / "baselines/rule_based_macro.py", "rule_macro_source"),
            "hardware_factory_source": audit.fingerprint(ROOT / "evaluation/run_benchmark.py", "classical_factory_source"),
            "prior_qualification": "not_applicable",
        }
        policies.append(entry)
    lows = {}
    for seed in range(1000, 1008):
        checkpoint, qualified, available, reason, provenance = old_stage(
            audit, baseline, campaign_status, seed, "low", None
        )
        entry = policy(f"old_low_{seed}", "old_low", seed)
        entry.update(
            hardware_checkpoint=checkpoint,
            available=available,
            prior_selection_qualified=qualified,
            unavailable_reason=reason,
            selection_provenance=provenance,
        )
        lows[seed] = entry
        policies.append(entry)
    for seed in range(1000, 1008):
        inner = lows[seed]["hardware_checkpoint"]
        checkpoint, qualified, available, reason, provenance = old_stage(
            audit, baseline, campaign_status, seed, "high", inner
        )
        entry = policy(f"old_pair_{seed}", "old_pair", seed, macro="sac_macro")
        entry.update(
            hardware_checkpoint=inner,
            macro_checkpoint=checkpoint,
            available=available and lows[seed]["available"],
            prior_selection_qualified=qualified,
            unavailable_reason=reason,
            matched_rule_id=f"old_low_{seed}",
            parent_selected_low_id=f"old_low_{seed}",
            selection_provenance=provenance,
        )
        if not entry["available"] and entry["unavailable_reason"] is None:
            entry["unavailable_reason"] = "Exact originally selected frozen hardware is unavailable."
        policies.append(entry)
    parent_inputs = {}
    for seed in PARENTS:
        parent_jobs = [job for job in restore_jobs if job["base_seed"] == seed]
        inputs = parent_jobs[0]["inputs"]
        audit.expect(all(job["inputs"] == inputs for job in parent_jobs), f"matched_parent_inputs:{seed}")
        parent_inputs[seed] = registered_inputs(audit, baseline, registration, seed, inputs)
    for job in sorted(restore_jobs, key=lambda record: (record["base_seed"], record["stream"])):
        audit.expect(
            job["id"] == f"base_{job['base_seed']}_stream_{job['stream']}_critic_restore",
            f"registered_job_identity:{job['id']}",
        )
        policies.append(
            restored_policy(
                audit, registration, readback, final_rows, pilot_status, job,
                parent_inputs[job["base_seed"]],
            )
        )
    audit.check_unchanged()
    created_at = datetime.now(timezone.utc).isoformat()
    inventory = {
        "schema": SCHEMA,
        "created_at": created_at,
        "policies": policies,
        "cells": make_cells(),
        "selection_gate": {
            **GATE,
            "required_episode_ticks": 17280,
            "require_complete_truncation_without_termination": True,
            "work_conservation_absolute_error_kwh_strictly_below": 1e-4,
            "backlog_tolerance_kwh": 1e-6,
            "prior_validation_cell_count": 4,
            "matched_validation_cell_count": 8,
        },
        "held_out_days": [207, 248, 249],
        "not_attempted_surfaces": [
            {
                "id": "learned_macro_on_restored_final_hardware",
                "status": "not_attempted",
                "hardware_policy_ids": [
                    entry["id"] for entry in policies if entry["group"] == "restored"
                ],
                "macro": "sac_macro",
                "reason": "The pilot registers hardware-only continuations. No macro was "
                "trained against these final hardware hashes. Original learned macros "
                "must not be transplanted onto restored weights.",
                "candidate_slots_added": 0,
            },
            {
                "id": "held_out_test_days",
                "status": "not_attempted",
                "days": [207, 248, 249],
                "reason": "Strict validation-only inventory; no held-out evaluation or selection.",
                "candidate_slots_added": 0,
            },
        ],
        "source_receipts": {
            "simulation_mode": "physical_v1",
            "baseline": {
                "root": str(BASELINE),
                "manifest": audit.fingerprint(BASELINE / "manifest.json", "baseline_manifest"),
                "protocol": audit.fingerprint(BASELINE / "protocol.yaml", "baseline_protocol"),
                "campaign_status": audit.fingerprint(BASELINE / "status.json", "baseline_campaign_status"),
                "protocol_id": baseline["protocol_id"],
                "source_id": baseline["source_id"],
                "source_snapshot": snapshot,
                "environment_protocol": protocol,
                "primary_source_files_verified": len(snapshot_members),
                "source_identity_scopes": "source_id hashes Python files; snapshot source_id "
                "also hashes configuration YAML. These different IDs are intentional.",
            },
            "pilot": {
                "root": str(PILOT),
                "registration": audit.fingerprint(PILOT / "registration.json", "pilot_registration"),
                "registration_id": registration["registration_id"],
                "final_readback": audit.fingerprint(PILOT / "reports/readback.json", "pilot_final_readback"),
                "final_runs": audit.fingerprint(PILOT / "reports/final_runs.csv", "pilot_final_runs"),
                "registered_source_files_verified": len(registration["source_files"]),
                "registered_runs": len(jobs),
                "included_arm": "critic_restore",
                "included_registered_finals": len(restore_jobs),
                "other_arms_not_in_candidate_universe": ["sham", "optimizer_reset"],
            },
            "execution": {
                "python": sys.executable,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "cpu_cap": 40,
                "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                "native_thread_limits": {key: os.environ[key] for key in THREAD_VARIABLES},
                "model_loading_or_unpickling": False,
                "rollouts_executed": 0,
                "training_executed": False,
            },
            "inference_caveats": CAVEATS,
            "absent_paths": sorted(audit.absences),
            "files": [
                {**receipt, "roles": sorted(receipt["roles"])}
                for _, receipt in sorted(audit.files.items())
            ],
        },
    }
    for error in inventory_errors(inventory):
        audit.expect(False, f"inventory_schema:{error}")
    selected_artifacts = {
        artifact["path"]
        for entry in policies
        for artifact in (entry["hardware_checkpoint"], entry["macro_checkpoint"])
        if artifact is not None
    }
    counts = {
        "policies": len(policies),
        "available_policies": sum(entry["available"] for entry in policies),
        "unavailable_policies": sum(not entry["available"] for entry in policies),
        "validation_cells": len(inventory["cells"]),
        "original_validation_cells": sum(cell["original_validation_cell"] for cell in inventory["cells"]),
        "matched_bias_counterparts": sum(not cell["original_validation_cell"] for cell in inventory["cells"]),
        "planned_available_policy_cell_rollouts": sum(entry["available"] for entry in policies) * len(inventory["cells"]),
        "unavailable_policy_cells": sum(not entry["available"] for entry in policies) * len(inventory["cells"]),
        "diagnostic_only_available_finals": sum(entry["diagnostic_only"] and entry["available"] for entry in policies),
        "unique_selected_model_archives": len(selected_artifacts),
        "available_frozen_inner_pairings": sum(entry["group"] == "old_pair" and entry["available"] for entry in policies),
        "trained_macro_manifest_pairings_checked": audit.manifest_pairings,
        "original_validation_history_suites": audit.validation_suites["original_history"],
        "restored_final_validation_suites": audit.validation_suites["restored_final"],
        "registered_parent_model_archives": 8,
        "opaque_registered_replay_hashes": 4,
        "source_files_and_receipts_hashed": len(audit.files),
        "bytes_hashed": audit.hash_bytes,
        "rollouts_executed": 0,
    }
    expected_counts = {
        "policies": 32, "available_policies": 29, "unavailable_policies": 3,
        "validation_cells": 8, "original_validation_cells": 4,
        "matched_bias_counterparts": 4, "planned_available_policy_cell_rollouts": 232,
        "unavailable_policy_cells": 24, "diagnostic_only_available_finals": 5,
        "unique_selected_model_archives": 25, "available_frozen_inner_pairings": 6,
        "trained_macro_manifest_pairings_checked": 7,
        "original_validation_history_suites": 106, "restored_final_validation_suites": 12,
    }
    for key, expected in expected_counts.items():
        audit.expect(counts[key] == expected, f"expected_count:{key}", counts[key], expected)
    self_tests = negative_self_tests(inventory) if not inventory_errors(inventory) else []
    verification = {
        "status": "passed" if not audit.discrepancies else "discrepancies",
        "counts": counts,
        "checks": audit.checks,
        "negative_validator_tests_passed": self_tests,
        "discrepancies": audit.discrepancies,
        "consumer_instruction": "Consume this inventory without reselecting models. "
        "Proceed only if verification.status is passed. Preserve unavailable policy "
        "cells and include available diagnostic-only final checkpoints.",
    }
    inventory["source_receipts"]["verification"] = verification
    audit_report = {
        "schema": "matched-validation-policy-inventory-audit-v1",
        "created_at": created_at,
        **verification,
        "unavailable_candidates": [
            {"id": entry["id"], "reason": entry["unavailable_reason"]}
            for entry in policies if not entry["available"]
        ],
        "diagnostic_only_available_finals": [
            entry["id"] for entry in policies if entry["available"] and entry["diagnostic_only"]
        ],
        "frozen_inner_pairings": [
            {
                "id": entry["id"],
                "matched_rule_id": entry["matched_rule_id"],
                "hardware_checkpoint": entry["hardware_checkpoint"],
                "macro_checkpoint": entry["macro_checkpoint"],
                "saved_inner_checkpoint": entry["selection_provenance"]["saved_c2g_context_inner_checkpoint"],
            }
            for entry in policies if entry["group"] == "old_pair" and entry["available"]
        ],
        "caveats": CAVEATS,
        "writes": [str(INVENTORY), str(REPORT)],
        "read_only_check_command": ".venv/bin/python -B outputs/matched_validation_20261031/build_policy_inventory.py --check",
    }
    return inventory, audit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify persistent outputs without writing")
    args = parser.parse_args()
    inventory, report = build_inventory()
    if args.check:
        stored = json.loads(INVENTORY.read_text())
        previous_report = json.loads(REPORT.read_text())
        expected = {key: value for key, value in inventory.items() if key != "created_at"}
        actual = {key: value for key, value in stored.items() if key != "created_at"}
        if inventory_errors(stored) or canonical_id(actual) != canonical_id(expected):
            raise AssertionError("Persistent inventory differs from the independently rebuilt audit")
        digest = hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
        if previous_report["inventory"]["sha256"] != digest:
            raise AssertionError("Persistent inventory/report hash mismatch")
    else:
        INVENTORY.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
        if json.loads(INVENTORY.read_text()) != inventory:
            raise AssertionError("Inventory read-after-write verification failed")
        report["inventory"] = {
            "path": str(INVENTORY),
            "sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        }
        REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if json.loads(REPORT.read_text()) != report:
            raise AssertionError("Audit report read-after-write verification failed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": "read_only_check" if args.check else "written_and_verified",
                "policies": report["counts"]["policies"],
                "available": report["counts"]["available_policies"],
                "cells": report["counts"]["validation_cells"],
                "planned_rollouts": report["counts"]["planned_available_policy_cell_rollouts"],
                "unavailable_policy_cells": report["counts"]["unavailable_policy_cells"],
                "diagnostic_only_finals": report["diagnostic_only_available_finals"],
                "discrepancies": report["discrepancies"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
