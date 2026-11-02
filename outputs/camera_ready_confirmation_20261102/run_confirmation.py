"""Execute every locked policy/cell slot and retain all runtime failures."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from confirmation_context import (
    ROOT,
    cap_cpu,
    case_key,
    effective_source_hashes,
    file_hash,
    identity,
    initialize_worker,
    read_json,
    technical_amendment_id,
    verify,
    write_json,
)
from confirmation_eval import evaluate_case


def completed_case(protocol, panel, policy, cell, destination):
    directory = Path(destination)
    paths = (
        [directory / "episode.json"] if (directory / "episode.json").exists() else []
    )
    paths += sorted(directory.glob("attempts/attempt_*/episode.json"))
    if len(paths) > 1:
        raise ValueError(
            "Multiple completed outcomes for one slot; never select a winner"
        )
    if not paths:
        return None
    path = paths[0]
    record = read_json(path)
    if (
        record["evaluation_registration_id"] != protocol["registration_id"]
        or record["panel"] != panel
        or record["policy"] != policy
        or record["cell"] != cell
        or "outcome" not in record
    ):
        raise ValueError("Existing episode does not match the immutable planned slot")
    for name, digest in record["trace_hashes"].items():
        if name not in {"hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz"}:
            raise ValueError("Unexpected cached trace path")
        if file_hash(path.parent / name) != digest:
            raise ValueError("A completed episode's immutable raw trace changed")
    return path, record


def next_attempt(destination):
    directory = Path(destination)
    if not directory.exists():
        return directory
    for index in range(1, 1_000_000):
        candidate = directory / "attempts" / f"attempt_{index:04d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Excessive failed attempts; manual investigation required")


def next_completion_log(output):
    if not (output / "completion.jsonl").exists():
        return output / "completion.jsonl"
    for index in range(1, 1_000_000):
        candidate = output / f"completion_resume_{index:04d}.jsonl"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Excessive execution attempts")


def execute(protocol, panel, policy, cell, destination):
    cached = completed_case(protocol, panel, policy, cell, destination)
    if cached is not None:
        path, record = cached
    else:
        target = next_attempt(destination)
        record = evaluate_case(protocol, panel, policy, cell, target)
        path = target / "episode.json"
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "panel": panel,
        "policy_id": policy["id"],
        "cell": cell,
        "complete": record["outcome"]["complete"],
        "qualified": record["outcome"]["qualified"],
        "reused_completed_slot": cached is not None,
        "technical_amendment_id": technical_amendment_id(),
    }


def run_panels(protocol, panels, commit):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
    failed_panels = []
    with ProcessPoolExecutor(
        max_workers=len(protocol["cpus"]),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as pool:
        for panel in panels:
            info = protocol["panels"][panel]
            policies = [
                row
                for row in protocol["policies"]
                if info["new_roster"] == "all" or row["kind"] == "scripted"
            ]
            output = ROOT / "evaluation" / panel
            output.mkdir(parents=True, exist_ok=True)
            expected = len(policies) * len(info["cells"])
            manifest = {
                "evaluation_registration_id": protocol["registration_id"],
                "implementation_commit": commit,
                "panel": panel,
                "scope": info["scope"],
                "policies": policies,
                "cells": info["cells"],
                "expected_episodes": expected,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "technical_amendment_id": technical_amendment_id(),
                "source_hashes": effective_source_hashes(protocol),
            }
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                previous = read_json(manifest_path)
                for key in (
                    "evaluation_registration_id",
                    "panel",
                    "scope",
                    "policies",
                    "cells",
                    "expected_episodes",
                ):
                    if previous[key] != manifest[key]:
                        raise ValueError(
                            "Resume manifest differs from the frozen panel"
                        )
            else:
                write_json(manifest_path, manifest)
            receipt_path = next_completion_log(output)
            status_path = output / "status.json"
            if status_path.exists():
                prior_status = read_json(status_path)
                previous_log = output / prior_status.get(
                    "completion_log", "completion.jsonl"
                )
                if prior_status["finished"] and (
                    file_hash(previous_log) != prior_status["completion_log_sha256"]
                ):
                    raise ValueError("A prior completed receipt log changed")
                write_json(
                    output / f"status_before_{receipt_path.stem}.json",
                    prior_status,
                )
            futures = {}
            for policy in policies:
                for cell in info["cells"]:
                    destination = output / policy["id"] / case_key(cell)
                    future = pool.submit(
                        execute,
                        protocol,
                        panel,
                        policy,
                        cell,
                        str(destination),
                    )
                    futures[future] = str(destination)
            completed, qualified, reused, failures = 0, 0, 0, []
            with receipt_path.open("x") as receipt:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        receipt.write(json.dumps(result, allow_nan=False) + "\n")
                        completed += 1
                        qualified += int(result["qualified"])
                        reused += int(result["reused_completed_slot"])
                    except Exception as error:
                        failure = {
                            "directory": futures[future],
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        print(json.dumps({"runtime_failure": failure}), flush=True)
                    if (completed + len(failures)) % 25 == 0:
                        receipt.flush()
                        write_json(
                            status_path,
                            {
                                "completed": completed,
                                "qualified": qualified,
                                "expected": expected,
                                "runtime_failures": failures,
                                "finished": False,
                                "completion_log": receipt_path.name,
                                "reused_completed_slots": reused,
                                "technical_amendment_id": technical_amendment_id(),
                            },
                        )
                        print(
                            f"{panel}: {completed}/{expected}; "
                            f"qualified={qualified}; runtime_failures={len(failures)}",
                            flush=True,
                        )
            write_json(
                status_path,
                {
                    "completed": completed,
                    "qualified": qualified,
                    "expected": expected,
                    "runtime_failures": failures,
                    "finished": True,
                    "completion_log_sha256": file_hash(receipt_path),
                    "completion_log": receipt_path.name,
                    "reused_completed_slots": reused,
                    "technical_amendment_id": technical_amendment_id(),
                    "execution_commit": commit,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if failures or completed != expected:
                failed_panels.append(panel)
    queue.close()
    queue.join_thread()
    if failed_panels:
        raise RuntimeError(
            f"Incomplete panels: {failed_panels}; all attempts/outcomes retained",
        )


def run(stage):
    cap_cpu(32)
    protocol = read_json(ROOT / "registration.json")
    commit = verify(protocol)
    if stage == "development":
        if (ROOT / "final_opening.json").exists():
            raise RuntimeError("Known-panel controls must precede confirmation")
        panels = ["development", "regression"]
    elif stage == "final":
        report_path = ROOT / "reports/development_report.json"
        report = read_json(report_path)
        if (
            report["registration_id"] != protocol["registration_id"]
            or not report["technical_validation_passed"]
        ):
            raise RuntimeError("Known-panel technical validation must pass first")
        opening_path = ROOT / "final_opening.json"
        opening = {
            "registration_id": protocol["registration_id"],
            "implementation_commit": commit,
            "opened_utc": datetime.now(timezone.utc).isoformat(),
            "panels": ["calendar", "peak_noise"],
            "development_report_sha256": file_hash(report_path),
            "old_failed_reservation_not_reopened": True,
            "no_new_training_or_checkpoint_selection": True,
            "technical_amendment_id": technical_amendment_id(),
            "source_hashes": effective_source_hashes(protocol),
        }
        if opening_path.exists():
            prior = read_json(opening_path)
            body = {key: value for key, value in prior.items() if key != "opening_id"}
            if identity(body) != prior["opening_id"]:
                raise ValueError("The immutable opening identity changed")
            for key in (
                "registration_id",
                "panels",
                "development_report_sha256",
                "technical_amendment_id",
                "source_hashes",
            ):
                if prior[key] != opening[key]:
                    raise ValueError("Cannot resume a different confirmation opening")
        else:
            opening["opening_id"] = identity(opening)
            write_json(opening_path, opening)
        panels = ["calendar", "peak_noise"]
    else:
        raise ValueError("Unknown execution stage")
    run_panels(protocol, panels, commit)
    verify(protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("development", "final"))
    run(parser.parse_args().stage)
