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
    file_hash,
    identity,
    initialize_worker,
    read_json,
    verify,
    write_json,
)
from confirmation_eval import evaluate_case


def execute(protocol, panel, policy, cell, destination):
    record = evaluate_case(protocol, panel, policy, cell, destination)
    path = Path(destination) / "episode.json"
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "panel": panel,
        "policy_id": policy["id"],
        "cell": cell,
        "complete": record["outcome"]["complete"],
        "qualified": record["outcome"]["qualified"],
    }


def run_panels(protocol, panels, commit):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
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
            output.mkdir(parents=True, exist_ok=False)
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
            }
            write_json(output / "manifest.json", manifest)
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
            completed, qualified, failures = 0, 0, []
            receipt_path = output / "completion.jsonl"
            with receipt_path.open("x") as receipt:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        receipt.write(json.dumps(result, allow_nan=False) + "\n")
                        completed += 1
                        qualified += int(result["qualified"])
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
                            output / "status.json",
                            {
                                "completed": completed,
                                "qualified": qualified,
                                "expected": expected,
                                "runtime_failures": failures,
                                "finished": False,
                            },
                        )
                        print(
                            f"{panel}: {completed}/{expected}; "
                            f"qualified={qualified}; runtime_failures={len(failures)}",
                            flush=True,
                        )
            write_json(
                output / "status.json",
                {
                    "completed": completed,
                    "qualified": qualified,
                    "expected": expected,
                    "runtime_failures": failures,
                    "finished": True,
                    "completion_log_sha256": file_hash(receipt_path),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if failures or completed != expected:
                raise RuntimeError(
                    f"{panel} is incomplete; "
                    "all failures and existing outcomes retained",
                )
    queue.close()
    queue.join_thread()


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
        if opening_path.exists():
            raise FileExistsError("Never reopen or overwrite the confirmation ledger")
        opening = {
            "registration_id": protocol["registration_id"],
            "implementation_commit": commit,
            "opened_utc": datetime.now(timezone.utc).isoformat(),
            "panels": ["calendar", "peak_noise"],
            "development_report_sha256": file_hash(report_path),
            "old_failed_reservation_not_reopened": True,
            "no_new_training_or_checkpoint_selection": True,
        }
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
