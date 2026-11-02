"""Run all declared PID references and audit them against frozen matched inputs."""

from __future__ import annotations

import json
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from audit_pid import audit_slot, frozen_inputs, summarize
from pid_context import (
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
from pid_reference import execute
from run_confirmation import next_completion_log


def main():
    cap_cpu(32)
    protocol = read_json(ROOT / "registration.json")
    commit = verify(protocol)
    opening_path = ROOT / "reference_opening.json"
    if opening_path.exists():
        opening = read_json(opening_path)
        if (
            opening["registration_id"] != protocol["registration_id"]
            or identity({k: v for k, v in opening.items() if k != "opening_id"})
            != opening["opening_id"]
        ):
            raise ValueError("PID reference opening changed")
    else:
        opening = {
            "registration_id": protocol["registration_id"],
            "implementation_commit": commit,
            "opened_utc": datetime.now(timezone.utc).isoformat(),
            "role": protocol["role"],
        }
        opening["opening_id"] = identity(opening)
        write_json(opening_path, opening)
    output = ROOT / "evaluation"
    output.mkdir(exist_ok=True)
    receipt_path = next_completion_log(output)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
    completed, failures = [], []
    with (
        ProcessPoolExecutor(
            max_workers=len(protocol["cpus"]),
            mp_context=context,
            initializer=initialize_worker,
            initargs=(queue,),
        ) as pool,
        receipt_path.open("x") as stream,
    ):
        futures = {}
        for panel, info in protocol["panels"].items():
            for policy in protocol["policies"]:
                for cell in info["cells"]:
                    destination = output / panel / policy["id"] / case_key(cell)
                    future = pool.submit(
                        execute,
                        protocol,
                        panel,
                        policy,
                        cell,
                        str(destination),
                    )
                    futures[future] = str(destination)
        for future in as_completed(futures):
            try:
                result = future.result()
                completed.append(result)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            except Exception as error:
                failure = {
                    "directory": futures[future],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print("RUNTIME FAILURE", failure, flush=True)
            if (len(completed) + len(failures)) % 20 == 0:
                stream.flush()
                print(
                    f"PID references {len(completed)}/{len(futures)}; "
                    f"qualified={sum(row['qualified'] for row in completed)}",
                    flush=True,
                )
    queue.close()
    queue.join_thread()
    write_json(
        output / "status.json",
        {
            "expected": protocol["expected_new_episodes"],
            "completed": len(completed),
            "runtime_failures": failures,
            "completion_log": receipt_path.name,
            "completion_log_sha256": file_hash(receipt_path),
        },
    )
    if failures or len(completed) != protocol["expected_new_episodes"]:
        raise RuntimeError("PID comparison incomplete; all attempts retained")
    expected = {
        (panel, policy["id"], case_key(cell))
        for panel, info in protocol["panels"].items()
        for policy in protocol["policies"]
        for cell in info["cells"]
    }
    keys = {
        (row["panel"], row["policy_id"], case_key(row["cell"])) for row in completed
    }
    if keys != expected or len(keys) != len(completed):
        raise ValueError("PID reference inventory is not one-to-one")
    references = frozen_inputs()
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
    with ProcessPoolExecutor(
        max_workers=len(protocol["cpus"]),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as pool:
        futures = [
            pool.submit(
                audit_slot,
                protocol,
                row,
                references[(row["panel"], case_key(row["cell"]))],
            )
            for row in completed
        ]
        audited = [future.result() for future in as_completed(futures)]
    queue.close()
    queue.join_thread()
    audited.sort(
        key=lambda row: (row["panel"], row["policy_id"], case_key(row["cell"]))
    )
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    if (reports / "pid_report.json").exists():
        raise FileExistsError("Never overwrite the audited PID report")
    write_json(reports / "pid_cases.json", audited)
    report = {
        "registration_id": protocol["registration_id"],
        "role": protocol["role"],
        "episodes": len(audited),
        "complete": sum(row["complete"] for row in audited),
        "qualified": sum(row["qualified"] for row in audited),
        "raw_checks": sum(row["raw_checks"] for row in audited),
        "case_records_sha256": file_hash(reports / "pid_cases.json"),
        "inputs_match_frozen_confirmation": True,
        "summary": summarize(audited, protocol),
        "previous_primary_test_and_models_unchanged": True,
    }
    write_json(reports / "pid_report.json", report)
    verify(protocol)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("episodes", "complete", "qualified", "raw_checks")
            }
        )
    )


if __name__ == "__main__":
    main()
