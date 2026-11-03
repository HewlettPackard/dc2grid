"""Run the closed 35-case Table 1 panel without training or outcome-based retries."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
):
    os.environ[_name] = "1"

from prepare_table1 import ROOT, verify  # noqa: E402
from table1_eval import run_case  # noqa: E402


def initialize(queue):
    os.sched_setaffinity(0, {queue.get()})


def evaluate(job):
    protocol, policy, seed, destination = job
    result = run_case(protocol, policy, seed, Path(destination))
    return {
        "policy": policy["id"],
        "seed": seed,
        "result_path": str(Path(destination) / "episode.json"),
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-case-only", action="store_true")
    parser.add_argument("--remaining", action="store_true")
    args = parser.parse_args()
    if args.first_case_only and args.remaining:
        raise ValueError("Choose only one explicit execution stage")
    protocol = verify(require_committed=True)
    jobs = []
    retained = []
    for policy in protocol["policies"]:
        for seed in protocol["panel"]["seeds"]:
            destination = (
                ROOT / "cases" / policy["id"].replace("+", "__") / f"seed_{seed}"
            )
            if args.first_case_only and (
                policy != protocol["policies"][0] or seed != 100
            ):
                continue
            if destination.exists():
                completed = destination / "episode.json"
                if not args.remaining or not completed.is_file():
                    raise FileExistsError(
                        "Existing attempt must be preserved and inspected: "
                        f"{destination}"
                    )
                retained.append(
                    {"policy": policy["id"], "seed": seed, "path": str(completed)}
                )
                continue
            jobs.append((protocol, policy, seed, str(destination)))
    if args.remaining and len(retained) != 1:
        raise ValueError(
            "The staged continuation expects exactly one retained complete "
            "first-case record"
        )
    label = (
        "first_case"
        if args.first_case_only
        else "remaining"
        if args.remaining
        else "full"
    )
    status_path = ROOT / f"execution_{label}.json"
    if status_path.exists():
        raise FileExistsError(
            "The requested execution stage already has a retained record"
        )
    results = []
    errors = []
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["execution"]["worker_cpus"]:
        queue.put(cpu)
    pool = ProcessPoolExecutor(
        max_workers=min(len(jobs), protocol["execution"]["workers"]),
        mp_context=context,
        initializer=initialize,
        initargs=(queue,),
    )

    def save():
        status_path.write_text(
            json.dumps(
                {
                    "registration_id": protocol["registration_id"],
                    "stage": label,
                    "planned_this_stage": len(jobs),
                    "retained": retained,
                    "completed": results,
                    "execution_errors": errors,
                    "outcome_based_retries": False,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    save()
    try:
        futures = {pool.submit(evaluate, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            error = future.exception()
            if error is not None:
                errors.append(
                    {
                        "policy": job[1]["id"],
                        "seed": job[2],
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "attempt_directory": job[3],
                    }
                )
                save()
                for pending in futures:
                    pending.cancel()
                raise error
            outcome = future.result()
            results.append(
                {key: value for key, value in outcome.items() if key != "result"}
            )
            save()
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "planned": len(jobs),
                        "policy": outcome["policy"],
                        "seed": outcome["seed"],
                    }
                ),
                flush=True,
            )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        queue.close()
    verify(require_committed=True)
    print(
        json.dumps(
            {
                "registration_id": protocol["registration_id"],
                "stage": label,
                "completed": len(results),
                "execution_errors": len(errors),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
