"""Bounded one-learner-per-GPU supervisor; no retries or hidden missing runs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from register import verify_training_gate
from study_common import ROOT, THREAD_KEYS, load_json, write_json


def gpu_status() -> dict:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    result = {}
    for line in process.stdout.splitlines():
        index, free, utilization = [int(value.strip()) for value in line.split(",")]
        result[index] = {"free_mib": free, "utilization_percent": utilization}
    return result


def main():
    protocol = load_json(ROOT / "registration.json")
    verify_training_gate(protocol)
    target = ROOT / "training_status.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite or relaunch a prior campaign")
    os.sched_setaffinity(0, {protocol["cpu_groups"][0][0]})
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    jobs = []
    for variant in protocol["variants"]:
        for seed in protocol["macro_seeds"]:
            slot = len(jobs)
            jobs.append(
                {
                    "id": f"{variant}_seed_{seed}",
                    "variant": variant,
                    "seed": seed,
                    "gpu": protocol["gpus"][slot],
                    "cpus": protocol["cpu_groups"][slot],
                    "output": str(ROOT / "runs" / f"{variant}_seed_{seed}"),
                    "state": "pending",
                    "started_unix": None,
                }
            )
    running = {}
    begun = time.monotonic()

    def publish(state):
        write_json(
            target,
            {
                "state": state,
                "pid": os.getpid(),
                "updated_unix": time.time(),
                "registration_id": protocol["registration_id"],
                "jobs": jobs,
                "planned_runs": len(jobs),
                "completed_runs": sum(job["state"] == "completed" for job in jobs),
                "runtime_failed_runs": sum(
                    job["state"] == "runtime_failure" for job in jobs
                ),
            },
        )

    publish("running")
    try:
        while any(job["state"] in ("pending", "running") for job in jobs):
            available = gpu_status()
            for job in jobs:
                if job["state"] != "pending":
                    continue
                hardware = available[job["gpu"]]
                if hardware["free_mib"] < 8192 or hardware["utilization_percent"] > 80:
                    if time.monotonic() - begun > 900:
                        job.update(
                            state="runtime_failure",
                            reason="GPU availability gate timed out",
                        )
                    continue
                environment = dict(os.environ)
                environment.update({key: "1" for key in THREAD_KEYS})
                environment.update(
                    CUDA_VISIBLE_DEVICES=str(job["gpu"]),
                    CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    MPLCONFIGDIR=str(ROOT / "cache/matplotlib" / job["id"]),
                )
                handle = (ROOT / "logs" / f"{job['id']}.log").open("x")
                command = [
                    sys.executable,
                    str(ROOT / "learn.py"),
                    "--protocol",
                    str(ROOT / "registration.json"),
                    "--variant",
                    job["variant"],
                    "--seed",
                    str(job["seed"]),
                    "--output",
                    job["output"],
                    "--device",
                    "cuda:0",
                    "--cpus",
                    *map(str, job["cpus"]),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[job["id"]] = (process, handle)
                job.update(
                    state="running",
                    pid=process.pid,
                    started_unix=time.time(),
                    gpu_readiness=hardware,
                )
                print(
                    "Started",
                    job["id"],
                    "GPU",
                    job["gpu"],
                    "PID",
                    process.pid,
                    flush=True,
                )
            for job in jobs:
                if job["state"] != "running":
                    continue
                process, handle = running[job["id"]]
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                from pathlib import Path

                status_path = Path(job["output"]) / "status.json"
                status = load_json(status_path) if status_path.exists() else None
                complete = (
                    code == 0 and status is not None and status["state"] == "completed"
                )
                job.update(
                    state="completed" if complete else "runtime_failure",
                    returncode=code,
                    qualified_checkpoint=bool(
                        status and status.get("qualified_checkpoint")
                    ),
                    final_macro_transitions=status["macro_transitions"]
                    if status
                    else None,
                )
                print("Finished", job["id"], job["state"], flush=True)
            publish("running")
            if any(job["state"] in ("pending", "running") for job in jobs):
                time.sleep(5)
        publish(
            "completed_with_runtime_failures"
            if any(job["state"] == "runtime_failure" for job in jobs)
            else "completed"
        )
    finally:
        for process, handle in running.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=60)
            if not handle.closed:
                handle.close()


if __name__ == "__main__":
    main()
