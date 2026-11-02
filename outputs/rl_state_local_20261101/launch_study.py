"""One bounded paired learner per GPU; never overwrite or silently resume a campaign."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from study import (
    ROOT,
    implementation_commit,
    read_json,
    verify_registration,
    write_json,
)


def launch(protocol_path, stage, smoke_path):
    protocol = read_json(protocol_path)
    verify_registration(protocol)
    smoke = read_json(smoke_path)
    if (
        smoke["status"] != "completed"
        or not smoke["smoke_only"]
        or smoke["source_hashes"] != protocol["source_hashes"]
        or smoke["registration_id"] != protocol["registration_id"]
        or not smoke["final"]["all_qualified"]
    ):
        raise ValueError("A clean and qualifying native learner lifecycle is required")
    implementation_commit(protocol)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    available = {}
    for line in query.stdout.splitlines():
        gpu, free, utilization = [int(item.strip()) for item in line.split(",")]
        available[gpu] = (free, utilization)
    for gpu in protocol["gpus"]:
        free, utilization = available[gpu]
        if free < 8192 or utilization > 80:
            raise RuntimeError(f"GPU {gpu} is too busy; nothing launched")
    target = ROOT / f"{stage}_status.json"
    logs = ROOT / f"{stage}_logs"
    if target.exists() or logs.exists():
        raise FileExistsError(
            "A previous campaign exists; inspect it instead of relaunching"
        )
    logs.mkdir()
    processes, jobs = [], []
    try:
        for variant in protocol["variants"]:
            for seed in protocol["seeds"]:
                slot = len(jobs)
                gpu, cpus = protocol["gpus"][slot], protocol["cpu_groups"][slot]
                output = ROOT / f"runs/{stage}/{variant}_{seed}"
                command = [
                    sys.executable,
                    str(ROOT / "low_learn.py"),
                    "--protocol",
                    str(protocol_path),
                    "--stage",
                    stage,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                    "--cpus",
                    *map(str, cpus),
                ]
                environment = os.environ.copy()
                environment.update(
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                with (logs / f"{variant}_{seed}.log").open("w") as handle:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                processes.append(process)
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "gpu": gpu,
                        "cpus": cpus,
                        "pid": process.pid,
                        "output": str(output),
                        "status": "running",
                    }
                )
        while True:
            for process, job in zip(processes, jobs, strict=True):
                code = process.poll()
                if code is not None:
                    job.update(
                        status="completed" if code == 0 else "failed", exit_code=code
                    )
            running = any(process.poll() is None for process in processes)
            write_json(
                target,
                {
                    "registration_id": protocol["registration_id"],
                    "stage": stage,
                    "state": "running" if running else "finished",
                    "updated_unix": time.time(),
                    "jobs": jobs,
                },
            )
            if not running:
                break
            time.sleep(10)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(
                "A learner failed; all native receipts and failures are retained"
            )
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    verify_registration(protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "registration.json")
    parser.add_argument("--stage", choices=("pilot", "extended"), required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    args = parser.parse_args()
    launch(args.protocol, args.stage, args.smoke_receipt)
