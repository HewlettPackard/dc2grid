"""Launch at most one standalone SAC learner per GPU on distinct CPU groups."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, read_json, write_json
from prepare_study import verify_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "registration.json")
    parser.add_argument(
        "--smoke-receipt",
        type=Path,
        default=ROOT / "training_smoke_v2/completed.json",
    )
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    verify_protocol(protocol)
    smoke = read_json(args.smoke_receipt)
    if (
        smoke["status"] != "completed"
        or not smoke["smoke_only"]
        or smoke["code_hashes"] != protocol["code_hashes"]
    ):
        raise ValueError("A clean native full-lifecycle smoke is required")
    target = ROOT / "training_status.json"
    if target.exists():
        raise FileExistsError("Refusing to relaunch or overwrite a prior campaign")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    available = {}
    for line in query.stdout.splitlines():
        gpu, free, used = [int(item.strip()) for item in line.split(",")]
        available[gpu] = {"free_mib": free, "utilization_percent": used}
    for gpu in protocol["gpus"]:
        if (
            available[gpu]["free_mib"] < 8192
            or available[gpu]["utilization_percent"] > 80
        ):
            raise RuntimeError(f"GPU {gpu} is too busy; no learners launched")
    jobs, processes = [], []
    logs = ROOT / "training_logs"
    logs.mkdir(exist_ok=False)
    try:
        for variant in protocol["variants"]:
            for seed in protocol["seeds"]:
                slot = len(jobs)
                gpu, cpus = protocol["gpus"][slot], protocol["cpu_groups"][slot]
                environment = os.environ.copy()
                environment.update(
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                output = ROOT / "runs" / f"{variant}_{seed}"
                log = (logs / f"{variant}_{seed}.log").open("w")
                command = [
                    sys.executable,
                    str(ROOT / "train.py"),
                    "--protocol",
                    str(args.protocol),
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                    "--cpus",
                    *map(str, cpus),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                log.close()
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
                        "command": command,
                    }
                )
        while True:
            for job, process in zip(jobs, processes, strict=True):
                code = process.poll()
                if code is not None:
                    job.update(
                        status="completed" if code == 0 else "failed", exit_code=code
                    )
            write_json(
                target,
                {
                    "registration_id": protocol["registration_id"],
                    "updated_unix": time.time(),
                    "jobs": jobs,
                    "state": "running"
                    if any(p.poll() is None for p in processes)
                    else "finished",
                },
            )
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(10)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("One or more SAC fits failed; native receipts retained")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise


if __name__ == "__main__":
    main()
