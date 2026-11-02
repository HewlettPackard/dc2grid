"""Run eight bounded union-prior fits with one GPU and five physical cores each."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ctx import ROOT, committed, read_json, verify, write_json


def launch(protocol_path, stage, smoke_receipts):
    protocol = read_json(protocol_path)
    verify(protocol)
    committed(protocol)
    covered = set()
    for path in smoke_receipts:
        smoke = read_json(path)
        if (
            smoke["status"] != "completed"
            or not smoke["smoke_only"]
            or smoke["registration_id"] != protocol["registration_id"]
            or smoke["source_hashes"] != protocol["source_hashes"]
            or not smoke["final"]["all_qualified"]
        ):
            raise ValueError("A clean native full-lifecycle smoke is required")
        covered.add(smoke["variant"])
    if covered != set(protocol["variants"]):
        raise ValueError("Both reward paths need native lifecycle verification")
    if stage == "extended":
        decision = read_json(ROOT / "extension_decision.json")
        if not decision["extend_both_arms"]:
            raise ValueError("No registered extension was earned")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    resources = {}
    for line in result.stdout.splitlines():
        gpu, free, used = map(int, line.replace(" ", "").split(","))
        resources[gpu] = free, used
    for gpu in protocol["gpus"]:
        if resources[gpu][0] < 8192 or resources[gpu][1] > 80:
            raise RuntimeError(f"GPU {gpu} is too busy; no fits started")
    status_path = ROOT / f"{stage}_status.json"
    logs = ROOT / f"{stage}_logs"
    if status_path.exists() or logs.exists():
        raise FileExistsError("Never overwrite or invisibly retry a campaign")
    logs.mkdir()
    jobs, processes = [], []
    try:
        for variant in protocol["variants"]:
            for seed in protocol["seeds"]:
                index = len(jobs)
                gpu, cpus = protocol["gpus"][index], protocol["cpu_groups"][index]
                command = [
                    sys.executable,
                    str(ROOT / "train_value.py"),
                    "--protocol",
                    str(protocol_path),
                    "--stage",
                    stage,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
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
                        "pid": process.pid,
                        "gpu": gpu,
                        "cpus": cpus,
                        "status": "running",
                        "output": str(ROOT / f"runs/{stage}/{variant}_{seed}"),
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
                status_path,
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
            raise RuntimeError("A union learner failed; native evidence is retained")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    verify(protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "registration.json")
    parser.add_argument("--stage", choices=("pilot", "extended"), required=True)
    parser.add_argument("--smoke-receipts", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    launch(args.protocol, args.stage, args.smoke_receipts)
