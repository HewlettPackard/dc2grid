"""Supervise one bounded SAC learner per GPU, hardware first and macro second."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from baselines.train_corrected_sac import (
    ROOT,
    THREAD_VARIABLES,
    validate_protocol,
    write_json,
)
from c2g_env import C2GFastEnv
from c2g_env.experiments.identity import (
    capture_source,
    configuration_id,
    file_fingerprint,
    source_id,
)


def verify_resources(protocol: dict) -> None:
    count = len(protocol["seeds"])
    if not count or len(set(protocol["seeds"])) != count:
        raise ValueError("Seeds must be nonempty and unique")
    if len(protocol["gpus"]) != count or len(protocol["cpu_groups"]) != count:
        raise ValueError("Assign exactly one GPU and CPU group per seed")
    if len(set(protocol["gpus"])) != count:
        raise ValueError("Each learner must have a different GPU")
    cpus = [cpu for group in protocol["cpu_groups"] for cpu in group]
    if len(cpus) != len(set(cpus)):
        raise ValueError("CPU groups must not overlap")
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError(
            "Requested CPUs are not available in the supervisor's affinity"
        )
    if any(len(group) < protocol["n_envs"] + 1 for group in protocol["cpu_groups"]):
        raise ValueError(
            "Each group needs one learner CPU plus one per training environment"
        )
    if protocol["threads"] != 1:
        raise ValueError("This launcher reserves one native/PyTorch thread per process")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    for gpu in protocol["gpus"]:
        free, _ = torch.cuda.mem_get_info(gpu)
        if free < 4 * 2**30:
            raise RuntimeError(
                f"GPU {gpu} has less than 4 GiB free; refusing oversubscription"
            )


def preflight(protocol: dict) -> dict:
    """Validate structural data coverage without running a controller or test day."""
    days = sorted(
        set(protocol["train_days"]) | {c["start_day"] for c in protocol["validation"]}
    )
    for day in days:
        env = C2GFastEnv(scenario=protocol["scenario"], start_day=day)
        try:
            observation, _ = env.reset(seed=0)
            if not np.isfinite(observation).all():
                raise ValueError(f"Nonfinite initial observation on day {day}")
        finally:
            env.close()
    files = [
        ROOT / "data/processed/energy/NYC.csv",
        ROOT / "data/processed/weather/NYC.csv",
        *(ROOT / "data/processed/workload_traces").glob("*.csv"),
    ]
    return {
        "validated_train_validation_days": days,
        "held_out_days_not_executed": protocol["held_out_days"],
        "data": {
            path.relative_to(ROOT).as_posix(): file_fingerprint(path) for path in files
        },
    }


def launch(
    protocol: dict,
    output: Path,
    *,
    only_low: bool = False,
) -> None:
    validate_protocol(protocol)
    verify_resources(protocol)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite campaign: {output}")
    output.mkdir(parents=True)
    protocol_file = output / "protocol.yaml"
    protocol_file.write_text(yaml.safe_dump(protocol, sort_keys=False))
    campaign_source = source_id(ROOT)
    manifest = {
        "protocol_id": configuration_id(protocol),
        "protocol": protocol,
        "source_id": campaign_source,
        "source_snapshot": capture_source(ROOT, output / "source"),
        **preflight(protocol),
        "only_low": only_low,
        "selection_uses_held_out_days": False,
        "supervisor_pid": os.getpid(),
    }
    write_json(output / "manifest.json", manifest)
    jobs = []
    for seed, gpu, cpus in zip(
        protocol["seeds"], protocol["gpus"], protocol["cpu_groups"]
    ):
        directory = output / f"seed_{seed}"
        directory.mkdir()
        jobs.append(
            {
                "seed": seed,
                "gpu": gpu,
                "cpus": cpus,
                "directory": directory,
                "stage": "low",
                "state": "pending",
            }
        )

    def start(job: dict) -> None:
        if source_id(ROOT) != campaign_source:
            raise RuntimeError(
                "Source changed during campaign; "
                "refusing to mix revisions between stages"
            )
        environment = dict(os.environ)
        environment.update({key: str(protocol["threads"]) for key in THREAD_VARIABLES})
        environment.update(
            CUDA_VISIBLE_DEVICES=str(job["gpu"]),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            UV_NO_SYNC="1",
            MPLCONFIGDIR=str(job["directory"] / "matplotlib"),
        )
        command = [
            sys.executable,
            "-m",
            "baselines.train_corrected_sac",
            "--protocol",
            str(protocol_file),
            "--stage",
            job["stage"],
            "--seed",
            str(job["seed"]),
            "--output",
            str(job["directory"] / job["stage"]),
            "--device",
            "cuda:0",
            "--cpus",
            *map(str, job["cpus"]),
        ]
        if job["stage"] == "high":
            command.extend(
                [
                    "--inner-checkpoint",
                    str(job["directory"] / "low/best_model/best_model.zip"),
                ]
            )
        log_path = job["directory"] / f"{job['stage']}.log"
        with log_path.open("x") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        job.update(process=process, state="running", pid=process.pid, command=command)
        print(
            f"Started seed={job['seed']} stage={job['stage']} "
            f"GPU={job['gpu']} PID={process.pid}",
            flush=True,
        )

    def snapshot(state: str) -> None:
        write_json(
            output / "status.json",
            {
                "state": state,
                "supervisor_pid": os.getpid(),
                "updated_unix": time.time(),
                "jobs": [
                    {
                        key: value
                        for key, value in job.items()
                        if key not in ("process", "directory")
                    }
                    for job in jobs
                ],
            },
        )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    finished = False
    try:
        for job in jobs:
            start(job)
        snapshot("running")
        while any(job["state"] == "running" for job in jobs):
            time.sleep(10)
            for job in jobs:
                if job["state"] != "running":
                    continue
                code = job["process"].poll()
                if code is None:
                    continue
                job["returncode"] = code
                if code != 0:
                    job["state"] = "failed"
                elif job["stage"] == "low" and not only_low:
                    best = job["directory"] / "low/best_model/best_model.zip"
                    selection = job["directory"] / "low/best_model/selection.json"
                    if not best.exists() or not selection.exists():
                        job.update(
                            state="blocked",
                            reason="No fully qualified hardware validation checkpoint",
                        )
                    else:
                        selected = json.loads(selection.read_text())
                        if not selected["all_cells_qualified"]:
                            raise RuntimeError(
                                "Unqualified checkpoint reached the macro stage"
                            )
                        if (
                            selected["checkpoint"]["sha256"]
                            != file_fingerprint(best)["sha256"]
                        ):
                            raise RuntimeError(
                                "Selected hardware checkpoint hash changed"
                            )
                        job["stage"] = "high"
                        start(job)
                else:
                    stage_status = json.loads(
                        (job["directory"] / job["stage"] / "status.json").read_text()
                    )
                    if not stage_status["qualified_checkpoint"] and not protocol.get(
                        "pilot_only"
                    ):
                        job.update(
                            state="blocked",
                            reason=(
                                f"No fully qualified {job['stage']} "
                                "validation checkpoint"
                            ),
                        )
                    else:
                        job["state"] = "completed"
            snapshot("running")
        successful = all(job["state"] == "completed" for job in jobs)
        snapshot("completed" if successful else "incomplete")
        finished = True
        if not successful:
            raise RuntimeError(
                "Some seeds failed or lacked a qualified checkpoint; "
                "inspect campaign status"
            )
    finally:
        for job in jobs:
            process = job.get("process")
            if process is not None and process.poll() is None:
                process.terminate()
        for job in jobs:
            process = job.get("process")
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if not finished:
            for job in jobs:
                if job["state"] == "running":
                    job["state"] = "interrupted"
            snapshot("failed_or_interrupted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "conf/retraining/corrected_sac.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-low", action="store_true")
    parser.add_argument(
        "--pilot-steps",
        type=int,
        help="Bounded first-seed pilot; forces only-low and validation at the end.",
    )
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if args.pilot_steps is not None:
        if args.pilot_steps < 2 * protocol["n_envs"]:
            parser.error("--pilot-steps must allow both rollout and optimizer updates")
        protocol["seeds"] = protocol["seeds"][:1]
        protocol["gpus"] = protocol["gpus"][:1]
        protocol["cpu_groups"] = protocol["cpu_groups"][:1]
        protocol["low"].update(
            timesteps=args.pilot_steps,
            learning_starts=args.pilot_steps // 8,
            eval_freq=args.pilot_steps + 1,
            checkpoint_freq=args.pilot_steps // 2,
            status_freq=max(protocol["n_envs"], args.pilot_steps // 8),
        )
        protocol["pilot_only"] = True
    launch(
        protocol,
        args.output.resolve(),
        only_low=args.only_low or args.pilot_steps is not None,
    )


if __name__ == "__main__":
    main()
