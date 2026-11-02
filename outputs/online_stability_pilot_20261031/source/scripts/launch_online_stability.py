"""Register and schedule isolated online SAC trials after baseline slots release."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import yaml
from baselines.online_stability import ROOT, verify_registration

from baselines.train_corrected_sac import THREAD_VARIABLES, write_json
from c2g_env.experiments.identity import configuration_id, file_fingerprint

TERMINAL_STATES = {"completed", "blocked", "failed", "interrupted", "cancelled"}


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


def slot_released(baseline_job, alive=process_alive):
    return baseline_job["state"] in TERMINAL_STATES and not alive(baseline_job["pid"])


def gpu_capacity():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in output.splitlines():
        gpu, free_mb, utilization = (int(value.strip()) for value in line.split(","))
        result[gpu] = {"free_mb": free_mb, "utilization": utilization}
    return result


def checkpoint_inputs(campaign, seed, late_step, donor_step):
    late = campaign / f"seed_{seed}/low/checkpoints/step_{late_step}"
    donor = campaign / f"seed_{seed}/low/checkpoints/step_{donor_step}"
    late_receipt = json.loads((late / "checkpoint.json").read_text())
    donor_receipt = json.loads((donor / "checkpoint.json").read_text())
    result = {
        "late_model": late_receipt["model"],
        "late_replay": late_receipt["replay_buffer"],
        "donor_model": donor_receipt["model"],
    }
    for artifact in result.values():
        if file_fingerprint(Path(artifact["path"]))["sha256"] != artifact["sha256"]:
            raise RuntimeError(f"Input hash mismatch: {artifact['path']}")
    return result


def register(protocol_file):
    output = ROOT.parent
    destination = output / "registration.json"
    if destination.exists():
        raise FileExistsError(
            "Registration already exists; do not mutate a registered pilot"
        )
    pilot = yaml.safe_load(protocol_file.read_text())
    campaign = Path(pilot["baseline_campaign"]).resolve()
    baseline_manifest = json.loads((campaign / "manifest.json").read_text())
    environment = yaml.safe_load((campaign / "protocol.yaml").read_text())
    rollout_block = environment["n_envs"] * environment["algorithm"]["train_freq"]
    for key in ("online_transitions", "evaluation_interval", "checkpoint_interval"):
        if pilot[key] <= 0 or pilot[key] % rollout_block:
            raise ValueError(f"{key} must be a positive multiple of {rollout_block}")
    if pilot["critic_calibration_steps"] < 0 or pilot["threads"] != 1:
        raise ValueError("Invalid calibration or native thread budget")
    if pilot["resource_policy"] != "wait_for_baseline_slot_release":
        raise ValueError("Pilot must wait for baseline resource release")
    if set(pilot["arms"]) != {"sham", "optimizer_reset", "critic_restore"}:
        raise ValueError("Register the three prespecified treatments")
    if len(pilot["continuation_streams"]) < 2:
        raise ValueError("Replicated continuation streams are required")
    for key in ("base_seeds", "continuation_streams", "arms"):
        if len(pilot[key]) != len(set(pilot[key])):
            raise ValueError(f"Duplicate entries in {key}")
    files = {}
    for folder in ("c2g_env", "baselines", "evaluation", "scripts", "conf", "tests"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".yaml"):
                files[path.relative_to(ROOT).as_posix()] = file_fingerprint(path)[
                    "sha256"
                ]
    inputs = {
        seed: checkpoint_inputs(campaign, seed, pilot["late_step"], pilot["donor_step"])
        for seed in pilot["base_seeds"]
    }
    jobs = [
        {
            "id": f"base_{seed}_stream_{stream}_{arm}",
            "base_seed": seed,
            "stream": stream,
            "arm": arm,
            "inputs": inputs[seed],
        }
        for seed, stream, arm in product(
            pilot["base_seeds"],
            pilot["continuation_streams"],
            pilot["arms"],
        )
    ]
    random.Random(pilot["scheduler_seed"]).shuffle(jobs)
    slots = [
        {
            "gpu": gpu,
            "cpus": cpus,
        }
        for gpu, cpus in zip(environment["gpus"], environment["cpu_groups"])
    ]
    cpus = [cpu for slot in slots for cpu in slot["cpus"]]
    if len(cpus) != len(set(cpus)) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Pilot CPU groups overlap or are unavailable")
    if any(len(s["cpus"]) < environment["n_envs"] + 1 for s in slots):
        raise ValueError("A slot lacks the required learner/environment CPUs")
    if pilot["maximum_concurrent_runs"] > len(slots):
        raise ValueError("Concurrency exceeds registered slots")
    body = {
        "schema": "registered-online-sac-value-state-v1",
        "output_root": str(output),
        "baseline_campaign": str(campaign),
        "baseline_source_id": baseline_manifest["source_id"],
        "baseline_source_snapshot": baseline_manifest["source_snapshot"],
        "protocol_file": file_fingerprint(protocol_file.resolve()),
        "pilot_protocol": pilot,
        "environment_protocol": environment,
        "source_files": files,
        "slots": slots,
        "jobs": jobs,
        "registered_job_count": len(jobs),
        "new_online_transitions_planned": len(jobs) * pilot["online_transitions"],
        "attempt_retention": (
            "All registered jobs and validation cells, including failures"
        ),
        "scope": (
            "Nested continuation streams on selected parent checkpoints; "
            "no from-scratch or confirmatory population claim"
        ),
    }
    registration = {**body, "registration_id": configuration_id(body)}
    write_json(destination, registration)
    print(f"Registered {len(jobs)} jobs: {destination}", flush=True)
    return destination


def launch(registration_path):
    registration = json.loads(registration_path.read_text())
    verify_registration(registration)
    output = Path(registration["output_root"])
    status_path = output / "status.json"
    if status_path.exists():
        raise FileExistsError(
            "Refusing to relaunch or silently retry an existing pilot"
        )
    campaign = Path(registration["baseline_campaign"])
    jobs = [{**job, "state": "pending"} for job in registration["jobs"]]
    slots = registration["slots"]
    processes = {}
    healthy_exit = False

    def snapshot(state, waiting=None, capacity_wait=None):
        write_json(
            status_path,
            {
                "state": state,
                "supervisor_pid": os.getpid(),
                "updated_unix": time.time(),
                "registration_id": registration["registration_id"],
                "registered_jobs": len(jobs),
                "counts": {
                    key: sum(job["state"] == key for job in jobs)
                    for key in ("pending", "running", "completed", "failed")
                },
                "waiting_for_baseline_gpus": waiting or [],
                "waiting_for_capacity_gpus": capacity_wait or [],
                "jobs": [
                    {k: v for k, v in job.items() if k != "inputs"} for job in jobs
                ],
            },
        )

    def start(job, slot):
        verify_registration(registration)
        directory = output / "runs" / job["id"]
        directory.parent.mkdir(exist_ok=True)
        environment = dict(os.environ)
        environment.update({key: "1" for key in THREAD_VARIABLES})
        environment.update(
            CUDA_VISIBLE_DEVICES=str(slot["gpu"]),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            UV_NO_SYNC="1",
            MPLCONFIGDIR=str(output / "plot-cache" / job["id"]),
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "baselines.online_stability",
            "--registration",
            str(registration_path),
            "--job",
            job["id"],
            "--cpus",
            *map(str, slot["cpus"]),
            "--device",
            "cuda:0",
        ]
        log = output / "runs" / f"{job['id']}.log"
        with log.open("x") as handle:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        job.update(
            state="running",
            gpu=slot["gpu"],
            cpus=slot["cpus"],
            pid=child.pid,
            started_unix=time.time(),
        )
        processes[job["id"]] = child
        print(
            f"Started {job['id']} on released GPU {slot['gpu']}, PID {child.pid}",
            flush=True,
        )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Pilot supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        snapshot("waiting_for_baseline")
        while any(job["state"] in ("pending", "running") for job in jobs):
            for job in jobs:
                if job["state"] != "running":
                    continue
                code = processes[job["id"]].poll()
                if code is not None:
                    state_file = output / "runs" / job["id"] / "status.json"
                    state = (
                        json.loads(state_file.read_text())
                        if state_file.exists()
                        else None
                    )
                    successful = (
                        code == 0
                        and state is not None
                        and state["state"] == "completed"
                    )
                    job.update(
                        state="completed" if successful else "failed",
                        returncode=code,
                        finished_unix=time.time(),
                    )
                    if not successful:
                        job["failure_detail"] = (
                            "Inspect retained run log/failure.json; no retry scheduled"
                        )
            baseline = json.loads((campaign / "status.json").read_text())
            owners = {job["gpu"]: job for job in baseline["jobs"]}
            busy = {job["gpu"] for job in jobs if job["state"] == "running"}
            waiting = []
            capacity_wait = []
            capacity = None
            for slot in slots:
                if slot["gpu"] in busy:
                    continue
                if not slot_released(owners[slot["gpu"]]):
                    waiting.append(slot["gpu"])
                    continue
                pending = next((job for job in jobs if job["state"] == "pending"), None)
                if (
                    pending is None
                    or len(busy)
                    >= registration["pilot_protocol"]["maximum_concurrent_runs"]
                ):
                    break
                if capacity is None:
                    capacity = gpu_capacity()
                available = capacity[slot["gpu"]]
                if available["free_mb"] < 4096 or available["utilization"] >= 90:
                    capacity_wait.append(slot["gpu"])
                    continue
                start(pending, slot)
                busy.add(slot["gpu"])
            remaining = any(job["state"] in ("pending", "running") for job in jobs)
            snapshot(
                ("running" if busy else "waiting_for_baseline")
                if remaining
                else (
                    "completed"
                    if all(j["state"] == "completed" for j in jobs)
                    else "completed_with_failures"
                ),
                waiting,
                capacity_wait,
            )
            if remaining:
                time.sleep(10)
        healthy_exit = True
    finally:
        for child in processes.values():
            if child.poll() is None:
                child.terminate()
        for child in processes.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        if not healthy_exit:
            snapshot("interrupted_or_failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--protocol", type=Path, default=ROOT.parent / "protocol.yaml")
    parser.add_argument(
        "--registration", type=Path, default=ROOT.parent / "registration.json"
    )
    args = parser.parse_args()
    if args.register:
        register(args.protocol)
    else:
        launch(args.registration.resolve())


if __name__ == "__main__":
    main()
