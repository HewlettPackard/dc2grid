from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import pandas as pd

from c2g_env.experiments.identity import configuration_id, file_fingerprint
from c2g_env.physics.workload import WorkloadOrchestrator

from .assume_backend import ThreeLBackend, local_directory, require_backend
from .config import (
    MODULE_ROOT,
    REPO_ROOT,
    contained_path,
    load_config,
    three_l_config,
    validate_numbers,
)
from .env import ThreeLevelEnv
from .market import ThreeLData, timestamp


def prepare(config: dict):
    root, backend, scenario, calendar = three_l_config(config)
    if calendar["time_step"] != "15min":
        raise ValueError(
            "This integration requires 3L's 15-minute settlement calendar."
        )
    start = timestamp(calendar["start_date"])
    score_start = start + pd.Timedelta(hours=config["simulation"]["warmup_hours"])
    end = score_start + pd.Timedelta(hours=config["simulation"]["scoring_hours"])
    if end > timestamp(calendar["end_date"]):
        raise ValueError("The requested experiment exceeds the loaded 3L calendar.")
    data = ThreeLData(scenario)
    data.require_coverage(start, end)
    require_backend(root)
    return root, backend, scenario, data, start, score_start, end


def write_json(path: Path, value) -> None:
    path = contained_path(path)
    with local_directory(path.parent), path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, default=runtime_value, allow_nan=False, indent=2)


def runtime_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    from baselines.metrics_callback import _runtime_json_value

    return _runtime_json_value(value)


def provenance(config, root, scenario, plant) -> dict:
    sources = list(MODULE_ROOT.glob("*.py")) + list(
        (REPO_ROOT / "c2g_env" / "physics").glob("*.py")
    )
    sources.extend((root / "assume_env").rglob("*.py"))
    sources.extend((root / "dc_env" / "policies").glob("*.py"))
    sources.extend((REPO_ROOT / "baselines").glob("*.py"))
    sources.extend(
        (
            REPO_ROOT / "c2g_env" / "env_low_level.py",
            REPO_ROOT / "c2g_env" / "env_high_level.py",
            REPO_ROOT / "c2g_env" / "thermal_limits.py",
            REPO_ROOT / "c2g_env" / "experiments" / "identity.py",
            REPO_ROOT / "baselines" / "metrics_callback.py",
            REPO_ROOT / "evaluation" / "run_benchmark.py",
            REPO_ROOT / "c2g_env" / "obs_indices.py",
        )
    )
    inputs = list(scenario.glob("*.csv")) + [scenario / "config.yaml", root / "uv.lock"]
    inputs += [REPO_ROOT / config["c2g"]["config"], root / config["three_l"]["config"]]
    trace_dir = REPO_ROOT / plant.global_config["trace_dir"]
    inputs.extend(
        trace_dir / name for name in WorkloadOrchestrator._TRACE_FILES.values()
    )
    revisions = {}
    for label, directory in (("c2g", REPO_ROOT), ("three_l", root)):
        revisions[label] = subprocess.run(
            ["git", "--no-pager", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return {
        "config": config,
        "config_id": configuration_id(config),
        "revisions": revisions,
        "source_fingerprints": [
            file_fingerprint(path) for path in sorted(set(sources))
        ],
        "input_fingerprints": [file_fingerprint(path) for path in sorted(set(inputs))],
        "versions": {
            name: metadata.version(name)
            for name in (
                "assume-framework",
                "mango-agents",
                "numpy",
                "pandas",
                "gymnasium",
                "stable-baselines3",
                "torch",
                "scipy",
            )
        },
        "python": {"version": sys.version, "executable": sys.executable},
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "battery_backend": "c2g_env.physics.bess._SimpleBESSModel",
        "workload_forecast": (
            "shared causal queue and nominal-facility projection through lead and "
            "delivery; trailing observed hourly inputs, persistence before that "
            "hour is observed; aggregate backlog; no job deadlines"
        ),
        "schedule_execution": (
            "accepted DA/IDM fills define the operating plan; the plan caps the "
            "hardware batch throttle"
        ),
        "demand_interpretation": (
            "C2G is additional demand, not a replacement for background demand"
        ),
        "settlement": "signed DA+ID+BM purchases; residual at delivery-slot BM price",
        "timebase": "timezone-naive 3L scenario labels; epoch conversion as UTC",
        "frequency_model": "not_modeled",
        "network_voltage_model": "not_modeled",
    }


def execute(config: dict, output: Path | None = None) -> Path:
    validate_numbers(config)
    root, backend_config, scenario, data, start, score_start, end = prepare(config)
    output = contained_path(output or MODULE_ROOT / "outputs" / uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    env = ThreeLevelEnv(
        config,
        data,
        start,
        end,
        score_start,
        lambda ledger: ThreeLBackend(
            root,
            backend_config,
            ledger,
            output / "backend",
            end,
            config["simulation"]["seed"],
        ),
    )
    write_json(
        output / "provenance.json", provenance(config, root, scenario, env.plant)
    )
    try:
        record_episode(env, output)
    finally:
        env.close()
    write_json(output / "summary.json", env.summary())
    write_json(output / "decisions.json", env.controllers.decisions)
    write_json(output / "policy_calls.json", env.controllers.policy_calls)
    write_json(output / "plan_revisions.json", env.controllers.plan_revisions)
    write_json(output / "settlements.json", env.records)
    write_json(
        output / "trades.json", [asdict(trade) for trade in env.ledger.trades.values()]
    )
    write_json(output / "clearing_events.json", env.backend.clearing_events)
    return output


def record_episode(env: ThreeLevelEnv, output: Path) -> None:
    output = contained_path(output)
    with (
        local_directory(output),
        (output / "transitions.jsonl").open("x", encoding="utf-8") as handle,
    ):
        observation, info = env.reset(seed=int(env.config["simulation"]["seed"]))
        write_json(
            output / "controllers.json",
            {
                "upper": env.controllers.manifest(),
                "inner": env.hardware.manifest(),
            },
        )
        _record(handle, {"event": "reset", "observation": observation, "info": info})
        while not env.finished:
            action = env.hardware_action()
            observation, reward, terminated, truncated, info = env.step(action)
            _record(
                handle,
                {
                    "event": "step",
                    "action": action,
                    "controller": env.hardware.last_decision,
                    "observation": observation,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                },
            )


def _record(handle, record) -> None:
    handle.write(json.dumps(record, default=runtime_value, allow_nan=False) + "\n")
    handle.flush()


def evaluate(config: dict, seeds: list[int], output: Path | None = None) -> Path:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be a nonempty, unique list.")
    output = contained_path(output or MODULE_ROOT / "outputs" / uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for seed in seeds:
        run_config = deepcopy(config)
        run_config["simulation"]["seed"] = seed
        validate_numbers(run_config)
        config_path = output / f"config-seed-{seed}.json"
        result_path = output / f"seed-{seed}"
        write_json(config_path, run_config)
        print(f"Starting seed {seed}: {result_path}", flush=True)
        _run_worker(config_path, result_path, output / f"seed-{seed}.log")
        with (result_path / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append({"output_directory": str(result_path), **summary})
        print(
            f"Seed {seed}: completed={summary['completed']}, "
            f"scored_cost_eur={summary['scored_cost_eur']:.2f}",
            flush=True,
        )
    write_json(output / "evaluation.json", {"seeds": seeds, "runs": summaries})
    pd.json_normalize(summaries).to_csv(
        output / "evaluation.csv", index=False, mode="x"
    )
    return output


def _run_worker(config_path: Path, output: Path, log_path: Path) -> None:
    command = [
        sys.executable,
        "-B",
        "-m",
        "c2g_env.experiments.3l-power-market.run",
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(
            f"Evaluation worker failed with exit code {result.returncode}; "
            f"diagnostics: {log_path}"
        )


def _apply_overrides(config, args) -> None:
    for argument, section, key in (
        ("inner_controller", "controllers", "inner"),
        ("model_dir", "controllers", "model_dir"),
        ("warmup_hours", "simulation", "warmup_hours"),
        ("scoring_hours", "simulation", "scoring_hours"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config[section][key] = value
    validate_numbers(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-level Ireland markets with C2G DC physics."
    )
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config.yaml")
    parser.add_argument(
        "--inner-controller",
        help="Existing C2G hardware controller; otherwise use the configuration.",
    )
    parser.add_argument(
        "--model-dir", help="PPO/SAC directory containing final_model.zip."
    )
    parser.add_argument("--warmup-hours", type=float)
    parser.add_argument("--scoring-hours", type=float)
    parser.add_argument("--seeds", type=int, nargs="+", help="Fresh process per seed.")
    parser.add_argument(
        "--output", type=Path, help="New module-contained output directory."
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate data and primary dependencies."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    _apply_overrides(config, args)
    if args.check:
        prepare(config)
        print(
            "Data and primary dependencies checked; "
            "complete 3L imports are checked at startup."
        )
        return
    output = (
        evaluate(config, args.seeds, args.output)
        if args.seeds is not None
        else execute(config, args.output)
    )
    print(f"Run results: {output}")


if __name__ == "__main__":
    main()
