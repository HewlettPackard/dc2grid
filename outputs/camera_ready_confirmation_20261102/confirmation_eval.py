"""Execute authorized confirmation episodes without changing any frozen policy."""

from __future__ import annotations

import traceback
from pathlib import Path

# isort: off
from confirmation_context import file_hash, read_json, write_json
from controls import load_model, make_environment, scripted_action
import gymnasium as gym
import pandas as pd
# isort: on

_MODELS = {}


def evaluate_case(protocol, panel, policy, cell, destination):
    if (
        panel not in protocol["panels"]
        or cell not in protocol["panels"][panel]["cells"]
    ):
        raise ValueError("Episode is outside the registered panel")
    if policy not in protocol["policies"]:
        raise ValueError("Policy is outside the closed confirmation roster")
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    task_protocol = read_json(protocol["task_protocol"]["path"])
    environment = None
    try:
        task, environment = make_environment(
            task_protocol,
            cell,
            protocol["panels"][panel]["cells"],
            policy,
        )
        wrapper_stack, current = [], environment
        while True:
            wrapper_stack.append(type(current).__name__)
            if not isinstance(current, gym.Wrapper):
                break
            current = current.env
        model = None
        if policy["kind"] == "neural":
            key = policy["checkpoint"]["sha256"]
            if key not in _MODELS:
                _MODELS[key] = load_model(policy)
            model = _MODELS[key]
        elif policy["kind"] != "scripted":
            raise ValueError("Unknown confirmation policy kind")
        observation, reset_info = environment.reset()
        while True:
            action = (
                model.predict(observation, deterministic=True)[0]
                if model is not None
                else scripted_action(policy, observation)
            )
            observation, _, terminated, truncated, _ = environment.step(action)
            if truncated:
                raise AssertionError("The 26-hour task must not expose a timeout")
            if terminated:
                break
        trace_hashes = {}
        for name, rows in task.terminal_traces().items():
            if rows:
                path = directory / name
                pd.DataFrame(rows).to_csv(
                    path, index=False, compression="gzip", mode="x"
                )
                trace_hashes[name] = file_hash(path)
        record = {
            "evaluation_registration_id": protocol["registration_id"],
            "panel": panel,
            "scope": protocol["panels"][panel]["scope"],
            "policy": policy,
            "cell": cell,
            "reset_info": reset_info,
            "outcome": task.episode_result,
            "trace_hashes": trace_hashes,
            "training_performed": False,
            "potential_used_at_evaluation": False,
            "actual_deployment_environment_stack": wrapper_stack,
            "inherited_task_metadata_describes_the_parent_method_study": True,
        }
        write_json(directory / "episode.json", record)
        return record
    except Exception as error:
        write_json(
            directory / "failure.json",
            {
                "panel": panel,
                "policy": policy,
                "cell": cell,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if environment is not None:
            environment.close()
