"""Use the unchanged native PID and an explicitly labelled full-work variant."""

from __future__ import annotations

from pathlib import Path

# isort: off
from pid_context import file_hash, read_json, write_json
from economy_eval import evaluate_episode
from evaluate import make_hardware
from run_confirmation import next_attempt
import numpy as np
# isort: on


class FullWorkPID:
    def __init__(self, base):
        self.base = base

    def reset(self):
        self.base.reset()

    def predict(self, observation, deterministic=True):
        action, state = self.base.predict(observation, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[-1] != 4 or not np.isfinite(action).all():
            raise ValueError("PID did not return four finite actuator commands")
        action[..., 0] = 1.0
        return action, state


def hardware_factory(policy, protocol):
    if policy["hardware"] not in ("pid", "cooling_first"):
        raise ValueError("Only registered PID references or the analytic test fixture")
    hardware = make_hardware(policy, protocol)
    if policy.get("full_work", False):
        if policy["hardware"] != "pid":
            raise ValueError("The full-work variant must wrap the native PID")
        return FullWorkPID(hardware)
    return hardware


def completed_case(protocol, panel, policy, cell, destination):
    root = Path(destination)
    paths = [root / "episode.json"] if (root / "episode.json").exists() else []
    paths += sorted(root.glob("attempts/attempt_*/episode.json"))
    if len(paths) > 1:
        raise ValueError("Never choose among multiple completed outcomes")
    if not paths:
        return None
    path = paths[0]
    outcome = read_json(path)
    if (
        outcome["study_registration_id"] != protocol["registration_id"]
        or outcome["policy"] != policy
        or outcome["cell"] != cell
        or cell not in protocol["panels"][panel]["cells"]
    ):
        raise ValueError("Existing PID outcome differs from the locked slot")
    for name, digest in outcome["trace_hashes"].items():
        if name not in {"hardware.csv.gz", "macro.csv.gz", "recovery.csv.gz"}:
            raise ValueError("Unexpected cached PID trace name")
        if file_hash(path.parent / name) != digest:
            raise ValueError("A completed PID trace changed")
    return path, outcome


def execute(protocol, panel, policy, cell, destination):
    if (
        policy not in protocol["policies"]
        or cell not in protocol["panels"][panel]["cells"]
    ):
        raise ValueError("PID episode lies outside the declared comparison")
    cached = completed_case(protocol, panel, policy, cell, destination)
    if cached is not None:
        path, outcome = cached
    else:
        target = next_attempt(destination)
        outcome = evaluate_episode(
            protocol,
            policy,
            cell,
            target,
            hardware_factory=hardware_factory,
        )
        path = target / "episode.json"
        write_json(
            target / "reference_scope.json",
            {
                "registration_id": protocol["registration_id"],
                "panel": panel,
                "role": protocol["role"],
                "full_work_variant": policy["full_work"],
                "original_confirmation_registration_id": protocol[
                    "confirmation_registration_id"
                ],
                "episode_sha256": file_hash(path),
                "no_training_or_gain_tuning": True,
                "not_part_of_the_original_primary_test": True,
            },
        )
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "panel": panel,
        "policy_id": policy["id"],
        "cell": cell,
        "complete": bool(outcome["complete"] and outcome["recovery"]["complete"]),
        "qualified": outcome["qualified"],
        "reused_completed_slot": cached is not None,
    }
