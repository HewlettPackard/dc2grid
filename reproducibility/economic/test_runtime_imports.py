"""Opt-in import/controller probe on a relocated source-only checkout; no episodes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import adapter

PROBE = r"""
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import adapter
root, output = Path(sys.argv[2]), Path(sys.argv[3])
manifest, policies, _ = adapter.load_metadata()
adapter._bootstrap(root, manifest, output)
import numpy as np
import gymnasium as gym
from controls import scripted_action
from fraction_macro import FractionalDispatchTask
from track_b.economy_macro import EconomicDispatchTask
from train_macro import MacroLatentEnv
from pid_reference import FullWorkPID, hardware_factory
assert issubclass(FractionalDispatchTask, EconomicDispatchTask)
observation = np.zeros(23, dtype=np.float32)
for name, expected in (("script_zero", 0.0), ("script_f100_always", 1.0)):
    spec = policies["confirmation/" + name]["policy"]
    np.testing.assert_array_equal(scripted_action(spec, observation), [expected])
hardware_observation = np.zeros(18, dtype=np.float32)
hardware_observation[:2] = 30 / 35
hardware_observation[2] = 0.5
hardware_observation[6] = 0.25
hardware_observation[17] = 0.5
native = hardware_factory(policies["pid/pid_native"]["policy"], {})
full = hardware_factory(policies["pid/pid_full_work"]["policy"], {})
assert isinstance(full, FullWorkPID)
native.reset()
full.reset()
native_action = native.predict(hardware_observation, deterministic=True)[0]
full_action = full.predict(hardware_observation, deterministic=True)[0]
assert full_action[0] == 1.0
np.testing.assert_array_equal(full_action[1:], native_action[1:])
class SpacesOnly(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    def reset(self, **kwargs):
        raise AssertionError("This test never simulates")
    def step(self, action):
        raise AssertionError("This test never simulates")
wrapper = MacroLatentEnv(SpacesOnly())
latent = np.array([0.5], dtype=np.float32)
np.testing.assert_allclose(wrapper.action(latent), np.tanh(latent), atol=1e-7)
adapter._check_import_origins(root, manifest)
print(json.dumps({
    "frozen_imports_resolved_from_relocated_checkout": True,
    "zero_and_full_fraction_scripts_checked": True,
    "explicit_PID_factory_and_full_work_throttle_checked": True,
    "PPO_latent_wrapper_checked": True,
    "simulator_resets": 0,
    "simulator_episodes": 0,
    "training": False,
}))
"""


@unittest.skipUnless(
    os.environ.get("DC2GRID_FROZEN_ROOT"),
    "Set DC2GRID_FROZEN_ROOT for an optional read-only-source import probe",
)
class RelocatedRuntimeImports(unittest.TestCase):
    def test_real_protocols_relocate_without_private_or_historical_inputs(self):
        manifest, policies, _ = adapter.load_metadata()
        original = Path(os.environ["DC2GRID_FROZEN_ROOT"]).resolve(strict=True)
        work = Path(".test-fixtures") / f"inputs-{os.getpid()}-{uuid.uuid4().hex}"
        work.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, work)
        checkout = (work / "protocol-only-checkout").resolve()
        checkout.mkdir()
        for row in [*manifest["configs"], *manifest["protocols"]]:
            source = adapter.checkout_path(original, row["checkout_relative"])
            self.assertEqual(adapter.file_hash(source), row["sha256"])
            target = checkout / row["checkout_relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for relative in (
            "data/processed/workload_traces",
            "data/processed/energy",
            "data/processed/weather",
        ):
            (checkout / relative).mkdir(parents=True)
        for key in (
            "absolute/economic_sac_1200",
            "fractional/potential_sac_1200",
            "confirmation/model__potential_sac_1200",
            "pid/pid_full_work",
        ):
            entry = policies[key]
            output = (work / entry["backend"]).resolve()
            output.mkdir()
            protocol, policy = adapter._prepare_inputs(
                checkout, manifest, entry, output
            )
            original_protocol = adapter.read_json(adapter.mapped_path(
                checkout, entry["protocol_recorded_path"], manifest
            ))
            self.assertEqual(
                protocol["registration_id"], original_protocol["registration_id"]
            )
            self.assertEqual(policy["id"], entry["policy"]["id"])
            if entry["backend"] == "confirmation":
                task = adapter.read_json(Path(protocol["task_protocol"]["path"]))
                task_original = adapter.read_json(adapter.mapped_path(
                    checkout, original_protocol["task_protocol"]["path"], manifest
                ))
                task["config_path"] = task_original["config_path"]
                self.assertEqual(task, task_original)
                protocol["policies"] = original_protocol["policies"]
                protocol["task_protocol"] = original_protocol["task_protocol"]
            else:
                protocol["config_path"] = original_protocol["config_path"]
            self.assertEqual(protocol, original_protocol)

    def test_imports_and_controllers_without_simulator_reset(self):
        manifest, _, _ = adapter.load_metadata()
        original = Path(os.environ["DC2GRID_FROZEN_ROOT"]).resolve(strict=True)
        work = Path(".test-fixtures") / f"imports-{os.getpid()}-{uuid.uuid4().hex}"
        work.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, work)
        checkout = (work / "source-only-checkout").resolve()
        checkout.mkdir()
        output = (work / "probe-result").resolve()
        output.mkdir()
        for row in [*manifest["sources"], *manifest["configs"]]:
            source = adapter.checkout_path(original, row["checkout_relative"])
            self.assertEqual(adapter.file_hash(source), row["sha256"])
            target = checkout / row["checkout_relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        environment = adapter._environment(output)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                PROBE,
                str(adapter.HERE),
                str(checkout),
                str(output),
            ],
            env=environment,
            cwd=output,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(receipt["frozen_imports_resolved_from_relocated_checkout"])
        self.assertEqual(receipt["simulator_resets"], 0)
        self.assertEqual(receipt["simulator_episodes"], 0)
        for row in manifest["sources"]:
            self.assertEqual(
                adapter.file_hash(checkout / row["checkout_relative"]), row["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
