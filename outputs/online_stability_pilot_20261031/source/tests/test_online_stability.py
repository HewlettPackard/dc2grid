import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from baselines.online_stability import (
    Diagnostics,
    actor_identity,
    apply_treatment,
    calibrate_critic,
    fingerprint_state,
    run_job,
    set_update_rng,
    stream_seed,
    verify_registration,
)
from scripts.launch_online_stability import slot_released
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines.train_corrected_sac import EpisodeConditions
from c2g_env import C2GFastEnv
from c2g_env.experiments.identity import configuration_id, file_fingerprint
from c2g_env.train_envs import RuleMacroWrappedEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("online-pilot")
    settings = yaml.safe_load((ROOT / "c2g_env/config.yaml").read_text())
    settings["global"]["episode_ticks"] = 12
    config = directory / "env.yaml"
    config.write_text(yaml.safe_dump(settings))
    environment = {
        "scenario": "default",
        "simulation_mode": "physical_v1",
        "config_path": str(config),
        "threads": 1,
        "n_envs": 2,
        "train_days": [0, 20],
        "train_biases": [0.0, 0.5],
        "validation": [
            {"start_day": 196, "regd_load_bias": 0.0, "seed": 90001},
            {"start_day": 196, "regd_load_bias": 0.5, "seed": 90002},
        ],
        "held_out_days": [207, 248, 249],
        "selection": {
            "min_terminal_soc": 0.2,
            "max_backlog_fraction": 0.01,
            "max_thermal_warning_fraction": 0.01,
        },
        "algorithm": {
            "train_freq": 1,
            "batch_size": 2,
            "tau": 0.005,
            "learning_rate": 0.0003,
        },
        "low": {"gamma": 0.99},
    }
    vec = DummyVecEnv(
        [lambda: RuleMacroWrappedEnv(C2GFastEnv(config_path=config)) for _ in range(2)]
    )
    model = SAC(
        "MlpPolicy",
        vec,
        learning_starts=0,
        batch_size=2,
        buffer_size=64,
        train_freq=1,
        gradient_steps=-1,
        policy_kwargs={"net_arch": [8, 8]},
        device="cpu",
        seed=1006,
    )
    model.c2g_context = {"simulation_mode": "physical_v1", "test_only": True}
    model.learn(4)
    donor = directory / "donor.zip"
    model.save(donor)
    model.learn(6, reset_num_timesteps=False)
    late, replay = directory / "late.zip", directory / "replay.pkl"
    model.save(late)
    model.save_replay_buffer(replay)
    vec.close()
    registration = {
        "source_files": {},
        "protocol_file": file_fingerprint(config),
        "environment_protocol": environment,
        "pilot_protocol": {
            "online_transitions": 8,
            "critic_calibration_steps": 2,
            "evaluation_interval": 4,
            "checkpoint_interval": 4,
            "status_interval": 2,
        },
    }
    registration["registration_id"] = configuration_id(registration)
    inputs = {
        "late_model": file_fingerprint(late),
        "late_replay": file_fingerprint(replay),
        "donor_model": file_fingerprint(donor),
    }
    return directory, registration, inputs


def test_treatments_and_calibration_preserve_actor_entropy(prepared):
    _, _, inputs = prepared
    identities = {}
    for arm in ("sham", "optimizer_reset", "critic_restore"):
        model = SAC.load(inputs["late_model"]["path"], device="cpu")
        model.load_replay_buffer(inputs["late_replay"]["path"])
        donor = SAC.load(inputs["donor_model"]["path"], device="cpu")
        before = actor_identity(model)
        identities[arm] = apply_treatment(model, arm, donor)
        assert actor_identity(model) == before
        set_update_rng(1006, 0, 20)
        calibrate_critic(model, 2, 2)
        assert actor_identity(model) == before
    assert (
        identities["sham"]["actor_and_entropy"]
        == identities["critic_restore"]["actor_and_entropy"]
    )
    assert identities["sham"]["critic"] == identities["optimizer_reset"]["critic"]
    assert identities["sham"]["target"] == identities["optimizer_reset"]["target"]
    assert (
        identities["critic_restore"]["critic"]
        != identities["optimizer_reset"]["critic"]
    )
    assert (
        identities["optimizer_reset"]["critic_optimizer"]
        == identities["critic_restore"]["critic_optimizer"]
    )
    assert (
        identities["sham"]["critic_optimizer"]
        != identities["optimizer_reset"]["critic_optimizer"]
    )


@pytest.mark.parametrize("arm", ["sham", "optimizer_reset", "critic_restore"])
def test_real_online_collection_and_final_update_accounting(prepared, tmp_path, arm):
    _, registration, inputs = prepared
    job = {
        "id": f"test_{arm}",
        "base_seed": 1006,
        "stream": 0,
        "arm": arm,
        "inputs": inputs,
    }
    run_job(copy.deepcopy(registration), job, tmp_path / arm, [28, 29, 30], "cpu")
    output = tmp_path / arm
    status = json.loads((output / "status.json").read_text())
    initial = json.loads((output / "initialization.json").read_text())
    final_validation = json.loads((output / "validation/online_8.json").read_text())
    assert status["state"] == "completed"
    assert status["new_online_transitions"] == 8
    assert status["new_normal_sac_updates"] == 8
    assert status["actual_replay_position"] == status["expected_replay_position"]
    assert status["actual_replay_position"] != initial["initial_replay_position"]
    assert final_validation["new_online_transitions"] == 8
    assert final_validation["new_normal_sac_updates"] == 8
    assert len(final_validation["cells"]) == 2
    assert all(cell["fast_steps"] == 12 for cell in final_validation["cells"])
    final_model = SAC.load(output / "final_model.zip", device="cpu")
    assert final_model._n_updates - initial["base_sac_updates"] == 8
    assert final_model.c2g_context["experiment"] == "online-value-state-pilot"
    assert (output / "checkpoints/online_8/replay_buffer.pkl").exists()
    for artifact in inputs.values():
        assert file_fingerprint(Path(artifact["path"]))["sha256"] == artifact["sha256"]


def test_diagnostics_are_observation_reward_transparent(prepared):
    _, registration, _ = prepared
    config = registration["environment_protocol"]["config_path"]
    original = C2GFastEnv(config_path=config)
    instrumented = Diagnostics(C2GFastEnv(config_path=config))
    a, _ = original.reset(seed=100)
    b, _ = instrumented.reset(seed=100)
    np.testing.assert_array_equal(a, b)
    action = np.array([0.7, 0.7, 0.7, 0.1], dtype=np.float32)
    for _ in range(4):
        before_commitment = instrumented.fast.committed_mw
        a, ra, ta, xa, ia = original.step(action)
        b, rb, tb, xb, ib = instrumented.step(action)
        np.testing.assert_array_equal(a, b)
        assert (ra, ta, xa) == (rb, tb, xb)
        assert ib["pilot_committed_mw"] == before_commitment
        assert ib["pilot_normalized_error_sq"] == pytest.approx(
            (
                ia["tracking_err_kw"]
                / max(before_commitment * 1000, original._tracking_floor_kw)
            )
            ** 2
        )
    original.close()
    instrumented.close()


def test_stream_identity_and_condition_pairing(prepared):
    _, registration, _ = prepared
    protocol = registration["environment_protocol"]
    seeds = [
        stream_seed(1006, stream, phase)
        for stream in (0, 1, 2)
        for phase in (10, 20, 21, 30, 31)
    ]
    assert len(set(seeds)) == len(seeds)
    envs = [
        EpisodeConditions(
            RuleMacroWrappedEnv(C2GFastEnv(config_path=protocol["config_path"])),
            protocol["train_days"],
            protocol["train_biases"],
            seed=stream_seed(1006, 1, 10),
        )
        for _ in range(3)
    ]
    for _ in range(3):
        records = [env.reset()[1]["condition"] for env in envs]
        assert records[0] == records[1] == records[2]
    for env in envs:
        env.close()
    set_update_rng(1006, 0, 30)
    a = fingerprint_state((np.random.random(4), torch.rand(4)))
    set_update_rng(1006, 0, 30)
    b = fingerprint_state((np.random.random(4), torch.rand(4)))
    assert a == b


def test_baseline_slots_cannot_be_stolen():
    assert not slot_released({"state": "running", "pid": 42}, alive=lambda pid: False)
    assert not slot_released({"state": "completed", "pid": 42}, alive=lambda pid: True)
    assert slot_released({"state": "completed", "pid": 42}, alive=lambda pid: False)
    assert slot_released({"state": "blocked", "pid": 42}, alive=lambda pid: False)


def test_registered_settings_cannot_change(prepared):
    _, registration, _ = prepared
    verify_registration(registration)
    changed = copy.deepcopy(registration)
    changed["pilot_protocol"]["online_transitions"] += 2
    with pytest.raises(RuntimeError, match="registration identity"):
        verify_registration(changed)
