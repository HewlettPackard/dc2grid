from __future__ import annotations

import copy

import pytest

# isort: off
from train_macro import (
    MacroJournal,
    MacroLatentEnv,
    attach_update_counters,
    initialize_actor,
    warmup_and_calibrate,
)
from evaluate_macro import evaluate_case, scripted_action
from evaluate_low import audit_primary
from economy_context import PRIOR, cap_cpu, read_json
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
# isort: on


class MacroFixture(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(23,),
            dtype=np.float32,
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(23, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1
        done = self.steps == 3
        return (
            np.zeros(23, dtype=np.float32),
            0.0,
            done,
            False,
            {
                "episode_result": {"complete": True, "fixture": True} if done else None,
            },
        )


def make_model(algorithm):
    cap_cpu()
    if algorithm == "sac":
        return SAC(
            "MlpPolicy",
            MacroFixture(),
            buffer_size=128,
            batch_size=4,
            learning_starts=16,
            train_freq=1,
            gradient_steps=1,
            ent_coef=0.0001,
            gamma=1.0,
            learning_rate=1e-4,
            policy_kwargs={"net_arch": [16, 16]},
            seed=1200,
            device="cpu",
        )
    return PPO(
        "MlpPolicy",
        MacroLatentEnv(MacroFixture()),
        batch_size=4,
        n_steps=8,
        n_epochs=3,
        ent_coef=0.0,
        gamma=1.0,
        learning_rate=1e-4,
        policy_kwargs={
            "net_arch": {"pi": [16, 16], "vf": [16, 16]},
            "activation_fn": torch.nn.ReLU,
            "ortho_init": False,
        },
        seed=1200,
        device="cpu",
    )


@pytest.mark.parametrize("algorithm", ("ppo", "sac"))
def test_native_economic_training_lifecycle_and_complete_transition_capture(
    algorithm,
    tmp_path,
):
    model = make_model(algorithm)
    initialization = initialize_actor(model, algorithm, 1200, 0.05, [16, 16])
    assert initialization["initial_deterministic_overlay_mw"] == 0.0
    model.set_logger(configure(str(tmp_path / "metrics"), ["csv"]))
    handles = attach_update_counters(model, algorithm)
    callback = MacroJournal(tmp_path)
    warmup_and_calibrate(
        model,
        algorithm,
        warmup_steps=16,
        critic_updates=8,
        callback=callback,
    )
    model.learn(16, callback=callback, reset_num_timesteps=False)
    receipt = callback.save_transitions(model)
    assert receipt["transitions"] == 32
    assert not receipt["partial"]
    assert callback.episodes == 10
    assert model.c2g_update_counts["critic"] == model.c2g_update_counts["actor"] + 8
    with np.load(receipt["path"]) as data:
        assert data["observations"].shape == (32, 1, 23)
        assert data["actions"].shape == (32, 1, 1)
    for handle in handles:
        handle.remove()
    model.get_env().close()


def test_paired_initial_actor_parameters_and_physical_mapping():
    ppo, sac = make_model("ppo"), make_model("sac")
    first = initialize_actor(ppo, "ppo", 1201, 0.05, [16, 16])
    second = initialize_actor(sac, "sac", 1201, 0.05, [16, 16])
    assert first == second
    wrapper = MacroLatentEnv(MacroFixture())
    for value in (-20.0, -1.2, 0.0, 1.2, 20.0):
        latent = np.array([value], dtype=np.float32)
        expected = sac.policy.unscale_action(
            torch.tanh(torch.from_numpy(latent)).numpy()
        )
        np.testing.assert_array_equal(wrapper.action(latent), expected)
    ppo.get_env().close()
    sac.get_env().close()


def test_scripted_margin_uses_only_current_observations():
    observation = np.zeros(23, dtype=np.float32)
    observation[20] = 0.3
    observation[17] = 0.2
    policy = {"mode": "margin", "threshold_usd_mwh": 10.0}
    np.testing.assert_array_equal(scripted_action(policy, observation), [1])
    observation[20] = 0.1
    np.testing.assert_array_equal(scripted_action(policy, observation), [0])


def test_parent_evaluation_and_raw_audit_on_training_fixture(tmp_path):
    protocol = copy.deepcopy(read_json(PRIOR / "registration.json"))
    cell = {"start_day": 16, "seed": 97000020, "regd_load_bias": 0.5}
    protocol["development_cells"] = [cell]
    policy = {
        "id": "unit_one",
        "family": "unit_one",
        "kind": "scripted",
        "mode": "one",
    }
    result = evaluate_case(protocol, policy, cell, tmp_path / "episode")
    assert result["qualified"]
    assert result["service_fast_steps"] == 17280
    assert result["recovery_fast_steps"] == 1440
    assert audit_primary(tmp_path / "episode", result) > 190
    np.testing.assert_allclose(result["reward_sum"], result["excess_cash_usd"] / 100)
