from __future__ import annotations

import copy
import io
import math

import pytest

# isort: off
from ppo_adapter import (
    ACTION_HIGH,
    ACTION_LOW,
    HistoryPPOController,
    LatentActionEnv,
    initialize_from_bc,
    native_from_latent,
)
from economy_context import PRIOR, cap_cpu, read_json
from economy_eval import evaluate_episode
from evaluate_low import audit_primary
from history_policy import HistorySACController
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
# isort: on


class PhysicalFixture(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(24,),
            dtype=np.float32,
        )
        self.executed = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(24, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.executed.append(action.copy())
        self.steps += 1
        return np.zeros(24, dtype=np.float32), 0.0, self.steps == 3, False, {}


def make_ppo(env=None):
    cap_cpu()
    return PPO(
        "MlpPolicy",
        LatentActionEnv(env if env is not None else PhysicalFixture()),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=0.0,
        ent_coef=0.0,
        seed=711,
        device="cpu",
        policy_kwargs={
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": torch.nn.ReLU,
            "log_std_init": math.log(0.01),
        },
    )


@pytest.mark.parametrize("seed", (1100, 1101, 1102, 1103))
def test_exact_initializer_transfer_and_stock_save_load(seed):
    protocol = read_json(PRIOR / "registration.json")
    teacher = SAC.load(protocol["initializers"][str(seed)]["path"], device="cpu")
    model = make_ppo()
    receipt = initialize_from_bc(model, teacher)
    assert receipt["actor_body_sha256"] == receipt["teacher_body_sha256"]
    assert receipt["actor_mean_sha256"] == receipt["teacher_mean_sha256"]
    assert torch.equal(model.policy.log_std, teacher.actor.log_std.bias)
    observations = (
        np.random.default_rng(19).uniform(0, 1, (1024, 24)).astype(np.float32)
    )
    expected = teacher.predict(observations, deterministic=True)[0]
    actual = native_from_latent(model.predict(observations, deterministic=True)[0])
    np.testing.assert_array_equal(actual, expected)
    saved = io.BytesIO()
    model.save(saved)
    saved.seek(0)
    restored = PPO.load(saved, device="cpu")
    np.testing.assert_array_equal(
        native_from_latent(restored.predict(observations, deterministic=True)[0]),
        expected,
    )
    controller = HistoryPPOController(restored)
    observation = observations[0, :18]
    action = controller.predict(observation)[0]
    following = controller.observer.features(observation)
    np.testing.assert_array_equal(following[18:22], action)
    model.get_env().close()


def test_latent_clipping_is_physically_neutral_and_invalid_input_is_loud():
    negative = np.full(3, -20, dtype=np.float32)
    positive = np.full(3, 20, dtype=np.float32)
    np.testing.assert_array_equal(native_from_latent(negative), ACTION_LOW)
    np.testing.assert_array_equal(native_from_latent(positive), ACTION_HIGH)
    np.testing.assert_array_equal(
        native_from_latent(np.full(3, 50, dtype=np.float32)),
        native_from_latent(positive),
    )
    with pytest.raises(ValueError, match="finite"):
        native_from_latent([0, np.nan, 0])
    with pytest.raises(ValueError):
        native_from_latent([0, 0])
    with pytest.raises(ValueError, match="declared latent box"):
        LatentActionEnv(PhysicalFixture()).action([0, 50, 0])


def test_stock_ppo_keeps_unclipped_gaussian_samples_and_exact_log_probabilities():
    fixture = PhysicalFixture()
    model = make_ppo(fixture)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.fill_(50.0)
    model.learn(total_timesteps=8)
    buffer = model.rollout_buffer
    assert np.all(buffer.actions > 20)
    assert len(fixture.executed) == 8
    for action in fixture.executed:
        np.testing.assert_array_equal(action, ACTION_HIGH)
    with torch.no_grad():
        _, probabilities, _ = model.policy.evaluate_actions(
            torch.as_tensor(buffer.observations),
            torch.as_tensor(buffer.actions),
        )
    np.testing.assert_array_equal(probabilities.numpy(), buffer.log_probs.reshape(-1))
    assert np.isfinite(buffer.advantages).all()
    for info in model.get_env().buf_infos:
        assert not info["TimeLimit.truncated"]
    model.get_env().close()


def test_initial_ppo_matches_bc_over_the_complete_physical_horizon(tmp_path):
    protocol = copy.deepcopy(read_json(PRIOR / "registration.json"))
    protocol["final_cells"] = []
    teacher = SAC.load(protocol["initializers"]["1100"]["path"], device="cpu")
    model = make_ppo()
    initialize_from_bc(model, teacher)
    policy = {
        "id": "initial_actor_parity",
        "family": "unit_fixture",
        "hardware": "initial_actor_fixture",
        "macro": "rule",
    }
    cell = {
        "start_day": protocol["train_days"][0],
        "seed": 97000002,
        "regd_load_bias": 0.5,
    }
    reference = evaluate_episode(
        protocol,
        policy,
        cell,
        tmp_path / "bc",
        hardware_factory=lambda spec, config: HistorySACController(teacher),
    )
    current = evaluate_episode(
        protocol,
        policy,
        cell,
        tmp_path / "ppo",
        hardware_factory=lambda spec, config: HistoryPPOController(model),
    )
    for key in (
        "service",
        "recovery",
        "grid",
        "decomposition",
        "cashflow_26h_usd",
        "regulation_revenue_usd",
        "service_terminal",
        "qualified",
    ):
        assert current[key] == reference[key], key
    assert current["complete"] and current["qualified"]
    assert audit_primary(tmp_path / "ppo", current) > 190
    model.get_env().close()
