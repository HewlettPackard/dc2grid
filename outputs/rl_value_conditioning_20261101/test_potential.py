# isort: off
from ctx import ACTION_LOW, ACTION_HIGH
from value_replay import (
    potential,
    shaped_rewards,
    PotentialReplayBuffer,
    ConditionedSAC,
)
from low_core import ReferenceSAC
from fit_bc import SpaceOnlyEnv, parameter_digest
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.logger import configure
# isort: on


def test_potential_telescopes_at_healthy_and_early_true_terminals():
    gamma = 0.99
    generator = np.random.default_rng(31)
    for length in (37, 17280):
        observations = torch.zeros((length + 1, 24), dtype=torch.float64)
        observations[:, 23] = 1 - torch.arange(length + 1) / 17280
        observations[:, 3:5] = torch.tensor(generator.random((length + 1, 2)))
        rewards = torch.tensor(generator.normal(size=(length, 1)))
        dones = torch.zeros((length, 1), dtype=torch.float64)
        dones[-1] = 1
        data = ReplayBufferSamples(
            observations=observations[:-1],
            next_observations=observations[1:],
            actions=torch.zeros((length, 3)),
            dones=dones,
            rewards=rewards,
        )
        discount = torch.tensor(gamma ** np.arange(length)).reshape(-1, 1)
        for config in (
            {"kind": "constant", "mean": 0.5},
            {"kind": "linear_causal", "weights": [0.1, 0.2, 0.3, 0.1], "alpha": 1},
        ):
            original = torch.sum(discount * rewards)
            changed = torch.sum(discount * shaped_rewards(data, config, gamma))
            expected = original - potential(observations[:1], config, gamma)[0, 0]
            torch.testing.assert_close(changed, expected, rtol=1e-12, atol=1e-10)


def make_model(conditioned):
    kwargs = {}
    if conditioned:
        kwargs = {
            "replay_buffer_class": PotentialReplayBuffer,
            "replay_buffer_kwargs": {
                "potential_config": {"kind": "constant", "mean": 0.0},
                "potential_gamma": 0.99,
            },
        }
    kind = ConditionedSAC if conditioned else ReferenceSAC
    model = kind(
        "MlpPolicy",
        SpaceOnlyEnv(24),
        seed=29,
        device="cpu",
        buffer_size=512,
        learning_starts=32,
        batch_size=32,
        ent_coef=1e-5,
        learning_rate=1e-4,
        policy_kwargs={"net_arch": [32, 32]},
        **kwargs,
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    model.freeze_reference()
    model.configure_reference("fixed_reference_kl", 0.1)
    generator = np.random.default_rng(50)
    observations = generator.normal(size=(128, 24)).astype(np.float32)
    observations[:, 23] = generator.uniform(0, 1, 128)
    model.attach_demonstrations(
        observations,
        np.tile(np.array([0.7, 0.7, 0], dtype=np.float32), (128, 1)),
        919,
    )
    for obs in observations:
        model.replay_buffer.add(
            obs[None],
            obs[None],
            np.zeros((1, 3), np.float32),
            np.array([0.5], np.float32),
            np.array([False]),
            [{}],
        )
    return model


def test_zero_potential_preserves_reference_sac_updates_bitwise():
    models = [make_model(False), make_model(True)]
    for model in models:
        np.random.seed(717)
        torch.manual_seed(717)
        model.train(4, 32)
    assert parameter_digest(models[0].policy) == parameter_digest(models[1].policy)


def test_storage_stays_original_and_inference_needs_no_reward_transform(tmp_path):
    model = make_model(True)
    model.replay_buffer.potential_config = {"kind": "constant", "mean": 0.5}
    stored = model.replay_buffer.rewards.copy()
    model.train(4, 32)
    np.testing.assert_array_equal(stored, model.replay_buffer.rewards)
    path = tmp_path / "conditioned.zip"
    model.save(path)
    inference = SAC.load(path, device="cpu")
    assert inference.replay_buffer_class is ReplayBuffer
    assert np.array_equal(inference.action_space.low, ACTION_LOW)
    assert np.array_equal(inference.action_space.high, ACTION_HIGH)
    observation = np.zeros(24, np.float32)
    np.testing.assert_array_equal(
        model.predict(observation, deterministic=True)[0],
        inference.predict(observation, deterministic=True)[0],
    )


def test_mixed_prior_is_one_mean_over_equal_halves(monkeypatch):
    model = make_model(True)
    model.configure_mixture(0.1)
    seen = []
    original = model.reference_kl

    def record(observations):
        seen.append(observations.detach().clone())
        return original(observations)

    monkeypatch.setattr(model, "reference_kl", record)
    model.train(1, 32)
    assert len(seen) == 1 and seen[0].shape == (64, 24)
    torch.testing.assert_close(
        original(seen[0]), 0.5 * (original(seen[0][:32]) + original(seen[0][32:]))
    )


def test_mismatched_discount_and_implicit_resume_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="discounts must agree"):
        ConditionedSAC(
            "MlpPolicy",
            SpaceOnlyEnv(24),
            gamma=0.99,
            replay_buffer_class=PotentialReplayBuffer,
            replay_buffer_kwargs={
                "potential_config": {"kind": "constant", "mean": 0},
                "potential_gamma": 0.98,
            },
        )
    model = make_model(True)
    path = tmp_path / "inference.zip"
    model.save(path)
    with pytest.raises(ValueError, match="explicitly configured"):
        ConditionedSAC.load(path, device="cpu")


def test_replay_preserves_early_true_terminal_mask():
    space = SpaceOnlyEnv(24)
    buffer = PotentialReplayBuffer(
        8,
        space.observation_space,
        space.action_space,
        device="cpu",
        potential_config={"kind": "constant", "mean": 0.5},
        potential_gamma=0.99,
    )
    observation = np.zeros((1, 24), dtype=np.float32)
    observation[:, 23] = 0.8
    following = observation.copy()
    following[:, 23] -= 1 / 17280
    buffer.add(
        observation,
        following,
        np.zeros((1, 3), dtype=np.float32),
        np.array([-1.0], dtype=np.float32),
        np.array([True]),
        [{"TimeLimit.truncated": False}],
    )
    sample = buffer.sample(1)
    assert sample.dones.item() == 1
    expected = (
        -1
        - potential(torch.from_numpy(observation), buffer.potential_config, 0.99).item()
    )
    assert sample.rewards.item() == pytest.approx(expected, abs=1e-6)


def test_full_reward_probe_uses_the_training_timeout_mask():
    from train_value import reward_probe

    model = make_model(True)
    model.replay_buffer.potential_config = {"kind": "constant", "mean": 0.5}
    model.replay_buffer.dones[0:2, 0] = 1
    model.replay_buffer.timeouts[0, 0] = 1
    result = reward_probe(model)
    assert result["sample_size"] == 128
    assert result["timeout_count"] == 1
    assert result["true_terminal_count"] == 1
    assert result["early_true_terminal_count"] == 1
    assert result["conditioned_min_max"][0] == result["terminal_conditioned_min_max"][0]
