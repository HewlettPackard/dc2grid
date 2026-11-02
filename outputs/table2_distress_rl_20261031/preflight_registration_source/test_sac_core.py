import numpy as np
import torch
from fit_bc import SpaceOnlyEnv, parameter_digest
from sac_core import GuidedSAC
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure


def model_with_replay(kind):
    model = kind(
        "MlpPolicy",
        SpaceOnlyEnv(24),
        seed=71,
        device="cpu",
        ent_coef=1e-5,
        batch_size=32,
        buffer_size=1024,
        policy_kwargs={"net_arch": [32, 32]},
    )
    model.set_logger(configure(format_strings=[]))
    rng = np.random.default_rng(21)
    for _ in range(128):
        model.replay_buffer.add(
            rng.normal(size=(1, 24)).astype(np.float32),
            rng.normal(size=(1, 24)).astype(np.float32),
            rng.uniform(-1, 1, size=(1, 3)).astype(np.float32),
            np.array([rng.normal()], dtype=np.float32),
            np.array([rng.uniform() < 0.1]),
            [{}],
        )
    return model


def test_zero_regularizer_matches_sb3_parameter_updates():
    standard, guided = model_with_replay(SAC), model_with_replay(GuidedSAC)
    assert parameter_digest(standard.policy) == parameter_digest(guided.policy)
    for model in (standard, guided):
        np.random.seed(421)
        torch.manual_seed(421)
        model.train(3, 32)
    assert parameter_digest(standard.policy) == parameter_digest(guided.policy)
    assert standard._n_updates == guided._n_updates == guided.actor_updates == 3


def test_calibration_does_not_update_actor_or_its_optimizer():
    model = model_with_replay(GuidedSAC)
    before = parameter_digest(model.actor)
    result = model.calibrate_critics(4)
    assert result["actor_digest_unchanged"] == before
    assert not model.actor.optimizer.state
    assert model.actor_updates == 0 and model._n_updates == 4
    assert model.critic.optimizer.state


def test_guided_update_and_checkpoint_roundtrip(tmp_path):
    model = model_with_replay(GuidedSAC)
    rng = np.random.default_rng(189)
    observations = rng.normal(size=(128, 24)).astype(np.float32)
    targets = np.tile(np.array([0.7, 0.7, 0], dtype=np.float32), (128, 1))
    model.attach_demonstrations(observations, targets, 731)
    model.calibrate_critics(4)
    calibration = model.calibrate_regularization()
    model.bc_weight = calibration["bc_weight"]
    actor_before = parameter_digest(model.actor)
    model.train(3, 32)
    assert model.actor_updates == 3 and model._n_updates == 7
    assert parameter_digest(model.actor) != actor_before
    path = tmp_path / "model.zip"
    model.save(path)
    restored = GuidedSAC.load(path, device="cpu")
    assert restored.demo_observations is None
    assert parameter_digest(restored.policy) == parameter_digest(model.policy)
    assert restored.actor_updates == 3
