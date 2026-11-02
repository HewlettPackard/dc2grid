from __future__ import annotations

import pytest

# isort: off
from train_low import (
    actor_identity,
    attach_update_counters,
    warmup_and_calibrate,
)
from ppo_adapter import LatentActionEnv
from test_ppo_adapter import PhysicalFixture
from economy_context import cap_cpu
from evaluate_low import family_means
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
# isort: on


@pytest.mark.parametrize("algorithm", ("ppo", "sac"))
def test_native_critic_calibration_preserves_actor_and_counts_actual_updates(
    algorithm,
    tmp_path,
):
    cap_cpu()
    fixture = PhysicalFixture()
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            LatentActionEnv(fixture),
            n_steps=8,
            batch_size=4,
            n_epochs=3,
            ent_coef=0.0,
            learning_rate=1e-4,
            device="cpu",
            seed=73,
            policy_kwargs={
                "net_arch": {"pi": [16, 16], "vf": [16, 16]},
                "activation_fn": torch.nn.ReLU,
            },
        )
    else:
        model = SAC(
            "MlpPolicy",
            fixture,
            learning_starts=16,
            batch_size=4,
            buffer_size=128,
            train_freq=1,
            gradient_steps=1,
            ent_coef=1e-5,
            learning_rate=1e-4,
            device="cpu",
            seed=73,
            policy_kwargs={"net_arch": [16, 16]},
        )
    model.set_logger(configure(str(tmp_path / algorithm), ["csv"]))
    handles = attach_update_counters(model, algorithm)
    before = actor_identity(model, algorithm)
    result = warmup_and_calibrate(
        model,
        algorithm,
        warmup_steps=16,
        critic_updates=8,
    )
    assert result["actor_unchanged"] == before
    assert actor_identity(model, algorithm) == before
    assert model.c2g_update_counts == {"actor": 0, "critic": 8, "critic_only": 8}
    assert model.num_timesteps == 16
    model.learn(16, reset_num_timesteps=False)
    assert model.num_timesteps == 32
    assert model.c2g_update_counts["actor"] > 0
    assert model.c2g_update_counts["critic"] == model.c2g_update_counts["actor"] + 8
    for handle in handles:
        handle.remove()
    model.get_env().close()


def test_missing_initializer_cannot_produce_a_survivor_family_mean():
    episode = {
        "policy": {"family": "native_ppo"},
        "complete": True,
        "qualified": True,
        "recovery": {"complete": True},
    }
    result = family_means(
        {"only_one_of_four": [episode]},
        {"native_ppo": 4},
    )["native_ppo"]
    assert result["model_count"] == 1
    assert not result["complete_cohort"]
    assert result["strata"] == {}
