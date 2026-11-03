import csv
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from stable_baselines3 import SAC

from baselines.macro_metrics_callback import C2GMacroMetricsCallback
from baselines.sac_provenance import bind_training_context
from baselines.train_hierarchical_sac import _make_sac_inner_fn
from baselines.train_rule_macro_sac import make_env_fn
from c2g_env.experiments.identity import check_policy_semantics
from evaluation.run_benchmark import load_sb3_agent


def test_unknown_checkpoint_requires_explicit_mismatch():
    model = SimpleNamespace()
    assert check_policy_semantics(model, "legacy_v0") == "legacy_v0"
    with pytest.raises(ValueError, match="Checkpoint semantics"):
        check_policy_semantics(model, "physical_v1")
    with pytest.warns(RuntimeWarning, match="frozen-policy diagnosis"):
        assert (
            check_policy_semantics(model, "physical_v1", allow_mismatch=True)
            == "legacy_v0"
        )


def test_short_sac_update_checkpoint_and_frozen_partner(tmp_path):
    env = make_env_fn("default", 100, simulation_mode="physical_v1")()
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=1,
        batch_size=2,
        buffer_size=16,
        policy_kwargs={"net_arch": [8, 8]},
        device="cpu",
        seed=100,
    )
    bind_training_context(model, env, tmp_path)
    model.learn(total_timesteps=4)
    assert model._n_updates > 0
    path = tmp_path / "final_model.zip"
    model.save(path)
    agent = load_sb3_agent(
        "sac", "default", 100, str(path), simulation_mode="physical_v1"
    )
    assert agent.checkpoint["simulation_mode"] == "physical_v1"
    assert agent._model.c2g_context["training_seed"] == 100
    frozen = _make_sac_inner_fn(str(path), "physical_v1")
    obs, _ = env.reset(seed=101)
    np.testing.assert_array_equal(frozen(obs, np.zeros(2)), agent.predict(obs)[0])
    assert frozen.checkpoint["sha256"] == agent.checkpoint["sha256"]
    with pytest.raises(ValueError, match="Checkpoint semantics"):
        _make_sac_inner_fn(str(path), "legacy_v0")
    env.close()


def test_macro_callback_weights_fast_steps_and_records_faults(tmp_path):
    callback = C2GMacroMetricsCallback(csv_path=tmp_path / "metrics.csv")
    callback.model = SimpleNamespace(logger=MagicMock())
    base = {
        "temp_A_max": 34.0,
        "temp_B_max": 32.0,
        "bess_soc_end": 0.4,
        "regulation_revenue": 3.0,
        "electricity_cost": 50.0,
        "terminated": False,
        "truncated": False,
    }
    for steps, squares, done in [(2, 2_000_000, False), (1, 4_000_000, True)]:
        info = {
            **base,
            "sub_steps_run": steps,
            "tracking_error_sq_sum": squares,
            "terminated": done,
        }
        callback.locals = {"rewards": [1.0], "infos": [info], "dones": [done]}
        assert callback._on_step()
    with (tmp_path / "metrics.csv").open() as handle:
        result = next(csv.DictReader(handle))
    assert float(result["tracking_rmse_kw"]) == pytest.approx(np.sqrt(2_000_000))
    assert float(result["hardware_steps"]) == 3
    assert float(result["electricity_cost"]) == 100
    assert int(result["terminated"]) == 1
    assert float(result["soc_final"]) == 0.4
    assert callback._episodes == {}


@pytest.mark.parametrize("macro", [False, True])
def test_ppo_normalizer_and_space_semantics_survive_save(tmp_path, macro):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from c2g_env import C2GFastEnv, C2GMacroEnv

    env_class = C2GMacroEnv if macro else C2GFastEnv
    vec = VecNormalize(DummyVecEnv([lambda: env_class()]), norm_reward=False)
    obs = vec.reset()
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=2,
        batch_size=2,
        policy_kwargs={"net_arch": [8, 8]},
        device="cpu",
    )
    model.save(tmp_path / "final_model.zip")
    vec.save(tmp_path / "vec_normalize.pkl")
    agent = load_sb3_agent(
        "ppo_macro" if macro else "ppo",
        "default",
        100,
        str(tmp_path),
        simulation_mode="physical_v1",
    )
    assert agent.checkpoint["simulation_mode"] == "physical_v1"
    assert agent._obs_normalizer.observation_space.shape == obs.shape[1:]
    np.testing.assert_array_equal(agent._obs_normalizer.obs_rms.mean, vec.obs_rms.mean)
    vec.close()
    agent._obs_normalizer.close()


def test_low_level_callback_does_not_count_early_fault_as_survival():
    from baselines.metrics_callback import (
        _accumulate_step,
        _compute_episode_metrics,
        _make_episode_buffer,
    )

    buffer = _make_episode_buffer()
    _accumulate_step(
        buffer,
        1.0,
        {
            "tick": 500,
            "episode_ticks": 17280,
            "terminated": True,
            "truncated": False,
        },
    )
    assert _compute_episode_metrics(buffer)["ep/survived"] == 0
