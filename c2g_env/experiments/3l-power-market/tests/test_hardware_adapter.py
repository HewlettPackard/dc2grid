from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from baselines.rule_based_mpc import RuleBasedController
from c2g_env import env_high_level
from c2g_env.obs_indices import Fast


@pytest.fixture
def make_env(modules, config, ireland_data):
    environments = []

    def create(inner="rule_based", model_dir=None):
        cfg = deepcopy(config)
        cfg["controllers"].update(inner=inner, model_dir=model_dir)
        start = pd.Timestamp("2025-01-01")
        env = modules.env.ThreeLevelEnv(
            cfg,
            ireland_data,
            start,
            start + 2 * modules.market.SLOT,
            start + modules.market.SLOT,
            lambda ledger: SimpleNamespace(
                start=lambda controllers: None,
                advance_to=lambda when: None,
                close=lambda: None,
            ),
        )
        environments.append(env)
        env.reset(seed=0)
        return env

    yield create
    for env in environments:
        env.close()


def commit(modules, env, target_mw, price=-20.0):
    at = env.now.floor("15min")
    end = at + modules.market.SLOT
    env.ledger.record_trade(
        modules.market.Trade("BM", "target", at, end, target_mw, price, at)
    )
    env.ledger.record_price("BM", at, end, price, at)


def test_adapter_maps_actual_physics_and_3l_context(modules, make_env):
    env = make_env()
    commit(modules, env, env.plant.state.facility_mw - 2.0)
    state = asdict(env.plant.state)
    tick = env.plant.workload._tick
    obs, audit = env.hardware.observe(env.now)
    assert obs.dtype == np.float32 and obs.shape == (18,)
    assert obs[Fast.TEMP_A] == pytest.approx(
        env.plant.state.temp_a_c / env.plant.thermal.T_safe
    )
    assert obs[Fast.SOC] == env.plant.state.soc
    assert obs[Fast.IS_SPIKE] == float(env.plant.workload_state.is_spike_active)
    assert obs[Fast.FREQ_DEV] == 0.0 and obs[Fast.VPCC] == 1.0
    assert obs[Fast.LMP] == pytest.approx(-20.0 / 200.0)
    assert audit["price_source"] == "BM_clearing"
    assert audit["requested_reduction_mw"] == pytest.approx(2.0)
    assert audit["represented_reduction_mw"] == pytest.approx(2.0)
    assert audit["ambient_c"] == env.data.context(env.now)["ambient_c"]
    assert obs[Fast.GRID_LOAD] == pytest.approx(
        env.data.context(env.now)["system_demand_mw"] / env.data.demand_scale_mw
    )
    assert asdict(env.plant.state) == state and env.plant.workload._tick == tick


def test_rule_controller_output_is_capped_by_operating_plan(modules, make_env):
    env = make_env()
    commit(modules, env, env.plant.state.facility_mw - 2.0)
    env.controllers.plan.set(env.now, env.now + modules.market.SLOT, 0.25)
    obs, _ = env.hardware.observe(env.now)
    expected = RuleBasedController(
        committed_mw_max=env.hardware.context._committed_mw_max,
        bess_p_max_mw=env.plant.battery.P_MAX_MW,
    ).predict(obs)[0]
    action = env.hardware_action()
    assert expected[0] == 1.0
    assert action[0] == pytest.approx(0.25)
    np.testing.assert_array_equal(action[1:], expected[1:])
    assert action[3] == pytest.approx(2.0 / env.plant.battery.P_MAX_MW)
    assert env.hardware.last_decision["inner_throttle"] == 1.0
    assert env.hardware.last_decision["schedule_cap"] == 0.25
    assert env.hardware.capped_steps == 1
    assert isinstance(env.hardware.controller, RuleBasedController)
    assert env.hardware.predict_calls == 1
    env.step(action)
    np.testing.assert_array_equal(env.plant.previous_action, action)
    assert env.plant.state.batch_served_mw <= 0.25 * env.plant.batch_capacity_mw


def test_unscheduled_slots_are_uncapped_and_cap_only_lowers(modules, make_env):
    env = make_env()
    action = env.hardware_action()
    assert env.controllers.plan.cap(env.now) == 1.0
    assert action[0] == 1.0
    assert env.hardware.capped_steps == 0
    env.controllers.plan.set(env.now, env.now + modules.market.SLOT, 1.0)
    env.plant.thermal.temp_A = env.plant.thermal.T_safe - 0.1
    env.plant._state = replace(env.plant.state, temp_a_c=env.plant.thermal.temp_A)
    action = env.hardware_action()
    assert action[0] == 0.0
    assert env.hardware.last_decision["inner_throttle"] == 0.0
    assert env.hardware.capped_steps == 0


def test_unpriced_warmup_is_not_zero_commitment_and_cache_updates(modules, make_env):
    env = make_env()
    _, before = env.hardware.observe(env.now)
    assert before["target_import_mw"] is None
    assert before["price_source"] == "supplied_IDM_forecast_warmup"
    commit(modules, env, 10.0, price=0.0)
    _, after = env.hardware.observe(env.now)
    assert after["target_import_mw"] == 10.0
    assert after["price_eur_mwh"] == 0.0
    assert after["price_source"] == "BM_clearing"
    revision = env.ledger.revision
    commit(modules, env, 10.0, price=0.0)
    assert env.ledger.revision == revision


def test_accepted_position_is_available_without_a_price_row(modules, make_env):
    env = make_env()
    env.ledger.record_trade(
        modules.market.Trade(
            "IDM", "accepted", env.now, env.now + modules.market.SLOT, 10.0, 80, env.now
        )
    )
    _, audit = env.hardware.observe(env.now)
    assert audit["target_import_mw"] == 10.0
    assert audit["price_source"] == "supplied_IDM_forecast_warmup"


def test_missing_scored_price_does_not_use_forecast(make_env):
    env = make_env()
    with pytest.raises(RuntimeError, match="No live market price"):
        env.hardware.observe(env.score_start)


def test_adapter_does_not_construct_legacy_environment_or_weather(
    make_env, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("The controller adapter must not construct legacy context")

    monkeypatch.setattr("c2g_env.env_low_level.C2GFastEnv.__init__", forbidden)
    monkeypatch.setattr("c2g_env.env_low_level.MacroGridSignal.__init__", forbidden)
    monkeypatch.setattr("c2g_env.env_low_level.WeatherLoader.__init__", forbidden)
    env = make_env()
    env.hardware_action()
    assert env.hardware.last_decision["ambient_c"] == 7.4164124


@pytest.mark.parametrize(
    "inner", ("rule_based", "pid", "bang_bang", "mpc_fast", "random", "fixed")
)
def test_existing_classical_controller_families_execute(make_env, inner):
    env = make_env(inner)
    action = env.hardware_action()
    assert action.shape == (4,)
    assert env.action_space.contains(action)
    env.step(action)
    assert env.fast_steps == 1


def test_fixed_action_uses_existing_macro_defaults(modules, make_env, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed actions must not construct legacy environments")

    monkeypatch.setattr("c2g_env.env_low_level.C2GFastEnv.__init__", forbidden)
    monkeypatch.setattr(env_high_level.C2GMacroEnv, "__init__", forbidden)
    env = make_env("fixed")
    commit(modules, env, env.plant.state.facility_mw - 20.0)
    env.controllers.plan.set(env.now, env.now + modules.market.SLOT, 1.0)
    expected = np.array(
        [
            env_high_level._DEFAULT_THROTTLE,
            env_high_level._DEFAULT_PUMP,
            env_high_level._DEFAULT_HVAC,
            env_high_level._DEFAULT_BESS,
        ],
        dtype=np.float32,
    )
    action = env.hardware_action()
    np.testing.assert_array_equal(action, expected)
    env.step(action)
    np.testing.assert_array_equal(env.plant.previous_action, expected)
    action.fill(0)
    np.testing.assert_array_equal(env.hardware_action(), expected)
    assert env.hardware.manifest()["implementation"].endswith(".FixedActionController")


def test_thermal_protection_still_comes_from_c2g_rule_policy(make_env):
    env = make_env()
    env.plant._state = replace(env.plant.state, temp_a_c=env.plant.thermal.T_safe)
    action = env.hardware_action()
    assert action[0] == 0.0
    assert action[1] == action[2] == 1.0


def test_missing_checkpoint_and_unknown_controller_are_errors(make_env, tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        make_env("ppo", str(tmp_path))
    with pytest.raises(ValueError, match="Unknown inner"):
        make_env("not_a_controller")
    with pytest.raises(ValueError, match="only valid"):
        make_env("rule_based", str(tmp_path))
    with pytest.raises(ValueError, match="only valid"):
        make_env("fixed", str(tmp_path))


@pytest.mark.parametrize("algo", ("ppo", "sac"))
def test_learned_controller_loading_and_normalization(
    make_env, tmp_path, algo, monkeypatch
):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    source = make_env()
    wrapped = gym.wrappers.TransformObservation(
        source,
        lambda observation: source.hardware.observe(source.now)[0],
        observation_space=source.hardware.observation_space,
    )
    vector = DummyVecEnv([lambda: wrapped])
    normalizer = None
    if algo == "ppo":
        normalizer = VecNormalize(vector, training=False)
        normalizer.obs_rms.mean = np.full(18, 0.25)
        normalizer.obs_rms.var = np.full(18, 4.0)
        model = PPO(
            "MlpPolicy",
            normalizer,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            device="cpu",
            seed=0,
        )
        normalizer.save(str(tmp_path / "vec_normalize.pkl"))
    else:
        model = SAC(
            "MlpPolicy",
            vector,
            buffer_size=8,
            batch_size=2,
            policy_kwargs={"net_arch": [8]},
            device="cpu",
            seed=0,
        )
    # Untrained checkpoint fixtures validate loading, not learned-policy performance.
    model.save(str(tmp_path / "final_model"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Normalization must not construct the legacy environment")

    monkeypatch.setattr("c2g_env.env_low_level.C2GFastEnv.__init__", forbidden)
    env = make_env(algo, str(tmp_path))
    obs, _ = env.hardware.observe(env.now)
    reference = obs if normalizer is None else normalizer.normalize_obs(obs)
    np.testing.assert_array_equal(
        env.hardware_action(), model.predict(reference, deterministic=True)[0]
    )
    assert env.hardware.manifest()["checkpoint"] is not None
    assert env.hardware.manifest()["checkpoint"]["saved_seed"] == model.seed
    assert env.hardware.manifest()["checkpoint"]["saved_num_timesteps"] == 0
    assert (env.hardware.normalization is not None) == (algo == "ppo")
    env.step(env.hardware_action())
    assert env.fast_steps == 1
    if algo == "ppo":
        assert not env.hardware.controller._obs_normalizer.training
        assert not env.hardware.controller._obs_normalizer.norm_reward
        (tmp_path / "vec_normalize.pkl").unlink()
        with pytest.raises(FileNotFoundError, match="normalization"):
            make_env(algo, str(tmp_path))
