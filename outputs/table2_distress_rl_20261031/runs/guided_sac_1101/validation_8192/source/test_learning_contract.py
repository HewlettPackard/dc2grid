import json

# isort: off
import numpy as np
import pytest

from common import contracts
from controllers import CoolingFirstController, FullResidualMacro
from fast_task import StandaloneTask
from history_policy import HistorySACController, OwnHistory, hardware_action

from baselines.rule_based_macro import RuleBasedMacroController
from c2g_env import C2GMacroEnv
# isort: on


def test_own_history_contains_only_observation_action_and_clock():
    history = OwnHistory()
    observation = np.arange(18, dtype=np.float32) / 20
    first = history.features(observation)
    np.testing.assert_array_equal(first[:18], observation)
    np.testing.assert_array_equal(
        first[18:22], [1, np.float32(0.7), np.float32(0.7), 0]
    )
    assert first[23] == 1
    history.last_action = np.array([1, 0.9, 0.2, -0.4], dtype=np.float32)
    second = history.features(observation)
    np.testing.assert_array_equal(second[18:22], history.last_action)
    assert second[23] == pytest.approx(1 - 1 / 17280, abs=1e-7)
    history.reset()
    np.testing.assert_array_equal(history.features(observation), first)


def test_standalone_policy_does_not_project_to_teacher():
    class Model:
        @staticmethod
        def predict(observation, deterministic=True):
            assert observation.shape == (24,)
            return np.array([0.9, 0.4, -0.6], dtype=np.float32), None

    policy = HistorySACController(Model())
    action, _ = policy.predict(np.zeros(18, dtype=np.float32))
    np.testing.assert_array_equal(
        action, np.array([1, 0.9, 0.4, -0.6], dtype=np.float32)
    )


@pytest.mark.parametrize("action", [[0.1, 0.7, 0], [0.7, 1.01, 0], [0.7, 0.7, 2]])
def test_invalid_neural_command_fails(action):
    with pytest.raises(ValueError, match="outside"):
        hardware_action(action)


def test_rule_macro_training_matches_evaluation_cadence():
    protocol = {**contracts(), "n_envs": 1}
    cell = {"start_day": 16, "regd_load_bias": 0.5, "seed": 710048}
    task = StandaloneTask(protocol, seed=1, fixed_cell=cell)
    native = np.array([1, 0.7, 0.65, 0.1], dtype=np.float32)
    evaluation_infos = []

    def callback(pre, action, post, reward, done, info):
        evaluation_infos.append((pre.copy(), info))

    core = C2GMacroEnv(
        scenario="default",
        simulation_mode="physical_v1",
        config_path=protocol["config_path"],
        start_day=16,
        regd_load_bias=0.5,
        inner_action_fn=lambda obs, macro: native.copy(),
        sub_step_callback=callback,
    )
    try:
        observations, _ = task.reset()
        training_rows = []
        for _ in range(360):
            pre = observations[:18].copy()
            observations, reward, term, trunc, info = task.step(native[1:])
            training_rows.append((pre, reward, info))
            assert not term and not trunc
        macro = RuleBasedMacroController()
        obs, _ = core.reset(seed=cell["seed"])
        for _ in range(2):
            action, _ = macro.predict(obs)
            obs, _, term, trunc, _ = core.step(action)
            assert not term and not trunc
        assert len(evaluation_infos) == len(training_rows) == 360
        for (expected_obs, expected), (actual_obs, reward, actual) in zip(
            evaluation_infos, training_rows, strict=True
        ):
            np.testing.assert_array_equal(actual_obs, expected_obs)
            assert reward == expected["reward"]
            assert actual["tracking_err_kw"] == expected["tracking_err_kw"]
            assert actual["bess_soc"] == expected["bess_soc"]
    finally:
        core.close()
        task.close()


def test_full_finite_episode_journal_and_repeated_reset():
    protocol = {**contracts(), "n_envs": 1}
    cell = {"start_day": 16, "regd_load_bias": 0.5, "seed": 710048}
    task = StandaloneTask(protocol, seed=1, fixed_cell=cell)
    teacher = CoolingFirstController()
    try:
        first, _ = task.reset()
        observation = first.copy()
        for _ in range(17280):
            action, _ = teacher.predict(observation[:18])
            observation, _, terminated, truncated, info = task.step(action[1:])
        assert terminated and not truncated
        assert observation[-1] == 0
        assert info["service_truncated"] and not info["physical_terminated"]
        record = json.loads(json.dumps(info["episode_result"], allow_nan=False))
        assert record["complete"] and record["steps"] == 17280
        task.fast._episode_ticks = 18720
        second, _ = task.reset()
        assert task.fast._episode_ticks == 17280
        np.testing.assert_array_equal(first, second)
    finally:
        task.close()


def test_price_taker_sees_initial_posted_offer():
    protocol = contracts()
    core = C2GMacroEnv(
        scenario="default",
        simulation_mode="physical_v1",
        config_path=protocol["config_path"],
        start_day=16,
    )
    try:
        observation, _ = core.reset(seed=710048)
        assert observation[18] * 30 >= 5
        action, _ = FullResidualMacro().predict(observation)
        assert float(action[0]) * 30 <= core._pending_offer["residual_mw"]
    finally:
        core.close()


def test_fixed_condition_cannot_bypass_training_allowlist():
    protocol = {**contracts(), "n_envs": 1}
    cell = {"start_day": 72, "regd_load_bias": 0.5, "seed": 710048}
    task = StandaloneTask(protocol, seed=1, fixed_cell=cell)
    try:
        with pytest.raises(ValueError, match="training allowlist"):
            task.reset()
        assert task.episode_ordinal == 0
    finally:
        task.close()


def test_checkpoint_with_different_action_units_is_rejected(tmp_path):
    import gymnasium as gym
    from common import file_hash
    from evaluate import make_hardware
    from fit_bc import SpaceOnlyEnv
    from history_policy import ACTION_LOW, FEATURE_CONTRACT
    from stable_baselines3 import SAC

    env = SpaceOnlyEnv(24)
    env.action_space = gym.spaces.Box(
        ACTION_LOW, np.array([0.9, 0.9, 0.9], dtype=np.float32)
    )
    model = SAC(
        "MlpPolicy",
        env,
        device="cpu",
        seed=17,
        policy_kwargs={"net_arch": [16, 16]},
    )
    model.c2g_context = {
        "simulation_mode": "physical_v1",
        "feature_contract": FEATURE_CONTRACT,
    }
    path = tmp_path / "wrong_bounds.zip"
    model.save(path)
    with pytest.raises(ValueError, match="identity mismatch"):
        make_hardware(
            {
                "hardware": "history_sac",
                "checkpoint": {"path": str(path), "sha256": file_hash(path)},
            },
            contracts(),
        )


def test_unfingerprinted_history_code_is_rejected_before_rollout(tmp_path):
    from evaluate import run_panel

    with pytest.raises(ValueError, match="fingerprinted"):
        run_panel(
            {"code_hashes": {}},
            [{"hardware": "history_sac"}],
            [],
            tmp_path / "must_not_exist",
        )
    assert not (tmp_path / "must_not_exist").exists()
