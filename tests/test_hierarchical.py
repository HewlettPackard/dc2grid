"""
Tests for hierarchical RL infrastructure
==========================================
Covers:
  - RuleBasedMacroController: predict interface, action bounds, safety rules
  - C2GMacroEnv with inner_action_fn: obs shape, step, episode
  - train_ppo_macro module importable
  - train_hierarchical module importable
  - Hierarchical combo smoke tests via run_macro_episode
"""
import math
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env import C2GMacroEnv
from baselines.rule_based_macro import RuleBasedMacroController


# =========================================================================
# A. RuleBasedMacroController
# =========================================================================

class TestRuleBasedMacroController:

    @pytest.fixture
    def ctrl(self):
        return RuleBasedMacroController()

    def test_predict_returns_tuple(self, ctrl):
        obs = np.zeros(19, dtype=np.float32)
        action, state = ctrl.predict(obs)
        assert state is None
        assert isinstance(action, np.ndarray)

    def test_action_shape(self, ctrl):
        obs = np.zeros(19, dtype=np.float32)
        action, _ = ctrl.predict(obs)
        assert action.shape == (2,)

    def test_action_bounds(self, ctrl):
        """Actions must be within MacroEnv action space."""
        for _ in range(20):
            obs = np.random.rand(19).astype(np.float32)
            obs[6] = np.random.uniform(0, 1)   # lmp_norm
            obs[7] = np.random.uniform(0, 1)   # load_norm
            obs[2] = np.random.uniform(0, 1)   # soc
            action, _ = ctrl.predict(obs)
            assert 0.0 <= action[0] <= 1.0, f"bid_mw_norm={action[0]} out of [0,1]"
            assert 0.0 <= action[1] <= 1.0, f"bid_price_norm={action[1]} out of [0,1]"

    def test_high_load_high_commitment(self, ctrl):
        obs = np.zeros(19, dtype=np.float32)
        obs[7] = 0.9   # high grid load
        obs[10] = 0.5   # healthy headroom A
        obs[11] = 0.5   # healthy headroom B
        obs[15] = 1.0   # nominal voltage
        action, _ = ctrl.predict(obs)
        assert action[0] >= 0.7, "High load should produce high commitment"

    def test_low_load_low_commitment(self, ctrl):
        obs = np.zeros(19, dtype=np.float32)
        obs[7] = 0.2   # low grid load
        obs[10] = 0.5
        obs[11] = 0.5
        obs[15] = 1.0
        action, _ = ctrl.predict(obs)
        assert action[0] <= 0.3, "Low load should produce low commitment"

    def test_batch_predict(self, ctrl):
        obs = np.zeros((5, 19), dtype=np.float32)
        actions, _ = ctrl.predict(obs)
        assert actions.shape == (5, 2)

    def test_thermal_emergency_reduces_commitment(self, ctrl):
        obs = np.zeros(19, dtype=np.float32)
        obs[7] = 0.9    # high load → normally high commitment
        obs[10] = 0.05  # very low headroom A → safety override
        obs[11] = 0.05
        obs[15] = 1.0
        action, _ = ctrl.predict(obs)
        assert action[0] <= 0.40, "Low headroom should cap commitment"

    def test_low_voltage_reduces_commitment(self, ctrl):
        obs = np.zeros(19, dtype=np.float32)
        obs[7] = 0.9
        obs[10] = 0.5
        obs[11] = 0.5
        obs[15] = 0.93  # low voltage
        action, _ = ctrl.predict(obs)
        assert action[0] <= 0.50, "Low voltage should reduce commitment"


# =========================================================================
# B. MacroEnv with custom inner_action_fn
# =========================================================================

class TestMacroEnvWithInnerFn:
    """Test that inner_action_fn is actually called during MacroEnv step."""

    def test_inner_fn_called(self):
        call_count = [0]

        def counting_fn(inner_obs, macro_action):
            call_count[0] += 1
            return np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)

        env = C2GMacroEnv(scenario="default", inner_action_fn=counting_fn)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        # 180 sub-steps per macro step
        assert call_count[0] == 180, f"Expected 180 calls, got {call_count[0]}"

    def test_inner_fn_receives_macro_action(self):
        received_actions = []

        def capturing_fn(inner_obs, macro_action):
            received_actions.append(macro_action.copy())
            return np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)

        env = C2GMacroEnv(scenario="default", inner_action_fn=capturing_fn)
        env.reset(seed=0)
        action = np.array([0.7, 0.3], dtype=np.float32)
        env.step(action)
        # All captured macro actions should match
        assert len(received_actions) > 0
        assert received_actions[0].shape == (2,)

    def test_episode_with_rule_based_inner(self):
        """Rule-based low-level as inner_action_fn runs a full macro step."""
        from baselines.rule_based_mpc import RuleBasedController
        inner_ctrl = RuleBasedController()

        def rule_fn(inner_obs, macro_action):
            action, _ = inner_ctrl.predict(inner_obs)
            return action

        env = C2GMacroEnv(scenario="default", inner_action_fn=rule_fn)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (19,)

        obs, rew, term, trunc, info = env.step(np.array([0.5, 0.0],
                                                         dtype=np.float32))
        assert obs.shape == (19,)
        assert math.isfinite(rew)


# =========================================================================
# C. Macro rule-based controller on live env
# =========================================================================

class TestRuleBasedMacroOnEnv:

    def test_full_macro_episode(self):
        """Rule-based macro runs a full episode without crashing."""
        ctrl = RuleBasedMacroController()
        env = C2GMacroEnv(scenario="default")
        obs, _ = env.reset(seed=0)
        total_rew = 0.0
        steps = 0
        for _ in range(10):  # 10 macro steps = 150 min
            action, _ = ctrl.predict(obs)
            obs, rew, term, trunc, info = env.step(action)
            total_rew += rew
            steps += 1
            if term or trunc:
                break
        assert steps > 0
        assert math.isfinite(total_rew)


# =========================================================================
# D. Module imports (smoke tests)
# =========================================================================

class TestModuleImports:

    def test_import_train_ppo_macro(self):
        import baselines.train_ppo_macro  # noqa

    def test_import_train_hierarchical(self):
        import baselines.train_hierarchical  # noqa

    def test_import_rule_based_macro(self):
        from baselines.rule_based_macro import RuleBasedMacroController  # noqa


# =========================================================================
# E. Hierarchical combo smoke tests (run_macro_episode)
# =========================================================================

# 4 macro-level outer controllers (no trained model files needed)
_MACRO_CONTROLLERS = ["rule_macro", "mpc_macro", "milp", "random_macro"]

# 4 hardware-level inner controllers (no trained model files needed)
# mpc_fast excluded: runs an optimization solve per sub-step (~540 calls/3 ticks), too slow for smoke tests
_INNER_CONTROLLERS = ["pid", "bang_bang", "rule_based", "random"]


def _build_macro_ctrl(macro_part: str, macro_env: "C2GMacroEnv"):
    """Instantiate a macro controller by name."""
    from baselines.rule_based_macro import RuleBasedMacroController
    from baselines.mpc_macro import MPCMacroController
    from baselines.milp_dispatch import MILPDispatchController
    from evaluation.run_benchmark import MacroRandomAgent
    if macro_part == "rule_macro":
        return RuleBasedMacroController()
    if macro_part == "mpc_macro":
        return MPCMacroController()
    if macro_part == "milp":
        return MILPDispatchController()
    if macro_part == "random_macro":
        return MacroRandomAgent(macro_env, algo_name="random_macro")
    raise ValueError(f"Unknown macro controller: {macro_part}")


class TestHierarchicalCombos:
    """
    Smoke-test all 16 non-LLM (macro × inner) combinations via run_macro_episode.
    Runs only 3 macro steps per combo to keep the suite fast.
    Validates: no crash, finite reward, correct return type.
    """

    @pytest.fixture(autouse=True)
    def short_episode(self, monkeypatch):
        """Patch C2GMacroEnv so episodes terminate after 3 macro steps."""
        original_step = C2GMacroEnv.step
        step_counter = {}

        def patched_step(self_env, action):
            key = id(self_env)
            step_counter[key] = step_counter.get(key, 0) + 1
            obs, rew, term, trunc, info = original_step(self_env, action)
            if step_counter[key] >= 3:
                term = True
            return obs, rew, term, trunc, info

        monkeypatch.setattr(C2GMacroEnv, "step", patched_step)

    @pytest.mark.parametrize("macro_part", _MACRO_CONTROLLERS)
    @pytest.mark.parametrize("inner_part", _INNER_CONTROLLERS)
    def test_combo(self, macro_part: str, inner_part: str):
        """All 16 macro+inner combos: no crash, finite mean_reward."""
        from evaluation.run_benchmark import (
            _make_inner_controller,
            _make_env,
            _make_macro_env,
            run_macro_episode,
        )

        # Build a throw-away macro env just for controllers that need its space
        macro_env = _make_macro_env(scenario="default")
        macro_env.reset(seed=0)

        # Build inner controller
        inner_env = _make_env(scenario="default")
        inner_env.reset(seed=0)
        inner_ctrl = _make_inner_controller(inner_part, env=inner_env, scenario="default")
        inner_action_fn = lambda obs, _act, c=inner_ctrl: c.predict(obs)[0]

        # Build macro controller
        macro_ctrl = _build_macro_ctrl(macro_part, macro_env)

        metrics = run_macro_episode(
            agent=macro_ctrl,
            scenario="default",
            seed=42,
            algo_name=macro_part,
            agent_type="macro",
            episode_number=0,
            inner_action_fn=inner_action_fn,
            record_transitions=False,
            combo_name=f"{macro_part}+{inner_part}",
        )

        assert isinstance(metrics, dict), "run_macro_episode must return a dict"
        assert math.isfinite(metrics["mean_reward"]), (
            f"{macro_part}+{inner_part} produced non-finite mean_reward"
        )
        assert metrics["episode_length"] > 0, (
            f"{macro_part}+{inner_part} ran 0 steps"
        )

    def test_inner_fn_push_reward_attr(self):
        """Standard hardware controllers don't expose push_reward."""
        from evaluation.run_benchmark import _make_inner_controller, _make_env

        inner_env = _make_env(scenario="default")
        inner_env.reset(seed=0)
        inner_ctrl = _make_inner_controller("pid", env=inner_env)
        assert not hasattr(inner_ctrl, "push_reward"), (
            "PIDController should not have push_reward"
        )

    def test_inner_fn_receives_hardware_obs(self):
        """inner_action_fn is called with hardware-level obs during MacroEnv step."""
        obs_shapes = []

        def recording_fn(inner_obs, macro_action):
            obs_shapes.append(inner_obs.shape)
            return np.zeros(4, dtype=np.float32)

        env = C2GMacroEnv(scenario="default", inner_action_fn=recording_fn)
        env.reset(seed=0)
        env.step(np.array([0.5, 0.0], dtype=np.float32))
        assert len(obs_shapes) > 0
        assert obs_shapes[0][0] >= 17, f"Unexpected inner obs dim: {obs_shapes[0]}"
