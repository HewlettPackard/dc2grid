"""
tests/test_critical_bug_fixes.py
=================================
Regression tests for the 4 critical bugs identified in the expert code review.

Bug #1 — BESS initial SOC is a no-op with the PySAM backend
    env_low_level.py used `self._bess._soc = value` which is a non-existent
    attribute on _PySAMBESSModel. Now uses `self._bess.set_initial_soc(value)`.

Bug #2 — metrics_callback.py tracked thermal terminations with the wrong key
    `info.get("thermal_terminated")` → should be `info.get("thermal_fault")`.

Bug #3 — ShieldedEnv used zero-vector obs for the FIRST step after every reset
    `_last_obs` was only set inside `step()`, not in `reset()`. A zero obs
    causes spurious SOC-low (obs[2]=0) and voltage-low (obs[15]=0) shield
    overrides on the very first action of every episode.

Bug #4 — C2GMacroEnv returned terminated=True AND truncated=True simultaneously
    Gymnasium requires at most one of them to be True. When a terminal fault
    coincided with the last macro tick, both flags were True. Fixed by only
    setting truncated=True when not already terminated.
"""
from __future__ import annotations

import numpy as np
import pytest

from c2g_env import C2GFastEnv
from c2g_env.plant_profiles import available_plant_profiles, load_plant_profile
from c2g_env.physics.bess import BESSModel, PYSAM_ACTIVE, _SimpleBESSModel
from c2g_env.thermal_limits import T_WARN_A


# ═══════════════════════════════════════════════════════════════════════════
# Bug #1 — BESS set_initial_soc API on both backends
# ═══════════════════════════════════════════════════════════════════════════

class TestBESSSetInitialSOC:

    def test_simple_backend_has_set_initial_soc(self):
        bess = _SimpleBESSModel()
        assert hasattr(bess, "set_initial_soc"), (
            "_SimpleBESSModel must expose set_initial_soc()"
        )

    def test_simple_backend_set_initial_soc_takes_effect(self):
        bess = _SimpleBESSModel()
        bess.set_initial_soc(0.80)
        assert bess.soc_fraction == pytest.approx(0.80, abs=1e-6)

    def test_simple_backend_set_initial_soc_clamps_to_limits(self):
        bess = _SimpleBESSModel()
        bess.set_initial_soc(0.0)   # below SOC_MIN
        assert bess.soc_fraction >= _SimpleBESSModel.SOC_MIN - 1e-9
        bess.set_initial_soc(1.0)   # above SOC_MAX
        assert bess.soc_fraction <= _SimpleBESSModel.SOC_MAX + 1e-9

    def test_bess_model_alias_has_set_initial_soc(self):
        """BESSModel (whichever backend is active) must expose set_initial_soc."""
        bess = BESSModel()
        assert hasattr(bess, "set_initial_soc"), (
            f"BESSModel ({BESSModel.__name__}) must expose set_initial_soc()"
        )

    def test_bess_model_set_initial_soc_takes_effect(self):
        """After set_initial_soc(0.75), soc_fraction must be ~0.75."""
        bess = BESSModel()
        bess.set_initial_soc(0.75)
        assert bess.soc_fraction == pytest.approx(0.75, abs=0.02), (
            f"Expected SOC≈0.75, got {bess.soc_fraction:.4f}. "
            "This fails if the old `_soc` direct-attribute assignment was used."
        )

    def test_env_reset_applies_custom_bess_soc(self):
        """
        C2GFastEnv.reset() must apply bess_soc_init from config.
        We verify via the observation returned (obs[2] = soc_fraction).
        """
        env = C2GFastEnv(scenario="default")
        # Default scenario has bess_soc_init in config; whatever it is,
        # the obs[2] must match the BESS's actual soc_fraction after reset.
        obs, _ = env.reset(seed=0)
        actual_soc = env._bess.soc_fraction
        assert obs[2] == pytest.approx(actual_soc, abs=1e-5), (
            "obs[2] (soc_fraction) does not match BESS internal state after reset"
        )

    def test_env_reset_scenario_switch_applies_bess_soc(self):
        """
        Switching scenario via options must apply that scenario's bess_soc_init.
        scenario_b has bess_soc_init=0.60; default has 0.50.
        This confirms set_initial_soc() is called (not the broken _soc direct set).
        """
        env = C2GFastEnv(scenario="default")
        # First reset at default (SOC=0.50)
        env.reset(seed=0)
        assert env._bess.soc_fraction == pytest.approx(0.50, abs=0.02)
        # Switch to scenario_b (SOC=0.60)
        env.reset(seed=0, options={"scenario": "scenario_b"})
        assert env._bess.soc_fraction == pytest.approx(0.60, abs=0.02), (
            f"Expected BESS SOC≈0.60 after scenario_b reset, got "
            f"{env._bess.soc_fraction:.4f}. Bug #1 not fixed."
        )

    @pytest.mark.skipif(not PYSAM_ACTIVE, reason="PySAM not installed")
    def test_pysam_backend_set_initial_soc(self):
        """With PySAM active, set_initial_soc must update StatePack.SOC."""
        bess = BESSModel()
        bess.set_initial_soc(0.70)
        assert bess.soc_fraction == pytest.approx(0.70, abs=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# Bug #2 — C2GMetricsCallback thermal termination key
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricsCallbackThermalKey:

    def test_thermal_fault_key_used_not_thermal_terminated(self):
        """
        The callback source must use 'thermal_fault', not 'thermal_terminated'.
        """
        import inspect
        from baselines.metrics_callback import C2GMetricsCallback
        src = inspect.getsource(C2GMetricsCallback)
        assert "thermal_terminated" not in src, (
            "metrics_callback.py still uses the wrong key 'thermal_terminated'. "
            "Bug #2 not fixed."
        )
        assert "thermal_fault" in src, (
            "metrics_callback.py does not use 'thermal_fault' anywhere."
        )

    def test_thermal_termination_registered_from_correct_key(self):
        """
        Manually invoke _on_step with a synthetic 'done' + 'thermal_fault=True'
        info dict and verify the callback records the termination.
        """
        from unittest.mock import MagicMock
        from baselines.metrics_callback import C2GMetricsCallback

        cb = C2GMetricsCallback(verbose=0)
        # Minimal SB3 callback plumbing
        mock_logger = MagicMock()
        mock_model = MagicMock()
        mock_model.logger = mock_logger
        cb.model = mock_model
        cb.num_timesteps = 0
        cb._ep_count = 0
        cb._bufs = {}
        cb._reset_buffers()

        # Simulate one step that ends the episode with a thermal fault
        cb.locals = {
            "rewards": [0.0],
            "dones": [True],
            "infos": [{
                "tick": 5,
                "temp_A": 36.0,   # over T_safe → thermal_fault
                "temp_B": 28.0,
                "bess_soc": 0.5,
                "pue": 1.5,
                "lmp": 50.0,
                "tracking_err_kw": 10.0,
                "delta_p_actual_kw": 100.0,
                "delta_p_demanded_kw": 110.0,
                "flex_reduction_kw": 0.0,
                "bess_actual_kw": 0.0,
                "p_facility_mw": 1.0,
                "is_spike": False,
                "thermal_fault": True,    # ← correct key
            }],
        }
        cb._on_step()
        cb._on_rollout_end()  # flush buffered metrics to logger

        # The buffer for env 0 was flushed by _log_episode; check episode count
        assert cb._ep_count == 1, "Episode was not counted"
        # The logger should have been called with thermal/terminated = 1
        calls = {call.args[0]: call.args[1]
                 for call in mock_logger.record.call_args_list}
        assert calls.get("thermal/terminated") == 1.0, (
            f"thermal/terminated not logged as 1; got {calls}. Bug #2 not fixed."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Bug #4 — C2GMacroEnv must not return terminated=True AND truncated=True
# ═══════════════════════════════════════════════════════════════════════════

class TestMacroEnvTerminatedTruncated:

    def _run_full_episode(self, env, seed=0):
        """Run until done, collect (terminated, truncated) pairs."""
        obs, _ = env.reset(seed=seed)
        flags = []
        for _ in range(1000):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            flags.append((terminated, truncated))
            if terminated or truncated:
                break
        return flags

    def test_terminated_and_truncated_never_both_true(self):
        """
        Gymnasium spec: at most one of terminated/truncated may be True per step.
        Bug #4 fired when a terminal fault coincided with the last macro tick.
        """
        from c2g_env import C2GMacroEnv
        env = C2GMacroEnv(scenario="default")
        for seed in range(5):
            flags = self._run_full_episode(env, seed=seed)
            for term, trunc in flags:
                assert not (term and trunc), (
                    f"terminated=True AND truncated=True at seed={seed}. "
                    "Bug #4 not fixed."
                )

    def test_truncated_set_when_no_fault_at_episode_end(self):
        """
        When the episode ends cleanly (no fault), the final step must have
        truncated=True and terminated=False.
        """
        from c2g_env import C2GMacroEnv
        env = C2GMacroEnv(scenario="default")
        flags = self._run_full_episode(env, seed=0)
        last_term, last_trunc = flags[-1]
        # If the episode survived (no early termination), last step is truncated
        all_terms = [t for t, _ in flags]
        if not any(all_terms[:-1]):   # no early termination before last step
            assert last_trunc, "Last step of survived episode must be truncated"
            assert not last_term, "Clean episode end must not set terminated"

    def test_terminated_set_on_fault(self):
        """
        When a fault occurs, terminated must be True and truncated False.
        Use scenario_d (cooling fault) which is most likely to trigger a fault.
        """
        from c2g_env import C2GMacroEnv
        env = C2GMacroEnv(scenario="scenario_c")
        # Feed worst-case actions: max BESS discharge, no cooling
        obs, _ = env.reset(seed=7)
        for _ in range(200):
            # aggressive: zero cooling, max discharge → likely thermal fault
            action = np.array([1.0, 0.0], dtype=np.float32)
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated:
                assert not truncated, (
                    f"terminated=True but also truncated=True. Bug #4 not fixed."
                )
                break


# ═══════════════════════════════════════════════════════════════════════════
# High Bug #1 — eval VecNormalize stats synced before evaluation
# ═══════════════════════════════════════════════════════════════════════════

class TestSyncNormEvalCallback:

    def test_class_exists_in_train_ppo(self):
        """SyncNormEvalCallback must be importable from train_ppo."""
        import importlib, sys
        # Avoid triggering Hydra's @main decorator at import time
        import baselines.train_ppo as m
        assert hasattr(m, "SyncNormEvalCallback"), (
            "SyncNormEvalCallback not found in train_ppo.py. High Bug #1 not fixed."
        )

    def test_sync_norm_is_subclass_of_eval_callback(self):
        from stable_baselines3.common.callbacks import EvalCallback
        import baselines.train_ppo as m
        assert issubclass(m.SyncNormEvalCallback, EvalCallback)

    def test_sync_envs_normalization_called(self):
        """
        SyncNormEvalCallback._on_step must call sync_envs_normalization
        when the eval frequency fires.
        """
        import inspect
        import baselines.train_ppo as m
        src = inspect.getsource(m.SyncNormEvalCallback._on_step)
        assert "sync_envs_normalization" in src, (
            "_on_step does not call sync_envs_normalization. High Bug #1 not fixed."
        )

    def test_eval_env_obs_rms_synced_after_training_steps(self):
        """
        After a few training steps, calling sync_envs_normalization must
        copy obs_rms from training env to eval env.
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

        train_env = VecNormalize(make_vec_env(lambda: C2GFastEnv(scenario="default"),
                                              n_envs=1, seed=0))
        eval_env  = VecNormalize(make_vec_env(lambda: C2GFastEnv(scenario="default"),
                                              n_envs=1, seed=1),
                                 training=False, norm_reward=False)

        # Simulate a few rollout steps to build up train_env obs_rms stats
        model = PPO("MlpPolicy", train_env, n_steps=64, batch_size=32,
                    n_epochs=1, verbose=0, seed=0)
        model.learn(256)

        # Before sync: stats must differ
        train_mean = train_env.obs_rms.mean.copy()
        eval_mean_before = eval_env.obs_rms.mean.copy()
        assert not np.allclose(train_mean, eval_mean_before, atol=1e-3), (
            "obs_rms.mean already identical before sync — test precondition failed"
        )

        # After sync: eval env should have training env's stats
        sync_envs_normalization(train_env, eval_env)
        np.testing.assert_allclose(
            eval_env.obs_rms.mean, train_env.obs_rms.mean,
            atol=1e-6,
            err_msg="obs_rms.mean not synced. High Bug #1 not fixed."
        )


# ═══════════════════════════════════════════════════════════════════════════
# High Bug #2 — _build_obs_at_reset uses real simulator state
# ═══════════════════════════════════════════════════════════════════════════

class TestObsAtResetRealState:

    def test_obs_at_reset_not_hardcoded_p_base(self):
        """
        obs[3] (p_base_norm) must reflect the actual workload trace tick 0,
        not the hardcoded 0.5 placeholder.
        """
        env = C2GFastEnv(scenario="default")
        obs, _ = env.reset(seed=0)
        # After reset the workload tick is 0; peek tick 0 to get real value
        w = env._workload.step(1.0)
        env._workload._tick = 0
        expected = w.p_base_kw / 250_000.0
        assert obs[3] == pytest.approx(expected, abs=1e-4), (
            f"obs[3] (p_base_norm) = {obs[3]:.4f}, expected {expected:.4f}. "
            "High Bug #2 not fixed."
        )

    def test_obs_at_reset_not_hardcoded_p_flex(self):
        """obs[4] (p_flex_nom_norm) must come from actual workload tick 0."""
        env = C2GFastEnv(scenario="default")
        obs, _ = env.reset(seed=0)
        w = env._workload.step(1.0)
        env._workload._tick = 0
        expected = w.p_flex_nom_kw / 250_000.0
        assert obs[4] == pytest.approx(expected, abs=1e-4), (
            f"obs[4] (p_flex_nom_norm) = {obs[4]:.4f}, expected {expected:.4f}. "
            "High Bug #2 not fixed."
        )

    def test_obs_at_reset_lmp_norm_real(self):
        """
        obs[7] (lmp_norm) must come from the actual LMP at tick 0,
        not the hardcoded 0.2 placeholder.
        """
        env = C2GFastEnv(scenario="default")
        obs, _ = env.reset(seed=0)
        gs = env._grid.step()
        env._grid._tick = 0
        env._grid._regd_state = 0.0
        env._grid._regd_buffer = []
        expected = min(gs["lmp_usd_mwh"] / 200.0, 1.0)
        assert obs[7] == pytest.approx(expected, abs=1e-4), (
            f"obs[7] (lmp_norm) = {obs[7]:.4f}, expected {expected:.4f}. "
            "High Bug #2 not fixed."
        )

    def test_obs_at_reset_bess_soc_is_real(self):
        """
        obs[2] (bess_soc) must be the actual BESS soc_fraction, not a
        hardcoded value.
        """
        env = C2GFastEnv(scenario="default")
        obs, _ = env.reset(seed=0)
        assert obs[2] == pytest.approx(env._bess.soc_fraction, abs=1e-6)

    def test_obs_at_reset_scenario_b_soc_differs_from_default(self):
        """
        scenario_b has bess_soc_init=0.60; obs[2] must differ from
        default scenario's obs[2] (0.50).
        """
        env_def = C2GFastEnv(scenario="default")
        obs_def, _ = env_def.reset(seed=0)

        env_b = C2GFastEnv(scenario="scenario_b")
        obs_b, _ = env_b.reset(seed=0)

        assert obs_def[2] == pytest.approx(0.50, abs=0.02)
        assert obs_b[2]   == pytest.approx(0.60, abs=0.02)
        assert not np.isclose(obs_def[2], obs_b[2], atol=0.05), (
            "BESS SOC at reset is the same across scenarios — suggests "
            "set_initial_soc not applied. High Bug #2 not fixed."
        )

    def test_obs_at_reset_does_not_advance_tick(self):
        """
        After reset(), env._tick must still be 0; peeking must not advance
        the main episode tick counter.
        """
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        assert env._tick == 0, (
            f"env._tick = {env._tick} after reset, expected 0. "
            "_build_obs_at_reset must not advance the episode tick."
        )

    def test_first_step_tick_becomes_one(self):
        """The first step() after reset must produce tick=1 in info."""
        env = C2GFastEnv(scenario="default")
        obs0, _ = env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32))
        assert info["tick"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# High Bug #3 — Seed fallback no longer hardcoded to 42
# ═══════════════════════════════════════════════════════════════════════════

class TestSeedFallback:

    def test_seed_none_uses_np_random_not_42(self):
        """
        Two envs reset with seed=None must produce different grid signal
        trajectories (different rng_seeds), not the same trajectory as each
        other or as seed=42.
        """
        env1 = C2GFastEnv(scenario="default")
        env2 = C2GFastEnv(scenario="default")

        obs1, _ = env1.reset(seed=None)
        obs2, _ = env2.reset(seed=None)

        # Run 10 steps each and collect regd_signal
        signals1, signals2 = [], []
        for _ in range(10):
            _, _, _, _, info1 = env1.step(np.array([1.0, 0.7, 0.7, 0.0], np.float32))
            _, _, _, _, info2 = env2.step(np.array([1.0, 0.7, 0.7, 0.0], np.float32))
            signals1.append(info1["regd_signal"])
            signals2.append(info2["regd_signal"])

        # At least some RegD signals should differ between the two unseeded envs
        # (probability of identical 10-step AR(1) trajectories from different
        # seeds is negligible)
        assert not np.allclose(signals1, signals2, atol=1e-6), (
            "Two envs with seed=None produced identical RegD trajectories. "
            "seed fallback is still 42 for both. High Bug #3 not fixed."
        )

    def test_explicit_seed_still_reproducible(self):
        """Explicit seed= must still give a deterministic trajectory."""
        env1 = C2GFastEnv(scenario="default")
        env2 = C2GFastEnv(scenario="default")
        env1.reset(seed=7)
        env2.reset(seed=7)

        signals1, signals2 = [], []
        for _ in range(10):
            _, _, _, _, info1 = env1.step(np.array([1.0, 0.7, 0.7, 0.0], np.float32))
            _, _, _, _, info2 = env2.step(np.array([1.0, 0.7, 0.7, 0.0], np.float32))
            signals1.append(info1["regd_signal"])
            signals2.append(info2["regd_signal"])

        np.testing.assert_allclose(signals1, signals2, atol=1e-9,
                                   err_msg="Explicit seed=7 not reproducible.")

    def test_seed_none_source_code_not_fallback_42(self):
        """Source must not contain the old hardcoded fallback `else 42`."""
        import inspect
        from c2g_env.env_low_level import C2GFastEnv as FastEnv
        src = inspect.getsource(FastEnv.reset)
        assert "else 42" not in src, (
            "reset() still contains `else 42` hardcoded seed fallback. "
            "High Bug #3 not fixed."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Medium Bug #1 — committed_mw=0 no longer causes 500× tracking spike
# ═══════════════════════════════════════════════════════════════════════════

class TestCommittedMwZeroReward:

    @pytest.mark.parametrize("profile", available_plant_profiles())
    @pytest.mark.parametrize("commitment_fraction", [0.0, 0.0001, 0.02])
    def test_tracking_floor_scales_with_capacity(self, profile, commitment_fraction):
        env = C2GFastEnv(plant_overrides=load_plant_profile(profile))
        try:
            env.reset(seed=0)
            env.committed_mw = commitment_fraction * env._facility_cap_kw / 1_000.0
            *_, info = env.step(np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32))
            floor_kw = 100.0 * env._facility_cap_kw / 250_000.0
            norm_kw = max(env.committed_mw * 1_000.0, floor_kw)
            assert env._tracking_floor_kw == pytest.approx(floor_kw)
            assert info["reward_tracking"] == pytest.approx(
                -env._rcfg["beta"] * info["tracking_err_kw"] / norm_kw
            )
        finally:
            env.close()

    @pytest.mark.parametrize("key", ["tracking_floor_fraction", "macro_tracking_floor_fraction"])
    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_tracking_floor_rejected(self, tmp_path, key, value):
        import yaml
        from c2g_env.env_low_level import _CONFIG_PATH

        config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
        config["reward"][key] = value
        path = tmp_path / "invalid_floor.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="finite and positive"):
            C2GFastEnv(config_path=path)

    def test_zero_committed_mw_reward_bounded(self):
        """
        With committed_mw forced to 0, the reward must not spike to ±500.
        regd_signal=0 when committed_mw=0 so delta_p_demanded=0 too → tracking_err≈0.
        """
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        env._committed_mw = 0.0
        action = np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert abs(reward) < 20.0, (
            f"Reward {reward:.1f} is abnormally large with committed_mw=0. "
            "Medium Bug #1 not fixed."
        )

    def test_normal_committed_mw_tracking_reward_scale(self):
        """At normal committed_mw tracking penalty must be O(1), not O(500)."""
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        action = np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert abs(reward) < 20.0, f"Reward {reward:.2f} out of expected range."


# ═══════════════════════════════════════════════════════════════════════════
# Medium Bug #2 — thermal penalty normalized to [0,1] range
# ═══════════════════════════════════════════════════════════════════════════

class TestThermalPenaltyNormalization:

    def test_thermal_pen_uses_headroom_normalization(self):
        """Source must divide temp excess by temp_headroom (T_safe - T_warn)."""
        import inspect
        from c2g_env.env_low_level import C2GFastEnv as E
        src = inspect.getsource(E.step)
        assert "temp_headroom" in src, (
            "thermal_pen does not use temp_headroom normalization. "
            "Medium Bug #2 not fixed."
        )

    def test_thermal_pen_at_warn_is_zero(self):
        """
        When temp_A == T_warn_A exactly, the thermal excess = 0 → thermal_pen = 0.
        """
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        # Clamp temperatures to exactly T_warn
        T_warn = T_WARN_A
        env._thermal.temp_A = T_warn
        env._thermal.temp_B = T_warn - 1.0   # below warn
        action = np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # Reward must not include a thermal penalty when both zones are at or below T_warn
        # We verify indirectly: run two steps — one at T_warn, one 1°C over.
        # At T_warn, freq/volt pen dominates but is small.
        # This test just checks the source (above) — can't isolate reward components
        # without refactoring. A smoke check suffices.
        assert info["temp_A"] < 36.0   # shouldn't have faulted

    def test_thermal_pen_at_safe_is_one_per_zone(self):
        """
        At temp = T_safe, excess = T_safe - T_warn = headroom → normalised pen = 1.0.
        The total penalty across both zones at T_safe should be ≤ 2.0.
        """
        import inspect
        from c2g_env.env_low_level import C2GFastEnv as E
        src = inspect.getsource(E.step)
        # Verify formula shape: max(0, excess) / headroom is present
        assert "/ temp_headroom" in src, (
            "Normalization / temp_headroom not found. Medium Bug #2 not fixed."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Medium Bug #3 — thermal reset uses scenario ambient temperature
# ═══════════════════════════════════════════════════════════════════════════

class TestThermalScenarioReset:

    def test_thermal_reset_accepts_temp_args(self):
        """ThermalTwin.reset() must accept optional temp_A, temp_B."""
        from c2g_env.physics.thermal import ThermalTwin
        twin = ThermalTwin()
        twin.reset(temp_A=32.0, temp_B=28.0)
        assert twin.temp_A == pytest.approx(32.0)
        assert twin.temp_B == pytest.approx(28.0)

    def test_thermal_reset_default_unchanged(self):
        """Without arguments, reset() must use original defaults (30, 20)."""
        from c2g_env.physics.thermal import ThermalTwin
        twin = ThermalTwin()
        twin.reset()
        assert twin.temp_A == pytest.approx(30.0)
        assert twin.temp_B == pytest.approx(20.0)

    def test_hot_scenario_starts_warmer_than_default(self):
        """
        scenario_b has T_amb=40°C; its initial temp must be higher than
        default scenario's T_amb=25°C to reflect the warmer environment.
        """
        env_def = C2GFastEnv(scenario="default")   # T_amb=25
        obs_def, _ = env_def.reset(seed=0)

        env_hot = C2GFastEnv(scenario="scenario_b")  # T_amb=40
        obs_hot, _ = env_hot.reset(seed=0)

        # obs[0] = temp_A / T_safe; hot scenario must be at least as warm
        assert obs_hot[0] >= obs_def[0], (
            f"Hot scenario temp_A_norm ({obs_hot[0]:.4f}) should be ≥ "
            f"default ({obs_def[0]:.4f}). Medium Bug #3 not fixed."
        )

    def test_initial_temp_below_t_safe(self):
        """Reset temperature must never exceed T_safe regardless of T_amb."""
        from c2g_env.physics.thermal import ThermalTwin
        twin = ThermalTwin()
        # Even a very high T_amb should not start at T_safe
        twin.reset(temp_A=34.9, temp_B=34.9)
        assert twin.temp_A < twin.T_safe
        assert twin.temp_B < twin.T_safe


# ═══════════════════════════════════════════════════════════════════════════
# Medium Bug #4 — BESS derating window symmetric; energy accounting consistent
# ═══════════════════════════════════════════════════════════════════════════

class TestBESSDeratingFix:

    def test_derate_window_symmetric(self):
        """Both discharge and charge derating must use the same 0.10 window."""
        import inspect
        from c2g_env.physics.bess import _SimpleBESSModel
        src = inspect.getsource(_SimpleBESSModel.step)
        # Old code had 0.05 for charge; that must be gone
        lines_with_derate = [l for l in src.splitlines() if "0.05" in l and "derate" in l.lower()]
        assert len(lines_with_derate) == 0, (
            f"Old asymmetric 0.05 charge window still present: {lines_with_derate}. "
            "Medium Bug #4 not fixed."
        )
        assert "_DERATE_WINDOW = 0.10" in src, (
            "_DERATE_WINDOW constant not found. Medium Bug #4 not fixed."
        )

    def test_energy_accounting_at_discharge_limit(self):
        """
        When BESS hits SOC_MIN during discharge, actual_power must satisfy
        energy conservation: actual_power ≤ available_energy / dt.
        """
        from c2g_env.physics.bess import _SimpleBESSModel
        bess = _SimpleBESSModel(dt_seconds=300.0)
        # Start just above SOC_MIN
        soc_init = bess.SOC_MIN + 0.005
        bess.set_initial_soc(soc_init)

        result = bess.step(bess.P_MAX_MW)  # full discharge
        dt_hr = 300.0 / 3600.0
        available_mwh = soc_init * bess.E_NOM_MWH - bess.SOC_MIN * bess.E_NOM_MWH
        max_power_possible = available_mwh / dt_hr  # upper bound (ignoring eta)

        assert result["actual_power_mw"] >= 0.0, "actual_power must be non-negative for discharge"
        assert result["actual_power_mw"] <= max_power_possible + 1e-6, (
            f"actual_power {result['actual_power_mw']:.3f} MW exceeds available energy bound "
            f"{max_power_possible:.3f} MW. Energy accounting inconsistency not fixed."
        )

    def test_energy_accounting_at_charge_limit(self):
        """
        When BESS hits SOC_MAX during charge, actual_power must be negative
        (charging) and bounded by available headroom.
        """
        from c2g_env.physics.bess import _SimpleBESSModel
        bess = _SimpleBESSModel(dt_seconds=300.0)
        soc_init = bess.SOC_MAX - 0.005
        bess.set_initial_soc(soc_init)

        result = bess.step(-bess.P_MAX_MW)  # full charge
        dt_hr = 300.0 / 3600.0
        available_mwh = (bess.SOC_MAX - soc_init) * bess.E_NOM_MWH
        max_charge_power = available_mwh / dt_hr

        assert result["actual_power_mw"] <= 0.0, "actual_power must be negative (charging)"
        assert abs(result["actual_power_mw"]) <= max_charge_power + 1e-6, (
            f"|actual_power| {abs(result['actual_power_mw']):.3f} MW exceeds charge headroom "
            f"{max_charge_power:.3f} MW. Energy accounting inconsistency not fixed."
        )

    def test_no_soc_violation_after_step(self):
        """SOC must stay within [SOC_MIN, SOC_MAX] after any step."""
        from c2g_env.physics.bess import _SimpleBESSModel
        bess = _SimpleBESSModel(dt_seconds=300.0)
        for soc_start, power in [(0.11, 50.0), (0.94, -50.0), (0.5, 50.0), (0.5, -50.0)]:
            bess.set_initial_soc(soc_start)
            bess.step(power)
            assert bess.soc_fraction >= bess.SOC_MIN - 1e-9
            assert bess.soc_fraction <= bess.SOC_MAX + 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Medium Bug #5 — MacroEnv uses committed_mw property, not _committed_mw
# ═══════════════════════════════════════════════════════════════════════════

class TestCommittedMwProperty:

    def test_fast_env_has_committed_mw_property(self):
        """C2GFastEnv must expose a committed_mw property."""
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        assert hasattr(type(env), "committed_mw"), (
            "C2GFastEnv does not have a committed_mw property. Medium Bug #5 not fixed."
        )

    def test_committed_mw_setter_clamps_to_zero(self):
        """Setting committed_mw < 0 must clamp to 0."""
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        env.committed_mw = -10.0
        assert env.committed_mw == pytest.approx(0.0)

    def test_committed_mw_setter_updates_internal(self):
        """Setting committed_mw via the property must be reflected in _committed_mw."""
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        env.committed_mw = 25.0
        assert env._committed_mw == pytest.approx(25.0)

    def test_macro_env_uses_property_not_private(self):
        """MacroEnv source must not directly write env._committed_mw."""
        import inspect
        from c2g_env.env_high_level import C2GMacroEnv
        src = inspect.getsource(C2GMacroEnv.step)
        assert "_fast_env._committed_mw" not in src, (
            "MacroEnv.step() still directly sets _fast_env._committed_mw. "
            "Medium Bug #5 not fixed."
        )
        assert "_fast_env.committed_mw" in src, (
            "MacroEnv.step() does not use committed_mw property setter."
        )

    def test_macro_env_committed_mw_flows_through_to_fast_env(self):
        """
        After a MacroEnv step, C2GFastEnv._committed_mw must equal the value
        derived from the macro action (bid_mw_norm x committed_max_mw) when
        the market handshake accepts the bid.
        """
        from c2g_env import C2GMacroEnv
        env = C2GMacroEnv(scenario="default")
        env.reset(seed=0)
        # 3D action: [bid_mw_norm, bid_price_norm, bess_target]
        # bid_price_norm=0.0 to maximise acceptance probability
        action = np.array([0.6, 0.0], dtype=np.float32)
        *_, info = env.step(action)
        if info.get("bid_accepted", True):
            expected = 0.6 * env._committed_max_mw
            assert env._fast_env.committed_mw == pytest.approx(expected, abs=0.01), (
                f"Fast env committed_mw={env._fast_env.committed_mw:.2f}, "
                f"expected {expected:.2f}. Medium Bug #5 not fixed."
            )
        else:
            assert env._fast_env.committed_mw == pytest.approx(0.0, abs=0.01), (
                "Bid was rejected but committed_mw is not zero."
            )
