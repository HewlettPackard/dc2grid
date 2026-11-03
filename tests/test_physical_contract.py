from pathlib import Path

import numpy as np
import pytest
import yaml

from c2g_env import C2GFastEnv, C2GMacroEnv
from c2g_env.replay import sample_previous
from c2g_env.train_envs import RuleMacroWrappedEnv

ROOT = Path(__file__).resolve().parents[1]
NOMINAL = np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)


@pytest.fixture
def short_config(tmp_path):
    cfg = yaml.safe_load((ROOT / "c2g_env/config.yaml").read_text())
    cfg["global"]["episode_ticks"] = 361
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_timestamp_hold_uses_time_not_row_number():
    times = np.array(
        ["2023-01-01T00:00", "2023-01-01T00:01", "2023-01-01T00:05"],
        dtype="datetime64[s]",
    )
    targets = np.array(
        ["2023-01-01T00:00:30", "2023-01-01T00:04", "2023-01-01T00:05"],
        dtype="datetime64[s]",
    )
    np.testing.assert_array_equal(
        sample_previous(times, np.array([10, 20, 30]), targets, max_age_seconds=300),
        [10, 20, 30],
    )
    with pytest.raises(ValueError, match="coverage"):
        sample_previous(
            times,
            np.array([10, 20, 30]),
            np.array(["2023-01-01T00:20"], dtype="datetime64[s]"),
            max_age_seconds=300,
        )
    with pytest.raises(ValueError, match="before"):
        sample_previous(
            times,
            np.array([10, 20, 30]),
            np.array(["2022-12-31T23:59"], dtype="datetime64[s]"),
            max_age_seconds=300,
        )


@pytest.mark.parametrize("day", [0, 207, 248, 249])
def test_calendar_covers_full_day_and_bias_is_observed_at_reset(day):
    unbiased = C2GFastEnv(start_day=day)
    biased = C2GFastEnv(start_day=day, regd_load_bias=0.5)
    obs0, _ = unbiased.reset(seed=100)
    obs1, info = biased.reset(seed=100)
    times = biased._grid.calendar_timestamps
    assert times[-1] - times[0] == np.timedelta64(24, "h")
    assert info["simulation_mode"] == "physical_v1"
    assert "calendar_climatology_2024" in biased._weather.source
    expected = np.clip(obs0[6] + 0.5 * biased._grid._load_anomaly[0], -1, 1)
    assert obs1[6] == pytest.approx(expected, abs=1e-7)
    assert obs1[8] == pytest.approx(
        biased._grid.load_at_tick(0) / biased._grid._load_max
    )


def test_nominal_reference_has_zero_relief_and_preserves_service(short_config):
    env = C2GFastEnv(config_path=short_config)
    env.reset(seed=100)
    for _ in range(10):
        _, _, _, _, info = env.step(NOMINAL)
        assert info["facility_relief_kw"] == pytest.approx(0, abs=1e-8)
        assert info["delta_p_actual_kw"] == pytest.approx(0, abs=1e-8)
        assert info["work_served_kwh"] == pytest.approx(info["work_arrived_kwh"])
        assert info["backlog_work_kwh"] == 0


def test_unused_service_capacity_does_not_create_tracking_credit(short_config):
    env = C2GFastEnv(config_path=short_config)
    env.reset(seed=100)
    env._workload._p_flex_arr[:] = env._workload.p_flex_idle_kw + 20_000
    action = NOMINAL.copy()
    action[0] = 0.5
    _, _, _, _, info = env.step(action)
    assert info["flex_reduction_kw"] == pytest.approx(0)
    assert info["facility_relief_kw"] == pytest.approx(0, abs=1e-8)
    assert info["p_flex_served_kw"] == pytest.approx(29_600)
    assert info["work_served_kwh"] == pytest.approx(20_000 * 5 / 3600)


def test_deferral_repayment_idle_heat_and_work_conservation(short_config, monkeypatch):
    env = C2GFastEnv(config_path=short_config)
    env.reset(seed=100)
    env._workload._p_flex_arr[:] = env._workload.p_flex_idle_kw + 10_000
    original_step = env._thermal.step
    heat = []

    def observe_heat(**kwargs):
        heat.append(kwargs["p_it_A_mw"] + kwargs["p_it_B_mw"])
        return original_step(**kwargs)

    monkeypatch.setattr(env._thermal, "step", observe_heat)
    rows = []
    for throttle in [0, 0, 1]:
        action = NOMINAL.copy()
        action[0] = throttle
        rows.append(env.step(action)[-1])
    assert rows[0]["p_flex_served_kw"] == env._workload.p_flex_idle_kw
    assert rows[-1]["flex_reduction_kw"] == pytest.approx(-20_000)
    assert rows[-1]["facility_relief_kw"] < 0
    assert sum(r["work_arrived_kwh"] for r in rows) == pytest.approx(
        sum(r["work_served_kwh"] for r in rows) + rows[-1]["backlog_work_kwh"]
    )
    np.testing.assert_allclose(heat, [r["p_total_it_mw"] for r in rows], atol=1e-10)


def test_battery_tracking_and_facility_relief_are_separate(short_config):
    env = C2GFastEnv(config_path=short_config)
    env.reset(seed=100)
    action = NOMINAL.copy()
    action[3] = 0.5
    info = env.step(action)[-1]
    assert info["facility_relief_kw"] == pytest.approx(0, abs=1e-8)
    assert info["delta_p_actual_kw"] == pytest.approx(info["bess_actual_kw"])
    assert info["p_grid_import_mw"] == pytest.approx(
        info["p_facility_mw"] - info["bess_actual_kw"] / 1000
    )


def test_market_calls_cannot_change_exogenous_regd(short_config):
    a = C2GFastEnv(config_path=short_config)
    b = C2GFastEnv(config_path=short_config)
    a.reset(seed=123)
    b.reset(seed=123)
    for _ in range(10):
        b._grid.step_rmcp()
        b._grid.clear_bid(0, 10)
        b._grid._step_frequency(100)
        oa = a.step(NOMINAL)[0]
        ob = b.step(NOMINAL)[0]
        np.testing.assert_array_equal(oa[[6, 7, 8]], ob[[6, 7, 8]])


def test_posted_offer_commitment_and_interval_settlement(short_config):
    observations, rows = [], []

    def inner(obs, macro_action):
        observations.append(obs.copy())
        return NOMINAL.copy()

    env = C2GMacroEnv(
        config_path=short_config,
        inner_action_fn=inner,
        sub_step_callback=lambda pre, action, post, reward, done, info: rows.append(
            info
        ),
    )
    obs, _ = env.reset(seed=100)
    offered_price = float(env._pending_offer["rmcp_usd"])
    assert obs[17] == pytest.approx(offered_price / env._rmcp_max)
    _, _, _, _, info = env.step(np.array([0.8, 0], dtype=np.float32))
    assert info["rmcp_usd"] == offered_price
    assert observations[0][17] * env._committed_max_mw == pytest.approx(
        info["committed_mw"], abs=1e-6
    )
    assert info["electricity_cost"] == pytest.approx(
        sum(r["lmp"] * r["p_grid_import_mw"] * r["duration_hours"] for r in rows)
    )
    assert info["tracking_rmse_kw"] == pytest.approx(
        np.sqrt(np.mean([r["tracking_err_kw"] ** 2 for r in rows]))
    )
    assert info["settlement_hours"] == pytest.approx(0.25)
    env.step(np.array([0.8, 0], dtype=np.float32))
    _, _, term, trunc, partial = env.step(np.array([0.8, 0], dtype=np.float32))
    assert not term and trunc
    assert partial["sub_steps_run"] == 1
    assert partial["settlement_hours"] == pytest.approx(5 / 3600)
    assert partial["regulation_revenue"] == pytest.approx(
        partial["rate"] * partial["committed_mw"] * partial["perf_score"] * 5 / 3600
    )


def test_rule_wrapper_observes_commitment_before_action_and_matches_macro(short_config):
    from baselines.rule_based_macro import RuleBasedMacroController

    wrapped = RuleMacroWrappedEnv(C2GFastEnv(config_path=short_config))
    macro = C2GMacroEnv(
        config_path=short_config, inner_action_fn=lambda obs, action: NOMINAL.copy()
    )
    macro_controller = RuleBasedMacroController()
    obs, _ = wrapped.reset(seed=100)
    macro_obs, _ = macro.reset(seed=100)
    for _ in range(2):
        macro_info = macro.step(macro_controller.predict(macro_obs)[0])
        macro_obs = macro_info[0]
        for _ in range(180):
            assert obs[17] * wrapped.env._committed_mw_max == pytest.approx(
                wrapped.env.committed_mw,
                abs=1e-6,
            )
            obs = wrapped.step(NOMINAL)[0]
        np.testing.assert_allclose(obs[:17], macro._last_hw_obs[:17], atol=1e-6)


def test_legacy_mode_retains_historical_throttle_credit(short_config):
    env = C2GFastEnv(config_path=short_config, simulation_mode="legacy_v0")
    env.reset(seed=100)
    action = NOMINAL.copy()
    action[0] = 0.5
    info = env.step(action)[-1]
    assert info["flex_reduction_kw"] == pytest.approx(0.5 * info["p_flex_nom_kw"])
    assert info["reward_throughput"] == 0.5


def test_invalid_mode_and_replay_options_fail_explicitly():
    with pytest.raises(ValueError, match="simulation_mode"):
        C2GFastEnv(simulation_mode="typo")
    with pytest.raises(ValueError, match="start_day"):
        C2GFastEnv(start_day=0.5).reset(seed=100)


@pytest.mark.parametrize("mode", ["physical_v1", "legacy_v0"])
def test_macro_observation_box_metadata_matches_its_bounds(short_config, mode):
    env = C2GMacroEnv(config_path=short_config, simulation_mode=mode)
    try:
        observation, _ = env.reset(seed=100)
        space = env.observation_space
        expected_high = np.array(
            [
                2,
                2,
                1,
                1,
                2,
                1,
                1,
                1,
                np.inf if mode == "physical_v1" else 2,
                1,
                1,
                1,
                1,
                1,
                1,
                1.1,
                2,
                5,
                5,
            ],
            dtype=np.float32,
        )
        expected_low = np.zeros(19, dtype=np.float32)
        expected_low[14] = -1
        np.testing.assert_array_equal(space.low, expected_low)
        np.testing.assert_array_equal(space.high, expected_high)
        np.testing.assert_array_equal(space.bounded_below, np.isfinite(space.low))
        np.testing.assert_array_equal(space.bounded_above, np.isfinite(space.high))
        assert space.c2g_simulation_mode == mode
        assert space.contains(observation)
        space.seed(37)
        for _ in range(32):
            sample = space.sample()
            assert np.isfinite(sample).all()
            assert space.contains(sample)
    finally:
        env.close()
