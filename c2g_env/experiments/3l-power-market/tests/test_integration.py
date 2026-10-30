import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


class KnownPriceTestBackend:
    """Test double for delivery metering, not a replacement market implementation."""

    def __init__(self, ledger, market):
        self.ledger, self.market = ledger, market
        self.closed = False

    def start(self, controllers):
        self.controllers = controllers

    def advance_to(self, when):
        slot = when.floor("15min")
        if self.ledger.price("BM", slot, slot + self.market.SLOT, when) is None:
            self.ledger.record_price("BM", slot, slot + self.market.SLOT, -20, slot)

    def close(self):
        self.closed = True


def test_delivery_advances_180_ticks_without_bids(modules, config, ireland_data):
    start = pd.Timestamp("2025-01-01")
    env = modules.env.ThreeLevelEnv(
        config,
        ireland_data,
        start,
        start + modules.market.SLOT,
        start,
        lambda ledger: KnownPriceTestBackend(ledger, modules.market),
    )
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    metered = 0.0
    for _ in range(180):
        observation, _, terminated, truncated, info = env.step(
            env.plant.nominal_action()
        )
        metered += info["plant"]["net_import_mw"] * env.plant.dt_seconds / 3600
        assert isinstance(terminated, bool)
        assert not terminated
    assert truncated
    assert env.observation_space.contains(observation)
    assert env.summary()["completed"]
    remaining = modules.env.MARKET_OBSERVATION_FIELDS.index(
        "remaining_interval_seconds"
    )
    assert observation["market"][remaining] == 0
    assert env.fast_steps == 180
    assert len(env.records) == 1
    assert env.records[0]["metered_mwh"] == pytest.approx(metered)
    assert env.scored_cost_eur == pytest.approx(metered * -20)
    assert all(value == 0 for value in env.summary()["decisions_by_market"].values())
    env.close()
    assert env.backend.closed


def test_slot_settlement_records_observed_profile_and_scored_flag(
    modules, config, ireland_data
):
    start = pd.Timestamp("2025-01-01")
    env = modules.env.ThreeLevelEnv(
        config,
        ireland_data,
        start,
        start + 2 * modules.market.SLOT,
        start + modules.market.SLOT,
        lambda ledger: KnownPriceTestBackend(ledger, modules.market),
    )
    _, info = env.reset(seed=0)
    assert info["scored"] is False
    samples = []
    for _ in range(180):
        _, _, _, _, info = env.step(env.plant.nominal_action())
        samples.append(info["plant"]["batch_arrived_mw"])
    assert info["scored"] is False
    recorded = env.controllers.profile.slots[pd.Timedelta(0)]
    assert recorded["batch_arrived_mw"] == pytest.approx(sum(samples) / 180)
    assert set(recorded) == set(env.controllers.profile.FIELDS)
    _, _, _, _, info = env.step(env.plant.nominal_action())
    assert info["scored"] is True
    assert env.controllers.profile.hour(start) is None
    env.close()


def test_midnight_callback_sees_completed_inputs_before_settlement(
    modules, config, ireland_data, monkeypatch
):
    start = pd.Timestamp("2025-01-01 23:45")
    midnight = start + modules.market.SLOT
    env = modules.env.ThreeLevelEnv(
        config,
        ireland_data,
        start,
        midnight,
        start,
        lambda ledger: KnownPriceTestBackend(ledger, modules.market),
    )
    env.reset(seed=0)
    profile = env.controllers.profile
    for offset in range(95):
        profile.record(
            start.normalize() + offset * modules.market.SLOT,
            {"base_a_mw": 30.0, "base_b_mw": 20.0, "batch_arrived_mw": 2.0},
        )
    record = profile.record
    records, seen = [], []

    def observe(slot_start, values):
        records.append(slot_start)
        record(slot_start, values)

    advance = env.backend.advance_to

    def opening(when):
        if when == midnight:
            assert start not in env.ledger.settled
            assert len(profile.slots) == 96
            seen.append(profile.hour(start.floor("h")))
        else:
            assert profile.hour(start.floor("h")) is None
        advance(when)

    monkeypatch.setattr(profile, "record", observe)
    monkeypatch.setattr(env.backend, "advance_to", opening)
    try:
        arrivals = []
        for _ in range(180):
            _, _, _, _, info = env.step(env.plant.nominal_action())
            arrivals.append(info["plant"]["batch_arrived_mw"])
        assert records == [start]
        assert len(seen) == 1
        assert seen[0]["batch_arrived_mw"] == pytest.approx(
            (3 * 2.0 + sum(arrivals) / 180) / 4
        )
        assert len(env.records) == 1
        assert start in env.ledger.settled
    finally:
        env.close()


def test_plan_cap_defers_work_into_backlog_and_releases_it(
    modules, config, ireland_data
):
    slot = modules.market.SLOT
    start = pd.Timestamp("2025-01-01")
    env = modules.env.ThreeLevelEnv(
        config,
        ireland_data,
        start,
        start + 2 * slot,
        start,
        lambda ledger: KnownPriceTestBackend(ledger, modules.market),
    )
    env.reset(seed=0)
    env.controllers.plan.set(start, start + slot, 0.0)
    env.controllers.plan.set(start + slot, start + 2 * slot, 1.0)
    arrived = 0.0
    for _ in range(180):
        _, _, _, _, info = env.step(env.hardware_action())
        arrived += info["plant"]["batch_arrived_mw"] * env.plant.dt_seconds / 3600
        assert info["applied_action"][0] == 0.0
        assert info["plant"]["batch_served_mw"] == 0.0
    assert env.plant.state.backlog_work_mwh == pytest.approx(arrived)
    assert env.hardware.capped_steps == 180
    for _ in range(180):
        _, _, _, truncated, info = env.step(env.hardware_action())
    assert truncated
    assert env.plant.state.backlog_work_mwh < arrived
    assert env.summary()["inner_capped_steps"] == 180
    # The test backend prices every slot, so a zero position is an active target.
    assert env.summary()["scored_request_mae_mw"] > 0
    env.close()


def test_demand_unit_rating_is_independent_of_warm_start(
    modules, config, plant, monkeypatch
):
    from types import SimpleNamespace

    pytest.importorskip("assume")
    root, _, _, _ = modules.config.three_l_config(config)
    monkeypatch.syspath_prepend(str(root))
    adapters = importlib.import_module(
        "c2g_env.experiments.3l-power-market.assume_adapters"
    )
    bridge = SimpleNamespace(controllers=SimpleNamespace(plant=plant))
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    forecaster = SimpleNamespace(index=index)
    unit = adapters.C2GDemandUnit(
        bridge=bridge,
        id="dc",
        unit_operator="dc",
        bidding_strategies={},
        forecaster=forecaster,
    )
    electrical, thermal = plant.electrical, plant.thermal
    it_mw = (
        electrical.n_racks_A * electrical.p_max_rack_A_kw
        + electrical.n_racks_B * electrical.p_max_rack_B_kw
    ) / 1000
    assert unit.min_power == pytest.approx(
        -(it_mw + thermal.max_hvac_mw + thermal.P_PUMP_MAX_MW)
    )
    assert -unit.min_power > plant.state.facility_mw


def test_timestamp_and_numpy_serialization(tmp_path):
    import json

    import numpy as np

    runner = importlib.import_module("c2g_env.experiments.3l-power-market.run")
    path = tmp_path / "record.json"
    runner.write_json(
        path, {"time": pd.Timestamp("2025-01-01"), "value": np.float32(2)}
    )
    assert json.loads(path.read_text()) == {"time": "2025-01-01T00:00:00", "value": 2.0}


def test_recorded_episode_has_every_physical_transition(
    modules, config, ireland_data, tmp_path
):
    import json

    runner = importlib.import_module("c2g_env.experiments.3l-power-market.run")
    start = pd.Timestamp("2025-01-01")
    env = modules.env.ThreeLevelEnv(
        config,
        ireland_data,
        start,
        start + modules.market.SLOT,
        start,
        lambda ledger: KnownPriceTestBackend(ledger, modules.market),
    )
    try:
        runner.record_episode(env, tmp_path)
        rows = [
            json.loads(line)
            for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 181
        assert rows[0]["event"] == "reset"
        assert len(rows[0]["observation"]["market"]) == 12
        assert rows[-1]["truncated"]
        assert rows[-1]["info"]["fast_steps"] == 180
        assert len(rows[-1]["controller"]["observation"]) == 18
        assert env.hardware.predict_calls == 180
        assert env.summary()["scored_financially_complete"]
        summary = env.summary()
        assert summary["scored_metered_mwh"] == pytest.approx(
            env.records[0]["metered_mwh"]
        )
        assert summary["scored_cost_eur"] == pytest.approx(
            summary["scored_procurement_cost_eur"]
            + summary["scored_imbalance_cost_eur"]
        )
        assert summary["scored_tracking_hours"] == pytest.approx(0.25)
    finally:
        env.close()


def test_provenance_fingerprints_workload_inputs(modules, config, plant, monkeypatch):
    runner = importlib.import_module("c2g_env.experiments.3l-power-market.run")
    root, _, scenario, _ = modules.config.three_l_config(config)
    monkeypatch.setattr(runner.metadata, "version", lambda name: "test-only-version")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    result = runner.provenance(config, root, scenario, plant)
    trace_dir = modules.config.REPO_ROOT / plant.global_config["trace_dir"]
    for name in modules.plant.WorkloadOrchestrator._TRACE_FILES.values():
        assert runner.file_fingerprint(trace_dir / name) in result["input_fingerprints"]
    defaults_source = modules.config.REPO_ROOT / "c2g_env" / "env_high_level.py"
    assert runner.file_fingerprint(defaults_source) in result["source_fingerprints"]
    assert result["thread_environment"]["OMP_NUM_THREADS"] == "1"


def test_write_boundary(modules, tmp_path):
    assert modules.config.contained_path(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="remain inside"):
        modules.config.contained_path(modules.config.REPO_ROOT / "outside.json")
    runner = importlib.import_module("c2g_env.experiments.3l-power-market.run")
    with pytest.raises(ValueError, match="remain inside"):
        runner.write_json(modules.config.REPO_ROOT / "outside.json", {})


def test_evaluation_seeds_use_separate_workers(config, tmp_path, monkeypatch):
    import json

    runner = importlib.import_module("c2g_env.experiments.3l-power-market.run")
    calls = []

    def worker(config_path, output, log_path):
        cfg = json.loads(config_path.read_text())
        calls.append((cfg["simulation"]["seed"], output, log_path))
        output.mkdir()
        runner.write_json(
            output / "summary.json",
            {
                "seed": cfg["simulation"]["seed"],
                "completed": True,
                "scored_cost_eur": 100.0 + cfg["simulation"]["seed"],
            },
        )

    monkeypatch.setattr(runner, "_run_worker", worker)
    output = runner.evaluate(config, [0, 2], tmp_path / "evaluation")
    assert [seed for seed, _, _ in calls] == [0, 2]
    assert config["simulation"]["seed"] == 0
    frame = pd.read_csv(output / "evaluation.csv")
    assert list(frame["seed"]) == [0, 2]
    assert list(frame["scored_cost_eur"]) == [100.0, 102.0]
    assert len(json.loads((output / "evaluation.json").read_text())["runs"]) == 2
    with pytest.raises(ValueError, match="unique"):
        runner.evaluate(config, [0, 0], tmp_path / "duplicate")


def test_worker_failure_is_reported(tmp_path, monkeypatch):
    from types import SimpleNamespace

    runner = importlib.import_module("c2g_env.experiments.3l-power-market.run")
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7)
    )
    with pytest.raises(RuntimeError, match="exit code 7"):
        runner._run_worker(
            tmp_path / "config.json", tmp_path / "output", tmp_path / "worker.log"
        )


@pytest.mark.parametrize("fault_tick", (1, 180))
def test_thermal_fault_and_open_positions(modules, config, ireland_data, fault_tick):
    start = pd.Timestamp("2025-01-01")
    env = modules.env.ThreeLevelEnv(
        config,
        ireland_data,
        start,
        start + modules.market.SLOT,
        start,
        lambda ledger: KnownPriceTestBackend(ledger, modules.market),
    )
    env.reset(seed=0)
    env.ledger.record_trade(
        modules.market.Trade("EOM", "remaining", start, env.end_time, 8, 50, start)
    )
    for _ in range(fault_tick - 1):
        env.step(env.plant.nominal_action())
    env.plant.thermal.T_safe = 0
    _, _, terminated, _, _ = env.step(env.plant.nominal_action())
    summary = env.summary()
    assert terminated
    assert not summary["completed"]
    assert summary["thermal_fault"]
    assert summary["scored_financially_complete"] == (fault_tick == 180)
    assert summary["open_future_positions_mwh"] == pytest.approx(
        8 * (env.end_time - env.now).total_seconds() / 3600
    )
    if fault_tick == 1:
        assert summary["unsettled_partial_interval_mwh"] > 0
    env.close()


def test_backend_context_restores_process_state(modules, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", "previous")
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    previous_cwd, previous_bytecode = Path.cwd(), sys.dont_write_bytecode
    with pytest.raises(RuntimeError, match="context failure"):
        with modules.assume_backend.local_directory(tmp_path):
            assert Path.cwd() == tmp_path
            assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "cache")
            assert sys.dont_write_bytecode
            raise RuntimeError("context failure")
    assert Path.cwd() == previous_cwd
    assert sys.dont_write_bytecode == previous_bytecode
    assert os.environ["MPLCONFIGDIR"] == "previous"
    assert "NUMBA_CACHE_DIR" not in os.environ


@pytest.mark.parametrize("export_fails", (False, True))
def test_backend_shutdown_is_complete_and_idempotent(
    modules, tmp_path, export_fails, monkeypatch
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock

    events = []

    async def store():
        events.append("store")
        if export_fails:
            raise OSError("export failed")

    async def shutdown():
        events.append("shutdown")

    backend = modules.assume_backend.ThreeLBackend(
        tmp_path,
        {},
        modules.market.PositionLedger(),
        tmp_path,
        pd.Timestamp("2025-01-01"),
        0,
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    set_event_loop = Mock(wraps=asyncio.set_event_loop)
    monkeypatch.setattr(asyncio, "set_event_loop", set_event_loop)
    backend.world = SimpleNamespace(
        loop=loop,
        output_role=SimpleNamespace(store_dfs=store),
        container=SimpleNamespace(shutdown=shutdown),
    )
    backend.activated = True
    if export_fails:
        with pytest.raises(OSError, match="export failed"):
            backend.close()
    else:
        backend.close()
    assert loop.is_closed()
    assert not backend.activated
    assert backend.world is None
    assert events == ["store", "shutdown"]
    set_event_loop.assert_called_once_with(None)
    backend.close()


@pytest.mark.skipif(
    os.environ.get("C2G_3L_LIVE_TEST") != "1",
    reason="Live 3L test requires provisioned dependencies and C2G_3L_LIVE_TEST=1.",
)
def test_live_three_market_day_rollover(modules, config):
    runner = importlib.import_module("c2g_env.experiments.3l-power-market.run")
    config["simulation"]["scoring_hours"] = 1
    output = runner.execute(config)
    import json

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"]
    assert summary["upper_policy_calls"]["L1"] > 0
    assert summary["upper_policy_calls"]["L2"] > 0
    assert summary["inner_predict_calls"] == summary["fast_steps"]
    assert summary["inner_controller"]["implementation"] == (
        "baselines.rule_based_mpc.RuleBasedController"
    )
    assert summary["scored_financially_complete"]
