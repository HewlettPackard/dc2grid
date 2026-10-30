import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_signed_positions_settlement_and_idempotency(modules):
    market = modules.market
    ledger = market.PositionLedger()
    start = pd.Timestamp("2025-01-02")
    end = start + market.SLOT
    da = market.Trade("EOM", "da", start, start + pd.Timedelta(hours=1), 10, 100, start)
    ledger.record_trade(da)
    ledger.record_trade(da)
    ledger.record_trade(market.Trade("IDM", "id", start, end, -2, 80, start))
    ledger.record_trade(market.Trade("BM", "bm", start, end, 1, -20, start))
    ledger.record_price("BM", start, end, -20, start)
    settled = ledger.settle(start, end, 2.5, scored=True)
    assert settled["positions_mwh"] == {"EOM": 2.5, "IDM": -0.5, "BM": 0.25}
    assert settled["imbalance_mwh"] == 0.25
    assert settled["procurement_cost_eur"] == 205
    assert settled["total_cost_eur"] == 200
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record_trade(replace(da, purchase_mw=11))
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(start, end, 2.5, scored=True)


def test_missing_prices_fail_scoring_not_silently_default(modules):
    ledger = modules.market.PositionLedger()
    start = pd.Timestamp("2025-01-01")
    with pytest.raises(RuntimeError, match="No live BM"):
        ledger.settle(start, start + modules.market.SLOT, 1.0, scored=True)
    result = ledger.settle(start, start + modules.market.SLOT, 1.0, scored=False)
    assert result["status"] == "unpriced_warmup"
    assert result["total_cost_eur"] is None


def test_future_price_is_not_visible(modules):
    ledger = modules.market.PositionLedger()
    start = pd.Timestamp("2025-01-02")
    clear = start - pd.Timedelta(minutes=15)
    ledger.record_price("BM", start, start + modules.market.SLOT, 100, clear)
    assert (
        ledger.price(
            "BM", start, start + modules.market.SLOT, clear - pd.Timedelta(seconds=1)
        )
        is None
    )
    assert ledger.price("BM", start, start + modules.market.SLOT, clear) == 100


def test_clock_does_not_overshoot_and_runs_without_orders(modules):
    events = [3.0, 12.0]
    executed = []
    clock = SimpleNamespace(time=0.0)
    clock.get_next_activity = lambda: events[0] if events else None
    clock.set_time = lambda value: setattr(clock, "time", value)

    async def drain():
        while events and events[0] <= clock.time:
            executed.append(events.pop(0))

    asyncio.run(modules.market.advance_clock(clock, 5.0, drain))
    assert clock.time == 5.0
    assert executed == [3.0]
    asyncio.run(modules.market.advance_clock(clock, 15.0, drain))
    assert clock.time == 15.0
    assert executed == [3.0, 12.0]
    with pytest.raises(ValueError, match="backwards"):
        asyncio.run(modules.market.advance_clock(clock, 14.0, drain))


def test_3l_weather_uses_current_interval(ireland_data):
    at = pd.Timestamp("2025-01-01")
    first = ireland_data.context(at)["ambient_c"]
    assert first == 7.4164124
    assert (
        ireland_data.context(at + pd.Timedelta(minutes=14, seconds=55))["ambient_c"]
        == first
    )
    assert ireland_data.context(at + pd.Timedelta(minutes=15))["ambient_c"] != first
    with pytest.raises(ValueError, match="does not cover"):
        ireland_data.require_coverage(at - pd.Timedelta(days=1), at)
    with pytest.raises(ValueError, match="not a BM"):
        ireland_data.forecast("BM", at)


def test_partial_acceptance_is_metered_from_clearing(modules, tmp_path):
    m = modules.market
    start = pd.Timestamp("2025-01-02")
    backend = modules.assume_backend.ThreeLBackend(
        tmp_path, {}, m.PositionLedger(), tmp_path, start + m.SLOT, 0
    )
    backend.world = SimpleNamespace(clock=SimpleNamespace(time=m.epoch_seconds(start)))
    fills = []
    backend.controllers = SimpleNamespace(
        on_fill=lambda bid, trade: fills.append((bid, trade))
    )
    bid = m.Bid("BM", start, start + m.SLOT, 10.0, 3000, 0.5)
    backend.submitted["partial"] = bid
    accepted = [
        {
            "unit_id": modules.assume_backend.UNIT_ID,
            "bid_id": "partial",
            "start_time": start,
            "end_time": start + m.SLOT,
            "volume": -10,
            "accepted_volume": -3,
            "accepted_price": -25,
        }
    ]
    meta = [{"product_start": start, "product_end": start + m.SLOT, "price": -25}]
    backend.capture("BM", (accepted, [], meta, {}))
    backend.capture("BM", (accepted, [], meta, {}))
    assert backend.ledger.energy(start, start + m.SLOT) == 0.75
    assert backend.ledger.trade_cost(start, start + m.SLOT) == -18.75
    assert len(fills) == 1
    assert fills[0][0] is bid and fills[0][1].purchase_mw == 3.0


def test_dependency_check_does_not_write_or_install(modules, monkeypatch):
    root, _, _, _ = modules.config.three_l_config(modules.config.load_config())
    monkeypatch.setattr(
        modules.assume_backend.importlib.util, "find_spec", lambda name: None
    )
    with pytest.raises(ModuleNotFoundError, match="no packages were installed"):
        modules.assume_backend.require_backend(root)


def test_three_l_paths_resolve_outside_repository_cwd(modules, config, tmp_path):
    with modules.assume_backend.local_directory(tmp_path):
        root, backend, scenario, _ = modules.config.three_l_config(config)
    expected = modules.config.REPO_ROOT / "site-packages" / "3L_Grid"
    assert root == expected.resolve()
    assert Path(backend["sim"]["inputs_path"]).is_relative_to(root)
    assert scenario.is_relative_to(root)


def test_config_paths_must_be_posix_style(modules, config, tmp_path):
    import yaml

    config["three_l"]["root"] = "site-packages\\3L_Grid"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forward slashes"):
        modules.config.load_config(path)
    for value in (
        modules.config.load_config()["c2g"]["config"],
        modules.config.load_config()["three_l"]["config"],
    ):
        assert "\\" not in value


def test_three_l_import_uses_relocated_source(modules, config, tmp_path, monkeypatch):
    backend = modules.assume_backend
    root, _, _, _ = modules.config.three_l_config(config)
    monkeypatch.setattr(backend.sys, "path", backend.sys.path.copy())
    monkeypatch.delitem(backend.sys.modules, "assume_env", raising=False)
    marker = object()

    def import_probe(name):
        assert name == "assume_env.create_world"
        spec = backend.importlib.util.find_spec("assume_env")
        assert spec is not None and spec.origin is not None
        assert Path(spec.origin).resolve().parent == root / "assume_env"
        return marker

    monkeypatch.setattr(backend.importlib, "import_module", import_probe)
    with backend.local_directory(tmp_path):
        assert backend.import_three_l(root) is marker


def test_three_l_import_rejects_stale_checkout(modules, config, monkeypatch):
    backend = modules.assume_backend
    root, _, _, _ = modules.config.three_l_config(config)
    stale = modules.config.REPO_ROOT / "3L_Grid" / "assume_env" / "__init__.py"
    monkeypatch.setitem(
        backend.sys.modules, "assume_env", SimpleNamespace(__file__=str(stale))
    )
    with pytest.raises(RuntimeError, match="different assume_env checkout"):
        backend.import_three_l(root)
