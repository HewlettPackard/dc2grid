from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_da_plan_invokes_native_policy_and_conserves_work(
    modules, plant, ireland_data, config, monkeypatch
):
    control = modules.controllers.ThreeControllers(
        plant, ireland_data, modules.market.PositionLedger(), config
    )
    original = control.day_ahead.policy.plan
    observed = []

    def native_call(**kwargs):
        observed.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(control.day_ahead.policy, "plan", native_call)
    start = pd.Timestamp("2025-01-02")
    products = [
        (start + pd.Timedelta(hours=i), start + pd.Timedelta(hours=i + 1), None)
        for i in range(3)
    ]
    data = SimpleNamespace(
        forecast=lambda market, at: [100, 10, 50][
            int((at - start).total_seconds() / 3600)
        ]
    )
    plant._state = replace(plant.state, batch_arrived_mw=2.0, backlog_work_mwh=0.0)
    monkeypatch.setattr(plant.workload, "_p_flex_max_kw", 3000.0)
    forecast = modules.forecast.DeliveryForecast(
        plant, control.profile, control.plan.service, start
    )
    decisions = control.day_ahead.decide(products, forecast, data, 3000)
    planned = control.day_ahead.last_call["output"]["allocation_mwh"]
    assert len(observed) == 1
    assert planned.sum() == 6.0
    assert (np.cumsum(planned) <= np.cumsum([2.0, 2.0, 2.0])).all()
    assert (planned <= 3.0).all()
    assert planned[1] == 3.0
    assert len(decisions) == 3
    assert all(bid.service_per_mw >= 0 for bid in decisions)
    assert sum(
        bid.forecast["delivery"]["served_work_mwh"] for bid in decisions
    ) == pytest.approx(6.0)
    assert (
        control.day_ahead.last_call["workload_forecast_sources"]
        == ["current_persistence"] * 3
    )
    assert type(control.day_ahead.policy).__module__ == "policies.rule_based_l1"


def test_da_forecast_uses_observed_same_hour_profile(
    modules, plant, ireland_data, config
):
    control = modules.controllers.ThreeControllers(
        plant, ireland_data, modules.market.PositionLedger(), config
    )
    slot = modules.market.SLOT
    history_day = pd.Timestamp("2025-01-01")
    for i in range(4):
        control.profile.record(
            history_day + pd.Timedelta(hours=3) + i * slot,
            {"base_a_mw": 10.0 + i, "base_b_mw": 5.0, "batch_arrived_mw": 4.0},
        )
    start = pd.Timestamp("2025-01-02 03:00")
    products = [
        (start, start + pd.Timedelta(hours=1), None),
        (start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=2), None),
    ]
    bids = control.bid("EOM", products, 3000, -500, pd.Timestamp("2025-01-01"))
    call = control.day_ahead.last_call
    assert call["workload_forecast_sources"] == [
        "observed_profile",
        "current_persistence",
    ]
    assert call["inputs"]["defer_arrivals_mwh"][0] == pytest.approx(4.0)
    assert call["inputs"]["defer_arrivals_mwh"][1] == pytest.approx(
        plant.state.batch_arrived_mw
    )
    delivery = bids[0].forecast["delivery"]
    assert delivery["arrived_work_mwh"] == pytest.approx(4.0)
    assert bids[0].purchase_mw == pytest.approx(delivery["facility_mwh"])
    assert control.profile.hour(start + pd.Timedelta(hours=1)) is None
    with pytest.raises(ValueError, match="hourly"):
        control.profile.hour(start + slot)
    # Submission does not define the plan; acceptance does.
    assert control.plan.service == {}


def test_native_planning_shortfall_does_not_drop_c2g_work(
    modules, plant, ireland_data, config
):
    start = pd.Timestamp("2025-01-02")
    plant._state = replace(plant.state, backlog_work_mwh=plant.batch_capacity_mw * 2.0)
    before = asdict(plant.state)
    control = modules.controllers.ThreeControllers(
        plant, ireland_data, modules.market.PositionLedger(), config
    )
    control.bid(
        "EOM", [(start, start + pd.Timedelta(hours=1), None)], 3000, -500, start
    )
    result = control.day_ahead.last_call["output"]
    assert result["would_drop_mwh"] > 0
    assert result["allocation_mwh"].sum() + result["would_drop_mwh"] == pytest.approx(
        plant.state.backlog_work_mwh + plant.state.batch_arrived_mw
    )
    assert asdict(plant.state) == before


def test_three_separate_decisions_do_not_advance_plant(
    modules, plant, ireland_data, config
):
    start = pd.Timestamp("2025-01-02")
    ledger = modules.market.PositionLedger()
    control = modules.controllers.ThreeControllers(plant, ireland_data, ledger, config)
    tick = plant.workload._tick
    for name in ("EOM", "IDM", "BM"):
        duration = pd.Timedelta(hours=1) if name == "EOM" else modules.market.SLOT
        decisions = control.bid(
            name,
            [(start, start + duration, None)],
            3000,
            -500,
            start - pd.Timedelta(hours=1),
        )
        assert all(decision.market == name for decision in decisions)
    assert plant.workload._tick == tick
    assert control.day_ahead.calls == control.intraday.calls == 1
    assert {call["market"] for call in control.policy_calls} == {"EOM", "IDM"}
    root, _, _, _ = modules.config.three_l_config(config)
    import inspect

    assert Path(inspect.getfile(type(control.intraday.policy))).is_relative_to(root)


@pytest.mark.parametrize("offset", (0, 95))
@pytest.mark.parametrize("side", ("buy", "sell"))
def test_native_l2_prices_quantities_and_midnight_calendar(
    modules, plant, ireland_data, config, offset, side
):
    slot = modules.market.SLOT
    day = pd.Timestamp("2025-01-02")
    at = day + offset * slot
    now = at - 9 * slot
    ledger = modules.market.PositionLedger()
    plant._state = replace(plant.state, backlog_work_mwh=1000.0)
    control = modules.controllers.ThreeControllers(plant, ireland_data, ledger, config)
    control.plan.set(at, at + slot, 0.5)
    forecast = modules.forecast.DeliveryForecast(
        plant, control.profile, control.plan.service, now
    )
    preview = forecast.fork()
    preview.advance(at)
    low = preview.fork().advance(at + slot, 0.0).mean_mw
    baseline = preview.fork().advance(at + slot, 0.5).mean_mw
    high = preview.fork().advance(at + slot, 1.0).mean_mw
    committed = (low + high) / 2.0
    ledger.record_trade(
        modules.market.Trade("EOM", "da", at, at + slot, committed, 100, now)
    )
    ledger.record_price(
        "EOM", day, day + pd.Timedelta(days=1), 100, day - pd.Timedelta(hours=12)
    )
    data = SimpleNamespace(forecast=lambda market, at: 50.0 if side == "buy" else 150.0)
    decisions = control.intraday.decide(
        [(at, at + slot, None)],
        forecast,
        data,
        ledger,
        3000,
        -500,
        now,
    )
    assert len(decisions) == 1
    bid = decisions[0]
    assert bid.start == at and bid.end == at + slot
    sign = 1 if side == "buy" else -1
    headroom = high - baseline if side == "buy" else baseline - low
    assert bid.purchase_mw == pytest.approx(sign * min((high - low) / 2.0, headroom))
    assert (
        bid.forecast["delivery"]["facility_mwh"] / 0.25 - baseline
    ) == pytest.approx(bid.purchase_mw, rel=0, abs=1e-8)
    margin = (
        control.intraday.policy.margin_buy
        if side == "buy"
        else control.intraday.policy.margin_sell
    )
    assert bid.price_eur_mwh == 100 - sign * margin
    assert control.intraday.policy.LEAD_INTERVALS == 9
    assert control.intraday.policy.SLOTS_PER_GATE == 1
    assert control.intraday.calls == 1
    assert set(control.intraday.last_call["da_reference_sources"]) == {"EOM_clearing"}


def test_balancing_targets_the_revised_plan_after_intraday_fill(
    modules, plant, ireland_data, config
):
    start = pd.Timestamp("2025-01-02")
    end = start + modules.market.SLOT
    plant._state = replace(plant.state, backlog_work_mwh=1000.0)
    ledger = modules.market.PositionLedger()
    control = modules.controllers.ThreeControllers(plant, ireland_data, ledger, config)
    forecast = modules.forecast.DeliveryForecast(plant, control.profile, {}, start)
    low = forecast.fork().advance(end, 0.0).mean_mw
    high = forecast.fork().advance(end, 1.0).mean_mw
    da_bid = modules.market.Bid(
        "EOM",
        start,
        end,
        forecast.fork().advance(end, 0.5).mean_mw,
        3000,
        0.5,
        1 / (high - low),
    )
    da_trade = modules.market.Trade(
        "EOM", "da", start, end, da_bid.purchase_mw, 100, start
    )
    ledger.record_trade(da_trade)
    control.on_fill(da_bid, da_trade)
    assert control.plan.at(start, plant) == 0.5
    id_bid = modules.market.Bid("IDM", start, end, -2.0, 150, 0.5, 1 / (high - low))
    id_trade = modules.market.Trade("IDM", "id", start, end, -2.0, 150, start)
    ledger.record_trade(id_trade)
    control.on_fill(id_bid, id_trade)
    revised = 0.5 - 2.0 / (high - low)
    assert control.plan.at(start, plant) == pytest.approx(revised)
    assert control.plan.cap(start) == pytest.approx(revised)
    bid = control.bid("BM", [(start, end, None)], 3000, -500, start)[0]
    assert bid.purchase_mw == pytest.approx(
        forecast.fork().advance(end, revised).mean_mw - (da_bid.purchase_mw - 2.0),
        abs=1e-6,
    )
    assert len(control.plan_revisions) == 2
    assert control.plan_revisions[1]["clipped_service"] == 0.0
    with pytest.raises(ValueError, match="does not match"):
        control.on_fill(id_bid, da_trade)


def test_partial_day_ahead_fill_and_clipped_intraday_revision(
    modules, plant, ireland_data, config
):
    start = pd.Timestamp("2025-01-02")
    end = start + pd.Timedelta(hours=1)
    control = modules.controllers.ThreeControllers(
        plant, ireland_data, modules.market.PositionLedger(), config
    )
    bid = modules.market.Bid("EOM", start, end, 100.0, 3000, 0.4, 0.01)
    control.on_fill(
        bid, modules.market.Trade("EOM", "da", start, end, 90.0, 100, start)
    )
    assert control.plan.at(start, plant) == pytest.approx(0.3)
    assert control.plan.at(start + 3 * modules.market.SLOT, plant) == pytest.approx(0.3)
    slot_end = start + modules.market.SLOT
    sell = modules.market.Bid("IDM", start, slot_end, -50.0, 150, 0.3, 0.01)
    control.on_fill(
        sell, modules.market.Trade("IDM", "id", start, slot_end, -50.0, 150, start)
    )
    assert control.plan.at(start, plant) == 0.0
    assert control.plan.at(start + modules.market.SLOT, plant) == pytest.approx(0.3)
    assert control.plan_revisions[-1]["clipped_service"] == pytest.approx(-0.2)
    assert control.plan.cap(pd.Timestamp("2025-01-03")) == 1.0


def test_da_receives_delivery_backlog_not_opening_inventory(
    modules, plant, ireland_data, config
):
    plant._state = replace(plant.state, backlog_work_mwh=10.0, batch_arrived_mw=2.0)
    control = modules.controllers.ThreeControllers(
        plant, ireland_data, modules.market.PositionLedger(), config
    )
    now = pd.Timestamp("2025-01-01")
    start = now + pd.Timedelta(days=1)
    bids = control.bid(
        "EOM", [(start, start + pd.Timedelta(hours=1), None)], 3000, -500, now
    )
    call = control.day_ahead.last_call
    assert call["inputs"]["queued_chunks"] == [(0.0, 1)]
    assert call["lead_projection"]["backlog_start_work_mwh"] == 10.0
    assert call["lead_projection"]["served_work_mwh"] == pytest.approx(58.0)
    assert bids[0].forecast["delivery"]["served_work_mwh"] == pytest.approx(2.0)
    assert control.plan.service == {}
    assert plant.state.backlog_work_mwh == 10.0


@pytest.mark.parametrize("price", (50.0, 150.0))
def test_empty_queue_has_no_intraday_service_headroom(
    modules, plant, ireland_data, config, price
):
    plant._state = replace(plant.state, backlog_work_mwh=0.0, batch_arrived_mw=0.0)
    control = modules.controllers.ThreeControllers(
        plant, ireland_data, modules.market.PositionLedger(), config
    )
    now = pd.Timestamp("2025-01-02")
    start = now + 9 * modules.market.SLOT
    end = start + modules.market.SLOT
    control.ledger.record_price("EOM", now, now + pd.Timedelta(days=1), 100.0, now)
    control.data = SimpleNamespace(forecast=lambda market, at: price)
    bids = control.bid("IDM", [(start, end, None)], 3000, -500, now)
    assert bids == []
    assert control.intraday.calls == 1
    assert control.intraday.last_call["inputs"]["queue_cap_mw"] == 0.0
    assert control.intraday.last_call["inputs"]["max_defer_mw"] == 0.0


@pytest.mark.parametrize("fraction", (0.5, 1.0))
def test_accepted_id_work_is_unavailable_to_later_delivery(
    modules, plant, ireland_data, config, fraction
):
    plant._state = replace(plant.state, backlog_work_mwh=10.0, batch_arrived_mw=0.0)
    ledger = modules.market.PositionLedger()
    control = modules.controllers.ThreeControllers(plant, ireland_data, ledger, config)
    now = pd.Timestamp("2025-01-02")
    slot = modules.market.SLOT
    start, end = now + 9 * slot, now + 10 * slot
    control.plan.set(now, end + slot, 0.0)
    ledger.record_price("EOM", now, now + pd.Timedelta(days=1), 100, now)
    control.data = SimpleNamespace(forecast=lambda market, at: 50.0)
    first = control.bid("IDM", [(start, end, None)], 3000, -500, now)[0]
    assert control.plan.cap(start) == 0.0
    accepted = modules.market.Trade(
        "IDM", "first", start, end, fraction * first.purchase_mw, 95, now + slot
    )
    ledger.record_trade(accepted)
    control.on_fill(first, accepted)
    assert control.plan_revisions[-1]["clipped_service"] == 0.0
    consumed = control.plan.cap(start) * plant.batch_capacity_mw * 0.25
    assert consumed > 0
    later = control.bid("IDM", [(end, end + slot, None)], 3000, -500, now + slot)[0]
    projection = control.intraday.last_call["delivery_projection"]
    assert projection["baseline"]["backlog_start_work_mwh"] == pytest.approx(
        10.0 - consumed, rel=0, abs=1e-10
    )
    assert projection["maximum"]["served_work_mwh"] == pytest.approx(
        10.0 - consumed, rel=0, abs=1e-10
    )
    assert later.forecast["delivery"]["served_work_mwh"] + consumed <= 10.0 + 1e-10


def test_bm_overpurchase_changes_only_through_accepted_offsets(
    modules, plant, ireland_data, config
):
    plant._state = replace(plant.state, backlog_work_mwh=0.0, batch_arrived_mw=2.0)
    start = pd.Timestamp("2025-01-02")
    end = start + modules.market.SLOT
    ledger = modules.market.PositionLedger()
    control = modules.controllers.ThreeControllers(plant, ireland_data, ledger, config)
    control.plan.set(start, end, 1.0)
    ledger.record_trade(
        modules.market.Trade("EOM", "excess", start, end, 250.0, 100, start)
    )
    bid = control.bid("BM", [(start, end, None)], 3000, -500, start)[0]
    assert bid.purchase_mw < 0
    assert bid.price_eur_mwh == -500
    assert ledger.energy(start, end) == 62.5
    expected = bid.forecast["delivery"]["facility_mwh"]
    assert expected / 0.25 - 250.0 == pytest.approx(bid.purchase_mw)
    partial = modules.market.Trade(
        "BM", "offset", start, end, bid.purchase_mw / 2, 80, start
    )
    ledger.record_trade(partial)
    control.on_fill(bid, partial)
    ledger.record_price("BM", start, end, 80, start)
    settlement = ledger.settle(start, end, expected, scored=True)
    assert settlement["imbalance_mwh"] == pytest.approx(bid.purchase_mw * 0.25 / 2)
    assert control.plan.cap(start) == 1.0


def test_da_work_below_idle_floor_has_no_releasable_energy(
    modules, plant, ireland_data, config
):
    plant._state = replace(
        plant.state,
        base_a_mw=1.0,
        base_b_mw=1.0,
        backlog_work_mwh=0.0,
        batch_arrived_mw=2.0,
    )
    start = pd.Timestamp("2025-01-02")
    end = start + pd.Timedelta(hours=1)
    control = modules.controllers.ThreeControllers(
        plant, ireland_data, modules.market.PositionLedger(), config
    )
    bid = control.bid("EOM", [(start, end, None)], 3000, -500, start)[0]
    assert bid.service_fraction > 0
    assert bid.forecast["delivery"]["served_work_mwh"] == pytest.approx(2.0)
    assert bid.service_per_mw == 0.0
    control.on_fill(
        bid,
        modules.market.Trade(
            "EOM", "partial", start, end, bid.purchase_mw / 2, 100, start
        ),
    )
    assert control.plan.cap(start) == bid.service_fraction
