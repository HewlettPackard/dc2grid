import pickle
import random
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest


def flat_workload(plant, backlog=0.0, arrivals=9.0):
    plant._state = replace(
        plant.state,
        base_a_mw=30.0,
        base_b_mw=20.0,
        batch_arrived_mw=arrivals,
        backlog_work_mwh=backlog,
    )
    plant.workload._backlog_kw = backlog * 3_600_000 / plant.dt_seconds
    plant.workload._p_base_a_arr[:] = 30_000.0
    plant.workload._p_base_b_arr[:] = 20_000.0
    plant.workload._p_flex_arr[:] = arrivals * 1000


@pytest.mark.parametrize("rate", (0.0, 0.5, 1.0))
@pytest.mark.parametrize("backlog", (0.0, 2.6))
def test_executable_energy_matches_nominal_physics_through_queue_drain(
    modules, plant, rate, backlog
):
    flat_workload(plant, backlog)
    start = pd.Timestamp("2025-01-02")
    forecast = modules.forecast.DeliveryForecast(
        plant, modules.forecast.ObservedProfile(), {}, start
    )
    window = forecast.advance(start + modules.market.SLOT, rate)
    reference = deepcopy(plant)
    action = reference.nominal_action()
    action[0] = rate
    served, metered = 0.0, 0.0
    for _ in range(180):
        state = reference.step(action, plant.state.ambient_c)
        served += state.batch_served_mw * plant.dt_seconds / 3600
        metered += state.net_import_mw * plant.dt_seconds / 3600
    assert window.facility_mwh == pytest.approx(metered, rel=0, abs=1e-10)
    assert window.served_work_mwh == pytest.approx(served, rel=0, abs=1e-10)
    assert window.backlog_end_work_mwh == pytest.approx(
        reference.state.backlog_work_mwh, rel=0, abs=1e-10
    )
    assert backlog + window.arrived_work_mwh == pytest.approx(
        window.served_work_mwh + window.backlog_end_work_mwh, rel=0, abs=1e-10
    )
    assert window.served_work_mwh <= backlog + 9.0 * 0.25 + 1e-10


def test_lead_consumes_work_and_empty_projection_is_noop(modules, plant):
    flat_workload(plant, backlog=2.6)
    start = pd.Timestamp("2025-01-02")
    forecast = modules.forecast.DeliveryForecast(
        plant, modules.forecast.ObservedProfile(), {}, start
    )
    empty = forecast.advance(start)
    assert empty.arrived_work_mwh == empty.served_work_mwh == empty.facility_mwh == 0
    assert empty.backlog_end_work_mwh == 2.6
    lead = forecast.advance(start + 9 * modules.market.SLOT)
    assert lead.backlog_end_work_mwh == 0
    assert lead.served_work_mwh == pytest.approx(2.6 + 9 * 2.25)
    delivery = forecast.advance(start + 10 * modules.market.SLOT)
    assert delivery.served_work_mwh == pytest.approx(2.25)
    assert delivery.backlog_start_work_mwh == 0


def test_forecast_does_not_change_live_state_or_use_future_trace(modules, plant):
    start = pd.Timestamp("2025-01-02")
    before = pickle.dumps(plant)
    random_before = pickle.dumps((random.getstate(), np.random.get_state()))
    profile = modules.forecast.ObservedProfile()
    projection = modules.forecast.DeliveryForecast(plant, profile, {}, start)
    expected = projection.advance(start + modules.market.SLOT)
    projection.fork().advance(start + 2 * modules.market.SLOT)
    assert pickle.dumps(plant) == before
    assert pickle.dumps((random.getstate(), np.random.get_state())) == random_before
    for field in ("_p_flex_arr", "_p_base_a_arr", "_p_base_b_arr"):
        getattr(plant.workload, field)[:] = np.nan
    actual = modules.forecast.DeliveryForecast(plant, profile, {}, start).advance(
        start + modules.market.SLOT
    )
    assert asdict(actual) == asdict(expected)


def test_profile_and_plan_are_snapshotted_per_decision(modules, plant):
    start = pd.Timestamp("2025-01-02")
    profile = modules.forecast.ObservedProfile()
    service = {start: 0.0}
    old = modules.forecast.DeliveryForecast(plant, profile, service, start)
    for offset in range(4):
        profile.record(
            start - pd.Timedelta(days=1) + offset * modules.market.SLOT,
            {"base_a_mw": 30.0, "base_b_mw": 20.0, "batch_arrived_mw": 2.0},
        )
    service[start] = 1.0
    assert old.inputs(start)[1] == "current_persistence"
    assert old.advance(start + modules.market.SLOT).served_work_mwh == 0
    new = modules.forecast.DeliveryForecast(plant, profile, service, start)
    assert new.inputs(start)[1] == "observed_profile"
    assert new.advance(start + modules.market.SLOT).served_work_mwh == pytest.approx(
        0.5
    )


@pytest.mark.parametrize("invalid", (-1.0, float("nan"), float("inf")))
def test_profile_rejects_invalid_workload(modules, invalid):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        modules.forecast.ObservedProfile().record(
            pd.Timestamp("2025-01-01"),
            {"base_a_mw": 30.0, "base_b_mw": 20.0, "batch_arrived_mw": invalid},
        )


def test_forecast_rejects_invalid_calendar_and_service(modules, plant):
    start = pd.Timestamp("2025-01-02")
    profile = modules.forecast.ObservedProfile()
    with pytest.raises(ValueError, match="origins"):
        modules.forecast.DeliveryForecast(
            plant, profile, {}, start + pd.Timedelta(seconds=5)
        )
    forecast = modules.forecast.DeliveryForecast(plant, profile, {}, start)
    for end in (start - modules.market.SLOT, start + pd.Timedelta(seconds=5)):
        with pytest.raises(ValueError, match="nonpast settlement boundary"):
            forecast.advance(end)
    with pytest.raises(ValueError, match="service fraction"):
        forecast.advance(start + modules.market.SLOT, float("nan"))
