from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .market import SLOT, hours, timestamp
from .plant import C2GPlant


class ObservedProfile:
    """Trailing same-time-of-day means of observed plant inputs (no trace lookahead)."""

    FIELDS = ("base_a_mw", "base_b_mw", "batch_arrived_mw")

    def __init__(self):
        self.slots: dict[pd.Timedelta, dict[str, float]] = {}

    def record(self, slot_start, means: dict[str, float]) -> None:
        slot = timestamp(slot_start)
        if slot != slot.floor("15min"):
            raise ValueError("Profile samples must start on a settlement slot.")
        values = {field: float(means[field]) for field in self.FIELDS}
        if not np.isfinite(list(values.values())).all() or min(values.values()) < 0:
            raise ValueError("Profile samples must be finite and nonnegative.")
        self.slots[slot - slot.normalize()] = values

    def hour(self, start) -> dict[str, float] | None:
        start = timestamp(start)
        if start != start.floor("h"):
            raise ValueError("Profile lookups are per hourly DA product.")
        keys = [start + i * SLOT - start.normalize() for i in range(4)]
        if any(key not in self.slots for key in keys):
            return None
        return {
            field: float(np.mean([self.slots[key][field] for key in keys]))
            for field in self.FIELDS
        }


@dataclass(frozen=True)
class ForecastWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    backlog_start_work_mwh: float
    backlog_end_work_mwh: float
    arrived_work_mwh: float
    served_work_mwh: float
    facility_mwh: float

    @property
    def mean_mw(self) -> float:
        return self.facility_mwh / hours(self.start, self.end)


class DeliveryForecast:
    """Causal planned-service projection with nominal cooling and an idle BESS."""

    def __init__(self, plant: C2GPlant, profile: ObservedProfile, service: dict, now):
        self.plant = plant.copy_physics()
        self.capacity_mw = plant.batch_capacity_mw
        self.now = timestamp(now)
        self.backlog_work_mwh = plant.state.backlog_work_mwh
        self.ambient_c = plant.state.ambient_c
        self.service = service.copy()
        self.persistence = {
            field: getattr(plant.state, field) for field in ObservedProfile.FIELDS
        }
        self.hourly = {
            hour: profile.hour(self.now.normalize() + pd.Timedelta(hours=hour))
            for hour in range(24)
        }
        self.cooling = tuple(float(value) for value in plant.nominal_action()[1:3])
        if self.now != self.now.floor("15min"):
            raise ValueError("Forecast origins must align with settlement slots.")
        values = [self.backlog_work_mwh, *self.persistence.values()]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Forecast workload must be finite and nonnegative.")

    def fork(self) -> DeliveryForecast:
        result = copy(self)
        result.plant = self.plant.copy_physics()
        return result

    def inputs(self, at) -> tuple[dict[str, float], str]:
        values = self.hourly[timestamp(at).hour]
        if values is None:
            return self.persistence, "current_persistence"
        return values, "observed_profile"

    def advance(self, end, service_fraction: float | None = None) -> ForecastWindow:
        end = timestamp(end)
        if end < self.now or end != end.floor("15min"):
            raise ValueError("Forecast end must be a nonpast settlement boundary.")
        if service_fraction is not None and not 0 <= service_fraction <= 1:
            raise ValueError("Forecast service fraction must be in [0, 1].")
        start, backlog = self.now, self.backlog_work_mwh
        arrived, served, energy = 0.0, 0.0, 0.0
        while self.now < end:
            at = self.now
            values, _ = self.inputs(at)
            rate = self.service.get(at, 1.0)
            if service_fraction is not None:
                rate = service_fraction
            work, consumption = self._advance_slot(values, rate)
            arrived += values["batch_arrived_mw"] * hours(at, at + SLOT)
            served += work
            energy += consumption
            self.now += SLOT
        return ForecastWindow(
            start, end, backlog, self.backlog_work_mwh, arrived, served, energy
        )

    def _advance_slot(
        self, values: dict[str, float], rate: float
    ) -> tuple[float, float]:
        dt_hours = self.plant.dt_seconds / 3600
        arrivals = values["batch_arrived_mw"] * dt_hours
        capacity = self.capacity_mw * rate * dt_hours
        served, energy = 0.0, 0.0
        for _ in range(round(SLOT.total_seconds() / self.plant.dt_seconds)):
            available = self.backlog_work_mwh + arrivals
            work = min(available, capacity)
            self.backlog_work_mwh = available - work
            result = self.plant._facility(
                values["base_a_mw"],
                values["base_b_mw"],
                work / dt_hours,
                *self.cooling,
                self.ambient_c,
            )
            served += work
            energy += result["p_facility_mw"] * dt_hours
        return served, energy

    def service_limit(self, end) -> float:
        arrivals = sum(
            self.inputs(at)[0]["batch_arrived_mw"] * hours(at, at + SLOT)
            for at in pd.date_range(self.now, end, freq=SLOT, inclusive="left")
        )
        return min(
            1.0,
            (self.backlog_work_mwh + arrivals)
            / (self.capacity_mw * hours(self.now, end)),
        )

    def rate_for_power(self, end, target_mw: float, maximum_rate: float) -> float:
        return float(
            brentq(
                lambda rate: self.fork().advance(end, rate).mean_mw - target_mw,
                0.0,
                maximum_rate,
            )
        )
