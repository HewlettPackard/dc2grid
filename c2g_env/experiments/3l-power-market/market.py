from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Protocol

import numpy as np
import pandas as pd

MARKETS = ("EOM", "IDM", "BM")
SLOT = pd.Timedelta(minutes=15)


def timestamp(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("Use the 3L scenario's finite, timezone-naive timestamps.")
    return result


def hours(start, end) -> float:
    duration = (timestamp(end) - timestamp(start)).total_seconds() / 3600
    if duration <= 0:
        raise ValueError("Delivery end must follow delivery start.")
    return duration


def overlap_hours(start, end, other_start, other_end) -> float:
    return max(
        0.0, (min(end, other_end) - max(start, other_start)).total_seconds() / 3600
    )


@dataclass(frozen=True)
class Bid:
    market: str
    start: pd.Timestamp
    end: pd.Timestamp
    purchase_mw: float
    price_eur_mwh: float
    service_fraction: float
    # Service-fraction change per accepted MW; converts fills into plan revisions.
    service_per_mw: float = 0.0
    forecast: dict | None = None


@dataclass(frozen=True)
class Trade:
    market: str
    bid_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    purchase_mw: float
    price_eur_mwh: float
    cleared_at: pd.Timestamp


class PositionLedger:
    def __init__(self):
        self.trades: dict[tuple, Trade] = {}
        self.prices: dict[tuple, tuple[float, pd.Timestamp]] = {}
        self.settled: set[pd.Timestamp] = set()
        self.revision = 0

    def _record_unique(self, records, key, value, conflict_message) -> bool:
        if key in records:
            if records[key] != value:
                raise ValueError(conflict_message)
            return False
        records[key] = value
        self.revision += 1
        return True

    def record_trade(self, trade: Trade) -> bool:
        self._validate_product(trade.market, trade.start, trade.end, trade.cleared_at)
        if not np.isfinite([trade.purchase_mw, trade.price_eur_mwh]).all():
            raise ValueError("Trade quantity and price must be finite.")
        key = (trade.market, trade.bid_id, trade.start, trade.end)
        return self._record_unique(
            self.trades, key, trade, f"Conflicting duplicate clearing: {key}"
        )

    @staticmethod
    def _validate_product(market, start, end, cleared_at) -> None:
        if market not in MARKETS:
            raise ValueError(f"Unsupported market: {market}")
        hours(start, end)
        if timestamp(cleared_at) > timestamp(start):
            raise ValueError("Clearing must not arrive after delivery starts.")
        if any(timestamp(t) != timestamp(t).floor("15min") for t in (start, end)):
            raise ValueError("Delivery must align with the 3L settlement clock.")

    def record_price(self, market, start, end, price, cleared_at) -> None:
        start, end, cleared_at = map(timestamp, (start, end, cleared_at))
        self._validate_product(market, start, end, cleared_at)
        if not np.isfinite(price):
            raise ValueError("Clearing price must be finite.")
        key = (market, start, end)
        value = (float(price), cleared_at)
        self._record_unique(self.prices, key, value, f"Conflicting price for {key}")

    def price(self, market, start, end, as_of) -> float | None:
        matches = [
            price
            for (name, left, right), (price, cleared) in self.prices.items()
            if name == market and left <= start and right >= end and cleared <= as_of
        ]
        if len(matches) > 1:
            raise ValueError("Ambiguous overlapping market prices.")
        return matches[0] if matches else None

    def energy(self, start, end, market: str | None = None) -> float:
        return sum(
            trade.purchase_mw * overlap_hours(start, end, trade.start, trade.end)
            for trade in self.trades.values()
            if market is None or trade.market == market
        )

    def trade_cost(self, start, end, market: str | None = None) -> float:
        return sum(
            trade.purchase_mw
            * trade.price_eur_mwh
            * overlap_hours(start, end, trade.start, trade.end)
            for trade in self.trades.values()
            if market is None or trade.market == market
        )

    def settle(self, start, end, metered_mwh: float, *, scored: bool) -> dict:
        start, end = timestamp(start), timestamp(end)
        if end - start != SLOT:
            raise ValueError("Settle one complete 15-minute delivery interval.")
        if start in self.settled:
            raise ValueError(f"Delivery interval already settled: {start}")
        if not np.isfinite(metered_mwh):
            raise ValueError("Metered energy must be finite.")
        price = self.price("BM", start, end, end)
        if scored and price is None:
            raise RuntimeError(
                f"No live BM clearing price for scored interval {start}."
            )
        positions = {market: self.energy(start, end, market) for market in MARKETS}
        imbalance = metered_mwh - sum(positions.values())
        procurement = self.trade_cost(start, end)
        imbalance_cost = None if price is None else imbalance * price
        self.settled.add(start)
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "scored": scored,
            "metered_mwh": metered_mwh,
            "positions_mwh": positions,
            "imbalance_mwh": imbalance,
            "bm_price_eur_mwh": price,
            "procurement_cost_eur": procurement,
            "costs_by_market_eur": {
                name: self.trade_cost(start, end, name) for name in MARKETS
            },
            "imbalance_cost_eur": imbalance_cost,
            "total_cost_eur": None
            if imbalance_cost is None
            else procurement + imbalance_cost,
            "status": "unpriced_warmup" if price is None else "settled",
        }


class ThreeLData:
    def __init__(self, scenario: Path):
        self.temperature = self._read(scenario / "temperature_df.csv", ["temp_c"])
        self.forecasts = self._read(
            scenario / "forecasts_df.csv", ["price_EOM", "price_IDM"]
        )
        self.imbalance = self._read(
            scenario / "system_imbalance_df.csv", ["net_imbalance_mw"]
        )
        demand = self._read(scenario / "demand_df.csv")
        self.demand = demand.sum(axis=1)
        ratings = pd.read_csv(scenario / "demand_units.csv")["max_power"]
        if not np.isfinite(ratings).all() or (ratings > 0).any():
            raise ValueError("3L demand-unit ratings must be finite and nonpositive.")
        self.demand_scale_mw = float(-ratings.sum())
        if self.demand_scale_mw <= 0:
            raise ValueError("3L demand-unit ratings must define positive capacity.")

    @staticmethod
    def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if columns is not None:
            frame = frame[columns]
        if (
            frame.empty
            or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
        ):
            raise ValueError(f"Invalid time index: {path}")
        if (
            frame.index.tz is not None
            or not np.isfinite(frame.to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"Invalid timestamps or numeric values: {path}")
        if not (frame.index.to_series().diff().dropna() == SLOT).all():
            raise ValueError(f"Expected complete 15-minute data: {path}")
        return frame

    def require_coverage(self, start, end) -> None:
        for data in (self.temperature, self.forecasts, self.imbalance, self.demand):
            if start < data.index[0] or end > data.index[-1] + SLOT:
                raise ValueError(f"3L input does not cover [{start}, {end}).")

    def context(self, when) -> dict[str, float]:
        slot = timestamp(when).floor("15min")
        return {
            "ambient_c": float(self.temperature.loc[slot, "temp_c"]),
            "system_demand_mw": float(self.demand.loc[slot]),
            "system_imbalance_mw": float(self.imbalance.loc[slot, "net_imbalance_mw"]),
        }

    def forecast(self, market: str, when) -> float:
        if market not in ("EOM", "IDM"):
            raise ValueError("3L supplies EOM/IDM forecasts, not a BM forecast.")
        return float(
            self.forecasts.loc[timestamp(when).floor("15min"), f"price_{market}"]
        )


class ExternalClock(Protocol):
    time: float

    def get_next_activity(self) -> float | None: ...

    def set_time(self, value: float) -> None: ...


async def advance_clock(
    clock: ExternalClock, target: float, drain: Callable[[], Awaitable[None]]
) -> None:
    if target < clock.time:
        raise ValueError("Market time cannot move backwards.")
    await drain()
    while clock.time < target:
        upcoming = clock.get_next_activity()
        next_time = target if upcoming is None else min(target, upcoming)
        if next_time <= clock.time:
            raise RuntimeError(
                "The market clock has an undrained or non-advancing event."
            )
        clock.set_time(next_time)
        await drain()


def epoch_seconds(when: datetime | pd.Timestamp) -> float:
    return timestamp(when).value / 1_000_000_000
