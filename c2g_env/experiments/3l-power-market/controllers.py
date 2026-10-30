from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .assume_backend import import_three_l
from .config import REPO_ROOT
from .forecast import DeliveryForecast, ObservedProfile
from .market import SLOT, Bid, PositionLedger, ThreeLData, Trade, hours, timestamp
from .plant import C2GPlant


class OperatingPlan:
    def __init__(self):
        self.service: dict[pd.Timestamp, float] = {}

    def set(self, start, end, service: float) -> None:
        if not 0 <= service <= 1:
            raise ValueError("Scheduled service must be in [0, 1].")
        for slot in pd.date_range(start, end, freq=SLOT, inclusive="left"):
            self.service[slot] = float(service)

    def at(self, when, plant: C2GPlant) -> float:
        slot = timestamp(when).floor("15min")
        if slot in self.service:
            return self.service[slot]
        # Before the first DA delivery, serve the observed arrival rate.
        return min(1.0, plant.state.batch_arrived_mw / plant.batch_capacity_mw)

    def cap(self, when) -> float:
        # Hardware throttle ceiling; unscheduled slots are uncapped.
        return self.service.get(timestamp(when).floor("15min"), 1.0)


class DayAheadController:
    def __init__(self, policy):
        self.policy = policy
        self.calls = 0
        self.last_call: dict = {}

    def decide(
        self, products, forecast: DeliveryForecast, data: ThreeLData, ceiling: float
    ) -> list[Bid]:
        products = sorted(products)
        hour = pd.Timedelta(hours=1)
        expected = [
            (products[0][0] + i * hour, products[0][0] + (i + 1) * hour, None)
            for i in range(len(products))
        ]
        if products != expected:
            raise ValueError("Native L1 requires consecutive hourly DA products.")
        lead = forecast.advance(products[0][0])
        hourly, sources = zip(*(forecast.inputs(start) for start, _, _ in products))
        inputs = {
            "price_eur_per_mwh": np.array(
                [data.forecast("EOM", start) for start, _, _ in products]
            ),
            "defer_arrivals_mwh": np.array(
                [values["batch_arrived_mw"] for values in hourly]
            ),
            "queued_chunks": [(forecast.backlog_work_mwh, len(products))],
            "max_defer_mw": forecast.capacity_mw,
            "deadline_hours": len(products),
            "horizon_hours": len(products),
        }
        self.calls += 1
        result = self.policy.plan(**inputs)
        self.last_call = {
            "inputs": inputs,
            "lead_projection": asdict(lead),
            "workload_forecast_sources": list(sources),
            "output": asdict(result),
        }
        return [
            self._bid(forecast, end, float(rate), ceiling)
            for (_, end, _), rate in zip(products, result.rates, strict=True)
        ]

    @staticmethod
    def _bid(forecast, end, rate, ceiling) -> Bid:
        low = forecast.fork().advance(end, 0.0)
        delivery = forecast.advance(end, rate)
        marginal = delivery.mean_mw - low.mean_mw
        if marginal < 0:
            raise ValueError("Executable batch service must not reduce facility power.")
        # Work below the electrical idle floor has no incremental energy to release.
        conversion = rate / marginal if marginal > 0 else 0.0
        return Bid(
            "EOM",
            delivery.start,
            end,
            delivery.mean_mw,
            ceiling,
            rate,
            conversion,
            {"delivery": asdict(delivery)},
        )


class IntradayController:
    def __init__(self, policy):
        self.policy = policy
        self.calls = 0
        self.last_call: dict = {}

    @staticmethod
    def _reference_prices(data, ledger, day, now):
        prices, sources = [], []
        for at in pd.date_range(day, periods=24, freq="h"):
            price = ledger.price("EOM", at, at + pd.Timedelta(hours=1), now)
            source = "EOM_clearing"
            if price is None:
                price = data.forecast("EOM", at)
                source = "supplied_EOM_forecast"
            prices.append(price)
            sources.append(source)
        return np.array(prices), sources

    def _inputs(self, forecast, data, ledger, start, end, now):
        day = start.normalize()
        slots = pd.date_range(day, periods=96, freq=SLOT)
        maximum_rate = forecast.service_limit(end)
        baseline_rate = min(forecast.service.get(start, 1.0), maximum_rate)
        low = forecast.fork().advance(end, 0.0)
        baseline = forecast.fork().advance(end, baseline_rate)
        high = forecast.fork().advance(end, maximum_rate)
        da_prices, sources = self._reference_prices(data, ledger, day, now)
        slot_hours = hours(day, day + SLOT)
        position = ledger.energy(start, end) / slot_hours
        # L2 shifts executable service; BM corrects the position-to-load mismatch.
        buy_headroom = max(0.0, high.mean_mw - baseline.mean_mw)
        sell_headroom = max(0.0, baseline.mean_mw - low.mean_mw)
        inputs = {
            "current_interval": int((now - day) / SLOT),
            "committed_mw": np.array(
                [ledger.energy(at, at + SLOT) / slot_hours for at in slots]
            ),
            "idm_forecast": np.array([data.forecast("IDM", at) for at in slots]),
            "da_price_24h": da_prices,
            "max_total_mw": position + buy_headroom,
            "max_defer_mw": buy_headroom + sell_headroom,
            "min_total_mw": position - sell_headroom,
            "queue_cap_mw": buy_headroom,
        }
        projection = {
            "minimum": asdict(low),
            "baseline": asdict(baseline),
            "maximum": asdict(high),
            "baseline_service_fraction": baseline_rate,
            "maximum_service_fraction": maximum_rate,
            "workload_inputs": forecast.inputs(start)[0],
            "workload_forecast_source": forecast.inputs(start)[1],
        }
        return inputs, sources, projection

    def decide(self, products, forecast, data, ledger, ceiling, floor, now):
        if len(products) != 1 or products[0][1] - products[0][0] != SLOT:
            raise ValueError("Ireland IDM requires one 15-minute delivery product.")
        start, end, _ = products[0]
        if now != now.floor("15min") or start != start.floor("15min"):
            raise ValueError("IDM opening and delivery must align with the calendar.")
        self.policy.LEAD_INTERVALS = int((start - now) / SLOT)
        self.policy.SLOTS_PER_GATE = len(products)
        lead = forecast.advance(start)
        inputs, sources, projection = self._inputs(
            forecast, data, ledger, start, end, now
        )
        self.calls += 1
        result = self.policy.bid_gate(**inputs)
        self.last_call = {
            "inputs": inputs,
            "da_reference_sources": sources,
            "lead_projection": asdict(lead),
            "delivery_projection": projection,
            "lead_intervals": self.policy.LEAD_INTERVALS,
            "slots_per_gate": self.policy.SLOTS_PER_GATE,
            "output": [asdict(bid) for bid in result],
        }
        return [
            self._bid(bid, forecast, end, projection, ceiling, floor) for bid in result
        ]

    @staticmethod
    def _bid(bid, forecast, end, projection, ceiling, floor) -> Bid:
        quantity = bid.quantity_mw if bid.side == "buy" else -bid.quantity_mw
        baseline = projection["baseline_service_fraction"]
        power = projection["baseline"]["facility_mwh"] / hours(forecast.now, end)
        rate = forecast.rate_for_power(
            end, power + quantity, projection["maximum_service_fraction"]
        )
        delivery = forecast.fork().advance(end, rate)
        return Bid(
            "IDM",
            forecast.now,
            end,
            quantity,
            float(np.clip(bid.price, floor, ceiling)),
            baseline,
            (rate - baseline) / quantity,
            {
                **projection,
                "delivery": asdict(delivery),
                "revised_service_fraction": rate,
            },
        )


def policy_identity(policy) -> str:
    return f"{type(policy).__module__}.{type(policy).__qualname__}"


class ThreeControllers:
    def __init__(
        self, plant: C2GPlant, data: ThreeLData, ledger: PositionLedger, config: dict
    ):
        self.plant, self.data, self.ledger = plant, data, ledger
        self.plan = OperatingPlan()
        self.profile = ObservedProfile()
        root = (REPO_ROOT / config["three_l"]["root"]).resolve()
        l1 = import_three_l(root, "policies.rule_based_l1")
        l2 = import_three_l(root, "policies.rule_based_l2")
        self.day_ahead = DayAheadController(l1.RuleBasedL1Policy())
        self.intraday = IntradayController(l2.RuleBasedL2Policy())
        self.decisions: list[dict] = []
        self.policy_calls: list[dict] = []
        self.plan_revisions: list[dict] = []

    def manifest(self) -> dict:
        return {
            "L1": policy_identity(self.day_ahead.policy),
            "L2": policy_identity(self.intraday.policy),
            "l2_parameters": vars(self.intraday.policy).copy(),
            "l1_deadlines": "planning horizon only; no per-job deadlines or drops",
            "l1_workload_forecast": (
                "trailing same-hour observed profile; current persistence "
                "until each full hour is observed; queue rolled to delivery start"
            ),
            "executable_forecast": {
                "workload": "5-second queue conservation under accepted service caps",
                "cooling": self.plant.nominal_action()[1:3].tolist(),
                "ambient": "current 3L ambient persisted over the projection",
                "battery": "idle; forecast facility demand equals forecast net import",
                "hardware": "assumes service at the cap, not future hardware actions",
                "intraday": (
                    "remaining executable service translated into position bounds"
                ),
                "partial_fills": (
                    "linear service interpolation within quoted endpoints; "
                    "zero marginal energy below the electrical idle floor "
                    "does not reduce service"
                ),
            },
            "l2_timing": "lead and window from actual ASSUME opening/products",
            "plan_updates": (
                "DA fills set the schedule (shortfall reduces service); "
                "IDM fills revise it; the schedule caps the hardware throttle"
            ),
        }

    def _balancing_bids(self, products, forecast, ceiling, floor) -> list[Bid]:
        result = []
        for start, end, _ in sorted(products):
            lead = forecast.advance(start)
            delivery = forecast.advance(end)
            rate = self.plan.cap(start)
            delta = delivery.mean_mw - self.ledger.energy(start, end) / hours(
                start, end
            )
            result.append(
                Bid(
                    "BM",
                    start,
                    end,
                    delta,
                    ceiling if delta > 0 else floor,
                    rate,
                    forecast={"lead": asdict(lead), "delivery": asdict(delivery)},
                )
            )
        return result

    def on_fill(self, bid: Bid, trade: Trade) -> None:
        if (bid.market, bid.start, bid.end) != (trade.market, trade.start, trade.end):
            raise ValueError("Fill does not match the submitted bid product.")
        if trade.market == "BM":
            return
        if trade.market == "EOM":
            shortfall = bid.purchase_mw - trade.purchase_mw
            proposed = bid.service_fraction - shortfall * bid.service_per_mw
        else:
            proposed = bid.service_fraction + trade.purchase_mw * bid.service_per_mw
        service = min(1.0, max(0.0, proposed))
        self.plan.set(trade.start, trade.end, service)
        self.plan_revisions.append(
            {
                "market": trade.market,
                "bid_id": trade.bid_id,
                "start": trade.start.isoformat(),
                "end": trade.end.isoformat(),
                "requested_mw": bid.purchase_mw,
                "accepted_mw": trade.purchase_mw,
                "service_per_mw": bid.service_per_mw,
                "base_service_fraction": bid.service_fraction,
                "accepted_service_fraction": service,
                "clipped_service": proposed - service,
            }
        )

    def _record_policy_call(self, market, controller, now) -> None:
        self.policy_calls.append(
            {
                "market": market,
                "decided_at": timestamp(now).isoformat(),
                "policy": policy_identity(controller.policy),
                **controller.last_call,
            }
        )

    def bid(self, market, products, ceiling, floor, now) -> list[Bid]:
        if not products:
            return []
        forecast = DeliveryForecast(self.plant, self.profile, self.plan.service, now)
        if market == "EOM":
            bids = self.day_ahead.decide(products, forecast, self.data, ceiling)
            self._record_policy_call(market, self.day_ahead, now)
        elif market == "IDM":
            bids = self.intraday.decide(
                products,
                forecast,
                self.data,
                self.ledger,
                ceiling,
                floor,
                now,
            )
            self._record_policy_call(market, self.intraday, now)
        elif market == "BM":
            bids = self._balancing_bids(products, forecast, ceiling, floor)
        else:
            raise ValueError(f"Unknown market: {market}")
        self._remember(bids, now)
        return bids

    def _remember(self, bids: list[Bid], now) -> None:
        for bid in bids:
            self.decisions.append(
                {
                    "decided_at": timestamp(now).isoformat(),
                    "market": bid.market,
                    "start": bid.start.isoformat(),
                    "end": bid.end.isoformat(),
                    "purchase_mw": bid.purchase_mw,
                    "limit_price_eur_mwh": bid.price_eur_mwh,
                    "service_fraction": bid.service_fraction,
                    "service_per_mw": bid.service_per_mw,
                    "forecast": bid.forecast,
                }
            )
