from __future__ import annotations

import numpy as np
import pandas as pd
from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.markets.clearing_algorithms.simple import PayAsClearRole
from assume_env.src.markets.ordc_clearing import PayAsClearORDCRole

from .market import SLOT


class C2GBiddingStrategy(MinMaxStrategy):
    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        return unit.bridge.orders(market_config, product_tuples)


class C2GDemandUnit(SupportsMinMax):
    def __init__(self, *, bridge, **kwargs):
        super().__init__(technology="DataCentre", **kwargs)
        self.bridge = bridge
        plant = bridge.controllers.plant
        electrical, thermal = plant.electrical, plant.thermal
        it_rating_mw = (
            electrical.n_racks_A * electrical.p_max_rack_A_kw
            + electrical.n_racks_B * electrical.p_max_rack_B_kw
        ) / 1000.0
        # Nameplate rating from equipment ratings, independent of the warm-start state.
        self.min_power = -(it_rating_mw + thermal.max_hvac_mw + thermal.P_PUMP_MAX_MW)
        self.max_power = 0.0

    def calculate_min_max_power(self, start, end, product_type="energy"):
        count = len(pd.date_range(start, end, freq=SLOT, inclusive="left"))
        return np.full(count, self.min_power), np.zeros(count)

    def calculate_marginal_cost(self, start, power):
        # Consumption is billed by the delivery ledger, not a generator fuel-cost model.
        return 0.0


class RecordedPayAsClear(PayAsClearRole):
    def __init__(self, marketconfig, *, recorder):
        super().__init__(marketconfig)
        self.recorder = recorder

    def clear(self, orderbook, market_products):
        result = super().clear(orderbook, market_products)
        self.recorder.capture(self.marketconfig.market_id, result)
        return result


class RecordedORDC(PayAsClearORDCRole):
    def __init__(self, marketconfig, *, recorder):
        super().__init__(marketconfig)
        self.recorder = recorder

    def clear(self, orderbook, market_products):
        result = super().clear(orderbook, market_products)
        self.recorder.capture(self.marketconfig.market_id, result)
        return result
