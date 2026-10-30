from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Protocol

import gymnasium as gym
import numpy as np
import pandas as pd

from .controllers import ThreeControllers
from .hardware_adapter import C2GHardwareAdapter
from .market import MARKETS, SLOT, PositionLedger, ThreeLData, timestamp
from .plant import C2GPlant

MARKET_OBSERVATION_FIELDS = (
    "da_energy_mwh",
    "id_energy_mwh",
    "bm_energy_mwh",
    "metered_mwh",
    "bm_price_eur_mwh",
    "bm_price_available",
    "system_demand_mw",
    "system_imbalance_mw",
    "service_target",
    "remaining_interval_seconds",
    "eom_forecast_eur_mwh",
    "idm_forecast_eur_mwh",
)


class MarketRuntime(Protocol):
    def start(self, controllers: ThreeControllers) -> None: ...
    def advance_to(self, when: pd.Timestamp) -> None: ...
    def close(self) -> None: ...


class ThreeLevelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict,
        data: ThreeLData,
        start,
        end,
        score_start,
        backend_factory: Callable[[PositionLedger], MarketRuntime],
    ):
        self.config, self.data = config, data
        self.start_time, self.end_time, self.score_start = map(
            timestamp, (start, end, score_start)
        )
        if not self.start_time <= self.score_start < self.end_time:
            raise ValueError("Require start <= score_start < end.")
        for boundary in (self.start_time, self.end_time, self.score_start):
            if boundary != boundary.floor("15min"):
                raise ValueError(
                    "Episode boundaries must align with settlement intervals."
                )
        self.data.require_coverage(self.start_time, self.end_time)
        self.plant = C2GPlant(config["c2g"])
        self.backend_factory = backend_factory
        self.backend: MarketRuntime | None = None
        self.finished = False
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0, -1], dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "plant": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "market": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(len(MARKET_OBSERVATION_FIELDS),),
                    dtype=np.float64,
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.backend is not None:
            raise RuntimeError("Create a new environment for a new ASSUME episode.")
        self.now = self.start_time
        self.slot_start = self.now
        self.metered_mwh = 0.0
        self._slot_inputs, self._slot_steps = np.zeros(3), 0
        self.records: list[dict] = []
        self.fast_steps = 0
        self.scored_steps = 0
        self.scored_cost_eur = 0.0
        self.scored_arrived_work_mwh = 0.0
        self.scored_served_work_mwh = 0.0
        self.thermal_fault = False
        self._reset_statistics()
        self.ledger = PositionLedger()
        actual_seed = int(self.config["simulation"]["seed"] if seed is None else seed)
        self.actual_seed = actual_seed
        self.plant.reset(actual_seed, self.data.context(self.now)["ambient_c"])
        self.controllers = ThreeControllers(
            self.plant, self.data, self.ledger, self.config
        )
        self.hardware = C2GHardwareAdapter(self, actual_seed)
        self.backend = self.backend_factory(self.ledger)
        self.backend.start(self.controllers)
        self.backend.advance_to(self.now)
        info = self._info(self.now >= self.score_start)
        info["observation_fields"] = {
            "plant": list(asdict(self.plant.state)),
            "market": MARKET_OBSERVATION_FIELDS,
        }
        return self._observation(), info

    def hardware_action(self) -> np.ndarray:
        return self.hardware.act(self.now)

    def step(self, action):
        if self.backend is None or self.finished:
            raise RuntimeError("Reset an unfinished environment before stepping.")
        scored = self.now >= self.score_start
        ambient = self.data.context(self.now)["ambient_c"]
        target = self.hardware.market_context(self.now)["target_import_mw"]
        state = self.plant.step(action, ambient)
        dt_hours = self.plant.dt_seconds / 3600
        self.metered_mwh += state.net_import_mw * dt_hours
        self._slot_inputs += [state.base_a_mw, state.base_b_mw, state.batch_arrived_mw]
        self._slot_steps += 1
        self.now += pd.Timedelta(seconds=self.plant.dt_seconds)
        self.fast_steps += 1
        reward = self._physical_reward(scored, dt_hours, target)
        self._publish_completed_inputs()
        self.backend.advance_to(self.now)
        settlement = None
        if self.now == self.slot_start + SLOT:
            settlement = self._settle_slot(scored)
            if scored:
                cost = settlement["total_cost_eur"]
                self.scored_cost_eur += cost
                reward -= cost / self.config["reward"]["cost_scale_eur"]
        terminated = bool(
            max(state.temp_a_c, state.temp_b_c) > self.plant.thermal.T_safe
        )
        self.thermal_fault = terminated
        truncated = self.now >= self.end_time
        self.finished = terminated or truncated
        if self.finished:
            reward -= self._terminal_penalty() / self.config["reward"]["cost_scale_eur"]
        info = self._info(scored)
        info.update({"settlement": settlement, "thermal_fault": terminated})
        return self._observation(), float(reward), terminated, truncated, info

    def _publish_completed_inputs(self) -> None:
        if self.now != self.slot_start + SLOT:
            return
        self.controllers.profile.record(
            self.slot_start,
            dict(
                zip(
                    self.controllers.profile.FIELDS,
                    self._slot_inputs / self._slot_steps,
                )
            ),
        )

    def _settle_slot(self, scored: bool) -> dict:
        settlement = self.ledger.settle(
            self.slot_start, self.now, self.metered_mwh, scored=scored
        )
        self.records.append(settlement)
        self.slot_start, self.metered_mwh = self.now, 0.0
        self._slot_inputs, self._slot_steps = np.zeros(3), 0
        return settlement

    def _physical_reward(self, scored: bool, dt_hours: float, target) -> float:
        if not scored:
            return 0.0
        state = self.plant.state
        self.scored_steps += 1
        self.scored_arrived_work_mwh += state.batch_arrived_mw * dt_hours
        self.scored_served_work_mwh += state.batch_served_mw * dt_hours
        self._record_statistics(state, dt_hours, target)
        rcfg = self.plant.reward_config
        thermal = max(0.0, state.temp_a_c - rcfg["T_warn_A"])
        thermal += max(0.0, state.temp_b_c - rcfg["T_warn_B"])
        served = state.batch_served_mw / self.plant.batch_capacity_mw
        return dt_hours * (rcfg["alpha"] * served - rcfg["gamma_thermal"] * thermal)

    def _reset_statistics(self) -> None:
        self.statistics = dict.fromkeys(
            (
                "scored_facility_mwh",
                "scored_cooling_mwh",
                "scored_metered_mwh",
                "scored_battery_charge_mwh",
                "scored_battery_discharge_mwh",
                "scored_thermal_warning_hours",
                "scored_tracking_hours",
                "scored_tracking_abs_mwh",
                "scored_tracking_squared_mw2_hours",
                "scored_request_abs_mwh",
                "scored_pue_time_integral_hours",
            ),
            0.0,
        )
        self._max_temperatures = np.full(2, -np.inf)
        self._min_soc, self._max_soc = 1.0, 0.0

    def _record_statistics(self, state, dt_hours: float, target) -> None:
        stats = self.statistics
        stats["scored_facility_mwh"] += state.facility_mw * dt_hours
        stats["scored_cooling_mwh"] += state.cooling_mw * dt_hours
        stats["scored_metered_mwh"] += state.net_import_mw * dt_hours
        stats["scored_battery_charge_mwh"] += max(0, -state.battery_mw) * dt_hours
        stats["scored_battery_discharge_mwh"] += max(0, state.battery_mw) * dt_hours
        stats["scored_pue_time_integral_hours"] += state.pue * dt_hours
        self._max_temperatures = np.maximum(
            self._max_temperatures, [state.temp_a_c, state.temp_b_c]
        )
        self._min_soc = min(self._min_soc, state.soc)
        self._max_soc = max(self._max_soc, state.soc)
        rcfg = self.plant.reward_config
        warning = state.temp_a_c > rcfg["T_warn_A"] or state.temp_b_c > rcfg["T_warn_B"]
        stats["scored_thermal_warning_hours"] += float(warning) * dt_hours
        if target is not None:
            error = state.net_import_mw - target
            stats["scored_tracking_hours"] += dt_hours
            stats["scored_tracking_abs_mwh"] += abs(error) * dt_hours
            stats["scored_tracking_squared_mw2_hours"] += error**2 * dt_hours
            stats["scored_request_abs_mwh"] += (
                abs(state.facility_mw - target) * dt_hours
            )

    def _terminal_penalty(self) -> float:
        cfg, state = self.config["reward"], self.plant.state
        soc_shortfall = (
            max(0.0, self.plant.initial_soc - state.soc) * self.plant.battery.E_NOM_MWH
        )
        return (
            cfg["terminal_backlog_eur_per_work_mwh"] * state.backlog_work_mwh
            + cfg["terminal_soc_eur_per_mwh"] * soc_shortfall
        )

    def _observation(self) -> dict[str, np.ndarray]:
        # At truncation the observation still describes the last delivered interval.
        query_time = min(
            self.now, self.end_time - pd.Timedelta(seconds=self.plant.dt_seconds)
        )
        slot = query_time.floor("15min")
        bm = self.ledger.price("BM", slot, slot + SLOT, self.now)
        context = self.data.context(query_time)
        market = [
            *(self.ledger.energy(slot, slot + SLOT, name) for name in MARKETS),
            self.metered_mwh,
            0.0 if bm is None else bm,
            float(bm is not None),
            context["system_demand_mw"],
            context["system_imbalance_mw"],
            self.controllers.plan.at(query_time, self.plant),
            (slot + SLOT - self.now).total_seconds(),
            self.data.forecast("EOM", query_time),
            self.data.forecast("IDM", query_time),
        ]
        return {
            "plant": np.array(
                list(asdict(self.plant.state).values()), dtype=np.float64
            ),
            "market": np.array(market, dtype=np.float64),
        }

    def _info(self, scored: bool) -> dict:
        return {
            "timestamp": self.now.isoformat(),
            "scored": scored,
            "plant": asdict(self.plant.state),
            "fast_steps": self.fast_steps,
            "applied_action": self.plant.previous_action.copy(),
            "frequency_model": "not_modeled",
            "network_voltage_model": "not_modeled",
        }

    def summary(self) -> dict:
        return {
            "seed": self.actual_seed,
            "start": self.start_time.isoformat(),
            "score_start": self.score_start.isoformat(),
            "stop": self.now.isoformat(),
            "configured_end": self.end_time.isoformat(),
            "completed": self.finished
            and self.now == self.end_time
            and not self.thermal_fault,
            "thermal_fault": self.thermal_fault,
            "fast_steps": self.fast_steps,
            "scored_steps": self.scored_steps,
            "settled_intervals": len(self.records),
            "scored_cost_eur": self.scored_cost_eur,
            "scored_arrived_work_mwh": self.scored_arrived_work_mwh,
            "scored_served_work_mwh": self.scored_served_work_mwh,
            "terminal_backlog_work_mwh": self.plant.state.backlog_work_mwh,
            "initial_soc": self.plant.initial_soc,
            "terminal_soc": self.plant.state.soc,
            "terminal_penalty_eur": self._terminal_penalty(),
            "unsettled_partial_interval_mwh": self.metered_mwh,
            "scored_financially_complete": self.finished
            and self.now == self.end_time
            and self.metered_mwh == 0.0,
            "open_future_positions_mwh": self.ledger.energy(self.now, self.end_time),
            "unpriced_warmup_intervals": sum(
                row["status"] == "unpriced_warmup" for row in self.records
            ),
            "decisions_by_market": {
                name: sum(item["market"] == name for item in self.controllers.decisions)
                for name in MARKETS
            },
            "upper_policy_calls": {
                "L1": self.controllers.day_ahead.calls,
                "L2": self.controllers.intraday.calls,
            },
            "inner_controller": self.hardware.manifest(),
            "inner_predict_calls": self.hardware.predict_calls,
            "inner_predict_seconds": self.hardware.predict_seconds,
            "inner_max_predict_seconds": self.hardware.max_predict_seconds,
            "inner_capped_steps": self.hardware.capped_steps,
            "plan_revisions": len(self.controllers.plan_revisions),
            **self.statistics,
            **self._physical_summary(),
            **self._tracking_summary(),
            **self._financial_summary(),
        }

    def _physical_summary(self) -> dict:
        result = {
            "scored_peak_temp_a_c": self._max_temperatures[0],
            "scored_peak_temp_b_c": self._max_temperatures[1],
            "scored_min_soc": self._min_soc,
            "scored_max_soc": self._max_soc,
            "scored_mean_pue": None,
        }
        if not self.scored_steps:
            return dict.fromkeys(result)
        duration = self.scored_steps * self.plant.dt_seconds / 3600
        result["scored_mean_pue"] = (
            self.statistics["scored_pue_time_integral_hours"] / duration
        )
        return result

    def _tracking_summary(self) -> dict:
        duration = self.statistics["scored_tracking_hours"]
        if not duration:
            return {
                "scored_tracking_mae_mw": None,
                "scored_tracking_rmse_mw": None,
                "scored_request_mae_mw": None,
            }
        return {
            "scored_tracking_mae_mw": self.statistics["scored_tracking_abs_mwh"]
            / duration,
            "scored_tracking_rmse_mw": float(
                np.sqrt(self.statistics["scored_tracking_squared_mw2_hours"] / duration)
            ),
            "scored_request_mae_mw": self.statistics["scored_request_abs_mwh"]
            / duration,
        }

    def _financial_summary(self) -> dict:
        result = {
            "scored_procurement_cost_eur": 0.0,
            "scored_imbalance_cost_eur": 0.0,
            "scored_imbalance_mwh": 0.0,
            "scored_costs_by_market_eur": dict.fromkeys(MARKETS, 0.0),
            "scored_positions_mwh": dict.fromkeys(MARKETS, 0.0),
        }
        for row in self.records:
            if not row["scored"]:
                continue
            result["scored_procurement_cost_eur"] += row["procurement_cost_eur"]
            result["scored_imbalance_cost_eur"] += row["imbalance_cost_eur"]
            result["scored_imbalance_mwh"] += row["imbalance_mwh"]
            for name in MARKETS:
                result["scored_costs_by_market_eur"][name] += row[
                    "costs_by_market_eur"
                ][name]
                result["scored_positions_mwh"][name] += row["positions_mwh"][name]
        return result

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
