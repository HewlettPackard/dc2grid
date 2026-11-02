"""Finite, full-work economic dispatch; bids and physical settlement stay frozen."""

# ruff: noqa: E402

from __future__ import annotations

import copy
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent))

# isort: off
import economy_context  # noqa: E402,F401

# The frozen bootstrap sets its own plotting path. Subsequent optional imports
# must not create caches in those frozen experiment directories.
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common import assert_default_plant, cap_cpu
from controllers import (
    NOMINAL,
    CoolingFirstController,
    FullResidualMacro,
    battery_bounds,
    facility_power,
    liquid_temperature,
    pretransformer_power,
)
from macro_task import Ledger, score_transition
from market_policy import maximum_native_quantity
from recovery_ops import battery_dispatch, begin_recovery, closure, plant_state
from c2g_env import C2GMacroEnv
from c2g_env.obs_indices import Fast, Macro
# isort: on

TASK_SCHEMA = "track-b-fixed-full-residual-economic-dispatch-v1"
REQUIRED_GAMMA = 1.0
REWARD_SCALE_USD = 100.0
EXTRA_RESPONSE_MAX_MW = 6.0
GUARD_SERVICE_TARGET = 0.92
EXTERNAL_SERVICE_FLOOR = 0.90
CREDIT_RESERVE_MWH = 0.02
SERVICE_TICKS = 17280
RECOVERY_TICKS = 1440
WINDOW_TICKS = 180
DT_HOURS = 5.0 / 3600.0
FROZEN_CENTERED_BOUND_USD = 2824000.0
RATE_BOUND_USD_MW_H = 500.0
EVALUATION_SEEDS = frozenset(
    (*range(90001, 90005), 91001, 91002, 92001, 92002, 92003, *range(100, 105))
)

OBSERVATION_SCHEMA = (
    "mean_actual_temp_A_over_35",
    "mean_actual_temp_B_over_35",
    "end_soc",
    "mean_base_power_over_250MW",
    "mean_facility_power_over_250MW",
    "mean_absolute_regd",
    "mean_lmp_over_200",
    "mean_grid_load_norm",
    "mean_absolute_tracking_error_over_committed_capacity",
    "any_workload_spike",
    "zone_A_thermal_headroom_over_35",
    "zone_B_thermal_headroom_over_35",
    "previous_bid_mw_over_30",
    "previous_bid_price_native_action",
    "mean_frequency_deviation_norm",
    "mean_pcc_voltage_pu",
    "mean_backlog_norm",
    "currently_posted_rmcp_over_100",
    "currently_posted_residual_mw_over_30",
    "current_fast_signed_regd",
    "current_fast_lmp_over_200",
    "realized_capacity_error_credit_over_30MWh",
    "remaining_service_fraction",
)

_GRID_COLUMNS = (
    "grid_load_mw",
    "p_reference_mw",
    "p_facility_mw",
    "p_grid_import_mw",
    "temp_A",
    "temp_B",
    "hvac_effort",
    "delta_p_demanded_kw",
    "bess_actual_kw",
    "duration_hours",
    "reference_temp_A",
    "reference_temp_B",
)


def excess_cash_bound(normalization: dict) -> dict:
    """Enclose every valid prefix, including its first physical fault tick."""
    required = (
        "cash_prefix_bound_usd",
        "import_abs_mw",
        "nominal_abs_mw",
        "lmp_abs_usd_mwh",
        "ambient_abs_C",
    )
    if any(
        not math.isfinite(normalization[key]) or normalization[key] <= 0
        for key in required
    ):
        raise ValueError("Cash enclosure requires explicit finite positive support")
    if (
        normalization["cash_prefix_bound_usd"] != FROZEN_CENTERED_BOUND_USD
        or normalization["import_abs_mw"] != 400
        or normalization["nominal_abs_mw"] != 400
        or normalization["ambient_abs_C"] != 60
    ):
        raise ValueError("The frozen centered-cash enclosure cannot be changed")
    price = float(normalization["lmp_abs_usd_mwh"])
    ideal_bound = 24 * 30 * RATE_BOUND_USD_MW_H
    centered_support = ideal_bound + 26 * price * (400 + 400)
    if centered_support > FROZEN_CENTERED_BOUND_USD:
        raise ValueError("Declared support exceeds the frozen centered-cash bound")
    demand_bound = 24 * 30 * price  # |RegD| <= 1, including signed electricity.
    raw_bound = FROZEN_CENTERED_BOUND_USD + ideal_bound + demand_bound
    bound = float(math.ceil(raw_bound / 1000) * 1000)
    return {
        "centered_cash_bound_usd": FROZEN_CENTERED_BOUND_USD,
        "centered_support_usd": centered_support,
        "ideal_regulation_bound_usd": ideal_bound,
        "signed_demand_electricity_bound_usd": demand_bound,
        "unrounded_excess_bound_usd": raw_bound,
        "excess_cash_prefix_bound_usd": bound,
        "invalid_episode_return": -2 * bound / REWARD_SCALE_USD,
        "valid_return_absolute_bound": bound / REWARD_SCALE_USD,
    }


def excess_transition(
    before: float, increment: float, bound: float, *, terminal: bool, valid: bool
) -> tuple[float, float, float]:
    """Return reward, explicit non-money invalid adjustment (USD), cash prefix."""
    normalized, _, after = score_transition(
        before, increment, bound, 0.0, 0.0, terminal=terminal, valid=valid
    )
    adjustment = -2 * bound - after if terminal and not valid else 0.0
    return normalized * bound / REWARD_SCALE_USD, adjustment, after


def requested_extra_response(action) -> float:
    value = np.asarray(action)
    if (
        value.shape != (1,)
        or not np.issubdtype(value.dtype, np.number)
        or np.iscomplexobj(value)
        or not np.isfinite(value).all()
        or np.any(np.abs(value) > 1)
    ):
        raise ValueError("Economic dispatch requires one finite action in [-1,1]")
    return EXTRA_RESPONSE_MAX_MW * max(float(value[0]), 0.0)


def service_capacity_reference(info: dict) -> float:
    """Audit the true posted residual, not the rounded observation or our bid."""
    maximum = maximum_native_quantity(info["residual_mw"])
    if not info["bid_accepted"]:
        if info["committed_mw"] != 5.0:
            raise ValueError("A rejected window must retain the 5MW standing contract")
        return 5.0
    return maximum


@dataclass
class CapacityErrorBudget:
    credit_mwh: float = 0.0
    committed_mwh: float = 0.0
    absolute_error_mwh: float = 0.0
    minimum_credit_mwh: float = 0.0
    steps: int = 0

    def cap(self, committed_mw: float, remaining_window_hours: float) -> float:
        if (
            not math.isfinite(committed_mw)
            or not 5 <= committed_mw <= 30
            or not math.isfinite(remaining_window_hours)
            or not 0 < remaining_window_hours <= 0.25
            or not math.isfinite(self.credit_mwh)
        ):
            raise ValueError("Guard requires actual capacity and current-window time")
        return min(
            EXTRA_RESPONSE_MAX_MW,
            max(
                0.0,
                (1 - GUARD_SERVICE_TARGET) * committed_mw
                + (self.credit_mwh - CREDIT_RESERVE_MWH) / remaining_window_hours,
            ),
        )

    def update(self, committed_mw: float, error_kw: float, hours: float) -> None:
        if (
            not math.isfinite(error_kw)
            or not math.isfinite(committed_mw)
            or not 5 <= committed_mw <= 30
            or hours != DT_HOURS
        ):
            raise ValueError("Credit requires metered five-second service samples")
        absolute_error = abs(error_kw) / 1000
        self.credit_mwh += (
            (1 - GUARD_SERVICE_TARGET) * committed_mw - absolute_error
        ) * hours
        self.committed_mwh += committed_mw * hours
        self.absolute_error_mwh += absolute_error * hours
        self.minimum_credit_mwh = min(self.minimum_credit_mwh, self.credit_mwh)
        self.steps += 1


class EconomicDispatchController(CoolingFirstController):
    """Frozen nominal observer, with an explicit 0..6MW requested PCC overlay.

    This scoped copy of CoolingFirstController.action removes only its 1MW
    residual-coordinate limit. Pump residual and SOC residual remain zero.
    The original temperature-dependent pump/HVAC and BESS compensation remain.
    """

    def action(self, extra_response_mw: float = 0.0) -> np.ndarray:
        if (
            not math.isfinite(extra_response_mw)
            or not 0 <= extra_response_mw <= EXTRA_RESPONSE_MAX_MW
        ):
            raise ValueError("Extra requested PCC response must be within [0,6] MW")
        if self.observation is None:
            raise RuntimeError("Observe the current state before selecting an action")
        observation = self.observation
        ta = float(observation[Fast.TEMP_A]) * 35
        tb = float(observation[Fast.TEMP_B]) * 35
        ambient = self.ambient(observation)
        soc = float(observation[Fast.SOC])
        request = (
            float(observation[Fast.COMMITTED]) * 30 * float(observation[Fast.REGD])
        )
        # Retain the frozen float32 residual-zero arithmetic, including promotion.
        pump_residual = np.float32(0.0)
        pump = float(
            np.clip(
                NOMINAL + max(0, ta - 32.0) * 0.3 + 0.15 * pump_residual,
                0.15,
                1,
            )
        )
        lower_battery, upper_battery = battery_bounds(soc)
        battery = float(np.clip(60 * (soc - 0.50), lower_battery, upper_battery))
        nominal_battery = battery
        desired_relief = request - battery + extra_response_mw
        reference_next = liquid_temperature(
            self.reference_temperature_A, self.estimated_heat_A, NOMINAL, ambient
        )
        actual_next = liquid_temperature(ta, self.estimated_heat_A, pump, ambient)
        reference_cool = 35 * NOMINAL * max(0, reference_next - 30) / 30 + 1.5 * NOMINAL
        actual_cool = 35 * pump * max(0, actual_next - 30) / 30 + 1.5 * pump
        previous_cool = 35 * float(self.last_action[1]) * max(
            0, ta - 30
        ) / 30 + 1.5 * float(self.last_action[1])
        observed_power = float(observation[Fast.P_FAC]) * 250
        reference_pre = (
            pretransformer_power(observed_power)
            + reference_cool
            - previous_cool
            + 50 * (NOMINAL - float(self.last_action[2]))
        )
        if self.transformer_compensation:
            target_pre = pretransformer_power(
                facility_power(reference_pre) - desired_relief
            )
            cooling_change = target_pre - reference_pre
        else:
            cooling_change = -desired_relief
        unconstrained_hvac = (
            NOMINAL + (cooling_change - (actual_cool - reference_cool)) / 50
        )
        thermal_floor = float(np.clip(2 * (tb - 32.0), 0, 1))
        hvac = float(np.clip(unconstrained_hvac, thermal_floor, 1))
        if hvac != unconstrained_hvac:
            before = (
                reference_pre
                + actual_cool
                - reference_cool
                + 50 * (unconstrained_hvac - NOMINAL)
            )
            after = reference_pre + actual_cool - reference_cool + 50 * (hvac - NOMINAL)
            battery += facility_power(after) - facility_power(before)
        battery = float(np.clip(battery, lower_battery, upper_battery))
        action = np.array([1.0, pump, hvac, battery / 5], dtype=np.float32)
        self.last_action = action.copy()
        self.last_diagnostics = {
            "estimated_heat_A_mw": self.estimated_heat_A,
            "observer_reference_temperature_A": self.reference_temperature_A,
            "observed_contract_target_mw": request,
            "extra_response_mw": float(extra_response_mw),
            "desired_pcc_response_mw": request + extra_response_mw,
            "soc_target": 0.50,
            "pump_nominal": NOMINAL,
            "pump_thermal_compensation": pump - NOMINAL,
            "nominal_soc_battery_command_mw": nominal_battery,
            "hvac_limit_battery_compensation_mw": battery - nominal_battery,
            "thermal_hvac_floor": thermal_floor,
            "unconstrained_hvac": unconstrained_hvac,
        }
        return action


class EconomicDispatchTask(gym.Env):
    """One action per 15 minutes; the final step includes fixed two-hour recovery."""

    metadata = {"render_modes": []}
    required_gamma = REQUIRED_GAMMA
    task_schema = TASK_SCHEMA
    observation_schema = OBSERVATION_SCHEMA

    def __init__(
        self,
        protocol: dict,
        *,
        seed: int,
        rank: int = 0,
        fixed_cell: dict | None = None,
        record_trace: bool = False,
        evaluation_mode: bool = False,
        evaluation_cells: list[dict] | tuple[dict, ...] | None = None,
    ):
        super().__init__()
        cap_cpu()
        if not isinstance(evaluation_mode, bool):
            raise ValueError("evaluation_mode must be an explicit boolean")
        self.protocol = copy.deepcopy(protocol)
        self.seed_key, self.rank = seed, rank
        self.fixed_cell = copy.deepcopy(fixed_cell)
        self.record_trace = bool(record_trace)
        self._evaluation_mode = evaluation_mode
        self._evaluation_keys = frozenset()
        if self.evaluation_mode:
            if (
                self.fixed_cell is None
                or not isinstance(evaluation_cells, (list, tuple))
                or not evaluation_cells
            ):
                raise ValueError(
                    "Evaluation mode requires fixed_cell and a nonempty "
                    "evaluation_cells allowlist"
                )
            keys = tuple(self._evaluation_key(cell) for cell in evaluation_cells)
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate conditions in evaluation_cells")
            self._evaluation_keys = frozenset(keys)
        elif evaluation_cells is not None:
            raise ValueError("evaluation_cells requires evaluation_mode=True")
        self.episode_ordinal = 0
        self.last_episode = None
        self.done, self._closed, self._initialized = True, False, False
        self.bound_derivation = excess_cash_bound(self.protocol["normalization"])
        self.excess_bound_usd = self.bound_derivation["excess_cash_prefix_bound_usd"]
        self._validate_protocol()
        self.controller = EconomicDispatchController()
        self.macro = FullResidualMacro()
        self.core = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=self.protocol["config_path"],
            inner_action_fn=self._inner_action,
            sub_step_callback=self._service_callback,
        )
        self.fast = self.core._fast_env
        self._check_cadence()
        low, high = (
            self.core.observation_space.low.copy(),
            self.core.observation_space.high.copy(),
        )
        high[Macro.TRACKING_ERR] = np.inf
        self.observation_space = spaces.Box(
            np.r_[low, -1.0, 0.0, -np.inf, 0.0].astype(np.float32),
            np.r_[high, 1.0, 1.0, np.inf, 1.0].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space.c2g_simulation_mode = "physical_v1"
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    @property
    def episode_result(self):
        return self.last_episode

    @property
    def evaluation_mode(self):
        return self._evaluation_mode

    @property
    def condition_role(self):
        return "evaluation_only" if self.evaluation_mode else "training_only"

    @staticmethod
    def _evaluation_key(cell):
        if (
            not isinstance(cell, dict)
            or not {"start_day", "seed", "regd_load_bias"} <= cell.keys()
        ):
            raise ValueError("Evaluation conditions require day, seed and bias")
        day, seed, bias = (cell["start_day"], cell["seed"], cell["regd_load_bias"])
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
            for value in (day, seed)
        ) or (
            isinstance(bias, bool)
            or not isinstance(bias, (int, float, np.integer, np.floating))
            or not math.isfinite(bias)
        ):
            raise ValueError("Evaluation conditions require valid numeric fields")
        return int(day), int(seed), float(bias)

    def _validate_protocol(self):
        p = self.protocol
        if (
            isinstance(self.seed_key, bool)
            or not isinstance(self.seed_key, (int, np.integer))
            or self.seed_key < 0
            or isinstance(self.rank, bool)
            or not isinstance(self.rank, (int, np.integer))
            or isinstance(p["n_envs"], bool)
            or not isinstance(p["n_envs"], int)
            or p["n_envs"] < 1
            or self.rank < 0
            or (not self.evaluation_mode and self.rank >= p["n_envs"])
            or not p["train_days"]
            or not p["train_biases"]
            or len(set(p["train_days"])) != len(p["train_days"])
        ):
            raise ValueError("Invalid deterministic training schedule")
        for day in p["train_days"]:
            if isinstance(day, bool) or not isinstance(day, int) or day < 0:
                raise ValueError("Training days must be nonnegative integer indices")
        if any(not math.isfinite(bias) for bias in p["train_biases"]):
            raise ValueError("Training biases must be finite")
        if p["service_floor"] != EXTERNAL_SERVICE_FLOOR:
            raise ValueError("The published service floor remains exactly 0.90")
        for key, expected in {
            "min_terminal_soc": 0.2,
            "max_backlog_fraction": 0.01,
            "max_thermal_warning_fraction": 0.01,
        }.items():
            if p["service_gate"][key] != expected:
                raise ValueError(f"Frozen service gate changed: {key}")
        for key, expected in {
            "duration_seconds": 7200,
            "throttle": 1.0,
            "pump": 0.7,
            "hvac": 0.7,
            "committed_mw": 0.0,
            "regulation_revenue_usd": 0.0,
            "soc_tolerance": 1e-6,
            "backlog_tolerance_kwh": 1e-6,
            "temperature_tolerance_C": 0.1,
            "work_conservation_tolerance_kwh": 1e-4,
            "soc_target": "episode_initial_soc",
            "temperature_target": "continuing_nominal_reference_trajectory",
            "fixed_duration_even_if_closed_early": True,
            "terminate_on_existing_physical_faults": True,
        }.items():
            if p["recovery"][key] != expected:
                raise ValueError(f"Frozen recovery contract changed: {key}")
        expected_macro = (
            "TEMP_A",
            "TEMP_B",
            "SOC",
            "P_BASE",
            "P_FAC",
            "REGD",
            "LMP",
            "GRID_LOAD",
            "TRACKING_ERR",
            "IS_SPIKE",
            "HEADROOM_A",
            "HEADROOM_B",
            "BID_MW_PREV",
            "BESS_PREV",
            "FREQ_DEV",
            "VPCC",
            "BACKLOG",
            "RMCP",
            "REG_NEED",
        )
        if (
            tuple(getattr(Macro, key) for key in expected_macro) != tuple(range(19))
            or (Fast.REGD, Fast.LMP, Fast.COMMITTED, Fast.DIM) != (6, 7, 17, 18)
            or Macro.DIM != 19
        ):
            raise ValueError("Frozen observation-index contract changed")
        self.forbidden_seeds = set(EVALUATION_SEEDS)
        for key in (
            "development_cells",
            "final_cells",
            "sealed_cells",
            "sealed_test_cells",
            "test_cells",
            "evaluation_cells",
            "validation_cells",
        ):
            self.forbidden_seeds.update(int(cell["seed"]) for cell in p.get(key, []))
        if self.fixed_cell is not None:
            self._validate_condition(self.fixed_cell)

    def _check_cadence(self):
        if (
            self.fast._episode_ticks != SERVICE_TICKS
            or self.fast._dt != 5
            or self.core._substeps != WINDOW_TICKS
            or self.core._episode_macro_ticks != 96
            or self.core._committed_max_mw != 30
            or self.core._dr_baseline_mw != 5
            or self.core._dr_rate_usd_mw != 5
            or self.core._rmcp_max != 100
        ):
            raise ValueError("Expected frozen 24h service, 15min bids and 5MW fallback")

    def _validate_condition(self, cell):
        if self.evaluation_mode:
            if self._evaluation_key(cell) not in self._evaluation_keys:
                raise ValueError(
                    "Condition is outside the explicit evaluation_cells allowlist"
                )
            return
        if (
            cell["start_day"] not in self.protocol["train_days"]
            or cell["regd_load_bias"] not in self.protocol["train_biases"]
        ):
            raise ValueError("Condition is outside the training-only allowlist")
        if (
            isinstance(cell["seed"], bool)
            or not isinstance(cell["seed"], (int, np.integer))
            or cell["seed"] < 0
            or cell["seed"] in self.forbidden_seeds
        ):
            raise ValueError("Invalid training seed or excluded evaluation seed")

    def condition(self) -> dict:
        if self.evaluation_mode and self.fixed_cell is None:
            raise ValueError("Evaluation mode cannot enter training scheduling")
        if self.fixed_cell is not None:
            cell = dict(self.fixed_cell)
        else:
            days, biases = self.protocol["train_days"], self.protocol["train_biases"]
            slot = self.episode_ordinal * self.protocol["n_envs"] + self.rank
            cycle, position = divmod(slot, len(days))
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed_key, 713, cycle])
            )
            day_index = int(rng.permutation(len(days))[position])
            cell = {
                "start_day": days[day_index],
                "regd_load_bias": biases[(position + cycle) % len(biases)],
                "seed": int(
                    np.random.SeedSequence(
                        [self.seed_key, 157, cycle, position]
                    ).generate_state(1)[0]
                ),
                "curriculum_cycle": cycle,
                "curriculum_position": position,
            }
        self._validate_condition(cell)
        return cell

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("Cannot reset a closed task")
        if not self.done:
            raise RuntimeError("Cannot discard an incomplete physical episode")
        if options:
            raise ValueError("Conditions are controlled by the training protocol")
        super().reset(seed=seed)
        self.cell = self.condition()
        self.fast._episode_ticks = SERVICE_TICKS
        self.core._episode_macro_ticks = 96
        observation, reset_info = self.core.reset(
            seed=self.cell["seed"],
            options={key: self.cell[key] for key in ("start_day", "regd_load_bias")},
        )
        assert_default_plant(self.fast)
        self._check_cadence()
        self.initial = plant_state(self.fast)
        if self.initial["backlog_kwh"] != 0:
            raise ValueError("Expected the frozen empty initial work queue")
        self.controller.reset()
        self.service, self.recovery = Ledger(), Ledger()
        self.budget = CapacityErrorBudget()
        self.revenue = self.ideal_revenue = self.demand_electricity_usd = 0.0
        self.centered_cash = self.excess_cash = self.reward_sum = 0.0
        self.committed_capacity = self.delivered_capacity = self.maximum_capacity = 0.0
        self.pre_regd_sum = self.original_service_reward = 0.0
        self.original_macro_reward = self.original_recovery_reward = 0.0
        self.decisions = self.guard_binding_ticks = 0
        self.requested_extra_mwh = self.guarded_extra_mwh = 0.0
        self._requested_extra_mw = 0.0
        self._grid_rows = []
        self.service_rows, self.recovery_rows, self.macro_rows = [], [], []
        self._macro_observation = observation.copy()
        self._fast_observation = self.core._last_hw_obs.copy()
        self._service_terminal = None
        self._recovered = {"performed": False, "complete": False, "closed": False}
        self._reward_totals = {
            key: 0.0
            for key in (
                "actual_regulation_revenue_usd",
                "ideal_regulation_reference_usd",
                "service_centered_electricity_usd",
                "contractual_demand_electricity_usd",
                "recovery_centered_electricity_usd",
                "excess_cash_usd",
                "invalid_adjustment_usd",
            )
        }
        self.done, self._initialized = False, True
        self.episode_ordinal += 1
        return self._observation(), {
            **reset_info,
            "condition": dict(self.cell),
            "condition_role": self.condition_role,
            "task_schema": TASK_SCHEMA,
            "required_gamma": REQUIRED_GAMMA,
            "bound_derivation": dict(self.bound_derivation),
        }

    def _observation(self):
        observation = np.r_[
            self._macro_observation,
            self._fast_observation[Fast.REGD],
            self._fast_observation[Fast.LMP],
            self.budget.credit_mwh / 30,
            1 - self.service.steps / SERVICE_TICKS,
        ].astype(np.float32)
        if (
            observation.shape != (23,)
            or not np.isfinite(observation).all()
            or not self.observation_space.contains(observation)
        ):
            raise FloatingPointError("Invalid 23D causal economic observation")
        return observation

    def _inner_action(self, observation, macro_action):
        remaining = (self._window_end_tick - self.fast._tick) * DT_HOURS
        cap = self.budget.cap(self.fast.committed_mw, remaining)
        extra = min(self._requested_extra_mw, cap)
        self.controller.features(observation)
        action = self.controller.action(extra)
        self._tick_dispatch = {
            **self.controller.last_diagnostics,
            "contract_target_mw": (
                self.fast.committed_mw * 1000 * self.fast._prev_regd_signal / 1000
            ),
            "desired_extra_response_mw": self._requested_extra_mw,
            "guard_cap_mw": cap,
            "guarded_extra_response_mw": extra,
            "guard_applied": bool(extra < self._requested_extra_mw),
            "credit_before_mwh": self.budget.credit_mwh,
            "remaining_window_hours": remaining,
        }
        return action

    def _trace_row(self, pre, post, info):
        return {
            **info,
            **self.fast._elec.get_diagnostics(),
            "observed_committed_mw": float(pre[Fast.COMMITTED]) * 30,
            "committed_mw": self.fast.committed_mw,
            "observed_temp_B": float(pre[Fast.TEMP_B]) * 35,
            "observed_regd_signal": float(pre[Fast.REGD]),
            "p_base_norm": float(post[Fast.P_BASE]),
            "p_flex_norm": float(post[Fast.P_FLEX]),
            "reference_temp_A": self.fast._reference_thermal.temp_A,
            "reference_temp_B": self.fast._reference_thermal.temp_B,
            "load_anomaly": float(self.fast._grid._load_anomaly[info["tick"] - 1]),
        }

    def _service_callback(self, pre, action, post, reward, done, info):
        if action[0] != 1.0:
            raise AssertionError("The overlay must never throttle work")
        demand = info["delta_p_demanded_kw"] / 1000
        if not math.isclose(
            demand, self._tick_dispatch["contract_target_mw"], abs_tol=1e-10
        ):
            raise AssertionError("Economic dispatch changed contractual demand")
        self.service.update(self.fast, info, self.protocol["normalization"])
        self.budget.update(
            self.fast.committed_mw, info["tracking_err_kw"], info["duration_hours"]
        )
        self.demand_electricity_usd += info["lmp"] * demand * info["duration_hours"]
        self.pre_regd_sum += float(pre[Fast.REGD])
        self.original_service_reward += float(reward)
        dispatch = {
            **self._tick_dispatch,
            "credit_after_mwh": self.budget.credit_mwh,
            "realized_signed_tracking_error_mw": (
                info["delta_p_actual_kw"] - info["delta_p_demanded_kw"]
            )
            / 1000,
        }
        self.guard_binding_ticks += int(dispatch["guard_applied"])
        self.requested_extra_mwh += (
            dispatch["desired_extra_response_mw"] * info["duration_hours"]
        )
        self.guarded_extra_mwh += (
            dispatch["guarded_extra_response_mw"] * info["duration_hours"]
        )
        self._grid_rows.append(
            tuple(info[key] for key in _GRID_COLUMNS[:-2])
            + (
                self.fast._reference_thermal.temp_A,
                self.fast._reference_thermal.temp_B,
            )
        )
        if self.record_trace:
            self.service_rows.append({**self._trace_row(pre, post, info), **dispatch})

    def _recover(self, last_info):
        config = self.protocol["recovery"]
        self.recovery.initial_backlog_kwh = self.service.backlog_kwh
        extension = begin_recovery(self.fast, self.cell, last_info, config)
        if extension["recovery_ticks"] != RECOVERY_TICKS:
            raise ValueError("The physical recovery cannot be shortened")
        self._recovered = {
            "performed": True,
            "complete": False,
            "closed": False,
            "extension": extension,
        }
        observations, infos = deque(maxlen=WINDOW_TICKS), deque(maxlen=WINDOW_TICKS)
        pre = self._fast_observation.copy()
        pre[Fast.COMMITTED] = 0.0
        for _ in range(RECOVERY_TICKS):
            action = np.array(
                [
                    config["throttle"],
                    config["pump"],
                    config["hvac"],
                    battery_dispatch(self.fast._bess, self.initial["soc"]),
                ],
                dtype=np.float32,
            )
            post, reward, term, trunc, info = self.fast.step(action)
            self.recovery.update(self.fast, info, self.protocol["normalization"])
            self.original_recovery_reward += float(reward)
            if self.fast.committed_mw != 0 or info["delta_p_demanded_kw"] != 0:
                raise AssertionError("Recovery must have no regulation obligation")
            if self.record_trace:
                self.recovery_rows.append(self._trace_row(pre, post, info))
            observations.append(post)
            infos.append({**info, "p_flex_max_kw": self.fast._workload.p_flex_max_kw})
            pre = post
            if term or trunc:
                break
        complete = bool(not term and trunc and self.recovery.steps == RECOVERY_TICKS)
        terminal = plant_state(self.fast)
        self._recovered = {
            **self._recovered,
            "complete": complete,
            "physical_terminated": bool(term),
            "raw_fast_truncated": bool(trunc),
            "terminal": terminal,
            "ledger": self.recovery.report(),
            "regulation_revenue_usd": 0.0,
            "last_faults": {
                key: bool(info[key])
                for key in ("thermal_fault", "freq_fault", "voltage_fault")
            },
            **closure(terminal, self.initial, config, complete=complete),
        }
        self._fast_observation = post.copy()
        self._macro_observation = C2GMacroEnv.aggregate_macro_obs(
            list(observations),
            list(infos),
            T_safe=self.fast._thermal.T_safe,
            committed_mw=0.0,
            tracking_floor_kw=self.fast._macro_tracking_floor_kw,
            bid_mw_norm=0.0,
            bid_price_norm=0.0,
            rmcp_norm=0.0,
            reg_need_norm=0.0,
        )

    def _episode(self, *, service_complete, physical_term, service_trunc, reason):
        service = self.service.report() if self.service.steps else None
        terminal = self._service_terminal or plant_state(self.fast)
        gate = self.protocol["service_gate"]
        checks = {
            "complete": bool(service_complete),
            "terminal_soc": terminal["soc"] >= gate["min_terminal_soc"],
            "terminal_work": bool(
                service is not None
                and service["backlog_kwh"]
                <= gate["max_backlog_fraction"] * service["work_arrived_kwh"] + 1e-6
            ),
            "thermal_warnings": bool(
                service is not None
                and service["thermal_warning_fraction"]
                <= gate["max_thermal_warning_fraction"]
            ),
        }
        ratio = (
            self.delivered_capacity / self.maximum_capacity
            if self.maximum_capacity > 0
            else None
        )
        recovered = dict(self._recovered)
        if recovered["performed"] and "ledger" not in recovered:
            recovered["ledger"] = (
                self.recovery.report() if self.recovery.steps else None
            )
        complete = bool(service_complete and recovered["complete"])
        physical_qualified = bool(all(checks.values()) and recovered["closed"])
        delivered_gate = bool(ratio is not None and ratio >= EXTERNAL_SERVICE_FLOOR)
        observed_cash = (
            self.revenue - self.service.electricity_usd - self.recovery.electricity_usd
        )
        nominal = (
            self.service.nominal_electricity_usd + self.recovery.nominal_electricity_usd
        )
        grid = None
        if service_complete:
            import pandas as pd
            from evaluate import grid_metrics

            grid = grid_metrics(
                pd.DataFrame(self._grid_rows, columns=_GRID_COLUMNS),
                annual_median=self.fast._grid._load_median,
            )
        return {
            "task_schema": TASK_SCHEMA,
            "condition": dict(self.cell),
            "cell": dict(self.cell),
            "condition_role": self.condition_role,
            "decisions": self.decisions,
            "service_fast_steps": self.service.steps,
            "recovery_fast_steps": self.recovery.steps,
            "complete": complete,
            "service_complete": bool(service_complete),
            "service_physical_terminated": bool(physical_term),
            "physical_terminated": bool(
                physical_term or recovered.get("physical_terminated", False)
            ),
            "raw_service_truncated": bool(service_trunc),
            "termination_reason": reason,
            "true_terminal": self.done,
            "truncated": False,
            "timeout_bootstrap": False,
            "service_checks": checks,
            "physical_qualified": physical_qualified,
            "qualified": bool(physical_qualified and delivered_gate),
            "delivered_service_floor_required": True,
            "delivered_service_floor": EXTERNAL_SERVICE_FLOOR,
            "delivered_service_gate": delivered_gate,
            "delivered_service_ratio": ratio,
            "committed_mw_hours": self.committed_capacity,
            "performance_weighted_mw_hours": self.delivered_capacity,
            "maximum_reference_mw_hours": self.maximum_capacity,
            "initial": dict(self.initial),
            "service_terminal": terminal,
            "service": service,
            "recovery": recovered,
            "grid": grid,
            "regulation_revenue_usd": self.revenue,
            "ideal_regulation_reference_usd": self.ideal_revenue,
            "contractual_demand_electricity_usd": self.demand_electricity_usd,
            "observed_cashflow_usd": observed_cash,
            "cashflow_26h_usd": observed_cash if complete else None,
            "cashflow_closed_recovery_usd": (
                observed_cash if complete and recovered["closed"] else None
            ),
            "cash_vs_nominal_26h_usd": (
                observed_cash
                + self.service.nominal_electricity_usd
                + self.recovery.nominal_electricity_usd
                if complete
                else None
            ),
            "exogenous_reference_cash_usd": (
                self.ideal_revenue + self.demand_electricity_usd - nominal
                if complete
                else None
            ),
            "credit": {
                "target": GUARD_SERVICE_TARGET,
                "reserve_mwh": CREDIT_RESERVE_MWH,
                "final_mwh": self.budget.credit_mwh,
                "minimum_mwh": self.budget.minimum_credit_mwh,
                "absolute_error_mwh": self.budget.absolute_error_mwh,
                "metered_committed_mwh": self.budget.committed_mwh,
                "performance_reference_credit_mwh": (
                    self.delivered_capacity
                    - GUARD_SERVICE_TARGET * self.maximum_capacity
                ),
                "guard_binding_ticks": self.guard_binding_ticks,
                "requested_extra_mwh": self.requested_extra_mwh,
                "guarded_extra_mwh": self.guarded_extra_mwh,
                "absolute_service_guarantee_claimed": False,
            },
            "bound_derivation": dict(self.bound_derivation),
            "original_service_reward_sum": self.original_service_reward,
            "original_macro_reward_sum": self.original_macro_reward,
            "original_recovery_reward_sum": self.original_recovery_reward,
            "inventory_accumulation_is_not_an_action": True,
            "fixed_offsets_are_efficiency_dispatch_not_time_shifting": True,
        }

    def step(self, action):
        if self._closed or self.done:
            raise RuntimeError("Reset before stepping a completed task")
        requested = requested_extra_response(action)
        try:
            return self._advance(action, requested)
        except Exception as error:
            self.done = True
            self.last_episode = self._episode(
                service_complete=False,
                physical_term=False,
                service_trunc=False,
                reason="implementation_or_support_error",
            )
            self.last_episode.update(
                error=repr(error),
                reward_sum=self.reward_sum,
                exception_propagated=True,
                invalid_score_is_not_money=True,
            )
            raise

    def _advance(self, action, requested):
        self._requested_extra_mw = requested
        self._window_end_tick = min(self.fast._tick + WINDOW_TICKS, SERVICE_TICKS)
        before_cost = self.service.electricity_usd
        before_nominal = self.service.nominal_electricity_usd
        before_demand = self.demand_electricity_usd
        before_steps, before_regd = self.service.steps, self.pre_regd_sum
        before_credit = self.budget.credit_mwh
        native = self.macro.predict(self._macro_observation.copy())[0]
        self.decisions += 1
        observation, original_reward, term, trunc, info = self.core.step(native)
        self._macro_observation = observation.copy()
        self._fast_observation = self.core._last_hw_obs.copy()
        self.original_macro_reward += original_reward
        if (
            info["bid_price"] != 0
            or not 5 <= info["bid_mw"] <= min(30, info["residual_mw"])
            or not 0 <= info["rate"] <= RATE_BOUND_USD_MW_H
            or not 0 <= info["perf_score"] <= 1
            or info["sub_steps_run"] != self.service.steps - before_steps
            or not math.isclose(
                info["electricity_cost"],
                self.service.electricity_usd - before_cost,
                abs_tol=1e-6,
                rel_tol=1e-12,
            )
        ):
            raise AssertionError("Frozen bidding, settlement or metering changed")
        expected_rate = info["rmcp_usd"] if info["bid_accepted"] else 5.0
        if info["rate"] != expected_rate:
            raise AssertionError("Expected posted clearing price or fixed fallback")
        hours, capacity = info["duration_hours"], info["committed_mw"]
        self.maximum_capacity += service_capacity_reference(info) * hours
        self.committed_capacity += capacity * hours
        self.delivered_capacity += capacity * info["perf_score"] * hours
        ideal = info["rate"] * capacity * hours
        self.revenue += info["regulation_revenue"]
        self.ideal_revenue += ideal
        components = {
            "actual_regulation_revenue_usd": info["regulation_revenue"],
            "ideal_regulation_reference_usd": ideal,
            "service_centered_electricity_usd": (
                self.service.nominal_electricity_usd
                - before_nominal
                - (self.service.electricity_usd - before_cost)
            ),
            "contractual_demand_electricity_usd": (
                self.demand_electricity_usd - before_demand
            ),
            "recovery_centered_electricity_usd": 0.0,
        }
        self.done = bool(term or trunc)
        episode = None
        if self.done:
            self._service_terminal = plant_state(self.fast)
            complete = bool(
                trunc
                and not term
                and self.service.steps == SERVICE_TICKS
                and info["last_inner_info"]["truncated"]
                and not info["last_inner_info"]["terminated"]
                and self.fast._episode_ticks == SERVICE_TICKS
                and self.decisions == 96
            )
            if complete:
                self._recover(info["last_inner_info"])
                components["recovery_centered_electricity_usd"] = (
                    self.recovery.nominal_electricity_usd
                    - self.recovery.electricity_usd
                )
            reason = (
                "service_physical_fault"
                if term
                else "service_horizon_mismatch"
                if not complete
                else "recovery_physical_fault"
                if self._recovered.get("physical_terminated", False)
                else "recovery_horizon_mismatch"
                if not self._recovered["complete"]
                else "finite_26h_completion"
            )
            episode = self._episode(
                service_complete=complete,
                physical_term=term,
                service_trunc=trunc,
                reason=reason,
            )
            episode["service_last_faults"] = {
                key: bool(info["last_inner_info"][key])
                for key in ("thermal_fault", "freq_fault", "voltage_fault")
            }
        increment = (
            components["actual_regulation_revenue_usd"]
            - components["ideal_regulation_reference_usd"]
            + components["service_centered_electricity_usd"]
            - components["contractual_demand_electricity_usd"]
            + components["recovery_centered_electricity_usd"]
        )
        reward, adjustment, self.excess_cash = excess_transition(
            self.excess_cash,
            increment,
            self.excess_bound_usd,
            terminal=self.done,
            valid=episode is None or episode["qualified"],
        )
        self.centered_cash = (
            self.revenue
            + self.service.nominal_electricity_usd
            - self.service.electricity_usd
            + self.recovery.nominal_electricity_usd
            - self.recovery.electricity_usd
        )
        if abs(self.centered_cash) > FROZEN_CENTERED_BOUND_USD:
            raise ValueError("Actual centered cash exceeded its frozen enclosure")
        components.update(excess_cash_usd=increment, invalid_adjustment_usd=adjustment)
        for key, value in components.items():
            self._reward_totals[key] += value
        self.reward_sum += reward
        if episode is not None:
            expected_return = (
                self.excess_cash / REWARD_SCALE_USD
                if episode["qualified"]
                else -2 * self.excess_bound_usd / REWARD_SCALE_USD
            )
            if not math.isclose(self.reward_sum, expected_return, abs_tol=1e-8):
                raise AssertionError("Finite cash reward/clawback did not telescope")
            if episode["complete"] and not math.isclose(
                self.excess_cash,
                episode["cashflow_26h_usd"] - episode["exogenous_reference_cash_usd"],
                rel_tol=1e-10,
                abs_tol=1e-6,
            ):
                raise AssertionError("Actual 26h cash/reference identity failed")
            episode.update(
                reward_sum=self.reward_sum,
                excess_cash_usd=self.excess_cash,
                centered_cash_usd=self.centered_cash,
                reward_components=dict(self._reward_totals),
                invalid_score_is_not_money=True,
                required_gamma=REQUIRED_GAMMA,
                reward_scale_usd=REWARD_SCALE_USD,
            )
            self.last_episode = episode
        if self.record_trace:
            self.macro_rows.append(
                {
                    **{k: v for k, v in info.items() if k != "last_inner_info"},
                    "kind": "full_residual",
                    "signed_regd_mean": (self.pre_regd_sum - before_regd)
                    / (self.service.steps - before_steps),
                    "maximum_reference_offer_mw": maximum_native_quantity(
                        info["residual_mw"]
                    ),
                    "reference_capacity_mw": service_capacity_reference(info),
                    "overlay_action": float(np.asarray(action)[0]),
                    "desired_extra_response_mw": requested,
                    "credit_before_mwh": before_credit,
                    "credit_after_mwh": self.budget.credit_mwh,
                    **components,
                    "training_reward": reward,
                }
            )
        result_info = {
            "condition": dict(self.cell),
            "condition_role": self.condition_role,
            "native_market_action": native.copy(),
            "desired_extra_response_mw": requested,
            "service_fast_steps": self.service.steps - before_steps,
            "recovery_fast_steps": self.recovery.steps if self.done else 0,
            "bid_mw": info["bid_mw"],
            "bid_accepted": info["bid_accepted"],
            "committed_mw": capacity,
            "credit_mwh": self.budget.credit_mwh,
            "reward_components": components,
            "raw_service_truncated": bool(trunc),
            "physical_terminated": bool(
                term or self._recovered.get("physical_terminated", False)
            ),
            "true_terminal": self.done,
            "timeout_bootstrap": False,
            "episode_result": episode,
        }
        return self._observation(), float(reward), self.done, False, result_info

    def progress(self):
        if not self._initialized:
            return {"initialized": False, "closed": self._closed}
        return {
            "condition": dict(self.cell),
            "condition_role": self.condition_role,
            "decisions": self.decisions,
            "service_fast_steps": self.service.steps,
            "recovery_fast_steps": self.recovery.steps,
            "done": self.done,
            "excess_cash_usd": self.excess_cash,
            "reward_sum": self.reward_sum,
            "credit_mwh": self.budget.credit_mwh,
            "cashflow_26h_usd": (
                self.last_episode["cashflow_26h_usd"]
                if self.done and self.last_episode is not None
                else None
            ),
        }

    def terminal_traces(self) -> dict[str, list[dict]]:
        """Copy public raw rows for explicit caller persistence before reset.

        Invalid episodes retain their partial rows. Qualification and phase
        completeness must be read from episode_result, not inferred from files.
        """
        if not self.record_trace:
            raise ValueError("Raw trace export requires record_trace=True")
        if not self._initialized or not self.done or self.last_episode is None:
            raise RuntimeError("Raw trace export requires a terminal/closed episode")
        return copy.deepcopy(
            {
                "hardware.csv.gz": self.service_rows,
                "macro.csv.gz": self.macro_rows,
                "recovery.csv.gz": self.recovery_rows,
            }
        )

    def close(self):
        if self._closed:
            return
        try:
            if self._initialized and not self.done:
                self.last_episode = self._episode(
                    service_complete=False,
                    physical_term=False,
                    service_trunc=False,
                    reason="administrative_close_before_completion",
                )
                self.last_episode.update(
                    reward_sum=self.reward_sum,
                    true_terminal=False,
                    administrative_abort=True,
                    invalid_penalty_returned=False,
                )
            self.fast.close()
        finally:
            self.core.close()
            self.done, self._closed = True, True
