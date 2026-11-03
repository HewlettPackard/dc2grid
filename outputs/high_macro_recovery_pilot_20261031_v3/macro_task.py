"""Finite-horizon quantity SAC task with unchanged low-level control and recovery."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

# isort: off
from study_common import (
    OBSERVATION_DIM,
    TASK_SCHEMA,
    FrozenHardware,
)
from recovery_ops import battery_dispatch, begin_recovery, closure, plant_state

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from c2g_env import C2GMacroEnv
# isort: on


def quantity_bounds(core) -> tuple[np.float32, np.float32]:
    offer = core._pending_offer
    if offer is None:
        raise RuntimeError("Quantity choice requires the currently posted offer")
    maximum = core._committed_max_mw
    lower = np.float32(core._dr_baseline_mw / maximum)
    upper = np.float32(min(maximum, offer["residual_mw"]) / maximum)
    if float(lower) * maximum < core._dr_baseline_mw:
        lower = np.nextafter(lower, np.float32(np.inf))
    if float(upper) * maximum > min(maximum, offer["residual_mw"]):
        upper = np.nextafter(upper, np.float32(0.0))
    if not 0 <= lower <= upper <= 1:
        raise ValueError("Posted residual does not admit the registered quantity range")
    return lower, upper


def map_quantity(core, fraction: float) -> tuple[np.ndarray, float]:
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Quantity fraction must be finite and within [0,1]")
    lower, upper = quantity_bounds(core)
    native = np.float32(float(lower) + fraction * (float(upper) - float(lower)))
    native = np.clip(native, lower, upper)
    return np.array([native, 0.0], dtype=np.float32), float(
        upper
    ) * core._committed_max_mw


def potential(soc: float, kappa: float, *, target: float, margin: float) -> float:
    if margin <= 0 or not 0 <= kappa <= 1:
        raise ValueError("Invalid reserve potential parameters")
    return -kappa * min(max((target - soc) / margin, 0.0), 1.0)


def score_transition(
    cash_before: float,
    cash_increment: float,
    bound: float,
    phi_before: float,
    phi_after: float,
    *,
    terminal: bool,
    valid: bool,
) -> tuple[float, float, float]:
    cash_after = cash_before + cash_increment
    if bound <= 0 or not math.isfinite(cash_after) or abs(cash_after) > bound:
        raise ValueError("Centered cash breached the registered normalization bound")
    if terminal and not valid:
        base_reward = -2.0 - cash_before / bound
    else:
        base_reward = cash_increment / bound
    if terminal:
        phi_after = 0.0
    return base_reward, phi_after - phi_before, cash_after


@dataclass
class Ledger:
    initial_backlog_kwh: float = 0.0
    steps: int = 0
    hours: float = 0.0
    electricity_usd: float = 0.0
    nominal_electricity_usd: float = 0.0
    grid_energy_mwh: float = 0.0
    facility_energy_mwh: float = 0.0
    battery_throughput_mwh: float = 0.0
    work_arrived_kwh: float = 0.0
    work_served_kwh: float = 0.0
    backlog_kwh: float = 0.0
    backlog_exposure_kwh_hours: float = 0.0
    max_backlog_kwh: float = 0.0
    thermal_warning_ticks: int = 0
    error_squared: float = 0.0
    normalized_error_squared: float = 0.0
    signed_regd_sum: float = 0.0
    requested_energy_mwh: float = 0.0
    lmp_hours: float = 0.0

    def update(self, fast, info: dict, normalization: dict):
        values = (
            info["p_grid_import_mw"],
            info["p_reference_mw"],
            info["lmp"],
            info["duration_hours"],
            info["tracking_err_kw"],
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Nonfinite physical or price observation")
        if (
            abs(info["p_grid_import_mw"]) > normalization["import_abs_mw"]
            or abs(info["p_reference_mw"]) > normalization["nominal_abs_mw"]
            or abs(info["lmp"]) > normalization["lmp_abs_usd_mwh"] + 1e-8
            or abs(info["T_amb"]) > normalization["ambient_abs_C"]
        ):
            raise ValueError(
                "Cash-bound support exceeded at "
                f"{info['timestamp']}: grid={info['p_grid_import_mw']}, "
                f"reference={info['p_reference_mw']}, price={info['lmp']}, "
                f"ambient={info['T_amb']}"
            )
        hours, lmp = info["duration_hours"], info["lmp"]
        self.steps += 1
        self.hours += hours
        self.electricity_usd += info["p_grid_import_mw"] * lmp * hours
        self.nominal_electricity_usd += info["p_reference_mw"] * lmp * hours
        self.grid_energy_mwh += info["p_grid_import_mw"] * hours
        self.facility_energy_mwh += info["p_facility_mw"] * hours
        self.battery_throughput_mwh += abs(info["bess_actual_kw"]) * hours / 1000
        self.work_arrived_kwh += info["work_arrived_kwh"]
        self.work_served_kwh += info["work_served_kwh"]
        self.backlog_kwh = info["backlog_work_kwh"]
        self.backlog_exposure_kwh_hours += self.backlog_kwh * hours
        self.max_backlog_kwh = max(self.max_backlog_kwh, self.backlog_kwh)
        self.thermal_warning_ticks += int(info["thermal_warning"])
        self.error_squared += info["tracking_err_kw"] ** 2
        self.normalized_error_squared += (
            info["tracking_err_kw"]
            / max(fast.committed_mw * 1000, fast._tracking_floor_kw)
        ) ** 2
        self.signed_regd_sum += info["regd_signal"]
        self.requested_energy_mwh += info["delta_p_demanded_kw"] * hours / 1000
        self.lmp_hours += lmp * hours

    def report(self) -> dict:
        if not self.steps:
            raise ValueError("Cannot summarize an empty physical phase")
        conservation = (
            self.initial_backlog_kwh
            + self.work_arrived_kwh
            - self.work_served_kwh
            - self.backlog_kwh
        )
        if abs(conservation) >= 1e-4:
            raise AssertionError(f"Work conservation failure: {conservation}")
        return {
            **asdict(self),
            "work_conservation_error_kwh": conservation,
            "tracking_rmse_kw": math.sqrt(self.error_squared / self.steps),
            "normalized_rmse": math.sqrt(self.normalized_error_squared / self.steps),
            "thermal_warning_fraction": self.thermal_warning_ticks / self.steps,
            "time_weighted_lmp_usd_mwh": self.lmp_hours / self.hours,
        }


class MacroQuantityTask(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        protocol: dict,
        kappa: float,
        *,
        run_seed: int = 5000,
        rank: int = 0,
        cell: dict | None = None,
        record_trace: bool = False,
    ):
        super().__init__()
        self.protocol, self.kappa = protocol, float(kappa)
        self.run_seed, self.rank, self.fixed_cell = run_seed, rank, cell
        self.record_trace = record_trace
        self.episode_ordinal = 0
        self.hardware = FrozenHardware(protocol["hardware"])
        self.core = C2GMacroEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=protocol["config_path"],
            inner_action_fn=self.hardware,
            sub_step_callback=self._service_callback,
        )
        self._configured_service_ticks = self.core._fast_env._episode_ticks
        self._configured_macro_ticks = self.core._episode_macro_ticks
        low, high = (
            self.core.observation_space.low.copy(),
            self.core.observation_space.high.copy(),
        )
        high[8] = np.inf
        scales = protocol["observation_scales"]
        fast = self.core._fast_env
        request_bound = (
            fast._committed_mw_max
            * fast._episode_ticks
            * fast._dt
            / 3600
            / scales["requested_energy_mwh"]
        )
        cash_bound = (
            protocol["normalization"]["cash_prefix_bound_usd"] / scales["cash_usd"]
        )
        self.observation_space = spaces.Box(
            np.r_[low, [0, -1, -1, -request_bound, -cash_bound, 0, 0, 0, 0]].astype(
                np.float32
            ),
            np.r_[
                high,
                [
                    1,
                    1,
                    1,
                    request_bound,
                    cash_bound,
                    1 / protocol["service_gate"]["max_thermal_warning_fraction"],
                    1,
                    1,
                    1,
                ],
            ].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space.c2g_simulation_mode = "physical_v1"
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.done = True

    def _condition(self) -> dict:
        if self.fixed_cell is not None:
            return dict(self.fixed_cell)
        days, biases = self.protocol["train_days"], self.protocol["train_biases"]
        size = len(days) * len(biases)
        slot = self.episode_ordinal * self.protocol["n_envs"] + self.rank
        cycle, position = divmod(slot, size)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.run_seed, cycle, 5813])
        )
        choice = int(rng.permutation(size)[position])
        day, bias = days[choice // len(biases)], biases[choice % len(biases)]
        seed = int(
            np.random.SeedSequence(
                [self.run_seed, cycle, position, 971]
            ).generate_state(1)[0]
        )
        return {
            "id": f"train_{self.run_seed}_{cycle}_{position}",
            "start_day": day,
            "regd_load_bias": bias,
            "seed": seed,
            "curriculum_cycle": cycle,
            "curriculum_position": position,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Conditions are fixed by the registered curriculum")
        self.condition = self._condition()
        self.episode_ordinal += 1
        # Recovery extends the live limit; a new service episode must not inherit it.
        self.core._fast_env._episode_ticks = self._configured_service_ticks
        self.core._episode_macro_ticks = self._configured_macro_ticks
        observation, reset_info = self.core.reset(
            seed=self.condition["seed"],
            options={
                "start_day": self.condition["start_day"],
                "regd_load_bias": self.condition["regd_load_bias"],
            },
        )
        self.service_ticks = self.core._fast_env._episode_ticks
        self.initial = plant_state(self.core._fast_env)
        self.hardware_start_calls = self.hardware.predictions
        if self.initial["backlog_kwh"] != 0:
            raise ValueError("This task requires an empty initial work queue")
        self.service = Ledger()
        self.recovery = Ledger()
        self.revenue = self.centered_cash = 0.0
        self.committed_capacity = self.delivered_capacity = self.maximum_capacity = 0.0
        self.base_return = self.shaped_return = 0.0
        self.decisions = self.shaping_active_steps = 0
        self.last_signed_mean = 0.0
        self.phi = self._potential()
        self.initial_phi = self.phi
        self.done = False
        self.last_episode = None
        self.service_rows, self.recovery_rows, self.macro_rows = [], [], []
        return self._observation(observation), {
            **reset_info,
            "condition": self.condition,
            "task_schema": TASK_SCHEMA,
            "hardware_tensor_digest": self.hardware.digest,
        }

    def _potential(self) -> float:
        return potential(
            self.core._fast_env._bess.soc_fraction,
            self.kappa,
            target=self.protocol["potential"]["reserve_target_soc"],
            margin=self.protocol["potential"]["margin"],
        )

    def _record(self, pre, post, info, *, recovery=False):
        if not self.record_trace:
            return
        fast = self.core._fast_env
        row = {
            **info,
            **fast._elec.get_diagnostics(),
            "observed_committed_mw": float(pre[17]) * fast._committed_mw_max,
            "committed_mw": fast.committed_mw,
            "p_base_norm": float(post[3]),
            "p_flex_norm": float(post[4]),
            "load_anomaly": float(fast._grid._load_anomaly[info["tick"] - 1]),
            "reference_temp_A": fast._reference_thermal.temp_A,
            "reference_temp_B": fast._reference_thermal.temp_B,
        }
        (self.recovery_rows if recovery else self.service_rows).append(row)

    def _service_callback(self, pre, action, post, reward, done, info):
        self.service.update(self.core._fast_env, info, self.protocol["normalization"])
        self._record(pre, post, info)

    def _observation(self, core_observation):
        scales = self.protocol["observation_scales"]
        capacity = (
            self.core._committed_max_mw
            * self.service_ticks
            * self.core._fast_env._dt
            / 3600
        )
        work_ratio = (
            self.service.backlog_kwh / self.service.work_arrived_kwh
            if self.service.work_arrived_kwh > 0
            else 0.0
        )
        if self.service.work_arrived_kwh == 0 and self.service.backlog_kwh != 0:
            raise AssertionError("Positive work debt without any arrivals")
        exposure = (
            self.delivered_capacity / self.maximum_capacity
            if self.maximum_capacity
            else 1.0
        )
        extra = [
            1 - self.service.steps / self.service_ticks,
            self.core._fast_env._prev_regd_signal,
            self.last_signed_mean,
            self.service.requested_energy_mwh / scales["requested_energy_mwh"],
            self.centered_cash / scales["cash_usd"],
            self.service.thermal_warning_ticks
            / (
                self.service_ticks
                * self.protocol["service_gate"]["max_thermal_warning_fraction"]
            ),
            work_ratio,
            min(exposure, 1.0),
            self.maximum_capacity / capacity,
        ]
        result = np.r_[core_observation, extra].astype(np.float32)
        if result.shape != (OBSERVATION_DIM,) or not np.isfinite(result).all():
            raise FloatingPointError("Invalid augmented macro observation")
        return result

    def _recover(self, last_info: dict) -> tuple[dict, np.ndarray]:
        fast = self.core._fast_env
        recovery_cfg = self.protocol["recovery"]
        self.recovery.initial_backlog_kwh = self.service.backlog_kwh
        extension = begin_recovery(fast, self.condition, last_info, recovery_cfg)
        observations, infos = (
            deque(maxlen=fast.macro_substeps),
            deque(maxlen=fast.macro_substeps),
        )
        pre = self.core._last_hw_obs.copy()
        pre[17] = 0
        for _ in range(extension["recovery_ticks"]):
            action = np.array(
                [
                    recovery_cfg["throttle"],
                    recovery_cfg["pump"],
                    recovery_cfg["hvac"],
                    battery_dispatch(fast._bess, self.initial["soc"]),
                ],
                dtype=np.float32,
            )
            post, _, term, trunc, info = fast.step(action)
            self.recovery.update(fast, info, self.protocol["normalization"])
            self._record(pre, post, info, recovery=True)
            if fast.committed_mw != 0 or info["delta_p_demanded_kw"] != 0:
                raise AssertionError("Recovery inherited a service obligation")
            info["p_flex_max_kw"] = fast._workload.p_flex_max_kw
            observations.append(post)
            infos.append(info)
            pre = post
            if term or trunc:
                break
        complete = (
            self.recovery.steps == extension["recovery_ticks"] and trunc and not term
        )
        terminal = plant_state(fast)
        result = {
            "performed": True,
            "complete": complete,
            "physical_terminated": bool(term),
            "terminal": terminal,
            "extension": extension,
            "ledger": self.recovery.report(),
            "regulation_revenue_usd": 0.0,
            **closure(terminal, self.initial, recovery_cfg, complete=complete),
        }
        observation = C2GMacroEnv.aggregate_macro_obs(
            list(observations),
            list(infos),
            T_safe=fast._thermal.T_safe,
            committed_mw=0.0,
            tracking_floor_kw=fast._macro_tracking_floor_kw,
            bid_mw_norm=0.0,
            bid_price_norm=0.0,
            rmcp_norm=0.0,
            reg_need_norm=0.0,
        )
        self.last_signed_mean = float(
            np.mean([entry["regd_signal"] for entry in infos])
        )
        return result, observation

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset the finished finite-horizon task")
        action = np.asarray(action)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Quantity policy must return one finite action")
        native, maximum_accepted = map_quantity(self.core, float(action[0]))
        cash_before = self.centered_cash
        cost_before = self.service.electricity_usd
        nominal_before = self.service.nominal_electricity_usd
        regd_before, steps_before = self.service.signed_regd_sum, self.service.steps
        self.decisions += 1
        observation, _, physical_term, service_trunc, info = self.core.step(native)
        if (
            info["bid_price"] != 0
            or info["bid_mw"] > info["residual_mw"]
            or not math.isclose(
                info["accept_prob"],
                1 / (1 + math.exp(-0.15 * info["rmcp_usd"])),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise AssertionError("Quantity bounds no longer preserve common admission")
        self.revenue += info["regulation_revenue"]
        hours = info["duration_hours"]
        reference_capacity = (
            maximum_accepted if info["bid_accepted"] else self.core._dr_baseline_mw
        )
        self.maximum_capacity += reference_capacity * hours
        self.committed_capacity += info["committed_mw"] * hours
        self.delivered_capacity += info["committed_mw"] * info["perf_score"] * hours
        if self.delivered_capacity > self.maximum_capacity + 1e-6:
            raise AssertionError(
                "Delivered service exceeds its unit-tracking counterfactual"
            )
        self.last_signed_mean = (self.service.signed_regd_sum - regd_before) / (
            self.service.steps - steps_before
        )
        cash_increment = (
            info["regulation_revenue"]
            - (self.service.electricity_usd - cost_before)
            + (self.service.nominal_electricity_usd - nominal_before)
        )
        if self.service.steps > self._configured_service_ticks:
            raise RuntimeError("Service exceeded its configured physical horizon")
        if (
            service_trunc
            and not physical_term
            and (
                self.service.steps != self._configured_service_ticks
                or not info["last_inner_info"]["truncated"]
                or self.core._fast_env._episode_ticks != self._configured_service_ticks
            )
        ):
            raise RuntimeError("Macro/fast service horizon disagreement")
        self.done = bool(physical_term or service_trunc)
        episode = None
        valid = True
        if self.done:
            service_terminal = plant_state(self.core._fast_env)
            service_report = self.service.report()
            complete = (
                service_trunc
                and not physical_term
                and self.service.steps == self.service_ticks
            )
            gate = self.protocol["service_gate"]
            original_gate = (
                complete
                and service_terminal["soc"] >= gate["min_terminal_soc"]
                and self.service.backlog_kwh
                <= gate["max_backlog_fraction"] * self.service.work_arrived_kwh + 1e-6
                and service_report["thermal_warning_fraction"]
                <= gate["max_thermal_warning_fraction"]
            )
            exposure_ratio = self.delivered_capacity / self.maximum_capacity
            exposure_gate = exposure_ratio >= self.protocol["minimum_service_fraction"]
            recovered = {"performed": False, "complete": False, "closed": False}
            if complete:
                recovered, observation = self._recover(info["last_inner_info"])
                cash_increment += (
                    self.recovery.nominal_electricity_usd
                    - self.recovery.electricity_usd
                )
            valid = bool(original_gate and exposure_gate and recovered["closed"])
            money_complete = bool(complete and recovered["complete"])
            episode = {
                "condition": self.condition,
                "decisions": self.decisions,
                "service_complete": bool(complete),
                "expected_service_steps": self._configured_service_ticks,
                "raw_macro_truncated": bool(service_trunc),
                "raw_fast_truncated": bool(info["last_inner_info"]["truncated"]),
                "service_physical_terminated": bool(physical_term),
                "original_service_gate": bool(original_gate),
                "delivered_service_gate": bool(exposure_gate),
                "delivered_service_ratio": exposure_ratio,
                "qualified": valid,
                "initial": self.initial,
                "service_terminal": service_terminal,
                "service": service_report,
                "recovery": recovered,
                "regulation_revenue_usd": self.revenue,
                "committed_capacity_mw_hours": self.committed_capacity,
                "performance_weighted_capacity_mw_hours": self.delivered_capacity,
                "maximum_accepted_capacity_mw_hours": self.maximum_capacity,
                "observed_cashflow_usd": self.revenue
                - self.service.electricity_usd
                - self.recovery.electricity_usd,
                "cashflow_26h_usd": self.revenue
                - self.service.electricity_usd
                - self.recovery.electricity_usd
                if money_complete
                else None,
                "cashflow_closed_recovery_usd": self.revenue
                - self.service.electricity_usd
                - self.recovery.electricity_usd
                if money_complete and recovered["closed"]
                else None,
                "hardware_tensor_digest": self.hardware.verify(),
                "hardware_predictions": self.hardware.predictions
                - self.hardware_start_calls,
                "task_schema": TASK_SCHEMA,
            }
        phi_after = 0.0 if self.done else self._potential()
        base_reward, delta_phi, self.centered_cash = score_transition(
            cash_before,
            cash_increment,
            self.protocol["normalization"]["cash_prefix_bound_usd"],
            self.phi,
            phi_after,
            terminal=self.done,
            valid=valid,
        )
        self.base_return += base_reward
        self.shaped_return += base_reward + delta_phi
        self.shaping_active_steps += int(delta_phi != 0)
        self.phi = phi_after
        if episode is not None:
            if episode["hardware_predictions"] != self.service.steps:
                raise AssertionError("Frozen hardware decision cadence changed")
            expected = (
                self.centered_cash
                / self.protocol["normalization"]["cash_prefix_bound_usd"]
                if valid
                else -2.0
            )
            if not math.isclose(self.base_return, expected, abs_tol=1e-9):
                raise AssertionError("Terminal cash clawback identity failed")
            if not math.isclose(
                self.shaped_return, self.base_return - self.initial_phi, abs_tol=1e-9
            ):
                raise AssertionError("Reserve potential did not telescope")
            episode.update(
                base_return=self.base_return,
                shaped_return=self.shaped_return,
                centered_cash_usd=self.centered_cash,
                shaping_active_steps=self.shaping_active_steps,
                invalid_score_is_not_money=True,
            )
            self.last_episode = episode
        if self.record_trace:
            self.macro_rows.append(
                {
                    **{
                        key: value
                        for key, value in info.items()
                        if key != "last_inner_info"
                    },
                    "quantity_fraction": float(action[0]),
                    "base_reward": base_reward,
                    "potential_delta": delta_phi,
                    "training_reward": base_reward + delta_phi,
                    "centered_cash_increment_usd": cash_increment,
                }
            )
        result_info = {
            "condition": self.condition,
            "quantity_fraction": float(action[0]),
            "bid_mw": info["bid_mw"],
            "committed_mw": info["committed_mw"],
            "service_fast_steps": self.service.steps - steps_before,
            "recovery_fast_steps": self.recovery.steps if self.done else 0,
            "task_ended": self.done,
            "physical_terminated": bool(
                physical_term
                or (
                    episode is not None
                    and episode["recovery"].get("physical_terminated", False)
                )
            ),
            "base_reward": base_reward,
            "potential_delta": delta_phi,
        }
        if episode is not None:
            result_info["c2g_episode"] = episode
        return (
            self._observation(observation),
            base_reward + delta_phi,
            self.done,
            False,
            result_info,
        )

    def progress(self) -> dict:
        return {
            "condition": self.condition,
            "decisions": self.decisions,
            "service_steps": self.service.steps,
            "recovery_steps": self.recovery.steps,
            "done": self.done,
            "centered_cash_usd": self.centered_cash,
            "base_return": self.base_return,
            "shaped_return": self.shaped_return,
        }

    def worker_status(self) -> dict:
        import os

        import torch

        return {
            "pid": os.getpid(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "hardware_device": str(self.hardware.model.device),
            "hardware_tensor_digest": self.hardware.verify(),
            "hardware_load_seconds": self.hardware.load_seconds,
        }

    def close(self):
        self.core.close()
