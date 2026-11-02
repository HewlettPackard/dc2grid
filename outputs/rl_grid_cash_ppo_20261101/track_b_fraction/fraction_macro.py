"""Dev-informed fractional-window dispatch and optional invariant shaping."""

# ruff: noqa: E402

from __future__ import annotations

import copy
import math
import os
import sys
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
from track_b.economy_macro import (
    DT_HOURS,
    OBSERVATION_SCHEMA,
    REQUIRED_GAMMA,
    REWARD_SCALE_USD,
    TASK_SCHEMA as PARENT_TASK_SCHEMA,
    EconomicDispatchTask,
    requested_extra_response,
)

# Do not allow the frozen bootstrap's cache location to receive new writes.
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

import gymnasium as gym
import numpy as np
# isort: on

TASK_SCHEMA = "track-b-fractional-window-economic-dispatch-v1"
ACTION_SEMANTICS = "fraction_of_frozen_start_of_window_allowance"
POTENTIAL_PRICE_USD_PER_MWH = 10.0
V1_SNAPSHOT = "5e2bf2855a3da4ac8cfea1763df8e9d8b28e9f82"


def window_fraction(action) -> float:
    requested_extra_response(action)  # Retain the frozen action validation.
    return max(float(np.asarray(action)[0]), 0.0)


class FractionalDispatchTask(EconomicDispatchTask):
    """Spend a fraction of one post-handshake allowance, not each changing cap.

    Constructor, conditions, observer/controller, per-tick guard, metering,
    recovery, terminal handling and cash rewards are inherited unchanged.
    """

    task_schema = TASK_SCHEMA
    action_semantics = ACTION_SEMANTICS

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.fraction_windows = []
        self._pending_fraction = None
        self._window_dispatch = None
        return observation, {
            **info,
            "task_schema": TASK_SCHEMA,
            "parent_task_schema": PARENT_TASK_SCHEMA,
            "action_semantics": ACTION_SEMANTICS,
        }

    def _advance(self, action, requested):
        self._pending_fraction = window_fraction(action)
        self._window_dispatch = None
        # The actual MW request is unknown until the inherited handshake clears.
        observation, reward, term, trunc, info = super()._advance(action, 0.0)
        if self._window_dispatch is None:
            raise RuntimeError("A dispatched window must have a hardware tick")
        dispatch = {
            **self._window_dispatch,
            "desired_extra_response_mw": self._window_dispatch[
                "fixed_window_request_mw"
            ],
        }
        # The base's window summary uses its pre-handshake argument. Replace only
        # that summary with the actual fixed request already metered tick by tick.
        if self.record_trace:
            self.macro_rows[-1].update(dispatch)
        return (
            observation,
            reward,
            term,
            trunc,
            {
                **info,
                **dispatch,
                "task_schema": TASK_SCHEMA,
                "action_semantics": ACTION_SEMANTICS,
            },
        )

    def _inner_action(self, observation, macro_action):
        if self._window_dispatch is None:
            if self._pending_fraction is None:
                raise RuntimeError("Choose a window fraction before dispatch")
            full_hours = (self._window_end_tick - self.fast._tick) * DT_HOURS
            allowance = self.budget.cap(self.fast.committed_mw, full_hours)
            self._window_dispatch = {
                "dispatch_window_index": self.decisions - 1,
                "window_start_tick": self.fast._tick,
                "window_committed_mw": self.fast.committed_mw,
                "window_credit_start_mwh": self.budget.credit_mwh,
                "full_current_window_hours": full_hours,
                "initial_allowance_mw": allowance,
                "window_fraction": self._pending_fraction,
                "fixed_window_request_mw": self._pending_fraction * allowance,
            }
            self.fraction_windows.append(dict(self._window_dispatch))
            self._requested_extra_mw = self._window_dispatch["fixed_window_request_mw"]
        action = super()._inner_action(observation, macro_action)
        self._tick_dispatch.update(self._window_dispatch)
        return action

    def _episode(self, **kwargs):
        episode = super()._episode(**kwargs)
        windows = getattr(self, "fraction_windows", [])
        episode.update(
            task_schema=TASK_SCHEMA,
            parent_task_schema=PARENT_TASK_SCHEMA,
            action_semantics=ACTION_SEMANTICS,
            fraction_window_count=len(windows),
            fraction_windows=copy.deepcopy(windows),
            adaptive_dev_informed_followup=True,
            fresh_validation_claimed=False,
        )
        return episode


def credit_potential(observation) -> float:
    """10 USD/MWh times the observed credit, in the unchanged 100 USD units."""
    observation = np.asarray(observation)
    if (
        observation.shape != (len(OBSERVATION_SCHEMA),)
        or not np.issubdtype(observation.dtype, np.number)
        or np.iscomplexobj(observation)
        or not np.isfinite(observation).all()
    ):
        raise ValueError("Credit potential requires the finite causal 23D observation")
    value = (
        POTENTIAL_PRICE_USD_PER_MWH * (float(observation[21]) * 30) / REWARD_SCALE_USD
    )
    if not math.isfinite(value):
        raise ValueError("Nonfinite credit potential")
    return value


class CreditPotentialEnv(gym.Wrapper):
    """Training-only, gamma-one potential shaping; never replace monetary fields."""

    required_gamma = REQUIRED_GAMMA

    def __init__(
        self,
        env: FractionalDispatchTask,
        coefficient_usd_per_mwh: float = POTENTIAL_PRICE_USD_PER_MWH,
        *,
        gamma: float = 1.0,
    ):
        if not isinstance(env, FractionalDispatchTask):
            raise TypeError("CreditPotentialEnv requires a FractionalDispatchTask")
        if (
            isinstance(coefficient_usd_per_mwh, bool)
            or not isinstance(
                coefficient_usd_per_mwh, (int, float, np.integer, np.floating)
            )
            or coefficient_usd_per_mwh != POTENTIAL_PRICE_USD_PER_MWH
        ):
            raise ValueError(
                "This follow-up fixes the credit coefficient at 10 USD/MWh"
            )
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, (int, float, np.integer, np.floating))
            or gamma != REQUIRED_GAMMA
        ):
            raise ValueError("Invariant credit shaping requires gamma=1")
        if env.evaluation_mode:
            raise ValueError(
                "CreditPotentialEnv is training-only; deploy the unwrapped task"
            )
        super().__init__(env)
        self._coefficient_usd_per_mwh = float(coefficient_usd_per_mwh)
        self.last_episode = None
        self.raw_return = self.shaped_return = self.potential_delta_sum = 0.0
        self.initial_phi = self.phi = 0.0
        self.potential_rows = []
        self._done, self._closed, self._returned_terminal = True, False, False

    @property
    def episode_result(self):
        return self.last_episode

    @property
    def coefficient_usd_per_mwh(self):
        return self._coefficient_usd_per_mwh

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("Cannot reset a closed potential wrapper")
        observation, info = self.env.reset(seed=seed, options=options)
        self.initial_phi = self.phi = credit_potential(observation)
        if self.initial_phi != 0.0:
            raise ValueError("Invariant full-episode shaping requires initial Phi=0")
        self.raw_return = self.shaped_return = self.potential_delta_sum = 0.0
        self.potential_rows = []
        self._done, self._returned_terminal = False, False
        return observation, {
            **info,
            "potential_shaping": {
                "price_usd_per_mwh": POTENTIAL_PRICE_USD_PER_MWH,
                "initial_phi": self.initial_phi,
                "required_gamma": REQUIRED_GAMMA,
                "training_only": True,
            },
        }

    def _capture_episode(self):
        episode = copy.deepcopy(self.env.episode_result)
        if episode is None:
            return
        episode.update(
            raw_return=self.raw_return,
            shaped_return=self.shaped_return,
            potential_shaping={
                "price_usd_per_mwh": POTENTIAL_PRICE_USD_PER_MWH,
                "initial_phi": self.initial_phi,
                "final_phi": self.phi,
                "potential_delta_sum": self.potential_delta_sum,
                "required_gamma": REQUIRED_GAMMA,
                "training_only": True,
                "returned_terminal": self._returned_terminal,
                "telescoping_complete": self._returned_terminal,
                "coefficient_source": "already-registered margin10 heuristic",
                "adaptive_dev_informed_followup": True,
            },
        )
        self.last_episode = episode

    def step(self, action):
        if self._closed or self._done:
            raise RuntimeError("Reset before stepping a completed potential wrapper")
        try:
            observation, raw_reward, term, trunc, info = self.env.step(action)
        except Exception:
            if self.env.done:
                self._done = True
                self._capture_episode()
            raise
        if trunc:
            raise RuntimeError(
                "CreditPotentialEnv requires true terminals, not timeouts"
            )
        next_phi = 0.0 if term else credit_potential(observation)
        delta = next_phi - self.phi
        reward = float(raw_reward + delta)
        components = {
            "raw_reward": float(raw_reward),
            "shaped_reward": reward,
            "phi_before": self.phi,
            "phi_after": next_phi,
            "potential_delta": delta,
            "true_terminal": bool(term),
        }
        self.raw_return += float(raw_reward)
        self.shaped_return += reward
        self.potential_delta_sum += delta
        self.phi = next_phi
        self._done = self._returned_terminal = bool(term)
        if self.env.record_trace:
            self.potential_rows.append(dict(components))
        if term:
            if not math.isclose(
                self.shaped_return,
                self.raw_return - self.initial_phi,
                rel_tol=0.0,
                abs_tol=1e-8,
            ) or not math.isclose(self.potential_delta_sum, 0.0, abs_tol=1e-8):
                raise AssertionError("Credit potential did not telescope")
            if not math.isclose(
                self.raw_return,
                self.env.episode_result["reward_sum"],
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise AssertionError("The wrapper's raw return changed")
            self._capture_episode()
        return (
            observation,
            reward,
            term,
            False,
            {
                **info,
                "potential_shaping": components,
                "episode_result": self.last_episode if term else None,
            },
        )

    def terminal_traces(self) -> dict[str, list[dict]]:
        traces = self.env.terminal_traces()
        traces["potential.csv.gz"] = copy.deepcopy(self.potential_rows)
        return traces

    def progress(self):
        return {
            **self.env.progress(),
            "raw_return": self.raw_return,
            "shaped_return": self.shaped_return,
            "current_phi": self.phi,
            "potential_delta_sum": self.potential_delta_sum,
            "returned_terminal": self._returned_terminal,
        }

    def close(self):
        if self._closed:
            return
        try:
            self.env.close()
            if not self._returned_terminal:
                self._capture_episode()
        finally:
            self._done, self._closed = True, True


FractionalWindowTask = FractionalDispatchTask
PotentialEnv = CreditPotentialEnv
