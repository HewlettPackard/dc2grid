"""
baselines/pid_controller.py  —  Multi-Loop PID Controller
==========================================================
Three PID loops for thermal/SOC control, plus proportional BESS response.

Loop assignment
---------------
  Loop 0 (BESS → tracking):   bess_dispatch = regd_signal (proportional)
  Loop 1 (Pump → Zone A):     error = temp_A - T_setpoint  → pump_speed
  Loop 2 (HVAC → Zone B):     error = temp_B - T_setpoint  → hvac_effort
  Loop 3 (Throttle → SOC):    error = SOC - SOC_target     → throttle_bias

BESS uses direct proportional response (not PID) since regd_signal is an
external setpoint, not a controllable process variable — true feedback
control isn't possible. The other loops use standard PID with anti-windup.

Loop 3 (throttle) acts as a bias: throttle = 1.0 + bias (clamped [0, 1]).
When SOC is below target, bias < 0 → reduces throttle → sheds load →
frees capacity for BESS charging.

Default gains are hand-tuned for the default scenario and can be
overridden via constructor kwargs or conf/algo/pid.yaml.

Observation index map (C2GFastEnv, 17-D)
-----------------------------------------
  [0]  temp_A_norm     [1]  temp_B_norm     [2]  bess_soc
  [6]  regd_signal

Usage
-----
  from baselines.pid_controller import PIDController
  ctrl = PIDController()
  obs, _ = env.reset(seed=0)
  action, _ = ctrl.predict(obs)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from c2g_env.obs_indices import Fast as _F
from c2g_env.thermal_limits import T_SAFE as _T_SAFE


class _PIDLoop:
    """Single-channel PID with anti-windup integral clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float = 5.0,
        out_min: float = -1.0,
        out_max: float = 1.0,
        integral_limit: float = 10.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float) -> float:
        # Proportional
        p = self.kp * error
        # Integral with anti-windup clamping
        self._integral += error * self.dt
        self._integral = float(np.clip(
            self._integral, -self.integral_limit, self.integral_limit
        ))
        i = self.ki * self._integral
        # Derivative (backward difference)
        d = self.kd * (error - self._prev_error) / self.dt
        self._prev_error = error
        # Output clamp
        return float(np.clip(p + i + d, self.out_min, self.out_max))


class PIDController:
    """
    Multi-loop PID controller for ``C2GFastEnv``.

    Implements the SB3 ``predict(obs)`` interface.

    Parameters
    ----------
    T_setpoint_norm : float
        Target temperature (normalised by T_safe).  Default 30/35 ≈ 0.857.
    soc_target : float
        Target SOC for the throttle-bias loop.  Default 0.50.
    dt : float
        Environment timestep in seconds.  Default 5.0.
    bess_gain : float
        Proportional gain for BESS response to regd signal.  Default 1.0.
    pump_gains, hvac_gains, throttle_gains : tuple[float, float, float]
        (Kp, Ki, Kd) for each PID loop.
    """

    def __init__(
        self,
        T_setpoint_norm: float = 30.0 / _T_SAFE,
        soc_target: float = 0.50,
        dt: float = 5.0,
        bess_gain: float = 6.0,
        pump_gains: tuple[float, float, float] = (1.5, 0.02, 0.2),
        hvac_gains: tuple[float, float, float] = (1.5, 0.02, 0.2),
        flex_gain: float = 0.12,
        cool_gain: float = 0.10,
    ) -> None:
        self.T_sp = T_setpoint_norm
        self.soc_target = soc_target
        self.bess_gain = bess_gain
        self.flex_gain = flex_gain
        self.cool_gain = cool_gain

        # Loop 1: thermal error → pump speed (range [0.7, 1])
        self._pid_pump = _PIDLoop(*pump_gains, dt=dt, out_min=0.7, out_max=1.0)
        # Loop 2: thermal error → HVAC effort (range [0.7, 1])
        self._pid_hvac = _PIDLoop(*hvac_gains, dt=dt, out_min=0.7, out_max=1.0)

    def reset(self) -> None:
        """Reset all integrators (call between episodes)."""
        self._pid_pump.reset()
        self._pid_hvac.reset()

    def predict(
        self,
        obs: NDArray[np.float32],
        state=None,
        episode_start=None,
        deterministic: bool = True,
    ) -> tuple[NDArray[np.float32], None]:
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]
        actions = np.array([self._action_for(o) for o in obs],
                           dtype=np.float32)
        return (actions[0] if single else actions), None

    def _action_for(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        temp_A_n = float(obs[_F.TEMP_A])
        temp_B_n = float(obs[_F.TEMP_B])
        soc      = float(obs[_F.SOC])
        regd     = float(obs[_F.REGD])
        committed = float(obs[_F.COMMITTED]) if len(obs) > _F.COMMITTED else 0.1

        # ── BESS: proportional response scaled by commitment ─────────
        ideal_bess = self.bess_gain * committed * regd
        bess_dispatch = float(np.clip(ideal_bess, -1.0, 1.0))

        # SOC guards: disable dispatch near limits
        if soc < 0.12 and bess_dispatch > 0:
            bess_dispatch = 0.0
        if soc > 0.93 and bess_dispatch < 0:
            bess_dispatch = 0.0

        # ── Residual: unmet demand after BESS clipping ────────────────
        residual = ideal_bess - bess_dispatch

        # ── Loop 1: Pump controls Zone A temperature ─────────────────
        pump_error = temp_A_n - self.T_sp
        pump_speed = self._pid_pump.step(pump_error)

        # ── Loop 2: HVAC controls Zone B temperature ─────────────────
        hvac_error = temp_B_n - self.T_sp
        hvac_effort = self._pid_hvac.step(hvac_error)

        # ── Throttle + cooling assist when BESS saturates ─────────────
        if residual > 0.05:
            # BESS discharge saturated — shed load via throttle
            throttle = max(0.0, 1.0 - residual * self.flex_gain)
            # Reduce cooling to free electrical headroom
            cool_adj = min(0.3, residual * self.cool_gain)
            pump_speed = max(0.3, pump_speed - cool_adj)
            hvac_effort = max(0.3, hvac_effort - cool_adj)
        elif residual < -0.05:
            # BESS charge saturated — increase cooling to absorb power
            throttle = 1.0
            cool_adj = min(0.3, abs(residual) * self.cool_gain)
            pump_speed = min(1.0, pump_speed + cool_adj)
            hvac_effort = min(1.0, hvac_effort + cool_adj)
        else:
            throttle = 1.0

        return np.array([throttle, pump_speed, hvac_effort, bess_dispatch],
                        dtype=np.float32)
