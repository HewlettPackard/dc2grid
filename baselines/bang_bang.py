"""
baselines/bang_bang.py  —  Bang-Bang / Hysteresis Controller
============================================================
The simplest reasonable baseline: binary switching with hysteresis bands.

Policy logic
------------
1. **BESS dispatch** = sign(regd_signal) × 0.4 (moderate bang-bang)
   - Binary switching but at ±20 MW instead of ±50 MW to reduce overshoot.
   - SOC guard: disable discharge below 12%, disable charge above 93%.

2. **Throttle** (batch load shedding):
   - 0.7 when regd > 0.05 (grid wants reduced draw → shed batch)
   - 1.0 otherwise (max throughput)

3. **CDU pump** (Zone A):
   - ON  (1.0) when  temp_A_norm > 31/35 ≈ 0.886
   - OFF (0.3) when  temp_A_norm < 29/35 ≈ 0.829
   - Hysteresis holds previous state between thresholds.

4. **HVAC** (Zone B):
   - ON  (1.0) when  temp_B_norm > 31/35 ≈ 0.886
   - OFF (0.0) when  temp_B_norm < 29/35 ≈ 0.829
   - Same hysteresis logic.

The hysteresis bands prevent rapid on/off chattering and represent a
plausible "junior engineer's first controller" — the absolute floor
baseline for the benchmark.

Observation index map (C2GFastEnv, 17-D)
-----------------------------------------
  [0]  temp_A_norm     [1]  temp_B_norm     [2]  bess_soc
  [6]  regd_signal

Usage
-----
  from baselines.bang_bang import BangBangController
  ctrl = BangBangController()
  obs, _ = env.reset(seed=0)
  action, _ = ctrl.predict(obs)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from c2g_env.obs_indices import Fast as _F
from c2g_env.thermal_limits import T_SAFE as _T_SAFE

# Hysteresis thresholds (normalised: T / T_safe)
_PUMP_ON  = 31.0 / _T_SAFE   # ≈ 0.886
_PUMP_OFF = 29.0 / _T_SAFE   # ≈ 0.829
_HVAC_ON  = 31.0 / _T_SAFE
_HVAC_OFF = 29.0 / _T_SAFE

# SOC guard bands
_SOC_DISCHARGE_MIN = 0.12
_SOC_CHARGE_MAX    = 0.93


class BangBangController:
    """
    Hysteresis bang-bang controller for ``C2GFastEnv``.

    Implements the SB3 ``predict(obs)`` interface.
    """

    def __init__(self) -> None:
        self._pump_on: bool = False
        self._hvac_on: bool = False

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

        # ── Pump: hysteresis on Zone A temperature ────────────────────
        if temp_A_n >= _PUMP_ON:
            self._pump_on = True
        elif temp_A_n <= _PUMP_OFF:
            self._pump_on = False
        pump_speed = 1.0 if self._pump_on else 0.7

        # ── HVAC: hysteresis on Zone B temperature ────────────────────
        if temp_B_n >= _HVAC_ON:
            self._hvac_on = True
        elif temp_B_n <= _HVAC_OFF:
            self._hvac_on = False
        hvac_effort = 1.0 if self._hvac_on else 0.7

        # ── BESS: bang-bang at full committed magnitude ────────────────
        bess_mag = max(committed * 6.0, 0.05)
        if abs(regd) < 0.05:
            bess_dispatch = 0.0
        elif regd > 0:
            bess_dispatch = min(bess_mag, 1.0) if soc > _SOC_DISCHARGE_MIN else 0.0
        else:
            bess_dispatch = -min(bess_mag, 1.0) if soc < _SOC_CHARGE_MAX else 0.0

        # ── Residual: tracking demand minus BESS delivery ─────────────
        demand = committed * 6.0 * regd
        residual = demand - bess_dispatch

        # ── Throttle + cooling assist when BESS saturates ─────────────
        throttle = 1.0
        if residual > 0.05:
            # Discharge deficit — shed load, reduce cooling
            throttle = max(0.0, 1.0 - residual * 0.12)
            cool_adj = min(0.2, residual * 0.10)
            pump_speed = max(0.3, pump_speed - cool_adj)
            hvac_effort = max(0.3, hvac_effort - cool_adj)
        elif residual < -0.05:
            # Charge deficit — increase cooling to absorb power
            cool_adj = min(0.2, abs(residual) * 0.10)
            pump_speed = min(1.0, pump_speed + cool_adj)
            hvac_effort = min(1.0, hvac_effort + cool_adj)

        return np.array([throttle, pump_speed, hvac_effort, bess_dispatch],
                        dtype=np.float32)
