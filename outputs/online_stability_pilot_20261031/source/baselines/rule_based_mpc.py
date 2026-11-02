"""
baselines/rule_based_mpc.py  —  Rule-Based Controller (Classical Baseline)
===========================================================================
A deterministic, threshold-based controller that serves as the classical
MPC baseline for the C2G-Bench NeurIPS evaluation.

The controller implements simple, interpretable rules derived from the
physical constraints of the system.  It requires no training and runs in
O(1) per step.  It is the lower bound that all RL agents must beat.

Policy logic (evaluated in priority order)
------------------------------------------
1. **Thermal protection** (highest priority)
   - T_max >= T_critical (0.98): max cooling (pump=1, hvac=1), throttle=0
   - T_max >= T_warn (33/35 ≈ 0.943): ramp cooling/pump up, throttle down proportionally

2. **Grid regulation via BESS**
   - bess_gain = committed_mw_max / bess_p_max_mw  (scenario-parameterised)
   - raw_demand = bess_gain * committed_norm * regd_signal
   - bess_dispatch = clip(raw_demand, -1, 1)
   - SOC < 0.15: ramp down discharge;  SOC > 0.80: ramp down charge

3. **Multi-lever residual tracking assist**
   - residual = raw_demand - bess_dispatch
   - residual > 0.05 (discharge deficit, no spike): shed throttle, reduce pump/hvac
   - residual < -0.05 (charge deficit): increase pump/hvac

4. **Defaults** (when no rule fires)
   - throttle_batch = 1.0  (run all batch at full speed)
   - pump_speed_A   = 0.7
   - hvac_effort    = 0.7  (moderate cooling, energy-efficient)
   - bess_dispatch  = 0.0  (idle)

5. **Opportunistic BESS charge**
   - |regd_signal| < 0.10 and SOC < 0.40: bess_dispatch = -0.3

Observation index map (C2GFastEnv, 12-D)
-----------------------------------------
  [0]  temp_A_norm     [1]  temp_B_norm     [2]  bess_soc
  [3]  p_base_norm     [4]  p_flex_nom_norm [5]  p_facility_norm
  [6]  regd_signal     [7]  lmp_norm        [8]  grid_load_norm
  [9]  is_spike        [10] prev_throttle   [11] pue_norm

Usage
-----
  from baselines.rule_based_mpc import RuleBasedController
  ctrl = RuleBasedController()
  obs, _ = env.reset(seed=0)
  while True:
      action, _ = ctrl.predict(obs)
      obs, rew, term, trunc, _ = env.step(action)
      if term or trunc: break
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from c2g_env.thermal_limits import T_SAFE as _T_SAFE, T_WARN as _T_WARN


# Thresholds (kept in sync with config.yaml reward section)
_T_WARN_NORM   = _T_WARN / _T_SAFE   # T_warn / T_safe  (obs[0] and [1])
_T_SAFE_NORM   = 1.0            # T_safe / T_safe
_T_CRITICAL    = 0.98           # 0.5°C below T_safe (obs normalised)
_SOC_MIN_GUARD = 0.15           # SOC below which we protect BESS
_SOC_CHARGE    = 0.80           # SOC above which we prefer not to charge further
_REGD_THRESH   = 0.10           # minimum regd_signal magnitude to act on
_DEFAULT_HVAC  = 0.7            # nominal HVAC (matches MacroEnv default)

from c2g_env.obs_indices import Fast as _F


class RuleBasedController:
    """
    Deterministic rule-based controller for ``C2GFastEnv``.

    Implements the ``predict(obs)`` interface used by Stable-Baselines3
    so it can be plugged directly into the benchmark runner.

    Parameters
    ----------
    t_warn_norm : float
        Normalised temperature warning threshold (default: 33/35).
    t_critical_norm : float
        Normalised temperature just below T_safe (default: 0.98).
    committed_mw_max : float
        Maximum regulation capacity for the scenario (MW). Used to derive
        bess_gain = committed_mw_max / bess_p_max_mw.
    bess_p_max_mw : float
        BESS peak power rating for the scenario (MW).
    """

    def __init__(
        self,
        t_warn_norm: float     = _T_WARN_NORM,
        t_critical_norm: float = _T_CRITICAL,
        committed_mw_max: float = 30.0,
        bess_p_max_mw: float    = 5.0,
    ) -> None:
        self.t_warn_norm     = t_warn_norm
        self.t_critical_norm = t_critical_norm
        self.bess_gain       = committed_mw_max / bess_p_max_mw

    def predict(
        self,
        obs: NDArray[np.float32],
        state=None,
        episode_start=None,
        deterministic: bool = True,
    ) -> tuple[NDArray[np.float32], None]:
        """
        Compute action from observation.

        Matches the SB3 ``BasePolicy.predict`` interface so the controller
        can be used interchangeably with trained SB3 agents.

        Parameters
        ----------
        obs : ndarray of shape (13,) or (N, 13)

        Returns
        -------
        action : ndarray of shape (4,) or (N, 4)
        state  : None  (stateless controller)
        """
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]
        actions = np.array([self._action_for(o) for o in obs], dtype=np.float32)
        return (actions[0] if single else actions), None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _action_for(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        temp_A_n = float(obs[_F.TEMP_A])
        temp_B_n = float(obs[_F.TEMP_B])
        soc      = float(obs[_F.SOC])
        regd     = float(obs[_F.REGD])
        is_spike = float(obs[_F.IS_SPIKE]) > 0.5
        committed = float(obs[_F.COMMITTED]) if len(obs) > _F.COMMITTED else 0.1

        # ── Defaults ──────────────────────────────────────────────────
        throttle      = 1.0
        pump_speed    = 0.7       # moderate — keeps thermal reserve
        hvac_effort   = _DEFAULT_HVAC
        bess_dispatch = 0.0

        # ── Rule 1: Thermal protection ────────────────────────────────
        max_temp_n = max(temp_A_n, temp_B_n)
        if max_temp_n >= self.t_critical_norm:
            # Emergency: max cooling, shed all batch, max pump
            hvac_effort = 1.0
            pump_speed  = 1.0
            throttle    = 0.0
        elif max_temp_n >= self.t_warn_norm:
            # Warning: ramp up cooling and pump, reduce batch proportionally
            excess = (max_temp_n - self.t_warn_norm) / (self.t_critical_norm - self.t_warn_norm)
            hvac_effort = min(1.0, _DEFAULT_HVAC + excess)
            pump_speed  = min(1.0, 0.7 + excess)
            throttle    = max(0.0, 1.0 - excess)

        # ── Rule 2: Grid regulation via BESS ──────────────────────────
        raw_demand = self.bess_gain * committed * regd
        if abs(regd) >= _REGD_THRESH:
            bess_dispatch = float(np.clip(raw_demand, -1.0, 1.0))

            # Protect BESS near min SOC: reduce discharge command
            if soc < _SOC_MIN_GUARD and bess_dispatch > 0:
                scale = max(0.0, (soc - 0.10) / (_SOC_MIN_GUARD - 0.10))
                bess_dispatch *= scale

            # Protect BESS near max SOC: reduce charge command
            if soc > _SOC_CHARGE and bess_dispatch < 0:
                scale = max(0.0, (0.95 - soc) / (0.95 - _SOC_CHARGE))
                bess_dispatch *= scale

        # ── Rule 2b: Multi-lever assist when BESS saturates ──────────
        residual = raw_demand - bess_dispatch
        if residual > 0.05 and not is_spike:
            # Discharge deficit — shed throttle, reduce cooling
            throttle = max(0.0, min(throttle, 1.0 - residual * 0.12))
            cool_adj = min(0.3, residual * 0.10)
            pump_speed = max(0.3, pump_speed - cool_adj)
            hvac_effort = max(0.3, hvac_effort - cool_adj)
        elif residual < -0.05:
            # Charge deficit — increase cooling to absorb power
            cool_adj = min(0.3, abs(residual) * 0.10)
            pump_speed = min(1.0, pump_speed + cool_adj)
            hvac_effort = min(1.0, hvac_effort + cool_adj)

        # ── Rule 4: Opportunistic BESS charge when price is cheap ─────
        # (grid_load_norm < 0.4 → off-peak, regd signal quiet)
        if abs(regd) < _REGD_THRESH and soc < 0.40:
            bess_dispatch = -0.3    # gentle charge at 15 MW

        return np.array([throttle, pump_speed, hvac_effort, bess_dispatch], dtype=np.float32)
