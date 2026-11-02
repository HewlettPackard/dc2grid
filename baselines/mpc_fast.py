"""
baselines/mpc_fast.py  —  Model Predictive Control (Fast Env, Short Horizon)
=============================================================================
Rolling-horizon nonlinear MPC that uses a simplified prediction model
derived from the known thermal ODE and BESS dynamics.

At each step the controller solves:

    minimize  Σ_{k=0..H-1} [ β·|regd - bess_k| + γ·max(0, T_k - T_warn)
                              + δ_soc·1_{soc_k ∉ [0.12,0.90]}
                              - α·throttle_k ]

    subject to
        T_{k+1} = T_eq + (T_k - T_eq)·exp(-a·dt)        (thermal ODE)
        SOC_{k+1} = SOC_k - bess_k·P_max·dt / E_nom      (linear BESS)
        0 ≤ throttle_k ≤ 1,  0.15 ≤ pump_k ≤ 1,  0 ≤ hvac_k ≤ 1
        -1 ≤ bess_k ≤ 1,  0.10 ≤ SOC_k ≤ 0.95

The prediction model is a first-order approximation of the full physics.
It uses scipy.optimize.minimize (SLSQP) — no additional dependencies.

Re-solve frequency: every ``replan_every`` steps (default 1 = every step).
Only the first action of the plan is applied (receding horizon).

Parameters are configurable via constructor kwargs or conf/algo/mpc_fast.yaml.

Usage
-----
  from baselines.mpc_fast import MPCFastController
  ctrl = MPCFastController(horizon=24)
  obs, _ = env.reset(seed=0)
  action, _ = ctrl.predict(obs)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from c2g_env.obs_indices import Fast as _F
from c2g_env.thermal_limits import T_SAFE as _T_SAFE, T_WARN as _T_WARN

# ── Physical constants (from c2g_env/config.yaml & physics/) ──────────
_T_WARN_NORM = _T_WARN / _T_SAFE
_DT          = 5.0        # seconds per tick
_E_NOM_MWH   = 150.0
_P_BESS_MAX  = 50.0       # MW
_SOC_MIN     = 0.10
_SOC_MAX     = 0.95

# Thermal model linearisation (Zone A dominates; simplified single-zone)
# From thermal.py: C_A=27000 MJ/°C, K_liq=35 MW/°C at full pump
# τ = C / K ≈ 771 s ≈ 12.9 min;  a = K/C per second
_C_THERMAL = 27_000.0   # MJ/°C
_K_THERMAL = 35.0        # MW/°C (at pump_speed=1.0)
_K_ENV     = 0.5         # MW/°C envelope coupling

# Reward weights (from config.yaml)
_ALPHA = 1.0
_BETA  = 2.0
_GAMMA = 5.0
_DELTA_SOC = 0.5


class MPCFastController:
    """
    Short-horizon MPC for ``C2GFastEnv``.

    Parameters
    ----------
    horizon : int
        Prediction horizon in steps.  Default 24 (= 2 minutes).
    replan_every : int
        Re-solve every N steps.  Between re-solves, replay planned actions.
    max_iter : int
        Maximum SLSQP iterations per solve.
    T_amb : float
        Assumed ambient temperature for prediction model (°C).
    """

    def __init__(
        self,
        horizon: int = 24,
        replan_every: int = 1,
        max_iter: int = 50,
        T_amb: float = 25.0,
    ) -> None:
        self.H = horizon
        self.replan_every = replan_every
        self.max_iter = max_iter
        self.T_amb = T_amb

        # Action plan cache
        self._plan: list[NDArray[np.float32]] = []
        self._step_since_plan = 0

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
        # Use cached plan if available
        if self._plan and self._step_since_plan < self.replan_every:
            idx = min(self._step_since_plan, len(self._plan) - 1)
            self._step_since_plan += 1
            return self._plan[idx]

        # ── Extract current state ────────────────────────────────────
        T_A_norm = float(obs[_F.TEMP_A])
        soc      = float(obs[_F.SOC])
        regd     = float(obs[_F.REGD])

        T_A = T_A_norm * _T_SAFE  # de-normalise to °C

        # ── Build & solve the NLP ────────────────────────────────────
        H = self.H
        # Decision variables: [throttle_0..H-1, pump_0..H-1, hvac_0..H-1, bess_0..H-1]
        n_vars = 4 * H

        # Initial guess: reasonable defaults
        x0 = np.zeros(n_vars)
        x0[0:H]     = 1.0    # throttle = 1
        x0[H:2*H]   = 0.7    # pump = 0.7
        x0[2*H:3*H] = 0.5    # hvac = 0.5
        x0[3*H:]     = np.clip(regd * 2.0, -1.0, 1.0)  # bess ~ proportional to regd

        # Variable bounds
        bounds = (
            [(0.0, 1.0)] * H       # throttle
            + [(0.15, 1.0)] * H    # pump
            + [(0.0, 1.0)] * H     # hvac
            + [(-1.0, 1.0)] * H    # bess
        )

        # Closure captures current state
        T_amb = self.T_amb

        def objective(x: NDArray) -> float:
            throttle = x[0:H]
            pump     = x[H:2*H]
            bess     = x[3*H:4*H]

            cost = 0.0
            T_k = T_A
            soc_k = soc

            for k in range(H):
                # Thermal prediction (Zone A, exact exponential)
                K_eff = _K_THERMAL * max(0.15, pump[k])
                a = (K_eff + _K_ENV) / _C_THERMAL  # 1/s
                # Assume IT load ≈ 150 MW (typical) scaled by throttle
                P_IT = 150.0 * throttle[k]
                T_supply = 30.0  # nominal supply temp
                b = (P_IT + K_eff * T_supply + _K_ENV * T_amb) / _C_THERMAL
                T_eq = b / a if a > 1e-12 else T_k
                T_k = T_eq + (T_k - T_eq) * np.exp(-a * _DT)

                # BESS SOC prediction (linear, ignoring efficiency detail)
                eta = 0.95
                if bess[k] > 0:
                    # Discharge
                    soc_k -= bess[k] * _P_BESS_MAX * _DT / (3600.0 * _E_NOM_MWH * eta)
                else:
                    # Charge
                    soc_k -= bess[k] * _P_BESS_MAX * _DT * eta / (3600.0 * _E_NOM_MWH)
                soc_k = np.clip(soc_k, _SOC_MIN, _SOC_MAX)

                # Cost terms (minimise = negative reward)
                # Tracking: |regd - bess_dispatch| (assume regd is constant over horizon)
                tracking_err = abs(regd - bess[k]) / 2.0  # normalise
                cost += _BETA * tracking_err

                # Thermal penalty
                T_k_norm = T_k / _T_SAFE
                if T_k_norm > _T_WARN_NORM:
                    cost += _GAMMA * (T_k_norm - _T_WARN_NORM)

                # SOC penalty
                if soc_k < 0.12 or soc_k > 0.90:
                    cost += _DELTA_SOC

                # Throughput (negative = we want to maximise)
                cost -= _ALPHA * throttle[k]

            return cost

        # SOC feasibility constraint
        def soc_constraint(x: NDArray) -> NDArray:
            bess = x[3*H:4*H]
            soc_k = soc
            feasibility = np.zeros(2 * H)
            for k in range(H):
                eta = 0.95
                if bess[k] > 0:
                    soc_k -= bess[k] * _P_BESS_MAX * _DT / (3600.0 * _E_NOM_MWH * eta)
                else:
                    soc_k -= bess[k] * _P_BESS_MAX * _DT * eta / (3600.0 * _E_NOM_MWH)
                feasibility[k]     = soc_k - _SOC_MIN       # soc >= SOC_MIN
                feasibility[H + k] = _SOC_MAX - soc_k       # soc <= SOC_MAX
            return feasibility

        constraints = [{"type": "ineq", "fun": soc_constraint}]

        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-6},
        )

        # Extract plan
        x = result.x
        self._plan = []
        for k in range(H):
            action = np.array([
                x[k],          # throttle
                x[H + k],     # pump
                x[2*H + k],   # hvac
                x[3*H + k],   # bess
            ], dtype=np.float32)
            self._plan.append(action)
        self._step_since_plan = 1  # we're about to use step 0

        return self._plan[0]
