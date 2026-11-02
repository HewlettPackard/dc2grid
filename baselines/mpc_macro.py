"""
baselines/mpc_macro.py  —  Model Predictive Control (Macro Env, Long Horizon)
==============================================================================
Rolling-horizon MPC for ``C2GMacroEnv`` (15-minute decision intervals).

At each macro step the controller solves:

    maximize  Σ_{k=0..H-1} [ lmp_k · bess_k · P_max / 50  (LMP revenue)
                              + α · commit_k                 (throughput)
                              - γ · max(0, T_k - T_warn)    (thermal)
                              - δ_soc · 1_{soc ∉ safe}      (SOC)
                              - churn · |Δcommit_k|  ]       (churn)

    subject to
        T_{k+1} = f(T_k, commit_k)       (linearised thermal)
        SOC_{k+1} = SOC_k + bess_k·η·dt  (linear BESS)
        0 ≤ commit_k ≤ 1,  -1 ≤ bess_k ≤ 1
        0.10 ≤ SOC_k ≤ 0.95

Uses persistence forecast: LMP and grid load assumed constant over horizon.

Usage
-----
  from baselines.mpc_macro import MPCMacroController
  ctrl = MPCMacroController(horizon=8)
  obs, _ = env.reset(seed=0)
  action, _ = ctrl.predict(obs)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from c2g_env.obs_indices import Macro as _M
from c2g_env.thermal_limits import T_SAFE as _T_SAFE, T_WARN as _T_WARN

# ── Physical constants ────────────────────────────────────────────────
_DT_MACRO   = 900.0     # 15 minutes in seconds
_E_NOM_MWH  = 150.0
_P_BESS_MAX = 50.0      # MW
_SOC_MIN    = 0.10
_SOC_MAX    = 0.95

# Simplified thermal model (Zone A only, 15-min timescale)
# τ ≈ 771 s → at 900 s step: exp(-900/771) ≈ 0.31
_TAU_THERMAL = 771.0  # seconds
_THERMAL_DECAY = np.exp(-_DT_MACRO / _TAU_THERMAL)

# Reward weights
_ALPHA  = 1.0
_GAMMA  = 5.0
_DELTA_SOC = 0.5
_LMP_BONUS = 0.1
_CHURN_PEN = 0.05


class MPCMacroController:
    """
    Long-horizon MPC for ``C2GMacroEnv``.

    Parameters
    ----------
    horizon : int
        Prediction horizon in macro steps.  Default 8 (= 2 hours).
    max_iter : int
        Maximum SLSQP iterations.
    T_amb : float
        Assumed ambient temperature (°C).
    """

    def __init__(
        self,
        horizon: int = 8,
        max_iter: int = 80,
        T_amb: float = 25.0,
    ) -> None:
        self.H = horizon
        self.max_iter = max_iter
        self.T_amb = T_amb

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
        H = self.H

        # ── Current state from obs ───────────────────────────────────
        temp_A_mean = float(obs[_M.TEMP_A])
        soc         = float(obs[_M.SOC])
        lmp_norm    = float(obs[_M.LMP])
        load_norm   = float(obs[_M.GRID_LOAD])
        commit_prev = float(obs[_M.BID_MW_PREV])

        # De-normalise temperature (obs is T/T_safe for headroom, or °C mean)
        # In macro env, obs[0] is temp_A_mean in °C normalised by T_safe
        T_A = temp_A_mean * _T_SAFE if temp_A_mean < 2.0 else temp_A_mean

        # Decision variables: [commit_0..H-1, bess_0..H-1]
        n_vars = 2 * H
        x0 = np.zeros(n_vars)
        x0[0:H] = 0.5         # commit = 50%
        x0[H:]  = 0.0         # bess = idle

        bounds = (
            [(0.0, 1.0)] * H       # commit
            + [(-1.0, 1.0)] * H    # bess
        )

        T_amb = self.T_amb

        def objective(x: NDArray) -> float:
            commit = x[0:H]
            bess   = x[H:2*H]

            cost = 0.0
            T_k = T_A
            soc_k = soc
            prev_c = commit_prev

            for k in range(H):
                # Thermal prediction (simplified: higher commitment → more heat)
                # Equilibrium T depends on committed power
                P_IT = 150.0 * commit[k]  # MW approx
                T_eq = T_amb + P_IT / 35.0 + 30.0  # rough equilibrium
                T_k = T_eq + (T_k - T_eq) * _THERMAL_DECAY

                # BESS SOC
                eta = 0.95
                delta_soc = bess[k] * _P_BESS_MAX * _DT_MACRO / (3600.0 * _E_NOM_MWH)
                if bess[k] > 0:
                    soc_k -= delta_soc / eta
                else:
                    soc_k -= delta_soc * eta
                soc_k = np.clip(soc_k, _SOC_MIN, _SOC_MAX)

                # LMP revenue (persistence forecast)
                revenue = _LMP_BONUS * lmp_norm * max(0.0, bess[k]) * _P_BESS_MAX / 50.0

                # Throughput
                throughput = _ALPHA * commit[k]

                # Thermal penalty
                T_norm = T_k / _T_SAFE
                thermal_pen = _GAMMA * max(0.0, T_norm - _T_WARN / _T_SAFE)

                # SOC penalty
                soc_pen = _DELTA_SOC if (soc_k < 0.12 or soc_k > 0.90) else 0.0

                # Churn penalty
                churn = _CHURN_PEN * abs(commit[k] - prev_c)
                prev_c = commit[k]

                # Total (we minimise, so negate revenue + throughput)
                cost += -revenue - throughput + thermal_pen + soc_pen + churn

            return cost

        def soc_constraint(x: NDArray) -> NDArray:
            bess = x[H:2*H]
            soc_k = soc
            feas = np.zeros(2 * H)
            for k in range(H):
                eta = 0.95
                delta = bess[k] * _P_BESS_MAX * _DT_MACRO / (3600.0 * _E_NOM_MWH)
                if bess[k] > 0:
                    soc_k -= delta / eta
                else:
                    soc_k -= delta * eta
                feas[k]     = soc_k - _SOC_MIN
                feas[H + k] = _SOC_MAX - soc_k
            return feas

        constraints = [{"type": "ineq", "fun": soc_constraint}]

        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-6},
        )

        x = result.x
        action = np.array([x[0], x[H]], dtype=np.float32)
        return action
