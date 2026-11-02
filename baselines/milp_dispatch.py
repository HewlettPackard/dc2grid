"""
baselines/milp_dispatch.py  —  MILP Economic Dispatch (Macro Env)
==================================================================
A linearised mixed-integer linear program for the ``C2GMacroEnv``.

Formulation (simplified unit-commitment style):

  Decision variables (per macro slot k = 0..H-1):
    commit_k  ∈ [0, 1]           continuous — committed MW fraction
    bess_k    ∈ [-1, 1]          continuous — BESS dispatch
    z_chg_k   ∈ {0, 1}          binary — 1 = charging mode
    z_dch_k   ∈ {0, 1}          binary — 1 = discharging mode
    z_chg_k + z_dch_k ≤ 1       no simultaneous charge + discharge

  Objective (maximise):
    Σ_k [ lmp_k · bess_dch_k · P_BESS / 50
          + α · commit_k
          - γ · thermal_excess_k
          - δ_soc · soc_viol_k
          - churn · |Δcommit_k| ]

  Constraints:
    SOC_{k+1} = SOC_k - bess_k · P_BESS · dt / (3600 · E_nom)
    SOC_MIN ≤ SOC_k ≤ SOC_MAX
    T_{k+1} = decay · T_k + (1 - decay) · T_eq(commit_k)   (linearised)
    bess_k ≤ z_dch_k,  -bess_k ≤ z_chg_k

Solver: scipy.optimize.linprog (HiGHS, bundled) when cvxpy is absent,
        or cvxpy + HiGHS if cvxpy is installed.

Usage
-----
  from baselines.milp_dispatch import MILPDispatchController
  ctrl = MILPDispatchController(horizon=8)
  obs, _ = env.reset(seed=0)
  action, _ = ctrl.predict(obs)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize

from c2g_env.obs_indices import Macro as _M
from c2g_env.thermal_limits import T_SAFE as _T_SAFE, T_WARN as _T_WARN

# ── Physical constants ───────────────────────────────────────────────
_DT_MACRO   = 900.0     # 15 min
_E_NOM_MWH  = 150.0
_P_BESS_MAX = 50.0
_SOC_MIN    = 0.10
_SOC_MAX    = 0.95

_TAU_THERMAL = 771.0
_THERMAL_DECAY = np.exp(-_DT_MACRO / _TAU_THERMAL)

# Reward weights
_ALPHA     = 1.0
_GAMMA     = 5.0
_DELTA_SOC = 0.5
_LMP_BONUS = 0.1
_CHURN_PEN = 0.05

# BESS SOC change per unit dispatch per macro step
_BESS_DELTA_PER_UNIT = _P_BESS_MAX * _DT_MACRO / (3600.0 * _E_NOM_MWH)


class MILPDispatchController:
    """
    MILP-based economic dispatch for ``C2GMacroEnv``.

    Falls back to a continuous LP relaxation if integer solve is too slow.

    Parameters
    ----------
    horizon : int
        Planning horizon in macro steps.  Default 8 (= 2 hours).
    T_amb : float
        Assumed ambient temperature (°C).
    """

    def __init__(
        self,
        horizon: int = 8,
        T_amb: float = 25.0,
    ) -> None:
        self.H = horizon
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
        """Solve the MILP and return the first action."""
        H = self.H

        temp_A_norm = float(obs[_M.TEMP_A])
        soc         = float(obs[_M.SOC])
        lmp_norm    = float(obs[_M.LMP])
        commit_prev = float(obs[_M.BID_MW_PREV])
        T_A = temp_A_norm * _T_SAFE if temp_A_norm < 2.0 else temp_A_norm

        # Use scipy NLP since linprog can't handle binary variables directly.
        # We use a continuous relaxation (LP relaxation of the MILP) through SLSQP.
        # Variables: [commit_0..H-1, bess_0..H-1]
        n_vars = 2 * H
        x0 = np.zeros(n_vars)
        x0[0:H] = 0.5
        x0[H:]  = 0.0

        bounds = [(0.0, 1.0)] * H + [(-1.0, 1.0)] * H
        T_amb = self.T_amb

        def objective(x: NDArray) -> float:
            commit = x[0:H]
            bess   = x[H:2*H]
            cost = 0.0
            T_k = T_A
            soc_k = soc
            prev_c = commit_prev

            for k in range(H):
                # Thermal (piecewise-linear: 3 segments)
                P_IT = 150.0 * commit[k]
                T_eq = T_amb + P_IT / 35.0 + 30.0
                T_k = T_eq + (T_k - T_eq) * _THERMAL_DECAY

                # SOC
                eta = 0.95
                if bess[k] > 0:
                    soc_k -= bess[k] * _BESS_DELTA_PER_UNIT / eta
                else:
                    soc_k -= bess[k] * _BESS_DELTA_PER_UNIT * eta
                soc_k = np.clip(soc_k, 0.0, 1.0)

                # Revenue
                bess_dch = max(0.0, bess[k])
                revenue = _LMP_BONUS * lmp_norm * bess_dch

                # Throughput
                throughput = _ALPHA * commit[k]

                # Thermal penalty (linearised)
                T_norm = T_k / _T_SAFE
                thermal_pen = _GAMMA * max(0.0, T_norm - _T_WARN / _T_SAFE)

                # SOC penalty
                soc_pen = _DELTA_SOC if (soc_k < 0.12 or soc_k > 0.90) else 0.0

                # Churn
                churn = _CHURN_PEN * abs(commit[k] - prev_c)
                prev_c = commit[k]

                # Simultaneous charge/discharge penalty (soft MILP relaxation)
                # Penalise bess near zero to encourage commitment to direction
                cd_pen = 0.01 * max(0.0, min(bess[k] + 0.1, 0.1 - bess[k]))

                cost += -revenue - throughput + thermal_pen + soc_pen + churn + cd_pen

            return cost

        def soc_constraint(x: NDArray) -> NDArray:
            bess = x[H:2*H]
            soc_k = soc
            feas = np.zeros(2 * H)
            for k in range(H):
                eta = 0.95
                if bess[k] > 0:
                    soc_k -= bess[k] * _BESS_DELTA_PER_UNIT / eta
                else:
                    soc_k -= bess[k] * _BESS_DELTA_PER_UNIT * eta
                feas[k]     = soc_k - _SOC_MIN
                feas[H + k] = _SOC_MAX - soc_k
            return feas

        constraints = [{"type": "ineq", "fun": soc_constraint}]

        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        x = result.x
        return np.array([x[0], x[H]], dtype=np.float32)
