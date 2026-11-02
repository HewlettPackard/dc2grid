"""
baselines/rule_based_macro.py  —  Rule-Based Macro Controller
===============================================================
Deterministic macro-level controller for ``C2GMacroEnv``.

Implements simple commitment and dispatch rules that serve as the
classical baseline for the high-level decision problem.

Policy logic
------------
1. **Bid sizing** — proportional to grid load stress:
   - High load (grid_load > 0.7): bid 80% of capacity
   - Medium load (0.4–0.7):       bid 50% of capacity
   - Low load (< 0.4):            bid 20%

2. **Bid pricing** — undercut the grid's posted RMCP:
   - Bid at 80% of RMCP (low price → high acceptance probability)

3. **Safety overrides:**
   - If thermal headroom < 0.1: reduce bid to 30%
   - If under-frequency: increase bid by 20%
   - If low voltage: cap bid at 40%

Observation index map (C2GMacroEnv, 19-D)
-------------------------------------------
  [0]  temp_A_mean     [1]  temp_B_mean     [2]  bess_soc_end
  [3]  p_base_mean     [4]  p_facility_mean [5]  regd_mean
  [6]  lmp_mean        [7]  grid_load_mean  [8]  tracking_err_mean
  [9]  is_spike_any    [10] headroom_A      [11] headroom_B
  [12] bid_mw_prev     [13] bess_prev       [14] freq_dev_mean
  [15] v_pcc_mean      [16] backlog_norm    [17] rmcp_norm
  [18] reg_need_norm
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


from c2g_env.obs_indices import Macro as _M


class RuleBasedMacroController:
    """
    Deterministic rule-based controller for ``C2GMacroEnv``.

    Implements the ``predict(obs)`` interface compatible with SB3 so it
    can be used interchangeably with trained agents in the evaluation loop.
    """

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
        load_norm = float(obs[_M.GRID_LOAD])
        hroom_A   = float(obs[_M.HEADROOM_A])
        hroom_B   = float(obs[_M.HEADROOM_B])
        freq_dev  = float(obs[_M.FREQ_DEV])
        v_pcc     = float(obs[_M.VPCC])
        rmcp_norm = float(obs[_M.RMCP]) if len(obs) > _M.RMCP else 0.25

        # ── Bid sizing (proportional to grid load) ────────────────
        if load_norm > 0.7:
            bid_mw_norm = 0.80
        elif load_norm > 0.4:
            bid_mw_norm = 0.50
        else:
            bid_mw_norm = 0.20

        # ── Bid pricing (undercut RMCP for high acceptance) ───────
        # rmcp_norm = RMCP / rmcp_max, bid_price_norm maps to [0, 2*rmcp_max]
        # So bid_price_norm = 0.5 * rmcp_norm bids at RMCP;
        # we bid at 80% of RMCP for high acceptance
        bid_price_norm = float(np.clip(0.4 * rmcp_norm, 0.0, 1.0))

        # ── Safety overrides ─────────────────────────────────────
        # Thermal protection: reduce bid
        min_headroom = min(hroom_A, hroom_B)
        if min_headroom < 0.10:
            bid_mw_norm = min(bid_mw_norm, 0.30)

        # Frequency support: if under-frequency, increase bid
        if freq_dev < -0.3:
            bid_mw_norm = min(1.0, bid_mw_norm + 0.2)

        # Voltage support: if low voltage, reduce facility load bid
        if v_pcc < 0.96:
            bid_mw_norm = min(bid_mw_norm, 0.40)

        return np.array([bid_mw_norm, bid_price_norm], dtype=np.float32)
