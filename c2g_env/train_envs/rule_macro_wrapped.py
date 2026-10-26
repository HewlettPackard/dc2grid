"""
c2g_env/train_envs/rule_macro_wrapped.py
========================================
Wraps ``C2GFastEnv`` so that a rule-based macro controller periodically
updates ``committed_mw`` via market handshake, giving the low-level PPO
agent a realistic dynamic regulation signal during training.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import gymnasium as gym

from c2g_env import C2GFastEnv
from c2g_env.env_high_level import C2GMacroEnv
from baselines.rule_based_macro import RuleBasedMacroController

# Number of FastEnv steps per macro decision (180 × 5 s = 900 s = 15 min)
_SUBSTEPS = 180

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class RuleMacroWrappedEnv(gym.Wrapper):
    """
    Wraps ``C2GFastEnv`` so that a rule-based macro controller
    periodically updates ``committed_mw`` via market handshake.

    The low-level action/observation spaces are unchanged (4-D / 18-D).
    Every ``_SUBSTEPS`` steps the wrapper:
      1. Aggregates recent sub-observations into a 19-D macro obs
      2. Calls ``RuleBasedMacroController.predict()``
      3. Runs the 3-phase market handshake (RMCP → bid → clear)
      4. Sets ``env.committed_mw`` to the cleared value

    Reuses ``C2GMacroEnv.aggregate_macro_obs``, ``C2GMacroEnv.run_handshake``,
    and ``C2GMacroEnv.macro_obs_from_reset`` so that observation construction
    and market logic are defined in exactly one place.
    """

    def __init__(self, env: C2GFastEnv) -> None:
        super().__init__(env)

        self._macro_ctrl = RuleBasedMacroController()

        # Load handshake config from the same config file the env uses
        cfg_path = _CONFIG_PATH
        with open(cfg_path, encoding="utf-8") as fh:
            full_cfg = yaml.safe_load(fh)
        hs_cfg = full_cfg.get("handshake", {})
        self._rmcp_max = float(hs_cfg.get("rmcp_max", 100.0))

        scfg = env._scfg
        self._committed_max_mw = env._committed_mw_max
        self._dr_baseline_mw = env._dr_baseline_mw
        self._dr_rate_usd_mw = float(scfg.get("dr_rate_usd_mw", 5.0))

        # Tracking state
        self._sub_tick = 0
        self._obs_buffer: list[np.ndarray] = []
        self._info_buffer: list[dict] = []
        self._prev_bid_mw_norm = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm = 0.0
        self._last_reg_need_norm = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._committed_max_mw = self.env._committed_mw_max
        self._dr_baseline_mw = self.env._dr_baseline_mw
        self._sub_tick = 0
        self._obs_buffer = []
        self._info_buffer = []
        self._prev_bid_mw_norm = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm = 0.0
        self._last_reg_need_norm = 0.0

        # Perform initial macro decision to set committed_mw from rule-based bid
        self._do_macro_handshake()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # At the start of each new macro window (after the first), update commitment
        if self._sub_tick > 0 and self._sub_tick % _SUBSTEPS == 0:
            self._do_macro_handshake()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_buffer.append(obs)
        self._info_buffer.append(info)
        self._sub_tick += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Market handshake
    # ------------------------------------------------------------------

    def _do_macro_handshake(self) -> None:
        """Run rule-based macro bid + grid clearing to update committed_mw."""
        macro_obs = self._build_macro_obs()
        bid_action, _ = self._macro_ctrl.predict(macro_obs)

        bid_mw_norm = float(bid_action[0])
        bid_price_norm = float(bid_action[1])

        hs = C2GMacroEnv.run_handshake(
            self.env._grid,  # type: ignore[attr-defined]
            bid_mw_norm=bid_mw_norm,
            bid_price_norm=bid_price_norm,
            committed_max_mw=self._committed_max_mw,
            rmcp_max=self._rmcp_max,
            dr_baseline_mw=self._dr_baseline_mw,
            dr_rate_usd_mw=self._dr_rate_usd_mw,
        )

        self.env.committed_mw = hs["committed_mw"]  # type: ignore[attr-defined]
        self._prev_bid_mw_norm = bid_mw_norm
        self._prev_bid_price_norm = bid_price_norm
        self._last_rmcp_norm = hs["rmcp_norm"]
        self._last_reg_need_norm = hs["reg_need_norm"]

        # Reset buffer for next macro window
        self._obs_buffer = []
        self._info_buffer = []

    def _build_macro_obs(self) -> np.ndarray:
        """
        Build a 19-D macro observation from buffered sub-step data.

        Delegates to ``C2GMacroEnv.aggregate_macro_obs`` (with buffer) or
        ``C2GMacroEnv.macro_obs_from_reset`` (at reset, no buffer).
        """
        if not self._obs_buffer:
            # At reset — build from the env's current state
            inner_obs = np.zeros(18, dtype=np.float32)
            inner_obs[7] = 0.5  # grid_load_norm (triggers mid-range bid)
            inner_obs[15] = 1.0  # v_pcc_pu (nominal)
            T_safe = self.env._thermal.T_safe  # type: ignore[attr-defined]
            return C2GMacroEnv.macro_obs_from_reset(
                inner_obs,
                T_safe=T_safe,
                temp_A=self.env._thermal.temp_A,  # type: ignore[attr-defined]
                temp_B=self.env._thermal.temp_B,  # type: ignore[attr-defined]
                prev_bid_mw_norm=self._prev_bid_mw_norm,
                prev_bid_price_norm=self._prev_bid_price_norm,
            )

        # Inject p_flex_max_kw so aggregate_macro_obs can compute backlog_norms
        p_flex_max_kw = self.env._workload.p_flex_max_kw  # type: ignore[attr-defined]
        for info in self._info_buffer:
            info["p_flex_max_kw"] = p_flex_max_kw

        committed_mw = self.env.committed_mw  # type: ignore[attr-defined]
        return C2GMacroEnv.aggregate_macro_obs(
            self._obs_buffer, self._info_buffer,
            T_safe=self.env._thermal.T_safe,  # type: ignore[attr-defined]
            committed_mw=committed_mw,
            tracking_floor_kw=self.env._macro_tracking_floor_kw,
            bid_mw_norm=self._prev_bid_mw_norm,
            bid_price_norm=self._prev_bid_price_norm,
            rmcp_norm=self._last_rmcp_norm,
            reg_need_norm=self._last_reg_need_norm,
        )
