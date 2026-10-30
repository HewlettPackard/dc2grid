"""
Step 2.2 — C2G-MacroEnv  (High-Level / Grid-Manager Environment)
=================================================================
A ``gymnasium.Env`` that wraps :class:`C2GFastEnv` and operates at a
coarser 15-minute (900 s) timescale — 180 five-second sub-steps per action.

This is the decision layer of the hierarchical control stack:

  MacroEnv action (every 15 min)
      → translated into 180 repeated FastEnv actions (every 5 s)
      → each FastEnv step advances all physics engines

The MacroEnv agent participates in a 3-phase market handshake each macro
step:
  1. Grid posts offer  — RMCP and regulation need
  2. DC bids           — agent offers MW capacity at a price
  3. Grid clears       — probabilistic acceptance (sigmoid × quantity)

If the bid is accepted the DC commits the accepted MW at the clearing
price.  If rejected it falls back to a standing DR baseline contract.

Action space  (Box, 2-D, continuous)
-------------------------------------
  [0] bid_mw_norm    ∈ [0, 1]   → MW to offer to the grid,
                                   mapped to [0, committed_max_mw].
  [1] bid_price_norm ∈ [0, 1]   → asking price,
                                   mapped to [0, 2 × rmcp_max].

MacroEnv holds all FastEnv levers at fixed "safe defaults"
unless overridden by ``inner_action_fn`` (optional callback for research):
  • throttle_batch = 1.0   (do not throttle batch by default)
  • hvac_effort    = 0.7   (moderate cooling)
  • bess_dispatch  = 0.0   (neutral)

Observation space  (Box, 19-D)
---------------------------------
Aggregated over the 180 sub-steps to give the grid-manager a stable view:

  [0]  temp_A_mean        Mean Zone A temperature / T_safe
  [1]  temp_B_mean        Mean Zone B temperature / T_safe
  [2]  bess_soc_end       SOC at end of the macro-step         ∈ [0, 1]
  [3]  p_base_mean        Mean p_base_norm
  [4]  p_facility_mean    Mean p_facility_norm
  [5]  regd_mean          Mean |regd_signal|                   ∈ [0, 1]
  [6]  lmp_mean           Mean lmp_norm                        ∈ [0, 1]
  [7]  grid_load_mean     Mean load_norm                       ∈ [0, 1]
  [8]  tracking_err_mean  Mean |ΔP_demanded − ΔP_actual| / norm
  [9]  is_spike_any       1.0 if any sub-step had a GenAI spike
  [10] thermal_headroom_A (T_safe − T_A_max) / T_safe          ∈ [0, 1]
  [11] thermal_headroom_B (T_safe − T_B_max) / T_safe          ∈ [0, 1]
  [12] bid_mw_prev_norm   Previous macro-action [0] bid MW
  [13] bid_price_prev_norm Previous macro-action [1] bid price
  [14] freq_dev_mean      Mean normalised frequency deviation   ∈ [-1, 1]
  [15] v_pcc_mean         Mean PCC voltage (per-unit)           ∈ [0, 1.1]
  [16] backlog_norm_mean  Mean batch queue depth / p_flex_max   ∈ [0, 2]
  [17] rmcp_norm          Grid's posted RMCP / rmcp_max         ∈ [0, 5]
  [18] reg_need_norm      Grid's residual need / committed_max  ∈ [0, 5]

Reward
------
  R_macro = λ_rev  × regulation_revenue / 1000
           + mean_sub_reward
           − λ_elec × electricity_cost / 1000
           − λ_churn × |bid_mw_now − bid_mw_prev|

Usage
-----
  from c2g_env import C2GMacroEnv
  env  = C2GMacroEnv(scenario="scenario_b")
  obs, info = env.reset(seed=0)
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from c2g_env.env_low_level import C2GFastEnv

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Number of FastEnv steps per MacroEnv step (180 × 5 s = 900 s = 15 min)
_SUBSTEPS = 180

# Safe-default inner levers when MacroEnv has full control
_DEFAULT_THROTTLE = 1.0   # do not shed batch
_DEFAULT_PUMP     = 0.7   # moderate pump speed (thermal inertia reserve)
_DEFAULT_HVAC     = 0.7   # moderate cooling
_DEFAULT_BESS     = 0.0   # neutral BESS (low-level agent controls dispatch)


class C2GMacroEnv(gym.Env):
    """
    C2G-Bench High-Level Environment (Grid Manager / Macro Controller).

    Parameters
    ----------
    scenario : str
        Scenario key in ``config.yaml``.
    config_path : str or Path, optional
        Override path to ``config.yaml``.
    committed_max_mw : float, optional
        Maximum regulation capacity the agent may bid.
        Defaults to 2× the scenario's ``committed_mw``.
    inner_action_fn : callable, optional
        If provided, called as ``fn(macro_obs, macro_action) → low_action``
        to produce a 4-D FastEnv action for each sub-step.  Allows research
        into HRL decompositions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "default",
        config_path: str | Path | None = None,
        inner_action_fn: Callable | None = None,
        sub_step_callback: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        with open(cfg_path, encoding="utf-8") as fh:
            full_cfg = yaml.safe_load(fh)

        self._gcfg    = full_cfg["global"]
        self._scfg    = full_cfg[scenario]
        self._scenario = scenario

        # Handshake config
        hs_cfg = full_cfg.get("handshake", {})
        self._rmcp_max     = float(hs_cfg.get("rmcp_max", 100.0))
        self._lambda_rev   = float(hs_cfg.get("lambda_rev", 1.0))
        self._lambda_elec  = float(hs_cfg.get("lambda_elec", 0.5))
        self._lambda_churn = float(hs_cfg.get("lambda_churn", 0.05))

        # DR baseline (standing contract for rejected bids)
        self._dr_rate_usd_mw = float(self._scfg.get("dr_rate_usd_mw", 5.0))

        self._inner_action_fn = inner_action_fn
        self._sub_step_callback = sub_step_callback

        # --- Spaces ------------------------------------------------------
        # [0] bid_mw_norm, [1] bid_price_norm
        self.action_space = spaces.Box(
            low  = np.array([0.0, 0.0], dtype=np.float32),
            high = np.array([1.0, 1.0], dtype=np.float32),
            dtype= np.float32,
        )
        # 19-D observation (17 original + 2 market signals)
        self.observation_space = spaces.Box(
            low  = np.full(19, -1.0, dtype=np.float32),
            high = np.full(19,  5.0, dtype=np.float32),
            dtype= np.float32,
        )
        # Override bounds for strictly non-negative features
        _nonneg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
        self.observation_space.low[_nonneg] = 0.0
        self.observation_space.high[:17] = np.array([
            2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 2.0,
        ], dtype=np.float32)
        # freq_dev_mean can be negative (index 14)
        self.observation_space.low[14] = -1.0
        # bess_target_prev at [13] normalised to [0,1]
        self.observation_space.low[13] = 0.0
        # market signals [17, 18] bounded [0, 5]
        self.observation_space.high[17] = 5.0
        self.observation_space.high[18] = 5.0

        # Inner environment
        _fixed_action_values = kwargs.pop("fixed_action_values", None)
        _thermal_overrides = kwargs.pop("thermal_overrides", None)
        _plant_overrides = kwargs.pop("plant_overrides", None)
        if _fixed_action_values:
            from c2g_env.experiments.action_ablation_env import ActionAblationFastEnv
            self._fast_env = ActionAblationFastEnv(
                scenario=scenario, config_path=cfg_path,
                fixed_action_values=_fixed_action_values,
                thermal_overrides=_thermal_overrides,
                plant_overrides=_plant_overrides,
            )
        else:
            self._fast_env = C2GFastEnv(
                scenario=scenario, config_path=cfg_path,
                thermal_overrides=_thermal_overrides,
                plant_overrides=_plant_overrides,
            )

        self._committed_max_mw = self._fast_env._committed_mw_max
        self._dr_baseline_mw = self._fast_env._dr_baseline_mw

        # Episode tracking
        self._macro_tick        = 0
        self._prev_bid_mw_norm  = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm    = 0.0
        self._last_reg_need_norm = 0.0
        self._episode_macro_ticks = (
            int(self._gcfg["episode_ticks"]) // _SUBSTEPS
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset both the outer macro environment and the inner FastEnv.
        """
        super().reset(seed=seed)
        inner_obs, inner_info = self._fast_env.reset(seed=seed, options=options)
        self._last_hw_reset_info = inner_info
        self._committed_max_mw = self._fast_env._committed_mw_max
        self._dr_baseline_mw = self._fast_env._dr_baseline_mw

        self._macro_tick         = 0
        self._prev_bid_mw_norm   = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm     = 0.0
        self._last_reg_need_norm = 0.0
        self._last_hw_obs        = inner_obs  # persisted across macro steps

        # Build macro obs from the reset inner obs
        return self._obs_from_reset(inner_obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one 15-minute macro step with 3-phase market handshake.

        Parameters
        ----------
        action : ndarray of shape (2,)
            [bid_mw_norm, bid_price_norm]
        """
        action          = np.clip(action.astype(np.float32),
                                  self.action_space.low,
                                  self.action_space.high)
        bid_mw_norm     = float(action[0])
        bid_price_norm  = float(action[1])

        bid_mw    = bid_mw_norm * self._committed_max_mw
        bid_price = bid_price_norm * 2.0 * self._rmcp_max

        # =================================================================
        # Market handshake (Phases 1–3)
        # =================================================================
        hs = self.run_handshake(
            self._fast_env._grid,
            bid_mw_norm=bid_mw_norm,
            bid_price_norm=bid_price_norm,
            committed_max_mw=self._committed_max_mw,
            rmcp_max=self._rmcp_max,
            dr_baseline_mw=self._dr_baseline_mw,
            dr_rate_usd_mw=self._dr_rate_usd_mw,
        )
        committed_mw = hs["committed_mw"]
        rate = hs["rate"]
        offer = hs["offer"]
        result = hs["result"]
        rmcp_norm = hs["rmcp_norm"]
        reg_need_norm = hs["reg_need_norm"]
        self._last_rmcp_norm = rmcp_norm
        self._last_reg_need_norm = reg_need_norm

        self._fast_env.committed_mw = committed_mw

        # =================================================================
        # Run 180 sub-steps
        # =================================================================
        sub_rewards    = []
        sub_obs_list   = []
        sub_infos      = []
        temp_As, temp_Bs = [], []
        regd_abs, lmps, load_norms = [], [], []
        track_errs    = []
        bess_actuals  = []
        spike_any      = False
        backlog_norms  = []
        flex_reductions = []
        cool_deltas     = []
        p_pumps         = []
        p_hvacs         = []
        inner_actions   = []   # executed 4-D inner actions
        terminated     = False
        truncated      = False
        last_info: dict = {}

        for sub in range(_SUBSTEPS):
            if self._inner_action_fn is not None:
                inner_obs = sub_obs_list[-1] if sub_obs_list else self._last_hw_obs
                low_action = self._inner_action_fn(inner_obs, action)
            else:
                low_action = np.array([
                    _DEFAULT_THROTTLE,
                    _DEFAULT_PUMP,
                    _DEFAULT_HVAC,
                    _DEFAULT_BESS,
                ], dtype=np.float32)

            obs, rew, term, trunc, info = self._fast_env.step(low_action)
            if self._sub_step_callback is not None:
                _pre = sub_obs_list[-1] if sub_obs_list else self._last_hw_obs
                self._sub_step_callback(_pre, low_action, obs, rew, term or trunc, info)
            sub_rewards.append(rew)
            sub_obs_list.append(obs)
            sub_infos.append(info)

            temp_As.append(info["temp_A"])
            temp_Bs.append(info["temp_B"])
            regd_abs.append(abs(info["regd_signal"]))
            lmps.append(info["lmp"])
            load_norms.append(obs[8])   # grid_load_norm index
            track_errs.append(info["tracking_err_kw"])
            bess_actuals.append(info["bess_actual_kw"])
            flex_reductions.append(info["flex_reduction_kw"])
            cool_deltas.append(info["cool_delta_kw"])
            p_pumps.append(info["p_pump_mw"])
            p_hvacs.append(info["p_hvac_mw"])
            inner_actions.append([
                info.get("throttle_batch", 0.0),
                info.get("pump_speed_A", 0.0),
                info.get("hvac_effort", 0.0),
                info.get("bess_dispatch", 0.0),
            ])
            spike_any = spike_any or bool(info["is_spike"])
            backlog_norms.append(min(
                info["backlog_kw"] / self._fast_env._workload.p_flex_max_kw, 2.0
            ))
            last_info = info

            if term or trunc:
                terminated = term
                truncated  = trunc
                break

        self._last_hw_obs = sub_obs_list[-1]  # update for next macro period
        self._macro_tick += 1
        if self._macro_tick >= self._episode_macro_ticks and not terminated:
            truncated = True

        # -----------------------------------------------------------------
        # Aggregate observation (19-D)
        # -----------------------------------------------------------------
        T_safe   = self._fast_env._thermal.T_safe
        last_obs = sub_obs_list[-1]
        mean_lmp = float(np.mean(lmps))
        bess_soc_end = last_obs[2]

        # Inject p_flex_max_kw into infos so aggregate_macro_obs can
        # compute backlog_norms without reaching into the env.
        p_flex_max_kw = self._fast_env._workload.p_flex_max_kw
        for idx, info in enumerate(sub_infos):
            info["p_flex_max_kw"] = p_flex_max_kw

        obs = self.aggregate_macro_obs(
            sub_obs_list, sub_infos,
            T_safe=T_safe,
            committed_mw=committed_mw,
            tracking_floor_kw=self._fast_env._macro_tracking_floor_kw,
            bid_mw_norm=bid_mw_norm,
            bid_price_norm=bid_price_norm,
            rmcp_norm=rmcp_norm,
            reg_need_norm=reg_need_norm,
        )

        # -----------------------------------------------------------------
        # Macro reward (market handshake)
        # -----------------------------------------------------------------
        hours = _SUBSTEPS * self._fast_env._dt / 3600.0  # 0.25 h for 15-min

        # Tracking performance score [0, 1]
        tracking_err_norm = float(np.mean(track_errs)) / max(
            committed_mw * 1_000, self._fast_env._macro_tracking_floor_kw
        )
        perf_score = max(0.0, 1.0 - tracking_err_norm)

        # Regulation revenue: rate × MW × performance × hours
        regulation_revenue = rate * committed_mw * perf_score * hours

        # Electricity cost (always runs, prevents "never bid" exploit)
        mean_p_facility_norm = float(np.mean([o[5] for o in sub_obs_list]))
        electricity_cost = mean_lmp * mean_p_facility_norm * hours

        mean_sub_reward  = float(np.mean(sub_rewards))
        commit_churn_pen = abs(bid_mw_norm - self._prev_bid_mw_norm)

        r_regulation = self._lambda_rev  *  regulation_revenue / 1000.0
        r_sub        = mean_sub_reward
        r_elec       = -self._lambda_elec * electricity_cost   / 1000.0
        r_churn      = -self._lambda_churn * commit_churn_pen
        reward = float(r_regulation + r_sub + r_elec + r_churn)

        self._prev_bid_mw_norm = bid_mw_norm
        self._prev_bid_price_norm = bid_price_norm

        info_out = {
            "macro_tick":            self._macro_tick,
            "committed_mw":          committed_mw,
            "bid_mw":                bid_mw,
            "bid_price":             bid_price,
            "bid_accepted":          result["accepted"],
            "accept_prob":           result["accept_prob"],
            "clearing_price":        result["clearing_price"],
            "regulation_revenue":    regulation_revenue,
            "electricity_cost":      electricity_cost,
            "perf_score":            perf_score,
            "rate":                  rate,
            "mean_sub_reward":       mean_sub_reward,
            "commit_churn_pen":      commit_churn_pen,
            "mean_tracking_err":     float(np.mean(track_errs)),
            "mean_lmp":              mean_lmp,
            "rmcp_usd":             offer["rmcp_usd"],
            "residual_mw":          offer["residual_mw"],
            "spike_any":             spike_any,
            "temp_A_max":            max(temp_As),
            "temp_B_max":            max(temp_Bs),
            "bess_soc_end":          bess_soc_end,
            "backlog_norm_mean":     float(np.mean(backlog_norms)),
            "sub_steps_run":         len(sub_rewards),
            "scenario":              self._scenario,
            "last_inner_info":       last_info,
            "mean_flex_reduction_kw": float(np.mean(flex_reductions)),
            "mean_bess_actual_kw":   float(np.mean(bess_actuals)),
            "mean_cool_delta_kw":    float(np.mean(cool_deltas)),
            "mean_p_pump_mw":        float(np.mean(p_pumps)),
            "mean_p_hvac_mw":        float(np.mean(p_hvacs)),
            "mean_inner_throttle":   float(np.mean([a[0] for a in inner_actions])) if inner_actions else 0.0,
            "mean_inner_pump":       float(np.mean([a[1] for a in inner_actions])) if inner_actions else 0.0,
            "mean_inner_hvac":       float(np.mean([a[2] for a in inner_actions])) if inner_actions else 0.0,
            "mean_inner_bess":       float(np.mean([a[3] for a in inner_actions])) if inner_actions else 0.0,
            "reward_regulation":     r_regulation,
            "reward_sub":            r_sub,
            "reward_elec":           r_elec,
            "reward_churn":          r_churn,
        }
        return obs, reward, terminated, truncated, info_out

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Reusable static / class methods for macro-level logic
    # ------------------------------------------------------------------
    # These are extracted so that other environments (e.g. wrappers that
    # pair a rule-based macro with a learned low-level) can reuse the
    # same observation aggregation and market handshake logic without
    # duplicating code.

    @staticmethod
    def aggregate_macro_obs(
        sub_obs_list: list[np.ndarray],
        sub_infos: list[dict],
        *,
        T_safe: float,
        committed_mw: float,
        tracking_floor_kw: float,
        bid_mw_norm: float,
        bid_price_norm: float,
        rmcp_norm: float,
        reg_need_norm: float,
    ) -> np.ndarray:
        """
        Aggregate sub-step data into a 19-D macro observation.

        Parameters
        ----------
        sub_obs_list : list of ndarray (N, 18)
            Low-level observations collected over the macro window.
        sub_infos : list of dict
            Corresponding info dicts from ``C2GFastEnv.step()``.
        T_safe : float
            Thermal safety limit (°C).
        committed_mw : float
            Current grid commitment [MW].
        tracking_floor_kw : float
            Capacity-scaled minimum tracking denominator [kW].
        bid_mw_norm, bid_price_norm : float
            Previous bid action (normalised).
        rmcp_norm, reg_need_norm : float
            Latest market signals (normalised).

        Returns
        -------
        ndarray of shape (19,)
        """
        buf = np.array(sub_obs_list)  # (N, 18)

        temp_As = [info["temp_A"] for info in sub_infos]
        temp_Bs = [info["temp_B"] for info in sub_infos]
        regd_abs = [abs(info["regd_signal"]) for info in sub_infos]
        lmps = [info["lmp"] for info in sub_infos]
        track_errs = [info["tracking_err_kw"] for info in sub_infos]
        spike_any = any(info["is_spike"] for info in sub_infos)
        backlog_norms = [
            min(info["backlog_kw"] / max(info.get("p_flex_max_kw", 1.0), 1.0), 2.0)
            for info in sub_infos
        ]

        mean_lmp = float(np.mean(lmps))

        return np.array([
            np.mean(temp_As) / T_safe,                         # 0
            np.mean(temp_Bs) / T_safe,                         # 1
            float(buf[-1, 2]),                                 # 2  bess_soc_end
            float(np.mean(buf[:, 3])),                         # 3  p_base_mean
            float(np.mean(buf[:, 5])),                         # 4  p_facility_mean
            float(np.mean(regd_abs)),                          # 5  regd_mean
            min(mean_lmp / 200.0, 1.0),                        # 6  lmp_norm
            float(np.mean(buf[:, 8])),                         # 7  grid_load_mean
            float(np.mean(track_errs)) / max(committed_mw * 1_000, tracking_floor_kw),  # 8
            float(spike_any),                                  # 9
            max(0.0, (T_safe - max(temp_As)) / T_safe),       # 10 headroom_A
            max(0.0, (T_safe - max(temp_Bs)) / T_safe),       # 11 headroom_B
            bid_mw_norm,                                       # 12 bid_mw_prev
            bid_price_norm,                                    # 13 bid_price_prev
            float(np.mean(buf[:, 14])),                        # 14 freq_dev_mean
            float(np.mean(buf[:, 15])),                        # 15 v_pcc_mean
            float(np.mean(backlog_norms)),                     # 16 backlog_norm_mean
            float(np.clip(rmcp_norm, 0.0, 5.0)),              # 17 rmcp_norm
            float(np.clip(reg_need_norm, 0.0, 5.0)),          # 18 reg_need_norm
        ], dtype=np.float32)

    @staticmethod
    def macro_obs_from_reset(
        inner_obs: np.ndarray,
        *,
        T_safe: float,
        temp_A: float,
        temp_B: float,
        prev_bid_mw_norm: float = 0.5,
        prev_bid_price_norm: float = 0.5,
    ) -> np.ndarray:
        """
        Build macro observation from a single inner env reset observation.

        Parameters
        ----------
        inner_obs : ndarray of shape (18,)
            Observation from ``C2GFastEnv.reset()``.
        T_safe : float
            Thermal safety limit (°C).
        temp_A, temp_B : float
            Current zone temperatures (°C, raw, not normalised).
        prev_bid_mw_norm, prev_bid_price_norm : float
            Previous bid action (normalised).

        Returns
        -------
        ndarray of shape (19,)
        """
        return np.array([
            inner_obs[0],   # temp_A_norm
            inner_obs[1],   # temp_B_norm
            inner_obs[2],   # bess_soc
            inner_obs[3],   # p_base_norm
            inner_obs[5],   # p_facility_norm
            abs(inner_obs[6]),  # regd_abs
            inner_obs[7],   # lmp_norm
            inner_obs[8],   # grid_load_norm
            0.0,            # tracking_err_mean (no steps yet)
            0.0,            # is_spike_any
            max(0.0, (T_safe - temp_A) / T_safe),
            max(0.0, (T_safe - temp_B) / T_safe),
            prev_bid_mw_norm,
            prev_bid_price_norm,
            0.0,            # 14 freq_dev_mean (nominal)
            1.0,            # 15 v_pcc_mean (nominal)
            0.0,            # 16 backlog_norm_mean
            0.0,            # 17 rmcp_norm (no market signal yet)
            0.0,            # 18 reg_need_norm (no market signal yet)
        ], dtype=np.float32)

    @staticmethod
    def run_handshake(
        grid,
        *,
        bid_mw_norm: float,
        bid_price_norm: float,
        committed_max_mw: float,
        rmcp_max: float,
        dr_baseline_mw: float,
        dr_rate_usd_mw: float,
    ) -> dict:
        """
        Execute the 3-phase market handshake (RMCP → bid → clear).

        Parameters
        ----------
        grid : MacroGridSignal
            The grid model instance (provides ``step_rmcp`` / ``clear_bid``).
        bid_mw_norm, bid_price_norm : float
            Agent's normalised bid action.
        committed_max_mw : float
            Maximum biddable capacity [MW].
        rmcp_max : float
            RMCP normalisation ceiling [$/MWh].
        dr_baseline_mw : float
            Fallback DR commitment if bid rejected [MW].
        dr_rate_usd_mw : float
            Fallback DR rate [$/MW].

        Returns
        -------
        dict with keys:
            committed_mw, rate, rmcp_norm, reg_need_norm,
            offer (raw grid offer dict), result (raw clear_bid dict).
        """
        bid_mw = bid_mw_norm * committed_max_mw
        bid_price = bid_price_norm * 2.0 * rmcp_max

        # Phase 1: Grid posts RMCP + regulation need
        offer = grid.step_rmcp()
        rmcp_norm = offer["rmcp_usd"] / rmcp_max
        reg_need_norm = offer["residual_mw"] / max(committed_max_mw, 1.0)

        # Phase 3: Grid clears bid
        result = grid.clear_bid(bid_price, bid_mw)

        if result["accepted"]:
            committed_mw = result["accepted_mw"]
            rate = result["clearing_price"]
        else:
            committed_mw = dr_baseline_mw
            rate = dr_rate_usd_mw

        return {
            "committed_mw": committed_mw,
            "rate": rate,
            "rmcp_norm": rmcp_norm,
            "reg_need_norm": reg_need_norm,
            "offer": offer,
            "result": result,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _obs_from_reset(self, inner_obs: np.ndarray) -> np.ndarray:
        """Build macro observation from inner env's reset observation."""
        return self.macro_obs_from_reset(
            inner_obs,
            T_safe=self._fast_env._thermal.T_safe,
            temp_A=self._fast_env._thermal.temp_A,
            temp_B=self._fast_env._thermal.temp_B,
            prev_bid_mw_norm=self._prev_bid_mw_norm,
            prev_bid_price_norm=self._prev_bid_price_norm,
        )
