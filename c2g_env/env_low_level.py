"""
Step 2.1 — C2G-FastEnv  (Low-Level / Hardware-Controller Environment)
======================================================================
A ``gymnasium.Env`` that wraps all seven C2G-Bench physics engines.

Every call to ``step()`` advances the physical simulation by one 5-second
interval.  The RL agent controls four levers:

  Lever 1  ``throttle_batch``  – DVFS factor for schedulable batch jobs  [0, 1]
  Lever 2  ``pump_speed_A``    – Zone-A CDU circulating pump speed       [0, 1]
                                  (modulates liquid-loop heat-transfer;
                                   lower speed stores heat in water-loop
                                   thermal mass for grid regulation)
  Lever 3  ``hvac_effort``     – Zone-B HVAC fan + chiller effort         [0, 1]
  Lever 4  ``bess_dispatch``   – BESS power command (−1 = full charge,
                                  +1 = full discharge / inject to grid)   [-1, 1]

The environment is designed for facility-level frequency regulation
(PJM/NYISO RegD).  The reward incentivises:
  1. Batch throughput            (do not shed batch jobs unnecessarily)
  2. Grid-signal tracking        (match committed ΔP_demand within ±5%)
  3. Thermal safety              (zero tolerance for temp > T_warn)
  4. BESS health                 (avoid deep discharge below SOC_min + 2%)

Observation space  (16-D, all values normalised to approximately [0, 1])
------------------------------------------------------------------------
  [0]  temp_A_norm        = T_A / T_safe
  [1]  temp_B_norm        = T_B / T_safe
  [2]  bess_soc           = BESS state-of-charge                  ∈ [0, 1]
  [3]  p_base_norm        = p_base_kw / FACILITY_CAP_KW
  [4]  p_flex_nom_norm    = p_flex_nom_kw / FACILITY_CAP_KW
  [5]  p_facility_norm    = p_facility_mw × 1000 / FACILITY_CAP_KW
  [6]  regd_signal        = normalised RegD signal                 ∈ [-1, 1]
  [7]  lmp_norm           = LMP / 200  (soft cap at $200/MWh)
  [8]  grid_load_norm     = NYISO zone load / historical max       ∈ [0, 1]
  [9]  is_spike           = GenAI burst flag                       ∈ {0, 1}
  [10] prev_throttle      = throttle from previous step            ∈ [0, 1]
  ... (indices 11-15: prev_pump_speed, pue_norm, T_amb_norm, freq_dev_norm, v_pcc_pu)
  [16] backlog_norm       = batch backlog / p_flex_max_kw           ∈ [0, 2]
  [11] prev_pump_speed    = pump speed from previous step          ∈ [0, 1]
  [12] pue_norm           = PUE / 2.5
  [13] T_amb_norm         = ambient temp (weather) normalised      ∈ [0, 1]
  [14] freq_dev_norm      = (f_grid − f_nom) / 0.5  clipped       ∈ [-1, 1]
  [15] v_pcc_pu           = PCC voltage per-unit (Thévenin)        ∈ [0, 1.1]

Reward
------
  r = α·throttle
    − β·|ΔP_demanded_kW − ΔP_actual_kW| / (committed_mw × 1000)
    − γ·[max(0, T_A − T_warn_A) + max(0, T_B − T_warn_B)]
    − soc_penalty   if SOC < SOC_min + 0.02
    − δ_f·max(0, |Δf| − 0.2)           (frequency penalty, dead-band ±0.2 Hz)
    − δ_v·volt_violation                 (voltage penalty outside [0.95, 1.05] pu)

Termination
-----------
  • ``terminated=True``  if T_A > T_safe **or** T_B > T_safe  (thermal fault)
  • ``terminated=True``  if |f − f_nom| > 0.5 Hz              (UFLS / OFGT)
  • ``terminated=True``  if v_pcc < 0.90 pu                   (UV relay trip)
  • ``truncated=True``   after ``episode_ticks`` steps (time limit)

Usage
-----
  from c2g_env import C2GFastEnv
  env = C2GFastEnv(scenario="scenario_a")
  obs, info = env.reset(seed=0)
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from c2g_env.physics.workload   import WorkloadOrchestrator
from c2g_env.physics.thermal    import ThermalTwin
from c2g_env.physics.electrical import DatacenterElectrical
from c2g_env.physics.bess       import BESSModel
from c2g_env.physics.macro_grid import MacroGridSignal
from c2g_env.physics.weather    import WeatherLoader

# ---------------------------------------------------------------------------
# Constants — kept in sync with electrical.py rack parameters
# ---------------------------------------------------------------------------
_FACILITY_CAP_KW = 250_000.0    # Nameplate IT capacity (kW)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

_THERMAL_OVERRIDE_KEYS = frozenset({
    "C_A", "C_B", "K_liq", "K_air", "K_env_A", "K_env_B",
    "T_supply_A", "T_supply_B", "COP_base", "fault_factor", "T_amb",
})

# Plant capacity overrides — re-instantiate the facility at a different
# nameplate without editing physics defaults. Keys map to engine attributes;
# rack counts drive both the workload profile and the electrical model so IT
# nameplate and the observation-normalisation denominator stay consistent.
_PLANT_THERMAL_KEYS = frozenset({
    "C_A", "C_B", "K_liq", "K_air", "K_env_A", "K_env_B",
    "max_hvac_mw", "P_PUMP_MAX_MW",
})
_PLANT_ELEC_KEYS = frozenset({"S_xfmr_mva", "p_iron_mw", "p_lighting_mw", "p_network_mw"})
_PLANT_BESS_KEYS = frozenset({"E_NOM_MWH", "P_MAX_MW"})
_PLANT_RACK_KEYS = frozenset({"n_racks_a_flex", "n_racks_a_base", "n_racks_b"})
# Market-commitment terms scale with capacity too; they patch the scenario
# config (not an engine) so commit_norm stays scale-invariant.
_PLANT_SCENARIO_KEYS = frozenset({"committed_mw_max", "dr_baseline_mw"})
_PLANT_OVERRIDE_KEYS = (
    _PLANT_RACK_KEYS | _PLANT_ELEC_KEYS | _PLANT_THERMAL_KEYS
    | _PLANT_BESS_KEYS | _PLANT_SCENARIO_KEYS
)


class C2GFastEnv(gym.Env):
    """
    C2G-Bench Low-Level Environment (Hardware Controller).

    Parameters
    ----------
    scenario : str
        Key in ``config.yaml``: ``"default"``, ``"scenario_a"``,
        ``"scenario_b"``, or ``"scenario_c"``.
    config_path : str or Path, optional
        Override path to ``config.yaml``.
    plant_overrides : mapping, optional
        Runtime facility-capacity overrides (rack counts + electrical/thermal/
        BESS sizing). Takes precedence over any ``plant`` block in the scenario
        config, letting a Hydra ``plant_profiles`` group re-instantiate the
        plant without editing ``config.yaml``.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Space definitions
    # ------------------------------------------------------------------
    _ACT_LOW  = np.array([0.0,  0.0,  0.0, -1.0], dtype=np.float32)  # throttle, pump, hvac, bess
    _ACT_HIGH = np.array([1.0,  1.0,  1.0,  1.0], dtype=np.float32)

    _OBS_LOW  = np.array([
        0.0,  # temp_A_norm
        0.0,  # temp_B_norm
        0.0,  # bess_soc
        0.0,  # p_base_norm
        0.0,  # p_flex_nom_norm
        0.0,  # p_facility_norm
       -1.0,  # regd_signal (signed)
        0.0,  # lmp_norm
        0.0,  # grid_load_norm
        0.0,  # is_spike
        0.0,  # prev_throttle
        0.0,  # prev_pump_speed
        0.0,  # pue_norm
        0.0,  # T_amb_norm
       -1.0,  # freq_dev_norm (normalised frequency deviation)
        0.0,  # v_pcc_pu (PCC voltage in per-unit)
        0.0,  # backlog_norm (batch queue depth / p_flex_max_kw)
        0.0,  # committed_mw_norm (current DR commitment / committed_mw_max)
    ], dtype=np.float32)

    _OBS_HIGH = np.array([
        2.0,  # temp_A_norm (can temporarily exceed 1)
        2.0,  # temp_B_norm
        1.0,  # bess_soc
        1.0,  # p_base_norm
        1.0,  # p_flex_nom_norm
        2.0,  # p_facility_norm
        1.0,  # regd_signal
        1.0,  # lmp_norm
        1.0,  # grid_load_norm
        1.0,  # is_spike
        1.0,  # prev_throttle
        1.0,  # prev_pump_speed
        2.0,  # pue_norm
        1.0,  # T_amb_norm
        1.0,  # freq_dev_norm
        1.1,  # v_pcc_pu (slight overvoltage possible)
        2.0,  # backlog_norm (capped at 2 × p_flex_max)
        1.0,  # committed_mw_norm
    ], dtype=np.float32)

    def __init__(
        self,
        scenario: str = "default",
        config_path: str | Path | None = None,
        thermal_overrides: Mapping[str, float] | None = None,
        plant_overrides: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        with open(cfg_path, encoding="utf-8") as fh:
            full_cfg = yaml.safe_load(fh)

        self._gcfg   = full_cfg["global"]
        self._rcfg   = full_cfg["reward"]
        self._scfg   = full_cfg[scenario]
        self._scenario = scenario
        for key in ("tracking_floor_fraction", "macro_tracking_floor_fraction"):
            fraction = float(self._rcfg[key])
            if not np.isfinite(fraction) or fraction <= 0.0:
                raise ValueError(f"reward.{key} must be finite and positive")
        self._thermal_overrides = {
            str(key): float(value) for key, value in (thermal_overrides or {}).items()
        }
        unknown_thermal_keys = set(self._thermal_overrides) - _THERMAL_OVERRIDE_KEYS
        if unknown_thermal_keys:
            raise ValueError(
                f"Unknown thermal override key(s): {sorted(unknown_thermal_keys)}"
            )
        self._plant_overrides = {
            str(key): value for key, value in dict(plant_overrides or {}).items()
        }
        unknown_plant_keys = set(self._plant_overrides) - _PLANT_OVERRIDE_KEYS
        if unknown_plant_keys:
            raise ValueError(
                f"Unknown plant override key(s): {sorted(unknown_plant_keys)}"
            )

        self.action_space = spaces.Box(
            low=self._ACT_LOW, high=self._ACT_HIGH, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self._OBS_LOW, high=self._OBS_HIGH, dtype=np.float32
        )

        # Simulator handles (reset on each reset() call)
        self._workload  = None
        self._thermal   = None
        self._elec      = None
        self._bess      = None
        self._grid      = None
        self._weather   = None

        # Episode state
        self._tick           = 0
        self._prev_throttle  = 1.0
        self._prev_pump_speed = 1.0
        self._prev_regd_signal = 0.0
        # Market commitments scale with capacity; plant overrides win over the
        # scenario config so commit_norm and tracking stay scale-invariant.
        self._committed_mw_max = float(
            self._plant_overrides.get("committed_mw_max", self._scfg["committed_mw_max"])
        )
        self._dr_baseline_mw = float(
            self._plant_overrides.get("dr_baseline_mw", self._scfg["dr_baseline_mw"])
        )
        self._committed_mw   = self._dr_baseline_mw
        self._episode_ticks  = int(self._gcfg["episode_ticks"])
        self._dt             = float(self._gcfg["dt_seconds"])
        # IT-nameplate normalisation denominator; recomputed per-plant in reset().
        self._facility_cap_kw = _FACILITY_CAP_KW

    # ------------------------------------------------------------------
    # Public API helpers
    # ------------------------------------------------------------------

    @property
    def committed_mw(self) -> float:
        """Current regulation capacity commitment [MW]."""
        return self._committed_mw

    @committed_mw.setter
    def committed_mw(self, value: float) -> None:
        """Set regulation capacity commitment, clamped to [0, committed_mw_max]."""
        cap = self._committed_mw_max
        self._committed_mw = float(np.clip(value, 0.0, cap))

    # ------------------------------------------------------------------
    # Gymnasium core
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Re-initialise all simulators and return the initial observation.

        Parameters
        ----------
        seed : int, optional
            RNG seed passed to all stochastic simulators.
        options : dict, optional
            Optional dictionary that may contain ``"scenario"`` key to
            override the scenario for this episode.
        """
        super().reset(seed=seed)
        # Use Gymnasium's seeded RNG when no explicit seed is provided so
        # parallel worker envs (seeded independently by SB3's VecEnv) each
        # produce a distinct trajectory instead of all falling back to 42.
        rng_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))

        # Allow per-episode scenario override
        if options and "scenario" in options:
            cfg_path = _CONFIG_PATH
            with open(cfg_path, encoding="utf-8") as fh:
                full_cfg = yaml.safe_load(fh)
            self._scfg     = full_cfg[options["scenario"]]
            self._scenario = options["scenario"]
        scfg = self._scfg
        gcfg = self._gcfg

        # --- Build / reset simulators ------------------------------------
        # Runtime plant_overrides (e.g. a Hydra plant_profiles group) win over
        # any ``plant`` block carried in the scenario config.
        plant = self._plant_overrides or scfg.get("plant") or {}
        # Re-derive capacity-scaled market commitments (survives a per-episode
        # scenario override via options).
        self._dr_baseline_mw   = float(plant.get("dr_baseline_mw", scfg["dr_baseline_mw"]))
        self._workload = WorkloadOrchestrator(
            trace_dir=gcfg["trace_dir"], dt_seconds=self._dt, seed=rng_seed,
            n_racks_a_flex=plant.get("n_racks_a_flex"),
            n_racks_a_base=plant.get("n_racks_a_base"),
            n_racks_b=plant.get("n_racks_b"),
        )
        self._thermal = ThermalTwin(dt_seconds=self._dt)
        # T_safe is the single-source-of-truth silicon limit from config.yaml.
        self._thermal.T_safe = float(gcfg.get("T_safe", self._thermal.T_safe))
        # Pass scenario ambient temperature so reset provides a warm-start
        # that matches the physical steady state at the scenario's T_amb.
        # Formula: T_idle ≈ T_amb + P_idle / K  (equilibrium at zero IT load)
        # We keep it simple and use T_amb + 5°C as a universal warm-start.
        _T_amb_init = float(scfg.get("T_amb", 25.0))
        self._thermal.reset(
            temp_A=min(_T_amb_init + 5.0, self._thermal.T_safe - 1.0),
            temp_B=min(_T_amb_init + 5.0, self._thermal.T_safe - 1.0),
        )
        self._elec = DatacenterElectrical()
        self._elec.reset()
        self._bess = BESSModel(dt_seconds=self._dt)
        self._bess.reset()

        # --- Apply plant capacity overrides ------------------------------
        # Rack counts set IT nameplate; keep the electrical model in sync with
        # the workload so the round-trip util inversion and UPS loading stay
        # consistent. Extensive thermal/BESS/transformer terms scale together.
        if plant:
            if "n_racks_a_flex" in plant or "n_racks_a_base" in plant:
                self._elec.n_racks_A = int(
                    plant.get("n_racks_a_flex", 0) + plant.get("n_racks_a_base", 0)
                )
            if "n_racks_b" in plant:
                self._elec.n_racks_B = int(plant["n_racks_b"])
            for key in _PLANT_ELEC_KEYS:
                if key in plant:
                    setattr(self._elec, key, float(plant[key]))
            for key in _PLANT_THERMAL_KEYS:
                if key in plant:
                    setattr(self._thermal, key, float(plant[key]))
            for key in _PLANT_BESS_KEYS:
                if key in plant:
                    setattr(self._bess, key, float(plant[key]))
        # IT nameplate (kW) drives observation normalisation; derive it from the
        # electrical model so it can never desync from the rack counts.
        self._facility_cap_kw = (
            self._elec.n_racks_A * self._elec.p_max_rack_A_kw
            + self._elec.n_racks_B * self._elec.p_max_rack_B_kw
        )
        self._committed_mw_max = float(plant.get(
            "committed_mw_max",
            float(scfg["committed_mw_max"]) * self._facility_cap_kw / _FACILITY_CAP_KW,
        ))
        self._tracking_floor_kw = self._facility_cap_kw * float(self._rcfg["tracking_floor_fraction"])
        self._macro_tracking_floor_kw = self._facility_cap_kw * float(self._rcfg["macro_tracking_floor_fraction"])
        self._grid = MacroGridSignal(
            energy_dir=gcfg["energy_dir"],
            zone=gcfg["nyiso_zone"],
            dt_seconds=self._dt,
            committed_mw=self._dr_baseline_mw,
            seed=rng_seed,
            market=gcfg.get("grid_market", "nyiso_nyc"),
        )
        self._weather_driven = bool(scfg.get("weather_driven", True))
        _weather_market = scfg.get(
            "weather_market",
            scfg.get("grid_market", gcfg.get("grid_market", "nyiso_nyc")),
        )
        self._weather = WeatherLoader(
            weather_dir=gcfg.get("weather_dir", "data/processed/weather"),
            market=_weather_market,
            dt_seconds=self._dt,
            fallback_temp_c=float(scfg["T_amb"]),
        )

        # --- Apply scenario parameters -----------------------------------
        # T_amb: set initial value from scenario config.
        # When weather_driven=True, this is overridden each tick by WeatherLoader.
        # When weather_driven=False (stress scenarios), this fixed value is used throughout.
        self._thermal.T_amb = float(scfg["T_amb"])
        if scfg.get("cooling_fault", False):
            self._thermal.set_cooling_fault(
                active=True,
                fault_factor=float(scfg.get("cooling_fault_factor", 0.4)),
            )
        for key, value in self._thermal_overrides.items():
            setattr(self._thermal, key, value)
        if "fault_factor" in self._thermal_overrides:
            self._thermal.fault_active = self._thermal.fault_factor < 1.0
        if "T_amb" in self._thermal_overrides:
            self._weather_driven = False
        # Override initial BESS SOC if specified
        bess_soc_init = float(scfg.get("bess_soc_init", 0.5))
        self._bess.set_initial_soc(bess_soc_init)

        self._committed_mw   = self._dr_baseline_mw
        self._tick           = 0
        self._prev_throttle  = 1.0
        self._prev_pump_speed = 1.0
        self._prev_regd_signal = 0.0   # overwritten by _build_obs_at_reset

        return self._build_obs_at_reset(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance simulation by one 5-second timestep.

        Parameters
        ----------
        action : ndarray of shape (4,)
            [throttle_batch, pump_speed_A, hvac_effort, bess_dispatch]
            All values are expected in the bounds defined by ``action_space``.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action         = np.clip(action.astype(np.float32),
                                 self._ACT_LOW, self._ACT_HIGH)
        throttle_batch = float(action[0])
        pump_speed_A   = float(action[1])
        hvac_effort    = float(action[2])
        bess_dispatch  = float(action[3])   # [-1, 1]

        # Scale BESS dispatch to MW.
        # Positive = discharge (DC injects power back to grid side).
        # Negative = charge    (DC draws more from grid).
        bess_mw = bess_dispatch * getattr(self._bess, "P_MAX_MW", 50.0)

        # -----------------------------------------------------------------
        # 1. Workload  → IT power by zone
        # -----------------------------------------------------------------
        w = self._workload.step(throttle_batch)

        # -----------------------------------------------------------------
        # 2. BESS
        # -----------------------------------------------------------------
        bess_out = self._bess.step(bess_mw)

        # -----------------------------------------------------------------
        # 3. Thermal  ← zone IT powers (MW)
        # -----------------------------------------------------------------
        # Update ambient temperature: real/synthetic weather OR static scenario value
        if self._weather_driven:
            self._thermal.T_amb = self._weather.temp_c(self._tick)
        # else: T_amb remains as set during reset() (static scenario override)

        p_it_A_mw = (w.p_base_a_kw + w.p_flex_kw) / 1_000.0
        p_it_B_mw =  w.p_base_b_kw               / 1_000.0
        (temp_A, temp_B), (p_cool_A_mw, p_hvac_mw, p_pump_mw) = self._thermal.step(
            p_it_A_mw=p_it_A_mw,
            p_it_B_mw=p_it_B_mw,
            hvac_effort=hvac_effort,
            pump_speed=pump_speed_A,
        )

        # -----------------------------------------------------------------
        # 4. Electrical  (full facility accounting: UPS/PDU/XFMR + PUE)
        # -----------------------------------------------------------------
        util_A = DatacenterElectrical._inverse_rack_util(
            w.p_base_a_kw + w.p_flex_kw,
            self._elec.n_racks_A, self._elec.p_idle_rack_A_kw,
            self._elec.p_max_rack_A_kw - self._elec.p_idle_rack_A_kw,
            self._elec.alpha_A,
        )
        util_B = DatacenterElectrical._inverse_rack_util(
            w.p_base_b_kw,
            self._elec.n_racks_B, self._elec.p_idle_rack_B_kw,
            self._elec.p_max_rack_B_kw - self._elec.p_idle_rack_B_kw,
            self._elec.alpha_B,
        )
        # p_pump_mw is a separate facility electrical load (CDU circulating pump)
        elec = self._elec.step(util_A, util_B, p_cool_A_mw, p_hvac_mw + p_pump_mw)

        # -----------------------------------------------------------------
        # 5. ΔP tracking  (against the signal the agent observed last step)
        # -----------------------------------------------------------------
        # Positive regd_signal → grid wants DC to REDUCE net draw.
        # The DC can deliver this by:
        #   (a) Reducing batch load (flex shedding)    → flex_reduction_kw
        #   (b) Discharging BESS (inject energy back)  → bess_actual_kw
        #   (c) Modulating cooling loads (HVAC + pump) → cool_delta_kw
        bess_actual_kw     = bess_out["actual_power_mw"] * 1_000.0
        flex_reduction_kw  = (1.0 - throttle_batch) * w.p_flex_nom_kw

        # Cooling load deviation from nominal operating point.
        # Positive cool_delta_kw = cooling draw increased above nominal
        # (i.e. facility is consuming MORE from grid → less net reduction).
        cool_delta_kw = (p_hvac_mw + p_pump_mw
                         - self._thermal.p_cool_nominal_mw) * 1_000.0

        # ΔP_actual = how much the DC has actually reduced its net draw.
        # Subtracting cool_delta_kw: increased cooling = more draw = less reduction.
        delta_p_actual_kw  = flex_reduction_kw + bess_actual_kw - cool_delta_kw

        # ΔP_demanded uses the RegD signal from the *previous* observation,
        # i.e. the signal the agent actually saw and responded to.
        delta_p_demanded_kw = (self._committed_mw * 1_000.0
                               * self._prev_regd_signal)

        tracking_err_kw = abs(delta_p_demanded_kw - delta_p_actual_kw)

        # -----------------------------------------------------------------
        # 6. Advance grid for next observation
        # -----------------------------------------------------------------
        gs = self._grid.step(committed_mw=self._committed_mw)
        self._prev_regd_signal = float(gs["regd_signal"])

        # -----------------------------------------------------------------
        # 6b. Frequency & voltage signals (safety-critical)
        # -----------------------------------------------------------------
        # Swing equation: tracking deficit drives frequency deviation
        tracking_deficit_mw = (delta_p_demanded_kw - delta_p_actual_kw) / 1_000.0
        self._grid._step_frequency(tracking_deficit_mw)
        f_grid_hz = self._grid.f_grid
        f_nom     = self._grid.f_nom

        # PCC voltage from electrical model (Thévenin equivalent)
        v_pcc_pu  = float(elec.get("v_pcc_pu", 1.0))
        v_drop_pu = float(elec.get("v_drop_pu", 0.0))

        # -----------------------------------------------------------------
        # 7. Reward
        # -----------------------------------------------------------------
        alpha      = float(self._rcfg["alpha"])
        beta       = float(self._rcfg["beta"])
        gamma      = float(self._rcfg["gamma_thermal"])
        T_warn_A   = float(self._rcfg["T_warn_A"])
        T_warn_B   = float(self._rcfg["T_warn_B"])
        soc_pen_c  = float(self._rcfg["soc_penalty"])

        # Normalize thermal excess to [0, 1] per zone:
        #   0 at T_warn, 1 at T_safe, so gamma is dimensionless like alpha/beta.
        T_safe = self._thermal.T_safe
        temp_headroom = max(T_safe - T_warn_A, 1.0)   # degrees in budget (≥1 guard)
        thermal_pen = (
            max(0.0, temp_A - T_warn_A) / temp_headroom
            + max(0.0, temp_B - T_warn_B) / temp_headroom
        )
        soc_pen = soc_pen_c if bess_out["soc_fraction"] < 0.12 else 0.0

        # Frequency penalty (dead-band ±0.2 Hz)
        delta_freq_pen_c = float(self._rcfg.get("delta_freq_penalty", 2.0))
        freq_dev_hz = abs(f_grid_hz - f_nom)
        freq_pen = delta_freq_pen_c * max(0.0, freq_dev_hz - 0.2)

        # Voltage penalty (ANSI C84.1 Range A: [0.95, 1.05] pu)
        delta_volt_pen_c = float(self._rcfg.get("delta_volt_penalty", 5.0))
        volt_pen = delta_volt_pen_c * (
            max(0.0, 0.95 - v_pcc_pu) + max(0.0, v_pcc_pu - 1.05)
        )

        sla_backlog_pen_c = float(self._rcfg.get("sla_backlog_penalty", 2.0))
        backlog_pen = sla_backlog_pen_c * (
            w.backlog_kw / self._workload.p_flex_max_kw
        )

        # Scale the zero-commitment safeguard with the effective IT nameplate.
        norm_kw = max(self._committed_mw * 1_000.0, self._tracking_floor_kw)

        r_throughput =  alpha * throttle_batch
        r_tracking   = -beta  * (tracking_err_kw / norm_kw)
        r_thermal    = -gamma * thermal_pen
        r_soc        = -soc_pen
        r_freq       = -freq_pen
        r_volt       = -volt_pen
        r_backlog    = -backlog_pen
        reward = float(r_throughput + r_tracking + r_thermal + r_soc + r_freq + r_volt + r_backlog)

        # -----------------------------------------------------------------
        # 8. Termination / truncation
        # -----------------------------------------------------------------
        T_safe     = self._thermal.T_safe
        thermal_fault = bool(temp_A > T_safe or temp_B > T_safe)

        # Under-frequency load shedding (UFLS) / over-frequency trip
        f_ufls = self._grid.f_nom - 0.5   # e.g. 59.5 Hz (US) / 49.5 Hz (EU)
        f_ofgt = self._grid.f_nom + 0.5   # over-frequency generator trip
        freq_fault = bool(f_grid_hz < f_ufls or f_grid_hz > f_ofgt)

        # Under-voltage relay trip (ANSI C84.1 Range B violation)
        v_uvr = 0.90   # 0.90 pu → UPS transfer to battery or relay trip
        voltage_fault = bool(v_pcc_pu < v_uvr)

        terminated = thermal_fault or freq_fault or voltage_fault

        self._tick += 1
        truncated  = self._tick >= self._episode_ticks

        # -----------------------------------------------------------------
        # 9. Observation
        # -----------------------------------------------------------------
        self._prev_throttle  = throttle_batch
        self._prev_pump_speed = pump_speed_A
        obs = self._build_obs(temp_A, temp_B, bess_out, w, elec, gs,
                              f_grid_hz=f_grid_hz, f_nom=f_nom,
                              v_pcc_pu=v_pcc_pu)

        info = {
            "tick":                  self._tick,
            "temp_A":                temp_A,
            "temp_B":                temp_B,
            "bess_soc":              bess_out["soc_fraction"],
            "p_facility_mw":         elec["p_facility_mw"],
            "p_total_it_mw":         elec["p_total_it_mw"],
            "p_cooling_mw":          elec["p_cooling_mw"],
            "p_flex_nom_kw":         w.p_flex_nom_kw,
            "p_flex_served_kw":      w.p_flex_kw,
            "pue":                   elec["pue_dynamic"],
            "tracking_err_kw":       tracking_err_kw,
            "delta_p_actual_kw":     delta_p_actual_kw,
            "delta_p_demanded_kw":   delta_p_demanded_kw,
            "flex_reduction_kw":     flex_reduction_kw,
            "bess_actual_kw":        bess_actual_kw,
            "cool_delta_kw":         cool_delta_kw,
            "lmp":                   gs["lmp_usd_mwh"],
            "regd_signal":           gs["regd_signal"],
            "reward":                reward,
            "backlog_kw":            w.backlog_kw,
            "avg_delay_steps":       w.avg_delay_steps,
            "is_spike":              w.is_spike_active,
            "pump_speed_A":          pump_speed_A,
            "throttle_batch":        throttle_batch,
            "hvac_effort":           hvac_effort,
            "bess_dispatch":         bess_dispatch,
            "p_hvac_mw":             p_hvac_mw,
            "p_pump_mw":             p_pump_mw,
            "T_amb":                  self._thermal.T_amb,
            "weather_driven":          self._weather_driven,
            "weather_source":          self._weather.source,
            "thermal_fault":         thermal_fault,
            "freq_fault":            freq_fault,
            "voltage_fault":         voltage_fault,
            "terminated":            terminated,
            "truncated":             truncated,
            "thermal_warning":       bool(thermal_pen > 0.0),
            "freq_warning":          bool(freq_pen > 0.0),
            "voltage_warning":       bool(volt_pen > 0.0),
            "f_grid_hz":             f_grid_hz,
            "f_nom_hz":              f_nom,
            "freq_dev_hz":           f_grid_hz - f_nom,
            "v_pcc_pu":              v_pcc_pu,
            "v_drop_pu":             v_drop_pu,
            "freq_penalty":          freq_pen,
            "volt_penalty":          volt_pen,
            "reward_throughput":     r_throughput,
            "reward_tracking":       r_tracking,
            "reward_thermal":        r_thermal,
            "reward_soc":            r_soc,
            "reward_freq":           r_freq,
            "reward_volt":           r_volt,
            "reward_backlog":        r_backlog,
            "scenario":              self._scenario,
            "throughput_ratio":      throttle_batch,
            "bess_age_frac":         self._bess._age_frac,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass   # headless benchmarking environment

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        temp_A: float,
        temp_B: float,
        bess_out: dict,
        w,               # WorkloadState
        elec: dict,
        gs: dict,
        f_grid_hz: float = 60.0,
        f_nom: float = 60.0,
        v_pcc_pu: float = 1.0,
    ) -> np.ndarray:
        T_safe = self._thermal.T_safe
        return np.array([
            temp_A / T_safe,
            temp_B / T_safe,
            bess_out["soc_fraction"],
            w.p_base_kw        / self._facility_cap_kw,
            w.p_flex_nom_kw    / self._facility_cap_kw,
            min(elec["p_facility_mw"] * 1_000.0 / self._facility_cap_kw, 2.0),
            float(np.clip(gs["regd_signal"], -1.0, 1.0)),
            min(gs["lmp_usd_mwh"] / 200.0, 1.0),
            float(gs["load_norm"]),
            float(w.is_spike_active),
            self._prev_throttle,
            self._prev_pump_speed,
            min(elec["pue_dynamic"] / 2.5, 2.0),
            self._weather.temp_norm(self._tick),
            float(np.clip((f_grid_hz - f_nom) / 0.5, -1.0, 1.0)),  # freq_dev_norm
            float(np.clip(v_pcc_pu, 0.0, 1.1)),                     # v_pcc_pu
            min(w.backlog_kw / self._workload.p_flex_max_kw, 2.0),  # backlog_norm
            self._committed_mw / self._committed_mw_max,  # committed_mw_norm
        ], dtype=np.float32)

    def _build_obs_at_reset(self) -> np.ndarray:
        """Build observation from actual simulator state at tick 0.

        Peeks at each simulator's tick-0 output and then rewinds the
        internal tick counters so the first real step() sees a clean state.
        This replaces the old hardcoded placeholder values (0.5, 0.3, 0.8 …)
        that were wrong whenever the scenario or trace differed from defaults.
        """
        T_safe = self._thermal.T_safe

        # ── Peek workload tick 0 (advance then rewind) ───────────────────
        w = self._workload.step(1.0)     # full throttle → nominal p_flex
        self._workload._tick = 0
        self._workload._backlog_kw           = 0.0   # reset queue after peek
        self._workload._delay_accum_kw_steps = 0.0
        self._workload._total_served_kw      = 0.0

        # ── Peek grid tick 0 (advance then rewind) ───────────────────────
        gs = self._grid.step()
        self._prev_regd_signal = float(gs["regd_signal"])
        self._grid._tick = 0
        self._grid._regd_state = 0.0     # restore pre-step AR(1) state
        self._grid._regd_buffer = []     # restore empty neutrality buffer

        # ── Peek thermal + electrical (advance then rewind) ──────────────
        p_it_A_mw = (w.p_base_a_kw + w.p_flex_kw) / 1_000.0
        p_it_B_mw =  w.p_base_b_kw / 1_000.0
        temp_A_saved, temp_B_saved = self._thermal.temp_A, self._thermal.temp_B
        (_, _), (p_cool_A, p_hvac, p_pump) = self._thermal.step(
            p_it_A_mw=p_it_A_mw, p_it_B_mw=p_it_B_mw,
            hvac_effort=ThermalTwin.HVAC_NOM_EFFORT,
            pump_speed=ThermalTwin.PUMP_NOM_SPEED,
        )
        # Rewind thermal to its post-reset temperatures
        self._thermal.temp_A = temp_A_saved
        self._thermal.temp_B = temp_B_saved

        util_A = DatacenterElectrical._inverse_rack_util(
            w.p_base_a_kw + w.p_flex_kw,
            self._elec.n_racks_A, self._elec.p_idle_rack_A_kw,
            self._elec.p_max_rack_A_kw - self._elec.p_idle_rack_A_kw,
            self._elec.alpha_A,
        )
        util_B = DatacenterElectrical._inverse_rack_util(
            w.p_base_b_kw,
            self._elec.n_racks_B, self._elec.p_idle_rack_B_kw,
            self._elec.p_max_rack_B_kw - self._elec.p_idle_rack_B_kw,
            self._elec.alpha_B,
        )
        elec = self._elec.step(util_A, util_B, p_cool_A, p_hvac + p_pump)
        self._elec.reset()               # clear cached _last_state

        return np.array([
            self._thermal.temp_A / T_safe,
            self._thermal.temp_B / T_safe,
            self._bess.soc_fraction,
            w.p_base_kw        / self._facility_cap_kw,
            w.p_flex_nom_kw    / self._facility_cap_kw,
            min(elec["p_facility_mw"] * 1_000.0 / self._facility_cap_kw, 2.0),
            float(np.clip(gs["regd_signal"], -1.0, 1.0)),
            min(gs["lmp_usd_mwh"] / 200.0, 1.0),
            float(gs["load_norm"]),
            float(w.is_spike_active),
            self._prev_throttle,
            self._prev_pump_speed,
            min(elec["pue_dynamic"] / 2.5, 2.0),
            self._weather.temp_norm(0),
            0.0,   # freq_dev_norm (nominal frequency at reset)
            1.0,   # v_pcc_pu (nominal voltage at reset)
            0.0,   # backlog_norm (no deferred work at reset)
            self._committed_mw / self._committed_mw_max,  # committed_mw_norm
        ], dtype=np.float32)
