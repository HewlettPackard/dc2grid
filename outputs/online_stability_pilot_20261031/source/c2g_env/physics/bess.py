"""
Battery Energy Storage System (BESS) — 150 MWh / 50 MW Li-ion NMC
====================================================================
Two backend implementations, selected automatically at import time:

  1. **PySAM backend** (preferred) — NREL BatteryStateful electrochemical
     model with Shepherd voltage curve, I²R thermal losses, and temperature-
     dependent capacity.  Requires ``nrel-pysam`` to be installed.

  2. **Pure-Python fallback** (default) — equivalent-circuit model that
     reproduces the key non-linearities the RL agent must learn:
       - Non-linear round-trip efficiency: η(C-rate, SOC)
       - SOC-dependent power limits (derating near empty/full)
       - Internal resistance heat dissipation
       - Calendar + cycle capacity fade (linear approximation)

Both backends expose an identical public API so the RL environment is
completely backend-agnostic.

References
----------
[1] NREL PySAM BatteryStateful documentation.
    https://nrel-pysam.readthedocs.io/en/main/modules/BatteryStateful.html
[2] Blair, N., DiOrio, N., Freeman, J., Gilman, P., Janzou, S. (2018)
    "System Advisor Model (SAM) General Description (Version 2017.9.5),"
    NREL/TP-6A20-70414, National Renewable Energy Laboratory.
    https://www.nrel.gov/docs/fy18osti/70414.pdf
    — Shepherd-curve electrochemical battery model in SAM/PySAM.
[3] Xu, B., Zhao, J., Zheng, T., Litvinov, E., Kirschen, D.S. (2018)
    "Factoring the Cycle Aging Cost of Batteries Participating in
    Electricity Markets," IEEE Trans. Power Syst., 33(2), 2248–2259.
    DOI: 10.1109/TPWRS.2017.2733339
[4] Shepherd, C.M. (1965) "Design of Primary and Secondary Cells:
    An Equation Describing Battery Discharge," J. Electrochem. Soc.,
    112(7), 657–664.  DOI: 10.1149/1.2423244
    — Shepherd voltage-curve equation underlying the BatteryStateful model.
[5] Wang, J., Liu, P., Hicks-Garner, J., et al. (2011) "Cycle-life model
    for graphite-LiFePO4 cells," J. Power Sources, 196(8), 3942–3948.
    DOI: 10.1016/j.jpowsour.2010.11.134
    — DOD/C-rate/temperature capacity-fade model; linear fade fallback.
[6] Hesse, H.C., Schimpe, M., Kucevic, D., Jossen, A. (2017)
    "Lithium-Ion Battery Storage for the Grid — A Review of Stationary
    Battery Storage System Design Tailored for Applications in Modern
    Power Grids," Energies, 10(12), 2107.  DOI: 10.3390/en10122107
    — η(C-rate, SOC) round-trip efficiency; NMC parameters (η_peak, k_crate).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Try PySAM backend
# ---------------------------------------------------------------------------
try:
    import PySAM.BatteryStateful as _BatteryStateful
    _PYSAM_AVAILABLE = True
except ModuleNotFoundError:
    _PYSAM_AVAILABLE = False


# ===========================================================================
# Pure-Python equivalent-circuit BESS
# ===========================================================================

class _SimpleBESSModel:
    """
    Equivalent-circuit Li-ion NMC battery model (no PySAM dependency).

    Physics
    -------
    Round-trip efficiency:
        η_rt(C, SOC) = η_peak² - k_crate·|C|² - k_soc·(SOC - 0.5)²
        where C = P / E_nom  (C-rate, hr⁻¹)

    SOC dynamics (per dt):
        discharge: ΔSOC = -P_actual·dt / (η_discharge · E_nom)
        charge:    ΔSOC = +|P_actual|·η_charge·dt / E_nom

    Power derating near SOC limits (avoids cliff-edge hard cut-off):
        P_discharge_max(SOC) = P_max · min(1, (SOC - SOC_min) / 0.10)
        P_charge_max(SOC)    = P_max · min(1, (SOC_max - SOC) / 0.05)

    Internal resistance heat:
        P_heat = R_int · I² ≈ R_int · (P / V_nom)²   (approximated at pack level)
    """

    # Design specs (15 MWh / 5 MW — constrained for multi-lever control)
    E_NOM_MWH    = 15.0
    P_MAX_MW     = 5.0
    SOC_INIT     = 0.50
    SOC_MIN      = 0.10
    SOC_MAX      = 0.95
    ETA_PEAK     = 0.97     # one-way peak efficiency (charge or discharge)
    K_CRATE      = 0.008    # efficiency loss per (C-rate)²
    K_SOC        = 0.010    # efficiency loss per (SOC - 0.5)²
    V_NOM        = 800.0    # nominal pack voltage  (V)
    R_INT_OHM    = 0.002    # internal resistance (Ω)

    def __init__(self, dt_seconds: float = 300.0) -> None:
        self.dt = dt_seconds
        self._soc   = self.SOC_INIT
        self._age_frac = 0.0   # capacity fade [0, 1]; 0 = new, 1 = EOL

    # ------------------------------------------------------------------
    # Public interface (mirrors PySAM backend)
    # ------------------------------------------------------------------

    def reset(self) -> float:
        self._soc = self.SOC_INIT
        self._age_frac = 0.0
        return self.soc_mwh

    def set_initial_soc(self, fraction: float) -> None:
        """Override the current SOC without a full reset. fraction ∈ [0, 1]."""
        self._soc = float(np.clip(fraction, self.SOC_MIN, self.SOC_MAX))

    @property
    def soc_mwh(self) -> float:
        return self._soc * self.E_NOM_MWH * (1.0 - self._age_frac * 0.2)

    @property
    def soc_fraction(self) -> float:
        return self._soc

    def step(self, power_mw: float) -> dict:
        """
        Command the BESS for one timestep.

        Parameters
        ----------
        power_mw : Positive = discharge (inject to grid); negative = charge.

        Returns
        -------
        Same dict keys as the PySAM backend.
        """
        power_mw = float(np.clip(power_mw, -self.P_MAX_MW, self.P_MAX_MW))

        # SOC-dependent derating — symmetric 10% window for both directions
        # so the gradient of derated power w.r.t. SOC has the same magnitude
        # for discharge and charge (previously 0.10 vs 0.05, a 2× asymmetry).
        _DERATE_WINDOW = 0.10
        if power_mw > 0:  # discharge
            derate = min(1.0, (self._soc - self.SOC_MIN) / _DERATE_WINDOW)
            power_mw = power_mw * max(0.0, derate)
        else:              # charge
            derate = min(1.0, (self.SOC_MAX - self._soc) / _DERATE_WINDOW)
            power_mw = power_mw * max(0.0, derate)

        # C-rate and efficiency — computed on the derated power
        c_rate = abs(power_mw) / self.E_NOM_MWH   # hr⁻¹
        eta_one_way = max(0.70,
            self.ETA_PEAK
            - self.K_CRATE * c_rate ** 2
            - self.K_SOC   * (self._soc - 0.5) ** 2
        )

        dt_hr = self.dt / 3600.0
        if power_mw > 0:    # discharging
            energy_mwh = power_mw * dt_hr
            delta_soc  = -energy_mwh / (eta_one_way * self.E_NOM_MWH)
        else:                # charging
            energy_mwh = abs(power_mw) * dt_hr
            delta_soc  = energy_mwh * eta_one_way / self.E_NOM_MWH

        new_soc = float(np.clip(self._soc + delta_soc, self.SOC_MIN, self.SOC_MAX))

        # Detect if SOC limit was clipped by the np.clip above, and recompute
        # actual_power from available energy with the SAME efficiency that was
        # used for the SOC update (ensures energy accounting consistency).
        if power_mw > 0 and new_soc <= self.SOC_MIN + 1e-6:
            available_mwh = (self._soc - self.SOC_MIN) * self.E_NOM_MWH
            actual_power = available_mwh * eta_one_way / dt_hr  # discharge: energy/η → power
        elif power_mw < 0 and new_soc >= self.SOC_MAX - 1e-6:
            available_mwh = (self.SOC_MAX - self._soc) * self.E_NOM_MWH
            actual_power = -(available_mwh / eta_one_way / dt_hr)  # charge: energy×η → power
        else:
            actual_power = power_mw

        self._soc = new_soc

        # Heat dissipation via energy balance: P_heat = P_elec × (1 − η)
        # Using I²R with system-level current would require knowing the number
        # of parallel strings; energy balance is equivalent and unit-safe.
        heat_kw = abs(actual_power) * 1e3 * (1.0 - eta_one_way)

        # Linear calendar fade: 1% per 1000 equivalent full cycles
        self._age_frac += abs(actual_power) * dt_hr / (self.E_NOM_MWH * 1000.0)
        self._age_frac  = min(self._age_frac, 1.0)

        # Effective SOC-dependent power limits for next step
        p_disc_max = self.P_MAX_MW * min(1.0, (self._soc - self.SOC_MIN) / 0.10)
        p_chrg_max = self.P_MAX_MW * min(1.0, (self.SOC_MAX - self._soc) / 0.05)

        return {
            "actual_power_mw":    actual_power,
            "soc_mwh":            self.soc_mwh,
            "soc_fraction":       self._soc,
            "voltage_V":          self.V_NOM,          # simplified (constant)
            "current_A":          abs(actual_power) * 1e6 / max(self.V_NOM, 1.0),  # system-level proxy
            "temperature_C":      25.0 + heat_kw / 100.0,  # energy-balance proxy (°C)
            "heat_dissipated_kw": heat_kw,
            "max_charge_mw":      p_chrg_max,
            "max_discharge_mw":   p_disc_max,
        }


# ===========================================================================
# PySAM backend (used only when nrel-pysam is installed)
# ===========================================================================

class _PySAMBESSModel:
    """PySAM BatteryStateful backend.  Only instantiated when PySAM is available."""

    NOMINAL_ENERGY_MWH = 150.0
    MAX_POWER_MW       = 50.0
    INITIAL_SOC_PCT    = 50.0
    MIN_SOC_PCT        = 10.0
    MAX_SOC_PCT        = 95.0

    def __init__(self, dt_seconds: float = 300.0) -> None:
        self.DT_SECONDS = dt_seconds
        self._batt = self._create_battery()

    def _create_battery(self, initial_soc_pct: float | None = None):
        b = _BatteryStateful.new()
        b.Controls.control_mode       = 1
        b.Controls.dt_hr              = self.DT_SECONDS / 3600.0
        b.Controls.input_power        = 0.0
        b.ParamsCell.chem             = 1
        b.ParamsCell.Vnom_default     = 3.6
        b.ParamsCell.Vfull            = 4.1
        b.ParamsCell.Vexp             = 4.05
        b.ParamsCell.Vnom             = 3.4
        b.ParamsCell.Vcut             = 2.5
        b.ParamsCell.Qfull            = 3.2
        b.ParamsCell.Qexp             = 0.04
        b.ParamsCell.Qnom             = 2.8
        b.ParamsCell.C_rate           = 0.33
        b.ParamsCell.resistance       = 0.001
        b.ParamsCell.voltage_choice   = 0
        b.ParamsCell.initial_SOC      = (initial_soc_pct if initial_soc_pct is not None
                                         else self.INITIAL_SOC_PCT)
        b.ParamsCell.maximum_SOC      = self.MAX_SOC_PCT
        b.ParamsCell.minimum_SOC      = self.MIN_SOC_PCT
        b.ParamsCell.life_model       = 0
        b.ParamsCell.calendar_choice  = 0
        b.ParamsCell.calendar_q0      = 1.02
        b.ParamsCell.calendar_a       = 0.003
        b.ParamsCell.calendar_b       = -7280.0
        b.ParamsCell.calendar_c       = 930.0
        b.ParamsCell.cycling_matrix   = [
            [20,10000,97],[40,5000,95],[60,3000,90],[80,2000,85],[100,1000,80]
        ]
        b.ParamsPack.nominal_energy   = self.NOMINAL_ENERGY_MWH
        b.ParamsPack.nominal_voltage  = 800.0
        b.ParamsPack.Cp               = 1020.0
        b.ParamsPack.mass             = 200_000.0
        b.ParamsPack.surface_area     = 500.0
        b.ParamsPack.h                = 10.0
        b.ParamsPack.T_room_init      = 25.0
        b.ParamsPack.loss_choice      = 0
        b.ParamsPack.monthly_charge_loss    = [0.5] * 12
        b.ParamsPack.monthly_discharge_loss = [0.5] * 12
        b.ParamsPack.monthly_idle_loss      = [0.2] * 12
        b.ParamsPack.replacement_option = 0
        b.ParamsPack.cap_vs_temp = [[-10,80],[0,90],[25,100],[40,99]]
        b.setup()
        return b

    def reset(self) -> float:
        self._batt = self._create_battery()
        return self.soc_mwh

    def set_initial_soc(self, fraction: float) -> None:
        """Recreate the PySAM battery with a custom initial SOC. fraction ∈ [0, 1]."""
        pct = float(np.clip(fraction * 100.0, self.MIN_SOC_PCT, self.MAX_SOC_PCT))
        self._batt = self._create_battery(initial_soc_pct=pct)

    @property
    def soc_mwh(self) -> float:
        return self._batt.StatePack.SOC / 100.0 * self.NOMINAL_ENERGY_MWH

    @property
    def soc_fraction(self) -> float:
        return self._batt.StatePack.SOC / 100.0

    def step(self, power_mw: float) -> dict:
        power_mw = float(np.clip(power_mw, -self.MAX_POWER_MW, self.MAX_POWER_MW))
        self._batt.Controls.input_power = power_mw
        self._batt.execute(0)
        return {
            "actual_power_mw":    self._batt.StatePack.P,
            "soc_mwh":            self._batt.StatePack.SOC / 100.0 * self.NOMINAL_ENERGY_MWH,
            "soc_fraction":       self._batt.StatePack.SOC / 100.0,
            "voltage_V":          self._batt.StatePack.V,
            "current_A":          self._batt.StatePack.I,
            "temperature_C":      self._batt.StatePack.T_batt,
            "heat_dissipated_kw": self._batt.StatePack.heat_dissipated,
            "max_charge_mw":     -self._batt.StatePack.P_chargeable,
            "max_discharge_mw":   self._batt.StatePack.P_dischargeable,
        }


# ===========================================================================
# Public alias — pick the best available backend
# ===========================================================================

if _PYSAM_AVAILABLE:
    BESSModel = _PySAMBESSModel
else:
    BESSModel = _SimpleBESSModel

PYSAM_ACTIVE: bool = _PYSAM_AVAILABLE
