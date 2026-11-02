# c2g_env/physics/thermal.py
#
# Governing equations (per zone, lumped-capacitance energy balance):
#
#   Zone A (Liquid-Cooled, HPE Cray EX):
#     C_A dT_A/dt = P_IT,A - K_liq_eff*(T_A - T_supply,A) + K_env,A*(T_amb - T_A)
#     K_liq_eff  = K_liq * max(PUMP_MIN, pump_speed) * fault_factor
#       pump_speed ∈ [0, 1]: agent action controlling CDU circulating pump speed.
#       Slower pump → lower K_liq_eff → heat stored in water-loop thermal mass
#       (thermal-battery effect; exploitable for short-horizon grid regulation).
#     CDU chiller power: Q_heat_A / COP_liq
#     CDU pump power:    P_PUMP_MAX_MW * pump_speed  (circulating pump, separate)
#     COP_liq = COP_base_liq * max(0.3, 1 - beta_liq*(T_ref_A - T_supply_A))
#
#   Zone B (Air-Cooled, HPE ProLiant):
#     C_B dT_B/dt = P_IT,B - Q_HVAC + K_env,B*(T_amb - T_B)
#     Q_HVAC = min(P_hvac*COP,  K_eff*max(0, T_B - T_supply,B))
#     K_eff  = K_air*(0.3 + 0.7*hvac_effort)          # fans boost convection
#     COP    = COP_base * max(0.3, 1 - alpha*(T_amb - 25))
#                       * max(0.3, 1 - beta_air*(T_ref_B - T_supply_B))
#
# Integration: exact exponential (unconditionally stable for any dt).
#   For dT/dt = (P - K(T-T0) + Ke(Ta-T))/C   i.e.  b - a*T :
#     T(dt) = T_eq + (T_now - T_eq)*exp(-a*dt)    with  T_eq = b/a
#
# References
# ----------
# [1] Incropera, F.P., et al. (2007) "Fundamentals of Heat and Mass Transfer,"
#     6th ed., Wiley (ISBN 978-0-471-45728-2). Ch. 5: Transient Conduction —
#     lumped-capacitance ODE foundation for the C_dT/dt energy balance.
# [2] Moore, J.D., Chase, J.S., Ranganathan, P., Sharma, R. (2005)
#     "Making Scheduling 'Cool': Temperature-Aware Workload Placement in Data
#     Centers," USENIX ATC 2005, pp. 61–75.
#     https://www.usenix.org/legacy/publications/library/proceedings/usenix05/
#     tech/general/moore.html
# [3] Tang, Q., Gupta, S.K.S., Varsamopoulos, G. (2008) "Energy-efficient
#     Thermal-aware Task Scheduling for Homogeneous High-Performance Computing
#     Data Centers: A Cyber-Physical Approach," IEEE Trans. Parallel Distrib.
#     Syst., 19(11), 1458–1472.  DOI: 10.1109/TPDS.2008.111
# [4] ASHRAE TC 9.9 (2021) "Thermal Guidelines for Data Processing
#     Environments," 5th ed., ASHRAE. — supply temperature zone limits:
#     A1 = 15–27 °C, W3 = 5–40 °C used for T_supply_{A,B}_range.
#     https://www.ashrae.org/technical-resources/bookstore/datacom-series
# [5] Patankar, S.V. (2010) "Airflow and Cooling in a Data Center,"
#     J. Heat Transfer, 132(7), 073001. DOI: 10.1115/1.4001406
#     — raised-floor airflow model; K_env envelope coupling calibration.
# [6] Zimmermann, S., Meijer, I., Tiwari, M.K., Paredes, S., Michel, B.,
#     Poulikakos, D. (2012) "Aquasar: A hot water cooled data center with
#     direct energy reuse," Energy, 43(1), 237–245.
#     DOI: 10.1016/j.energy.2012.04.037
#     — single-phase liquid cooling CDU efficiency; pump-flow heat-transfer
#     coefficient model underpinning K_liq in Zone A.

import numpy as np

from c2g_env.thermal_limits import T_SAFE as _T_SAFE_DEFAULT


class ThermalTwin:
    """
    HPE Digital Twin for Data Center Thermodynamics.
    Models the thermal masses of Liquid-Cooled (Zone A) and Air-Cooled (Zone B)
    IT infrastructure using exact exponential integration for stability.

    Both supply-temperature setpoints (T_supply_A, T_supply_B) are adjustable.
    Lowering a setpoint improves cooling but degrades COP (more chiller work).
    """

    # CDU circulating pump constants (Zone A liquid loop)
    P_PUMP_MAX_MW = 1.5   # Max pump electrical draw (MW) — ~0.6% of 250 MW facility
    PUMP_MIN      = 0.15  # Minimum safe pump speed (maintains minimum server-blade flow)

    # Nominal (baseline) operating point for cooling — used as the
    # Customer Baseline Load (CBL) reference for RegD tracking.
    HVAC_NOM_EFFORT = 0.7  # Zone-B HVAC effort at nominal operation
    PUMP_NOM_SPEED  = 1.0  # Zone-A CDU pump speed at nominal operation

    def __init__(self, dt_seconds=300.0):
        self.dt = dt_seconds

        # -- Zone A: HPE Cray EX (Liquid Cooled) --------------------------
        self.temp_A = 30.0          # Current temperature          (deg C)
        self.C_A = 27_000.0         # Thermal capacitance  (MJ/deg C) — τ≈12.7 min at full K
        self.K_liq = 35.0           # Liquid-loop heat-transfer K   (MW/deg C) — sized for 150 MW: T_eq=34.2°C
        self.T_supply_A = 30.0      # Coolant supply temperature    (deg C)
        self.T_ref_A = 30.0         # Design-point supply temp      (deg C)
        self.T_supply_A_range = (20.0, 40.0)  # ASHRAE W3 allowable range
        self.COP_base_liq = 30.0    # Base CDU COP at design-point supply
        self.COP_beta_liq = 0.03    # COP degradation per deg C below T_ref_A
        self.K_env_A = 0.5          # Envelope coupling to ambient  (MW/deg C)

        # -- Zone B: HPE ProLiant (Air Cooled) -----------------------------
        self.temp_B = 20.0          # Current temperature           (deg C)
        self.C_B = 10_000.0         # Thermal capacitance  (MJ/deg C) — τ≈12-19 min across HVAC range
        self.K_air = 13.0           # Full-fan air-side K           (MW/deg C) — sized for 100 MW: safe at hvac≥0.7
        self.T_supply_B = 20.0      # CRAH supply-air temperature   (deg C)
        self.T_ref_B = 20.0         # Design-point supply temp      (deg C)
        self.T_supply_B_range = (15.0, 27.0)  # ASHRAE A1 allowable range
        self.max_hvac_mw = 50.0     # Max HVAC electrical draw      (MW)
        self.COP_base = 3.5         # Base COP at 25 deg C ambient
        self.COP_alpha = 0.02       # COP degradation per deg C > 25
        self.COP_beta_air = 0.04    # COP degradation per deg C below T_ref_B
        self.K_env_B = 0.5          # Envelope coupling to ambient  (MW/deg C)

        # -- Environment ---------------------------------------------------
        self.T_amb = 25.0           # Ambient outdoor temperature   (deg C)
        self.T_safe = _T_SAFE_DEFAULT  # High-Assurance Silicon Limit (deg C) — from config.yaml

        # -- Cooling fault injection (resilience testing) ------------------
        # fault_factor=1.0  → normal operation
        # fault_factor=0.4  → 60% pump efficiency loss (cooling failure)
        # Cleared by reset(); injected by ScenarioManager in env.py.
        self.fault_factor: float = 1.0
        self.fault_active: bool = False

    def set_supply_temps(self, T_supply_A=None, T_supply_B=None):
        """Set cooling setpoints, clamped to allowable ranges."""
        if T_supply_A is not None:
            lo, hi = self.T_supply_A_range
            self.T_supply_A = float(np.clip(T_supply_A, lo, hi))
        if T_supply_B is not None:
            lo, hi = self.T_supply_B_range
            self.T_supply_B = float(np.clip(T_supply_B, lo, hi))

    def reset(
        self,
        temp_A: float | None = None,
        temp_B: float | None = None,
    ):
        """Reset thermal state.

        Parameters
        ----------
        temp_A : float, optional
            Initial Zone A temperature [°C]. Defaults to 30.0 (design idle).
        temp_B : float, optional
            Initial Zone B temperature [°C]. Defaults to 20.0 (design idle).
        """
        self.temp_A = float(temp_A) if temp_A is not None else 30.0
        self.temp_B = float(temp_B) if temp_B is not None else 20.0
        self.T_supply_A = self.T_ref_A
        self.T_supply_B = self.T_ref_B
        self.fault_factor = 1.0
        self.fault_active = False
        return self.temp_A, self.temp_B

    @property
    def p_cool_nominal_mw(self) -> float:
        """Nominal cooling electrical draw [MW] at baseline operating point.

        This is the cooling power the facility would draw absent any
        active regulation — the CBL reference for ΔP tracking.
        """
        return (self.HVAC_NOM_EFFORT * self.max_hvac_mw
                + self.P_PUMP_MAX_MW * max(self.PUMP_MIN, self.PUMP_NOM_SPEED))

    def set_cooling_fault(self, active: bool, fault_factor: float = 0.4) -> None:
        """Inject / clear a cooling fault.

        Parameters
        ----------
        active : bool
            True  → fault active (pump degraded)
            False → normal operations
        fault_factor : float
            Fraction of nominal cooling capacity remaining.
            Default 0.4 (60% pump efficiency loss per spec).
        """
        self.fault_active = bool(active)
        self.fault_factor = float(fault_factor) if active else 1.0

    # ------------------------------------------------------------------ #
    #  Exact exponential integrator (stable for any dt)                    #
    # ------------------------------------------------------------------ #
    def _exp_step(self, T_now, P_src, K_cool, T_cool, K_env, C):
        """
        Solve  C dT/dt = P_src - K_cool*(T - T_cool) + K_env*(T_amb - T)
        over one timestep using the exact analytical solution.
        """
        K_total = K_cool + K_env
        if K_total < 1e-9:
            return T_now + P_src * self.dt / C
        T_eq = (P_src + K_cool * T_cool + K_env * self.T_amb) / K_total
        decay = np.exp(-K_total * self.dt / C)
        return T_eq + (T_now - T_eq) * decay

    # ------------------------------------------------------------------ #
    #  COP helpers                                                         #
    # ------------------------------------------------------------------ #
    def cop_liquid(self):
        """CDU chiller COP — degrades when supply temp is lowered below design point."""
        factor = max(0.3, 1.0 - self.COP_beta_liq * (self.T_ref_A - self.T_supply_A))
        return self.COP_base_liq * factor

    def cop_air(self):
        """HVAC chiller COP — degrades with higher ambient and lower supply temp."""
        f_amb = max(0.3, 1.0 - self.COP_alpha * (self.T_amb - 25.0))
        f_supply = max(0.3, 1.0 - self.COP_beta_air * (self.T_ref_B - self.T_supply_B))
        return self.COP_base * f_amb * f_supply

    # ------------------------------------------------------------------ #
    #  Main step                                                           #
    # ------------------------------------------------------------------ #
    def step(self, p_it_A_mw, p_it_B_mw, hvac_effort, pump_speed=1.0):
        """
        Advance thermal state by one timestep.

        Parameters
        ----------
        p_it_A_mw   : IT power delivered to Zone A (liquid-cooled)  [MW]
        p_it_B_mw   : IT power delivered to Zone B (air-cooled)     [MW]
        hvac_effort : Zone-B HVAC fan + chiller effort              [0, 1]
        pump_speed  : Zone-A CDU circulating pump speed fraction    [0, 1]
            Controls liquid-loop flow rate and therefore the effective
            heat-transfer coefficient (K_liq_eff = K_liq * pump_speed).
            Reducing pump speed lets the water loop absorb heat temporarily
            (thermal-battery / inertia exploitation for grid regulation).
            Clamped to PUMP_MIN (0.15) to maintain minimum server-blade flow.

        Returns
        -------
        temps         : (temp_A, temp_B)
        cooling_power : (p_cool_A_mw, p_hvac_mw, p_pump_mw)  electrical draw
            p_cool_A_mw : CDU chiller electrical draw  [MW]
            p_hvac_mw   : Zone-B HVAC draw             [MW]
            p_pump_mw   : CDU circulating pump draw    [MW]
        """
        hvac_effort = float(np.clip(hvac_effort, 0.0, 1.0))
        pump_speed  = float(np.clip(pump_speed,  0.0, 1.0))
        p_it_A_mw   = max(0.0, float(p_it_A_mw))
        p_it_B_mw   = max(0.0, float(p_it_B_mw))

        # -- Zone A (Liquid-Cooled) --- pump-speed-modulated liquid loop ---
        # K_liq_eff scales with pump speed (flow rate ∝ pump speed for
        # single-phase forced convection in turbulent regime).
        # Slowing the pump stores heat in the water-loop thermal mass
        # (τ_A = C_A/K_liq ≈ 12.7 min at full speed) — the RL agent can
        # exploit this inertia for short-horizon frequency regulation.
        k_liq_eff = self.K_liq * max(self.PUMP_MIN, pump_speed) * self.fault_factor
        self.temp_A = self._exp_step(
            self.temp_A, p_it_A_mw,
            k_liq_eff, self.T_supply_A,
            self.K_env_A, self.C_A,
        )

        # CDU chiller power: heat rejected to coolant / chiller COP.
        Q_heat_A = k_liq_eff * max(0.0, self.temp_A - self.T_supply_A)
        cop_liq = self.cop_liquid()
        p_cool_A_mw = Q_heat_A / cop_liq

        # CDU circulating pump electrical draw (linear with speed).
        p_pump_mw = self.P_PUMP_MAX_MW * max(self.PUMP_MIN, pump_speed)

        # -- Zone B (Air-Cooled) -------------------------------------------
        p_hvac_mw = hvac_effort * self.max_hvac_mw
        cop = self.cop_air()
        Q_cap = p_hvac_mw * cop                       # HVAC capacity (MW_th)

        # Fault also degrades air-side heat-transfer coefficients.
        K_eff = self.K_air * self.fault_factor * (0.3 + 0.7 * hvac_effort)
        T_cross = (self.T_supply_B + Q_cap / K_eff) if K_eff > 1e-9 else self.T_supply_B

        if self.temp_B < self.T_supply_B:
            # Below supply air: HVAC cannot cool further, ambient only
            self.temp_B = self._exp_step(
                self.temp_B, p_it_B_mw,
                0.0, self.T_supply_B,
                self.K_env_B, self.C_B,
            )
        elif self.temp_B < T_cross:
            # Physics-limited: cooling = K_eff * (T - T_supply)
            self.temp_B = self._exp_step(
                self.temp_B, p_it_B_mw,
                K_eff, self.T_supply_B,
                self.K_env_B, self.C_B,
            )
        else:
            # Capacity-limited: HVAC at max, constant Q_cap removal
            # ODE becomes: C dT/dt = (P_IT - Q_cap) + K_env*(T_amb - T)
            self.temp_B = self._exp_step(
                self.temp_B, p_it_B_mw - Q_cap,
                0.0, self.T_supply_B,
                self.K_env_B, self.C_B,
            )

        return (self.temp_A, self.temp_B), (p_cool_A_mw, p_hvac_mw, p_pump_mw)
