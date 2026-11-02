# c2g_env/physics/electrical.py
#
# Datacenter Electrical Model — HPE 250 MW Hyperscale Facility
#
# This module models the internal power delivery and conversion chain
# of a hyperscale data center, from the grid Point of Common Coupling
# (PCC) down to the IT silicon.  It captures the key non-linearities
# that an RL agent must learn to exploit for DOE Genesis-class
# orchestration.
#
# ┌─────────────────────────────────────────────────────────────────┐
# │                     Grid PCC (P_grid, Q_grid)                   │
# │                            │                                    │
# │                      ┌─────┴─────┐                              │
# │                      │  HV / MV  │  Transformer (η_xfmr, pf)    │
# │                      │  Xformer  │                              │
# │                      └─────┬─────┘                              │
# │                            │                                    │
# │              ┌─────────────┼─────────────┐                      │
# │              │             │             │                      │
# │         ┌────┴────┐   ┌────┴────┐  ┌─────┴─────┐                │
# │         │  UPS A  │   │  UPS B  │  │  BESS     │                │
# │         │(Liquid) │   │ (Air)   │  │ 150 MWh   │                │
# │         └────┬────┘   └────┬────┘  └─────┬─────┘                │
# │              │             │             │                      │
# │         ┌────┴────┐   ┌────┴────┐        │                      │
# │         │  PDU A  │   │  PDU B  │        │                      │
# │         └────┬────┘   └────┬────┘        │                      │
# │              │             │             │                      │
# │         ┌────┴────┐   ┌────┴────┐        │                      │
# │         │ IT Rack │   │ IT Rack │        │                      │
# │         │ Zone A  │   │ Zone B  │        │                      │
# │         │Cray EX  │   │ProLiant │        │                      │
# │         └─────────┘   └─────────┘        │                      │
# │              Cooling Power ──────────────┘                      │
# └─────────────────────────────────────────────────────────────────┘
#
# References:
#   [1] Barroso, L.A., Hölzle, U., Ranganathan, P. — "The Datacenter as a
#       Computer: Designing Warehouse-Scale Machines," 3rd ed.,
#       Morgan & Claypool, 2019 (ISBN 978-1-68173-565-4). Ch.6: Power
#       Provisioning. https://dl.acm.org/doi/book/10.5555/3238191
#   [2] Fan, X., Weber, W., Barroso, L.A. — "Power Provisioning for a
#       Warehouse-sized Computer," ACM ISCA 2007, pp. 13–23.
#       DOI: 10.1145/1273440.1250665  — Non-linear server power model.
#   [3] IEEE Std 3006.8-2018 — "Recommended Practice for Analyzing
#       Reliability Data for Equipment Used in Industrial and Commercial
#       Power Systems (PUE)."  https://ieeexplore.ieee.org/document/8353506
#   [4] ASHRAE TC 9.9 (2021) — "Thermal Guidelines for Data Processing
#       Environments," 5th ed.  https://www.ashrae.org/
#   [5] DOE — Genesis Mission Brief, November 2025. 250MW–1GW AI Campus.
#       https://www.energy.gov/
#   [6] Economou, D., Rivoire, S., Kozyrakis, C., Ranganathan, P. (2006)
#       "Full-System Power Analysis and Modeling for Server Environments,"
#       Workshop on Modeling, Benchmarking and Simulation (MoBS), ISCA 2006.
#       https://csl.stanford.edu/~christos/publications/2006.mantis.mobs.slides.pdf
#       — Empirical validation of superlinear exponent α ≈ 1.4 for servers.
#   [7] Shehabi, A., Smith, S., Sartor, D., et al. (2016) "United States Data
#       Center Energy Usage Report," Lawrence Berkeley National Laboratory,
#       LBNL-1005775. https://escholarship.org/content/qt84p772fc/qt84p772fc.pdf
#       — PUE benchmarks (1.2–1.8 range); dynamic PUE definition.
#
# Governing equations:
#
#   Server power (non-linear, per zone):
#     P_server(u) = N * [ P_idle + (P_max - P_idle) * u^alpha ]
#     where u = utilization [0,1], alpha ≈ 1.4 (GPU superlinear) [2]
#
#   UPS efficiency (load-dependent, double-conversion):
#     η_ups(x) = η_peak * x / (x + k_loss * (1 - x)^2 + k_noload)
#     where x = load fraction, η_peak ≈ 0.96  [1]
#
#   Transformer losses (copper + iron):
#     P_loss_xfmr = P_iron + P_copper * (P_load / S_rated)^2
#
#   Dynamic PUE:
#     PUE(t) = P_total_facility / P_IT = f(cooling, UPS, PDU, lighting)
#
#   Reactive power:
#     Q_grid = P_grid * tan(arccos(pf_composite))
#     pf_composite varies with load mix (GPUs ≈ 0.95, HVAC ≈ 0.85)

import numpy as np


class DatacenterElectrical:
    """
    HPE 250 MW Hyperscale Datacenter — Electrical Power Delivery Model.

    Models the full power conversion chain from grid PCC to silicon,
    including non-linear server power, UPS losses, transformer losses,
    PDU losses, and composite power factor.

    Designed for:
      - DOE Genesis Mission: 250MW+ datacenter-grid co-simulation
      - NeurIPS Benchmark Track: exposes non-trivial dynamics for RL
      - Computational efficiency: closed-form, no iteration needed
    """

    def __init__(self):
        # ── Facility-level parameters ────────────────────────────────
        self.S_xfmr_mva = 300.0        # Main transformer rating (MVA)
        self.p_iron_mw = 0.15           # Iron (no-load) losses (MW)
        self.p_copper_frac = 0.006      # Copper loss at rated load (fraction)

        # ── Zone A: HPE Cray EX — Liquid-cooled GPU cluster ─────────
        self.n_racks_A = 2000           # Number of rack units
        self.p_idle_rack_A_kw = 8.0     # Idle power per rack (kW)
        self.p_max_rack_A_kw = 75.0     # Peak power per rack (kW) — H100/H200
        self.alpha_A = 1.4              # GPU power-utilization exponent [2]

        # UPS Zone A (double-conversion, high-efficiency)
        self.ups_eta_peak_A = 0.97      # Peak UPS efficiency
        self.ups_k_loss_A = 0.03        # Quadratic loss coefficient
        self.ups_k_noload_A = 0.005     # No-load loss fraction

        # PDU Zone A (busway + tap-off boxes)
        self.pdu_loss_frac_A = 0.015    # 1.5% PDU losses

        # ── Zone B: HPE ProLiant — Air-cooled inference/DLRM ────────
        self.n_racks_B = 2500           # Number of rack units
        self.p_idle_rack_B_kw = 4.0     # Idle power per rack (kW)
        self.p_max_rack_B_kw = 40.0     # Peak power per rack (kW) — DL380a w/ inference GPU
        self.alpha_B = 1.2              # CPU/inference power exponent

        # UPS Zone B
        self.ups_eta_peak_B = 0.96      # Slightly lower efficiency
        self.ups_k_loss_B = 0.04
        self.ups_k_noload_B = 0.008

        # PDU Zone B
        self.pdu_loss_frac_B = 0.02     # 2% PDU losses (more cable runs)

        # ── Power Factor ─────────────────────────────────────────────
        self.pf_it_gpu = 0.95           # GPU PSU power factor
        self.pf_it_cpu = 0.92           # CPU/inference PSU power factor
        self.pf_hvac = 0.85             # HVAC motor power factor
        self.pf_lighting = 0.90         # Misc facility loads PF

        # ── Auxiliary / Facility loads ───────────────────────────────
        self.p_lighting_mw = 0.5        # Lighting, fire suppression, etc.
        self.p_network_mw = 1.5         # Networking, storage, control plane

        # ── Grid voltage at PCC (point of common coupling) ──────────
        # V_nom: nominal bus voltage at the PCC (medium-voltage side of XFMR)
        # Z_grid_pu: grid Thévenin impedance in per-unit (X/R ~ 10 for transmission)
        # V_min_safe: under-voltage relay threshold (ANSI C84.1: 0.90 pu)
        # V_max_safe: over-voltage protection threshold
        self.V_nom_kv:     float = 138.0   # typical HV feeder [kV]
        self.Z_grid_pu:    float = 0.04    # grid impedance in pu (stiff grid)
        self.V_min_safe_pu: float = 0.95   # ANSI C84.1 Range A minimum
        self.V_max_safe_pu: float = 1.05   # ANSI C84.1 Range A maximum

        # ── Cached state (updated each step) ─────────────────────────
        self._last_state = None

    def reset(self):
        self._last_state = None

    # ── Non-linear server power model [2] ────────────────────────────
    @staticmethod
    def _server_power_mw(util, n_racks, p_idle_kw, p_max_kw, alpha):
        """
        P_IT = N * [P_idle + (P_max - P_idle) * u^alpha]  (MW)

        The superlinear exponent alpha captures the non-linear
        relationship between GPU/CPU utilization and power draw.
        alpha > 1 means power grows faster than utilization.
        """
        util = np.clip(util, 0.0, 1.0)
        p_rack_kw = p_idle_kw + (p_max_kw - p_idle_kw) * (util ** alpha)
        return n_racks * p_rack_kw / 1000.0  # kW → MW

    @staticmethod
    def _inverse_rack_util(p_zone_kw: float, n_racks: int,
                           p_idle_kw: float, p_range_kw: float,
                           alpha: float) -> float:
        """
        Invert  P(u) = N × [p_idle + p_range × u^alpha]  to recover u ∈ [0, 1].
        """
        per_rack = p_zone_kw / max(n_racks, 1)
        frac = (per_rack - p_idle_kw) / max(p_range_kw, 1e-9)
        frac = float(np.clip(frac, 0.0, 1.0))
        return frac ** (1.0 / alpha)

    # ── UPS efficiency model ─────────────────────────────────────────
    @staticmethod
    def _ups_efficiency(load_frac, eta_peak, k_loss, k_noload):
        """
        η_UPS = η_peak * x / (x + k_loss*(1-x)^2 + k_noload)

        Low load → poor efficiency (no-load + fixed losses dominate).
        This creates an RL incentive to consolidate load.
        """
        x = np.clip(load_frac, 0.01, 1.0)
        return eta_peak * x / (x + k_loss * (1.0 - x) ** 2 + k_noload)

    # ── Transformer losses ───────────────────────────────────────────
    def _transformer_loss_mw(self, p_load_mw):
        """
        P_loss = P_iron + P_copper * (P_load / S_rated)^2

        Iron losses are constant (core magnetization).
        Copper losses scale with the square of the load current.
        """
        load_frac = min(p_load_mw / self.S_xfmr_mva, 1.5)
        return self.p_iron_mw + self.p_copper_frac * self.S_xfmr_mva * load_frac ** 2

    # ── Composite power factor ───────────────────────────────────────
    def _composite_pf(self, p_it_A, p_it_B, p_cool):
        """
        Weighted power factor across all load types.
        Returns (pf, Q_reactive_mvar).
        """
        total = p_it_A + p_it_B + p_cool + self.p_lighting_mw + self.p_network_mw
        if total < 0.1:
            return 0.95, 0.0

        # Weighted sum of reactive components
        q_it_A = p_it_A * np.tan(np.arccos(self.pf_it_gpu))
        q_it_B = p_it_B * np.tan(np.arccos(self.pf_it_cpu))
        q_cool = p_cool * np.tan(np.arccos(self.pf_hvac))
        q_misc = (self.p_lighting_mw + self.p_network_mw) * np.tan(np.arccos(self.pf_lighting))

        q_total = q_it_A + q_it_B + q_cool + q_misc
        s_total = np.sqrt(total ** 2 + q_total ** 2)
        pf = total / s_total if s_total > 0.1 else 0.95

        return pf, q_total

    # ── Main step ────────────────────────────────────────────────────
    def step(self, util_A, util_B, p_cool_A_mw, p_cool_B_mw):
        """
        Compute the full electrical state of the datacenter.

        Parameters
        ----------
        util_A : float
            Zone A GPU utilization [0, 1].
        util_B : float
            Zone B CPU/GPU utilization [0, 1].
        p_cool_A_mw : float
            Cooling power draw, Zone A (from ThermalTwin).
        p_cool_B_mw : float
            Cooling power draw, Zone B (from ThermalTwin).

        Returns
        -------
        dict with:
            p_it_A_mw      : IT power Zone A (at the server)
            p_it_B_mw      : IT power Zone B (at the server)
            p_total_it_mw  : Total IT power (both zones)
            p_ups_loss_mw  : Total UPS conversion losses
            p_pdu_loss_mw  : Total PDU distribution losses
            p_xfmr_loss_mw : Transformer losses
            p_cooling_mw   : Total cooling electrical draw
            p_aux_mw       : Auxiliary facility loads
            p_facility_mw  : Total facility power (at grid PCC)
            pue_dynamic    : Instantaneous PUE
            pf_composite   : Composite power factor
            q_reactive_mvar: Reactive power (MVAr)
            ups_eta_A      : UPS efficiency, Zone A
            ups_eta_B      : UPS efficiency, Zone B
        """
        util_A = float(np.clip(util_A, 0.0, 1.0))
        util_B = float(np.clip(util_B, 0.0, 1.0))
        p_cool_A_mw = max(0.0, float(p_cool_A_mw))
        p_cool_B_mw = max(0.0, float(p_cool_B_mw))

        # 1. IT power at the server (non-linear)
        p_it_A = self._server_power_mw(
            util_A, self.n_racks_A,
            self.p_idle_rack_A_kw, self.p_max_rack_A_kw, self.alpha_A
        )
        p_it_B = self._server_power_mw(
            util_B, self.n_racks_B,
            self.p_idle_rack_B_kw, self.p_max_rack_B_kw, self.alpha_B
        )
        p_total_it = p_it_A + p_it_B

        # 2. PDU losses (resistive, proportional to load)
        p_pdu_A = p_it_A * self.pdu_loss_frac_A
        p_pdu_B = p_it_B * self.pdu_loss_frac_B
        p_pdu_loss = p_pdu_A + p_pdu_B

        # 3. UPS losses (load-dependent efficiency)
        #    UPS sees IT load + PDU losses
        ups_load_A = p_it_A + p_pdu_A
        ups_cap_A = self.n_racks_A * self.p_max_rack_A_kw / 1000.0 * 1.1
        ups_frac_A = ups_load_A / ups_cap_A if ups_cap_A > 0 else 0.5
        ups_eta_A = self._ups_efficiency(
            ups_frac_A, self.ups_eta_peak_A,
            self.ups_k_loss_A, self.ups_k_noload_A
        )
        p_ups_in_A = ups_load_A / ups_eta_A if ups_eta_A > 0.1 else ups_load_A
        p_ups_loss_A = p_ups_in_A - ups_load_A

        ups_load_B = p_it_B + p_pdu_B
        ups_cap_B = self.n_racks_B * self.p_max_rack_B_kw / 1000.0 * 1.1
        ups_frac_B = ups_load_B / ups_cap_B if ups_cap_B > 0 else 0.5
        ups_eta_B = self._ups_efficiency(
            ups_frac_B, self.ups_eta_peak_B,
            self.ups_k_loss_B, self.ups_k_noload_B
        )
        p_ups_in_B = ups_load_B / ups_eta_B if ups_eta_B > 0.1 else ups_load_B
        p_ups_loss_B = p_ups_in_B - ups_load_B

        p_ups_loss = p_ups_loss_A + p_ups_loss_B

        # 4. Total load before transformer
        p_cooling = p_cool_A_mw + p_cool_B_mw
        p_aux = self.p_lighting_mw + self.p_network_mw
        p_pre_xfmr = p_ups_in_A + p_ups_in_B + p_cooling + p_aux

        # 5. Transformer losses
        p_xfmr_loss = self._transformer_loss_mw(p_pre_xfmr)

        # 6. Total facility power at grid PCC
        p_facility = p_pre_xfmr + p_xfmr_loss

        # 7. Dynamic PUE
        pue = p_facility / p_total_it if p_total_it > 0.1 else 1.0

        # 8. Power factor and reactive power
        pf, q_mvar = self._composite_pf(p_it_A, p_it_B, p_cooling)

        # 9. Voltage at PCC  (simplified steady-state voltage drop)
        # ΔV/V ≈ (P·R + Q·X) / V²  in per-unit with S_base = S_xfmr
        s_base_mva = max(self.S_xfmr_mva, 1.0)
        p_pu = p_facility / s_base_mva
        q_pu = q_mvar / s_base_mva
        # For typical transmission: X/R ≈ 10, so R_pu ≈ Z/√101, X_pu ≈ 10·R_pu
        xr_ratio = 10.0
        r_pu = self.Z_grid_pu / np.sqrt(1.0 + xr_ratio ** 2)
        x_pu = xr_ratio * r_pu
        v_drop_pu = p_pu * r_pu + q_pu * x_pu
        v_pcc_pu = float(np.clip(1.0 - v_drop_pu, 0.85, 1.10))

        self._last_state = {
            "p_it_A_mw": p_it_A,
            "p_it_B_mw": p_it_B,
            "p_total_it_mw": p_total_it,
            "p_ups_loss_mw": p_ups_loss,
            "p_pdu_loss_mw": p_pdu_loss,
            "p_xfmr_loss_mw": p_xfmr_loss,
            "p_cooling_mw": p_cooling,
            "p_aux_mw": p_aux,
            "p_facility_mw": p_facility,
            "pue_dynamic": pue,
            "pf_composite": pf,
            "q_reactive_mvar": q_mvar,
            "ups_eta_A": ups_eta_A,
            "ups_eta_B": ups_eta_B,
            "v_pcc_pu": v_pcc_pu,
            "v_drop_pu": v_drop_pu,
        }
        return self._last_state

    def get_diagnostics(self):
        """Return the last computed state for logging/debugging."""
        return self._last_state or {}
