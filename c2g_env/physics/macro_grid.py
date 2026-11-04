"""
Step 1.4 — Macro-Grid Signal Generator (The "Handshake")
=========================================================
Generates the two signals the C2G-Bench RL environment needs from the
regional grid:

  1. **Frequency Regulation Signal** (ΔP_grid, kW) — the real-time
     dispatch command the grid operator sends every timestep.
     The datacenter must match this by adjusting its three levers.

  2. **Locational Marginal Price** (LMP, $/MWh) — the wholesale energy
     price at the datacenter's Point of Common Coupling (PCC), used by
     the upper-level Market Orchestrator agent for economic decisions.

Frequency Regulation model
--------------------------
Modelled after PJM's RegD signal, which is designed for fast resources
(e.g., batteries, DVFS).  Key statistical properties reproduced:
  - Zero-mean over any 15-minute window (energy-neutral)
  - Band-limited noise: power spectral density peaks at 0.01–0.05 Hz
  - Normalised to [-1, 1]; scaled by the committed_mw capacity
  - Autocorrelation: AR(1) with ρ ≈ 0.8 at 5-min timestep

  ΔP_grid(t) = committed_mw × regD(t)   [MW]

LMP model
---------
Built on real NYISO 5-minute actual load data (11 load zones, 2023–2025).
A price proxy is derived from the load-duration curve:

  LMP(t) = lmp_base + lmp_slope × max(0, Load_t - Load_median)

This captures the non-linear marginal cost curve: prices spike sharply
when the grid is near peak load, creating a strong economic incentive for
the datacenter to reduce demand.  The zone is configurable (default: NYC).

References
----------
[1] PJM Manual 12: Balancing Operations, Section 4 (RegD signal spec).
    https://www.pjm.com/-/media/documents/manuals/m12.ashx
[2] NYISO Real-Time Actual Load data, 5-minute resolution.
    https://www.nyiso.com/real-time-dashboard
[3] Hogan, W.W. (2002): "Electricity Market Restructuring: Reforms of Reforms,"
    Journal of Regulatory Economics, 21(1), 103–132.
    DOI: 10.1023/A:1013557912617  — foundational LMP theory.
[4] Kundur, P. (1994) "Power System Stability and Control," McGraw-Hill
    (ISBN 978-0-07-035958-1), Ch. 12. — Swing equation (df/dt ∝ ΔP);
    damping coefficient D; frequency-deviation model calibration.
[5] Schweppe, F.C., Caramanis, M.C., Tabors, R.D., Bohn, R.E. (1988)
    "Spot Pricing of Electricity," Kluwer Academic
    (ISBN 978-0-89838-260-0). — LMP marginal-cost theory underpinning
    the load-duration LMP proxy.
[6] Kirby, B.J. (2005) "Frequency Regulation Basics and Trends,"
    Oak Ridge National Laboratory, ORNL/TM-2004/291.
    DOI: 10.2172/885974  https://www.osti.gov/biblio/885974
    — AR(1) statistical properties of PJM RegD (ρ≈0.8, zero-mean 15-min);
    ρ and σ calibration in MarketParams.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-market regulation and price calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketParams:
    """
    Calibration parameters for one ISO/market region.

    Parameters
    ----------
    name : str
        Human-readable ISO name (e.g. "PJM").
    region : str
        Sub-region or data-centre hub (e.g. "Northern Virginia / Ashburn").
    regulation_product : str
        Official regulation product name (e.g. "RegD", "FCR-N", "FCAS").
    regd_rho : float
        AR(1) autocorrelation at 5-second step.
    regd_sigma : float
        AR(1) noise standard deviation (stationary std ≈ sigma / sqrt(1 - rho^2)).
    window_ticks : int
        Energy-neutrality correction window in ticks (= settlement interval / 5 s).
    lmp_base_usd : float
        Off-peak base wholesale electricity price (USD/MWh).
    lmp_slope_usd : float
        Marginal price sensitivity above median load (USD/MWh per GW).
    load_mean_mw : float
        Reference mean zonal load (MW) — used for synthetic fallback.
    load_daily_amp_mw : float
        Daily sinusoidal load amplitude (MW) — used for synthetic fallback.
    load_noise_std_mw : float
        Load noise standard deviation (MW) — used for synthetic fallback.
    dc_hub : str
        Reference data-centre location for documentation purposes.
    data_url : str
        Canonical download URL for real regional load data.
    """
    name:               str
    region:             str
    regulation_product: str
    regd_rho:           float
    regd_sigma:         float
    window_ticks:       int
    lmp_base_usd:       float
    lmp_slope_usd:      float
    load_mean_mw:       float
    load_daily_amp_mw:  float
    load_noise_std_mw:  float
    dc_hub:             str
    data_url:           str
    zone_csv:           str | None = None   # energy CSV stem; None → synthetic

    # --- Market handshake (RMCP + regulation need) ---
    rmcp_base_usd:       float = 25.0    # base regulation market clearing price ($/MWh)
    rmcp_lmp_beta:       float = 0.3     # RMCP-LMP correlation coefficient
    rmcp_noise_std:      float = 5.0     # stochastic RMCP noise ($/MWh)
    reg_need_base_mw:    float = 50.0    # baseline regulation requirement (MW)
    reg_need_slope:      float = 0.5     # reg need increase with normalised load
    competitor_fill_rate: float = 0.7    # fraction already filled by competitors


#: Registry of supported markets.  Keys match the ``grid_market`` config field.
MARKET_PRESETS: dict[str, MarketParams] = {
    # ── 1. NYISO New York City ─────────────────────────────────────────────
    # Reference: NYISO OASIS real-time load, NYC zone (2023-2025)
    # RegD-equivalent signal; 15-min settlement.
    "nyiso_nyc": MarketParams(
        name               = "NYISO",
        region             = "New York City",
        regulation_product = "AGC / Secondary Reserve",
        regd_rho           = 0.9963,   # 0.80^(5/300)
        regd_sigma         = 0.022,
        window_ticks       = 180,      # 15 min
        lmp_base_usd       = 28.0,
        lmp_slope_usd      = 18.0,     # peak ≈ $118/MWh (5 GW above median)
        load_mean_mw       = 5_500.0,
        load_daily_amp_mw  = 1_500.0,
        load_noise_std_mw  = 300.0,
        dc_hub             = "Manhattan / New Jersey financial data centres",
        data_url           = "https://www.nyiso.com/load-data",
        zone_csv           = "NYC",
        rmcp_base_usd      = 25.0,
        reg_need_base_mw   = 40.0,
        competitor_fill_rate = 0.70,
    ),
    # ── 2. PJM Dominion (Northern Virginia) ───────────────────────────────
    # "Data Center Alley" — ~70 % of global internet traffic.
    # Same RegD product as our synthetic signal (PJM is the RegD origin).
    "pjm_dom": MarketParams(
        name               = "PJM",
        region             = "Northern Virginia / Ashburn",
        regulation_product = "RegD (Fast-Response Regulation)",
        regd_rho           = 0.9963,
        regd_sigma         = 0.022,
        window_ticks       = 180,
        lmp_base_usd       = 35.0,
        lmp_slope_usd      = 10.0,     # deep market; peak ≈ $115/MWh
        load_mean_mw       = 14_000.0,
        load_daily_amp_mw  =  3_000.0,
        load_noise_std_mw  =    800.0,
        dc_hub             = "Ashburn / Loudoun County VA (largest DC cluster on Earth)",
        data_url           = "https://dataminer2.pjm.com/feed/hrl_load_metered",
        zone_csv           = "PJM_DOM",
        rmcp_base_usd      = 30.0,
        reg_need_base_mw   = 80.0,
        competitor_fill_rate = 0.75,
    ),
    # ── 3. CAISO PG&E (Bay Area) ──────────────────────────────────────────
    # Duck curve: large midday solar ramp → rapid dispatchable ramp-up at dusk.
    # Higher base LMP (gas-heavy); slope reflects non-linear duck-curve ramp.
    "caiso_pgae": MarketParams(
        name               = "CAISO",
        region             = "Bay Area / San Jose & Santa Clara",
        regulation_product = "REGU / REGD (Regulation Up/Down)",
        regd_rho           = 0.9960,   # slightly more volatile due to solar variability
        regd_sigma         = 0.025,
        window_ticks       = 180,
        lmp_base_usd       = 48.0,
        lmp_slope_usd      = 28.0,     # duck curve creates steep non-linearity; peak ≈ $188
        load_mean_mw       = 26_000.0,
        load_daily_amp_mw  =  6_000.0, # strong solar duck curve
        load_noise_std_mw  =  1_500.0,
        dc_hub             = "Santa Clara / San Jose (Silicon Valley AI cluster)",
        data_url           = "https://oasis.caiso.com/mrioasis/logon.do",
        zone_csv           = "CAISO_PGAE",
        rmcp_base_usd      = 40.0,
        reg_need_base_mw   = 60.0,
        competitor_fill_rate = 0.65,
    ),
    # ── 4. ERCOT North (Dallas–Fort Worth) ────────────────────────────────
    # Isolated grid (no interstate interconnects) → extreme price spikes.
    # Winter Storm Uri hit $9,000/MWh cap. Fast-growing DC market.
    "ercot_north": MarketParams(
        name               = "ERCOT",
        region             = "Dallas / Fort Worth",
        regulation_product = "ECRS (Electricity Contingency Reserve Service)",
        regd_rho           = 0.9950,   # more volatile; isolated grid
        regd_sigma         = 0.030,
        window_ticks       = 180,
        lmp_base_usd       = 28.0,
        lmp_slope_usd      = 55.0,     # isolated grid → very steep; peak >> $200 (clips in obs)
        load_mean_mw       = 42_000.0,
        load_daily_amp_mw  = 12_000.0, # extreme AC load in Texas summers
        load_noise_std_mw  =  2_000.0,
        dc_hub             = "Dallas–Fort Worth / San Antonio (fastest-growing US DC market)",
        data_url           = "https://www.ercot.com/gridinfo/load/load_hist",
        zone_csv           = "ERCOT_NORTH",
        rmcp_base_usd      = 20.0,
        reg_need_base_mw   = 50.0,
        competitor_fill_rate = 0.60,
    ),
    # ── 5. ENTSO-E Germany (Frankfurt hub) ────────────────────────────────
    # FCR-N: responds to frequency deviation, not AGC dispatch → smoother signal.
    # 30-min EPEX settlement; deep interconnected market suppresses price spikes.
    # High renewable penetration (offshore wind in North Sea) → negative prices.
    "entso_de": MarketParams(
        name               = "ENTSO-E / EPEX",
        region             = "Germany (Frankfurt)",
        regulation_product = "FCR-N (Frequency Containment Reserve — Normal)",
        regd_rho           = 0.9980,   # FCR-N is smoother / more persistent
        regd_sigma         = 0.015,
        window_ticks       = 360,      # 30-min EPEX settlement interval
        lmp_base_usd       = 60.0,     # ~EUR 55 × 1.10 USD/EUR
        lmp_slope_usd      = 8.0,      # deep interconnected market; peak ≈ $156/MWh
        load_mean_mw       = 58_000.0,
        load_daily_amp_mw  = 10_000.0,
        load_noise_std_mw  =  3_000.0,
        dc_hub             = "Frankfurt (DE-CIX — largest internet exchange by traffic)",
        data_url           = "https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show",
        zone_csv           = "ENTSOE_DE",
        rmcp_base_usd      = 35.0,
        reg_need_base_mw   = 100.0,
        competitor_fill_rate = 0.80,
    ),
    # ── 6. AEMO New South Wales (Sydney) ──────────────────────────────────
    # 5-min dispatch intervals (NEM); South Australia has reached 100% renewable.
    # Extreme price events: AUD $15,500/MWh cap; negative prices are common.
    "aemo_nsw": MarketParams(
        name               = "AEMO / NEM",
        region             = "New South Wales / Sydney",
        regulation_product = "Regulation FCAS (Frequency Control Ancillary Service)",
        regd_rho           = 0.9945,   # 5-min dispatch → faster decorrelation
        regd_sigma         = 0.032,
        window_ticks       = 60,       # 5-min NEM dispatch interval
        lmp_base_usd       = 55.0,     # AUD ~$80 × 0.67 USD
        lmp_slope_usd      = 45.0,     # extreme events; peak ≈ $190/MWh (clips)
        load_mean_mw       =  8_200.0,
        load_daily_amp_mw  =  1_500.0,
        load_noise_std_mw  =    400.0,
        dc_hub             = "Western Sydney Aerotropolis (AUD $1B investment zone)",
        data_url           = "https://aemo.com.au/en/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data",
        zone_csv           = "AEMO_NSW",
        rmcp_base_usd      = 30.0,
        reg_need_base_mw   = 30.0,
        competitor_fill_rate = 0.65,
    ),
}



class MacroGridSignal:
    """
    Generates real-time frequency regulation and LMP price signals.

    Parameters
    ----------
    energy_dir:
        Directory with NYISO processed load CSVs (one per zone).
        Defaults to ``data/processed/energy`` relative to cwd.
    zone:
        Energy CSV stem to use for load data.  If None (default), uses the
        ``zone_csv`` field from the market preset.  Falls back to synthetic
        if the CSV does not exist.
    dt_seconds:
        Timestep in seconds.  Must match NYISO data resolution (300 s).
    committed_mw:
        Default regulation capacity committed to the grid operator [MW].
        The high-level agent overrides this each 15-minute interval.
    lmp_base_usd:
        Off-peak base LMP ($/MWh).  Calibrated to NYISO NYC historical mean.
    lmp_slope:
        Marginal price sensitivity above median load ($/MWh per GW).
    seed:
        RNG seed for the RegD signal generator.
    """

    def __init__(
        self,
        energy_dir: str | Path = "data/processed/energy",
        zone: str | None = None,
        dt_seconds: float = 300.0,
        committed_mw: float = 20.0,
        lmp_base_usd: float | None = None,
        lmp_slope: float | None = None,
        seed: int = 42,
        market: str = "nyiso_nyc",
        timestamp_replay: bool = False,
        start_day: int = 0,
        episode_ticks: int = 17280,
        regd_load_bias: float = 0.0,
    ) -> None:
        # Resolve market preset — explicit lmp_* args override the preset
        if market not in MARKET_PRESETS:
            raise ValueError(
                f"Unknown market {market!r}. "
                f"Available: {list(MARKET_PRESETS.keys())}"
            )
        self._market: MarketParams = MARKET_PRESETS[market]

        self.dt = dt_seconds
        self.committed_mw = committed_mw
        self.lmp_base_usd = lmp_base_usd if lmp_base_usd is not None else self._market.lmp_base_usd
        self.lmp_slope    = lmp_slope    if lmp_slope    is not None else self._market.lmp_slope_usd
        self._rng = np.random.default_rng(seed)
        self._timestamp_replay = timestamp_replay
        self._reset_signal_rngs(seed)
        self._calendar_load_mw: np.ndarray | None = None
        self._regd_bias = float(regd_load_bias)
        if not np.isfinite(self._regd_bias):
            raise ValueError("regd_load_bias must be finite")

        # Resolve zone: explicit arg > market preset default > "NYC"
        effective_zone = zone or self._market.zone_csv or "NYC"
        self._load_mw, self._load_median = self._load_zone_or_synthetic(
            Path(energy_dir), effective_zone, seed
        )
        self._n = len(self._load_mw)
        self._load_max = float(self._load_mw.max())
        if timestamp_replay:
            from c2g_env.replay import sample_previous

            path = Path(energy_dir) / f"{effective_zone}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Timestamp replay requires real load data: {path}")
            frame = pd.read_csv(path, parse_dates=["Time Stamp"])
            frame = frame.dropna(subset=["Load"]).sort_values("Time Stamp")
            self.calendar_start = frame["Time Stamp"].iloc[0].normalize() + pd.Timedelta(days=start_day)
            self.calendar_timestamps = (
                self.calendar_start.to_datetime64()
                + np.arange(episode_ticks + 1) * np.timedelta64(round(self.dt * 1e9), "ns")
            )
            self._calendar_load_mw = sample_previous(
                frame["Time Stamp"].to_numpy(), frame["Load"].to_numpy(),
                self.calendar_timestamps, max_age_seconds=600.0,
            )
            day_load = self._calendar_load_mw[:episode_ticks]
            mean, peak = day_load.mean(), day_load.max()
            self._load_anomaly = np.clip(
                (self._calendar_load_mw - mean) / max(peak - mean, 1e-9), -1.0, 1.0,
            )

        # AR(1) parameters from market preset
        self._regd_rho:   float = self._market.regd_rho
        self._regd_sigma: float = self._market.regd_sigma
        self._regd_state: float = 0.0

        # Energy-neutrality integrator (window = market settlement interval)
        self._regd_buffer: list[float] = []
        self._window_ticks: int = self._market.window_ticks

        # ── Grid frequency model (simplified swing equation) ──────────
        # f(t+1) = f(t) + dt/H * (ΔP_supply − ΔP_demand) + damping + noise
        # Nominal 60 Hz (US/ERCOT/AEMO) or 50 Hz (ENTSO-E)
        # H = inertia constant in seconds; D = load-damping coefficient
        self._f_nom: float = 50.0 if market == "entso_de" else 60.0
        self._f_grid: float = self._f_nom      # current frequency [Hz]
        self._H_inertia: float = 6.0            # system inertia [s] (typical 4-8s)
        self._D_damping: float = 1.5            # load-damping coefficient [pu]
        self._f_noise_std: float = 0.005         # stochastic mismatch noise [Hz]

        self._tick: int = 0

        # ── Market handshake state (RMCP + regulation need) ──────────
        self._rmcp: float = self._market.rmcp_base_usd
        self._reg_need: float = self._market.reg_need_base_mw
        self._residual: float = (
            self._reg_need * (1.0 - self._market.competitor_fill_rate)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, committed_mw: float | None = None) -> dict[str, float]:
        """
        Advance one timestep and return the grid signals.

        Parameters
        ----------
        committed_mw:
            Override the committed regulation capacity for this step [MW].
            If None, uses the value set at construction / last ``set_committed``.

        Returns
        -------
        dict with keys:
            delta_p_kw      : Regulation dispatch command [kW].
                              Positive = grid asks DC to *increase* consumption.
                              Negative = grid asks DC to *reduce* consumption.
            committed_mw    : Effective committed capacity this step [MW].
            lmp_usd_mwh     : Wholesale LMP at PCC [$/MWh].
            grid_load_mw    : Regional zone load [MW] (grid stress indicator).
            load_norm       : Grid load as fraction of historical max [0, 1].
            regd_signal     : Normalised RegD signal [-1, 1].
            tick            : Current tick index.
        """
        if committed_mw is not None:
            self.committed_mw = float(committed_mw)

        idx = self._tick % self._n
        load_mw = self.load_at_tick(self._tick)

        regd = self._step_regd()
        if self._timestamp_replay:
            regd = float(np.clip(regd + self._regd_bias * self._load_anomaly[self._tick], -1, 1))
        delta_p_kw = regd * self.committed_mw * 1_000.0   # MW → kW

        lmp = self._load_to_lmp(load_mw)
        load_norm = load_mw / (self._load_max if self._timestamp_replay else float(self._load_mw.max()))

        self._tick += 1
        return {
            "delta_p_kw":    delta_p_kw,
            "committed_mw":  self.committed_mw,
            "lmp_usd_mwh":   lmp,
            "grid_load_mw":  load_mw,
            "load_norm":     load_norm,
            "regd_signal":   regd,
            "f_grid_hz":     self._f_grid,
            "f_nom_hz":      self._f_nom,
            "tick":          idx,
        }

    def reset(self, seed: int | None = None) -> None:
        """Reset to the beginning of the dataset and clear RegD state."""
        self._tick = 0
        self._regd_state = 0.0
        self._regd_buffer = []
        self._f_grid = self._f_nom
        self._rmcp = self._market.rmcp_base_usd
        self._reg_need = self._market.reg_need_base_mw
        self._residual = (
            self._reg_need * (1.0 - self._market.competitor_fill_rate)
        )
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._reset_signal_rngs(seed)

    def _reset_signal_rngs(self, seed: int) -> None:
        streams = np.random.SeedSequence(seed).spawn(4)
        self._regd_rng, self._market_rng, self._bid_rng, self._frequency_rng = (
            np.random.default_rng(stream) for stream in streams
        )

    def load_at_tick(self, tick: int) -> float:
        if self._calendar_load_mw is not None:
            if tick < 0 or tick >= len(self._calendar_load_mw):
                raise ValueError(f"Tick {tick} is outside the configured replay horizon")
            return float(self._calendar_load_mw[tick])
        return float(self._load_mw[tick % self._n])

    def set_committed(self, committed_mw: float) -> None:
        """Update the committed regulation capacity [MW]."""
        self.committed_mw = float(np.clip(committed_mw, 0.0, 50.0))

    # ------------------------------------------------------------------
    # Market handshake: RMCP + bid clearing
    # ------------------------------------------------------------------

    def step_rmcp(self) -> dict[str, float]:
        """
        Compute the Regulation Market Clearing Price and residual need.

        Called once per macro interval (15-min) *before* the agent bids.
        The grid posts RMCP and regulation need; the agent observes these
        and decides how much MW to offer and at what price.

        Returns
        -------
        dict with keys:
            rmcp_usd     : Regulation Market Clearing Price [$/MWh].
            reg_need_mw  : Total regulation requirement [MW].
            residual_mw  : Residual need after competitor fills [MW].
            lmp_usd_mwh  : Current LMP [$/MWh] (for reference).
            load_norm    : Current normalised grid load [0, 1].
        """
        m = self._market
        tick = max(0, self._tick - 1) if self._timestamp_replay else self._tick
        load_mw = self.load_at_tick(tick)
        lmp = self._load_to_lmp(load_mw)
        load_norm = load_mw / (self._load_max if self._timestamp_replay else float(self._load_mw.max()))

        rng = self._market_rng if self._timestamp_replay else self._rng
        noise = float(rng.normal(0.0, m.rmcp_noise_std))
        self._rmcp = m.rmcp_base_usd + m.rmcp_lmp_beta * (lmp - m.lmp_base_usd) + noise
        self._rmcp = float(np.clip(self._rmcp, 0.0, 500.0))

        self._reg_need = m.reg_need_base_mw * (1.0 + m.reg_need_slope * load_norm)
        self._residual = self._reg_need * (1.0 - m.competitor_fill_rate)

        return {
            "rmcp_usd":    self._rmcp,
            "reg_need_mw": self._reg_need,
            "residual_mw": self._residual,
            "lmp_usd_mwh": lmp,
            "load_norm":   load_norm,
        }

    def clear_bid(self, bid_price: float, bid_mw: float) -> dict[str, float]:
        """
        Probabilistic bid acceptance using sigmoid price curve × quantity factor.

        Called once per macro interval *after* ``step_rmcp()`` and the agent's bid.

        Parameters
        ----------
        bid_price : float
            Agent's asking price [$/MWh].
        bid_mw : float
            Agent's offered regulation capacity [MW].

        Returns
        -------
        dict with keys:
            accepted       : bool — whether the bid was accepted.
            accepted_mw    : float — MW accepted (bid_mw if accepted, else 0).
            accept_prob    : float — computed acceptance probability.
            clearing_price : float — the RMCP at which the bid clears.
        """
        k_p = 0.15  # sigmoid temperature
        price_prob = 1.0 / (1.0 + np.exp(-k_p * (self._rmcp - bid_price)))
        quantity_factor = min(1.0, self._residual / max(bid_mw, 0.01))
        accept_prob = float(price_prob * quantity_factor)
        rng = self._bid_rng if self._timestamp_replay else self._rng
        accepted = bool(rng.random() < accept_prob)
        accepted_mw = bid_mw if accepted else 0.0
        return {
            "accepted":       accepted,
            "accepted_mw":    accepted_mw,
            "accept_prob":    accept_prob,
            "clearing_price": self._rmcp,
        }

    @property
    def horizon_ticks(self) -> int:
        """Total ticks in the loaded dataset before looping."""
        return self._n

    @property
    def lmp_stats(self) -> dict[str, float]:
        """Historical LMP statistics derived from zone load data."""
        lmps = np.array([self._load_to_lmp(l) for l in self._load_mw])
        return {
            "mean":  float(lmps.mean()),
            "std":   float(lmps.std()),
            "p95":   float(np.percentile(lmps, 95)),
            "max":   float(lmps.max()),
        }

    # ------------------------------------------------------------------
    # RegD signal generator
    # ------------------------------------------------------------------

    def _step_regd(self) -> float:
        """
        AR(1) process producing a PJM RegD-like normalised signal in [-1, 1].

        Energy neutrality is enforced with a mean-correction over rolling
        15-minute windows, matching PJM's requirement that RegD integrates
        to approximately zero over any 15-minute settlement period.
        """
        rng = self._regd_rng if self._timestamp_replay else self._rng
        noise = float(rng.normal(0.0, self._regd_sigma))
        self._regd_state = self._regd_rho * self._regd_state + noise

        # Soft clip: tanh keeps the signal smooth near ±1
        raw = float(np.tanh(self._regd_state))

        # Accumulate into the energy-neutrality buffer
        self._regd_buffer.append(raw)
        if len(self._regd_buffer) >= self._window_ticks:
            # Correct for accumulated drift
            drift = np.mean(self._regd_buffer)
            raw_corrected = raw - drift
            self._regd_buffer = []
        else:
            raw_corrected = raw

        return float(np.clip(raw_corrected, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Grid frequency model (swing equation approximation)
    # ------------------------------------------------------------------

    def _step_frequency(self, tracking_deficit_mw: float = 0.0) -> float:
        """
        Simplified swing-equation frequency model.

        When the datacenter fails to track the RegD signal (tracking_deficit_mw > 0),
        it contributes to a supply-demand imbalance that drives frequency down.
        Good tracking stabilises the grid.

        Parameters
        ----------
        tracking_deficit_mw : float
            Unsigned tracking error in MW.  Positive means the DC delivered
            *less* than demanded (i.e., net load is higher than scheduled).

        Returns
        -------
        float
            Grid frequency in Hz.
        """
        # Supply-demand imbalance due to DC tracking error
        # Negative ΔP means excess demand → frequency drops
        delta_p_pu = -tracking_deficit_mw / max(self._load_median, 1.0)

        # Stochastic background grid noise (other generators, load fluctuations)
        rng = self._frequency_rng if self._timestamp_replay else self._rng
        noise = float(rng.normal(0.0, self._f_noise_std))

        # Swing equation: df/dt = (1/2H) * (ΔP - D * Δf)
        delta_f = self._f_grid - self._f_nom
        df_dt = (1.0 / (2.0 * self._H_inertia)) * (
            delta_p_pu - self._D_damping * (delta_f / self._f_nom)
        )
        self._f_grid += df_dt * self.dt + noise

        # Hard clip to realistic bounds
        #   UFLS: under-frequency load shedding at 59.5 Hz (US) / 49.5 Hz (EU)
        #   OFGT: over-frequency at 60.5 Hz / 50.5 Hz
        f_min = self._f_nom - 0.5
        f_max = self._f_nom + 0.5
        self._f_grid = float(np.clip(self._f_grid, f_min, f_max))
        return self._f_grid

    @property
    def f_nom(self) -> float:
        """Nominal grid frequency [Hz]."""
        return self._f_nom

    @property
    def f_grid(self) -> float:
        """Current grid frequency [Hz]."""
        return self._f_grid

        # ------------------------------------------------------------------
    # LMP proxy from zone load
    # ------------------------------------------------------------------

    def _load_to_lmp(self, load_mw: float) -> float:
        """
        Non-linear LMP proxy: flat below median load, rising steeply above.

        LMP(t) = lmp_base + lmp_slope × max(0, Load_t - Load_median) / 1000
                                                                  [MW → GW]
        The division by 1000 converts the MW excess to GW for the slope units.
        """
        excess_gw = max(0.0, load_mw - self._load_median) / 1_000.0
        lmp = self.lmp_base_usd + self.lmp_slope * excess_gw
        return float(np.clip(lmp, 0.0, 500.0))   # cap at $500/MWh (spike limit)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_zone_or_synthetic(
        self, energy_dir: Path, zone: str, seed: int
    ) -> tuple[np.ndarray, float]:
        """
        Load zone load data from CSV; fall back to calibrated synthetic load
        if the file does not exist.

        Synthetic model (when real data unavailable):
            L(t) = L_mean + L_amp * sin(2π * h / 24 + π)     (daily cycle)
                         + σ_noise * ε(t)                      (AR(1) noise)
        where h is the hour-of-day.  Parameters come from the market preset so
        implied LMP peaks are statistically realistic for that ISO.
        """
        path = energy_dir / f"{zone}.csv"
        if path.exists():
            return self._load_zone(path)

        import warnings
        warnings.warn(
            f"MacroGridSignal: load file not found: {path}. "
            f"Using calibrated synthetic load for market '{self._market.name}' "
            f"({self._market.region}). "
            f"Download real data from: {self._market.data_url}",
            stacklevel=3,
        )
        return self._synthetic_load(seed)

    def _synthetic_load(self, seed: int) -> tuple[np.ndarray, float]:
        """Generate one year of synthetic 5-minute load using market calibration."""
        rng  = np.random.default_rng(seed + 999)
        m    = self._market
        # 365 × 24 × 12 = 105 120 five-minute ticks
        n    = 105_120
        t    = np.arange(n, dtype=np.float64)
        h    = (t * 5 / 60) % 24          # hour of day (fractional)
        # Daily cycle: peak in late afternoon, trough at 4 am
        daily = m.load_daily_amp_mw * np.sin(2 * np.pi * h / 24 - np.pi * 0.5)
        # AR(1) noise (ρ=0.95 at 5-min for load process, different from signal)
        rho_load, sigma_load = 0.97, m.load_noise_std_mw * np.sqrt(1 - 0.97**2)
        noise_arr = np.zeros(n)
        s = 0.0
        for i in range(n):
            s = rho_load * s + rng.normal(0, sigma_load)
            noise_arr[i] = s
        loads = np.maximum(m.load_mean_mw * 0.3, m.load_mean_mw + daily + noise_arr)
        return loads, float(np.median(loads))

    @staticmethod
    def _load_zone(path: Path) -> tuple[np.ndarray, float]:
        """
        Load zone load CSV and return (load_array_mw, median_mw).

        Expected columns: ``Time Stamp``, ``Load``  (NYISO OASIS format)
        """
        df = pd.read_csv(path, parse_dates=["Time Stamp"])
        if "Load" not in df.columns:
            raise ValueError(f"{path}: expected column 'Load', got {list(df.columns)}")
        df = df.dropna(subset=["Load"]).sort_values("Time Stamp").reset_index(drop=True)
        loads = df["Load"].values.astype(np.float64)
        return loads, float(np.median(loads))
