# c2g_env/physics/weather.py
#
# NOAA ISD weather loader for C2G-Bench.
#
# Reads hourly dry-bulb temperature (and dew-point) from preprocessed
# NOAA Integrated Surface Database CSVs in ``data/processed/weather/``.
#
# Physical effects in ThermalTwin (already parameterized):
#   Zone A:  T_eq incorporates K_env_A * T_amb  → envelope heat gain/loss
#   Zone B:  same K_env_B * T_amb coupling
#            COP = COP_base * max(0.3, 1 - COP_alpha * (T_amb - 25))
#                → warmer days require more fan energy for the same cooling output
#
# References
# ----------
# [1] Smith, A., Lott, J.N., Vose, R. (2011) "The Integrated Surface Database:
#     Recent Developments and Partnerships," Bulletin of the American
#     Meteorological Society, 92(6), 704–708.
#     DOI: 10.1175/2011BAMS3015.1  — NOAA ISD station archive; describes
#     quality-control flags applied during preprocessing.
# [2] Parton, W.J., Logan, J.A. (1981) "A model for diurnal variation in
#     soil and air temperature," Agricultural Meteorology, 23, 205–216.
#     DOI: 10.1016/0002-1571(81)90105-9  — truncated-sine + exponential
#     diurnal model; basis for the synthetic temperature fallback.
# [3] ASHRAE (2021) "ASHRAE Handbook — Fundamentals," Ch. 14: Climatic
#     Design Information. ASHRAE, Atlanta.
#     https://www.ashrae.org/technical-resources/ashrae-handbook
#     — source for annual_mean_c / annual_amp_c per data-centre location.
# [4] Lee, K.P., Chen, H.L. (2013) "Analysis of energy saving potential of
#     air-side free cooling for data centers in worldwide climate zones,"
#     Energy and Buildings, 64, 103–112.
#     DOI: 10.1016/j.enbuild.2013.04.019  — quantifies how outdoor dry-bulb
#     T_amb governs free-cooling availability and chiller COP, motivating
#     the weather-driven COP degradation term in ThermalTwin.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-market weather calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherParams:
    """
    Climate calibration for one data-centre location.

    Parameters
    ----------
    location_name : str
        Human-readable name (e.g. "New York City (Central Park)").
    zone_file : str
        CSV filename inside ``weather_dir`` (without extension).
        Set to ``None`` if no real data is available; synthetic model is used.
    t_amb_fallback_c : float
        Constant fallback if both real data and synthetic model fail.
    annual_mean_c : float
        Annual mean temperature (°C) — centre of the seasonal sinusoid.
    annual_amp_c : float
        Half-range of the seasonal cycle (°C).
        Temperature swings from (annual_mean - annual_amp) in winter to
        (annual_mean + annual_amp) in summer.
    daily_amp_c : float
        Half-range of the diurnal cycle (°C).
    norm_min_c : float
        Lower bound for observation normalization (°C).
    norm_max_c : float
        Upper bound for observation normalization (°C).
    data_url : str
        Download URL for real weather data.
    """
    location_name:    str
    zone_file:        str | None
    t_amb_fallback_c: float
    annual_mean_c:    float
    annual_amp_c:     float
    daily_amp_c:      float
    norm_min_c:       float
    norm_max_c:       float
    data_url:         str


#: Registry of weather calibrations.  Keys match ``MARKET_PRESETS`` in macro_grid.
WEATHER_PRESETS: dict[str, WeatherParams] = {
    # ── 1. NYISO New York City ─────────────────────────────────────────────
    # Source: NOAA ASOS station 725053, Central Park (real data available)
    "nyiso_nyc": WeatherParams(
        location_name    = "New York City (Central Park, station 725053)",
        zone_file        = "NYC",      # data/processed/weather/NYC.csv
        t_amb_fallback_c = 25.0,
        annual_mean_c    = 13.0,       # NYC annual mean ≈ 13°C
        annual_amp_c     = 12.5,       # winter ~0°C, summer ~25°C
        daily_amp_c      =  5.0,
        norm_min_c       = -20.0,
        norm_max_c       =  42.0,
        data_url         = "https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database",
    ),
    # ── 2. PJM Northern Virginia (Ashburn) ────────────────────────────────
    # Closest public proxy: Reagan National (DCA) — hot humid summers.
    # No local CSV bundled; synthetic fallback used.
    "pjm_dom": WeatherParams(
        location_name    = "Northern Virginia / Ashburn (Reagan National proxy)",
        zone_file        = "DCA",      # data/processed/weather/DCA.csv (ERA5/ISD-Lite)
        t_amb_fallback_c = 26.0,
        annual_mean_c    = 15.0,       # DC annual mean ≈ 15°C
        annual_amp_c     = 13.0,       # winter ~2°C, summer ~28°C
        daily_amp_c      =  7.0,       # more continental than NYC
        norm_min_c       = -15.0,
        norm_max_c       =  42.0,
        data_url         = "https://www.ncei.noaa.gov/pub/data/noaa/ (station 724050, DCA)",
    ),
    # ── 3. CAISO Bay Area (San Jose) ─────────────────────────────────────
    # Mediterranean climate: mild winters, cool foggy summers.
    # Duck curve driven by solar, not heat load.
    "caiso_pgae": WeatherParams(
        location_name    = "San Jose / Santa Clara (San José Mineta Intl proxy)",
        zone_file        = "SJC",      # data/processed/weather/SJC.csv (ERA5/ISD-Lite)
        t_amb_fallback_c = 18.0,
        annual_mean_c    = 14.5,       # San Jose annual mean ≈ 14.5°C
        annual_amp_c     =  6.5,       # very mild: winter ~8°C, summer ~21°C
        daily_amp_c      =  8.0,       # large diurnal swing (marine layer)
        norm_min_c       =  -5.0,
        norm_max_c       =  42.0,      # occasional heat dome events
        data_url         = "https://www.ncei.noaa.gov/pub/data/noaa/ (station 724945, SJC)",
    ),
    # ── 4. ERCOT North Texas (Dallas–Fort Worth) ──────────────────────────
    # Hot humid summers (> 40°C); rare but extreme winter events (Uri).
    # Largest cooling load of any US market.
    "ercot_north": WeatherParams(
        location_name    = "Dallas–Fort Worth (DFW station 722590)",
        zone_file        = "DFW",      # data/processed/weather/DFW.csv (ERA5/ISD-Lite)
        t_amb_fallback_c = 29.0,
        annual_mean_c    = 19.5,       # DFW annual mean ≈ 19.5°C
        annual_amp_c     = 15.0,       # winter ~4°C, summer ~34°C
        daily_amp_c      =  9.0,
        norm_min_c       = -15.0,      # Winter Storm Uri extreme
        norm_max_c       =  46.0,      # July heat peaks
        data_url         = "https://www.ncei.noaa.gov/pub/data/noaa/ (station 722590, DFW)",
    ),
    # ── 5. ENTSO-E Germany (Frankfurt) ────────────────────────────────────
    # Temperate oceanic; increasingly hot summers. Negative price events
    # correlate with windy/cold periods (high wind, low load).
    "entso_de": WeatherParams(
        location_name    = "Frankfurt am Main (station 106370, FRA)",
        zone_file        = "FRA",      # data/processed/weather/FRA.csv (ERA5/ISD-Lite)
        t_amb_fallback_c = 12.0,
        annual_mean_c    = 11.0,       # Frankfurt annual mean ≈ 11°C
        annual_amp_c     = 10.5,       # winter ~0.5°C, summer ~21.5°C
        daily_amp_c      =  7.0,
        norm_min_c       = -20.0,
        norm_max_c       =  40.0,      # 2019/2022 European heat domes
        data_url         = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/ (DWD)",
    ),
    # ── 6. AEMO New South Wales (Sydney) ──────────────────────────────────
    # Southern hemisphere: southern summer = Jan/Feb.
    # Sea breeze moderates coast; Western Sydney (DC hub) is 5°C hotter.
    "aemo_nsw": WeatherParams(
        location_name    = "Western Sydney Aerotropolis (Bankstown proxy, station 953933)",
        zone_file        = "BKT",      # data/processed/weather/BKT.csv (ERA5/ISD-Lite)
        t_amb_fallback_c = 21.0,
        annual_mean_c    = 18.5,       # Western Sydney annual mean ≈ 18.5°C
        annual_amp_c     =  9.0,       # winter ~9.5°C, summer ~27.5°C (southern hemi)
        daily_amp_c      =  8.0,
        norm_min_c       =  -5.0,
        norm_max_c       =  46.0,      # Black Summer heat records 2019-20
        data_url         = "http://www.bom.gov.au/climate/data/ (Bureau of Meteorology)",
    ),
}

# Ticks per 1-hour weather record at 5-second resolution
_TICKS_PER_HOUR = 720   # 3600 s / 5 s


class WeatherLoader:
    """
    Loads hourly NOAA ISD weather data and exposes per-tick ambient temperature.

    Priority chain for data source:
        1. Real CSV from ``weather_dir/{zone_file}.csv``   (highest fidelity)
        2. Calibrated synthetic seasonal+diurnal model     (uses WeatherParams)
        3. Constant fallback temperature                   (last resort)

    Parameters
    ----------
    weather_dir : str or Path
        Directory containing zone CSVs.
    market : str
        Key into WEATHER_PRESETS (e.g. ``"nyiso_nyc"``).  Selects zone file
        and calibration parameters.  Defaults to ``"nyiso_nyc"``.
    zone : str or None
        Override the zone filename from the preset.  Use when the CSV has a
        custom name.
    dt_seconds : float
        Simulation timestep in seconds (default 5).
    fallback_temp_c : float or None
        Override the preset's constant fallback temperature.
    """

    def __init__(
        self,
        weather_dir: str | Path = "data/processed/weather",
        market: str = "nyiso_nyc",
        zone: str | None = None,
        dt_seconds: float = 5.0,
        fallback_temp_c: float | None = None,
        replay_timestamps: np.ndarray | None = None,
    ) -> None:
        if market not in WEATHER_PRESETS:
            raise ValueError(
                f"Unknown weather market {market!r}. "
                f"Available: {list(WEATHER_PRESETS.keys())}"
            )
        self._preset = WEATHER_PRESETS[market]
        self._ticks_per_hour: int = max(1, round(3600.0 / dt_seconds))
        self._fallback = float(
            fallback_temp_c
            if fallback_temp_c is not None
            else self._preset.t_amb_fallback_c
        )

        # Resolve zone filename: explicit arg > preset > None
        zone_file = zone if zone is not None else self._preset.zone_file
        weather_dir = Path(weather_dir)
        path = weather_dir / f"{zone_file}.csv" if zone_file else None

        self._temps_c:     np.ndarray | None = None
        self._dewpoints_c: np.ndarray | None = None
        self._n_hours:     int = 0
        self._source:      str = "fallback"
        self._replay_temps: np.ndarray | None = None
        self._replay_dewpoints: np.ndarray | None = None

        if path and path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp_utc"])
            df = (df.dropna(subset=["temp_c"])
                    .sort_values("timestamp_utc")
                    .reset_index(drop=True))
            self._temps_c     = df["temp_c"].values.astype(np.float64)
            self._dewpoints_c = (df["dewpoint_c"].values.astype(np.float64)
                                 if "dewpoint_c" in df.columns else None)
            self._n_hours = len(self._temps_c)
            self._market_id = market
            self._source  = f"real:{path.name}"
            if replay_timestamps is not None:
                from c2g_env.replay import sample_previous

                times = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_localize(None)
                source_year = times.iloc[0].year
                targets = np.array([
                    pd.Timestamp(t).replace(year=source_year).to_datetime64()
                    for t in replay_timestamps
                ])
                previous = times.iloc[-1].replace(year=source_year - 1).to_datetime64()
                extended_times = np.concatenate(([previous], times.to_numpy()))
                self._replay_temps = sample_previous(
                    extended_times, np.r_[self._temps_c[-1], self._temps_c],
                    targets, max_age_seconds=7200.0,
                )
                if self._dewpoints_c is not None and np.isfinite(self._dewpoints_c).all():
                    self._replay_dewpoints = sample_previous(
                        extended_times, np.r_[self._dewpoints_c[-1], self._dewpoints_c],
                        targets, max_age_seconds=7200.0,
                    )
                self._source += f":calendar_climatology_{source_year}"
        else:
            if replay_timestamps is not None:
                raise FileNotFoundError(f"Timestamp replay requires real weather data: {path}")
            # Calibrated synthetic model
            if path:
                warnings.warn(
                    f"WeatherLoader [{market}]: file not found: {path}. "
                    f"Using calibrated synthetic climate for "
                    f"'{self._preset.location_name}'. "
                    f"Download real data from: {self._preset.data_url}",
                    stacklevel=2,
                )
            self._market_id = market
            self._temps_c, self._dewpoints_c = self._make_synthetic()
            self._n_hours = len(self._temps_c)
            self._source  = f"synthetic:{market}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def temp_c(self, tick: int) -> float:
        """Return ambient dry-bulb temperature (°C) at simulation tick."""
        if self._replay_temps is not None:
            return float(self._replay_temps[tick])
        if self._temps_c is None:
            return self._fallback
        hour_idx = (tick // self._ticks_per_hour) % self._n_hours
        return float(self._temps_c[hour_idx])

    def dewpoint_c(self, tick: int) -> float:
        """Return dew-point temperature (°C) at simulation tick."""
        if self._replay_dewpoints is not None:
            return float(self._replay_dewpoints[tick])
        if self._dewpoints_c is None or self._temps_c is None:
            return self._fallback - 10.0
        hour_idx = (tick // self._ticks_per_hour) % self._n_hours
        return float(self._dewpoints_c[hour_idx])

    def temp_norm(self, tick: int) -> float:
        """Return T_amb normalized to [0, 1] using preset bounds."""
        t = self.temp_c(tick)
        lo, hi = self._preset.norm_min_c, self._preset.norm_max_c
        return float(np.clip((t - lo) / (hi - lo), 0.0, 1.0))

    @property
    def loaded(self) -> bool:
        """True if real CSV data is loaded."""
        return self._source.startswith("real:")

    @property
    def source(self) -> str:
        """String describing the data source: 'real:NYC.csv', 'synthetic:pjm_dom', or 'fallback'."""
        return self._source

    @property
    def market(self) -> str:
        """The market key this loader was constructed with."""
        return self._source.split(":")[-1] if ":" in self._source else "unknown"

    # ------------------------------------------------------------------
    # Synthetic climate model
    # ------------------------------------------------------------------

    def _make_synthetic(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate one leap-year of hourly temperature using a two-harmonic model:

            T(d, h) = T_mean
                    + A_annual * sin(2π * d/365 + φ_annual)     [seasonal]
                    + A_daily  * sin(2π * h/24  - π*0.5)        [diurnal: peak at 14h]

        where d is the day-of-year and h is the hour-of-day.
        The phase φ_annual is set so the summer peak falls in mid-July (day 196)
        for the northern hemisphere, or mid-January (day 15) for the southern
        hemisphere (detected by annual_mean_c / seasonal sign convention).

        Dew-point is approximated as T - depression, where the depression
        is 8°C in temperate climates and 5°C in humid regions.
        """
        p    = self._preset
        n    = 366 * 24   # one leap-year in hours
        days = np.arange(n, dtype=np.float64) / 24.0   # fractional day
        hrs  = np.arange(n, dtype=np.float64) % 24.0

        # seasonal peak: day 196 ≈ mid-July (NH); flip for SH (aemo)
        # Southern hemisphere detected by negative sign convention on amp
        # (both amps are positive; we flip phase for SH markets)
        southern_hemi = ("aemo" in self._market_id
                         or "nsw" in self._market_id
                         or "aemo" in str(self._preset.location_name).lower())
        phase_offset = -np.pi * 0.5 if not southern_hemi else np.pi * 0.5

        seasonal = p.annual_amp_c * np.sin(
            2 * np.pi * days / 365.25 + phase_offset
        )
        diurnal = p.daily_amp_c * np.sin(2 * np.pi * hrs / 24.0 - np.pi * 0.5)

        temps = p.annual_mean_c + seasonal + diurnal

        # Dew-point: simpler depression (warmer → more humid → smaller gap)
        depression = np.where(temps > 20, 6.0, 10.0)
        dewpoints  = temps - depression

        return temps.astype(np.float64), dewpoints.astype(np.float64)
