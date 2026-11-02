"""
scripts/download_weather.py
──────────────────────────────────────────────────────────────────────────────
Download hourly weather data for all C2G-Bench market locations using the
Open-Meteo Historical Weather API (ERA5 reanalysis).

  - Free, no API key required
  - Global coverage, gap-free ERA5 reanalysis
  - Temperature + dew-point at 2 m, hourly

Output files:  data/processed/weather/{ZONE}.csv
  Columns: timestamp_utc, zone, station_id, station_name, temp_c, dewpoint_c

Usage
─────
    python scripts/download_weather.py [--year 2024] [--markets all]
    python scripts/download_weather.py --markets pjm_dom ercot_north --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "data" / "processed" / "weather"
OUT.mkdir(parents=True, exist_ok=True)

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# ── Station / location registry ────────────────────────────────────────────
# Coordinates match NOAA ISD stations used in WeatherParams presets.
STATIONS: dict[str, dict] = {
    "nyiso_nyc": {
        "zone":    "NYC",
        "lat":      40.7789,
        "lon":     -73.9692,
        "name":    "New York City (Central Park)",
    },
    "pjm_dom": {
        "zone":    "DCA",
        "lat":      38.8521,
        "lon":     -77.0377,
        "name":    "Reagan National Airport, Washington DC",
    },
    "caiso_pgae": {
        "zone":    "SJC",
        "lat":      37.3626,
        "lon":    -121.9290,
        "name":    "San Jose Mineta Airport",
    },
    "ercot_north": {
        "zone":    "DFW",
        "lat":      32.8969,
        "lon":     -97.0380,
        "name":    "Dallas-Fort Worth Airport",
    },
    "entso_de": {
        "zone":    "FRA",
        "lat":      50.0331,
        "lon":       8.5706,
        "name":    "Frankfurt Rhein-Main Airport",
    },
    "aemo_nsw": {
        "zone":    "BKT",
        "lat":     -33.9244,
        "lon":     150.9880,
        "name":    "Sydney Bankstown Airport",
    },
}


def _get(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "C2G-Bench/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    attempt {attempt+1}/{retries} failed ({exc}); "
                  f"retrying in {wait}s …", flush=True)
            time.sleep(wait)
    print(f"ERROR: could not fetch {url}", file=sys.stderr)
    sys.exit(1)


def download_station(market: str, meta: dict, year: int,
                     force: bool = False) -> Path:
    zone     = meta["zone"]
    out_path = OUT / f"{zone}.csv"

    if out_path.exists() and not force:
        print(f"  [{market}] {out_path.name} already exists — "
              f"skip (--force to re-download)")
        return out_path

    params = urllib.parse.urlencode({
        "latitude":  meta["lat"],
        "longitude": meta["lon"],
        "start_date": f"{year}-01-01",
        "end_date":   f"{year}-12-31",
        "hourly":    "temperature_2m,dewpoint_2m",
        "timezone":  "UTC",
        "format":    "json",
    })
    url = f"{OPEN_METEO_BASE}?{params}"
    print(f"  [{market}] Downloading ERA5 reanalysis "
          f"{meta['name']} ({year}) …", end=" ", flush=True)
    raw  = _get(url)
    data = json.loads(raw)

    times    = data["hourly"]["time"]          # "YYYY-MM-DDTHH:00"
    temps    = data["hourly"]["temperature_2m"]
    dewpts   = data["hourly"]["dewpoint_2m"]

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(times, utc=True),
        "zone":          zone,
        "station_id":    f"era5_{meta['lat']}_{meta['lon']}",
        "station_name":  meta["name"],
        "temp_c":        temps,
        "dewpoint_c":    dewpts,
    })
    df = df.dropna(subset=["temp_c"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"{len(df):,} rows → {out_path.relative_to(ROOT)}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download ERA5 weather data for C2G-Bench (Open-Meteo)")
    ap.add_argument("--year",    type=int, default=2024)
    ap.add_argument("--markets", nargs="+", default=["all"],
                    choices=list(STATIONS) + ["all"])
    ap.add_argument("--force",   action="store_true",
                    help="Re-download even if file already exists")
    args    = ap.parse_args()
    targets = list(STATIONS) if args.markets == ["all"] else args.markets

    print(f"Downloading ERA5 weather data (year={args.year})")
    print(f"Source: Open-Meteo Historical API (no API key required)")
    print(f"Output: {OUT}\n")

    for mkt in targets:
        download_station(mkt, STATIONS[mkt], args.year, force=args.force)
        time.sleep(0.3)   # be polite to the Open-Meteo servers

    print("\nDone.")


if __name__ == "__main__":
    main()
