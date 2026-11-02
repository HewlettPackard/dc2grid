"""
Download and preprocess NOAA ISD-Lite hourly weather data for NYISO-zone proxies.

Source:
https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database

Usage:
    python preprocessing/weather/download_noaa_isd.py --start-year 2023 --end-year 2026

Outputs:
    data/raw/weather/noaa_isd_lite/<ZONE>/<USAF-WBAN-YYYY>.gz
    data/processed/weather/<ZONE>.csv
    data/processed/weather/noaa_isd_zones_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

BASE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
RAW_ROOT = "data/raw/weather/noaa_isd_lite"
PROCESSED_ROOT = "data/processed/weather"


@dataclass(frozen=True)
class Station:
    zone: str
    usaf: str
    wban: str
    name: str

    @property
    def station_id(self) -> str:
        return f"{self.usaf}-{self.wban}"


DEFAULT_STATIONS: Dict[str, Station] = {
    "CAPITL": Station("CAPITL", "725180", "14735", "Albany Intl"),
    "CENTRL": Station("CENTRL", "725190", "14771", "Syracuse Hancock"),
    "DUNWOD": Station("DUNWOD", "725037", "94745", "Westchester County"),
    "GENESE": Station("GENESE", "725290", "14768", "Rochester Greater"),
    "HUD_VL": Station("HUD_VL", "725036", "14757", "Poughkeepsie"),
    "LONGIL": Station("LONGIL", "744860", "94789", "JFK Intl"),
    "MHK_VL": Station("MHK_VL", "725030", "14732", "LaGuardia"),
    "MILLWD": Station("MILLWD", "725037", "94745", "Westchester County"),
    "NORTH": Station("NORTH", "726227", "94790", "Watertown"),
    "NYC": Station("NYC", "725053", "94728", "Central Park"),
    "WEST": Station("WEST", "725280", "14733", "Buffalo Niagara"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA ISD-Lite weather data")
    parser.add_argument("--start-year", type=int, required=True, help="First year (inclusive)")
    parser.add_argument("--end-year", type=int, required=True, help="Last year (inclusive)")
    parser.add_argument(
        "--zones",
        type=str,
        default="",
        help="Optional comma-separated zone subset, e.g., NYC,LONGIL,CAPITL",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="HTTP timeout per file",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    os.makedirs(RAW_ROOT, exist_ok=True)
    os.makedirs(PROCESSED_ROOT, exist_ok=True)


def select_stations(zones_arg: str) -> List[Station]:
    if not zones_arg.strip():
        return list(DEFAULT_STATIONS.values())

    selected: List[Station] = []
    requested = [z.strip().upper() for z in zones_arg.split(",") if z.strip()]
    missing = [z for z in requested if z not in DEFAULT_STATIONS]
    if missing:
        raise ValueError(f"Unknown zones: {missing}. Valid zones: {sorted(DEFAULT_STATIONS)}")

    for zone in requested:
        selected.append(DEFAULT_STATIONS[zone])
    return selected


def build_url(station: Station, year: int) -> str:
    return f"{BASE_URL}/{year}/{station.station_id}-{year}.gz"


def raw_file_path(station: Station, year: int) -> str:
    zone_dir = os.path.join(RAW_ROOT, station.zone)
    os.makedirs(zone_dir, exist_ok=True)
    return os.path.join(zone_dir, f"{station.station_id}-{year}.gz")


def download_file(url: str, out_path: str, timeout_seconds: int) -> bool:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            data = response.read()
        with open(out_path, "wb") as f:
            f.write(data)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[WARN] Missing file (404): {url}")
            return False
        print(f"[WARN] HTTP error for {url}: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Download failed for {url}: {e}")
        return False


def parse_isd_lite_gz(gz_path: str) -> Iterable[Tuple[datetime, float, float]]:
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            air_temp_tenth_c = int(parts[4])
            dew_temp_tenth_c = int(parts[5])

            if air_temp_tenth_c == -9999:
                continue

            ts = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
            air_temp_c = air_temp_tenth_c / 10.0
            dew_temp_c = dew_temp_tenth_c / 10.0 if dew_temp_tenth_c != -9999 else float("nan")
            yield ts, air_temp_c, dew_temp_c


def write_zone_csv(station: Station, rows: List[Tuple[datetime, float, float]]) -> str:
    out_path = os.path.join(PROCESSED_ROOT, f"{station.zone}.csv")
    rows.sort(key=lambda x: x[0])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_utc",
            "zone",
            "station_id",
            "station_name",
            "temp_c",
            "dewpoint_c",
        ])
        for ts, temp_c, dew_c in rows:
            writer.writerow([ts.isoformat(), station.zone, station.station_id, station.name, temp_c, dew_c])

    return out_path


def write_merged_csv(all_rows: List[Tuple[str, datetime, str, str, float, float]]) -> str:
    out_path = os.path.join(PROCESSED_ROOT, "noaa_isd_zones_merged.csv")
    all_rows.sort(key=lambda x: (x[1], x[0]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_utc",
            "zone",
            "station_id",
            "station_name",
            "temp_c",
            "dewpoint_c",
        ])
        for zone, ts, station_id, station_name, temp_c, dew_c in all_rows:
            writer.writerow([ts.isoformat(), zone, station_id, station_name, temp_c, dew_c])

    return out_path


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year")

    ensure_dirs()
    stations = select_stations(args.zones)

    print(f"Downloading NOAA ISD-Lite for years {args.start_year}..{args.end_year}")
    print(f"Zones: {[s.zone for s in stations]}")

    merged_rows: List[Tuple[str, datetime, str, str, float, float]] = []

    for station in stations:
        zone_rows: List[Tuple[datetime, float, float]] = []
        print(f"\n[ZONE] {station.zone} ({station.station_id}, {station.name})")

        for year in range(args.start_year, args.end_year + 1):
            url = build_url(station, year)
            gz_path = raw_file_path(station, year)
            ok = download_file(url, gz_path, timeout_seconds=args.timeout_seconds)
            if not ok:
                continue

            count_before = len(zone_rows)
            for ts, temp_c, dew_c in parse_isd_lite_gz(gz_path):
                zone_rows.append((ts, temp_c, dew_c))
            count_added = len(zone_rows) - count_before
            print(f"  [OK] {year}: +{count_added} rows")

        if not zone_rows:
            print(f"  [WARN] No data for zone {station.zone}")
            continue

        out_zone = write_zone_csv(station, zone_rows)
        print(f"  [SAVE] {out_zone} ({len(zone_rows)} rows)")

        for ts, temp_c, dew_c in zone_rows:
            merged_rows.append((station.zone, ts, station.station_id, station.name, temp_c, dew_c))

    if merged_rows:
        out_merged = write_merged_csv(merged_rows)
        print(f"\n[SAVE] {out_merged} ({len(merged_rows)} rows)")
    else:
        print("\n[WARN] No rows downloaded. Check station IDs or year range.")


if __name__ == "__main__":
    main()
