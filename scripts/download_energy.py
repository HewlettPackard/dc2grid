"""
scripts/download_energy.py
──────────────────────────────────────────────────────────────────────────────
Download actual grid load data for all C2G-Bench market presets.

Sources (all free, no API key required)
───────
  pjm_dom    – EIA Open Data API  (hourly, PJM respondent)
  caiso_pgae – EIA Open Data API  (hourly, CISO respondent)
  ercot_north– EIA Open Data API  (hourly, ERCO respondent)
  entso_de   – Open Power System Data (OPSD, hourly, DE actual load)
  aemo_nsw   – AEMO price & demand monthly CSVs (30-min, NSW1 region)
  nyiso_nyc  – NYISO OASIS public bulk CSVs (5-min, NYC zone)

Output files: data/processed/energy/{ZONE}.csv
  Columns: Time Stamp, Load   (5-minute intervals, MW)

Usage
─────
    python scripts/download_energy.py [--year 2024] [--markets all]
    python scripts/download_energy.py --markets pjm_dom ercot_north --force
    python scripts/download_energy.py --markets nyiso_nyc
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed" / "energy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EIA_BASE  = "https://api.eia.gov/v2/electricity/rto/region-data/data"
EIA_KEY   = "DEMO_KEY"   # free, 25 req/day; register at eia.gov for higher limits
SMARD_BASE = "https://www.smard.de/app/chart_data/410/DE"   # hourly total consumption
AEMO_BASE  = "https://aemo.com.au/aemo/data/nem/priceanddemand"
NYISO_BASE = "https://mis.nyiso.com/public/csv/pal"


# ── helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 5, timeout: int = 120) -> bytes:
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


def _to_5min(df: pd.DataFrame, ts_col: str, val_col: str,
             year: int) -> pd.DataFrame:
    """Upsample / resample any interval to 5-minute, forward-fill gaps."""
    sub = df[[ts_col, val_col]].copy()
    sub[ts_col] = pd.to_datetime(sub[ts_col], errors="coerce", utc=False)
    if sub[ts_col].dt.tz is not None:
        sub[ts_col] = sub[ts_col].dt.tz_localize(None)
    sub = sub.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:55", freq="5min")
    sub = sub.reindex(sub.index.union(idx)).sort_index().ffill().bfill().reindex(idx)
    sub.index.name = "Time Stamp"
    sub.columns    = ["Load"]
    return sub.reset_index()


def _save(df: pd.DataFrame, zone: str) -> Path:
    path = OUT_DIR / f"{zone}.csv"
    df.to_csv(path, index=False)
    print(f"  → {path.relative_to(ROOT)}  ({len(df):,} rows, "
          f"{path.stat().st_size // 1024} KB)")
    return path


# ── EIA  (PJM / ERCOT / CAISO) ───────────────────────────────────────────────

EIA_MARKETS = {
    "pjm_dom":     ("PJM",  "PJM_DOM"),
    "caiso_pgae":  ("CISO", "CAISO_PGAE"),
    "ercot_north": ("ERCO", "ERCOT_NORTH"),
}


def download_eia(market: str, year: int, force: bool = False) -> None:
    respondent, zone = EIA_MARKETS[market]
    out = OUT_DIR / f"{zone}.csv"
    if out.exists() and not force:
        print(f"  [{market}] {out.name} exists — skip"); return

    print(f"  [{market}] Downloading EIA hourly demand ({respondent}, {year}) …")

    # EIA v2 API returns max 5000 rows per call — page through the full year
    rows_all = []
    offset   = 0
    page     = 0
    while True:
        params = urllib.parse.urlencode({
            "api_key":                    EIA_KEY,
            "frequency":                  "hourly",
            "data[0]":                    "value",
            "facets[respondent][]":       respondent,
            "facets[type][]":             "D",
            "start":                      f"{year}-01-01T00",
            "end":                        f"{year}-12-31T23",
            "sort[0][column]":            "period",
            "sort[0][direction]":         "asc",
            "length":                     5000,
            "offset":                     offset,
        })
        raw  = _get(f"{EIA_BASE}?{params}")
        data = json.loads(raw)
        rows = data.get("response", {}).get("data", [])
        if not rows:
            break
        rows_all.extend(rows)
        page += 1
        print(f"    page {page}: {len(rows)} rows (total {len(rows_all)})", flush=True)
        if len(rows) < 5000:
            break
        offset += 5000
        time.sleep(0.2)

    df = pd.DataFrame(rows_all)
    # EIA columns: period (YYYY-MM-DDTHH), value (MWh)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    _save(_to_5min(df, "period", "value", year), zone)


# ── SMARD  (ENTSO-E Germany total consumption, Bundesnetzagentur) ────────────

def download_entso_de(year: int, force: bool = False) -> None:
    """Download German total consumption via SMARD.de hourly JSON API."""
    zone = "ENTSOE_DE"
    out  = OUT_DIR / f"{zone}.csv"
    if out.exists() and not force:
        print(f"  [entso_de] {out.name} exists — skip"); return

    print(f"  [entso_de] Downloading SMARD DE total consumption (ENTSO-E, {year}) …")

    # 1. Fetch weekly chunk index
    idx_url = f"{SMARD_BASE}/index_hour.json"
    idx     = json.loads(_get(idx_url))
    year_start = int(pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp() * 1000)
    year_end   = int(pd.Timestamp(f"{year+1}-01-01", tz="UTC").timestamp() * 1000)
    chunks     = [t for t in idx["timestamps"] if year_start <= t < year_end]
    # Also include last chunk before year_start in case its data spills into year
    all_ts = sorted(idx["timestamps"])
    pre_chunks = [t for t in all_ts if t < year_start]
    if pre_chunks:
        chunks = [pre_chunks[-1]] + chunks
    print(f"    Fetching {len(chunks)} weekly chunks …")

    rows = []
    for ts_ms in chunks:
        url  = f"{SMARD_BASE}/410_DE_hour_{ts_ms}.json"
        data = json.loads(_get(url, retries=4))
        rows.extend(data.get("series", []))
        time.sleep(0.15)

    df = pd.DataFrame(rows, columns=["ts_ms", "load_mw"])
    df = df.dropna(subset=["load_mw"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[df["ts"].dt.year == year].copy()
    print(f"    {len(df):,} hourly rows for {year}")
    _save(_to_5min(df, "ts", "load_mw", year), zone)


# ── AEMO  (NSW) ────────────────────────────────────────────────────────────────

def download_aemo(year: int, force: bool = False) -> None:
    zone = "AEMO_NSW"
    out  = OUT_DIR / f"{zone}.csv"
    if out.exists() and not force:
        print(f"  [aemo_nsw] {out.name} exists — skip"); return

    print(f"  [aemo_nsw] Downloading AEMO NSW demand (monthly CSVs, {year}) …")
    frames = []
    for month in range(1, 13):
        fname = f"PRICE_AND_DEMAND_{year}{month:02d}_NSW1.csv"
        url   = f"{AEMO_BASE}/{fname}"
        print(f"    {fname} …", end=" ", flush=True)
        raw  = _get(url, retries=3)
        df_m = pd.read_csv(io.BytesIO(raw))
        frames.append(df_m)
        print(f"{len(df_m)} rows")
        time.sleep(0.3)

    df_all = pd.concat(frames, ignore_index=True)
    # Columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE
    # Use TRADE periods only (actual 5-min settlement)
    if "PERIODTYPE" in df_all.columns:
        df_all = df_all[df_all["PERIODTYPE"] == "TRADE"]
    _save(_to_5min(df_all, "SETTLEMENTDATE", "TOTALDEMAND", year), zone)


# ── NYISO  (NYC zone, 5-minute actual load) ────────────────────────────────────────

def download_nyiso(year: int, force: bool = False) -> None:
    """Download NYISO 5-minute actual load (PAL) for the NYC zone.

    Data source: NYISO OASIS public bulk CSVs (no auth required).
    URL pattern: https://mis.nyiso.com/public/csv/pal/{YYYYMM}01pal_csv.zip
    Each monthly ZIP contains daily files:  {YYYYMMDD}pal.csv
    Columns: Time Stamp, Name, PTID, Load
    We filter Name == "N.Y.C." and save as data/processed/energy/NYC.csv.
    """
    zone = "NYC"
    out  = OUT_DIR / f"{zone}.csv"
    if out.exists() and not force:
        print(f"  [nyiso_nyc] {out.name} exists — skip"); return

    print(f"  [nyiso_nyc] Downloading NYISO PAL 5-min load (NYC zone, {year}) …")
    frames = []
    for month in range(1, 13):
        zip_name = f"{year}{month:02d}01pal_csv.zip"
        url      = f"{NYISO_BASE}/{zip_name}"
        print(f"    {zip_name} …", end=" ", flush=True)
        raw = _get(url, retries=3, timeout=120)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                with zf.open(name) as f:
                    df_day = pd.read_csv(f)
                    # Filter to NYC zone only
                    if "Name" in df_day.columns:
                        df_day = df_day[df_day["Name"] == "N.Y.C."]
                    elif "name" in df_day.columns:
                        df_day = df_day[df_day["name"].str.upper() == "N.Y.C."]
                    frames.append(df_day)
        print(f"{sum(len(f) for f in frames)} rows total", flush=True)
        time.sleep(0.3)

    df_all = pd.concat(frames, ignore_index=True)
    # Normalise column names (NYISO uses mixed case)
    df_all.columns = [c.strip() for c in df_all.columns]
    ts_col   = next(c for c in df_all.columns if "time" in c.lower() or "timestamp" in c.lower())
    load_col = next(c for c in df_all.columns if "load" in c.lower())
    print(f"    {len(df_all):,} 5-min rows for {year} (NYC zone)")
    _save(_to_5min(df_all, ts_col, load_col, year), zone)


# ── main ──────────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "pjm_dom":     lambda year, force: download_eia("pjm_dom",     year, force),
    "caiso_pgae":  lambda year, force: download_eia("caiso_pgae",  year, force),
    "ercot_north": lambda year, force: download_eia("ercot_north", year, force),
    "entso_de":    download_entso_de,
    "aemo_nsw":    download_aemo,
    "nyiso_nyc":   download_nyiso,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download grid load data for C2G-Bench")
    ap.add_argument("--year",    type=int, default=2024)
    ap.add_argument("--markets", nargs="+", default=["all"],
                    choices=list(DOWNLOADERS) + ["all"])
    ap.add_argument("--force",   action="store_true",
                    help="Re-download even if output CSV already exists")
    args    = ap.parse_args()
    targets = list(DOWNLOADERS) if args.markets == ["all"] else args.markets

    print(f"\nDownloading energy load data  year={args.year}")
    print(f"US markets: EIA Open Data API (api_key=DEMO_KEY, 25 req/day free)")
    print(f"Germany:    Open Power System Data (OPSD)")
    print(f"Australia:  AEMO price & demand CSVs")
    print(f"New York:   NYISO OASIS public bulk CSVs (no auth)")
    print(f"Output: {OUT_DIR}\n")

    for mkt in targets:
        DOWNLOADERS[mkt](args.year, args.force)

    print("\nDone. Re-run notebook 08_energy_markets.ipynb to use the new data.")


if __name__ == "__main__":
    main()
