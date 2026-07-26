"""Fetch the two raw inputs: PJME hourly load (Kaggle) and Philadelphia hourly
temperature (Open-Meteo). Idempotent -- an existing file is left untouched.

Usage:
    conda run -n p3 python data/download.py
    conda run -n p3 python data/download.py --force   # re-download both
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    KAGGLE_DATASET,
    KAGGLE_FILE,
    LOAD_COL,
    OPEN_METEO_URL,
    RAW_DIR,
    RAW_LOAD_CSV,
    RAW_WEATHER_CSV,
    WEATHER_LAT,
    WEATHER_LON,
    ensure_dirs,
)

KAGGLE_HELP = """
Kaggle credentials not found.

  1. Log in to kaggle.com -> Account -> Settings -> API -> "Create New Token"
  2. Save the downloaded kaggle.json to:  %USERPROFILE%\\.kaggle\\kaggle.json
  3. Re-run:  conda run -n p3 python data/download.py

(See HUMAN_TASKS.md section 1. The dataset is a plain Dataset -- no competition
rules to accept.)
"""


def _kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = os.environ.get("KAGGLE_CONFIG_DIR")
    candidates = [Path(cfg) / "kaggle.json"] if cfg else []
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")
    return any(p.exists() for p in candidates)


def download_load(force: bool = False) -> Path:
    """Download PJME_hourly.csv from the Kaggle hourly-energy-consumption dataset."""
    if RAW_LOAD_CSV.exists() and not force:
        print(f"[skip] {RAW_LOAD_CSV.name} already present")
        return RAW_LOAD_CSV

    if not _kaggle_credentials_present():
        print(KAGGLE_HELP, file=sys.stderr)
        raise SystemExit(2)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[kaggle] downloading {KAGGLE_DATASET}:{KAGGLE_FILE} ...")
    api.dataset_download_file(KAGGLE_DATASET, KAGGLE_FILE, path=str(RAW_DIR), force=force)

    # The API lands either the bare csv or a same-named zip depending on version.
    zipped = RAW_DIR / f"{KAGGLE_FILE}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extract(KAGGLE_FILE, path=RAW_DIR)
        zipped.unlink()

    if not RAW_LOAD_CSV.exists():
        raise RuntimeError(f"expected {RAW_LOAD_CSV} after Kaggle download")
    return RAW_LOAD_CSV


def _load_date_range(path: Path) -> tuple[str, str]:
    """Read the load csv only to learn which weather window we need."""
    ts = pd.read_csv(path, usecols=["Datetime"], parse_dates=["Datetime"])["Datetime"]
    # Pad by a day on each side so UTC conversion never runs off the edge.
    lo = (ts.min() - pd.Timedelta(days=1)).date()
    hi = (ts.max() + pd.Timedelta(days=1)).date()
    return lo.isoformat(), hi.isoformat()


def download_weather(force: bool = False) -> Path:
    """Hourly 2m temperature for Philadelphia over the load series' span (UTC)."""
    if RAW_WEATHER_CSV.exists() and not force:
        print(f"[skip] {RAW_WEATHER_CSV.name} already present")
        return RAW_WEATHER_CSV

    if not RAW_LOAD_CSV.exists():
        raise RuntimeError("download the load series first -- it defines the date range")

    start, end = _load_date_range(RAW_LOAD_CSV)
    print(f"[open-meteo] temperature_2m {start} .. {end} @ ({WEATHER_LAT}, {WEATHER_LON})")

    frames = []
    # Chunk by 5-year blocks: one 17-year request is large enough to time out.
    for yr_lo in range(int(start[:4]), int(end[:4]) + 1, 5):
        yr_hi = min(yr_lo + 4, int(end[:4]))
        chunk_start = max(start, f"{yr_lo}-01-01")
        chunk_end = min(end, f"{yr_hi}-12-31")
        params = {
            "latitude": WEATHER_LAT,
            "longitude": WEATHER_LON,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=180)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        frames.append(pd.DataFrame(hourly))
        print(f"  [ok] {chunk_start} .. {chunk_end}  ({len(hourly['time'])} hours)")

    weather = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    weather = weather.rename(columns={"time": "datetime_utc", "temperature_2m": "temp_c"})
    weather.to_csv(RAW_WEATHER_CSV, index=False)
    return RAW_WEATHER_CSV


def summarize() -> None:
    """Phase 0 gate: print row counts and year spans of both raw files."""
    load = pd.read_csv(RAW_LOAD_CSV, parse_dates=["Datetime"])
    weather = pd.read_csv(RAW_WEATHER_CSV, parse_dates=["datetime_utc"])

    print("\n=== Phase 0 gate: raw data landed ===")
    print(
        f"{RAW_LOAD_CSV.name:32s} rows={len(load):>7,}  "
        f"{load['Datetime'].min().date()} .. {load['Datetime'].max().date()}  "
        f"{LOAD_COL}: {load[LOAD_COL].min():.0f}-{load[LOAD_COL].max():.0f} MW"
    )
    print(
        f"{RAW_WEATHER_CSV.name:32s} rows={len(weather):>7,}  "
        f"{weather['datetime_utc'].min().date()} .. {weather['datetime_utc'].max().date()}  "
        f"temp_c: {weather['temp_c'].min():.1f}-{weather['temp_c'].max():.1f} C"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    ensure_dirs()
    download_load(force=args.force)
    download_weather(force=args.force)
    summarize()


if __name__ == "__main__":
    main()
