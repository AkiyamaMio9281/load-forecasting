"""Project-wide paths and constants. Single source of truth for magic numbers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "app" / "artifacts"

RAW_LOAD_CSV = RAW_DIR / "PJME_hourly.csv"
RAW_WEATHER_CSV = RAW_DIR / "philadelphia_temperature.csv"
CLEAN_PARQUET = PROCESSED_DIR / "pjme_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "pjme_features.parquet"

# --- Data source (SPEC section 1) ---
KAGGLE_DATASET = "robikscube/hourly-energy-consumption"
KAGGLE_FILE = "PJME_hourly.csv"
LOAD_COL = "PJME_MW"
LOCAL_TZ = "US/Eastern"

WEATHER_LAT = 39.95
WEATHER_LON = -75.16
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# --- Cleaning thresholds (SPEC section 2) ---
MAX_INTERPOLATE_GAP_H = 6  # gaps <= this are linearly interpolated
MAD_THRESHOLD = 5.0  # deviation from (month, hour) median in MAD units
LOAD_MIN_MW = 10_000  # pandera bounds
LOAD_MAX_MW = 65_000

# --- Feature engineering (SPEC section 3) ---
HDD_BASE_C = 18.0
CDD_BASE_C = 24.0
LAGS = (24, 48, 168)

# --- Backtest protocol (SPEC section 4) ---
HORIZON = 24  # hours predicted per fold (one full day)
N_FOLDS = 52
FOLD_STEP_DAYS = 7
SEED = 42

MODEL_NAMES = ("naive", "snaive", "sarima", "prophet", "lgbm", "lstm")


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
