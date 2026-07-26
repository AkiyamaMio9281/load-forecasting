"""Synthetic PJME-shaped fixtures.

The tests must run on a clone with no Kaggle credentials and no downloaded data,
so everything here is generated. The generator deliberately reproduces the four
things that make the real file awkward -- DST fall-back duplicates, a DST
spring-forward hole, a short gap, a long gap and an outlier spike -- because those
are exactly the branches tests/test_clean.py needs to exercise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import clean
from src.config import LOAD_COL, LOCAL_TZ
from src.features import build_features

SYNTH_START = "2015-01-01"
SYNTH_END = "2017-12-31"
SEED = 42

# Injected defects, expressed in local wall-clock terms.
SHORT_GAP_START = pd.Timestamp("2016-05-10 03:00", tz=LOCAL_TZ)
SHORT_GAP_HOURS = 3
LONG_GAP_START = pd.Timestamp("2016-09-14 00:00", tz=LOCAL_TZ)
LONG_GAP_HOURS = 30
SPIKE_AT = pd.Timestamp("2017-03-15 14:00", tz=LOCAL_TZ)
SPIKE_VALUE = 61_000.0


def _utc_index() -> pd.DatetimeIndex:
    start = pd.Timestamp(SYNTH_START, tz=LOCAL_TZ).tz_convert("UTC")
    end = pd.Timestamp(SYNTH_END + " 23:00", tz=LOCAL_TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="h", tz="UTC", name="ts")


def make_weather(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Seasonal + diurnal temperature, in the Open-Meteo csv shape."""
    index = _utc_index() if index is None else index
    local = index.tz_convert(LOCAL_TZ)
    doy = local.dayofyear.to_numpy()
    hour = local.hour.to_numpy()
    temp = (
        12.0
        - 13.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    )
    rng = np.random.default_rng(SEED + 1)
    temp = temp + rng.normal(0, 1.5, len(index))
    return pd.DataFrame({"datetime_utc": index.tz_convert("UTC"), "temp_c": temp})


def make_raw_load(weather: pd.DataFrame) -> pd.DataFrame:
    """Load with daily/weekly/yearly shape and a U-shaped temperature response.

    Written out the way Kaggle writes it: a naive **local wall-clock** column. The
    UTC -> local conversion is what manufactures the DST duplicate and hole.
    """
    index = pd.DatetimeIndex(weather["datetime_utc"])
    local = index.tz_convert(LOCAL_TZ)
    temp = weather["temp_c"].to_numpy()

    hour, dow, doy = local.hour.to_numpy(), local.dayofweek.to_numpy(), local.dayofyear.to_numpy()
    daily = 4_500 * np.sin(2 * np.pi * (hour - 9) / 24) + 1_800 * np.sin(
        4 * np.pi * (hour - 6) / 24
    )
    weekly = np.where(dow >= 5, -2_600, 400.0)
    yearly = 1_200 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    weather_effect = 380 * np.maximum(18 - temp, 0) + 520 * np.maximum(temp - 24, 0)
    trend = np.linspace(0, 900, len(index))

    rng = np.random.default_rng(SEED)
    load = (
        30_000 + daily + weekly + yearly + weather_effect + trend + rng.normal(0, 550, len(index))
    )
    load = np.clip(load, 14_000, 60_000)

    raw = pd.DataFrame(
        {
            "Datetime": local.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"),
            LOAD_COL: load,
        }
    )

    # --- inject defects, addressing rows by their UTC timestamp ---
    drop = pd.date_range(
        SHORT_GAP_START.tz_convert("UTC"), periods=SHORT_GAP_HOURS, freq="h"
    ).append(pd.date_range(LONG_GAP_START.tz_convert("UTC"), periods=LONG_GAP_HOURS, freq="h"))
    keep = ~index.isin(drop)
    raw = raw[keep].reset_index(drop=True)

    spike_row = np.flatnonzero(index[keep] == SPIKE_AT.tz_convert("UTC"))
    raw.loc[spike_row[0], LOAD_COL] = SPIKE_VALUE
    return raw


@pytest.fixture(scope="session")
def synthetic_weather() -> pd.DataFrame:
    return make_weather()


@pytest.fixture(scope="session")
def synthetic_raw(synthetic_weather: pd.DataFrame) -> pd.DataFrame:
    return make_raw_load(synthetic_weather)


@pytest.fixture(scope="session")
def cleaned(synthetic_raw: pd.DataFrame, synthetic_weather: pd.DataFrame):
    return clean(raw=synthetic_raw, weather=synthetic_weather)


@pytest.fixture(scope="session")
def clean_df(cleaned) -> pd.DataFrame:
    return cleaned[0]


@pytest.fixture(scope="session")
def reconciliation(cleaned):
    return cleaned[1]


@pytest.fixture(scope="session")
def features(clean_df: pd.DataFrame) -> pd.DataFrame:
    return build_features(clean_df)
