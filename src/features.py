"""Feature construction for the tree/NN models (SPEC section 3).

Everything here hangs off one idea: the **operating day** and its **cutoff**.

    Operating day  = 05:00 UTC .. 04:00 UTC next day  (midnight EST)
    Cutoff of a day = 04:00 UTC that morning = the last hour we are allowed to see

We anchor the day to a fixed UTC offset rather than the true local midnight so the
horizon is *always* exactly 24 points and `lag_24` is *always* exactly 24 rows back.
A true-local-day definition gives a 23-hour and a 25-hour day each year, which
silently breaks both. The price is that during EDT the operating day starts at
01:00 local instead of 00:00 -- immaterial for a day-ahead exercise, and the
calendar features below still read the *true* local clock, which is what actually
drives demand.

Leakage contract: for a target timestamp `t` with cutoff `c`, every feature except
the `temp` family references only timestamps <= `c`. `lag_24` is the binding
constraint (`t - 24h == c` exactly when t is the last hour of the day), and
tests/test_no_leakage.py asserts the property fold by fold.

The `temp` family is deliberately exempt: it uses the realised temperature at `t`,
i.e. a **perfect weather forecast** assumption. This is stated in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CDD_BASE_C,
    CLEAN_PARQUET,
    FEATURES_PARQUET,
    HDD_BASE_C,
    HORIZON,
    LOAD_COL,
    LOCAL_TZ,
    ensure_dirs,
)

OP_DAY_OFFSET_H = 5  # 05:00 UTC == 00:00 EST

# Feature groups -- names are the contract with SPEC section 3.
LAG_FEATURES: dict[str, int] = {"lag_24": 24, "lag_48": 48, "lag_168": 168}
ROLL_FEATURES: dict[str, tuple[int, str]] = {
    "roll24_mean": (24, "mean"),
    "roll24_std": (24, "std"),
    "roll168_mean": (168, "mean"),
    "roll168_max": (168, "max"),
}
CALENDAR_FEATURES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
)
WEATHER_FEATURES: tuple[str, ...] = ("temp", "temp_lag24", "hdd", "cdd")

FEATURE_COLS: tuple[str, ...] = (
    *LAG_FEATURES,
    *ROLL_FEATURES,
    *CALENDAR_FEATURES,
    *WEATHER_FEATURES,
)
# Bookkeeping columns carried alongside the features for the backtest and analysis.
META_COLS: tuple[str, ...] = (
    "y",
    "cutoff_ts",
    "lead",
    "op_date",
    "bad_day",
    "local_hour",
    "local_month",
    "local_date",
)


# --------------------------------------------------------------------------- #
# Operating-day arithmetic
# --------------------------------------------------------------------------- #
def op_day_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First hour (05:00 UTC) of the operating day each timestamp belongs to."""
    shifted = index - pd.Timedelta(hours=OP_DAY_OFFSET_H)
    return shifted.floor("D") + pd.Timedelta(hours=OP_DAY_OFFSET_H)


def cutoff_for(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last observable hour before each timestamp's operating day opens."""
    return op_day_start(index) - pd.Timedelta(hours=1)


def op_date_for(index: pd.DatetimeIndex) -> np.ndarray:
    """Operating date as a naive date array (the fold's `target_date`)."""
    return op_day_start(index).tz_convert("UTC").normalize().tz_localize(None).date


def target_index_for(op_date: pd.Timestamp) -> pd.DatetimeIndex:
    """The 24 target timestamps of an operating day, given its (naive) date."""
    start = pd.Timestamp(op_date, tz="UTC") + pd.Timedelta(hours=OP_DAY_OFFSET_H)
    return pd.date_range(start, periods=HORIZON, freq="h", tz="UTC", name="ts")


def cutoff_of_op_date(op_date: pd.Timestamp) -> pd.Timestamp:
    """The single cutoff timestamp for an operating day."""
    return target_index_for(op_date)[0] - pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- #
# Leakage bookkeeping -- consumed by tests/test_no_leakage.py
# --------------------------------------------------------------------------- #
def referenced_timestamps(
    feature: str, target_ts: pd.DatetimeIndex, cutoff: pd.Timestamp
) -> pd.DatetimeIndex:
    """Every historical timestamp `feature` reads in order to be computed.

    Calendar features read no data at all (empty index). Weather features return
    their true reference even though they are exempt, so a test can assert the
    exemption is exactly the documented set and nothing more.
    """
    if feature in LAG_FEATURES:
        return target_ts - pd.Timedelta(hours=LAG_FEATURES[feature])
    if feature in ROLL_FEATURES:
        window, _ = ROLL_FEATURES[feature]
        return pd.date_range(end=cutoff, periods=window, freq="h", tz="UTC")
    if feature in CALENDAR_FEATURES:
        return pd.DatetimeIndex([], tz="UTC")
    if feature == "temp_lag24":
        return target_ts - pd.Timedelta(hours=24)
    if feature in ("temp", "hdd", "cdd"):
        return target_ts  # perfect-forecast assumption
    raise KeyError(f"unknown feature: {feature}")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _cyclical(values: np.ndarray, period: int, name: str) -> dict[str, np.ndarray]:
    radians = 2 * np.pi * values / period
    return {f"{name}_sin": np.sin(radians), f"{name}_cos": np.cos(radians)}


def _calendar_block(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features read the *local* clock -- demand follows human schedules."""
    import holidays

    local = index.tz_convert(LOCAL_TZ)
    local_dates = local.date
    us_holidays = holidays.US(years=range(local.year.min(), local.year.max() + 1))

    block: dict[str, np.ndarray] = {}
    block.update(_cyclical(local.hour.to_numpy(), 24, "hour"))
    block.update(_cyclical(local.dayofweek.to_numpy(), 7, "dow"))
    block.update(_cyclical(local.month.to_numpy(), 12, "month"))
    block["is_weekend"] = np.asarray(local.dayofweek >= 5).astype(int)
    block["is_holiday"] = np.fromiter(
        (d in us_holidays for d in local_dates), dtype=int, count=len(index)
    )
    return pd.DataFrame(block, index=index)


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Turn the clean UTC grid into the modelling table. One row per target hour."""
    if not clean.index.is_monotonic_increasing:
        clean = clean.sort_index()

    index = clean.index
    load = clean[LOAD_COL]
    temp = clean["temp_c"]

    out = pd.DataFrame(index=index)
    out["y"] = load.astype(float)

    # --- lags: read at the target timestamp, shifted back a fixed number of hours ---
    for name, hours in LAG_FEATURES.items():
        out[name] = load.shift(hours)

    # --- rolling stats: evaluated *once at the cutoff*, shared by all 24 leads ---
    cutoffs = cutoff_for(index)
    for name, (window, how) in ROLL_FEATURES.items():
        rolled = getattr(load.rolling(window), how)()
        out[name] = rolled.reindex(cutoffs).to_numpy()

    # --- calendar ---
    out = out.join(_calendar_block(index))

    # --- weather (perfect-forecast assumption, see module docstring) ---
    out["temp"] = temp.astype(float)
    out["temp_lag24"] = temp.shift(24)
    out["hdd"] = np.maximum(HDD_BASE_C - out["temp"], 0.0)
    out["cdd"] = np.maximum(out["temp"] - CDD_BASE_C, 0.0)

    # --- bookkeeping ---
    local = index.tz_convert(LOCAL_TZ)
    out["cutoff_ts"] = cutoffs
    out["lead"] = ((index - cutoffs) // pd.Timedelta(hours=1)).astype(int)
    out["op_date"] = op_date_for(index)
    out["bad_day"] = clean["bad_day"].to_numpy()
    out["local_hour"] = local.hour.to_numpy()
    out["local_month"] = local.month.to_numpy()
    out["local_date"] = local.date

    assert out["lead"].between(1, HORIZON).all(), "lead must lie in 1..24"
    return out


def usable_mask(features: pd.DataFrame) -> pd.Series:
    """Rows fit for training: no bad_day, no NaN in any model input."""
    return (features["bad_day"] == 0) & features[list(FEATURE_COLS)].notna().all(axis=1)


def main() -> None:
    ensure_dirs()
    clean = pd.read_parquet(CLEAN_PARQUET)
    feats = build_features(clean)
    feats.to_parquet(FEATURES_PARQUET)

    usable = usable_mask(feats)
    print("=== Feature table ===")
    print(f"rows            : {len(feats):,}  ({int(usable.sum()):,} usable)")
    print(f"span            : {feats.index.min()} .. {feats.index.max()}")
    print(f"model features  : {len(FEATURE_COLS)}  -> {', '.join(FEATURE_COLS)}")
    print(f"operating days  : {pd.Series(feats['op_date']).nunique():,}")
    print(f"written         : {FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
