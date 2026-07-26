"""Cleaning pipeline for the PJME hourly load series (SPEC section 2).

The five rules run in order and each one reports how many rows it touched, so the
whole pass is auditable via a reconciliation table.

Timezone policy -- the one decision everything else depends on:
    The raw Kaggle timestamps are US/Eastern *wall clock*. We localize them and
    immediately convert to UTC, then hold every downstream grid in UTC. That
    guarantees a strictly regular hourly index, so `lag_24` is always exactly 24
    rows back and a 24-point horizon is always exactly 24 hours -- neither is true
    on a local-time grid, which has a 23-hour and a 25-hour day every year.
    Human-behaviour features (hour, weekday, holiday) are still derived from the
    *local* clock, because that is what drives electricity demand.

Usage:
    conda run -n p3 python -m src.clean
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_PARQUET,
    LOAD_COL,
    LOAD_MAX_MW,
    LOAD_MIN_MW,
    LOCAL_TZ,
    MAD_THRESHOLD,
    MAX_INTERPOLATE_GAP_H,
    RAW_LOAD_CSV,
    RAW_WEATHER_CSV,
    ensure_dirs,
)

FLAG_COLS = ("imputed", "clipped", "bad_day")


class Reconciliation:
    """Append-only log of `(rule, rows_affected, note)` for the audit table."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(self, rule: str, affected: int, note: str = "") -> None:
        self._rows.append({"rule": rule, "rows_affected": affected, "note": note})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def render(self) -> str:
        df = self.to_frame()
        lines = ["=== Cleaning reconciliation ===", f"{'rule':<38}{'rows':>9}  note"]
        for _, r in df.iterrows():
            lines.append(f"{r['rule']:<38}{r['rows_affected']:>9,}  {r['note']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rule 1 + 2: timezone, duplicates, regular grid
# --------------------------------------------------------------------------- #
def localize_to_utc(raw: pd.DataFrame, rec: Reconciliation) -> pd.Series:
    """Rule 1 -- parse as US/Eastern wall clock and convert to UTC.

    DST fall-back produces two rows sharing one wall-clock stamp; SPEC says average
    them. That collapses two distinct UTC hours into one, so the vacated UTC hour
    reappears downstream as a 1-hour gap and is interpolated by rule 3.
    DST spring-forward stamps do not exist locally and are dropped here.
    """
    naive = pd.to_datetime(raw["Datetime"])
    n_raw = len(naive)

    # Averaging by wall clock covers both the DST fall-back pair and plain dupes.
    dup_wallclock = int(naive.duplicated().sum())
    by_wallclock = raw.assign(_naive=naive).groupby("_naive")[LOAD_COL].mean()
    rec.record(
        "1. DST fall-back / duplicate stamps",
        dup_wallclock,
        f"averaged; {n_raw:,} -> {len(by_wallclock):,} rows",
    )

    idx = by_wallclock.index.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="NaT")
    series = pd.Series(by_wallclock.to_numpy(), index=idx, name=LOAD_COL)

    n_nonexistent = int(series.index.isna().sum())
    series = series[series.index.notna()]
    rec.record("1. DST spring-forward stamps", n_nonexistent, "non-existent local time, dropped")

    series.index = series.index.tz_convert("UTC")
    series = series.groupby(level=0).mean().sort_index()
    series.index.name = "ts"
    return series


def build_regular_grid(series: pd.Series, rec: Reconciliation) -> pd.DataFrame:
    """Rule 2 -- reindex onto a complete hourly UTC grid; missing hours become NaN."""
    full = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC", name="ts")
    df = series.reindex(full).to_frame()
    rec.record(
        "2. Reindex to full hourly grid",
        int(df[LOAD_COL].isna().sum()),
        f"{len(full):,} slots, missing hours created as NaN",
    )
    return df


# --------------------------------------------------------------------------- #
# Rule 3: gaps
# --------------------------------------------------------------------------- #
def _gap_runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Return `(start_pos, length)` for each maximal run of True in `mask`."""
    values = mask.to_numpy()
    if not values.any():
        return []
    # Boundaries are where the mask flips; pad so runs at either edge are caught.
    padded = np.concatenate(([False], values, [False]))
    flips = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e - s)) for s, e in zip(flips[::2], flips[1::2], strict=False)]


def handle_gaps(df: pd.DataFrame, rec: Reconciliation) -> pd.DataFrame:
    """Rule 3 -- interpolate short gaps, quarantine the local day around long ones."""
    df = df.copy()
    df["imputed"] = 0
    df["bad_day"] = 0

    missing = df[LOAD_COL].isna()
    runs = _gap_runs(missing)
    short_runs = [(s, n) for s, n in runs if n <= MAX_INTERPOLATE_GAP_H]
    long_runs = [(s, n) for s, n in runs if n > MAX_INTERPOLATE_GAP_H]

    # Every missing hour gets filled -- the parquet contract is "no NaN" -- so every
    # filled hour is flagged `imputed`. The short/long distinction lives in bad_day.
    df["imputed"] = missing.astype(int)
    df[LOAD_COL] = df[LOAD_COL].interpolate(method="time", limit_direction="both")
    n_short = sum(n for _, n in short_runs)
    rec.record(
        "3. Short gaps (<=6h) interpolated",
        int(n_short),
        f"{len(short_runs)} runs, max {max((n for _, n in short_runs), default=0)}h",
    )

    # A long gap contaminates the whole local calendar day it falls in -- that day is
    # excluded from both training and evaluation.
    local_date = df.index.tz_convert(LOCAL_TZ).date
    bad_dates: set = set()
    for s, n in long_runs:
        bad_dates.update(local_date[s : s + n])
    if bad_dates:
        df.loc[np.isin(local_date, list(bad_dates)), "bad_day"] = 1
    n_long = sum(n for _, n in long_runs)
    rec.record(
        "3. Long gaps (>6h) -> bad_day",
        int(df["bad_day"].sum()),
        f"{n_long}h missing over {len(long_runs)} runs, {len(bad_dates)} local days quarantined",
    )
    return df


# --------------------------------------------------------------------------- #
# Rule 4: outliers
# --------------------------------------------------------------------------- #
def winsorize_outliers(df: pd.DataFrame, rec: Reconciliation) -> pd.DataFrame:
    """Rule 4 -- clip to median +/- 5*MAD within each (local month, local hour) cell.

    Conditioning on month and hour matters: a raw global threshold would flag every
    normal summer afternoon as an outlier. MAD is the unscaled median absolute
    deviation, per SPEC.
    """
    df = df.copy()
    local = df.index.tz_convert(LOCAL_TZ)
    key = pd.MultiIndex.from_arrays([local.month, local.hour], names=["month", "hour"])
    grouped = df[LOAD_COL].groupby(key)

    median = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    lower = median - MAD_THRESHOLD * mad
    upper = median + MAD_THRESHOLD * mad

    nonpositive = df[LOAD_COL] <= 0
    out_of_band = (df[LOAD_COL] < lower) | (df[LOAD_COL] > upper)
    flagged = nonpositive | out_of_band

    df["clipped"] = flagged.astype(int)
    df[LOAD_COL] = df[LOAD_COL].clip(lower=lower, upper=upper)
    rec.record(
        "4. Outliers winsorized",
        int(flagged.sum()),
        f"{int(nonpositive.sum())} non-positive, {int(out_of_band.sum())} beyond 5*MAD",
    )
    return df


# --------------------------------------------------------------------------- #
# Weather join
# --------------------------------------------------------------------------- #
def attach_weather(
    df: pd.DataFrame, rec: Reconciliation, weather: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Left-join Open-Meteo hourly temperature onto the UTC grid."""
    if weather is None:
        weather = pd.read_csv(RAW_WEATHER_CSV, parse_dates=["datetime_utc"])
    weather = weather.copy()
    weather["datetime_utc"] = pd.to_datetime(weather["datetime_utc"])
    if weather["datetime_utc"].dt.tz is None:
        weather["datetime_utc"] = weather["datetime_utc"].dt.tz_localize("UTC")
    temp = weather.set_index("datetime_utc")["temp_c"].sort_index()
    temp = temp[~temp.index.duplicated(keep="first")]

    df = df.copy()
    df["temp_c"] = temp.reindex(df.index)
    n_missing = int(df["temp_c"].isna().sum())
    df["temp_imputed"] = df["temp_c"].isna().astype(int)
    df["temp_c"] = df["temp_c"].interpolate(method="time", limit_direction="both")
    rec.record(
        "5. Weather joined", len(df) - n_missing, f"{n_missing} hours interpolated from neighbours"
    )
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """pandera contract: continuous index, plausible load and temperature, 0/1 flags."""
    import pandera as pa

    schema = pa.DataFrameSchema(
        columns={
            LOAD_COL: pa.Column(float, pa.Check.in_range(LOAD_MIN_MW, LOAD_MAX_MW), nullable=False),
            "temp_c": pa.Column(float, pa.Check.in_range(-40.0, 50.0), nullable=False),
            "imputed": pa.Column("int64", pa.Check.isin([0, 1])),
            "clipped": pa.Column("int64", pa.Check.isin([0, 1])),
            "bad_day": pa.Column("int64", pa.Check.isin([0, 1])),
            "temp_imputed": pa.Column("int64", pa.Check.isin([0, 1])),
        },
        index=pa.Index(pd.DatetimeTZDtype(tz="UTC"), unique=True, name="ts"),
        strict=True,
        ordered=False,
    )
    validated = schema.validate(df)

    # pandera has no "regular frequency" check, so assert it directly.
    deltas = validated.index.to_series().diff().dropna().unique()
    if not (len(deltas) == 1 and deltas[0] == pd.Timedelta("1h")):
        raise ValueError(f"index is not a continuous hourly grid: gaps {deltas}")
    return validated


# --------------------------------------------------------------------------- #
def clean(
    raw: pd.DataFrame | None = None, weather: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, Reconciliation]:
    """Run rules 1-5 and return the clean frame plus its reconciliation log."""
    rec = Reconciliation()
    if raw is None:
        raw = pd.read_csv(RAW_LOAD_CSV)
    rec.record("0. Raw rows read", len(raw), RAW_LOAD_CSV.name)

    series = localize_to_utc(raw, rec)
    df = build_regular_grid(series, rec)
    df = handle_gaps(df, rec)
    df = winsorize_outliers(df, rec)
    df = attach_weather(df, rec, weather)

    # int64 explicitly: numpy's default integer is int32 on Windows, which would
    # make the pandera dtype check pass on Linux CI and fail on the dev machine.
    for col in (*FLAG_COLS, "temp_imputed"):
        df[col] = df[col].astype("int64")
    df[LOAD_COL] = df[LOAD_COL].astype(float)
    df["temp_c"] = df["temp_c"].astype(float)

    df = validate(df)
    rec.record("5. pandera validation", len(df), "passed")
    return df, rec


def main() -> None:
    ensure_dirs()
    df, rec = clean()
    df.to_parquet(CLEAN_PARQUET)

    print(rec.render())
    local = df.index.tz_convert(LOCAL_TZ)
    usable = df[df["bad_day"] == 0]
    print("\n=== Phase 1 gate: clean series ===")
    print(f"rows           : {len(df):,}  ({len(usable):,} usable after bad_day)")
    print(f"span (UTC)     : {df.index.min()} .. {df.index.max()}")
    print(f"span (local)   : {local.min()} .. {local.max()}")
    print(
        f"{LOAD_COL:15s}: mean {df[LOAD_COL].mean():,.0f}  "
        f"min {df[LOAD_COL].min():,.0f}  max {df[LOAD_COL].max():,.0f} MW"
    )
    print(
        f"temp_c         : mean {df['temp_c'].mean():.1f}  "
        f"min {df['temp_c'].min():.1f}  max {df['temp_c'].max():.1f} C"
    )
    print(
        f"flags          : imputed={df['imputed'].sum():,}  "
        f"clipped={df['clipped'].sum():,}  bad_day={df['bad_day'].sum():,}"
    )
    print(f"written        : {CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
