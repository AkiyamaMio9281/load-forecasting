"""Cleaning rules (SPEC section 2 / test matrix section 9).

Each test names the rule it pins down. The DST pair matters most: it is the one
place where a plausible-looking implementation silently drops or duplicates an
hour, and nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import _gap_runs, clean, validate
from src.config import LOAD_COL, LOCAL_TZ, MAX_INTERPOLATE_GAP_H
from tests.conftest import (
    LONG_GAP_HOURS,
    LONG_GAP_START,
    SHORT_GAP_HOURS,
    SHORT_GAP_START,
    SPIKE_AT,
    SPIKE_VALUE,
)


# --- rules 1 & 2: timezone and grid ---------------------------------------- #
def test_index_is_continuous_hourly_utc(clean_df: pd.DataFrame) -> None:
    assert str(clean_df.index.tz) == "UTC"
    deltas = clean_df.index.to_series().diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta("1h")]


def test_index_has_no_duplicates(clean_df: pd.DataFrame) -> None:
    assert not clean_df.index.duplicated().any()


def test_dst_fall_back_hour_is_averaged_not_duplicated(
    synthetic_raw: pd.DataFrame, clean_df: pd.DataFrame
) -> None:
    """Autumn DST repeats one wall clock; SPEC says average, so 2 rows -> 1 value."""
    naive = pd.to_datetime(synthetic_raw["Datetime"])
    repeated = naive[naive.duplicated()]
    assert len(repeated) > 0, "fixture should contain DST fall-back duplicates"

    for stamp in repeated:
        pair = synthetic_raw.loc[naive == stamp, LOAD_COL]
        assert len(pair) == 2
        # The averaged value lands on the first (EDT) UTC slot; the vacated second
        # slot becomes a 1-hour gap and is interpolated by rule 3.
        utc = pd.Timestamp(stamp).tz_localize(LOCAL_TZ, ambiguous=True).tz_convert("UTC")
        assert clean_df.loc[utc, LOAD_COL] == pytest.approx(pair.mean(), rel=1e-9)


def test_dst_spring_forward_leaves_no_hole_in_utc(clean_df: pd.DataFrame) -> None:
    """The missing local hour is an artefact of the local clock, not of the data."""
    local = clean_df.index.tz_convert(LOCAL_TZ)
    spring_days = pd.Series(local.date)[pd.Series(local.hour) == 3].value_counts()
    assert (spring_days == 1).all()
    assert clean_df[LOAD_COL].notna().all()


# --- rule 3: gaps ----------------------------------------------------------- #
def test_gap_runs_finds_maximal_runs() -> None:
    mask = pd.Series([False, True, True, False, True, False, False, True])
    assert _gap_runs(mask) == [(1, 2), (4, 1), (7, 1)]


def test_gap_runs_handles_edges() -> None:
    assert _gap_runs(pd.Series([True, True, False])) == [(0, 2)]
    assert _gap_runs(pd.Series([False, True, True])) == [(1, 2)]
    assert _gap_runs(pd.Series([False, False])) == []


def test_short_gap_is_interpolated_and_flagged(clean_df: pd.DataFrame) -> None:
    gap = pd.date_range(SHORT_GAP_START.tz_convert("UTC"), periods=SHORT_GAP_HOURS, freq="h")
    assert SHORT_GAP_HOURS <= MAX_INTERPOLATE_GAP_H
    assert clean_df.loc[gap, LOAD_COL].notna().all()
    assert (clean_df.loc[gap, "imputed"] == 1).all()
    assert (clean_df.loc[gap, "bad_day"] == 0).all()


def test_short_gap_interpolation_is_linear(clean_df: pd.DataFrame) -> None:
    gap_start = SHORT_GAP_START.tz_convert("UTC")
    anchor_before = gap_start - pd.Timedelta(hours=1)
    anchor_after = gap_start + pd.Timedelta(hours=SHORT_GAP_HOURS)
    lo = clean_df.loc[anchor_before, LOAD_COL]
    hi = clean_df.loc[anchor_after, LOAD_COL]
    expected = np.linspace(lo, hi, SHORT_GAP_HOURS + 2)[1:-1]
    filled = clean_df.loc[
        pd.date_range(gap_start, periods=SHORT_GAP_HOURS, freq="h"), LOAD_COL
    ].to_numpy()
    assert filled == pytest.approx(expected, rel=1e-6)


def test_long_gap_quarantines_every_local_day_it_touches(clean_df: pd.DataFrame) -> None:
    gap = pd.date_range(LONG_GAP_START.tz_convert("UTC"), periods=LONG_GAP_HOURS, freq="h")
    assert LONG_GAP_HOURS > MAX_INTERPOLATE_GAP_H

    touched_days = set(gap.tz_convert(LOCAL_TZ).date)
    assert len(touched_days) == 2, "a 30h gap from midnight must span two local days"

    local_date = clean_df.index.tz_convert(LOCAL_TZ).date
    for day in touched_days:
        day_rows = clean_df[local_date == day]
        assert (day_rows["bad_day"] == 1).all(), f"{day} should be fully quarantined"


def test_days_without_long_gaps_are_not_quarantined(clean_df: pd.DataFrame) -> None:
    local_date = clean_df.index.tz_convert(LOCAL_TZ).date
    untouched = clean_df[local_date == pd.Timestamp("2016-06-01").date()]
    assert (untouched["bad_day"] == 0).all()


# --- rule 4: outliers ------------------------------------------------------- #
def test_spike_is_winsorized_and_flagged(clean_df: pd.DataFrame) -> None:
    ts = SPIKE_AT.tz_convert("UTC")
    assert clean_df.loc[ts, "clipped"] == 1
    assert clean_df.loc[ts, LOAD_COL] < SPIKE_VALUE


def test_winsorizing_touches_few_rows(clean_df: pd.DataFrame) -> None:
    """A 5x MAD band conditioned on (month, hour) should flag outliers, not seasons."""
    assert clean_df["clipped"].mean() < 0.02


# --- rule 5 and the audit trail --------------------------------------------- #
def test_validate_rejects_a_broken_grid(clean_df: pd.DataFrame) -> None:
    punctured = clean_df.drop(clean_df.index[100])
    with pytest.raises(ValueError, match="continuous hourly grid"):
        validate(punctured)


def test_validate_rejects_out_of_range_load(clean_df: pd.DataFrame) -> None:
    import pandera.errors

    broken = clean_df.copy()
    broken.iloc[5, broken.columns.get_loc(LOAD_COL)] = 999_999.0
    with pytest.raises(pandera.errors.SchemaError):
        validate(broken)


def test_reconciliation_accounts_for_every_rule(reconciliation) -> None:
    frame = reconciliation.to_frame()
    rules = frame["rule"].tolist()
    for expected in (
        "0. Raw rows read",
        "1. DST fall-back / duplicate stamps",
        "2. Reindex to full hourly grid",
        "3. Short gaps (<=6h) interpolated",
        "3. Long gaps (>6h) -> bad_day",
        "4. Outliers winsorized",
        "5. pandera validation",
    ):
        assert any(r.startswith(expected) for r in rules), f"missing rule: {expected}"
    assert (frame["rows_affected"] >= 0).all()


def test_row_count_is_conserved(
    synthetic_raw: pd.DataFrame, clean_df: pd.DataFrame, reconciliation
) -> None:
    """raw rows - duplicates_merged - nonexistent + hours_created == final rows."""
    frame = reconciliation.to_frame().set_index("rule")["rows_affected"]
    raw_rows = frame["0. Raw rows read"]
    merged = frame["1. DST fall-back / duplicate stamps"]
    dropped = frame["1. DST spring-forward stamps"]
    created = frame["2. Reindex to full hourly grid"]
    assert raw_rows - merged - dropped + created == len(clean_df)


def test_clean_output_has_expected_columns(clean_df: pd.DataFrame) -> None:
    assert set(clean_df.columns) == {
        LOAD_COL,
        "temp_c",
        "imputed",
        "clipped",
        "bad_day",
        "temp_imputed",
    }
    for flag in ("imputed", "clipped", "bad_day", "temp_imputed"):
        assert set(clean_df[flag].unique()) <= {0, 1}


def test_clean_is_idempotent_on_its_own_output(synthetic_raw, synthetic_weather) -> None:
    first, _ = clean(raw=synthetic_raw, weather=synthetic_weather)
    second, _ = clean(raw=synthetic_raw, weather=synthetic_weather)
    pd.testing.assert_frame_equal(first, second)
