"""Feature definitions (SPEC section 3 / test matrix section 9).

Every feature is checked against a value computed by hand from the clean frame,
not against another call into the same code path.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.config import CDD_BASE_C, HDD_BASE_C, HORIZON, LOAD_COL, LOCAL_TZ
from src.features import (
    FEATURE_COLS,
    LAG_FEATURES,
    OP_DAY_OFFSET_H,
    build_features,
    cutoff_for,
    cutoff_of_op_date,
    op_day_start,
    target_index_for,
    usable_mask,
)

PROBE = pd.Timestamp("2017-06-15 18:00", tz="UTC")  # arbitrary mid-series hour


# --- operating-day arithmetic ----------------------------------------------- #
def test_op_day_start_is_05_utc() -> None:
    index = pd.date_range("2017-06-15 00:00", periods=48, freq="h", tz="UTC")
    starts = op_day_start(index)
    assert set(starts.hour) == {OP_DAY_OFFSET_H}
    # 04:00 belongs to the *previous* operating day; 05:00 opens a new one.
    assert starts[4] == pd.Timestamp("2017-06-14 05:00", tz="UTC")
    assert starts[5] == pd.Timestamp("2017-06-15 05:00", tz="UTC")


def test_cutoff_is_one_hour_before_the_day_opens() -> None:
    index = pd.date_range("2017-06-15 05:00", periods=24, freq="h", tz="UTC")
    cutoffs = cutoff_for(index)
    assert (cutoffs == pd.Timestamp("2017-06-15 04:00", tz="UTC")).all()


def test_target_index_is_exactly_24_hours_even_across_dst() -> None:
    for day in ("2017-03-12", "2017-11-05", "2017-06-15"):
        idx = target_index_for(dt.date.fromisoformat(day))
        assert len(idx) == HORIZON
        assert (idx[-1] - idx[0]) == pd.Timedelta(hours=23)
        assert cutoff_of_op_date(dt.date.fromisoformat(day)) == idx[0] - pd.Timedelta(hours=1)


def test_lead_runs_1_to_24(features: pd.DataFrame) -> None:
    assert features["lead"].min() == 1
    assert features["lead"].max() == HORIZON
    expected = (features.index - features["cutoff_ts"]) // pd.Timedelta(hours=1)
    assert (features["lead"] == expected).all()


def test_every_operating_day_in_the_interior_has_24_rows(features: pd.DataFrame) -> None:
    counts = features.groupby("op_date").size()
    assert (counts.iloc[1:-1] == HORIZON).all()


# --- lags -------------------------------------------------------------------- #
@pytest.mark.parametrize("feature,hours", sorted(LAG_FEATURES.items()))
def test_lag_reads_the_value_that_many_hours_earlier(
    features: pd.DataFrame, clean_df: pd.DataFrame, feature: str, hours: int
) -> None:
    expected = clean_df.loc[PROBE - pd.Timedelta(hours=hours), LOAD_COL]
    assert features.loc[PROBE, feature] == pytest.approx(expected)


def test_lags_are_nan_only_at_the_start(features: pd.DataFrame) -> None:
    assert features["lag_168"].iloc[:168].isna().all()
    assert features["lag_168"].iloc[168:].notna().all()


# --- rolling windows --------------------------------------------------------- #
def test_roll24_mean_is_the_24h_window_ending_at_the_cutoff(
    features: pd.DataFrame, clean_df: pd.DataFrame
) -> None:
    cutoff = features.loc[PROBE, "cutoff_ts"]
    window = clean_df.loc[cutoff - pd.Timedelta(hours=23) : cutoff, LOAD_COL]
    assert len(window) == 24
    assert features.loc[PROBE, "roll24_mean"] == pytest.approx(window.mean())
    assert features.loc[PROBE, "roll24_std"] == pytest.approx(window.std(ddof=1))


def test_roll168_stats_use_the_week_ending_at_the_cutoff(
    features: pd.DataFrame, clean_df: pd.DataFrame
) -> None:
    cutoff = features.loc[PROBE, "cutoff_ts"]
    window = clean_df.loc[cutoff - pd.Timedelta(hours=167) : cutoff, LOAD_COL]
    assert len(window) == 168
    assert features.loc[PROBE, "roll168_mean"] == pytest.approx(window.mean())
    assert features.loc[PROBE, "roll168_max"] == pytest.approx(window.max())


def test_rolling_features_are_constant_within_an_operating_day(features: pd.DataFrame) -> None:
    """They are evaluated once at the cutoff, so all 24 leads must share a value."""
    day = features[features["op_date"] == dt.date(2017, 6, 15)]
    assert len(day) == HORIZON
    for col in ("roll24_mean", "roll24_std", "roll168_mean", "roll168_max"):
        assert day[col].nunique() == 1


# --- calendar ---------------------------------------------------------------- #
def test_cyclical_hour_encoding_matches_the_local_clock(features: pd.DataFrame) -> None:
    local_hour = PROBE.tz_convert(LOCAL_TZ).hour
    assert features.loc[PROBE, "hour_sin"] == pytest.approx(np.sin(2 * np.pi * local_hour / 24))
    assert features.loc[PROBE, "hour_cos"] == pytest.approx(np.cos(2 * np.pi * local_hour / 24))


def test_cyclical_encodings_are_unit_norm(features: pd.DataFrame) -> None:
    for (name,) in (("hour",), ("dow",), ("month",)):
        norm = features[f"{name}_sin"] ** 2 + features[f"{name}_cos"] ** 2
        assert norm.to_numpy() == pytest.approx(np.ones(len(features)))


def test_weekend_flag_follows_the_local_weekday(features: pd.DataFrame) -> None:
    local_dow = features.index.tz_convert(LOCAL_TZ).dayofweek
    assert (features["is_weekend"].to_numpy() == (local_dow >= 5).astype(int)).all()


def test_holiday_flag_catches_july_4th_and_not_july_5th(features: pd.DataFrame) -> None:
    july4 = features[features["local_date"] == dt.date(2017, 7, 4)]
    july5 = features[features["local_date"] == dt.date(2017, 7, 5)]
    assert (july4["is_holiday"] == 1).all()
    assert (july5["is_holiday"] == 0).all()


# --- weather ------------------------------------------------------------------ #
def test_temp_is_the_realised_value_at_the_target_hour(
    features: pd.DataFrame, clean_df: pd.DataFrame
) -> None:
    assert features.loc[PROBE, "temp"] == pytest.approx(clean_df.loc[PROBE, "temp_c"])


def test_temp_lag24_reads_a_day_earlier(features: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    expected = clean_df.loc[PROBE - pd.Timedelta(hours=24), "temp_c"]
    assert features.loc[PROBE, "temp_lag24"] == pytest.approx(expected)


def test_degree_days_are_one_sided_hinges(features: pd.DataFrame) -> None:
    temp = features["temp"].to_numpy()
    assert features["hdd"].to_numpy() == pytest.approx(np.maximum(HDD_BASE_C - temp, 0))
    assert features["cdd"].to_numpy() == pytest.approx(np.maximum(temp - CDD_BASE_C, 0))
    assert ((features["hdd"] > 0) & (features["cdd"] > 0)).sum() == 0


# --- table contract ------------------------------------------------------------ #
def test_every_spec_feature_is_present(features: pd.DataFrame) -> None:
    expected = {
        "lag_24",
        "lag_48",
        "lag_168",
        "roll24_mean",
        "roll24_std",
        "roll168_mean",
        "roll168_max",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
        "temp",
        "temp_lag24",
        "hdd",
        "cdd",
    }
    assert set(FEATURE_COLS) == expected
    assert expected <= set(features.columns)


def test_usable_mask_drops_bad_days_and_warmup(features: pd.DataFrame) -> None:
    mask = usable_mask(features)
    assert not mask.iloc[:168].any(), "warm-up rows lack lag_168"
    assert (features.loc[mask, "bad_day"] == 0).all()
    assert mask.sum() > 0.9 * (len(features) - 168 - features["bad_day"].sum())


def test_target_column_matches_the_clean_series(
    features: pd.DataFrame, clean_df: pd.DataFrame
) -> None:
    assert features["y"].to_numpy() == pytest.approx(clean_df[LOAD_COL].to_numpy())


def test_build_features_is_deterministic(clean_df: pd.DataFrame) -> None:
    a = build_features(clean_df)
    b = build_features(clean_df)
    pd.testing.assert_frame_equal(a, b)
