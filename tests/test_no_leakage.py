"""The leakage regression suite (SPEC section 3 / test matrix section 9).

Leakage is the failure mode that makes a forecasting project worthless while every
number on screen looks excellent, so this file checks it three independent ways:

1. **Declared contract** -- `referenced_timestamps` says which hours each feature
   reads; assert they all fall at or before the cutoff.
2. **Observed behaviour** -- corrupt the series *after* the cutoff, rebuild the
   features, and assert the fold's feature rows are bit-identical. This catches a
   leak even if the declaration is wrong, which is the whole point: it does not
   trust the module's own account of itself.
3. **Predictive sanity** -- destroy the feature/target relationship in training and
   assert accuracy collapses. A model that stays accurate is reading the answer.

Run after any change to src/features.py or src/backtest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LOAD_COL
from src.features import (
    CALENDAR_FEATURES,
    FEATURE_COLS,
    WEATHER_FEATURES,
    build_features,
    referenced_timestamps,
)
from src.metrics import mape
from src.models.lgbm import LgbmDirectModel
from src.models.naive import NaiveModel, SeasonalNaiveModel

# The only features permitted to read the target hour itself. Everything else must
# stay at or behind the cutoff. Keeping the set here, spelled out, means widening
# the exemption requires editing a test -- it cannot happen by accident.
PERFECT_FORECAST_EXEMPT = {"temp", "hdd", "cdd"}

N_PROBE_FOLDS = 6


@pytest.fixture(scope="module")
def folds(features: pd.DataFrame):
    from src.backtest import make_folds

    return make_folds(features, n_folds=N_PROBE_FOLDS)


# --- 1. declared contract ---------------------------------------------------- #
def test_exemption_set_is_exactly_the_documented_weather_features() -> None:
    assert set(WEATHER_FEATURES) > PERFECT_FORECAST_EXEMPT
    assert set(WEATHER_FEATURES) - PERFECT_FORECAST_EXEMPT == {"temp_lag24"}


def test_no_feature_reads_beyond_the_cutoff(folds) -> None:
    for fold in folds:
        for feature in FEATURE_COLS:
            if feature in PERFECT_FORECAST_EXEMPT:
                continue
            referenced = referenced_timestamps(feature, fold.target_index, fold.cutoff)
            if len(referenced) == 0:
                continue
            assert (
                referenced.max() <= fold.cutoff
            ), f"fold {fold.fold}: {feature} reads {referenced.max()} > cutoff {fold.cutoff}"


def test_lag_24_is_the_binding_constraint(folds) -> None:
    """It touches the cutoff exactly -- proof the boundary is tight, not slack."""
    fold = folds[0]
    referenced = referenced_timestamps("lag_24", fold.target_index, fold.cutoff)
    assert referenced.max() == fold.cutoff


def test_calendar_features_read_no_data(folds) -> None:
    fold = folds[0]
    for feature in CALENDAR_FEATURES:
        assert len(referenced_timestamps(feature, fold.target_index, fold.cutoff)) == 0


def test_every_feature_has_a_declared_reference(folds) -> None:
    fold = folds[0]
    for feature in FEATURE_COLS:
        referenced_timestamps(feature, fold.target_index, fold.cutoff)  # raises if undeclared


# --- 2. observed behaviour --------------------------------------------------- #
def _corrupt_after(
    clean_df: pd.DataFrame, cutoff: pd.Timestamp, columns: list[str]
) -> pd.DataFrame:
    """Replace `columns` with noise strictly after `cutoff`, leaving history intact."""
    rng = np.random.default_rng(7)
    corrupted = clean_df.copy()
    future = corrupted.index > cutoff
    for col in columns:
        corrupted.loc[future, col] = rng.uniform(15_000, 60_000, int(future.sum()))
    return corrupted


def test_features_do_not_change_when_the_future_load_is_corrupted(
    clean_df: pd.DataFrame, features: pd.DataFrame, folds
) -> None:
    fold = folds[-1]
    corrupted = build_features(_corrupt_after(clean_df, fold.cutoff, [LOAD_COL]))

    baseline = features.loc[fold.target_index, list(FEATURE_COLS)]
    observed = corrupted.loc[fold.target_index, list(FEATURE_COLS)]
    pd.testing.assert_frame_equal(baseline, observed)


def test_only_the_exempt_features_move_when_future_weather_is_corrupted(
    clean_df: pd.DataFrame, features: pd.DataFrame, folds
) -> None:
    """Pins the exemption from the other side: temp/hdd/cdd must react, and nothing else."""
    fold = folds[-1]
    rng = np.random.default_rng(11)
    corrupted_clean = clean_df.copy()
    future = corrupted_clean.index > fold.cutoff
    corrupted_clean.loc[future, "temp_c"] = rng.uniform(-20, 40, int(future.sum()))
    corrupted = build_features(corrupted_clean)

    baseline = features.loc[fold.target_index, list(FEATURE_COLS)]
    observed = corrupted.loc[fold.target_index, list(FEATURE_COLS)]
    changed = {c for c in FEATURE_COLS if not np.allclose(baseline[c], observed[c])}
    assert changed == PERFECT_FORECAST_EXEMPT


def test_history_slice_never_extends_past_the_cutoff(features: pd.DataFrame, folds) -> None:
    for fold in folds:
        history = features.loc[features.index <= fold.cutoff]
        assert history.index.max() <= fold.cutoff
        assert history.index.max() < fold.target_index.min()
        assert not history.index.intersection(fold.target_index).size


@pytest.mark.parametrize("model_factory", [NaiveModel, SeasonalNaiveModel, LgbmDirectModel])
def test_predictions_are_unchanged_when_the_future_is_corrupted(
    clean_df: pd.DataFrame, features: pd.DataFrame, folds, model_factory
) -> None:
    """End-to-end: a model refit on corrupted-future data must forecast identically."""
    fold = folds[-1]
    corrupted = build_features(_corrupt_after(clean_df, fold.cutoff, [LOAD_COL]))

    predictions = []
    for table in (features, corrupted):
        model = model_factory(seed=42)
        model.fit(table.loc[table.index <= fold.cutoff])
        predictions.append(
            model.predict(fold.target_index, table.loc[fold.target_index, list(FEATURE_COLS)])
        )
    assert predictions[0] == pytest.approx(predictions[1])


# --- 3. predictive sanity ------------------------------------------------------ #
def test_shuffling_the_training_targets_destroys_accuracy(features: pd.DataFrame, folds) -> None:
    """If accuracy survives a permuted target, the features are carrying the answer."""
    fold = folds[-1]
    history = features.loc[features.index <= fold.cutoff]
    exog = features.loc[fold.target_index, list(FEATURE_COLS)]
    y_true = features.loc[fold.target_index, "y"].to_numpy()

    honest = LgbmDirectModel(seed=42)
    honest.fit(history)
    honest_mape = mape(y_true, honest.predict(fold.target_index, exog))

    shuffled_history = history.copy()
    shuffled_history["y"] = history["y"].sample(frac=1.0, random_state=0).to_numpy()
    shuffled = LgbmDirectModel(seed=42)
    shuffled.fit(shuffled_history)
    shuffled_mape = mape(y_true, shuffled.predict(fold.target_index, exog))

    assert (
        honest_mape < 0.5 * shuffled_mape
    ), f"permuting the target barely hurt: {honest_mape:.4f} vs {shuffled_mape:.4f}"
