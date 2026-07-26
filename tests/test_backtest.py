"""Fold construction and the harness contract (SPEC section 4 / test matrix section 9).

A backtest that silently overlaps folds, drops hours, or scores two models on
different target points produces a comparison table that is confidently wrong.
These tests pin the split geometry down.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    MODEL_REGISTRY,
    build_summary,
    eligible_op_dates,
    make_folds,
    run_backtest,
    run_fold,
)
from src.config import FOLD_STEP_DAYS, HORIZON
from src.metrics import fold_maes, mase, summarize
from src.models.naive import NaiveModel, SeasonalNaiveModel

N_FOLDS_TEST = 12


@pytest.fixture(scope="module")
def folds(features: pd.DataFrame):
    return make_folds(features, n_folds=N_FOLDS_TEST)


# --- fold geometry ------------------------------------------------------------ #
def test_requested_number_of_folds(folds) -> None:
    assert len(folds) == N_FOLDS_TEST
    assert [f.fold for f in folds] == list(range(1, N_FOLDS_TEST + 1))


def test_folds_are_chronological_and_seven_days_apart(folds) -> None:
    dates = [f.target_date for f in folds]
    assert dates == sorted(dates)
    steps = {b - a for a, b in zip(dates, dates[1:], strict=False)}
    assert steps == {timedelta(days=FOLD_STEP_DAYS)}


def test_each_fold_predicts_exactly_24_consecutive_hours(folds) -> None:
    for fold in folds:
        assert len(fold.target_index) == HORIZON
        deltas = fold.target_index.to_series().diff().dropna().unique()
        assert list(deltas) == [pd.Timedelta("1h")]


def test_target_windows_never_overlap(folds) -> None:
    seen = pd.DatetimeIndex([])
    for fold in folds:
        assert seen.intersection(
            fold.target_index
        ).empty, f"fold {fold.fold} overlaps an earlier one"
        seen = seen.append(fold.target_index)
    assert len(seen) == len(folds) * HORIZON


def test_cutoff_immediately_precedes_the_target_window(folds) -> None:
    for fold in folds:
        assert fold.target_index[0] - fold.cutoff == pd.Timedelta(hours=1)


def test_training_window_expands(features: pd.DataFrame, folds) -> None:
    sizes = [len(features.loc[features.index <= f.cutoff]) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] - sizes[0] == (len(folds) - 1) * FOLD_STEP_DAYS * 24


def test_fold_rejects_a_misaligned_horizon(folds) -> None:
    from src.backtest import Fold

    fold = folds[0]
    with pytest.raises(ValueError, match="horizon"):
        Fold(
            fold=1,
            target_date=fold.target_date,
            cutoff=fold.cutoff,
            target_index=fold.target_index[:5],
        )
    with pytest.raises(ValueError, match="at or before the cutoff"):
        Fold(
            fold=1,
            target_date=fold.target_date,
            cutoff=fold.target_index[0],
            target_index=fold.target_index,
        )


# --- eligibility --------------------------------------------------------------- #
def test_quarantined_days_are_never_fold_targets(features: pd.DataFrame, folds) -> None:
    eligible = eligible_op_dates(features)
    bad_days = set(features.loc[features["bad_day"] == 1, "op_date"])
    assert not (eligible & bad_days)
    assert not ({f.target_date for f in folds} & bad_days)


def test_eligible_days_have_all_inputs_present(features: pd.DataFrame) -> None:
    from src.features import FEATURE_COLS

    eligible = eligible_op_dates(features)
    sample = features[features["op_date"].isin(list(eligible))]
    assert sample[list(FEATURE_COLS)].notna().all().all()


# --- running ------------------------------------------------------------------- #
def test_run_fold_returns_one_prediction_per_target_hour(features: pd.DataFrame, folds) -> None:
    y_pred, elapsed = run_fold(NaiveModel(), features, folds[0])
    assert y_pred.shape == (HORIZON,)
    assert np.isfinite(y_pred).all()
    assert elapsed >= 0


class _DivergingModel:
    """Stands in for an unconstrained ARIMA whose forecast grows geometrically."""

    name = "diverging"

    def fit(self, history: pd.DataFrame) -> None:
        self._level = float(history["y"].iloc[-1])

    def predict(self, horizon_index, future_exog=None) -> np.ndarray:
        return self._level * np.power(2.0, np.arange(1, len(horizon_index) + 1))


class _NanModel(_DivergingModel):
    name = "nan"

    def predict(self, horizon_index, future_exog=None) -> np.ndarray:
        return np.full(len(horizon_index), np.nan)


def test_run_fold_rejects_a_diverging_forecast(features: pd.DataFrame, folds) -> None:
    """A blow-up is finite and absurd -- isfinite alone would let 1e60 MW through."""
    with pytest.raises(ValueError, match="diverged"):
        run_fold(_DivergingModel(), features, folds[0])


def test_run_fold_rejects_non_finite_predictions(features: pd.DataFrame, folds) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        run_fold(_NanModel(), features, folds[0])


def test_plausibility_band_admits_a_real_forecast(features: pd.DataFrame, folds) -> None:
    """The guard must not be so tight that an honest model trips it."""
    from src.backtest import PLAUSIBLE_HIGH, PLAUSIBLE_LOW

    for fold in folds:
        history = features.loc[features.index <= fold.cutoff, "y"]
        y_pred, _ = run_fold(SeasonalNaiveModel(), features, fold)
        assert (y_pred > history.min() * PLAUSIBLE_LOW).all()
        assert (y_pred < history.max() * PLAUSIBLE_HIGH).all()


def test_naive_reproduces_the_previous_day(features: pd.DataFrame, folds) -> None:
    fold = folds[3]
    y_pred, _ = run_fold(NaiveModel(), features, fold)
    expected = features.loc[fold.target_index - pd.Timedelta(hours=24), "y"].to_numpy()
    assert y_pred == pytest.approx(expected)


def test_snaive_reproduces_the_previous_week(features: pd.DataFrame, folds) -> None:
    fold = folds[3]
    y_pred, _ = run_fold(SeasonalNaiveModel(), features, fold)
    expected = features.loc[fold.target_index - pd.Timedelta(hours=168), "y"].to_numpy()
    assert y_pred == pytest.approx(expected)


def test_results_frame_shape_and_truth_alignment(features: pd.DataFrame, folds) -> None:
    results = run_backtest("snaive", features, folds, verbose=False)
    assert len(results) == len(folds) * HORIZON
    assert results.groupby("fold").size().eq(HORIZON).all()
    assert set(results["lead"]) == set(range(1, HORIZON + 1))

    merged = results.set_index("ts")["y_true"]
    assert merged.to_numpy() == pytest.approx(features.loc[merged.index, "y"].to_numpy())


def test_every_registered_model_name_resolves() -> None:
    for name, factory in MODEL_REGISTRY.items():
        if name == "lstm":
            continue  # optional torch dependency, covered by its own test
        assert factory().name == name


# --- metrics ------------------------------------------------------------------- #
def test_mase_of_the_reference_against_itself_is_one(features: pd.DataFrame, folds) -> None:
    results = run_backtest("snaive", features, folds, verbose=False)
    assert mase(results, results) == pytest.approx(1.0)


def test_naive_and_snaive_are_scored_on_identical_points(features: pd.DataFrame, folds) -> None:
    a = run_backtest("naive", features, folds, verbose=False)
    b = run_backtest("snaive", features, folds, verbose=False)
    pd.testing.assert_series_equal(a["ts"], b["ts"])
    assert a["y_true"].to_numpy() == pytest.approx(b["y_true"].to_numpy())


def test_mase_rejects_a_misaligned_reference(features: pd.DataFrame, folds) -> None:
    results = run_backtest("snaive", features, folds, verbose=False)
    with pytest.raises(ValueError):
        mase(results, results[results["fold"] > 1])


def test_fold_maes_are_per_fold(features: pd.DataFrame, folds) -> None:
    results = run_backtest("naive", features, folds, verbose=False)
    maes = fold_maes(results)
    assert len(maes) == len(folds)
    assert (maes > 0).all()


def test_compare_models_detects_a_real_difference(features: pd.DataFrame, folds) -> None:
    from src.metrics import compare_models

    naive = run_backtest("naive", features, folds, verbose=False)
    snaive_res = run_backtest("snaive", features, folds, verbose=False)
    result = compare_models(naive, snaive_res, "naive", "snaive")

    assert result["n_folds"] == len(folds)
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["mean_diff"] == pytest.approx(result["mae_a"] - result["mae_b"], rel=1e-6)


def test_compare_models_finds_no_difference_against_itself(features: pd.DataFrame, folds) -> None:
    """Identical inputs must not produce a significant result."""
    from src.metrics import compare_models

    results = run_backtest("snaive", features, folds, verbose=False)
    comparison = compare_models(results, results, "a", "b")
    assert comparison["mean_diff"] == pytest.approx(0.0)
    assert comparison["p_value"] == 1.0
    assert not comparison["significant_at_05"]


def test_compare_models_rejects_mismatched_folds(features: pd.DataFrame, folds) -> None:
    from src.metrics import compare_models

    full = run_backtest("naive", features, folds, verbose=False)
    partial = full[full["fold"] > 1]
    with pytest.raises(ValueError, match="different folds"):
        compare_models(full, partial, "a", "b")


def test_summarize_reports_every_headline_metric(features: pd.DataFrame, folds) -> None:
    results = run_backtest("naive", features, folds, verbose=False)
    reference = run_backtest("snaive", features, folds, verbose=False)
    row = summarize(results, reference)
    assert set(row) == {"MAPE", "RMSE", "MAE", "MASE", "n_folds", "n_points"}
    assert row["n_folds"] == len(folds)
    assert row["n_points"] == len(folds) * HORIZON
    assert 0 < row["MAPE"] < 1


def test_build_summary_skips_models_without_results(tmp_path, monkeypatch) -> None:
    import src.backtest as bt

    monkeypatch.setattr(bt, "RESULTS_DIR", tmp_path)
    assert build_summary(["naive", "lgbm"]).empty
