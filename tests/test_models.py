"""Every model honours the same interface (SPEC section 5).

The backtest harness has no per-model branches, so it can only be as trustworthy as
this contract. One fold per model on the synthetic fixture is enough to catch the
failures that matter: wrong output length, NaNs, a fit that mutates shared state, or
a predict that quietly ignores the horizon it was handed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import MODEL_REGISTRY, make_folds
from src.config import HORIZON
from src.features import FEATURE_COLS
from src.models.base import ForecastModel
from src.models.sarima import fourier_terms

torch = pytest.importorskip  # referenced lazily in the lstm test below

ALWAYS_AVAILABLE = ["naive", "snaive", "sarima", "prophet", "lgbm"]


@pytest.fixture(scope="module")
def fold(features: pd.DataFrame):
    return make_folds(features, n_folds=1)[0]


@pytest.fixture(scope="module")
def fold_inputs(features: pd.DataFrame, fold):
    history = features.loc[features.index <= fold.cutoff]
    exog = features.loc[fold.target_index, list(FEATURE_COLS)]
    return history, exog


@pytest.mark.parametrize("name", ALWAYS_AVAILABLE)
def test_model_satisfies_the_protocol(name: str) -> None:
    model = MODEL_REGISTRY[name]()
    assert isinstance(model, ForecastModel)
    assert model.name == name


@pytest.mark.parametrize("name", ALWAYS_AVAILABLE)
def test_model_returns_24_finite_predictions(name: str, fold, fold_inputs) -> None:
    history, exog = fold_inputs
    model = MODEL_REGISTRY[name]()
    model.fit(history)
    y_pred = np.asarray(model.predict(fold.target_index, exog), dtype=float)

    assert y_pred.shape == (HORIZON,)
    assert np.isfinite(y_pred).all()
    # Sanity band -- a model that returns the right shape but nonsense values is
    # still broken, and every fixture day sits well inside this range.
    assert (y_pred > 5_000).all() and (y_pred < 80_000).all()


@pytest.mark.parametrize("name", ALWAYS_AVAILABLE)
def test_predict_before_fit_raises(name: str, fold, fold_inputs) -> None:
    _, exog = fold_inputs
    model = MODEL_REGISTRY[name]()
    with pytest.raises((RuntimeError, AttributeError, ValueError)):
        model.predict(fold.target_index, exog)


@pytest.mark.parametrize("name", ALWAYS_AVAILABLE)
def test_refitting_is_reproducible(name: str, fold, fold_inputs) -> None:
    history, exog = fold_inputs
    first = MODEL_REGISTRY[name]()
    first.fit(history)
    second = MODEL_REGISTRY[name]()
    second.fit(history)
    assert first.predict(fold.target_index, exog) == pytest.approx(
        second.predict(fold.target_index, exog), rel=1e-6
    )


# --- model-specific properties ---------------------------------------------- #
def test_baselines_refuse_to_guess_when_history_is_short(features: pd.DataFrame, fold) -> None:
    from src.models.naive import SeasonalNaiveModel

    model = SeasonalNaiveModel()
    model.fit(features.loc[features.index <= fold.cutoff].iloc[-24:])
    with pytest.raises(ValueError, match="lag"):
        model.predict(fold.target_index, None)


def test_fourier_terms_have_the_expected_shape_and_periodicity() -> None:
    index = pd.date_range("2017-01-01", periods=400, freq="h", tz="UTC")
    terms = fourier_terms(index)
    assert terms.shape == (400, 12)  # 2 periods x 3 harmonics x sin/cos
    assert (terms.abs().max() <= 1.0 + 1e-12).all()

    daily = terms["sin_24_1"].to_numpy()
    assert daily[:100] == pytest.approx(daily[24:124], abs=1e-9)


def test_sarima_truncates_its_training_window(features: pd.DataFrame, fold) -> None:
    from src.models.sarima import SarimaFourierModel

    model = SarimaFourierModel()
    history = features.loc[features.index <= fold.cutoff]
    truncated = model._truncate(history)
    assert len(truncated) < len(history)
    span_years = (truncated.index.max() - truncated.index.min()).days / 365.25
    assert span_years == pytest.approx(model.max_train_years, abs=0.02)


def test_lgbm_trains_one_model_per_lead(fold_inputs) -> None:
    from src.models.lgbm import LgbmDirectModel

    history, _ = fold_inputs
    model = LgbmDirectModel(params={"n_estimators": 30})
    model.fit(history)
    assert sorted(model.models_) == list(range(1, HORIZON + 1))


def test_lgbm_feature_importance_covers_every_feature(fold_inputs) -> None:
    from src.models.lgbm import LgbmDirectModel

    history, _ = fold_inputs
    model = LgbmDirectModel(params={"n_estimators": 30})
    model.fit(history)
    importance = model.feature_importance()
    assert set(importance.index) == set(FEATURE_COLS)
    assert importance.sum() == pytest.approx(1.0)
    assert importance.is_monotonic_decreasing


def test_lgbm_rejects_a_missing_exog_block(fold, fold_inputs) -> None:
    from src.models.lgbm import LgbmDirectModel

    history, _ = fold_inputs
    model = LgbmDirectModel(params={"n_estimators": 30})
    model.fit(history)
    with pytest.raises(ValueError, match="future_exog"):
        model.predict(fold.target_index, None)


def test_lstm_matches_the_interface(fold, fold_inputs) -> None:
    pytest.importorskip("torch", reason="lstm is an optional extra")
    from src.models.lstm import LstmModel

    history, exog = fold_inputs
    model = LstmModel(seed=42, epochs=2)
    model.fit(history)
    y_pred = model.predict(fold.target_index, exog)
    assert np.asarray(y_pred).shape == (HORIZON,)
    assert np.isfinite(y_pred).all()
