"""Error metrics and grouped error reports (SPEC section 6).

MASE is the metric that carries the comparison: it divides a model's MAE by the
MAE of seasonal naive on *the same target points*, so 1.0 is the "no better than
last week" line and the number is comparable across series and seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SNAIVE_REF = "snaive"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error. Load is strictly positive, so no guard needed."""
    return float(np.mean(np.abs(y_true - y_pred) / y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def fold_maes(results: pd.DataFrame) -> pd.Series:
    """MAE within each fold, indexed by fold id."""
    err = (results["y_true"] - results["y_pred"]).abs()
    return err.groupby(results["fold"]).mean()


def mase(results: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Global MASE: per-fold MAEs summed, then divided (SPEC section 6).

    `reference` must be the seasonal-naive backtest over the identical fold/target
    grid -- we assert that rather than trusting it, because a silently misaligned
    denominator would flatter every model.
    """
    model_maes = fold_maes(results)
    ref_maes = fold_maes(reference)
    if not model_maes.index.equals(ref_maes.index):
        raise ValueError("MASE reference covers different folds than the model")
    if len(results) != len(reference):
        raise ValueError("MASE reference covers a different number of target points")
    return float(model_maes.sum() / ref_maes.sum())


def summarize(results: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
    """Headline metrics for one model's backtest output."""
    y_true = results["y_true"].to_numpy()
    y_pred = results["y_pred"].to_numpy()
    row = {
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "n_folds": int(results["fold"].nunique()),
        "n_points": int(len(results)),
    }
    row["MASE"] = mase(results, reference) if reference is not None else np.nan
    return row


def compare_models(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """Is `a` really better than `b`, or is the gap fold-to-fold noise?

    A headline table invites reading a 0.02pp MAPE gap as a ranking. With 52 folds
    that gap is usually indistinguishable from zero. This runs a paired test on the
    per-fold MAE differences -- paired because both models forecast the *same* target
    points, which removes fold difficulty from the comparison and is far more
    sensitive than comparing two independent means.

    Wilcoxon signed-rank rather than a t-test: 52 fold errors are right-skewed
    (a few extreme-weather days dominate), so normality is not a safe assumption.
    """
    from scipy import stats

    mae_a, mae_b = fold_maes(a), fold_maes(b)
    if not mae_a.index.equals(mae_b.index):
        raise ValueError("cannot compare models scored on different folds")

    diff = mae_a - mae_b  # negative => a is better
    wins = int((diff < 0).sum())

    if (diff == 0).all():
        # Identical forecasts. scipy raises rather than returning a p-value here, but
        # the answer is well defined: no evidence whatsoever of a difference.
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = stats.wilcoxon(mae_a, mae_b)

    return {
        "model_a": name_a,
        "model_b": name_b,
        "mae_a": float(mae_a.mean()),
        "mae_b": float(mae_b.mean()),
        "mean_diff": float(diff.mean()),
        "a_wins_folds": wins,
        "n_folds": len(diff),
        "wilcoxon_stat": float(statistic),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
    }


def grouped_errors(results: pd.DataFrame, by: str) -> pd.DataFrame:
    """MAPE / RMSE / MAE broken out by a column of `results` (hour, month, ...)."""

    def _agg(g: pd.DataFrame) -> pd.Series:
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        return pd.Series(
            {"MAPE": mape(yt, yp), "RMSE": rmse(yt, yp), "MAE": mae(yt, yp), "n": len(g)}
        )

    return results.groupby(by, observed=True).apply(_agg, include_groups=False)
