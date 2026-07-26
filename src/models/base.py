"""The one interface the backtest framework knows about (SPEC section 5).

Every model -- a two-line naive rule and a 24-model LightGBM ensemble alike -- is
refit from scratch inside each fold and asked for exactly `HORIZON` numbers. The
backtest loop therefore contains no per-model branching at all, which is the whole
point: adding a model is adding a file, not editing the harness.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ForecastModel(Protocol):
    """Fit on history up to the cutoff, predict the next operating day."""

    name: str

    def fit(self, history: pd.DataFrame) -> None:
        """`history`: feature table rows with ts <= cutoff, including column `y`."""
        ...

    def predict(
        self, horizon_index: pd.DatetimeIndex, future_exog: pd.DataFrame | None
    ) -> np.ndarray:
        """Return one prediction per timestamp in `horizon_index` (length 24)."""
        ...


class BaseModel:
    """Shared plumbing: name, seed, and an optional training-window cap.

    `max_train_years` exists for the statistical models. SARIMA and Prophet refit
    on an expanding window across 52 folds cost hours on a 145k-point series; the
    design doc's fallback is to shorten their window and say so in the report.
    `None` means the full expanding window mandated by SPEC section 4.
    """

    name: str = "base"
    max_train_years: float | None = None

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _truncate(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.max_train_years is None:
            return history
        start = history.index.max() - pd.Timedelta(days=365.25 * self.max_train_years)
        return history.loc[history.index >= start]

    @staticmethod
    def _masked_y(history: pd.DataFrame) -> pd.Series:
        """Target series with quarantined days blanked out.

        Days behind a long gap carry interpolated values so the parquet stays gap
        free; letting a model *train* on them would be fitting our own straight
        lines. Returning NaN keeps the index contiguous (which the state-space
        models need) while telling them the value is unknown.
        """
        y = history["y"].astype(float).copy()
        y[history["bad_day"].to_numpy() == 1] = np.nan
        return y

    def fit(self, history: pd.DataFrame) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(
        self, horizon_index: pd.DatetimeIndex, future_exog: pd.DataFrame | None
    ) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
