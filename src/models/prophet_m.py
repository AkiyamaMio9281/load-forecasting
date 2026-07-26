"""Prophet (SPEC section 5).

Prophet's pitch for this problem is that it states its structure out loud --
trend + daily + weekly + yearly seasonality + a holiday effect -- so a stakeholder
can read why a forecast moved. It is included for that interpretability and as a
mid-strength reference point, not because it is expected to win.

Two operational notes:
  * Prophet insists on naive timestamps, so we strip the UTC tz on the way in and
    put it back on the way out. Everything stays UTC; only the label changes.
  * `max_train_years` caps the window. Prophet's MAP fit cost grows with sample
    count, and 145k hourly points x 52 folds is hours of wall clock for a model
    that has already saturated on 3 years of seasonal shape.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.base import BaseModel

# Prophet/cmdstan log a line per fit; 52 folds of that buries the real output.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetModel(BaseModel):
    """Additive decomposition with daily/weekly/yearly seasonality and US holidays."""

    name = "prophet"
    max_train_years: float | None = 3.0

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.model_ = None

    def fit(self, history: pd.DataFrame) -> None:
        from prophet import Prophet

        history = self._truncate(history)
        y = self._masked_y(history)
        frame = pd.DataFrame({"ds": history.index.tz_localize(None), "y": y.to_numpy()}).dropna()

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode="additive",
            uncertainty_samples=0,  # we only score the point forecast
        )
        model.add_country_holidays(country_name="US")
        model.fit(frame)
        self.model_ = model

    def predict(
        self, horizon_index: pd.DatetimeIndex, future_exog: pd.DataFrame | None = None
    ) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("prophet: predict called before fit")
        future = pd.DataFrame({"ds": horizon_index.tz_localize(None)})
        return self.model_.predict(future)["yhat"].to_numpy(dtype=float)
