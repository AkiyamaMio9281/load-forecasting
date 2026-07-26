"""The two baselines every other model has to beat.

Both are pure index arithmetic on the observed series -- no fitting, no parameters.
`snaive` doubles as the MASE denominator, so its correctness matters more than its
sophistication: a bug here silently rescales every model's headline number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class _LagBaseline(BaseModel):
    """Predict y(t) = y(t - lag). Requires the lag to reach past the cutoff."""

    lag_hours: int

    def fit(self, history: pd.DataFrame) -> None:
        self._history = history["y"]

    def predict(
        self, horizon_index: pd.DatetimeIndex, future_exog: pd.DataFrame | None = None
    ) -> np.ndarray:
        source = horizon_index - pd.Timedelta(hours=self.lag_hours)
        values = self._history.reindex(source)
        if values.isna().any():
            raise ValueError(
                f"{self.name}: history lacks {int(values.isna().sum())} of the "
                f"{len(source)} timestamps at lag {self.lag_hours}h"
            )
        return values.to_numpy(dtype=float)


class NaiveModel(_LagBaseline):
    """Yesterday, same hour."""

    name = "naive"
    lag_hours = 24


class SeasonalNaiveModel(_LagBaseline):
    """Last week, same hour and weekday -- the MASE reference."""

    name = "snaive"
    lag_hours = 168
