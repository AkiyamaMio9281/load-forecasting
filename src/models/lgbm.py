"""LightGBM, direct multi-horizon strategy (SPEC section 5) -- the main model.

**Direct, not recursive.** We train 24 independent regressors, one per lead
h = 1..24, each predicting `y(cutoff + h)` from features known at the cutoff.
The alternative -- one model applied recursively, feeding its own output back in
as `lag_24` -- compounds its errors across the horizon and, worse, would need
predicted lags at prediction time while it saw *true* lags during training. Direct
models never face that train/serve mismatch: every horizon is a plain supervised
problem on the exact feature vector it will see in production. The cost is 24x the
training work, which at ~6k rows per model is a few seconds.

Because each lead model only sees one row per operating day, its natural training
set is ~1/24 of the table -- so an "N-day" validation window is N *rows*, not 24N.
That is why `val_days` is generous compared with a recursive setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZON
from src.features import FEATURE_COLS
from src.models.base import BaseModel

PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbose": -1,
    "n_jobs": -1,
}


class LgbmDirectModel(BaseModel):
    """One LightGBM regressor per lead; `predict` stitches the 24 outputs together."""

    name = "lgbm"

    def __init__(self, seed: int = 42, val_days: int = 90, params: dict | None = None) -> None:
        super().__init__(seed=seed)
        self.val_days = val_days
        self.params = {**PARAMS, "random_state": seed, **(params or {})}
        self.models_: dict[int, object] = {}
        self.best_iterations_: dict[int, int] = {}

    def fit(self, history: pd.DataFrame) -> None:
        import lightgbm as lgb

        train = history[(history["bad_day"] == 0) & history[list(FEATURE_COLS)].notna().all(axis=1)]
        if train.empty:
            raise ValueError("lgbm: no usable training rows before the cutoff")

        self.models_.clear()
        self.best_iterations_.clear()
        for lead in range(1, HORIZON + 1):
            subset = train[train["lead"] == lead]
            x = subset[list(FEATURE_COLS)]
            y = subset["y"]

            # Chronological holdout -- the last `val_days` rows of this lead.
            n_val = min(self.val_days, max(0, len(subset) // 5))
            model = lgb.LGBMRegressor(**self.params)
            if n_val >= 10:
                model.fit(
                    x.iloc[:-n_val],
                    y.iloc[:-n_val],
                    eval_set=[(x.iloc[-n_val:], y.iloc[-n_val:])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                self.best_iterations_[lead] = int(
                    model.best_iteration_ or self.params["n_estimators"]
                )
            else:
                model.fit(x, y)
                self.best_iterations_[lead] = int(self.params["n_estimators"])
            self.models_[lead] = model

    def predict(
        self, horizon_index: pd.DatetimeIndex, future_exog: pd.DataFrame | None
    ) -> np.ndarray:
        if future_exog is None:
            raise ValueError("lgbm requires future_exog (the feature rows of the target day)")
        if not self.models_:
            raise RuntimeError("lgbm: predict called before fit")

        x = future_exog[list(FEATURE_COLS)]
        # Row i of the horizon is lead i+1 by construction of the operating day.
        return np.array(
            [
                self.models_[lead].predict(x.iloc[[i]])[0]
                for i, lead in enumerate(range(1, len(horizon_index) + 1))
            ],
            dtype=float,
        )

    def feature_importance(self) -> pd.Series:
        """Gain importance averaged over the 24 lead models, as a share of total."""
        if not self.models_:
            raise RuntimeError("lgbm: feature_importance called before fit")
        gains = np.mean(
            [m.booster_.feature_importance(importance_type="gain") for m in self.models_.values()],
            axis=0,
        )
        series = pd.Series(gains, index=list(FEATURE_COLS), name="gain")
        return (series / series.sum()).sort_values(ascending=False)
