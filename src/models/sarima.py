"""ARIMA with Fourier seasonal terms (SPEC section 5).

A textbook SARIMA for this series would want s=168 to capture the weekly cycle.
That is unusable: the state vector grows with s, so a seasonal order of 168 turns
one fit into hours. The standard workaround -- and the one the design doc calls
for -- is to move seasonality out of the AR/MA structure and into **deterministic
Fourier regressors**: K=3 harmonics of the 24-hour cycle plus K=3 of the 168-hour
cycle, 12 exogenous columns, fed to a plain ARIMA(2,0,2). The ARIMA part then only
has to model short-range dependence in the deseasonalised residual, which it can
do cheaply.

Order is fixed across folds by design (SPEC: "no re-identification per fold") --
re-running an information criterion inside every fold would let model selection
see 52 different datasets and quietly turn the backtest into a search.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models.base import BaseModel

DAILY_PERIOD = 24
WEEKLY_PERIOD = 168
N_HARMONICS = 3
ORDER = (2, 0, 2)
# Fourier phase is measured from a fixed epoch so train and forecast rows share it.
EPOCH = pd.Timestamp("2000-01-01", tz="UTC")


def fourier_terms(
    index: pd.DatetimeIndex,
    periods: tuple[int, ...] = (DAILY_PERIOD, WEEKLY_PERIOD),
    n_harmonics: int = N_HARMONICS,
) -> pd.DataFrame:
    """sin/cos harmonics of each period, phased from a fixed epoch."""
    hours = (index - EPOCH) / pd.Timedelta(hours=1)
    cols: dict[str, np.ndarray] = {}
    for period in periods:
        for k in range(1, n_harmonics + 1):
            angle = 2 * np.pi * k * hours / period
            cols[f"sin_{period}_{k}"] = np.sin(angle)
            cols[f"cos_{period}_{k}"] = np.cos(angle)
    return pd.DataFrame(cols, index=index)


class SarimaFourierModel(BaseModel):
    """ARIMA(2,0,2) on the load level with Fourier seasonal exogenous terms.

    `max_train_years` is the design doc's escape hatch for fit cost: the Kalman
    filter is linear in sample size, and 16 years x 52 folds is not worth the
    marginal accuracy of pre-2015 data for a day-ahead horizon. The truncation is
    reported alongside the results.
    """

    name = "sarima"
    max_train_years: float | None = 2.0

    def __init__(self, seed: int = 42, order: tuple[int, int, int] = ORDER) -> None:
        super().__init__(seed=seed)
        self.order = order
        self.result_ = None

    def fit(self, history: pd.DataFrame) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        history = self._truncate(history)
        y = self._masked_y(history)  # NaN on quarantined days; ARIMA treats them as missing
        exog = fourier_terms(history.index)

        with warnings.catch_warnings():
            # Convergence chatter on a 17k-point series is noise, not information.
            warnings.simplefilter("ignore")
            # enforce_stationarity is NOT optional here. Relaxing it lets the optimiser
            # put an AR root inside the unit circle, and the forecast then grows
            # geometrically -- observed as predictions reaching 1e60 MW on one dataset
            # while looking perfectly sane on another. A constrained fit that converges
            # slightly worse beats an unconstrained one that occasionally detonates.
            model = ARIMA(
                y,
                exog=exog,
                order=self.order,
                trend="c",
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            self.result_ = model.fit(method_kwargs={"warn_convergence": False})

    def predict(
        self, horizon_index: pd.DatetimeIndex, future_exog: pd.DataFrame | None = None
    ) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("sarima: predict called before fit")
        exog = fourier_terms(horizon_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.result_.forecast(steps=len(horizon_index), exog=exog)
        return np.asarray(forecast, dtype=float)
