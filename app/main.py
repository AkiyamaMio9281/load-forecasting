"""FastAPI day-ahead forecast service (SPEC section 8).

    POST /forecast  {"date": "2018-07-01"}
      -> {"date": ..., "model": "lgbm", "predictions": [{"ts": ..., "mw": ...} x 24]}

Cold-start design (design doc pitfall 5): the artifact is loaded and the feature
table built **once during lifespan startup**, not per request. Cloud Run reuses a
warm process across requests, so paying ~2s once turns into ~5ms per forecast;
doing it in the handler would pay it every time.

Scope note, stated in the response itself rather than buried: this serves a
historical dataset. The shipped model is retrained through the end of that series,
so a requested date inside the training span is in-sample. `trained_through` is
returned with every forecast so a caller can see exactly where they stand.
"""

from __future__ import annotations

import datetime as dt
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import ARTIFACTS_DIR, HORIZON
from src.features import FEATURE_COLS, build_features, target_index_for

DEFAULT_ARTIFACT = ARTIFACTS_DIR / "lgbm.joblib"


def artifact_path() -> Path:
    """Env override exists so tests can point at a small fixture artifact."""
    return Path(os.environ.get("FORECAST_ARTIFACT", DEFAULT_ARTIFACT))


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ForecastRequest(BaseModel):
    # `dt.date` rather than a bare `date`: a field named `date` annotated with a
    # type also named `date` is unresolvable to pydantic.
    date: dt.date = Field(..., description="Operating day to forecast, YYYY-MM-DD")


class HourlyPrediction(BaseModel):
    ts: str = Field(..., description="UTC timestamp of the hour")
    mw: float = Field(..., description="Predicted load in megawatts")


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    date: dt.date
    model: str
    trained_through: str
    predictions: list[HourlyPrediction]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model: str
    trained_through: str
    available_dates: list[dt.date]
    backtest_metrics: dict


# --------------------------------------------------------------------------- #
# Lifespan: load once, serve many
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    path = artifact_path()
    if not path.exists():
        raise RuntimeError(
            f"no serving artifact at {path}. Build one with: "
            "conda run -n p3 python -m src.train_final --model lgbm"
        )
    artifact = joblib.load(path)

    state = app.state
    state.artifact = artifact
    state.model = artifact["model"]
    state.model_name = artifact["model_name"]
    state.trained_through = pd.Timestamp(artifact["trained_through"])
    # Same code path as training -- see src/train_final.py for why this matters.
    state.features = build_features(artifact["clean"])

    op_dates = pd.Series(state.features["op_date"])
    complete = op_dates.value_counts()
    usable = state.features[state.features[list(FEATURE_COLS)].notna().all(axis=1)]
    state.servable_dates = sorted(set(usable["op_date"]) & set(complete[complete == HORIZON].index))
    yield


app = FastAPI(
    title="PJM day-ahead load forecast",
    description="24-hour-ahead hourly electricity load forecast for the PJM East zone.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    servable = app.state.servable_dates
    return HealthResponse(
        status="ok",
        model=app.state.model_name,
        trained_through=str(app.state.trained_through),
        available_dates=[servable[0], servable[-1]],
        backtest_metrics=app.state.artifact.get("backtest_metrics", {}),
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    servable = app.state.servable_dates
    if request.date not in set(servable):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{request.date} cannot be forecast: outside the covered range "
                f"{servable[0]}..{servable[-1]}, or the day lacks complete inputs"
            ),
        )

    horizon_index = target_index_for(request.date)
    future_exog = app.state.features.loc[horizon_index, list(FEATURE_COLS)]
    predictions = app.state.model.predict(horizon_index, future_exog)

    return ForecastResponse(
        date=request.date,
        model=app.state.model_name,
        trained_through=str(app.state.trained_through),
        predictions=[
            HourlyPrediction(ts=ts.isoformat(), mw=round(float(mw), 1))
            for ts, mw in zip(horizon_index, predictions, strict=False)
        ],
    )
