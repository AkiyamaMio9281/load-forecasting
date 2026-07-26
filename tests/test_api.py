"""Forecast service contract (SPEC section 8 / test matrix section 9).

The app is exercised against a small artifact built from the synthetic fixture, so
these tests need neither Kaggle data nor a trained production model.
"""

from __future__ import annotations

import datetime as dt

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import HORIZON
from src.features import build_features
from src.models.lgbm import LgbmDirectModel


@pytest.fixture(scope="module")
def client(clean_df: pd.DataFrame, tmp_path_factory, monkeypatch=None):
    """A live app backed by a miniature artifact trained on the fixture series."""
    import os

    features = build_features(clean_df)
    model = LgbmDirectModel(seed=42, params={"n_estimators": 40})
    model.fit(features)

    artifact = {
        "artifact_version": 1,
        "model_name": "lgbm",
        "model": model,
        "clean": clean_df,
        "trained_through": features.index.max(),
        "trained_at": "2018-01-01T00:00:00+00:00",
        "seed": 42,
        "backtest_metrics": {"MAPE": 0.031, "MASE": 0.62},
    }
    path = tmp_path_factory.mktemp("artifacts") / "lgbm.joblib"
    joblib.dump(artifact, path)

    os.environ["FORECAST_ARTIFACT"] = str(path)
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    os.environ.pop("FORECAST_ARTIFACT", None)


# --- health -------------------------------------------------------------------- #
def test_health_reports_model_and_coverage(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "lgbm"
    assert len(body["available_dates"]) == 2
    assert body["backtest_metrics"]["MASE"] == pytest.approx(0.62)


# --- happy path ---------------------------------------------------------------- #
def test_forecast_returns_24_hourly_points(client) -> None:
    target = client.get("/health").json()["available_dates"][1]
    response = client.post("/forecast", json={"date": target})
    assert response.status_code == 200

    body = response.json()
    assert body["date"] == target
    assert body["model"] == "lgbm"
    assert len(body["predictions"]) == HORIZON


def test_forecast_timestamps_are_consecutive_utc_hours(client) -> None:
    target = client.get("/health").json()["available_dates"][1]
    body = client.post("/forecast", json={"date": target}).json()

    stamps = pd.to_datetime([p["ts"] for p in body["predictions"]], utc=True)
    assert list(stamps.to_series().diff().dropna().unique()) == [pd.Timedelta("1h")]
    assert stamps[0].hour == 5  # operating day opens at 05:00 UTC


def test_predicted_load_is_physically_plausible(client) -> None:
    target = client.get("/health").json()["available_dates"][1]
    body = client.post("/forecast", json={"date": target}).json()
    values = [p["mw"] for p in body["predictions"]]
    assert all(10_000 < v < 70_000 for v in values), values


def test_forecast_is_deterministic(client) -> None:
    target = client.get("/health").json()["available_dates"][1]
    first = client.post("/forecast", json={"date": target}).json()
    second = client.post("/forecast", json={"date": target}).json()
    assert first == second


# --- rejections ----------------------------------------------------------------- #
def test_malformed_date_is_422(client) -> None:
    assert client.post("/forecast", json={"date": "not-a-date"}).status_code == 422
    assert client.post("/forecast", json={"date": "2018-13-45"}).status_code == 422


def test_missing_body_field_is_422(client) -> None:
    assert client.post("/forecast", json={}).status_code == 422


def test_date_outside_coverage_is_422(client) -> None:
    response = client.post("/forecast", json={"date": "1999-01-01"})
    assert response.status_code == 422
    assert "outside the covered range" in response.json()["detail"]


def test_date_before_warmup_is_rejected(client) -> None:
    """The first week has no lag_168, so it must not be servable."""
    first_servable = dt.date.fromisoformat(client.get("/health").json()["available_dates"][0])
    too_early = first_servable - dt.timedelta(days=1)
    assert client.post("/forecast", json={"date": too_early.isoformat()}).status_code == 422


def test_openapi_documents_the_forecast_route(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/forecast" in schema["paths"]
    assert "post" in schema["paths"]["/forecast"]
