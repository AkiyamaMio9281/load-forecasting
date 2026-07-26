"""Retrain the winning model on the full series and package it for serving.

The backtest answers "which model, and how good"; this answers "what do we ship".
The artifact bundles three things:

  * the fitted model,
  * the **clean input series** it needs to derive features for a requested date, and
  * provenance -- what it was trained through, and the backtest numbers it earned.

Shipping the series rather than a pre-computed feature matrix is deliberate: the
service then calls the very same `build_features` the backtest used, so there is no
second implementation to drift out of sync. That mismatch -- features built one way
in training and another way in production -- is the classic way a good offline
model becomes a bad online one.

Usage:
    conda run -n p3 python -m src.train_final --model lgbm
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import pandas as pd

from src.backtest import MODEL_REGISTRY, results_path
from src.config import ARTIFACTS_DIR, CLEAN_PARQUET, RESULTS_DIR, SEED, ensure_dirs
from src.features import build_features

ARTIFACT_VERSION = 1


def build_artifact(model_name: str = "lgbm") -> dict:
    """Fit `model_name` on every usable hour and return the servable bundle."""
    clean = pd.read_parquet(CLEAN_PARQUET)
    features = build_features(clean)

    model = MODEL_REGISTRY[model_name]()
    model.fit(features)

    backtest_metrics = {}
    summary_path = RESULTS_DIR / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path).set_index("model")
        if model_name in summary.index:
            backtest_metrics = summary.loc[model_name].to_dict()

    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": model_name,
        "model": model,
        "clean": clean,
        "trained_through": features.index.max(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "backtest_metrics": backtest_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lgbm", choices=sorted(MODEL_REGISTRY))
    args = parser.parse_args()

    ensure_dirs()
    artifact = build_artifact(args.model)
    path = ARTIFACTS_DIR / f"{args.model}.joblib"
    joblib.dump(artifact, path, compress=3)

    size_mb = path.stat().st_size / 1e6
    print("=== Serving artifact ===")
    print(f"model           : {artifact['model_name']}")
    print(f"trained through : {artifact['trained_through']}")
    print(f"history rows    : {len(artifact['clean']):,}")
    print(f"backtest        : {json.dumps(artifact['backtest_metrics'], default=str)}")
    print(f"written         : {path}  ({size_mb:.1f} MB)")
    if not results_path(args.model).exists():
        print("[warn] no backtest results on disk for this model -- provenance is thin")


if __name__ == "__main__":
    main()
