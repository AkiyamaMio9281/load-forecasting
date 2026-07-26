"""Rolling-origin backtest harness (SPEC section 4).

    for each of the last 52 weeks:
        cutoff  = 04:00 UTC on the target operating day
        train   = everything at or before the cutoff        (expanding window)
        predict = the 24 hours of the target operating day
        step back 7 days and repeat

Two properties make the numbers trustworthy, and both are enforced here rather
than left to each model:

1. **The model never sees past the cutoff.** `history` is sliced by timestamp, and
   `future_exog` only carries columns whose leakage contract is verified in
   tests/test_no_leakage.py. A model physically cannot reach the target values.
2. **Every model is scored on identical target points.** The fold list is built
   once from the data, not per model, so MASE denominators line up by
   construction and `metrics.mase` re-asserts it.

Usage:
    conda run -n p3 python -m src.backtest --models naive,snaive,sarima,prophet,lgbm
    conda run -n p3 python -m src.backtest --models lgbm --folds 3   # smoke
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics
from src.config import (
    FEATURES_PARQUET,
    FOLD_STEP_DAYS,
    HORIZON,
    N_FOLDS,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from src.features import FEATURE_COLS, cutoff_of_op_date, target_index_for
from src.models.base import BaseModel

# Multiples of the observed training range that a forecast may not leave. Wide enough
# that no sane model trips them, tight enough to catch a divergence immediately.
PLAUSIBLE_LOW = 0.25
PLAUSIBLE_HIGH = 4.0


# --------------------------------------------------------------------------- #
# Model registry -- imports are lazy so a missing optional dep (torch) only
# breaks the model that needs it.
# --------------------------------------------------------------------------- #
def _naive() -> BaseModel:
    from src.models.naive import NaiveModel

    return NaiveModel(seed=SEED)


def _snaive() -> BaseModel:
    from src.models.naive import SeasonalNaiveModel

    return SeasonalNaiveModel(seed=SEED)


def _sarima() -> BaseModel:
    from src.models.sarima import SarimaFourierModel

    return SarimaFourierModel(seed=SEED)


def _prophet() -> BaseModel:
    from src.models.prophet_m import ProphetModel

    return ProphetModel(seed=SEED)


def _lgbm() -> BaseModel:
    from src.models.lgbm import LgbmDirectModel

    return LgbmDirectModel(seed=SEED)


def _lstm() -> BaseModel:
    from src.models.lstm import LstmModel

    return LstmModel(seed=SEED)


MODEL_REGISTRY: dict[str, Callable[[], BaseModel]] = {
    "naive": _naive,
    "snaive": _snaive,
    "sarima": _sarima,
    "prophet": _prophet,
    "lgbm": _lgbm,
    "lstm": _lstm,
}


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fold:
    """One rolling origin: what may be seen, and what must be predicted."""

    fold: int
    target_date: date
    cutoff: pd.Timestamp
    target_index: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if len(self.target_index) != HORIZON:
            raise ValueError(
                f"fold {self.fold}: horizon is {len(self.target_index)}, want {HORIZON}"
            )
        if self.target_index.min() <= self.cutoff:
            raise ValueError(f"fold {self.fold}: target starts at or before the cutoff")


def eligible_op_dates(features: pd.DataFrame) -> set[date]:
    """Operating days that can serve as a fold target.

    A day qualifies only if it has all 24 hours, carries no `bad_day` flag, and has
    every model input present -- otherwise some models could be scored on fewer
    points than others and the comparison stops being like-for-like.
    """
    grouped = features.groupby("op_date")
    complete = grouped.size() == HORIZON
    clean = grouped["bad_day"].max() == 0
    inputs_present = (
        features[list(FEATURE_COLS)].notna().all(axis=1).groupby(features["op_date"]).all()
    )
    ok = complete & clean & inputs_present.reindex(complete.index, fill_value=False)
    return set(ok[ok].index)


def make_folds(
    features: pd.DataFrame, n_folds: int = N_FOLDS, step_days: int = FOLD_STEP_DAYS
) -> list[Fold]:
    """Build the fold list back from the last usable operating day.

    Note on `step_days=7` (the SPEC default): a step that is a multiple of 7 lands
    every fold on the same weekday. That is fine for the headline comparison -- all
    models are scored on identical points -- but it means the 52 folds cannot
    support a weekday-vs-weekend error breakdown, because only one weekday is ever
    evaluated. Use `--step 1` for the diagnostic run that needs full weekday
    coverage; `daily_folds()` is the convenience wrapper.
    """
    eligible = eligible_op_dates(features)
    if not eligible:
        raise ValueError("no operating day satisfies the fold eligibility rules")

    last = max(eligible)
    candidates = [last - timedelta(days=step_days * i) for i in range(n_folds)][::-1]
    kept = [d for d in candidates if d in eligible]
    skipped = [d for d in candidates if d not in eligible]
    if skipped:
        print(
            f"[folds] skipped {len(skipped)} ineligible target dates: "
            f"{', '.join(str(d) for d in skipped[:5])}{' ...' if len(skipped) > 5 else ''}"
        )

    if step_days % 7 == 0 and len({d.weekday() for d in kept}) == 1:
        print(
            f"[folds] every target lands on a {kept[0]:%A} (step {step_days}d). Headline "
            "metrics are unaffected, but weekday/weekend breakdowns need --step 1."
        )

    return [
        Fold(
            fold=i + 1, target_date=d, cutoff=cutoff_of_op_date(d), target_index=target_index_for(d)
        )
        for i, d in enumerate(kept)
    ]


def daily_folds(features: pd.DataFrame, days: int = 364) -> list[Fold]:
    """Every day of the last year as its own fold -- the diagnostic protocol.

    The 52-fold headline protocol steps 7 days, so it only ever evaluates one
    weekday. Grouped error analysis (by hour, weekday, month) runs on this denser
    grid instead, and the report says which protocol produced which number.
    """
    return make_folds(features, n_folds=days, step_days=1)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_fold(model: BaseModel, features: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, float]:
    """Refit `model` on everything up to the cutoff and predict the target day."""
    history = features.loc[features.index <= fold.cutoff]
    future_exog = features.loc[fold.target_index, list(FEATURE_COLS)]

    started = time.perf_counter()
    model.fit(history)
    y_pred = np.asarray(model.predict(fold.target_index, future_exog), dtype=float)
    elapsed = time.perf_counter() - started

    if y_pred.shape != (HORIZON,):
        raise ValueError(f"{model.name} returned shape {y_pred.shape}, want ({HORIZON},)")
    if not np.isfinite(y_pred).all():
        raise ValueError(f"{model.name} returned non-finite predictions in fold {fold.fold}")

    # A diverging model produces numbers that are finite and absurd -- an unconstrained
    # ARIMA once returned 1e60 MW here, which `isfinite` happily accepted. Bound the
    # output by the training range so a blow-up fails the fold instead of quietly
    # entering the comparison table.
    observed = history["y"]
    low, high = observed.min() * PLAUSIBLE_LOW, observed.max() * PLAUSIBLE_HIGH
    if (y_pred < low).any() or (y_pred > high).any():
        raise ValueError(
            f"{model.name} fold {fold.fold}: predictions outside [{low:,.0f}, {high:,.0f}] MW "
            f"(got [{y_pred.min():,.3g}, {y_pred.max():,.3g}]) -- the fit has diverged"
        )
    return y_pred, elapsed


def run_backtest(
    model_name: str, features: pd.DataFrame, folds: list[Fold], verbose: bool = True
) -> pd.DataFrame:
    """Run every fold for one model and return the tidy per-hour result frame."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_name!r}; known: {sorted(MODEL_REGISTRY)}")

    rows: list[pd.DataFrame] = []
    timings: list[float] = []
    for fold in folds:
        model = MODEL_REGISTRY[model_name]()  # refit from scratch, no state carried over
        y_pred, elapsed = run_fold(model, features, fold)
        timings.append(elapsed)

        target = features.loc[fold.target_index]
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold.fold,
                    "target_date": fold.target_date,
                    "ts": fold.target_index,
                    "lead": target["lead"].to_numpy(),
                    "hour": target["local_hour"].to_numpy(),
                    "month": target["local_month"].to_numpy(),
                    "is_weekend": target["is_weekend"].to_numpy(),
                    "y_true": target["y"].to_numpy(),
                    "y_pred": y_pred,
                }
            )
        )
        if verbose and (fold.fold % 10 == 0 or fold is folds[-1]):
            print(
                f"  [{model_name}] fold {fold.fold}/{len(folds)} "
                f"({fold.target_date})  {elapsed:.2f}s"
            )

    results = pd.concat(rows, ignore_index=True)
    results.attrs["fit_seconds_total"] = float(np.sum(timings))
    results.attrs["fit_seconds_median"] = float(np.median(timings))
    return results


def results_path(model_name: str) -> Path:
    return RESULTS_DIR / f"backtest_{model_name}.csv"


def load_results(model_name: str) -> pd.DataFrame:
    return pd.read_csv(results_path(model_name), parse_dates=["ts"])


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def build_summary(model_names: list[str]) -> pd.DataFrame:
    """Assemble results/summary.csv from whichever backtests are on disk."""
    reference = (
        load_results(metrics.SNAIVE_REF) if results_path(metrics.SNAIVE_REF).exists() else None
    )
    if reference is None:
        print(f"[warn] {metrics.SNAIVE_REF} results absent -- MASE will be NaN")

    rows = []
    for name in model_names:
        if not results_path(name).exists():
            continue
        res = load_results(name)
        row = {"model": name, **metrics.summarize(res, reference)}
        rows.append(row)

    columns = ["MAPE", "RMSE", "MASE", "MAE", "n_folds", "n_points"]
    if not rows:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="model"))
    return pd.DataFrame(rows).set_index("model")[columns]


def render_summary(summary: pd.DataFrame) -> str:
    show = summary.copy()
    show["MAPE"] = (show["MAPE"] * 100).map("{:.2f}%".format)
    show["RMSE"] = show["RMSE"].map("{:,.0f}".format)
    show["MAE"] = show["MAE"].map("{:,.0f}".format)
    show["MASE"] = show["MASE"].map("{:.3f}".format)
    return show.to_string()


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default="naive,snaive", help="comma-separated model names from the registry"
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="number of folds")
    parser.add_argument(
        "--step",
        type=int,
        default=FOLD_STEP_DAYS,
        help="days between folds; 1 gives the dense diagnostic protocol",
    )
    parser.add_argument("--suffix", default="", help="appended to output filenames")
    parser.add_argument("--features", default=str(FEATURES_PARQUET))
    args = parser.parse_args()

    ensure_dirs()
    features = pd.read_parquet(args.features)
    folds = make_folds(features, n_folds=args.folds, step_days=args.step)
    print(f"[folds] {len(folds)} folds, {folds[0].target_date} .. {folds[-1].target_date}")

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in requested:
        print(f"\n=== {name} ===")
        started = time.perf_counter()
        results = run_backtest(name, features, folds)
        path = results_path(name + args.suffix)
        results.to_csv(path, index=False)
        print(
            f"  wrote {path}  "
            f"(total {time.perf_counter() - started:.1f}s, "
            f"median {results.attrs['fit_seconds_median']:.2f}s/fold)"
        )

    if not args.suffix:
        summary = build_summary(list(MODEL_REGISTRY))
        summary.to_csv(RESULTS_DIR / "summary.csv")
        print("\n=== summary ===")
        print(render_summary(summary))


if __name__ == "__main__":
    main()
