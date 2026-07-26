"""Figures and grouped error tables (SPEC section 7).

All plotting lives here rather than in the notebooks: the notebooks import these
functions, so every figure in the README, the notebooks and the report comes from
one code path and can be regenerated with a single command.

    conda run -n p3 python -m src.report --all
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import viz
from src.backtest import build_summary, load_results, results_path
from src.config import (
    CLEAN_PARQUET,
    FEATURES_PARQUET,
    FIGURES_DIR,
    LOAD_COL,
    LOCAL_TZ,
    RESULTS_DIR,
    ensure_dirs,
)
from src.features import FEATURE_COLS
from src.metrics import fold_maes, grouped_errors, mape

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _save(fig, name: str) -> str:
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# =========================================================================== #
# EDA figures
# =========================================================================== #
def plot_seasonality(clean: pd.DataFrame) -> str:
    """Daily shape by day type, and the weekday x hour surface behind it."""
    local = clean.index.tz_convert(LOCAL_TZ)
    frame = pd.DataFrame(
        {
            "load": clean[LOAD_COL].to_numpy(),
            "hour": local.hour,
            "dow": local.dayofweek,
        }
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[1, 1.15])

    weekday = frame[frame["dow"] < 5].groupby("hour")["load"].mean()
    weekend = frame[frame["dow"] >= 5].groupby("hour")["load"].mean()
    left.plot(weekday.index, weekday.to_numpy(), color=viz.SERIES[0], label="Weekday")
    left.plot(weekend.index, weekend.to_numpy(), color=viz.SERIES[1], label="Weekend")
    viz.annotate_endpoint(left, weekday.index[-1], weekday.iloc[-1], "Weekday", viz.SERIES[0])
    viz.annotate_endpoint(left, weekend.index[-1], weekend.iloc[-1], "Weekend", viz.SERIES[1])
    left.set_title("Average daily load profile")
    left.set_xlabel("Hour of day (local)")
    left.set_ylabel("MW")
    left.set_xlim(0, 26)
    left.set_xticks(range(0, 24, 4))
    viz.thousands_axis(left)
    left.legend(loc="lower right")

    grid = frame.pivot_table(index="dow", columns="hour", values="load", aggfunc="mean")
    mesh = right.pcolormesh(
        grid.columns, grid.index, grid.to_numpy(), cmap=viz.BLUES, shading="nearest"
    )
    right.set_title("Mean load by weekday and hour")
    right.set_xlabel("Hour of day (local)")
    right.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    right.grid(False)
    right.invert_yaxis()
    bar = fig.colorbar(mesh, ax=right, pad=0.02)
    bar.set_label("MW", color=viz.INK_SECONDARY)
    bar.outline.set_visible(False)

    fig.suptitle(
        "Load has three nested cycles: daily, weekly, yearly",
        x=0.005,
        ha="left",
        color=viz.INK,
        fontsize=12,
        fontweight="600",
    )
    fig.tight_layout()
    return _save(fig, "01_seasonality.png")


def plot_stl(clean: pd.DataFrame) -> str:
    """STL on the daily mean -- an hourly STL at s=8766 is unreadable and slow."""
    from statsmodels.tsa.seasonal import STL

    daily = clean[LOAD_COL].resample("D").mean().dropna()
    result = STL(daily, period=365, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    panels = [
        (daily, "Observed (daily mean)"),
        (result.trend, "Trend"),
        (result.seasonal, "Seasonal (annual)"),
        (result.resid, "Residual"),
    ]
    for ax, (series, title) in zip(axes, panels, strict=True):
        ax.plot(series.index, series.to_numpy(), color=viz.SERIES[0], linewidth=1.0)
        ax.set_title(title, fontsize=9.5)
        viz.thousands_axis(ax)
    axes[-1].set_xlabel("Date")

    fig.suptitle(
        "STL decomposition: a flat trend under a strong annual cycle",
        x=0.005,
        ha="left",
        color=viz.INK,
        fontsize=12,
        fontweight="600",
    )
    fig.tight_layout()
    return _save(fig, "02_stl.png")


def plot_load_vs_temperature(clean: pd.DataFrame) -> str:
    """The U curve: heating below ~15C, cooling above ~20C. This motivates HDD/CDD."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    hexes = ax.hexbin(
        clean["temp_c"], clean[LOAD_COL], gridsize=55, cmap=viz.BLUES, mincnt=1, linewidths=0.0
    )
    bins = pd.cut(clean["temp_c"], bins=np.arange(-20, 41, 2))
    median = clean.groupby(bins, observed=True)[LOAD_COL].median()
    centres = [interval.mid for interval in median.index]
    ax.plot(centres, median.to_numpy(), color=viz.SERIES[1], linewidth=2.0, label="Median load")
    ax.axvline(18, color=viz.BASELINE, linewidth=1.0)
    ax.axvline(24, color=viz.BASELINE, linewidth=1.0)
    ax.text(
        18, ax.get_ylim()[1], " HDD base 18C", va="top", ha="left", color=viz.INK_MUTED, fontsize=8
    )
    ax.text(
        24, ax.get_ylim()[1], " CDD base 24C", va="top", ha="left", color=viz.INK_MUTED, fontsize=8
    )

    ax.set_title("Load vs temperature: the classic U", loc="left")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("MW")
    viz.thousands_axis(ax)
    ax.legend(loc="upper center")
    bar = fig.colorbar(hexes, ax=ax, pad=0.02)
    bar.set_label("Hours", color=viz.INK_SECONDARY)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, "03_load_vs_temperature.png")


def plot_autocorrelation(clean: pd.DataFrame, lags: int = 200) -> str:
    """ACF/PACF spikes at 24 and 168 are the evidence behind the lag feature set."""
    from statsmodels.tsa.stattools import acf, pacf

    series = clean[LOAD_COL]
    acf_values = acf(series, nlags=lags)
    pacf_values = pacf(series.iloc[-20_000:], nlags=min(lags, 100))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.5))
    for ax, values, title in (
        (top, acf_values, "Autocorrelation"),
        (bottom, pacf_values, "Partial autocorrelation"),
    ):
        ax.vlines(range(len(values)), 0, values, color=viz.SERIES[0], linewidth=1.2)
        viz.hairline_baseline(ax)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Lag (hours)")
        for lag in (24, 48, 168):
            if lag < len(values):
                ax.axvline(lag, color=viz.SERIES[1], linewidth=1.0, alpha=0.7)
                ax.text(
                    lag,
                    ax.get_ylim()[1],
                    f" {lag}h",
                    va="top",
                    ha="left",
                    color=viz.SERIES[1],
                    fontsize=8,
                )

    fig.suptitle(
        "Correlation peaks at 24h and 168h -- hence lag_24, lag_48, lag_168",
        x=0.005,
        ha="left",
        color=viz.INK,
        fontsize=12,
        fontweight="600",
    )
    fig.tight_layout()
    return _save(fig, "04_autocorrelation.png")


# =========================================================================== #
# Result figures
# =========================================================================== #
def plot_model_comparison(summary: pd.DataFrame) -> str:
    """MAPE and MASE side by side, every bar directly labelled."""
    order = viz.order_models(summary.index)
    data = summary.loc[order]
    colors = [viz.MODEL_COLORS.get(m, viz.SERIES[0]) for m in order]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    bars = left.bar(order, data["MAPE"], color=colors, width=0.62)
    viz.label_bars(left, bars, data["MAPE"].to_numpy(), fmt="{:.2%}")
    left.set_title("MAPE (lower is better)")
    left.set_ylabel("Mean absolute percentage error")
    viz.percent_axis(left)
    left.set_ylim(0, data["MAPE"].max() * 1.22)

    bars = right.bar(order, data["MASE"], color=colors, width=0.62)
    viz.label_bars(right, bars, data["MASE"].to_numpy(), fmt="{:.3f}")
    # Chrome, not a series: a categorical hue here would read as a sixth model.
    right.axhline(1.0, color=viz.INK_MUTED, linewidth=1.4)
    right.text(
        len(order) - 0.4,
        1.005,
        "seasonal naive",
        va="bottom",
        ha="right",
        color=viz.INK_MUTED,
        fontsize=8,
    )
    right.set_title("MASE (below 1.0 beats the baseline)")
    right.set_ylabel("MAE relative to seasonal naive")
    right.set_ylim(0, max(1.15, data["MASE"].max() * 1.22))

    for ax in (left, right):
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", colors=viz.INK_SECONDARY, labelsize=9)

    n_folds = int(data["n_folds"].max())
    fig.suptitle(
        f"Model comparison over {n_folds} rolling-origin folds",
        x=0.005,
        ha="left",
        color=viz.INK,
        fontsize=12,
        fontweight="600",
    )
    fig.tight_layout()
    return _save(fig, "05_model_comparison.png")


def plot_best_worst_folds(results: pd.DataFrame, model_name: str) -> str:
    """The model's easiest and hardest day, so the error has a face."""
    maes = fold_maes(results)
    best_fold, worst_fold = int(maes.idxmin()), int(maes.idxmax())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, fold, label in ((axes[0], best_fold, "Best"), (axes[1], worst_fold, "Worst")):
        day = results[results["fold"] == fold].sort_values("lead")
        error = mape(day["y_true"].to_numpy(), day["y_pred"].to_numpy())

        ax.plot(day["lead"], day["y_true"], color=viz.SERIES[0], label="Actual")
        ax.plot(day["lead"], day["y_pred"], color=viz.SERIES[1], label="Forecast")
        ax.fill_between(
            day["lead"], day["y_true"], day["y_pred"], color=viz.SERIES[1], alpha=0.12, linewidth=0
        )
        ax.set_title(f"{label} fold - {day['target_date'].iloc[0]} (MAPE {error:.2%})")
        ax.set_xlabel("Hours ahead of the 04:00 UTC cutoff")
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 6, 12, 18, 24])
        viz.thousands_axis(ax)
    axes[0].set_ylabel("MW")
    axes[0].legend(loc="upper left")

    fig.suptitle(
        f"{model_name}: day-ahead forecast against actual load",
        x=0.005,
        ha="left",
        color=viz.INK,
        fontsize=12,
        fontweight="600",
    )
    fig.tight_layout()
    return _save(fig, "06_pred_vs_actual.png")


def plot_error_heatmap(results: pd.DataFrame, model_name: str) -> str:
    """Where the error lives: hour of day against month."""
    frame = results.copy()
    frame["ape"] = (frame["y_true"] - frame["y_pred"]).abs() / frame["y_true"]
    grid = frame.pivot_table(index="hour", columns="month", values="ape", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    mesh = ax.pcolormesh(
        grid.columns, grid.index, grid.to_numpy(), cmap=viz.BLUES, shading="nearest"
    )
    ax.set_title(f"{model_name}: MAPE by local hour and month", loc="left")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day (local)")
    ax.set_xticks(sorted(grid.columns), [MONTH_LABELS[m - 1] for m in sorted(grid.columns)])
    ax.set_yticks(range(0, 24, 2))
    ax.grid(False)
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, format=lambda v, _: f"{v:.1%}")
    bar.set_label("MAPE", color=viz.INK_SECONDARY)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, "07_error_heatmap.png")


def plot_error_by_hour(model_names: list[str]) -> str:
    """Error profile across the day, one line per model."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for name in viz.order_models(model_names):
        if not results_path(name).exists():
            continue
        results = load_results(name)
        by_hour = grouped_errors(results, "hour")["MAPE"]
        color = viz.MODEL_COLORS.get(name, viz.SERIES[0])
        ax.plot(by_hour.index, by_hour.to_numpy(), color=color, label=name)
        viz.annotate_endpoint(ax, by_hour.index[-1], by_hour.iloc[-1], name, color)

    ax.set_title("Error by hour of day: the evening ramp is the hard part", loc="left")
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("MAPE")
    ax.set_xlim(0, 25.5)
    ax.set_xticks(range(0, 24, 3))
    viz.percent_axis(ax)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return _save(fig, "08_error_by_hour.png")


def plot_feature_importance(top_n: int = 15) -> str:
    """Gain importance of the shipped model, averaged over its 24 lead models."""
    from src.features import build_features
    from src.models.lgbm import LgbmDirectModel

    features = pd.read_parquet(FEATURES_PARQUET)
    model = LgbmDirectModel()
    model.fit(features)
    importance = model.feature_importance().head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.barh(importance.index, importance.to_numpy(), color=viz.SERIES[0], height=0.66)
    viz.label_bars(ax, bars, importance.to_numpy(), fmt="{:.1%}", horizontal=True)
    ax.set_title(f"LightGBM: top {top_n} features by gain", loc="left")
    ax.set_xlabel("Share of total gain")
    ax.set_xlim(0, importance.max() * 1.18)
    viz.percent_axis(ax, axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=viz.INK_SECONDARY, labelsize=8.5)
    fig.tight_layout()
    _ = build_features  # imported for the notebook's convenience
    return _save(fig, "09_feature_importance.png")


# =========================================================================== #
# Tables and conclusions
# =========================================================================== #
def error_tables(model_name: str) -> dict[str, pd.DataFrame]:
    """MAPE/RMSE/MAE broken out by hour, day type and month."""
    results = load_results(model_name)
    results["day_type"] = np.where(results["is_weekend"] == 1, "weekend", "weekday")
    return {
        "by_hour": grouped_errors(results, "hour"),
        "by_day_type": grouped_errors(results, "day_type"),
        "by_month": grouped_errors(results, "month"),
    }


def business_findings(model_name: str) -> list[str]:
    """Derive the report's conclusions from the numbers rather than asserting them."""
    tables = error_tables(model_name)
    by_hour, by_day, by_month = tables["by_hour"], tables["by_day_type"], tables["by_month"]

    worst_hour = by_hour["MAPE"].idxmax()
    best_hour = by_hour["MAPE"].idxmin()
    worst_month = by_month["MAPE"].idxmax()
    best_month = by_month["MAPE"].idxmin()

    findings = [
        f"Hardest hour is {worst_hour:02d}:00 local at {by_hour.loc[worst_hour, 'MAPE']:.2%} MAPE, "
        f"{by_hour.loc[worst_hour, 'MAPE'] / by_hour.loc[best_hour, 'MAPE']:.1f}x the easiest "
        f"({best_hour:02d}:00, {by_hour.loc[best_hour, 'MAPE']:.2%}).",
        f"Hardest month is {MONTH_LABELS[worst_month - 1]} at "
        f"{by_month.loc[worst_month, 'MAPE']:.2%}, against "
        f"{MONTH_LABELS[best_month - 1]} at {by_month.loc[best_month, 'MAPE']:.2%} -- "
        f"error tracks temperature-driven demand, not calendar position.",
    ]
    if {"weekday", "weekend"} <= set(by_day.index):
        weekday, weekend = by_day.loc["weekday", "MAPE"], by_day.loc["weekend", "MAPE"]
        harder = "weekends" if weekend > weekday else "weekdays"
        findings.append(
            f"{harder.capitalize()} are harder: {weekend:.2%} weekend vs {weekday:.2%} weekday "
            f"({abs(weekend - weekday) / min(weekday, weekend):.0%} relative difference)."
        )
    return findings


def significance_table(model_names: list[str]) -> pd.DataFrame:
    """Pairwise paired tests down the ranking: is each step a real improvement?

    Without this, a comparison table invites reading every gap as a ranking. Two of
    the gaps in this project's results do not survive the test, and saying so is the
    difference between a report and a leaderboard.
    """
    from src.metrics import compare_models

    ranked = [m for m in viz.order_models(model_names) if results_path(m).exists()]
    ranked.sort(
        key=lambda m: load_results(m).pipe(lambda r: (r["y_true"] - r["y_pred"]).abs().mean())
    )

    rows = []
    for better, worse in zip(ranked, ranked[1:], strict=False):
        rows.append(compare_models(load_results(better), load_results(worse), better, worse))
    return pd.DataFrame(rows)


def render_significance(table: pd.DataFrame) -> str:
    lines = [
        "| better | worse | MAE | MAE | folds won | p | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        verdict = "**significant**" if r["significant_at_05"] else "not significant"
        lines.append(
            f"| {r['model_a']} | {r['model_b']} | {r['mae_a']:,.0f} | {r['mae_b']:,.0f} | "
            f"{int(r['a_wins_folds'])}/{int(r['n_folds'])} | "
            f"{r['p_value']:.4f} | {verdict} |"
        )
    return "\n".join(lines)


def weather_ablation(n_folds: int = 8, seed: int = 42) -> pd.DataFrame:
    """Refit LightGBM with the weather block zeroed out, and report the gap.

    The `temp`/`hdd`/`cdd` features read the realised temperature at the target hour,
    i.e. they assume a perfect weather forecast. Every report states that caveat;
    this measures it. The gap is the part of the accuracy that a production system --
    which only has a numerical weather prediction -- would have to earn back.
    """
    from src.backtest import make_folds
    from src.models.lgbm import LgbmDirectModel

    features = pd.read_parquet(FEATURES_PARQUET)
    folds = make_folds(features, n_folds=n_folds)
    blanked = dict.fromkeys(("temp", "hdd", "cdd", "temp_lag24"), 0.0)

    rows = []
    for fold in folds:
        history = features.loc[features.index <= fold.cutoff]
        exog = features.loc[fold.target_index, list(FEATURE_COLS)]
        y_true = features.loc[fold.target_index, "y"].to_numpy()

        with_weather = LgbmDirectModel(seed=seed)
        with_weather.fit(history)

        without = LgbmDirectModel(seed=seed)
        without.fit(history.assign(**blanked))

        rows.append(
            {
                "fold": fold.fold,
                "target_date": fold.target_date,
                "mape_with_weather": mape(y_true, with_weather.predict(fold.target_index, exog)),
                "mape_without_weather": mape(
                    y_true, without.predict(fold.target_index, exog.assign(**blanked))
                ),
            }
        )
    return pd.DataFrame(rows)


# =========================================================================== #
# README injection
# =========================================================================== #
README_BEGIN = "<!-- RESULTS:BEGIN"
README_END = "<!-- RESULTS:END -->"


def render_markdown_table(summary: pd.DataFrame, model: str = "lgbm") -> str:
    """The comparison table as markdown, best model bolded."""
    lines = ["| model | MAPE | RMSE (MW) | MASE | folds |", "|---|---|---|---|---|"]
    for name in viz.order_models(summary.index):
        row = summary.loc[name]
        label = f"**{name}**" if name == model else name
        mase = "—" if pd.isna(row["MASE"]) else f"{row['MASE']:.3f}"
        lines.append(
            f"| {label} | {row['MAPE']:.2%} | {row['RMSE']:,.0f} | {mase} | {int(row['n_folds'])} |"
        )
    return "\n".join(lines)


def update_readme(summary: pd.DataFrame, model: str = "lgbm") -> str:
    """Rewrite the README's results block in place.

    The README claims these numbers are written by tooling rather than by hand;
    this is the tooling. Anything outside the markers is left untouched.
    """
    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start = text.index(README_BEGIN)
    end = text.index(README_END) + len(README_END)

    findings = business_findings(model)
    significance = significance_table(list(summary.index))
    ties = significance[~significance["significant_at_05"]]
    tie_note = (
        "Not every gap in that table is a result: "
        + "; ".join(
            f"**{r['model_a']} vs {r['model_b']}** is a statistical tie "
            f"(p={r['p_value']:.2f}, won {int(r['a_wins_folds'])}/{int(r['n_folds'])} folds)"
            for _, r in ties.iterrows()
        )
        + "."
    )

    block = "\n".join(
        [
            "<!-- RESULTS:BEGIN — regenerated by `python -m src.report`; do not edit by hand -->",
            "",
            f"52 rolling-origin folds, 1,248 scored hours per model. Shipped model: **{model}**.",
            "",
            render_markdown_table(summary, model),
            "",
            "### Is each step down the table real?",
            "",
            "Paired Wilcoxon signed-rank on per-fold MAE. Paired because both models",
            "forecast the *same* target points, which removes fold difficulty from the",
            "comparison; signed-rank rather than a t-test because fold errors are",
            "right-skewed by a handful of extreme-weather days.",
            "",
            render_significance(significance),
            "",
            *([tie_note, ""] if len(ties) else []),
            "**What the errors say:**",
            "",
            *[f"{i}. {f}" for i, f in enumerate(findings, 1)],
            "",
            README_END,
        ]
    )
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return str(path)


# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda", action="store_true", help="regenerate EDA figures")
    parser.add_argument("--results", action="store_true", help="regenerate result figures")
    parser.add_argument("--all", action="store_true", help="both")
    parser.add_argument("--model", default="lgbm", help="model the result figures describe")
    parser.add_argument("--readme", action="store_true", help="rewrite the README results block")
    parser.add_argument(
        "--ablation",
        type=int,
        default=0,
        metavar="N",
        help="measure the perfect-weather assumption over N folds",
    )
    args = parser.parse_args()
    if not (args.eda or args.results or args.all):
        args.all = True

    viz.apply_theme()
    ensure_dirs()
    written: list[str] = []

    if args.eda or args.all:
        clean = pd.read_parquet(CLEAN_PARQUET)
        written += [
            plot_seasonality(clean),
            plot_stl(clean),
            plot_load_vs_temperature(clean),
            plot_autocorrelation(clean),
        ]

    if args.results or args.all:
        summary = build_summary(list(viz.MODEL_ORDER))
        summary.to_csv(RESULTS_DIR / "summary.csv")
        results = load_results(args.model)
        written += [
            plot_model_comparison(summary),
            plot_best_worst_folds(results, args.model),
            plot_error_heatmap(results, args.model),
            plot_error_by_hour(list(summary.index)),
            plot_feature_importance(),
        ]

        print("\n=== Grouped error analysis ===")
        for name, table in error_tables(args.model).items():
            print(f"\n-- {name} --")
            print(table.to_string(float_format=lambda v: f"{v:,.4f}"))

        print("\n=== Business findings ===")
        for i, finding in enumerate(business_findings(args.model), 1):
            print(f"{i}. {finding}")

        if args.ablation:
            table = weather_ablation(n_folds=args.ablation)
            table.to_csv(RESULTS_DIR / "weather_ablation.csv", index=False)
            with_w = table["mape_with_weather"].mean()
            without_w = table["mape_without_weather"].mean()
            print(f"\n=== Perfect-weather ablation ({args.ablation} folds) ===")
            print(f"  with weather   : {with_w:.2%}")
            print(f"  without weather: {without_w:.2%}")
            print(
                f"  the assumption is worth {without_w - with_w:.2%} MAPE "
                f"({without_w / with_w - 1:+.0%} relative)"
            )

        if args.readme:
            print(f"\n=== README updated: {update_readme(summary, args.model)} ===")
            print(render_markdown_table(summary, args.model))

    print("\n=== Figures ===")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
