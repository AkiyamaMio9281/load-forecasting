"""Chart theme: one palette, one set of chrome rules, used by every figure.

Colour choices are not per-chart taste. The categorical slots below are assigned in
fixed order (so a model keeps its hue across every figure), and the sequential ramp
is a single hue light-to-dark (so magnitude reads as magnitude). Three of the
categorical slots sit below 3:1 against the surface, so charts that use them carry
visible value labels rather than relying on the swatch alone.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in assignment order. Never cycled, never reordered per chart.
SERIES = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

# Chrome and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Single-hue sequential ramp (light -> dark) for magnitude encodings.
SEQUENTIAL_STEPS = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
BLUES = LinearSegmentedColormap.from_list("p3_blues", SEQUENTIAL_STEPS)

# Fixed model -> slot map so a model is the same colour in every figure.
MODEL_COLORS = {
    "naive": SERIES[4],
    "snaive": SERIES[3],
    "sarima": SERIES[2],
    "prophet": SERIES[1],
    "lgbm": SERIES[0],
    "lstm": SERIES[6],
}
MODEL_ORDER = ("naive", "snaive", "sarima", "prophet", "lstm", "lgbm")


def apply_theme() -> None:
    """Recessive chrome: hairline solid grid, no box, muted ticks, generous padding."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.dpi": 140,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 140,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linestyle": "-",  # never dashed -- dashes read as "threshold"
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def order_models(names) -> list[str]:
    """Sort model names into the canonical weak-to-strong order."""
    known = [m for m in MODEL_ORDER if m in set(names)]
    return known + sorted(set(names) - set(known))


def label_bars(ax, bars, values, fmt="{:.2f}", pad=0.01, horizontal=False) -> None:
    """Direct value labels -- the relief for low-contrast slots, not decoration."""
    span = max(values) if len(values) else 1.0
    for bar, value in zip(bars, values, strict=True):
        if horizontal:
            ax.text(
                bar.get_width() + span * pad,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
                ha="left",
                color=INK_SECONDARY,
                fontsize=8,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + span * pad,
                fmt.format(value),
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=8,
            )


def annotate_endpoint(ax, x, y, text, color) -> None:
    """Selective direct label at a line's endpoint instead of a value per point."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        color=color,
        fontsize=8.5,
        fontweight="600",
    )


def hairline_baseline(ax, y: float = 0.0) -> None:
    ax.axhline(y, color=BASELINE, linewidth=1.0, zorder=1)


def percent_axis(ax, axis: str = "y", decimals: int = 1) -> None:
    """Percent ticks. `decimals` matters: at 0 a 1.6%..3.4% range prints '2%' twice."""
    formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def thousands_axis(ax, axis: str = "y") -> None:
    formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def diverging_norm(values: np.ndarray):
    """Symmetric normalisation so a diverging ramp's midpoint really means zero."""
    limit = float(np.nanmax(np.abs(values)))
    return mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
