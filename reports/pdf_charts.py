"""Static charts for the PDF, drawn with matplotlib from the same AnalysisResult tables and the
same theme colours as the dashboard.

Why matplotlib and not Plotly images: exporting Plotly to PNG needs a headless Chrome, which is
unreliable on Streamlit Community Cloud. matplotlib draws the same numbers with no browser.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # no screen needed

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

from dashboard import theme  # noqa: E402
from utils.formatting import format_date, format_value  # noqa: E402

WIDTH_IN = 6.7        # fits the A4 text width
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": theme.AXIS,
    "axes.labelcolor": theme.INK_MUTED,
    "axes.titlecolor": theme.INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": theme.INK_MUTED,
    "ytick.color": theme.INK_MUTED,
    "axes.facecolor": theme.SURFACE,
    "figure.facecolor": "white",
})


def _to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _value_axis(ax, fmt: str, axis: str = "y") -> None:
    formatter = FuncFormatter(lambda v, _pos: format_value(v, fmt, compact=True))
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(formatter)
    target.set_major_locator(MaxNLocator(5))
    ax.grid(axis=axis, color=theme.GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _month_label(text: str) -> str:
    """'2026-01' -> 'Jan 26'; anything else unchanged."""
    try:
        return pd.Timestamp(f"{text}-01").strftime("%b %y") if len(text) == 7 else text
    except ValueError:
        return text


def line_png(df: pd.DataFrame, x: str, y: str, fmt: str, title: str, height: float = 2.3,
             date_axis: bool = True) -> bytes | None:
    if df is None or df.empty or y not in df or df[y].notna().sum() < 2:
        return None
    data = df[[x, y]].dropna().sort_values(x)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
    xs = data[x] if date_axis else data[x].astype(str).map(_month_label)
    ax.plot(xs, data[y], color=theme.PRIMARY, linewidth=1.6)
    ax.fill_between(range(len(xs)) if not date_axis else xs, data[y], color=theme.PRIMARY, alpha=0.10)
    ax.scatter([xs.iloc[-1]], [data[y].iloc[-1]], color=theme.PRIMARY, s=14, zorder=3)
    ax.annotate(format_value(data[y].iloc[-1], fmt, compact=True), (xs.iloc[-1], data[y].iloc[-1]),
                textcoords="offset points", xytext=(-4, 6), ha="right", color=theme.INK_SECONDARY)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    _value_axis(ax, fmt)
    if date_axis:
        ticks = xs.iloc[:: max(1, len(xs) // 6)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_date(t).split(" ", 1)[1] if len(xs) > 20 else format_date(t)
                            for t in ticks])
    else:
        ax.tick_params(axis="x", labelrotation=0)
        step = max(1, len(xs) // 12)
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)
    return _to_png(fig)


def hbar_png(df: pd.DataFrame, category: str, value: str, fmt: str, title: str,
             ascending: bool = False, top_n: int | None = None,
             reference: float | None = None, reference_label: str = "Overall") -> bytes | None:
    if df is None or df.empty or value not in df or df[value].notna().sum() == 0:
        return None
    data = df[[category, value]].dropna().sort_values(value, ascending=ascending)
    if top_n:
        data = data.head(top_n)
    data = data.iloc[::-1]
    height = max(1.8, 0.32 * len(data) + 0.8)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
    labels = [str(c) if len(str(c)) <= 42 else str(c)[:40] + "..." for c in data[category]]
    ax.barh(labels, data[value], color=theme.PRIMARY, height=0.55)
    span = max(abs(float(data[value].max())), 1e-9)
    for i, v in enumerate(data[value]):
        ax.text(v + span * 0.01, i, format_value(v, fmt, compact=True), va="center",
                color=theme.INK_SECONDARY, fontsize=8)
    if reference is not None and not pd.isna(reference):
        ax.axvline(reference, color=theme.INK_MUTED, linewidth=0.8)   # explained in the caption
    ax.set_title(title)
    ax.set_xlim(right=span * 1.18)
    _value_axis(ax, fmt, axis="x")
    ax.tick_params(axis="y", length=0)
    return _to_png(fig)


def share_png(df: pd.DataFrame, category: str, series: dict[str, str], title: str) -> bytes | None:
    cols = {n: c for n, c in series.items() if c in df}
    if df is None or df.empty or not cols:
        return None
    data = df.sort_values(list(cols.values())[0])
    height = max(2.0, 0.42 * len(data) + 0.9)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
    n = len(cols)
    bar_h = 0.7 / n
    positions = range(len(data))
    for i, (name, col) in enumerate(cols.items()):
        # First series on top within each group, matching the legend order.
        ax.barh([p + ((n - 1) / 2 - i) * bar_h for p in positions], data[col], height=bar_h * 0.9,
                color=theme.CATEGORICAL[i], label=name)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(data[category].astype(str))
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _value_axis(ax, "percent", axis="x")
    ax.tick_params(axis="y", length=0)
    return _to_png(fig)


def funnel_png(funnel: pd.DataFrame, title: str) -> bytes | None:
    if funnel is None or funnel.empty:
        return None
    data = funnel[funnel["stage"] != "revenue"].sort_values("stage_order")
    if len(data) < 2:
        return None
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.42 * len(data) + 0.9))
    labels = list(data["label"])[::-1]
    values = list(data["value"])[::-1]
    rates = list(data["rate_from_previous"])[::-1]
    colors = theme.ORDINAL_BLUES[:len(data)][::-1]
    # Log scale: impressions are ~100,000x conversions; a linear funnel would hide the lower stages.
    ax.barh(labels, values, color=colors, height=0.6)
    ax.set_xscale("log")
    for i, (v, r) in enumerate(zip(values, rates)):
        text = format_value(v, "count") + ("" if pd.isna(r) else f"  ({format_value(r, 'percent')} of previous)")
        ax.text(v * 1.15, i, text, va="center", color=theme.INK_SECONDARY, fontsize=8)
    ax.set_xlim(right=max(values) * 60)
    ax.set_xticks([])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    return _to_png(fig)
