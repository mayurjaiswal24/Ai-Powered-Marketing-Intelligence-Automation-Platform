"""Static charts for the PDF, drawn with matplotlib from the same AnalysisResult tables and the
same theme colours as the dashboard.

Why matplotlib and not Plotly images: exporting Plotly to PNG needs a headless Chrome, which is
unreliable on Streamlit Community Cloud. matplotlib draws the same numbers with no browser.
"""

from __future__ import annotations

import io
import textwrap

import matplotlib

matplotlib.use("Agg")  # no screen needed

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

from analytics.funnel import biggest_drop_off  # noqa: E402
from dashboard import theme  # noqa: E402
from dashboard.chart_standard import group_other  # noqa: E402
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


def _wrap(text, width: int = 28) -> str:
    """Wrap a long category label onto two lines (then shorten), so labels never collide."""
    lines = textwrap.wrap(str(text), width) or [""]
    if len(lines) > 2:
        lines = [lines[0], lines[1][:width - 3] + "..."]
    return "\n".join(lines)


def _title(ax, title: str, pad: float = 6, subtitle: str | None = None) -> None:
    """Insight title in bold; the subtitle (metric, unit, period) on a smaller grey line below it,
    as in the app (docs/CHART_STYLE_GUIDE.md)."""
    if subtitle:
        ax.set_title(textwrap.fill(title, 90) + "\n", pad=pad)
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, pad + 2),
                    textcoords="offset points", fontsize=7.5, color=theme.INK_MUTED, va="bottom")
    else:
        ax.set_title(textwrap.fill(title, 90), pad=pad)


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
             date_axis: bool = True, subtitle: str | None = None) -> bytes | None:
    if df is None or df.empty or y not in df or df[y].notna().sum() < 2:
        return None
    data = df[[x, y]].dropna().sort_values(x)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
    xs = data[x] if date_axis else data[x].astype(str).map(_month_label)
    ax.plot(xs, data[y], color=theme.PRIMARY, linewidth=1.6)
    ax.fill_between(range(len(xs)) if not date_axis else xs, data[y], color=theme.PRIMARY, alpha=0.10)
    ax.scatter([xs.iloc[-1]], [data[y].iloc[-1]], color=theme.PRIMARY, s=14, zorder=3)
    ax.annotate(format_value(data[y].iloc[-1], fmt, compact=True), (xs.iloc[-1], data[y].iloc[-1]),
                textcoords="offset points", xytext=(-4, 6), ha="right", color=theme.INK_SECONDARY,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    _title(ax, title, subtitle=subtitle)
    ax.set_ylim(bottom=0)
    _value_axis(ax, fmt)
    if date_axis:
        ticks = xs.iloc[:: max(1, len(xs) // 6)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_date(t).split(" ", 1)[1] if len(xs) > 20 else format_date(t)
                            for t in ticks])
    else:
        ax.tick_params(axis="x", labelrotation=0)
        step = max(1, -(-len(xs) // 8))       # at most ~8 month labels, so they never collide
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)
    return _to_png(fig)


def hbar_png(df: pd.DataFrame, category: str, value: str, fmt: str, title: str,
             ascending: bool = False, top_n: int | None = None,
             reference: float | None = None, reference_label: str = "Overall",
             subtitle: str | None = None, color_by: dict[str, str] | None = None) -> bytes | None:
    """Sorted horizontal bars. Colour as in the app: fixed entity colours with `color_by`
    (channels), otherwise the top bar in the focus colour and the rest in context grey."""
    if df is None or df.empty or value not in df or df[value].notna().sum() == 0:
        return None
    data = df[[category, value]].dropna().sort_values(value, ascending=ascending)
    if top_n:
        data = data.head(top_n)
    data = data.iloc[::-1]
    height = max(1.8, 0.32 * len(data) + 0.8)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
    labels = [_wrap(c) for c in data[category]]
    if color_by:
        bar_colors = [color_by.get(str(c), theme.NEUTRAL) for c in data[category]]
    else:
        top = str(data[category].iloc[-1])            # the first row after sorting (drawn at the top)
        bar_colors = [theme.FOCUS if str(c) == top else theme.CONTEXT for c in data[category]]
    ax.barh(labels, data[value], color=bar_colors, height=0.55)
    span = max(abs(float(data[value].max())), 1e-9)
    for i, v in enumerate(data[value]):
        ax.text(v + span * 0.01, i, format_value(v, fmt, compact=True), va="center",
                color=theme.INK_SECONDARY, fontsize=8, zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
    if reference is not None and not pd.isna(reference):
        ax.axvline(reference, color=theme.INK_MUTED, linewidth=0.8, linestyle=":")
        # Labelled at the bottom of the plot, where the shortest bar leaves room.
        ax.annotate(f"{reference_label}: {format_value(reference, fmt)}", xy=(reference, 0),
                    xycoords=("data", "axes fraction"), xytext=(3, 2), textcoords="offset points",
                    fontsize=7, color=theme.INK_MUTED, va="bottom")
    _title(ax, title, subtitle=subtitle)
    ax.set_xlim(right=span * 1.18)
    _value_axis(ax, fmt, axis="x")
    ax.tick_params(axis="y", length=0)
    return _to_png(fig)


SHARE_SERIES_COLORS = [theme.CONTEXT, theme.FOCUS, theme.ACCENT]   # same as the app


def share_png(df: pd.DataFrame, category: str, series: dict[str, str], title: str,
              subtitle: str | None = None) -> bytes | None:
    cols = {n: c for n, c in series.items() if c in df}
    if df is None or df.empty or not cols:
        return None
    data = group_other(df, category, list(cols.values()), sort_by=list(cols.values())[0])
    data = data.sort_values(list(cols.values())[0])
    height = max(2.0, 0.42 * len(data) + 0.9)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
    n = len(cols)
    bar_h = 0.7 / n
    positions = range(len(data))
    for i, (name, col) in enumerate(cols.items()):
        # First series on top within each group, matching the legend order.
        ys = [p + ((n - 1) / 2 - i) * bar_h for p in positions]
        ax.barh(ys, data[col], height=bar_h * 0.9, color=SHARE_SERIES_COLORS[i], label=name)
        for y, v in zip(ys, data[col]):
            if pd.notna(v):
                ax.text(v, y, " " + format_value(v, "percent"), va="center", fontsize=6.5,
                        color=theme.INK_SECONDARY)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([_wrap(c) for c in data[category]])
    # Legend in its own row above the plot and below the title (never over the bars or title).
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, frameon=False, fontsize=8,
              borderaxespad=0.2, handlelength=1.2)
    _title(ax, title, pad=22, subtitle=subtitle)
    ax.set_xlim(right=float(data[list(cols.values())].max().max()) * 1.15)
    _value_axis(ax, "percent", axis="x")
    ax.tick_params(axis="y", length=0)
    return _to_png(fig)


def funnel_png(funnel: pd.DataFrame, title: str, subtitle: str | None = None) -> bytes | None:
    """Same design as the app: one bar per step = share of people who move on (0-100%), labelled
    with the count reached; the biggest drop after the click stage in red, the rest grey."""
    if funnel is None or funnel.empty:
        return None
    data = funnel[funnel["stage"] != "revenue"].sort_values("stage_order").reset_index(drop=True)
    if len(data) < 2:
        return None
    worst = biggest_drop_off(data)
    steps = data.iloc[1:]
    rows = [f"{p} → {c}" for p, c in zip(data["label"].iloc[:-1], steps["label"])][::-1]
    rates = list(steps["rate_from_previous"])[::-1]
    values = list(steps["value"])[::-1]
    names = [str(c).lower() for c in steps["label"]][::-1]
    colors = [theme.BAD if worst is not None and st == worst["stage"] else theme.CONTEXT
              for st in steps["stage"]][::-1]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.42 * len(steps) + 1.0))
    ax.barh([_wrap(r, 34) for r in rows], rates, color=colors, height=0.55)
    for i, (r, v, name) in enumerate(zip(rates, values, names)):
        ax.text((0 if pd.isna(r) else r) + 1, i,
                f"{format_value(r, 'percent')} move on · {format_value(v, 'count')} {name}",
                va="center", color=theme.INK_SECONDARY, fontsize=8)
    ax.set_xlim(0, 160)                       # room for the labels to the right of the 100% mark
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=theme.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    first = data.iloc[0]
    note = f"Starting from {format_value(first['value'], 'count')} {str(first['label']).lower()}"
    _title(ax, title, subtitle=f"{subtitle} · {note}" if subtitle else note)
    return _to_png(fig)
