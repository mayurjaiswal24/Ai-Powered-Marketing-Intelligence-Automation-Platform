"""Chart builders. Each takes a result table and returns a Plotly figure, or None when the data
needed is missing (the page then shows an empty state instead of a blank or fake chart).

Rules applied everywhere: one theme template, thin marks, one y-axis per chart, numbers in
tooltips and axes formatted with utils/formatting.py (Indian grouping, ₹, %, x), no 3D.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard import theme
from utils.formatting import format_date, format_value

CHART_HEIGHT = 320


def _has(df: pd.DataFrame | None, *cols: str) -> bool:
    return df is not None and not df.empty and all(c in df and df[c].notna().any() for c in cols)


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    vmin, vmax = min(vmin, 0.0), max(vmax, 0.0)
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _format_axis(fig: go.Figure, values, fmt: str, axis: str = "y") -> None:
    """Indian-formatted tick labels (₹4.2 Cr, 12,34,567, 12.5%, 3.00x) for a value axis."""
    vals = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return
    ticks = _nice_ticks(float(vals.min()), float(vals.max()))
    text = [format_value(t, fmt, compact=True) for t in ticks]
    update = dict(tickmode="array", tickvals=ticks, ticktext=text)
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**update)


def wrap_label(text, width: int = theme.LABEL_WRAP, sep: str = "<br>") -> str:
    """Break a long label at word boundaries so it never runs into its neighbour."""
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return sep.join(lines)


def _base(title: str, height: int = CHART_HEIGHT) -> go.Figure:
    fig = go.Figure()
    title = wrap_label(title, theme.TITLE_WRAP)
    extra = 20 * title.count("<br>")          # a wrapped title needs one more line of room
    fig.update_layout(template=theme.TEMPLATE_NAME, title=title, height=height + extra,
                      showlegend=False, margin=dict(t=theme.TOP_MARGIN + extra))
    return fig


def _show_legend(fig: go.Figure) -> None:
    """Legend row under the title, with the top margin grown so they never overlap."""
    extra = fig.layout.margin.t - theme.TOP_MARGIN
    fig.update_layout(showlegend=True, margin=dict(t=theme.TOP_MARGIN_WITH_LEGEND + extra),
                      height=fig.layout.height + theme.TOP_MARGIN_WITH_LEGEND - theme.TOP_MARGIN)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------------------------
# Line / trend
# ---------------------------------------------------------------------------------------------

def line_chart(df: pd.DataFrame, x: str, y: str, fmt: str, title: str,
               x_label_fmt: str = "date") -> go.Figure | None:
    """Single-series trend with a light area wash; the latest value is labelled at the end."""
    if not _has(df, x, y):
        return None
    data = df[[x, y]].dropna(subset=[y]).sort_values(x)
    if data.empty:
        return None
    x_text = (data[x].map(format_date) if x_label_fmt == "date" else data[x].astype(str))
    hover = [f"{xt}<br><b>{format_value(v, fmt)}</b>" for xt, v in zip(x_text, data[y])]
    fig = _base(title)
    fig.add_trace(go.Scatter(
        x=data[x], y=data[y], mode="lines", line=dict(color=theme.PRIMARY, width=2),
        fill="tozeroy", fillcolor=_hex_to_rgba(theme.PRIMARY, 0.10),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    last = data.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last[x]], y=[last[y]], mode="markers+text", text=[format_value(last[y], fmt, compact=True)],
        textposition="top left", textfont=dict(color=theme.INK_SECONDARY, size=12),
        marker=dict(size=8, color=theme.PRIMARY, line=dict(color=theme.SURFACE, width=2)),
        hoverinfo="skip"))
    _format_axis(fig, data[y], fmt)
    fig.update_layout(hovermode="x unified" if len(data) > 1 else "closest")
    return fig


# ---------------------------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------------------------

def bar_chart(df: pd.DataFrame, category: str, value: str, fmt: str, title: str,
              top_n: int | None = None, ascending: bool = False,
              reference: float | None = None, reference_label: str = "Overall") -> go.Figure | None:
    """Horizontal bars, one colour, best at the top; optional reference line (e.g. overall)."""
    if not _has(df, category, value):
        return None
    data = df[[category, value]].dropna().sort_values(value, ascending=ascending)
    if top_n:
        data = data.head(top_n)
    data = data.iloc[::-1]                      # plotly draws the first row at the bottom
    labels = data[category].astype(str)
    height = max(CHART_HEIGHT, 44 + 30 * len(data))
    fig = _base(title, height)
    fig.add_trace(go.Bar(
        x=data[value], y=labels.map(wrap_label), orientation="h", marker=dict(color=theme.PRIMARY),
        text=[format_value(v, fmt, compact=True) for v in data[value]], textposition="outside",
        textfont=dict(color=theme.INK_SECONDARY, size=12), cliponaxis=False,
        customdata=[f"{c}<br><b>{format_value(v, fmt)}</b>" for c, v in zip(labels, data[value])],
        hovertemplate="%{customdata}<extra></extra>"))
    if reference is not None and not pd.isna(reference):
        fig.add_vline(x=reference, line=dict(color=theme.INK_MUTED, width=1))
        fig.add_annotation(x=reference, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"{reference_label}: {format_value(reference, fmt)}",
                           font=dict(color=theme.INK_MUTED, size=11))
    _format_axis(fig, list(data[value]) + ([reference] if reference is not None else []), fmt, axis="x")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    fig.update_layout(bargap=0.4)
    return fig


def share_comparison_chart(df: pd.DataFrame, category: str, series: dict[str, str],
                           title: str) -> go.Figure | None:
    """Two or three shares side by side per category (e.g. share of spend vs share of revenue)."""
    cols = [c for c in series.values() if c in df]
    if not _has(df, category) or not cols:
        return None
    data = df.sort_values(cols[0], ascending=True)
    labels = data[category].astype(str)
    fig = _base(title, max(CHART_HEIGHT, 60 + 44 * len(data)))
    for i, (name, col) in enumerate((n, c) for n, c in series.items() if c in df):
        fig.add_trace(go.Bar(
            x=data[col], y=labels.map(wrap_label), orientation="h", name=name,
            marker=dict(color=theme.CATEGORICAL[i]),
            customdata=[f"{c}<br>{name}: <b>{format_value(v, 'percent')}</b>" for c, v in zip(labels, data[col])],
            hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    _show_legend(fig)
    _format_axis(fig, pd.concat([data[c] for c in cols]), "percent", axis="x")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    return fig


def index_chart(df: pd.DataFrame, category: str, index_col: str, title: str,
                lower_is_better: bool) -> go.Figure | None:
    """Efficiency index vs the overall average (100). Bars grow left/right from 100 and are
    coloured by better/worse, with the words 'better'/'worse' in the tooltip (never colour alone)."""
    if not _has(df, category, index_col):
        return None
    data = df[[category, index_col]].dropna().sort_values(index_col, ascending=not lower_is_better)
    data = data.iloc[::-1]
    diff = data[index_col] - 100
    better = diff < 0 if lower_is_better else diff > 0
    labels = data[category].astype(str)
    fig = _base(title, max(CHART_HEIGHT, 44 + 30 * len(data)))
    fig.add_trace(go.Bar(
        x=diff, y=labels.map(wrap_label), base=100, orientation="h",
        marker=dict(color=[theme.GOOD if b else theme.BAD for b in better]),
        text=[f"{v:.0f}" for v in data[index_col]], textposition="outside", cliponaxis=False,
        textfont=dict(color=theme.INK_SECONDARY, size=12),
        customdata=[f"{c}<br>Index <b>{v:.0f}</b> ({'better' if b else 'worse'} than average)"
                    for c, v, b in zip(labels, data[index_col], better)],
        hovertemplate="%{customdata}<extra></extra>"))
    fig.add_vline(x=100, line=dict(color=theme.INK_MUTED, width=1))
    fig.add_annotation(x=100, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                       text="Average = 100", font=dict(color=theme.INK_MUTED, size=11))
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    return fig


# ---------------------------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------------------------

def funnel_chart(funnel: pd.DataFrame, title: str) -> go.Figure | None:
    """Funnel of the count stages that exist (revenue is a value, shown in the table instead)."""
    if not _has(funnel, "stage", "value"):
        return None
    data = funnel[funnel["stage"] != "revenue"].sort_values("stage_order")
    if len(data) < 2:
        return None
    colors = theme.ORDINAL_BLUES[:len(data)]
    text = []
    for _, row in data.iterrows():
        rate = row["rate_from_previous"]
        text.append(format_value(row["value"], "count")
                    + ("" if pd.isna(rate) else f"  ({format_value(rate, 'percent')} of previous)"))
    fig = _base(title, max(CHART_HEIGHT, 70 + 52 * len(data)))
    fig.add_trace(go.Funnel(
        y=data["label"], x=data["value"], text=text, textinfo="text",
        textfont=dict(color=theme.INK, size=12), marker=dict(color=colors),
        connector=dict(line=dict(color=theme.GRID, width=1)),
        customdata=text, hovertemplate="%{y}<br><b>%{customdata}</b><extra></extra>"))
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------------------------
# Anomaly detail
# ---------------------------------------------------------------------------------------------

def anomaly_chart(weekly: pd.DataFrame, metric: str, fmt: str, start, end, expected: float | None,
                  title: str) -> go.Figure | None:
    """Weekly history of one metric for one entity, with the anomaly period shaded and the
    expected level marked across it."""
    if not _has(weekly, "period", metric):
        return None
    fig = line_chart(weekly, "period", metric, fmt, title)
    if fig is None:
        return None
    fig.add_vrect(x0=pd.Timestamp(start), x1=pd.Timestamp(end), fillcolor=theme.WARNING,
                  opacity=0.12, line_width=0, layer="below")
    fig.add_annotation(x=pd.Timestamp(start), y=1.0, yref="paper", yanchor="bottom", xanchor="left",
                       showarrow=False, text="Unusual period", font=dict(color=theme.INK_MUTED, size=11))
    if expected is not None and not pd.isna(expected):
        fig.add_shape(type="line", x0=pd.Timestamp(start), x1=pd.Timestamp(end), y0=expected,
                      y1=expected, line=dict(color=theme.INK_SECONDARY, width=2))
        fig.add_annotation(x=pd.Timestamp(end), y=expected, xanchor="left", showarrow=False,
                           text=f"Expected ~{format_value(expected, fmt, compact=True)}",
                           font=dict(color=theme.INK_SECONDARY, size=11))
    return fig
