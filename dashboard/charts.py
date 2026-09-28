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
from analytics.funnel import biggest_drop_off
from dashboard.chart_standard import OWNED_SUFFIX, group_other, label_crosses_line
from utils.formatting import format_date, format_value

CHART_HEIGHT = 320
MAX_INCIDENT_MARKERS = 8        # the highest-ranked incidents only, so a trend line stays readable
LABEL_ROOM = 0.25               # share of the value axis a bar's value label needs on a phone


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


def _base(title: str, height: int = CHART_HEIGHT, subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()
    title = wrap_label(title, theme.TITLE_WRAP)
    extra = 20 * title.count("<br>")          # a wrapped title needs one more line of room
    title_spec = dict(text=title)
    if subtitle:                              # metric, unit and period on a smaller grey line
        title_spec["subtitle"] = dict(text=subtitle, font=dict(size=theme.SUBTITLE_SIZE, color=theme.INK_MUTED))
        title_spec["pad"] = dict(t=16, l=8)   # a little more room so a two-line title is never clipped
        extra += 28
    fig.update_layout(template=theme.TEMPLATE_NAME, title=title_spec, height=height + extra,
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
               x_label_fmt: str = "date", subtitle: str | None = None,
               reference: float | None = None, reference_label: str = "Average") -> go.Figure | None:
    """Single-series trend with a light area wash; the latest value is labelled at the end.
    Optional labelled reference line (e.g. the average for the selected data)."""
    if not _has(df, x, y):
        return None
    data = df[[x, y]].dropna(subset=[y]).sort_values(x)
    if data.empty:
        return None
    fig = _base(title, subtitle=subtitle)
    if len(data) < 2:                          # one point is not a trend: say so inside the chart
        fig.add_annotation(text="Not enough data to show a trend (only one period).", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           font=dict(color=theme.INK_MUTED, size=13))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    x_text = (data[x].map(format_date) if x_label_fmt == "date" else data[x].astype(str))
    hover = [f"{xt}<br><b>{format_value(v, fmt)}</b>" for xt, v in zip(x_text, data[y])]
    if reference is not None and not pd.isna(reference):
        hover = [f"{h}<br>{_vs_reference(v, reference, reference_label)}" for h, v in zip(hover, data[y])]
    fig.add_trace(go.Scatter(
        x=data[x], y=data[y], mode="lines", line=dict(color=theme.PRIMARY, width=2),
        fill="tozeroy", fillcolor=_hex_to_rgba(theme.PRIMARY, 0.10),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    last = data.iloc[-1]
    # The latest-value label sits on the side of the point away from the reference line.
    below = reference is not None and not pd.isna(reference) and last[y] < reference
    fig.add_trace(go.Scatter(
        x=[last[x]], y=[last[y]], mode="markers+text", text=[format_value(last[y], fmt, compact=True)],
        textposition="bottom left" if below else "top left", textfont=dict(color=theme.INK_SECONDARY, size=12),
        marker=dict(size=8, color=theme.PRIMARY, line=dict(color=theme.SURFACE, width=2)),
        hoverinfo="skip"))
    if reference is not None and not pd.isna(reference):
        fig.add_hline(y=reference, line=dict(color=theme.INK_MUTED, width=1, dash="dot"))
        fig.add_annotation(x=0, xref="paper", y=reference, yanchor="bottom", xanchor="left", showarrow=False,
                           text=f"{reference_label}: {format_value(reference, fmt, compact=True)}",
                           font=dict(color=theme.INK_MUTED, size=11), bgcolor=_hex_to_rgba(theme.SURFACE, 0.85))
    _format_axis(fig, list(data[y]) + ([reference] if reference is not None else []), fmt)
    fig.update_layout(hovermode="x unified" if len(data) > 1 else "closest")
    return fig


def _vs_reference(value, reference, label: str) -> str:
    """'+12.3% vs Average' for tooltips (a display comparison, not a KPI)."""
    if value is None or pd.isna(value) or not reference:
        return f"{label}: N/A"
    return f"{(value - reference) / abs(reference) * 100:+.1f}% vs {label}"


def add_incident_markers(fig: go.Figure, incidents: pd.DataFrame | None) -> go.Figure:
    """Mark anomaly incidents on a trend line: a light band over each incident period and a
    marker at the top with the incident name and its estimated ₹ impact on hover."""
    if fig is None or incidents is None or incidents.empty or not fig.data:
        return fig
    line = fig.data[0]
    values = pd.Series(line.y, dtype=float).dropna()
    if values.empty:
        return fig
    x_min, x_max = pd.Series(line.x).min(), pd.Series(line.x).max()
    shown = incidents.sort_values("rank").head(MAX_INCIDENT_MARKERS)
    shown = shown[(shown["period_end"] >= x_min) & (shown["period_start"] <= x_max)]
    if shown.empty:
        return fig
    top = float(values.max()) * 1.06
    hover = []
    for _, inc in shown.iterrows():
        direction = inc.get("impact_direction")
        impact = (f"{format_value(inc['impact_inr'], 'money', compact=True)} {direction} ({inc['impact_label']})"
                  if direction in ("loss", "gain") and inc.get("impact_inr") else "Not estimated")
        hover.append(f"<b>Incident #{inc['rank']}</b><br>{inc['headline']}<br>Estimated impact: {impact}")
        fig.add_vrect(x0=inc["period_start"], x1=inc["period_end"], fillcolor=theme.WARNING, opacity=0.10,
                      line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=list(shown["period_start"]), y=[top] * len(shown), mode="markers", name="Incidents",
        marker=dict(symbol="triangle-down", size=11, color=theme.WARNING, line=dict(color=theme.INK, width=0.6)),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------------------------

def bar_chart(df: pd.DataFrame, category: str, value: str, fmt: str, title: str,
              top_n: int | None = None, ascending: bool = False,
              reference: float | None = None, reference_label: str = "Overall",
              subtitle: str | None = None, color_by: dict[str, str] | None = None,
              share_total: float | None = None) -> go.Figure | None:
    """Horizontal bars, best at the top; optional labelled reference line (e.g. overall).
    Colour: the fixed entity colour when `color_by` is given (channels), otherwise the top bar in
    the focus colour and the rest in context grey. Hover: value, share of `share_total` (for
    values that add up) and the difference from the reference line."""
    if not _has(df, category, value):
        return None
    data = df[[category, value]].dropna().sort_values(value, ascending=ascending)
    if top_n:
        data = data.head(top_n)
    labels_top_first = data[category].astype(str)
    data = data.iloc[::-1]                      # plotly draws the first row at the bottom
    labels = data[category].astype(str)
    wrapped = labels.map(wrap_label)
    row = 46 if wrapped.str.contains("<br>").any() else 30     # two-line labels need taller rows
    height = max(CHART_HEIGHT, 44 + row * len(data))
    fig = _base(title, height, subtitle)
    if color_by:
        colors = [color_by.get(c, theme.NEUTRAL) for c in labels]
    else:
        focus = labels_top_first.iloc[0]
        colors = [theme.FOCUS if c == focus and c != theme.OTHER_LABEL else theme.CONTEXT for c in labels]
    hover = []
    for c, v in zip(labels, data[value]):
        lines = [c, f"<b>{format_value(v, fmt)}</b>"]
        if share_total:
            lines.append(f"{format_value(v / share_total * 100, 'percent')} of total")
        if reference is not None and not pd.isna(reference):
            lines.append(_vs_reference(v, reference, reference_label))
        hover.append("<br>".join(lines))
    # A value label that the reference line would cross is printed just after the line instead.
    axis_max = max([float(v) for v in data[value]] + ([float(reference)] if reference is not None else []))
    moved = [label_crosses_line(v, reference, axis_max, LABEL_ROOM) for v in data[value]]
    fig.add_trace(go.Bar(
        x=data[value], y=wrapped, orientation="h", marker=dict(color=colors),
        text=["" if m else format_value(v, fmt, compact=True) for v, m in zip(data[value], moved)],
        textposition="outside",
        textfont=dict(color=theme.INK_SECONDARY, size=12), cliponaxis=False,
        customdata=np.column_stack([hover, list(labels)]),     # [hover text, raw label for click events]
        hovertemplate="%{customdata[0]}<extra></extra>"))
    for v, w, m in zip(data[value], wrapped, moved):
        if m:
            fig.add_annotation(x=reference, y=w, xanchor="left", xshift=4, showarrow=False,
                               text=format_value(v, fmt, compact=True),
                               font=dict(color=theme.INK_SECONDARY, size=12))
    if reference is not None and not pd.isna(reference):
        fig.add_vline(x=reference, line=dict(color=theme.INK_MUTED, width=1, dash="dot"),
                      layer="below")          # value labels stay readable where the line crosses them
        fig.add_annotation(x=reference, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"{reference_label}: {format_value(reference, fmt)}",
                           font=dict(color=theme.INK_MUTED, size=11))
    _format_axis(fig, list(data[value]) + ([reference] if reference is not None else []), fmt, axis="x")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    fig.update_layout(bargap=0.4, margin=dict(r=theme.VALUE_LABEL_MARGIN))   # room for the value labels
    return fig


SHARE_SERIES_COLORS = [theme.CONTEXT, theme.FOCUS, theme.ACCENT]   # spend = context, result = focus


def share_comparison_chart(df: pd.DataFrame, category: str, series: dict[str, str],
                           title: str, subtitle: str | None = None,
                           owned: set[str] | None = None) -> go.Figure | None:
    """Two or three shares side by side per category (e.g. share of spend in grey vs share of
    revenue in the focus colour). More than 8 categories fold into "Other" (shares add up).
    Owned channels (`owned`) are labelled "(owned)" and drawn lighter and hatched, so they are
    not read as paid media."""
    cols = [c for c in series.values() if c in df]
    if not _has(df, category) or not cols:
        return None
    data = group_other(df, category, cols, sort_by=cols[0])
    data = data.sort_values(cols[0], ascending=True)
    is_owned = [c in (owned or set()) for c in data[category].astype(str)]
    display_order = [wrap_label(f"{c}{OWNED_SUFFIX}" if o else c) for c, o in zip(data[category].astype(str), is_owned)]
    if is_owned[0] and not all(is_owned):
        # Plotly draws the legend swatch from the first bar, so a paid row goes first in the data;
        # the axis keeps the display order.
        first = is_owned.index(False)
        data = pd.concat([data.iloc[[first]], data.drop(data.index[first])])
        is_owned = [is_owned[first]] + is_owned[:first] + is_owned[first + 1:]
    labels = pd.Series([f"{c}{OWNED_SUFFIX}" if o else c for c, o in zip(data[category].astype(str), is_owned)])
    fig = _base(title, max(CHART_HEIGHT, 60 + 44 * len(data)), subtitle)
    for i, (name, col) in enumerate((n, c) for n, c in series.items() if c in df):
        color = SHARE_SERIES_COLORS[i]
        fig.add_trace(go.Bar(
            x=data[col], y=labels.map(wrap_label), orientation="h", name=name,
            marker=dict(color=[_hex_to_rgba(color, 0.45) if o else color for o in is_owned],
                        pattern=dict(shape=["/" if o else "" for o in is_owned], fgcolor=color, size=6))
            if any(is_owned) else dict(color=color),
            text=[format_value(v, "percent") for v in data[col]], textposition="outside", cliponaxis=False,
            textfont=dict(color=theme.INK_SECONDARY, size=11),
            customdata=[f"{c}<br>{name}: <b>{format_value(v, 'percent')}</b>" for c, v in zip(labels, data[col])],
            hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, margin=dict(r=theme.VALUE_LABEL_MARGIN))
    fig.update_yaxes(categoryorder="array", categoryarray=display_order)
    _show_legend(fig)
    _format_axis(fig, pd.concat([data[c] for c in cols]), "percent", axis="x")
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    return fig


def index_chart(df: pd.DataFrame, category: str, index_col: str, title: str,
                lower_is_better: bool, subtitle: str | None = None) -> go.Figure | None:
    """Efficiency index vs the overall average (100). Bars grow left/right from 100 and are
    coloured by better/worse, with the words 'better'/'worse' in the tooltip (never colour alone)."""
    if not _has(df, category, index_col):
        return None
    data = df[[category, index_col]].dropna().sort_values(index_col, ascending=not lower_is_better)
    data = data.iloc[::-1]
    diff = data[index_col] - 100
    better = diff < 0 if lower_is_better else diff > 0
    labels = data[category].astype(str)
    fig = _base(title, max(CHART_HEIGHT, 44 + 30 * len(data)), subtitle)
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
    fig.update_layout(margin=dict(r=theme.VALUE_LABEL_MARGIN))              # room for the value labels
    return fig


# ---------------------------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------------------------

def funnel_chart(funnel: pd.DataFrame, title: str, subtitle: str | None = None) -> go.Figure | None:
    """Funnel as one bar per step: the share of people who move on from the previous stage
    (0-100%), labelled with the count reached. All steps share one 0-100% scale, so the small
    lower stages stay visible (a width-by-count funnel hides them). The biggest drop after the
    click stage (the same step as the funnel finding) is red; the other steps are grey."""
    if not _has(funnel, "stage", "value"):
        return None
    data = funnel[funnel["stage"] != "revenue"].sort_values("stage_order").reset_index(drop=True)
    if len(data) < 2:
        return None
    worst = biggest_drop_off(data)
    steps = data.iloc[1:]
    rows = [f"{p} → {c}" for p, c in zip(data["label"].iloc[:-1], steps["label"])]
    rates = list(steps["rate_from_previous"])
    text = [f"{format_value(r, 'percent')} move on · {format_value(v, 'count')} {str(c).lower()}"
            for r, v, c in zip(rates, steps["value"], steps["label"])]
    colors = [theme.BAD if worst is not None and stage == worst["stage"] else theme.CONTEXT
              for stage in steps["stage"]]
    hover = [f"{row}<br><b>{format_value(r, 'percent')}</b> move on, "
             f"{format_value(None if pd.isna(r) else 100 - r, 'percent')} drop off"
             f"<br>{format_value(v, 'count')} reached" for row, r, v in zip(rows, rates, steps["value"])]
    first = data.iloc[0]
    note = f"Starting from {format_value(first['value'], 'count')} {str(first['label']).lower()}"
    fig = _base(title, max(CHART_HEIGHT, 70 + 52 * len(steps)),
                f"{subtitle} · {note}" if subtitle else note)
    fig.add_trace(go.Bar(
        x=rates[::-1], y=[wrap_label(r) for r in rows[::-1]], orientation="h",
        marker=dict(color=colors[::-1]), text=text[::-1], textposition="outside", cliponaxis=False,
        textfont=dict(color=theme.INK_SECONDARY, size=12),
        customdata=hover[::-1], hovertemplate="%{customdata}<extra></extra>"))
    # The axis runs past 100% so the labels printed after each bar fit inside the chart.
    fig.update_xaxes(range=[0, 175], showgrid=True, gridcolor=theme.GRID, tickmode="array",
                     tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"])
    fig.update_yaxes(showgrid=False, automargin=True)
    fig.update_layout(bargap=0.45, margin=dict(r=16))
    return fig


# ---------------------------------------------------------------------------------------------
# Anomaly detail
# ---------------------------------------------------------------------------------------------

def anomaly_chart(weekly: pd.DataFrame, metric: str, fmt: str, start, end, expected: float | None,
                  title: str, subtitle: str | None = None) -> go.Figure | None:
    """Weekly history of one metric for one entity, with the anomaly period shaded and the
    expected level marked across it."""
    if not _has(weekly, "period", metric):
        return None
    fig = line_chart(weekly, "period", metric, fmt, title, subtitle=subtitle)
    if fig is None:
        return None
    fig.add_vrect(x0=pd.Timestamp(start), x1=pd.Timestamp(end), fillcolor=theme.WARNING,
                  opacity=0.12, line_width=0, layer="below")
    fig.add_annotation(x=pd.Timestamp(start), y=1.0, yref="paper", yanchor="bottom", xanchor="left",
                       showarrow=False, text="Unusual Period", font=dict(color=theme.INK_MUTED, size=11))
    if expected is not None and not pd.isna(expected):
        fig.add_shape(type="line", x0=pd.Timestamp(start), x1=pd.Timestamp(end), y0=expected,
                      y1=expected, line=dict(color=theme.INK_SECONDARY, width=2))
        fig.add_annotation(x=pd.Timestamp(end), y=expected, xanchor="left", showarrow=False,
                           text=f"Expected ~{format_value(expected, fmt, compact=True)}",
                           font=dict(color=theme.INK_SECONDARY, size=11))
    return fig


# ---------------------------------------------------------------------------------------------
# Profitability (U4)
# ---------------------------------------------------------------------------------------------

def breakeven_scatter(df: pd.DataFrame, title: str, subtitle: str | None = None) -> go.Figure | None:
    """Each campaign's ROAS (up) against its own break-even ROAS (across), bubble size = spend,
    colour = profitability status (the status is also in the legend and tooltip). Campaigns above
    the labelled diagonal break-even line make money after gross profit; below it they lose money."""
    if not _has(df, "roas", "break_even_roas", "campaign"):
        return None
    data = df.dropna(subset=["roas", "break_even_roas"])
    if data.empty:
        return None
    fig = _base(title, CHART_HEIGHT + 80, subtitle)
    spend = data["spend"].astype(float).clip(lower=0) if "spend" in data else pd.Series(1.0, index=data.index)
    biggest = float(spend.max()) or 1.0
    sizes = 8 + 30 * np.sqrt(spend / biggest)                 # area grows with spend
    for status, color in theme.PROFIT_STATUS_COLORS.items():
        rows = data[data["status"] == status]
        if rows.empty:
            continue
        hover = [f"<b>{r.campaign}</b><br>ROAS <b>{format_value(r.roas, 'ratio')}</b> vs break-even "
                 f"{format_value(r.break_even_roas, 'ratio')}<br>Headroom {format_value(r.roas_headroom, 'percent')}"
                 f"<br>Contribution {format_value(r.contribution, 'money', compact=True)}"
                 f"<br>Spend {format_value(r.spend, 'money', compact=True)}<br>{status}"
                 for r in rows.itertuples()]
        fig.add_trace(go.Scatter(
            x=rows["break_even_roas"], y=rows["roas"], mode="markers", name=status,
            marker=dict(size=sizes[rows.index], color=_hex_to_rgba(color, 0.75), line=dict(color=color, width=1)),
            customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    # Break-even ROAS values sit close together, so the x axis is zoomed to their range (the ROAS
    # axis starts at zero); the line ROAS = break-even ROAS runs across the whole x range.
    x_low, x_high = float(data["break_even_roas"].min()), float(data["break_even_roas"].max())
    pad = max((x_high - x_low) * 0.15, x_high * 0.05)
    x_bottom, x_top = max(x_low - pad, 0.0), x_high + pad
    y_top = max(float(data["roas"].max()), x_top) * 1.1
    fig.add_shape(type="line", x0=x_bottom, y0=x_bottom, x1=x_top, y1=x_top, line=dict(color=theme.INK_MUTED, width=1, dash="dot"),
                  layer="below")
    fig.add_annotation(x=x_top, y=x_top, xanchor="right", yanchor="bottom", showarrow=False,
                       text="Break-Even Line", font=dict(color=theme.INK_MUTED, size=11),
                       bgcolor=_hex_to_rgba(theme.SURFACE, 0.85))
    fig.update_xaxes(range=[x_bottom, x_top], title_text="Break-Even ROAS (x)", showgrid=True, gridcolor=theme.GRID)
    fig.update_yaxes(range=[0, y_top], title_text="ROAS (x)", showgrid=True, gridcolor=theme.GRID)
    fig.update_xaxes(tickformat=".2f", ticksuffix="x")
    _format_axis(fig, [0, y_top], "ratio", axis="y")
    _show_legend(fig)
    fig.update_layout(hovermode="closest")
    return fig
