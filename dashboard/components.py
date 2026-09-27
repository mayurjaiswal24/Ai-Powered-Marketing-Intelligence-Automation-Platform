"""Reusable UI pieces: headers, KPI cards, empty states, findings and formatted tables."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from analytics.kpis import KPI_REGISTRY
from utils.formatting import format_change, format_count, format_date, format_value

MONEY_COLUMNS = {"spend", "budget", "revenue", "gross_profit", "baseline_money", "observed_money"}
COUNT_COLUMNS = {"rows", "impressions", "clicks", "leads", "qualified_leads", "opportunities",
                 "conversions", "customers", "orders", "active_days", "value", "days"}
NICE_NAMES = {
    "campaign": "Campaign", "channel": "Channel", "segment": "Segment", "rows": "Rows",
    "first_date": "First day", "last_date": "Last day", "active_days": "Active days",
    "trend": "Trend (4 wks)", "trend_change_pct": "Change (4 wks)", "growth_trend": "Trend (4 wks)",
    "growth_pct": "Growth (4 wks)", "anomalies": "Anomalies", "label": "Stage",
    "rate_from_previous": "Conversion from previous", "drop_off_pct": "Drop-off",
}


# ---------------------------------------------------------------------------------------------
# Text blocks
# ---------------------------------------------------------------------------------------------

def page_header(title: str, caption: str | None = None) -> None:
    st.markdown(f"# {html.escape(title)}")
    if caption:
        st.markdown(f"<div class='mi-caption'>{html.escape(caption)}</div>", unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.markdown(f"<div class='mi-empty'>{html.escape(message)}</div>", unsafe_allow_html=True)


def show_chart(fig, empty_message: str) -> None:
    """Draw a chart, or a clear empty state when the data for it does not exist."""
    if fig is None:
        empty_state(empty_message)
    else:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def findings_list(findings, limit: int | None = None) -> None:
    for f in findings[:limit] if limit else findings:
        st.markdown(
            f"<div class='mi-finding'><span class='mi-tag'>{html.escape(f.type)}</span>"
            f"<span class='mi-title'>{html.escape(f.title)}</span><br>{html.escape(f.text)}</div>",
            unsafe_allow_html=True)


def friendly_error(exc: Exception) -> None:
    """Plain-English error. Known errors carry a user_message; anything else gets a generic one
    (the technical detail is kept out of the UI)."""
    message = getattr(exc, "user_message", None) or (
        "Something went wrong while preparing this view. Please reload the data and try again. "
        "If it keeps happening, check that the file is a normal marketing export.")
    st.error(message)


# ---------------------------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------------------------

def kpi_cards(kpis: dict, deltas: dict | None, keys: list[str], per_row: int = 4) -> None:
    """Metric cards for the available KPIs among `keys`, with change vs the previous period.
    Cost KPIs (CPL, CAC) show a fall as good (green) and a rise as bad (red)."""
    shown = [k for k in keys if k in kpis and kpis[k].available]
    for i in range(0, len(shown), per_row):
        cols = st.columns(per_row)
        for col, key in zip(cols, shown[i:i + per_row]):
            k = kpis[key]
            delta, delta_color = None, "off"
            d = deltas.get(key) if deltas else None
            if d is not None and d.pct_change is not None:
                delta = format_change(d.pct_change)
                better = KPI_REGISTRY[key].higher_is_better
                delta_color = "off" if better is None else ("normal" if better else "inverse")
            help_text = f"{KPI_REGISTRY[key].formula_text}. {k.note}".strip()
            with col:
                st.metric(k.label, k.formatted_compact, delta=delta, delta_color=delta_color,
                          help=help_text, border=True)


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def column_label(col: str) -> str:
    if col in NICE_NAMES:
        return NICE_NAMES[col]
    if col in KPI_REGISTRY:
        return KPI_REGISTRY[col].label
    if col.endswith("_share_pct"):
        return f"{col.removesuffix('_share_pct').replace('_', ' ').capitalize()} share"
    if col.endswith("_index"):
        base = col.removesuffix("_index")
        return f"{KPI_REGISTRY[base].label if base in KPI_REGISTRY else base} index"
    return col.replace("_", " ").capitalize()


def column_format(col: str, series: pd.Series) -> str:
    if col in KPI_REGISTRY:
        return KPI_REGISTRY[col].fmt
    if col.endswith("_pct") or col in ("rate_from_previous", "drop_off_pct"):
        return "percent"
    if col.endswith("_index"):
        return "index"
    if col in MONEY_COLUMNS:
        return "money"
    if col in COUNT_COLUMNS:
        return "count"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "text"


def format_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """A display copy with every number formatted the same way as the KPI cards and reports.
    Missing values show as N/A. (Sorting happens BEFORE formatting, via the page's controls.)"""
    columns = [c for c in (columns or list(df.columns)) if c in df]
    out = pd.DataFrame(index=df.index)
    for col in columns:
        fmt = column_format(col, df[col])
        if fmt == "date":
            out[column_label(col)] = df[col].map(format_date)
        elif fmt == "index":
            out[column_label(col)] = df[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:.0f}")
        elif fmt == "text":
            out[column_label(col)] = df[col].astype(object).where(df[col].notna(), "N/A").astype(str)
        elif fmt == "count":
            out[column_label(col)] = df[col].map(format_count)
        else:
            out[column_label(col)] = df[col].map(lambda v, f=fmt: format_value(v, f))
    return out.reset_index(drop=True)


def show_table(df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    if df is None or df.empty:
        empty_state("No rows to show for the current filters.")
        return
    kwargs = {"height": height} if height else {}
    st.dataframe(format_table(df, columns), hide_index=True, width="stretch", **kwargs)
