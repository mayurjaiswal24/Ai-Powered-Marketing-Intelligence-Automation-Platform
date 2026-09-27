"""Reusable UI pieces: headers, KPI cards, empty states, findings and formatted tables."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from analytics.kpis import KPI_REGISTRY
from dashboard.tables import column_format, column_label, format_table  # noqa: F401
from utils.formatting import format_change

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

def show_table(df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    if df is None or df.empty:
        empty_state("No rows to show for the current filters.")
        return
    kwargs = {"height": height} if height else {}
    st.dataframe(format_table(df, columns), hide_index=True, width="stretch", **kwargs)
