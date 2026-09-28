"""Reusable UI pieces: headers, KPI cards, empty states, findings and formatted tables."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import config.settings as config_settings
from analytics.kpis import KPI_REGISTRY
from dashboard.tables import column_format, column_label, format_table  # noqa: F401
from utils.formatting import format_change

# ---------------------------------------------------------------------------------------------
# Text blocks
# ---------------------------------------------------------------------------------------------

def page_header(title: str, caption: str | None = None, eyebrow: str | None = None,
                scope: str | None = None) -> None:
    """The same header on every page: section label, title, one-line description and (on
    analysis pages) which data is shown. The look comes from theme.APP_CSS."""
    parts = ["<div class='mi-page-header'>"]
    if eyebrow:
        parts.append(f"<div class='mi-eyebrow'>{html.escape(eyebrow)}</div>")
    parts.append(f"<div class='mi-page-title'>{html.escape(title)}</div>")
    if caption:
        parts.append(f"<p>{html.escape(caption)}</p>")
    if scope:
        parts.append(f"<div class='mi-scope'>{html.escape(scope)}</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def brand_header() -> None:
    """Sidebar top: a small bar-chart logo mark and the product name."""
    st.markdown(
        "<div class='mi-brand'><div class='mi-logo'><span style='height:9px'></span>"
        "<span style='height:14px'></span><span style='height:19px'></span></div>"
        f"<div><div class='mi-brand-name'>{html.escape(config_settings.APP_NAME)}</div>"
        "<div class='mi-brand-sub'>Marketing analytics &amp; AI insights</div></div></div>",
        unsafe_allow_html=True)


def signature() -> None:
    """Sidebar footer on every page: creator (LinkedIn, new tab) and version."""
    st.markdown(
        "<div class='mi-signature'>Created by "
        f"<a href='{html.escape(config_settings.CREATOR_LINKEDIN)}' target='_blank' "
        f"rel='noopener noreferrer'>{html.escape(config_settings.CREATOR_NAME)}</a>"
        f"<br>v{html.escape(config_settings.APP_VERSION)}</div>",
        unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f"<div class='mi-sidebar-label'>{html.escape(text)}</div>", unsafe_allow_html=True)


def steps_guide(steps: list[tuple[str, str]]) -> None:
    """Numbered how-it-works cards (one row on a laptop, stacked on a phone)."""
    cards = "".join(
        f"<div class='mi-step'><span class='mi-step-no'>{i}</span><span class='mi-step-title'>"
        f"{html.escape(title)}</span><p>{html.escape(text)}</p></div>"
        for i, (title, text) in enumerate(steps, start=1))
    st.markdown(f"<div class='mi-steps'>{cards}</div>", unsafe_allow_html=True)


def card_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='mi-card-title'>{html.escape(title)}</div>"
                + (f"<div class='mi-card-sub'>{html.escape(subtitle)}</div>" if subtitle else ""),
                unsafe_allow_html=True)


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
