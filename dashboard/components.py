"""Reusable UI pieces: headers, KPI cards, empty states, findings and formatted tables."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import config.settings as config_settings
from analytics.kpis import KPI_REGISTRY
from dashboard.tables import column_format, column_label, format_table  # noqa: F401
from utils.formatting import format_change, title_case, title_case_label

# ---------------------------------------------------------------------------------------------
# Text blocks
# ---------------------------------------------------------------------------------------------

def page_header(title: str, caption: str | None = None, eyebrow: str | None = None,
                scope: str | list[str] | None = None) -> None:
    """The same header on every page: section label, title, one-line description and (on
    analysis pages) which data is shown, as one chip per active filter. The look comes from
    theme.APP_CSS."""
    parts = ["<div class='mi-page-header'>"]
    if eyebrow:
        parts.append(f"<div class='mi-eyebrow'>{html.escape(eyebrow)}</div>")
    parts.append(f"<div class='mi-page-title'>{html.escape(title)}</div>")
    if caption:
        parts.append(f"<p>{html.escape(caption)}</p>")
    if scope:
        chips = [scope] if isinstance(scope, str) else scope
        parts.append("<div class='mi-chips'>" + "".join(
            f"<span class='mi-scope'>{html.escape(chip)}</span>" for chip in chips) + "</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


_DEFAULT_NAME = "Marketing Intelligence Platform"


def brand_header() -> None:
    """Sidebar top: a small bar-chart logo mark and the product name."""
    st.markdown(
        "<div class='mi-brand'><div class='mi-logo'><span style='height:9px'></span>"
        "<span style='height:14px'></span><span style='height:19px'></span></div>"
        f"<div><div class='mi-brand-name'>{html.escape(_detail('APP_NAME') or _DEFAULT_NAME)}</div>"
        "<div class='mi-brand-sub'>Marketing analytics &amp; AI insights</div></div></div>",
        unsafe_allow_html=True)


def _detail(name: str) -> str:
    """A product/creator detail from config/settings.py, or "" if it is missing, so the sidebar
    footer and the About page can never fail because of a settings change."""
    value = getattr(config_settings, name, "")
    return value if isinstance(value, str) else ""


def _creator_link() -> str:
    name, url = html.escape(_detail("CREATOR_NAME")), _detail("CREATOR_LINKEDIN")
    if url.startswith("https://"):
        return (f"<a href='{html.escape(url)}' target='_blank' "
                f"rel='noopener noreferrer'>{name}</a>")
    return name


def signature() -> None:
    """Sidebar footer on every page: creator (LinkedIn, new tab) and version."""
    parts = []
    if _detail("CREATOR_NAME"):
        parts.append(f"Created by {_creator_link()}")
    if _detail("APP_VERSION"):
        parts.append(f"v{html.escape(_detail('APP_VERSION'))}")
    if parts:
        st.markdown(f"<div class='mi-signature'>{'<br>'.join(parts)}</div>", unsafe_allow_html=True)


def creator_card() -> None:
    """About page: who built the platform. Rows whose setting is missing are simply left out."""
    card_title("Created by")
    parts = []
    if _detail("CREATOR_NAME"):
        parts.append(f"<div class='mi-creator-name'>{_creator_link()}</div>")
    for label, name in (("Education", "CREATOR_TITLE"), ("Previously", "CREATOR_EDUCATION_PREVIOUS"),
                        ("Focus", "CREATOR_FOCUS")):
        if _detail(name):
            parts.append(f"<div class='mi-creator-row'><b>{label}:</b> "
                         f"{html.escape(_detail(name))}</div>")
    if _detail("CREATOR_OPEN_TO"):
        parts.append(f"<div class='mi-open-to'>{html.escape(_detail('CREATOR_OPEN_TO'))}</div>")
    if _detail("APP_VERSION"):
        parts.append(f"<div class='mi-creator-row'>{html.escape(_detail('APP_NAME'))} "
                     f"v{html.escape(_detail('APP_VERSION'))}</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    if _detail("CREATOR_LINKEDIN").startswith("https://"):
        st.link_button("Connect on LinkedIn", _detail("CREATOR_LINKEDIN"), icon=":material/open_in_new:")


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


def empty_state(message: str, icon: str = "info", action=None) -> None:
    """The same empty state everywhere: an icon, one sentence and, when there is a useful next
    step, one action button. `icon` is a Material Symbols name; `action` is (label, callback, key)."""
    st.markdown(f"<div class='mi-empty'><span class='mi-empty-icon' aria-hidden='true'>"
                f"{html.escape(icon)}</span><span>{html.escape(message)}</span></div>",
                unsafe_allow_html=True)
    if action is not None:
        label, callback, key = action
        st.button(label, key=key, on_click=callback, type="primary")


# Chart toolbar (docs/CHART_STYLE_GUIDE.md): Download as PNG and Reset zoom only.
CHART_CONFIG = {
    "displaylogo": False,
    "modeBarButtons": [["toImage", "resetScale2d"]],
    "toImageButtonOptions": {"format": "png", "filename": "chart", "scale": 2},
}


def show_chart(fig, empty_message: str, key: str | None = None, selectable: bool = False):
    """Draw a chart, or a clear empty state when the data for it does not exist. With
    `selectable`, clicking a bar reruns the page and the click event is returned."""
    if fig is None:
        empty_state(empty_message)
        return None
    if selectable:
        return st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=key, on_select="rerun",
                               selection_mode="points")
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=key)
    return None


def findings_list(findings, limit: int | None = None) -> None:
    for f in findings[:limit] if limit else findings:
        st.markdown(
            f"<div class='mi-finding'><span class='mi-tag'>{html.escape(f.type)}</span>"
            f"<span class='mi-title'>{html.escape(title_case_label(f.title))}</span><br>{html.escape(f.text)}</div>",
            unsafe_allow_html=True)


def friendly_error(exc: Exception) -> None:
    """Plain-English error. Known errors carry a user_message; anything else gets a generic one
    (the technical detail is kept out of the UI)."""
    message = getattr(exc, "user_message", None) or (
        "Something went wrong while preparing this view. Please reload the data and try again. "
        "If it keeps happening, check that the file is a standard marketing export.")
    st.error(message)


# ---------------------------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------------------------

def kpi_cards(kpis: dict, deltas: dict | None, keys: list[str], per_row: int = 4,
              sparklines: dict[str, list] | None = None, notes: dict[str, str] | None = None) -> None:
    """Metric cards for the available KPIs among `keys`, with change vs the previous period and
    (when given) a sparkline of the last 12 weeks. Cost KPIs (CPL, CAC) show a fall as good
    (green) and a rise as bad (red). `notes` (U5): a small line under a card, e.g.
    'Target ₹9,000 · On Target'; cards without a note are unchanged."""
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
            spec = KPI_REGISTRY[key]
            help_text = f"{spec.description}  \nFormula: {spec.formula_text}." + (f"  \n{k.note}" if k.note else "")
            spark = (sparklines or {}).get(key)
            chart = dict(chart_data=spark, chart_type="line") if spark is not None and len(spark) > 1 else {}
            with col:
                st.metric(title_case(k.label), k.formatted_compact, delta=delta, delta_color=delta_color,
                          help=help_text, border=True, **chart)
                if notes and key in notes:
                    st.caption(notes[key])


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def show_table(df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None,
               wrap_headers: bool = False) -> None:
    """`wrap_headers`: a small static table whose long column names wrap instead of being cut off."""
    if df is None or df.empty:
        empty_state("No rows match the current filters.", icon="filter_alt_off")
        return
    if wrap_headers:
        st.table(format_table(df, columns), hide_index=True)
        return
    kwargs = {"height": height} if height else {}
    st.dataframe(format_table(df, columns), hide_index=True, width="stretch", **kwargs)
