"""Marketing Intelligence Platform - Streamlit entry point.

Run with:  streamlit run app.py   (inside the activated virtual environment)
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

# Browser tab: product name and the "MJ" monogram (assets/brand/mj_icon.png).
_ICON = Path(__file__).resolve().parent / "assets" / "brand" / "mj_icon.png"
st.set_page_config(page_title="Marketing Intelligence Platform",
                   page_icon=str(_ICON) if _ICON.exists() else None,
                   layout="wide", initial_sidebar_state="auto")

from dashboard import components as ui  # noqa: E402  (after set_page_config)
from dashboard import layout  # noqa: E402
from dashboard.theme import APP_CSS  # noqa: E402

logger = logging.getLogger("marketing_intelligence")

PAGE_RENDERERS = {
    "Executive Overview": layout.page_overview,
    "Performance Trends": layout.page_trends,
    "Channels": layout.page_channels,
    "Campaigns": layout.page_campaigns,
    "Funnel": layout.page_funnel,
    "Segments": layout.page_segments,
    "Anomalies": layout.page_anomalies,
    "Profitability": layout.page_profitability,
    "Targets": layout.page_targets,
}


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    layout.startup()                  # public mode: delete runs older than 24 hours (once)
    page, filters = layout.sidebar()
    try:
        if page == "Upload & Profile":
            layout.page_upload()
        elif page == "About":
            layout.page_about()
        elif layout.current_output() is None:
            ui.page_header(page, layout.PAGE_DESCRIPTIONS.get(page),
                           eyebrow=layout.PAGE_SECTION.get(page) or None)
            layout.not_ready()
        elif page in PAGE_RENDERERS:
            PAGE_RENDERERS[page](filters)
        elif page == "AI Insights":
            layout.page_ai()
        elif page == "Data Quality":
            layout.page_quality()
        elif page == "Reports":
            layout.page_reports()
    except Exception as exc:  # noqa: BLE001 - plain-English message instead of a traceback
        logger.exception("Error while rendering %s", page)
        ui.friendly_error(exc)


main()
