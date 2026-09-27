"""Marketing Intelligence Platform - Streamlit entry point.

Run with:  streamlit run app.py   (inside the activated virtual environment)
"""

from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(page_title="Marketing Intelligence Platform", layout="wide",
                   initial_sidebar_state="expanded")

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
}


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    layout.startup()                  # public mode: delete runs older than 24 hours (once)
    page, filters = layout.sidebar()
    try:
        if page == "Upload & Profile":
            layout.page_upload()
        elif layout.current_output() is None:
            ui.page_header(page)
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
