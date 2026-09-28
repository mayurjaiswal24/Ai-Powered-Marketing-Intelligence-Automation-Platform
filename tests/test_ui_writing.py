"""UI writing standard (docs/UI_STYLE_GUIDE.md): Title Case helpers, empty states with an action,
KPI tooltips from the registry, and the exact About copy."""

from __future__ import annotations

from pathlib import Path

import pytest

import database.connection as dbc
from config.settings import Settings
from dashboard import layout
from utils.formatting import as_sentence, format_date_range, title_case, title_case_label

ROOT = Path(__file__).resolve().parents[1]


def test_title_case_rules():
    assert title_case("Click-to-lead rate") == "Click-to-Lead Rate"
    assert title_case("Share of spend") == "Share of Spend"
    assert title_case("Cost per click (CPC)") == "Cost per Click (CPC)"
    assert title_case("where the money goes and what it returns") == "Where the Money Goes and What It Returns"
    assert title_case("ROAS by paid channel") == "ROAS by Paid Channel"          # acronyms kept
    assert title_case("LinkedIn Ads") == "LinkedIn Ads"                          # brand casing kept
    assert title_case(None) is None


def test_label_only_title_case_keeps_entity_names():
    assert title_case_label("Top campaign by conversions: summer push - brand") == \
        "Top Campaign by Conversions: summer push - brand"


def test_sentence_and_date_range():
    assert as_sentence("extra spaces removed") == "Extra spaces removed."
    assert as_sentence("Already a sentence.") == "Already a sentence."
    assert as_sentence("") == ""
    import pandas as pd
    assert format_date_range(pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-30")) == "1 Oct 2025 – 30 Sep 2026"


def test_about_copy_is_exact():
    titles = [t for t, _ in layout.ABOUT_FEATURES]
    assert titles == ["Upload Any Marketing Export", "Clean and Validate the Data", "Calculate Verified KPIs",
                      "Detect What Changed", "Explain Results in Plain Language", "Share the Results"]
    assert all(text.endswith(".") and text[0].isupper() for _, text in layout.ABOUT_FEATURES)
    assert layout.ABOUT_FOOTER.startswith("Built with Python, pandas, Streamlit, Plotly, SQLite, Google "
                                          "Gemini, ReportLab, Matplotlib and XlsxWriter.")
    assert layout.PAGE_DESCRIPTIONS["About"] == "What the platform does and who created it."


def test_page_descriptions_are_sentences():
    for page, text in layout.PAGE_DESCRIPTIONS.items():
        assert text[0].isupper() and text.endswith("."), page


@pytest.fixture
def app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)


def test_empty_state_offers_one_action_that_works(app):
    at = app.run()
    at.sidebar.radio(key="page").set_value("Funnel").run()
    assert any("No analysis is open yet" in m.value for m in at.markdown)
    at.button(key="empty_go_upload").click().run()
    assert not at.exception and at.sidebar.radio(key="page").value == "Upload & Profile"


def test_kpi_tooltips_and_clear_filters(app):
    at = app.run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    at.sidebar.radio(key="page").set_value("Executive Overview").run()
    cards = {m.label: m for m in at.metric}
    assert "Lead-to-Conversion Rate" in cards                   # Title Case card label
    help_text = cards["CPL"].help
    assert "Average cost of one lead." in help_text and "Formula: Spend / Leads." in help_text
    at.sidebar.radio(key="date_preset").set_value("Last 30 days").run()
    at.sidebar.button(key="clear_filters").click().run()
    assert not at.exception and at.sidebar.radio(key="date_preset").value == "All dates"
