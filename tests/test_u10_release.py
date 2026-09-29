"""U10 tests: one version number (APP_VERSION) shown identically in the sidebar, the About page, the PDF
footers and the Excel title block; the About page's Version Roadmap renders in both views."""

import io
from pathlib import Path

import pytest
from openpyxl import load_workbook
from pypdf import PdfReader

import config.settings as config_settings
import database.connection as dbc
from config.settings import Settings
from dashboard import layout
from dashboard.views import basic
from reports.excel_report import generate_workbook
from reports.pdf_report import generate_pdf
from reports.summary_report import generate_summary_pdf

ROOT = Path(__file__).resolve().parent.parent
SHOWN = f"v{config_settings.APP_VERSION}"


@pytest.fixture(scope="module")
def output(report_outputs):
    return next(iter(report_outputs.values()))


def test_version_is_2_0_and_in_the_signature():
    assert config_settings.APP_VERSION == "2.0"
    assert config_settings.REPORT_SIGNATURE == ("Prepared with Marketing Intelligence Platform v2.0 · "
                                                "Created by Mayur Jaiswal")


def test_pdf_footers_and_excel_title_block_show_the_version(output, tmp_path):
    generate_pdf(output.analysis, tmp_path / "r.pdf")
    pages = [p.extract_text() for p in PdfReader(str(tmp_path / "r.pdf")).pages]
    assert all(config_settings.REPORT_SIGNATURE in " ".join(t.split()) for t in pages[1:])   # every footer
    buf = io.BytesIO()
    generate_summary_pdf(output.analysis, buf)
    assert config_settings.REPORT_SIGNATURE in " ".join(PdfReader(buf).pages[0].extract_text().split())
    generate_workbook(output.analysis, tmp_path / "w.xlsx")
    assert load_workbook(tmp_path / "w.xlsx")["Executive_KPIs"]["A3"].value == config_settings.REPORT_SIGNATURE


def test_roadmap_badges_follow_app_version():
    now = layout.roadmap_html("2.0")
    assert now.count("Current") == 1 and now.count("Planned") == 2        # the v3.0 name is also "Planned"
    assert now.index("v2.0 · Decision Intelligence") < now.index(">Current<") < now.index("v3.0 · Planned")
    for version, name, text in layout.ABOUT_ROADMAP:
        assert f"v{version} · {name}" in now and text.replace("'", "&#x27;") in now
    later = layout.roadmap_html("3.0")
    assert "mi-badge-planned" not in later and later.index(">Current<") > later.index("v3.0 · Planned")


def test_roadmap_copy_is_exact():
    assert [(v, n) for v, n, _ in layout.ABOUT_ROADMAP] == [
        ("1.0", "Foundation"), ("2.0", "Decision Intelligence"), ("3.0", "Planned")]
    assert layout.ABOUT_ROADMAP[1][2].startswith("Insight-driven visualisation, profitability and break-even")
    assert layout.ABOUT_ROADMAP[2][2].endswith("scheduled reports and alerts, team workspaces.")


def test_version_shown_in_sidebar_and_about_in_both_views(tmp_path, monkeypatch, output):
    from streamlit.testing.v1 import AppTest
    s = Settings(database_path=str(tmp_path / "app.db"))
    monkeypatch.setattr(dbc, "settings", s)
    monkeypatch.setattr(config_settings, "settings", s)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    at.session_state["output"] = output
    at.run()
    for view in ("Professional", "Basic"):
        at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).set_value(view).run()
        at.sidebar.radio(key="page" if view == "Professional" else basic.PAGE_KEY).set_value("About").run()
        assert not at.exception and not at.error, view
        main = " ".join(m.value for m in at.main.markdown)
        assert "Version Roadmap" in main and "mi-badge-current" in main, view
        assert "v2.0 · Decision Intelligence" in main and f"Marketing Intelligence Platform {SHOWN}" in main
        assert SHOWN in " ".join(m.value for m in at.sidebar.markdown), view
