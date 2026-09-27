"""Phase 09 tests: the executive PDF is generated, consistent with the dashboard, and adaptive."""

import re
from pathlib import Path

import pytest
from pypdf import PdfReader

import database.connection as dbc
import reports.pdf_report as pdf_report
from config.settings import Settings
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from database import repository as repo
from reports.pdf_report import AI_NOT_GENERATED, export_pdf, generate_pdf

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "clean": ("marketing_clean.csv", None),
    "messy": ("marketing_messy.xlsx", None),
    "no_revenue": ("marketing_no_revenue.csv", None),
    "no_margin": ("marketing_no_margin.csv", None),
    "meta": ("meta_ads_export_style.csv", {"Results": "leads"}),
}


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """Analyse each dataset and generate its PDF once; return (result, text, pages)."""
    folder = tmp_path_factory.mktemp("pdf")
    out = {}
    for name, (file, overrides) in DATASETS.items():
        raw, report = load_raw(SAMPLE_DIR / file)
        output = run_pipeline(prepare(raw, report, overrides), db_path=folder / "t.db")
        path = folder / f"{name}.pdf"
        generate_pdf(output.analysis, path)
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        out[name] = (output.analysis, text, len(reader.pages))
    return out


def squash(text: str) -> str:
    """Remove line breaks and spaces so values split across lines still match."""
    return re.sub(r"\s+", "", text)


def test_pdf_generates_for_every_dataset(reports):
    for name, (_, text, pages) in reports.items():
        assert pages >= 5, name
        assert "Marketing Intelligence Report" in text
        assert "Executive summary" in text and "Methodology" in text and "Limitations" in text


def test_rupee_sign_renders(reports):
    _, text, _ = reports["clean"]
    assert "₹" in text


def test_kpi_values_match_dashboard(reports):
    result, text, _ = reports["clean"]
    flat = squash(text)
    for k in result.kpis.values():
        if k.available:
            assert squash(k.formatted) in flat, (k.key, k.formatted)       # KPI overview table
    for key in ("revenue", "spend", "roas", "leads", "cpl", "conversions", "cac"):
        assert squash(result.kpis[key].formatted_compact) in flat, key     # same as dashboard cards


def test_findings_and_anomalies_included(reports):
    result, text, _ = reports["clean"]
    flat = squash(text)
    for f in result.findings:
        assert squash(f.text) in flat, f.id
    assert "Performance concerns" in text
    assert "trackingoutage" in flat.lower()


def test_content_labels_and_no_invented_ai(reports):
    _, text, _ = reports["clean"]
    assert AI_NOT_GENERATED in text
    assert "VERIFIED METRIC" in text and "ANALYTICAL FINDING" in text
    # AI labels appear only in the cover's "how to read" legend, never on content.
    assert text.count("AI INTERPRETATION") == 1
    assert text.count("RECOMMENDATION") == 1 and text.count("HYPOTHESIS") == 1


def test_sections_absent_without_data(reports):
    no_rev = reports["no_revenue"][1]
    assert "Revenue by month" not in no_rev
    assert "no revenue field was detected" in no_rev
    # Chart titles are inside the images, so check the text captions under the charts.
    assert "overall ROAS" not in no_rev and "overall CPL" in no_rev

    no_margin = reports["no_margin"][1]
    assert "ROI is hidden because it needs gross profit or margin data" in no_margin
    assert "overall ROAS" in no_margin

    # Section headings are numbered ("5. Funnel analysis"); the Limitations section may still
    # EXPLAIN why an analysis is missing, which is intended.
    meta = reports["meta"][1]
    headings = {re.sub(r"^\d+\. ", "", line) for line in meta.splitlines() if re.match(r"^\d+\. ", line)}
    assert "Channel analysis" not in headings
    assert "Segment, geography and product analysis" not in headings
    assert "Funnel analysis" in headings                    # impressions -> clicks -> leads exist
    assert "Channel analysis is unavailable" in meta


def test_messy_report_mentions_quality_work(reports):
    result, text, _ = reports["messy"]
    flat = squash(text)
    assert "Duplicaterowsremoved" in flat and "25" in text
    assert "80rowshavenousablerevenue" in flat


def test_export_saves_file_and_records_it(tmp_path):
    db = tmp_path / "e.db"
    raw, report = load_raw(SAMPLE_DIR / "test_c_large.csv")
    output = run_pipeline(prepare(raw, report), db_path=db)
    path = export_pdf(output.analysis, exports_dir=tmp_path / "exports", db_path=db)
    assert path.exists() and path.parent.name == str(output.run_id)
    assert path.name.startswith("Marketing_Intelligence_Report_") and path.suffix == ".pdf"
    recorded = repo.list_reports(output.run_id, db_path=db)
    assert recorded["report_type"].tolist() == ["pdf"] and recorded["file_path"].iloc[0] == str(path)


def test_ai_content_is_labelled_when_provided(tmp_path):
    raw, report = load_raw(SAMPLE_DIR / "test_b_medium.csv")
    output = run_pipeline(prepare(raw, report), db_path=tmp_path / "a.db")
    ai = {"interpretations": [{"text": "Spend shifted towards search."}],
          "recommendations": [{"text": "Review the weakest campaign's landing page."}]}
    path = tmp_path / "ai.pdf"
    generate_pdf(output.analysis, path, ai=ai)
    text = "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)
    assert AI_NOT_GENERATED not in text
    assert "Spend shifted towards search." in text and text.count("RECOMMENDATION") == 2


def test_reports_page_generates_pdf(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    monkeypatch.setattr(pdf_report, "EXPORTS_DIR", tmp_path / "exports")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    at.sidebar.radio(key="page").set_value("Reports").run()
    at.button(key="generate_pdf").click().run()
    assert not at.exception and not at.error
    assert any("Report ready" in s.value for s in at.success)
    assert list((tmp_path / "exports").rglob("*.pdf"))
