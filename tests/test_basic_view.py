"""U8 tests: Basic and Professional views (plain-language sentences, Basic pages, view switch,
Summary Report PDF). No real AI calls: the AI client factory is replaced and must never be used."""

import io
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

import ai.client
import ai.demo_seed
import config.settings as config_settings
import dashboard.demo
import database.connection as dbc
from analytics.targets import targets_analysis
from config.settings import Settings, load_settings
from dashboard import layout
from dashboard.pipeline import load_raw, prepare, run_pipeline
from dashboard.views import basic
from reports.summary_report import generate_summary_pdf
from utils import plain_language as pl

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample"
SEED = ROOT / "data" / "demo" / "kalpa_demo_seed.json"


@pytest.fixture(scope="module")
def kalpa(tmp_path_factory):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    return run_pipeline(prepare(raw, report), db_path=tmp_path_factory.mktemp("db") / "k.db")


@pytest.fixture(scope="module")
def insights():
    return json.loads(SEED.read_text(encoding="utf-8"))["ai"]["insights"]


def _all_basic_text(output, insights) -> list[str]:
    a, df = output.analysis, output.clean.clean_df
    best, weakest, metric = pl.best_and_weakest(a)
    summary, recs = pl.ai_summary(insights)
    lines = [f"{k.title} {k.value} {k.sentence} {k.status_text} {k.basis}"
             for k in pl.key_numbers(a, df, pacing=a.pacing)]
    lines += pl.overview_lines(a, df) + pl.profit_lines(a.profitability) + pl.pacing_lines(a.pacing)
    lines += pl.money_lines(a) + [pl.campaign_line(r, metric) for _, r in best.iterrows()]
    lines += [pl.campaign_line(r, metric) for _, r in weakest.iterrows()]
    lines += [f"{x['title']} {x['what']} {x['impact']}" for x in pl.top_alerts(a)] + summary + recs
    lines += [f"{t}: {d}" for t, d in pl.GLOSSARY] + list(basic.BASIC_DESCRIPTIONS.values())
    return lines


# --- Plain-language sentences --------------------------------------------------------------------

def test_key_numbers_equal_the_analysis_result(kalpa):
    a, df = kalpa.analysis, kalpa.clean.clean_df
    numbers = {k.key: k for k in pl.key_numbers(a, df, pacing=a.pacing)}
    assert list(numbers) == ["spend", "revenue", "roas", "cpl", "cac"]
    for key in ("spend", "revenue", "cpl", "cac"):
        assert numbers[key].value == a.kpis[key].formatted_compact
        assert a.kpis[key].formatted_compact in numbers[key].sentence or a.kpis[key].formatted in numbers[key].sentence
    assert numbers["roas"].value == f"₹{a.kpis['roas'].value:.2f}" == "₹4.57"
    assert numbers["roas"].sentence == "For every ₹1 spent, you got ₹4.57 back."
    assert numbers["cpl"].sentence.endswith(f"cost {a.kpis['cpl'].formatted} on average.")
    assert all(k.status in ("good", "watch", "bad", "neutral") for k in numbers.values())
    assert numbers["spend"].basis.startswith("This month's spending")          # budget pacing decides spend


def test_colours_follow_targets_then_trend(kalpa):
    a, df = kalpa.analysis, kalpa.clean.clean_df
    cpl = a.kpis["cpl"].value
    loose = {k.key: k for k in pl.key_numbers(a, df, targets={"cpl": cpl * 2})}
    tight = {k.key: k for k in pl.key_numbers(a, df, targets={"cpl": cpl / 2})}
    assert loose["cpl"].status == "good" and "target" in loose["cpl"].basis
    assert tight["cpl"].status == "bad"
    trend = {k.key: k for k in pl.key_numbers(a, df)}
    assert "previous 30 days" in trend["cpl"].basis


def test_no_unexplained_abbreviations_in_basic_text(kalpa, insights):
    for line in _all_basic_text(kalpa, insights):
        assert pl.unexplained_terms(line) == [], line
        assert "E0" not in line and " I0" not in line and " I1" not in line, line      # no evidence IDs


def test_ai_text_is_made_plain():
    assert pl.plain_ai_text("Shift budget (ROAS 1.55x) to Affiliate (E05, E10).") == \
        "Shift budget (₹1.55 back per ₹1 spent) to Affiliate."
    assert pl.plain_ai_text("losses similar to the ₹1.6 L impact from incident I09") == \
        "losses similar to the ₹1.6 L impact from an earlier incident"
    assert pl.plain_ai_text("CPL and CAC rose") == "cost per lead and cost per customer rose"
    assert pl.unexplained_terms("ROAS (return on ad spend) fell") == []
    assert pl.unexplained_terms("ROAS fell") == ["ROAS"]


def test_alerts_are_the_top_problem_incidents(kalpa):
    a = kalpa.analysis
    alerts = pl.top_alerts(a)
    problems = a.incidents[a.incidents["sentiment"] == "negative"].sort_values("rank").head(3)
    assert [x["impact_inr"] for x in alerts] == list(problems["impact_inr"])
    assert alerts[0]["title"] == "East (region)" and alerts[0]["impact"] == "Estimated impact: about ₹11.8 L lost."


def test_best_and_weakest_are_paid_and_big_enough(kalpa):
    best, weakest, metric = pl.best_and_weakest(kalpa.analysis)
    assert metric == "roas" and len(best) == len(weakest) == 3
    assert best["roas"].min() > weakest["roas"].max()
    assert "Email" not in set(best["channel"]) | set(weakest["channel"])
    assert (best["spend_share_pct"] >= 1.0).all() and (weakest["spend_share_pct"] >= 1.0).all()


def test_default_view_setting():
    assert Settings().default_view == "Professional"
    assert load_settings({"DEFAULT_VIEW": "basic"}).default_view == "Basic"
    assert load_settings({"DEFAULT_VIEW": "expert"}).default_view == "Professional"


# --- Summary Report PDF --------------------------------------------------------------------------

def test_summary_pdf_is_short_plain_and_signed(kalpa, insights):
    a = kalpa.analysis
    buffer = io.BytesIO()
    targets = targets_analysis(kalpa.clean.clean_df, {"cpl": 250}, a.capabilities, a.profitability, kpis=a.kpis)
    generate_summary_pdf(a, buffer, df=kalpa.clean.clean_df, ai=insights, targets=targets)
    pages = PdfReader(io.BytesIO(buffer.getvalue())).pages
    text = " ".join(p.extract_text() for p in pages)
    assert 2 <= len(pages) <= 3, len(pages)
    assert config_settings.REPORT_SIGNATURE in pages[0].extract_text()
    for piece in ("Marketing Summary Report", "How Are We Doing?", "₹4.57", a.kpis["spend"].formatted_compact,
                  "Where the Money Goes", "Top Alerts", "AI Summary", "targets are on track"):
        assert piece in text, piece


# --- The app --------------------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    """The app with a temporary database, AI switched on with a fake key, and a client factory that
    fails the test if it is ever used (Basic must never call the AI)."""
    from streamlit.testing.v1 import AppTest
    s = Settings(database_path=str(tmp_path / "app.db"), ai_enabled=True, gemini_api_key="test-key",
                 gemini_model="test-model")
    monkeypatch.setattr(dbc, "settings", s)
    monkeypatch.setattr(config_settings, "settings", s)
    monkeypatch.setattr(ai.client, "client_from_settings",
                        lambda *_a, **_k: pytest.fail("the AI client was requested"))
    monkeypatch.setattr(ai.demo_seed, "SEED_PATH", SEED)
    monkeypatch.setattr(dashboard.demo, "DEMO_SNAPSHOT_PATH", ROOT / "data" / "demo" / "kalpa_demo_snapshot.pkl.gz")
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)


def _main_text(at) -> str:
    parts = [m.value for m in at.main.markdown if "<style" not in m.value]        # CSS is not reader text
    parts += [c.value for c in at.main.caption]
    parts += [i.value for i in at.main.info] + [b.label for b in at.main.button] + [e.label for e in at.main.expander]
    return " ".join(parts)


def test_toggle_appears_only_with_an_analysis_and_every_page_renders_in_both_views(app, kalpa):
    at = app.run()
    assert not [r for r in at.sidebar.radio if r.key == basic.VIEW_WIDGET_KEY]         # no analysis yet
    at.session_state["output"] = kalpa
    at.run()
    assert at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).value == "Professional"      # DEFAULT_VIEW
    for page in layout.PAGES:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
    at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).set_value("Basic").run()
    for page in basic.BASIC_PAGES:
        at.sidebar.radio(key=basic.PAGE_KEY).set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not any("Something went wrong" in e.value for e in at.error), page
        if page in basic.RENDERERS:
            assert pl.unexplained_terms(_main_text(at)) == [], page
            assert not at.main.dataframe, page                                         # no dense tables


def test_basic_numbers_and_switching_never_rerun_or_call_ai(app, kalpa, monkeypatch):
    calls = []
    monkeypatch.setattr(layout, "run_pipeline", lambda *a, **k: calls.append("run") or kalpa)
    at = app.run()
    at.session_state["output"] = kalpa
    at.run()
    at.sidebar.radio(key="page").set_value("Channels").run()
    at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).set_value("Basic").run()
    assert at.sidebar.radio(key=basic.PAGE_KEY).value == "Where the Money Goes"      # the matching page
    at.sidebar.radio(key=basic.PAGE_KEY).set_value("How Are We Doing?").run()
    text = _main_text(at)
    for k in pl.key_numbers(kalpa.analysis, kalpa.clean.clean_df, pacing=kalpa.analysis.pacing):
        assert k.value in text and k.sentence in text
    at.sidebar.radio(key=basic.PAGE_KEY).set_value("AI Summary").run()
    at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).set_value("Professional").run()
    assert at.session_state["page"] == "AI Insights"
    at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).set_value("Basic").run()
    assert not at.exception and calls == [] and at.session_state["output"] is kalpa


def test_see_full_details_opens_the_professional_page(app, kalpa):
    at = app.run()
    at.session_state["output"] = kalpa
    at.session_state[basic.VIEW_KEY] = "Basic"
    at.run()
    at.sidebar.radio(key=basic.PAGE_KEY).set_value("Top Alerts").run()
    at.button(key="basic_full_Anomalies").click().run()
    assert not at.exception
    assert at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).value == "Professional"
    assert at.sidebar.radio(key="page").value == "Anomalies"


def test_ai_summary_shows_saved_insights_without_ids(app):
    at = app.run()
    at.button(key="load_sample").click().run()
    at.run()                                      # the sidebar shows the toggle from the next run on
    at.sidebar.radio(key=basic.VIEW_WIDGET_KEY).set_value("Basic").run()
    at.sidebar.radio(key=basic.PAGE_KEY).set_value("AI Summary").run()
    assert not at.exception
    text = _main_text(at)
    assert "Top 3 Recommendations" in text and "₹1.55 back per ₹1 spent" in text
    assert "E05" not in text and "I09" not in text and pl.unexplained_terms(text) == []
    assert any(b.label == "See Evidence" for b in at.main.button)


def test_default_basic_view_and_session_only_choice_in_public_mode(app, kalpa, monkeypatch):
    s = Settings(database_path=config_settings.settings.database_path, public_mode=True, default_view="Basic")
    monkeypatch.setattr(config_settings, "settings", s)
    monkeypatch.setattr(dbc, "settings", s)
    first = app.run()
    first.session_state["output"] = kalpa
    first.run()
    assert first.sidebar.radio(key=basic.VIEW_WIDGET_KEY).value == "Basic"          # DEFAULT_VIEW=Basic
    assert first.sidebar.radio(key=basic.PAGE_KEY).value == "Upload & Profile"
    first.sidebar.radio(key=basic.VIEW_WIDGET_KEY).set_value("Professional").run()
    assert not first.exception
    from streamlit.testing.v1 import AppTest
    second = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()   # another visitor
    second.session_state["output"] = kalpa
    second.run()
    assert second.sidebar.radio(key=basic.VIEW_WIDGET_KEY).value == "Basic"         # not shared or saved


def test_summary_report_downloads_from_basic_reports(app, kalpa):
    at = app.run()
    at.session_state["output"] = kalpa
    at.session_state[basic.VIEW_KEY] = "Basic"
    at.run()
    at.sidebar.radio(key=basic.PAGE_KEY).set_value("Reports").run()
    at.button(key="generate_summary").click().run()
    assert not at.exception
    assert any("summary report is ready" in s.value for s in at.success)
    assert any(b.label == "Generate PDF" for b in at.main.button)                      # full reports still there
