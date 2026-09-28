"""Phase 08 tests: chart/table builders for every dataset variant, filters, and app smoke tests."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import database.connection as dbc
from analytics.trends import time_series
from config.settings import Settings
from dashboard import charts
from dashboard.components import column_label, format_table
from dashboard.filters import Filters, apply_filters, available_filters, describe, preset_range
from dashboard.pipeline import SAMPLE_DATASETS, load_raw, prepare, run_pipeline, sample_path
from dashboard.theme import CATEGORICAL, color_map

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    """Run the whole pipeline once for every runnable sample (the Meta export needs a choice)."""
    db = tmp_path_factory.mktemp("db") / "t.db"
    results = {}
    for label in SAMPLE_DATASETS:
        raw, report = load_raw(sample_path(label))
        overrides = {"Results": "leads"} if "Meta" in label else None
        prep = prepare(raw, report, overrides)
        results[label] = run_pipeline(prep, db_path=db)
    return results


# --- Chart builders ----------------------------------------------------------------------------

def test_charts_build_for_every_dataset(outputs):
    for label, out in outputs.items():
        a = out.analysis
        df = out.clean.clean_df
        for grain in ("day", "week", "month"):
            trend = time_series(df, grain)
            for metric, fmt in (("spend", "money"), ("leads", "count"), ("revenue", "money"), ("roas", "ratio")):
                fig = charts.line_chart(trend, "period", metric, fmt, "t",
                                        x_label_fmt="text" if grain == "month" else "date")
                assert fig is None or isinstance(fig, go.Figure), label
                if metric in ("spend", "leads"):
                    assert fig is not None, (label, metric)
        if not a.channels.empty:
            assert isinstance(charts.bar_chart(a.channels, "channel", "cpl", "money", "t", reference=100), go.Figure)
            assert isinstance(charts.index_chart(a.channels, "channel", "cpl_index", "t", True), go.Figure)
            assert isinstance(charts.share_comparison_chart(a.channels, "channel",
                              {"Spend": "spend_share_pct", "Leads": "leads_share_pct"}, "t"), go.Figure)
        assert isinstance(charts.funnel_chart(a.funnel, "t"), go.Figure), label


def test_charts_return_none_when_data_is_missing(outputs):
    no_rev = outputs["Lead generation only - no revenue (CSV)"].analysis
    assert "revenue" not in no_rev.channels
    assert charts.bar_chart(no_rev.channels, "channel", "roas", "ratio", "t") is None
    assert charts.line_chart(time_series(outputs["Lead generation only - no revenue (CSV)"].clean.clean_df, "week"),
                             "period", "revenue", "money", "t") is None
    empty = pd.DataFrame()
    assert charts.bar_chart(empty, "channel", "cpl", "money", "t") is None
    assert charts.funnel_chart(empty, "t") is None
    assert charts.funnel_chart(pd.DataFrame({"stage": ["clicks"], "value": [5], "stage_order": [1],
                                             "rate_from_previous": [None], "label": ["Clicks"]}), "t") is None
    assert charts.anomaly_chart(empty, "spend", "money", "2026-01-01", "2026-01-07", 10, "t") is None


def test_chart_axes_use_indian_formatting(outputs):
    a = outputs["Kalpa Learning - clean sample (CSV)"].analysis
    fig = charts.bar_chart(a.channels, "channel", "spend", "money", "Spend")
    ticks = fig.layout.xaxis.ticktext
    assert any("Cr" in t or "L" in t for t in ticks)
    labels = fig.data[0].text
    assert all(t.startswith("₹") for t in labels)


def test_anomaly_chart(outputs):
    out = outputs["Kalpa Learning - clean sample (CSV)"]
    row = out.analysis.anomalies.iloc[0]
    from dashboard.layout import _entity_weekly
    weekly = _entity_weekly(out.clean.clean_df, row["entity_type"], row["entity"])
    fig = charts.anomaly_chart(weekly, row["metric"], "count", row["period_start"], row["period_end"],
                               row["baseline"], "t")
    assert isinstance(fig, go.Figure)
    assert any(s.type == "rect" for s in fig.layout.shapes)


def test_colour_follows_entity_not_rank():
    all_channels = ["Paid Search", "Email", "Video", "Affiliate"]
    full = color_map(all_channels)
    assert full["Affiliate"] == CATEGORICAL[0]
    # The same channel keeps its colour; the mapping is built from ALL entities, not the filter.
    assert color_map(all_channels)["Video"] == full["Video"]
    assert len(set(full.values())) == 4


# --- Tables and filters ------------------------------------------------------------------------

def test_format_table(outputs):
    a = outputs["Kalpa Learning - clean sample (CSV)"].analysis
    shown = format_table(a.channels, ["channel", "spend", "spend_share_pct", "cpl", "roas", "ctr", "cpl_index"])
    assert list(shown.columns) == ["Channel", "Spend", "Spend Share", "CPL", "ROAS", "CTR", "CPL Index"]
    assert shown["Spend"].str.startswith("₹").all()
    assert shown["ROAS"].str.endswith("x").all() and shown["CTR"].str.endswith("%").all()
    nan_table = format_table(pd.DataFrame({"roas": [None, 2.0]}))
    assert nan_table["ROAS"].tolist() == ["N/A", "2.00x"]
    assert column_label("lead_to_conversion_rate") == "Lead-to-Conversion Rate"


def test_filters(outputs):
    df = outputs["Kalpa Learning - clean sample (CSV)"].clean.clean_df
    cols = [c for c, _ in available_filters(df)]
    assert {"channel", "campaign_name", "region", "product_category", "customer_segment"} <= set(cols)
    start, end = preset_range("Last 30 days", df["date"].min(), df["date"].max())
    assert (end - start).days == 29
    f = Filters(start=start, end=end, dimensions={"channel": ["Email"]},
                data_start=df["date"].min(), data_end=df["date"].max())
    sub = apply_filters(df, f)
    assert set(sub["channel"]) == {"Email"} and sub["date"].min() >= start
    assert f.active and "Channel: Email" in describe(f)
    assert describe(Filters()) == "Showing all data"
    no_rev = outputs["Lead generation only - no revenue (CSV)"].clean.clean_df
    assert "revenue" not in no_rev


# --- Pipeline ----------------------------------------------------------------------------------

def test_pipeline_saves_and_reports_reupload(outputs, tmp_path):
    for out in outputs.values():
        assert out.save_error is None and out.run_id is not None
    raw, report = load_raw(sample_path("Kalpa Learning - clean sample (CSV)"))
    prep = prepare(raw, report)
    first = run_pipeline(prep, db_path=tmp_path / "x.db")
    again = run_pipeline(prep, db_path=tmp_path / "x.db")
    assert not first.is_reupload and again.is_reupload and again.run_id != first.run_id


def test_pipeline_survives_database_failure(tmp_path):
    raw, report = load_raw(sample_path("Kalpa Learning - clean sample (CSV)"))
    out = run_pipeline(prepare(raw, report), db_path=tmp_path)     # a folder: cannot be opened
    assert out.run_id is None and "could not be opened" in out.save_error
    assert out.analysis.kpis["spend"].available                     # analysis still shown


# --- App smoke tests (streamlit.testing) -------------------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)


PAGES = ["Executive Overview", "Performance Trends", "Channels", "Campaigns", "Funnel", "Segments",
         "Anomalies", "AI Insights", "Data Quality", "Reports"]


def _visit_all_pages(at):
    for page in PAGES:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])


def test_app_sample_dataset_end_to_end(app):
    at = app.run()
    assert not at.exception
    assert "Marketing Intelligence Platform" in at.sidebar.markdown[0].value
    at.button(key="load_sample").click().run()
    assert [m.value for m in at.metric][1] == "8,471"
    at.button(key="run_analysis").click().run()
    assert not at.exception
    assert any("Analysis ready" in s.value for s in at.success)
    _visit_all_pages(at)
    at.sidebar.radio(key="date_preset").set_value("Last 90 days").run()
    at.sidebar.radio(key="page").set_value("Executive Overview").run()
    cards = {m.label: m for m in at.metric}
    assert {"Revenue", "Spend", "Leads", "Conversions", "ROAS", "CPL", "CAC"} <= set(cards)
    assert cards["Spend"].value.startswith("₹") and cards["Spend"].delta
    assert not at.exception


def test_app_messy_excel_end_to_end(app):
    at = app.run()
    at.selectbox(key="sample_choice").set_value("Kalpa Learning - messy sample (Excel)").run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    assert not at.exception and any("Analysis ready" in s.value for s in at.success)
    at.sidebar.radio(key="page").set_value("Data Quality").run()
    assert {m.label: m.value for m in at.metric}["Duplicates Removed"] == "25"
    _visit_all_pages(at)


def test_app_requires_confirmation_for_ambiguous_column(app):
    at = app.run()
    at.selectbox(key="sample_choice").set_value("Meta Ads export - needs a mapping decision (CSV)").run()
    at.button(key="load_sample").click().run()
    assert at.button(key="run_analysis").disabled
    assert any("Results" in w.value for w in at.warning)
    at.selectbox(key="confirm_Results").set_value("leads").run()
    at.button(key="apply_mapping").click().run()
    assert not at.button(key="run_analysis").disabled
    at.button(key="run_analysis").click().run()
    assert not at.exception
    _visit_all_pages(at)


def test_app_no_revenue_hides_revenue_kpis(app):
    at = app.run()
    at.selectbox(key="sample_choice").set_value("Lead generation only - no revenue (CSV)").run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    at.sidebar.radio(key="page").set_value("Executive Overview").run()
    labels = {m.label for m in at.metric}
    assert "Revenue" not in labels and "ROAS" not in labels
    assert {"Spend", "Leads", "CPL"} <= labels
    _visit_all_pages(at)


def test_pages_before_analysis_show_empty_state(app):
    at = app.run()
    at.sidebar.radio(key="page").set_value("Executive Overview").run()
    assert not at.exception
    assert any("No analysis is open yet" in m.value for m in at.markdown)


# --- About page and sidebar footer (static: no data, no database, no AI settings) ---------------

def _about_ok(at):
    at.sidebar.radio(key="page").set_value("About").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    text = " ".join(m.value for m in at.markdown)
    assert "What This Platform Does" in text and "Mayur Jaiswal" in text
    assert "Created by" in " ".join(m.value for m in at.sidebar.markdown)


def test_about_renders_with_no_data(app):
    at = app.run()
    _about_ok(at)


def test_about_renders_with_data_loaded(app):
    at = app.run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    _about_ok(at)


def test_about_and_footer_survive_missing_settings(app, monkeypatch):
    import config.settings as cfg
    for name in ("APP_NAME", "APP_VERSION", "CREATOR_NAME", "CREATOR_LINKEDIN", "CREATOR_TITLE",
                 "CREATOR_EDUCATION_PREVIOUS", "CREATOR_FOCUS", "CREATOR_OPEN_TO"):
        monkeypatch.delattr(cfg, name)
    at = app.run()
    at.sidebar.radio(key="page").set_value("About").run()
    assert not at.exception and not at.error
    assert any("What This Platform Does" in m.value for m in at.markdown)


def test_every_menu_page_renders_after_loading_sample(app):
    from dashboard.layout import PAGES as MENU
    at = app.run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    for page in MENU:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not any("Something went wrong" in e.value for e in at.error), page
