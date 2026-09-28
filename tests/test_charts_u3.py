"""U3 visualisation standard: shared colours app/PDF, insight titles that match the AnalysisResult,
"Other" grouping, channel drill-down and incident markers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd
import plotly.graph_objects as go
import pytest

import database.connection as dbc
from analytics.channel import paid_channels
from analytics.kpis import KPI_REGISTRY, period_comparison
from config.settings import TREND_WINDOW_DAYS, Settings
from dashboard import chart_standard as cs
from dashboard import charts, theme
from reports import pdf_charts
from utils.formatting import format_value

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    from dashboard.pipeline import DEFAULT_SAMPLE, load_raw, prepare, run_pipeline, sample_path
    db = tmp_path_factory.mktemp("u3") / "t.db"
    raw, report = load_raw(sample_path(DEFAULT_SAMPLE))
    return run_pipeline(prepare(raw, report), db_path=db)


def _hex(color) -> str:
    return mcolors.to_hex(color).lower()


# --- Colour ------------------------------------------------------------------------------------

def test_channel_colours_are_fixed_and_identical_in_app_and_pdf(output, monkeypatch):
    paid = paid_channels(output.analysis.channels)
    names = list(output.analysis.channels["channel"])
    colours = theme.entity_colors(names)
    assert colours["Paid Search"] == theme.CHANNEL_COLORS["paid search"]
    assert len(set(colours.values())) == len(names)                 # every channel its own colour
    # Filtering never changes a channel's colour: the map comes from the full channel list.
    assert theme.entity_colors(names)["Video"] == colours["Video"]
    assert theme.entity_colors(["Google Ads"])["Google Ads"] == colours["Paid Search"]   # platform alias

    fig = charts.bar_chart(paid, "channel", "roas", "ratio", "t", color_by=colours)
    app = {c: col.lower() for c, col in zip(fig.data[0].customdata[:, 1], fig.data[0].marker.color)}

    captured = {}
    monkeypatch.setattr(pdf_charts, "_to_png", lambda f: captured.setdefault("fig", f))
    pdf_charts.hbar_png(paid, "channel", "roas", "ratio", "t", color_by=colours)
    ax = captured["fig"].axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    pdf = {label: _hex(bar.get_facecolor()) for label, bar in zip(labels, ax.patches)}
    assert app == pdf == {c: colours[c].lower() for c in paid["channel"]}


def test_share_chart_uses_context_grey_and_focus_colour_in_both(output, monkeypatch):
    ch = output.analysis.channels
    series = {"Share of Spend": "spend_share_pct", "Share of Revenue": "revenue_share_pct"}
    fig = charts.share_comparison_chart(ch, "channel", series, "t")
    assert [t.marker.color for t in fig.data] == [theme.CONTEXT, theme.PRIMARY]
    captured = {}
    monkeypatch.setattr(pdf_charts, "_to_png", lambda f: captured.setdefault("fig", f))
    pdf_charts.share_png(ch, "channel", series, "t")
    used = {_hex(p.get_facecolor()) for p in captured["fig"].axes[0].patches}
    assert used == {theme.CONTEXT.lower(), theme.PRIMARY.lower()}


# --- Insight titles ----------------------------------------------------------------------------

def test_share_title_numbers_match_the_analysis(output):
    ch = output.analysis.channels
    title = cs.share_title(ch, "channel", "revenue_share_pct", "Revenue", "neutral")
    row = ch.set_index("channel").loc["Paid Search"]
    assert title == (f"Paid Search Brings {format_value(row['revenue_share_pct'], 'percent')} of Revenue "
                     f"on {format_value(row['spend_share_pct'], 'percent')} of Spend")
    flat = ch.assign(revenue_share_pct=ch["spend_share_pct"] + 1)
    assert cs.share_title(flat, "channel", "revenue_share_pct", "Revenue", "neutral") == "neutral"


def test_ranking_title_numbers_match_the_analysis(output):
    a = output.analysis
    paid = paid_channels(a.channels)
    best = paid.sort_values("roas", ascending=False).iloc[0]
    title = cs.ranking_title(paid, "channel", "roas", "neutral", reference=a.paid_kpis["roas"].value,
                             reference_label="Paid Average")
    assert title == (f"{best['channel']} Has the Best ROAS: {format_value(best['roas'], 'ratio')} "
                     f"vs {format_value(a.paid_kpis['roas'].value, 'ratio')} Paid Average")
    cheapest = paid.sort_values("cpl").iloc[0]                   # lower CPL is better
    assert cs.ranking_title(paid, "channel", "cpl", "n").startswith(f"{cheapest['channel']} Leads on CPL")
    assert cs.ranking_title(paid.head(1), "channel", "roas", "neutral") == "neutral"


def test_trend_and_funnel_titles_match_the_analysis(output):
    rows = output.clean.clean_df
    end = rows["date"].max().normalize()
    deltas = period_comparison(rows, end - pd.Timedelta(days=TREND_WINDOW_DAYS - 1), end).deltas
    daily = output.analysis.trends["day"]
    checked = set()
    for metric in ("revenue", "spend", "leads", "conversions", "roas", "cpl"):
        change = deltas[metric].pct_change
        label = KPI_REGISTRY[metric].label
        title = cs.trend_title(rows, metric, "neutral")
        if abs(change) < 5:                                        # inside the flat band
            assert title == "neutral"
        else:
            assert title == (f"{label} {'Rose' if change > 0 else 'Fell'} "
                             f"{format_value(abs(change), 'percent')} in the Last {TREND_WINDOW_DAYS} Days")
            checked.add(metric)
        # The PDF uses the daily trend table of the AnalysisResult and gets the same title.
        assert cs.trend_title(daily, metric, "neutral", date_col="period") == title
    assert checked                                                 # at least one real takeaway tested

    f = output.analysis.funnel
    title = cs.funnel_title(f, "Marketing Funnel")
    leads = f.set_index("stage").loc["leads", "rate_from_previous"]
    assert title == f"Biggest Drop After the Click: Only {format_value(leads, 'percent')} of Clicks Become Leads"
    # Without a Clicks stage before the drop, the title stays generic.
    no_clicks = f[f["stage"] != "clicks"].assign(
        rate_from_previous=lambda d: d["rate_from_previous"].where(d["stage"] != "impressions"))
    assert cs.funnel_title(no_clicks, "Marketing Funnel").startswith("Biggest Drop: Only ")


def test_subtitle_names_metric_unit_and_period():
    text = cs.subtitle("revenue", "week", pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-30"))
    assert text == "Revenue (₹) · Weekly · 1 Oct 2025 – 30 Sep 2026"
    fig = charts.line_chart(pd.DataFrame({"p": pd.date_range("2026-01-05", periods=4, freq="W-MON"),
                                          "v": [1.0, 2, 3, 4]}), "p", "v", "count", "Title", subtitle=text)
    assert fig.layout.title.subtitle.text == text


# --- "Other" grouping --------------------------------------------------------------------------

def test_more_than_eight_categories_fold_into_other():
    df = pd.DataFrame({"segment": [f"S{i}" for i in range(12)], "spend": [float(100 - i) for i in range(12)],
                       "roas": [2.0] * 12})
    out = cs.group_other(df, "segment", ["spend"], sort_by="spend")
    assert len(out) == theme.MAX_CATEGORIES
    other = out.iloc[-1]
    assert other["segment"] == theme.OTHER_LABEL
    assert other["spend"] == df["spend"].iloc[7:].sum()            # additive values are summed
    assert pd.isna(other["roas"])                                  # ratios are never summed
    assert out["spend"].sum() == df["spend"].sum()
    assert cs.group_other(df.head(8), "segment", ["spend"], "spend") is not None
    assert len(cs.group_other(df.head(8), "segment", ["spend"], "spend")) == 8
    fig = charts.bar_chart(out, "segment", "spend", "money", "t")
    assert theme.CONTEXT in list(fig.data[0].marker.color)


# --- Interactivity -----------------------------------------------------------------------------

def test_incident_markers_render_on_trend_lines(output):
    from analytics.trends import time_series
    trend = time_series(output.clean.clean_df, "week")
    fig = charts.line_chart(trend, "period", "revenue", "money", "t")
    incidents = output.analysis.incidents
    charts.add_incident_markers(fig, incidents)
    marker = [t for t in fig.data if t.name == "Incidents"]
    assert len(marker) == 1 and len(marker[0].x) == min(len(incidents), charts.MAX_INCIDENT_MARKERS)
    assert all("₹" in text or "Not estimated" in text for text in marker[0].customdata)


def test_single_point_trend_says_not_enough_data():
    fig = charts.line_chart(pd.DataFrame({"p": [pd.Timestamp("2026-01-05")], "v": [5.0]}), "p", "v",
                            "count", "t")
    assert any("Not enough data" in a.text for a in fig.layout.annotations)


def test_funnel_chart_labels_every_drop_off(output):
    fig = charts.funnel_chart(output.analysis.funnel, "t")
    texts = list(fig.data[0].text)
    assert len(texts) == 4 and all("move on" in t for t in texts)
    assert theme.BAD in list(fig.data[0].marker.color)             # the biggest drop stands out


@pytest.fixture
def app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)


def test_channel_drill_down_and_trend_incidents_render(app):
    at = app.run()
    at.button(key="load_sample").click().run()
    if any(b.key == "run_analysis" for b in at.button):
        at.button(key="run_analysis").click().run()
    at.sidebar.radio(key="page").set_value("Channels").run()
    assert not at.exception
    at.session_state["channel_drill"] = "Paid Search"
    at.run()
    assert not at.exception
    assert any(b.label == "Back to All Channels" for b in at.button)
    text = " ".join(str(getattr(e, "value", "")) for e in at.main)
    assert "Campaigns in Paid Search" in text
    at.button(key="channel_back").click().run()
    assert not any(b.label == "Back to All Channels" for b in at.button)

    at.sidebar.radio(key="page").set_value("Performance Trends").run()
    assert not at.exception
    assert at.toggle(key="trend_incidents").value is True
    at.toggle(key="trend_incidents").set_value(False).run()
    assert not at.exception


# --- Owner review fixes (2026-09-29) -----------------------------------------------------------

def test_partial_edge_weeks_leave_the_line_but_not_the_totals(output):
    from analytics.trends import time_series
    trend = time_series(output.clean.clean_df, "week")
    shown, dropped = cs.full_weeks(trend)
    edges = int(trend["days"].iloc[0] < 7) + int(trend["days"].iloc[-1] < 7)
    assert dropped == (edges > 0) and len(shown) == len(trend) - edges
    assert (shown["days"] == 7).iloc[[0, -1]].all()
    assert trend["revenue"].sum() == output.analysis.kpis["revenue"].value        # table itself unchanged
    # A table with fewer than two full weeks is left as it is.
    tiny = pd.DataFrame({"period": pd.date_range("2026-01-05", periods=2, freq="W-MON"), "days": [3, 7]})
    assert cs.full_weeks(tiny)[1] is False


def test_owned_channels_are_labelled_and_lighter_in_both(output, monkeypatch):
    ch = output.analysis.channels
    owned = cs.owned_names(ch["channel"])
    assert owned == {"Email"}
    series = {"Share of Spend": "spend_share_pct", "Share of Revenue": "revenue_share_pct"}
    fig = charts.share_comparison_chart(ch, "channel", series, "t", owned=owned)
    assert "Email (owned)" in list(fig.data[0].y)
    i = list(fig.data[0].y).index("Email (owned)")
    assert fig.data[0].marker.pattern.shape[i] == "/" and fig.data[0].marker.color[i].startswith("rgba")
    assert fig.data[0].marker.pattern.shape[0] == ""        # legend swatch comes from a paid bar
    captured = {}
    monkeypatch.setattr(pdf_charts, "_to_png", lambda f: captured.setdefault("fig", f))
    pdf_charts.share_png(ch, "channel", series, "t", owned=owned)
    ax = captured["fig"].axes[0]
    assert "Email (owned)" in [t.get_text() for t in ax.get_yticklabels()]
    assert any(p.get_hatch() for p in ax.patches)


def test_value_labels_move_clear_of_the_average_line(output, monkeypatch):
    paid = paid_channels(output.analysis.channels)
    avg = output.analysis.paid_kpis["roas"].value
    fig = charts.bar_chart(paid, "channel", "roas", "ratio", "t", reference=avg)
    moved = [a for a in fig.layout.annotations if a.x == avg and a.xanchor == "left"]
    assert moved and all(t == "" for t, v in zip(fig.data[0].text, fig.data[0].x)
                         if cs.label_crosses_line(v, avg, max(max(fig.data[0].x), avg), charts.LABEL_ROOM))
    shown = [t for t in fig.data[0].text if t] + [a.text for a in moved]
    assert sorted(shown) == sorted(format_value(v, "ratio", compact=True) for v in paid["roas"])
    # Line chart: the latest-value label sits on the far side of the reference line.
    df = pd.DataFrame({"p": pd.date_range("2026-01-05", periods=3, freq="W-MON"), "v": [5.0, 4.0, 3.0]})
    fig = charts.line_chart(df, "p", "v", "ratio", "t", reference=4.5)
    assert fig.data[1].textposition == "bottom left"
