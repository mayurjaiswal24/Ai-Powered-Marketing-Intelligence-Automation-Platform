"""Basic view (U8): the same analysis in plain words for non-specialists.

Pages: How Are We Doing? · Where the Money Goes · What's Working and What's Not · Top Alerts ·
AI Summary · Reports (plus Upload & Profile and About, the same in both views). Every number comes
from the AnalysisResult already in the session: switching views never re-runs the analysis and never
calls the AI. Sentences come from utils/plain_language.py and charts from the shared builders.
Basic pages cover the full dataset (no sidebar filters). "See Full Details" opens the matching
Professional page.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

import config.settings as config_settings
from analytics.kpis import KPI_REGISTRY
from dashboard import chart_standard as cs
from dashboard import charts, components as ui, layout, theme
from utils import plain_language as pl
from utils.formatting import title_case

BASIC = "Basic"
PROFESSIONAL = "Professional"
VIEW_KEY = "view_choice"            # the chosen view (session only, never saved)
VIEW_WIDGET_KEY = "view_toggle"
PAGE_KEY = "page_basic"             # the Basic menu (the Professional menu uses "page")

BASIC_SECTIONS = [
    ("Data", ["Upload & Profile"]),
    ("Results", ["How Are We Doing?", "Where the Money Goes", "What's Working and What's Not", "Top Alerts"]),
    ("AI", ["AI Summary"]),
    ("Output", ["Reports"]),
    ("", ["About"]),
]
BASIC_PAGES = [page for _, pages in BASIC_SECTIONS for page in pages]
BASIC_SECTION = {page: section for section, pages in BASIC_SECTIONS for page in pages}
BASIC_ICONS = {"How Are We Doing?": "dashboard", "Where the Money Goes": "payments",
               "What's Working and What's Not": "thumbs_up_down", "Top Alerts": "notification_important",
               "AI Summary": "auto_awesome"}
BASIC_DESCRIPTIONS = {
    "How Are We Doing?": "The headline numbers in plain words, and whether each one looks good.",
    "Where the Money Goes": "Which channels get the money and what each one brings back.",
    "What's Working and What's Not": "The campaigns that bring back the most and the least for the money.",
    "Top Alerts": "The biggest problems found in the data, with what each one is estimated to have cost.",
    "AI Summary": "Google Gemini's saved summary and its top recommendations, in plain words.",
    "Reports": "Download a short Summary Report, or the full report and workbook.",
}
# Basic page -> the Professional page with the full details (and back).
FULL_DETAILS = {"How Are We Doing?": "Executive Overview", "Where the Money Goes": "Channels",
                "What's Working and What's Not": "Campaigns", "Top Alerts": "Anomalies",
                "AI Summary": "AI Insights", "Reports": "Reports", "Upload & Profile": "Upload & Profile",
                "About": "About"}
BASIC_FOR = {pro: basic for basic, pro in FULL_DETAILS.items()}
BASIC_HOME = "How Are We Doing?"
CHART_HEIGHT_BASIC = 420            # "simple large charts"


# ---------------------------------------------------------------------------------------------
# View state
# ---------------------------------------------------------------------------------------------

def _state():
    return st.session_state


def current_view() -> str:
    """Basic or Professional. Without an analysis the Professional menu is shown (the toggle
    appears once an analysis exists); the choice is remembered for the session."""
    if layout.current_output() is None:
        return PROFESSIONAL
    return _state().get(VIEW_KEY) or config_settings.settings.default_view


def _switch_view() -> None:
    """Toggle callback: keep the reader on the matching page of the other view."""
    state = _state()
    new = state.get(VIEW_WIDGET_KEY) or PROFESSIONAL
    if new == BASIC:
        state[PAGE_KEY] = BASIC_FOR.get(state.get("page"), BASIC_HOME)
    else:
        state["page"] = FULL_DETAILS.get(state.get(PAGE_KEY), "Executive Overview")
    state[VIEW_KEY] = new


def open_professional(page: str) -> None:
    """'See Full Details' / 'See Evidence': switch to Professional on `page`."""
    state = _state()
    state[VIEW_KEY] = state[VIEW_WIDGET_KEY] = PROFESSIONAL
    state["page"] = page


def view_toggle() -> str:
    """The 'View: Basic | Professional' switch at the top of the sidebar (only once an analysis
    exists). Returns the view in use."""
    view = current_view()
    if layout.current_output() is None:
        return view
    state = _state()
    state[VIEW_KEY] = view
    if state.get(VIEW_WIDGET_KEY) != view:
        state[VIEW_WIDGET_KEY] = view
    st.radio("View", [BASIC, PROFESSIONAL], key=VIEW_WIDGET_KEY, horizontal=True, on_change=_switch_view,
             help="Basic: the key results in plain words. Professional: every chart, table and setting. "
                  "Switching does not re-run the analysis.")
    if view == BASIC and state.get(PAGE_KEY) not in BASIC_PAGES:
        state[PAGE_KEY] = BASIC_FOR.get(state.get("page"), BASIC_HOME)
    if view == PROFESSIONAL and "page" not in state and state.get(PAGE_KEY):
        state["page"] = FULL_DETAILS.get(state[PAGE_KEY], "Executive Overview")
    return view


# ---------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------

def _header(page: str) -> None:
    ui.page_header(page, BASIC_DESCRIPTIONS.get(page), eyebrow=BASIC_SECTION.get(page) or None,
                   scope="Full dataset")


def _full_details(page: str, label: str = "See Full Details") -> None:
    target = FULL_DETAILS[page]
    st.button(label, key=f"basic_full_{target}", on_click=open_professional, args=(target,),
              icon=":material/open_in_new:", help=f"Opens {target} in the Professional view.")


def _glossary() -> None:
    with st.expander("What Do These Words Mean?", icon=":material/help:"):
        st.markdown("\n".join(f"- **{html.escape(term)}**: {html.escape(text)}" for term, text in pl.GLOSSARY))


def _bullets(lines: list[str]) -> None:
    if lines:
        st.markdown("\n".join(f"- {line}" for line in lines))


def key_number_cards(numbers: list[pl.KeyNumber]) -> None:
    """Large cards: the number, one plain sentence, and a coloured status with its reason
    (colour is never the only signal: the status is also written out)."""
    cards = []
    for k in numbers:
        tip = html.escape(f"{k.status_text}. {k.basis}" if k.basis else k.status_text, quote=True)
        cards.append(
            f"<div class='mi-basic-card mi-basic-{k.status}' title='{tip}'>"
            f"<div class='mi-basic-title'>{html.escape(k.title)}</div>"
            f"<div class='mi-basic-value'>{html.escape(k.value)}</div>"
            f"<div class='mi-basic-sentence'>{html.escape(k.sentence)}</div>"
            f"<div class='mi-basic-status'><span class='mi-basic-dot'></span>{html.escape(k.status_text)}"
            + (f"<span class='mi-basic-basis'> · {html.escape(k.basis)}</span>" if k.basis else "")
            + "</div></div>")
    st.markdown(f"<div class='mi-basic-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _large(fig):
    if fig is not None:
        fig.update_layout(height=max(CHART_HEIGHT_BASIC, fig.layout.height or 0))
    return fig


def _output():
    return layout.current_output()


# ---------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------

def page_doing() -> None:
    page = "How Are We Doing?"
    output = _output()
    a, df = output.analysis, output.clean.clean_df
    _header(page)
    pacing = layout._export_pacing(output)
    numbers = pl.key_numbers(a, df, targets=layout.current_targets(), pacing=pacing)
    if numbers:
        key_number_cards(numbers)
        st.caption(f"Colours: a target's status when you set one; money spent against this month's budget; "
                   f"otherwise the last {config_settings.BASIC_COMPARE_DAYS} days compared with the "
                   f"{config_settings.BASIC_COMPARE_DAYS} days before.")
    else:
        ui.empty_state("This data has no spend, revenue, lead or customer numbers to summarise.", icon="info")
    st.markdown("## In Plain Words")
    _bullets(pl.overview_lines(a, df))
    for title, lines in (("Profit", pl.profit_lines(layout._export_profitability(output))),
                         ("Your Targets", pl.target_lines(layout._export_targets(output))),
                         ("This Month's Budget", pl.pacing_lines(pacing))):
        if lines:
            st.markdown(f"## {title}")
            _bullets(lines)
    trend = a.trends.get("week")
    outcome = next((m for m in ("revenue", "conversions", "leads") if trend is not None and m in trend), None)
    if outcome:
        shown, dropped = cs.full_weeks(trend)
        start, end = cs.date_span(df)
        title = title_case(f"{pl.plain_name(outcome)} per week")
        ui.show_chart(_large(charts.line_chart(shown, "period", outcome, KPI_REGISTRY[outcome].fmt, title,
                                               subtitle=cs.subtitle(outcome, "week", start, end,
                                                                    note=cs.PARTIAL_WEEKS_NOTE if dropped else None))),
                      "No weekly trend is available because the data has no usable date column.")
    _full_details(page)
    _glossary()


def page_money() -> None:
    page = "Where the Money Goes"
    output = _output()
    a = output.analysis
    _header(page)
    ch = a.channels
    if ch is None or ch.empty or "spend_share_pct" not in ch:
        ui.empty_state("This needs a channel or platform column and a spend column.", icon="table_chart")
        _glossary()
        return
    series = {"Share of Spend": "spend_share_pct"}
    if "revenue_share_pct" in ch:
        series["Share of Revenue"] = "revenue_share_pct"
    elif "leads_share_pct" in ch:
        series["Share of Leads"] = "leads_share_pct"
    result_col = list(series.values())[-1]
    start, end = cs.date_span(output.clean.clean_df)
    title = cs.share_title(ch, "channel", result_col, list(series)[-1].removeprefix("Share of "),
                           "Where the Money Goes and What It Brings Back")
    ui.show_chart(_large(charts.share_comparison_chart(ch, "channel", series, title,
                                                       subtitle=cs.subtitle(None, None, start, end,
                                                                            note="Share of total (%)"),
                                                       owned=cs.owned_names(ch["channel"]))),
                  "Channel analysis is not available for this data.")
    _bullets(pl.money_lines(a))
    _full_details(page)
    _glossary()


def page_working() -> None:
    page = "What's Working and What's Not"
    output = _output()
    a = output.analysis
    _header(page)
    best, weakest, metric = pl.best_and_weakest(a)
    if metric is None:
        ui.empty_state("This needs campaign names with spend and revenue or leads.", icon="table_chart")
        _glossary()
        return
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("## Working Well")
        _bullets([pl.campaign_line(r, metric) for _, r in best.iterrows()])
    with right:
        st.markdown("## Not Working as Well")
        _bullets([pl.campaign_line(r, metric) for _, r in weakest.iterrows()] or
                 ["Every campaign is among the best here: there are too few to compare."])
    shown = pd.concat([best, weakest]).drop_duplicates("campaign")
    overall = a.kpis.get(metric)
    overall = overall.value if overall is not None and overall.available else None
    title = ("Money Back per ₹1 Spent: Best and Weakest Campaigns" if metric == "roas"
             else "Cost to Get One Lead: Best and Weakest Campaigns")
    start, end = cs.date_span(output.clean.clean_df)
    ui.show_chart(_large(charts.bar_chart(shown, "campaign", metric, KPI_REGISTRY[metric].fmt, title,
                                          ascending=metric == "cpl", reference=overall, reference_label="Overall",
                                          subtitle=cs.subtitle(None, None, start, end,
                                                               note=f"{len(shown)} Paid Campaigns"))),
                  "No campaigns can be compared.")
    st.caption(f"Campaigns with less than {config_settings.BASIC_MIN_SPEND_SHARE:g}% of all spend are left out: "
               "their results are too small to judge.")
    prof = layout._export_profitability(output)
    losers = prof.loss_makers(3) if prof is not None and prof.available else pd.DataFrame()
    if not losers.empty:
        st.markdown("## Losing Money")
        _bullets([f"{r.campaign} lost {pl.money(-r.contribution)} after the cost of what was sold."
                  for r in losers.itertuples()])
    _full_details(page)
    _glossary()


def page_alerts() -> None:
    page = "Top Alerts"
    a = _output().analysis
    _header(page)
    alerts = pl.top_alerts(a)
    if not alerts:
        ui.empty_state("No problems stood out in this data.", icon="task_alt")
    for i, alert in enumerate(alerts, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {alert['title']}** · {alert['period']}")
            st.markdown(alert["what"])
            st.markdown(f"**{alert['impact']}**")
    if alerts:
        st.caption("Alerts show what changed and by how much, not why. The ₹ amounts are estimates.")
    _full_details(page)
    _glossary()


def _saved_insights() -> dict | None:
    """Saved AI insights for this analysis (this session's, or the saved ones). Never a new call."""
    run = _state().get("ai_run")
    if run is not None and getattr(run, "ok", False):
        return run.insights
    settings = config_settings.settings
    if not settings.ai_cache_enabled:
        return None
    from ai.cache import get_insights
    output = _output()
    cached = get_insights(output.analysis, settings, None, run_id=output.run_id, allow_call=False)
    return cached.insights if cached is not None else None


def page_ai_summary() -> None:
    page = "AI Summary"
    _header(page)
    summary, recs = pl.ai_summary(_saved_insights())
    if not summary and not recs:
        ui.empty_state("No AI summary has been saved for this analysis yet. The results on the other pages "
                       "are complete without it.", icon="auto_awesome")
        if config_settings.settings.ai_enabled:
            _full_details(page, "Create One in AI Insights")
        _glossary()
        return
    st.info("Written by Google Gemini from the verified numbers. Check before acting on it.",
            icon=":material/auto_awesome:")
    for text in summary:
        st.markdown(text)
    if recs:
        st.markdown("## Top 3 Recommendations")
        st.markdown("\n".join(f"{i}. {text}" for i, text in enumerate(recs, start=1)))
    _full_details(page, "See Evidence")
    _glossary()


def page_reports() -> None:
    output = _output()
    _header("Reports")
    with st.container(border=True):
        ui.card_title("Summary Report (PDF)", "Two to three pages in plain words: the key numbers, where the "
                      "money goes, what's working, the top alerts and the AI summary if one is saved.")
        if st.button("Generate Summary Report", key="generate_summary", type="primary",
                     icon=":material/picture_as_pdf:"):
            from reports.pdf_report import ReportError
            from reports.summary_report import export_summary_pdf
            try:
                with st.spinner("Building the summary report…"):
                    path = export_summary_pdf(output.analysis, df=output.clean.clean_df, ai=_saved_insights(),
                                              profitability=layout._export_profitability(output),
                                              targets=layout._export_targets(output),
                                              pacing=layout._export_pacing(output))
                _state()["summary_path"] = str(path)
            except ReportError as exc:
                ui.friendly_error(exc)
        path = _state().get("summary_path")
        if path and Path(path).exists():
            st.success(f"Your summary report is ready: {Path(path).name}")
            st.download_button("Download Summary Report", Path(path).read_bytes(), file_name=Path(path).name,
                               mime="application/pdf", key="download_summary", icon=":material/download:")
    st.markdown("## Full Reports")
    layout.page_reports(show_header=False)


RENDERERS = {
    "How Are We Doing?": page_doing,
    "Where the Money Goes": page_money,
    "What's Working and What's Not": page_working,
    "Top Alerts": page_alerts,
    "AI Summary": page_ai_summary,
    "Reports": page_reports,
}
