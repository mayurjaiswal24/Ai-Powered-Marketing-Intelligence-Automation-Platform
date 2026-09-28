"""The dashboard pages. Every number shown comes from the AnalysisResult (full dataset) or from
the same analytics functions re-run on the filtered rows."""

from __future__ import annotations

import html
import io
import logging
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.campaign import campaign_table, rank_campaigns
from analytics.channel import channel_table, owned_channels, paid_channels
from analytics.common import campaign_key
from analytics.funnel import biggest_drop_off, funnel_by_channel, funnel_table
from analytics.kpis import KPI_REGISTRY, compute_kpis, group_kpis, period_comparison
from analytics.segments import SEGMENT_DIMENSIONS, available_dimensions, segment_table
from analytics.trends import time_series
from config.fields import FIELD_BY_NAME, FIELDS, channel_type
import config.settings as config_settings
from dashboard import charts, components as ui, theme
from dashboard.filters import (DATE_PRESETS, Filters, apply_dimension_filters, apply_filters,
                               available_filters, describe, preset_range)
from dashboard.pipeline import (DEFAULT_SAMPLE, SAMPLE_DATASETS, load_raw, prepare,
                                reopen_run, run_pipeline, sample_path)
from ingestion.loader import IngestionError
from processing.cleaner import CleaningError
from reports.excel_report import WorkbookError, export_workbook
from reports.pdf_report import ReportError, export_pdf
from utils.formatting import format_count, format_date, format_datetime_ist, format_value

logger = logging.getLogger("marketing_intelligence")

# Sidebar menu: sections in order, each with its pages. A blank section name = divider only.
NAV_SECTIONS = [
    ("Data", ["Upload & Profile", "Data Quality"]),
    ("Analysis", ["Executive Overview", "Performance Trends", "Channels", "Campaigns", "Funnel",
                  "Segments", "Anomalies"]),
    ("AI", ["AI Insights"]),
    ("Output", ["Reports"]),
    ("", ["About"]),
]
PAGES = [page for _, pages in NAV_SECTIONS for page in pages]
PAGE_SECTION = {page: section for section, pages in NAV_SECTIONS for page in pages}
PAGE_ICONS = {
    "Upload & Profile": "upload_file", "Data Quality": "fact_check", "Executive Overview": "dashboard",
    "Performance Trends": "trending_up", "Channels": "hub", "Campaigns": "campaign",
    "Funnel": "filter_alt", "Segments": "pie_chart", "Anomalies": "notification_important",
    "AI Insights": "auto_awesome", "Reports": "description", "About": "info",
}
PAGE_DESCRIPTIONS = {
    "Upload & Profile": "Turn any marketing export into verified KPIs, insights and ready-to-share "
                        "reports in minutes.",
    "Data Quality": "What was fixed, flagged or excluded while cleaning, and what it means for the numbers.",
    "Executive Overview": "The headline numbers, where the money goes and the most important findings.",
    "Performance Trends": "How results and efficiency move over time.",
    "Channels": "Compare paid channels on cost and return; owned channels are shown separately.",
    "Campaigns": "Rank campaigns, spot the strongest and weakest, and see which are trending.",
    "Funnel": "How many people move from one stage to the next, and where most are lost.",
    "Segments": "Performance by customer segment, geography and product.",
    "Anomalies": "Unusual periods found automatically, ranked by estimated rupee impact.",
    "AI Insights": "Google Gemini interprets the verified findings. It never calculates the numbers; "
                   "every statement cites the evidence it is based on.",
    "Reports": "Executive PDF and analytical Excel workbook for the full dataset (dashboard filters "
               "do not apply to exports).",
    "About": "What this platform does and who built it.",
}
OVERVIEW_KPIS = ["revenue", "spend", "leads", "conversions", "roas", "cpl", "cac",
                 "lead_to_conversion_rate"]
NOT_READY = "Load a dataset and run the analysis first (see Upload & Profile in the sidebar)."


# ---------------------------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------------------------

def _state():
    return st.session_state


def _reset_for_new_file(raw_df, report, label):
    s = _state()
    s["raw_df"], s["report"], s["source_label"] = raw_df, report, label
    s["overrides"] = {}
    for key in ("output", "prep", "prep_key", "view", "view_key", "pdf_path", "excel_path", "ai_run", "ai_confirm"):
        s.pop(key, None)


def current_prep():
    """Profile + mapping for the loaded file, recomputed only when the file or the user's
    mapping choices change. Gemini is asked (once per column layout) about the columns the
    rules could not map; a failed or unavailable attempt is not repeated in this session."""
    s = _state()
    if "raw_df" not in s:
        return None
    key = (s["report"].file_hash, tuple(sorted(s.get("overrides", {}).items(), key=str)))
    if s.get("prep_key") != key:
        layout, assistant = _assistant(s["raw_df"].columns)
        s["prep"] = prepare(s["raw_df"], s["report"], s.get("overrides") or None, assistant=assistant)
        s["ai_map_session_calls"] = s.get("ai_map_session_calls", 0) + assistant.calls_made
        if s["prep"].ai.source == "none" and s["prep"].ai.note:
            s.setdefault("ai_map_tried", {})[layout] = s["prep"].ai.note
        s["prep_key"] = key
    return s["prep"]


def _session_id() -> str:
    """A random id for this browser session (public mode: runs and choices stay in it)."""
    s = _state()
    if "session_id" not in s:
        s["session_id"] = uuid.uuid4().hex
    return s["session_id"]


def _public() -> bool:
    return bool(config_settings.settings.public_mode)


def _assistant(columns):
    """The mapping assistant for this session: remembers choices, asks Gemini when allowed."""
    import ai.client as ai_client
    from ai.mapping import MappingAssistant, layout_key
    settings = config_settings.settings
    s = _state()
    tried = s.get("ai_map_tried", {})
    layout = layout_key(columns)
    return layout, MappingAssistant(
        settings, lambda: ai_client.client_from_settings(settings), session_id=_session_id(),
        session_calls=s.get("ai_map_session_calls", 0), allow_call=layout not in tried,
        blocked_note=tried.get(layout, ""))


@st.cache_resource(show_spinner=False)
def _startup_housekeeping(public_mode: bool) -> list[int]:
    """Once per app start: in public mode, delete runs older than 24 hours (and beyond
    KEEP_LAST_RUNS). Cached, so it runs once per server process, not on every page view."""
    if not public_mode:
        return []
    from database.connection import DatabaseError
    from database.housekeeping import housekeep
    try:
        return housekeep()
    except (DatabaseError, OSError):
        return []


def startup() -> None:
    _startup_housekeeping(bool(config_settings.settings.public_mode))


def current_output():
    return _state().get("output")


# ---------------------------------------------------------------------------------------------
# Sidebar: navigation + filters
# ---------------------------------------------------------------------------------------------

def _nav_label(page: str) -> str:
    return f":material/{PAGE_ICONS[page]}: {page}"


def _nav_section_starts() -> dict[int, str]:
    """1-based menu position of the first page in each section -> section label."""
    starts, position = {}, 1
    for section, pages in NAV_SECTIONS:
        starts[position] = section
        position += len(pages)
    return starts


def _header(page: str, filters: Filters | None = None, title: str | None = None) -> None:
    """The standard page header; analysis pages also say which data is shown."""
    ui.page_header(title or page, PAGE_DESCRIPTIONS.get(page), eyebrow=PAGE_SECTION.get(page) or None,
                   scope=_filter_caption(filters) if filters is not None else None)


def sidebar() -> tuple[str, Filters | None]:
    with st.sidebar:
        ui.brand_header()
        st.markdown(theme.nav_css(_nav_section_starts()), unsafe_allow_html=True)
        with st.container(key="mi_nav"):
            page = st.radio("Menu", PAGES, key="page", label_visibility="collapsed",
                            format_func=_nav_label, width="stretch")
        filters = None
        if page != "About":           # About is static: no filters, no data
            try:
                filters = _sidebar_filters()
            except Exception:  # noqa: BLE001 - a filter problem must not hide the menu or footer
                logger.exception("Could not build the sidebar filters")
                st.caption("Filters are unavailable for this analysis.")
        ui.signature()
        return page, filters


def _sidebar_filters() -> Filters | None:
    """Date and dimension filters for the analysis pages (only once an analysis exists)."""
    output = current_output()
    if output is None:
        return None
    df = output.clean.clean_df
    st.divider()
    ui.section_label("Filters")
    filters = Filters()
    if "date" in df and df["date"].notna().any():
        data_start, data_end = df["date"].min(), df["date"].max()
        filters.data_start, filters.data_end = data_start, data_end
        preset = st.radio("Date range", DATE_PRESETS, key="date_preset", horizontal=False)
        if preset == "Custom range":
            picked = st.date_input("From - to", value=(data_start.date(), data_end.date()),
                                   min_value=data_start.date(), max_value=data_end.date(),
                                   key="date_custom")
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                filters.start, filters.end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
            else:
                filters.start, filters.end = data_start, data_end
        else:
            filters.start, filters.end = preset_range(preset, data_start, data_end)
    for col, label in available_filters(df):
        options = sorted(df[col].dropna().astype(str).unique())
        filters.dimensions[col] = st.multiselect(label, options, key=f"filter_{col}",
                                                 placeholder="All")
    if filters.active and st.button("Clear filters", key="clear_filters"):
        for col, _ in available_filters(df):
            st.session_state[f"filter_{col}"] = []
        st.session_state["date_preset"] = DATE_PRESETS[0]
        st.rerun()
    meta = output.analysis.metadata
    st.caption(f"Dataset: {meta.get('dataset_name')}"
               + (f" · Analysis #{output.run_id}" if output.run_id else ""))
    return filters


# ---------------------------------------------------------------------------------------------
# Filtered view (recomputed with the analytics functions, cached per filter selection)
# ---------------------------------------------------------------------------------------------

def filtered_view(filters: Filters) -> dict:
    s = _state()
    output = current_output()
    key = (output.analysis.metadata.get("created_at"), filters.key())
    if s.get("view_key") == key:
        return s["view"]
    full = output.clean.clean_df
    caps = output.analysis.capabilities
    df = apply_filters(full, filters)
    deltas = None
    if filters.date_filtered and filters.start is not None:
        comparison = period_comparison(apply_dimension_filters(full, filters), filters.start,
                                       filters.end, capabilities=caps)
        deltas = comparison.deltas if not comparison.note else None
    view = {
        "df": df,
        "kpis": compute_kpis(df, caps),
        "deltas": deltas,
        "campaigns": campaign_table(df) if not df.empty else pd.DataFrame(),
        "channels": channel_table(df) if not df.empty else pd.DataFrame(),
        "funnel": funnel_table(df) if not df.empty else pd.DataFrame(),
        "funnel_by_channel": funnel_by_channel(df) if not df.empty else pd.DataFrame(),
    }
    s["view"], s["view_key"] = view, key
    return view


def _filter_caption(filters: Filters | None) -> str:
    return describe(filters) if filters is not None else "Showing all data"


# ---------------------------------------------------------------------------------------------
# Page: Upload & Profile
# ---------------------------------------------------------------------------------------------

def page_upload() -> None:
    _header("Upload & Profile")
    if _public():
        st.info("Demo app: your data is kept only for this session and deleted automatically.")
    ui.steps_guide([
        ("Upload", "Add a CSV or Excel export from any ad platform, CRM or campaign tracker."),
        ("Check mapping", "The app recognises your columns and cleans the data. Confirm anything "
                          "it is unsure about."),
        ("Analyse", "Run the analysis to get verified KPIs, channel and campaign insights, "
                    "incidents and reports."),
    ])
    left, right = st.columns([3, 2], gap="large")
    with left, st.container(border=True):
        limit_mb = config_settings.settings.upload_limit_mb      # 10 MB in public mode
        ui.card_title("Upload your data", f"CSV or Excel, up to {limit_mb} MB. One row per day and "
                      "campaign works best.")
        uploaded = st.file_uploader("Marketing data file", type=["csv", "xlsx"], key="uploader",
                                    max_upload_size=limit_mb, label_visibility="collapsed",
                                    help="One row per day and campaign works best. Maximum size: "
                                         f"{limit_mb} MB.")
        if uploaded is not None and _state().get("uploaded_id") != uploaded.file_id:
            _state()["uploaded_id"] = uploaded.file_id
            try:
                raw_df, report = load_raw(uploaded)
                _reset_for_new_file(raw_df, report, uploaded.name)
            except IngestionError as exc:
                ui.friendly_error(exc)
    with right, st.container(border=True):
        ui.card_title("Try the Demo", "No file at hand? Explore the platform with realistic sample "
                      "data from an education brand.")
        label = st.selectbox("Sample dataset", list(SAMPLE_DATASETS),
                             index=list(SAMPLE_DATASETS).index(DEFAULT_SAMPLE), key="sample_choice")
        if st.button("Load sample dataset", key="load_sample", type="primary",
                     icon=":material/play_arrow:"):
            try:
                raw_df, report = load_raw(sample_path(label))
                _reset_for_new_file(raw_df, report, label)
                _open_demo_snapshot(report)
            except IngestionError as exc:
                ui.friendly_error(exc)

    prep = current_prep()
    if prep is None:
        output = current_output()
        if output is not None:
            meta = output.analysis.metadata
            st.success(f"Opened saved analysis run #{output.run_id} ({meta.get('dataset_name')}, "
                       f"{format_count(meta.get('rows'))} rows, analysed {_format_timestamp(meta.get('created_at'))}). "
                       "Use the sidebar to explore it, or upload a file to start a new analysis.")
        else:
            ui.empty_state("No data loaded yet. Upload a file or load a sample dataset to begin.")
        _recent_analyses()
        return

    report, profile = prep.report, prep.profile
    st.markdown("## Dataset profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File", report.filename if len(report.filename) < 28 else report.filename[:25] + "...",
              border=True, help=report.filename)
    c2.metric("Rows", format_count(profile.rows), border=True)
    c3.metric("Columns", format_count(len(profile.column_names)), border=True)
    c4.metric("Date range", "N/A" if profile.date_min is None else
              f"{format_date(profile.date_min)} - {format_date(profile.date_max)}", border=True)
    for warning in report.warnings:
        st.info(warning)

    tab_quality, tab_mapping, tab_caps = st.tabs(["Data-quality findings", "Field mapping",
                                                  "What this data supports"])
    with tab_quality:
        if not profile.findings:
            st.success("No data-quality problems were found.")
        for f in profile.findings:
            st.markdown(f"- {f.message}")
        st.caption("These are fixed or flagged automatically during cleaning; every change is logged "
                   "on the Data Quality page.")
    with tab_mapping:
        _mapping_editor(prep)
    with tab_caps:
        for cap in prep.validation.capabilities.values():
            st.markdown(f"- **{cap.label}**: {'available' if cap.enabled else 'not available'}. "
                        f"{cap.reason}")
        for note in prep.validation.notes:
            st.caption(note)

    st.markdown("## Run analysis")
    for message in prep.validation.blocking:
        st.warning(message)
    _ai_mapping_banner(prep)
    _confirm_columns(prep)
    _mapping_checks(prep)
    _final_mapping_summary(prep)
    state = _state()
    if current_output() is not None and state.get("output_prep_key") not in (None, state.get("prep_key")):
        st.info("The mapping changed since the last analysis. Press Run analysis to update the results.")
    run = st.button("Run analysis", type="primary", key="run_analysis",
                    disabled=not prep.validation.can_analyse)
    if run:
        try:
            with st.status("Running analysis...", expanded=True) as status:
                output = run_pipeline(prep, progress=lambda msg: status.write(msg),
                                      session_id=_session_id() if _public() else None)
                status.update(label="Analysis complete", state="complete", expanded=False)
            _state()["output"] = output
            _state()["output_prep_key"] = _state().get("prep_key")
            _preload_demo(output, prep.report)
            for key in ("view", "view_key", "pdf_path", "excel_path", "ai_run", "ai_confirm"):
                _state().pop(key, None)
        except (CleaningError, IngestionError) as exc:
            ui.friendly_error(exc)
        except Exception as exc:  # noqa: BLE001 - never show a traceback in the UI
            ui.friendly_error(exc)
    output = current_output()
    if output is not None:
        meta = output.analysis.metadata
        st.success(f"Analysis ready: {format_count(meta['rows'])} clean rows. Open Executive "
                   "Overview in the sidebar.")
        if _state().pop("demo_opened", False):
            st.caption("The finished sample analysis was opened instantly (it ships with the app, "
                       "with its saved AI insights). Your own uploads always run the full analysis.")
        if output.is_reupload:
            st.info("This exact file was uploaded before; a new analysis run was recorded for it.")
        if output.save_error:
            st.warning(f"The results are shown but were not saved: {output.save_error}")
        if "raw_df" in _state():
            st.button("Edit mapping and re-run", key="edit_rerun", on_click=_open_mapping_editor,
                      help="Opens 'Change any mapping' above with your current choices filled in.")
    _recent_analyses()


def _recent_analyses() -> None:
    """Previous runs saved in the database, reopened without recomputing."""
    from database import repository
    from database.connection import DatabaseError
    try:
        recent = repository.list_recent_runs(limit=8, session_id=_session_id() if _public() else None)
    except DatabaseError as exc:
        ui.friendly_error(exc)
        return
    st.markdown("## Recent analyses")
    message = _state().pop("deleted_message", None)
    if message:
        st.success(message)
    if recent.empty:
        st.caption("Analyses you run are saved here so you can reopen them later without recalculating.")
        return
    st.caption("Reopening shows the saved results instantly (no recalculation and no AI call). "
               f"Only your {config_settings.settings.keep_last_runs} most recent analyses are kept.")
    state = _state()
    grid = st.columns(2, gap="medium")
    for i, row in enumerate(recent.itertuples()):
        card = grid[i % 2].container(border=True)
        reports = f" · {row.reports} report{'s' if row.reports != 1 else ''}" if row.reports else ""
        card.markdown(
            f"<div class='mi-run-name' title='{html.escape(str(row.file_name))}'>"
            f"{html.escape(str(row.file_name))}</div><div class='mi-run-meta'>Analysis #{row.run_id} · "
            f"{format_count(row.row_count)} rows · {_format_timestamp(row.created_at)}{reports}</div>",
            unsafe_allow_html=True)
        c3, c4, _ = card.columns([1, 1, 2])
        if c4.button("Delete", key=f"delete_run_{row.run_id}", icon=":material/delete:",
                     width="stretch"):
            state["confirm_delete"] = int(row.run_id)
        if state.get("confirm_delete") == int(row.run_id) and _owns_run(int(row.run_id)):
            with card:
                _confirm_delete(int(row.run_id), row.file_name)
        if c3.button("Open", key=f"open_run_{row.run_id}", icon=":material/folder_open:",
                     width="stretch") and _owns_run(int(row.run_id)):
            try:
                output = reopen_run(int(row.run_id))
            except DatabaseError as exc:
                ui.friendly_error(exc)
                continue
            state = _state()
            for key in ("raw_df", "report", "source_label", "overrides", "prep", "prep_key", "view",
                        "view_key", "pdf_path", "excel_path", "ai_run", "ai_confirm", "uploaded_id"):
                state.pop(key, None)
            state["output"] = output
            st.rerun()


_STATUS_LABELS = {
    "confirmed": "Confirmed", "high_confidence": "High confidence (inferred)",
    "uncertain": "Uncertain - please confirm", "unmapped": "Not recognised",
    "derived": "Derived metric (recalculated)", "not_used": "Not used (on purpose)",
    "ignored": "Not used (your choice)"}
_NOT_USED = "Not used"
_LINE_BREAK = "\n"      # markdown lists inside st.success / st.warning need real line breaks


def _open_demo_snapshot(report) -> None:
    """Kalpa clean sample: open the finished analysis that ships with the app (no waiting)."""
    from dashboard.demo import load_demo_snapshot, restore_demo_run
    payload = load_demo_snapshot(report.file_hash)
    if payload is None:
        return
    prep, output = restore_demo_run(payload, config_settings.settings,
                                    session_id=_session_id() if _public() else None)
    state = _state()
    state["prep"], state["prep_key"] = prep, (report.file_hash, ())
    state["output"], state["output_prep_key"] = output, state["prep_key"]
    state["demo_opened"] = True


def _preload_demo(output, report) -> None:
    """Kalpa sample on a database without saved insights: store the demo seed's insights."""
    from ai.demo_seed import preload_demo_insights
    try:
        preload_demo_insights(output, report, config_settings.settings)
    except Exception:  # noqa: BLE001 - the demo seed is a convenience; never break a run
        pass


def _owns_run(run_id: int) -> bool:
    """Public mode: a visitor may open or delete only runs made in their own session."""
    if not _public():
        return True
    from database import repository
    from database.connection import DatabaseError
    try:
        return repository.run_session_id(run_id) == _session_id()
    except DatabaseError:
        return False


def _confirm_delete(run_id: int, file_name: str) -> None:
    """Second click needed: deleting removes the saved results, reports and cached AI insights."""
    st.warning(f"Delete analysis run #{run_id} ({file_name})? Its saved results, PDF/Excel "
               "reports and cached AI insights will be removed. This cannot be undone.")
    yes, no = st.columns(2)
    if yes.button("Yes, delete", key=f"confirm_delete_{run_id}", type="primary"):
        from database.connection import DatabaseError
        from database.housekeeping import delete_analysis
        state = _state()
        state.pop("confirm_delete", None)
        try:
            delete_analysis(run_id)
        except DatabaseError as exc:
            ui.friendly_error(exc)
            return
        output = state.get("output")
        if output is not None and output.run_id == run_id:      # it was open: close it
            for key in ("output", "view", "view_key", "pdf_path", "excel_path", "ai_run", "ai_confirm"):
                state.pop(key, None)
        state["deleted_message"] = f"Analysis run #{run_id} was deleted."
        st.rerun()
    if no.button("Cancel", key=f"cancel_delete_{run_id}"):
        _state().pop("confirm_delete", None)
        st.rerun()


def _source_label(m) -> str:
    if m.source == "user":
        return "your choice"
    return "AI" if m.source in ("ai", "ai_rejected") else "rule"


def _mapping_editor(prep) -> None:
    """Every column: what it is used as, who decided (rule / AI / your choice), the AI check
    badge and the reason. AI rows are highlighted. Any column can be changed below."""
    from ai.mapping import BADGES
    frame = prep.mapping.to_frame()
    table = pd.DataFrame({
        "Column in file": frame["column"],
        "Used as": frame["maps_to"],
        "Source": [_source_label(m) for m in prep.mapping.columns],
        "Check": [BADGES.get(prep.badges.get(m.column, ""), "") for m in prep.mapping.columns],
        "Status": frame["status"].map(_STATUS_LABELS).fillna(frame["status"]),
        "Why": frame["reason"],
    })
    badge_of = [prep.badges.get(m.column, "") if m.source in ("ai", "ai_rejected") else None
                for m in prep.mapping.columns]

    def highlight(row):
        badge = badge_of[row.name]
        colour = "" if badge is None else theme.AI_ROW_BACKGROUND.get(badge, theme.AI_ROW_BACKGROUND[""])
        return [f"background-color: {colour}" if colour else ""] * len(row)

    st.dataframe(table.style.apply(highlight, axis=1), hide_index=True, width="stretch")
    st.caption("Highlighted rows were mapped by Gemini. Columns that need a decision are also shown "
               "next to the Run analysis button below.")
    _all_columns_editor(prep)


def _open_mapping_editor() -> None:
    """Used by the 'Edit mapping and re-run' buttons (also as a button callback)."""
    _state()["edit_mapping"] = True
    _state()["page"] = "Upload & Profile"


def _all_columns_editor(prep) -> None:
    """A dropdown for every column, whoever mapped it. Changes are applied together, remembered
    for this column layout, and win over AI and rules next time."""
    from ai.mapping import layout_key
    layout = layout_key(prep.raw_df.columns)[:10]
    with st.expander("Change any mapping", expanded=bool(_state().get("edit_mapping"))):
        with st.form(f"mapping_form_{layout}"):
            picks: dict[str, str | None] = {}
            cols = st.columns(3)
            for i, m in enumerate(prep.mapping.columns):
                current = m.field if m.status in ("confirmed", "high_confidence") and m.field else _NOT_USED
                choices = _field_choices(m.field, [f for f, _ in m.candidates])
                picked = cols[i % 3].selectbox(
                    f"{m.column}  ({_source_label(m)})", choices, index=choices.index(current),
                    format_func=_choice_label, key=f"map_{layout}_{m.column}")
                picks[m.column] = None if picked == _NOT_USED else picked
            submitted = st.form_submit_button("Apply changes", key=f"apply_all_{layout}")
        if submitted:
            changed = {}
            for m in prep.mapping.columns:
                before = m.field if m.status in ("confirmed", "high_confidence") else None
                if picks[m.column] != before:
                    changed[m.column] = picks[m.column]
            _apply_user_choices(prep, changed)


def _field_choices(first: str | None = None, suggestions: list[str] = ()) -> list[str]:
    """Suggestions first, then every other field, then 'Not used'."""
    ordered = [f for f in ([first] if first else []) + list(suggestions) if f]
    ordered += [f.name for f in FIELDS if f.name not in ordered]
    return list(dict.fromkeys(ordered)) + [_NOT_USED]


def _choice_label(choice: str) -> str:
    return FIELD_BY_NAME[choice].label if choice in FIELD_BY_NAME else choice


def _confirm_columns(prep) -> None:
    """Dropdowns for columns that need a decision, right where the user is about to run.
    The app's suggestion is pre-selected, but nothing is used until 'Apply choices' is clicked
    (except a Profit column, which starts as 'Not used': using it could double-count spend)."""
    overrides = _state().setdefault("overrides", {})
    pending = [m for m in prep.mapping.columns if m.status == "uncertain"]
    chosen = [m for m in prep.mapping.columns if m.source == "user"]
    if not pending and not chosen:
        return
    if pending:
        st.markdown("**Please confirm what these columns mean**")
        st.caption("Business-critical fields (Date, Spend, Revenue, Leads, Conversions) are never "
                   "guessed; analysis starts once you confirm them. Other columns are optional.")
    picks: dict[str, str | None] = {}
    groups = [(pending, None)]
    if chosen:
        groups.append((chosen, st.expander(f"Your mapping choices ({len(chosen)})",
                                           expanded=not pending)))
    for group, container in groups:
        for m in group:
            suggestions = [f for f, _ in m.candidates]
            default = (overrides.get(m.column, _NOT_USED) if m.column in overrides
                       else _NOT_USED if m.source in ("profit", "ai_rejected")
                       else suggestions[0] if suggestions else _NOT_USED)
            default = default or _NOT_USED
            choices = _field_choices(m.field, suggestions)
            with container if container is not None else st.container():
                picked = st.selectbox(
                    f"'{m.column}' contains", choices, index=choices.index(default),
                    format_func=_choice_label, key=f"confirm_{m.column}",
                    help=m.reason or None)
                if m.status == "uncertain" and m.reason:
                    st.caption(f"Why asked: {m.reason}.")
            picks[m.column] = None if picked == _NOT_USED else picked
    if st.button("Apply choices", key="apply_mapping", type="secondary"):
        _apply_user_choices(prep, picks)


def _apply_user_choices(prep, choices: dict[str, str | None]) -> None:
    """Use the user's choices now and remember them for this column layout (user > AI > rules)."""
    state = _state()
    if choices:
        state.setdefault("overrides", {}).update(choices)
        _layout, assistant = _assistant(prep.raw_df.columns)
        assistant.remember_choices(prep.raw_df.columns, choices)
    state.pop("prep_key", None)
    state["edit_mapping"] = False
    st.rerun()


def _mapping_checks(prep) -> None:
    """The file's own calculations and the plausibility warnings, before the run."""
    checks = prep.crosscheck.checks
    verified = [c for c in checks if c.status == "verified"]
    if verified:
        st.success(_LINE_BREAK.join(["Verified by your file's own calculations:"]
                                    + [f"- {c.message}" for c in verified]))
    for c in checks:
        if c.status == "mismatch":
            st.warning(c.message)
        elif c.status in ("alternative", "unchecked"):
            st.info(c.message)
    if prep.warnings:
        from ingestion.plausibility import HEADLINE
        st.warning(_LINE_BREAK.join([HEADLINE] + [f"- {w.message}" for w in prep.warnings]))



def _final_mapping_summary(prep) -> None:
    """One compact line per used field (with who decided and the AI check), right above the
    Run analysis button."""
    from ai.mapping import BADGES
    verified = prep.crosscheck.verified_fields
    parts = []
    for m in prep.mapping.columns:
        if m.status not in ("confirmed", "high_confidence"):
            continue
        tags = [_source_label(m)]
        if m.column in prep.badges:
            tags.append(BADGES[prep.badges[m.column]])
        elif m.field in verified:
            tags.append("verified ✓")
        parts.append(f"{FIELD_BY_NAME[m.field].label} ← '{m.column}' ({', '.join(tags)})")
    skipped = [m.column for m in prep.mapping.columns if m.status not in ("confirmed", "high_confidence")]
    st.markdown("**Final mapping summary**")
    st.caption(" · ".join(parts))
    if skipped:
        st.caption(f"Not used ({len(skipped)}): " + ", ".join(skipped))


def _ai_mapping_banner(prep) -> None:
    """How many columns Gemini mapped, the privacy note, and why AI was not used (if so)."""
    from ai.mapping import PRIVACY_NOTE
    ai_columns = prep.ai_mapped_columns
    if ai_columns:
        st.info(f"Gemini mapped {len(ai_columns)} column{'s' if len(ai_columns) != 1 else ''} "
                "(highlighted). Review them or change any mapping below.")
    if prep.ai.note:
        st.caption(prep.ai.note)
    if ai_columns or config_settings.settings.ai_enabled:
        st.caption(PRIVACY_NOTE)


# ---------------------------------------------------------------------------------------------
# Page: Executive Overview
# ---------------------------------------------------------------------------------------------

def page_overview(filters: Filters) -> None:
    output = current_output()
    view = filtered_view(filters)
    _header("Executive Overview", filters)
    if view["df"].empty:
        ui.empty_state("No rows match the current filters.")
        return
    ui.kpi_cards(view["kpis"], view["deltas"], OVERVIEW_KPIS)
    if view["deltas"] is None:
        st.caption("Change arrows compare with the previous period of the same length. Choose "
                   "'Last 30 days' or 'Last 90 days' in the sidebar to see them.")
    for note in output.analysis.notes:
        st.caption(note)

    left, right = st.columns(2, gap="large")
    trend = time_series(view["df"], "week")
    outcome = next((m for m in ("revenue", "conversions", "leads") if m in trend), None)
    with left:
        fig = charts.line_chart(trend, "period", outcome, KPI_REGISTRY[outcome].fmt,
                                f"Weekly {KPI_REGISTRY[outcome].label.lower()}") if outcome else None
        ui.show_chart(fig, "No trend is available (the data has no usable date column).")
    with right:
        ch = view["channels"]
        series = {"Share of spend": "spend_share_pct"}
        if "revenue_share_pct" in ch:
            series["Share of revenue"] = "revenue_share_pct"
        elif "leads_share_pct" in ch:
            series["Share of leads"] = "leads_share_pct"
        fig = charts.share_comparison_chart(ch, "channel", series, "Where the money goes and what it returns") \
            if not ch.empty else None
        ui.show_chart(fig, "Channel analysis is not available for this data.")

    st.markdown("## Key findings")
    st.caption("Rule-based analytical findings for the full dataset (not affected by filters).")
    if output.analysis.findings:
        ui.findings_list(output.analysis.findings, limit=5)
    else:
        ui.empty_state("No findings could be produced from this data.")


# ---------------------------------------------------------------------------------------------
# Page: Performance Trends
# ---------------------------------------------------------------------------------------------

def page_trends(filters: Filters) -> None:
    view = filtered_view(filters)
    _header("Performance Trends", filters)
    if "date" not in view["df"] or view["df"].empty:
        ui.empty_state("Trends need a date column and at least one row in the current filters.")
        return
    grain_label = st.segmented_control("Time grain", ["Daily", "Weekly", "Monthly"], default="Weekly",
                                       key="trend_grain") or "Weekly"
    grain = {"Daily": "day", "Weekly": "week", "Monthly": "month"}[grain_label]
    trend = time_series(view["df"], grain)
    if trend.empty:
        ui.empty_state("No trend data for the current filters.")
        return
    xfmt = "text" if grain == "month" else "date"
    efficiency_options = [k for k in ("roas", "cpl", "cac", "ctr", "lead_to_conversion_rate", "cpc")
                          if k in trend and trend[k].notna().any()]
    items = [(m, f"{KPI_REGISTRY[m].label}") for m in ("revenue", "spend", "conversions", "leads")
             if m in trend and trend[m].notna().any()][:3]
    cols = st.columns(2, gap="large")
    for i, (metric, label) in enumerate(items):
        with cols[i % 2]:
            ui.show_chart(charts.line_chart(trend, "period", metric, KPI_REGISTRY[metric].fmt,
                                            f"{label} by {grain_label.lower().rstrip('ly') or 'day'}",
                                            x_label_fmt=xfmt), f"No {label.lower()} data.")
    with cols[len(items) % 2]:
        if efficiency_options:
            eff = st.selectbox("Efficiency metric", efficiency_options, key="trend_eff",
                               format_func=lambda k: KPI_REGISTRY[k].label)
            ui.show_chart(charts.line_chart(trend, "period", eff, KPI_REGISTRY[eff].fmt,
                                            f"{KPI_REGISTRY[eff].label} over time", x_label_fmt=xfmt),
                          "No efficiency data.")
    if (trend["days"] < {"day": 1, "week": 7, "month": 28}[grain]).any():
        st.caption("The first or last period may be partial (fewer days of data), so it can look "
                   "lower than the others.")


# ---------------------------------------------------------------------------------------------
# Page: Channels
# ---------------------------------------------------------------------------------------------

def page_channels(filters: Filters) -> None:
    view = filtered_view(filters)
    _header("Channels", filters)
    ch = view["channels"]
    if ch.empty:
        ui.empty_state("Channel analysis needs a channel or platform column.")
        return
    paid, owned = paid_channels(ch), owned_channels(ch)
    df = view["df"]
    key = "channel" if "channel" in df else "platform"
    paid_kpis = compute_kpis(df[df[key].map(channel_type) == "paid"]) if not paid.empty else {}
    metrics = [k for k in ("roas", "cpl", "cac", "ctr", "cpc", "lead_to_conversion_rate",
                           "revenue", "spend", "leads", "conversions") if k in paid and paid[k].notna().any()]
    if metrics:
        left, right = st.columns(2, gap="large")
        with left:
            metric = st.selectbox("Compare paid channels on", metrics, key="channel_metric",
                                  format_func=lambda k: KPI_REGISTRY[k].label)
            ref = paid_kpis.get(metric)
            better_high = KPI_REGISTRY[metric].higher_is_better
            ui.show_chart(charts.bar_chart(paid, "channel", metric, KPI_REGISTRY[metric].fmt,
                                           f"{KPI_REGISTRY[metric].label} by paid channel",
                                           ascending=better_high is False,
                                           reference=ref.value if ref and metric not in
                                           ("revenue", "spend", "leads", "conversions") else None,
                                           reference_label="Paid average"),
                          "No data for this metric.")
        with right:
            index_cols = [c for c in paid.columns if c.endswith("_index") and paid[c].notna().any()]
            if index_cols:
                idx = st.selectbox("Efficiency index (paid-media average = 100)", index_cols, key="channel_index",
                                   format_func=lambda c: KPI_REGISTRY[c.removesuffix("_index")].label)
                base = idx.removesuffix("_index")
                ui.show_chart(charts.index_chart(paid, "channel", idx, f"{KPI_REGISTRY[base].label} index vs "
                                                 "paid-media average",
                                                 lower_is_better=KPI_REGISTRY[base].higher_is_better is False),
                              "No index available.")
    cols = ["channel", "spend", "spend_share_pct", "impressions", "clicks", "ctr", "cpc", "leads", "cpl",
            "conversions", "lead_to_conversion_rate", "cac", "revenue", "revenue_share_pct", "roas", "roi"]
    st.markdown("## Paid channels")
    ui.show_table(paid.sort_values("spend", ascending=False) if "spend" in paid else paid, cols)
    if not owned.empty:
        st.markdown("## Owned channels (not ranked)")
        st.caption("Owned channels use the company's own audience (for example its email list) and have "
                   "mostly fixed costs, so their ROAS is not comparable with paid media. They are left "
                   "out of the rankings and efficiency indices above.")
        ui.show_table(owned, cols)


# ---------------------------------------------------------------------------------------------
# Page: Campaigns
# ---------------------------------------------------------------------------------------------

def page_campaigns(filters: Filters) -> None:
    output = current_output()
    view = filtered_view(filters)
    _header("Campaigns", filters)
    table = view["campaigns"]
    if table.empty:
        ui.empty_state("Campaign analysis needs a campaign name or ID column.")
        return
    anomalies = output.analysis.anomalies
    counts = anomalies[anomalies["entity_type"] == "campaign"].groupby("entity").size() \
        if not anomalies.empty else pd.Series(dtype=int)
    table = table.assign(anomalies=table["campaign"].map(counts).fillna(0).astype(int))

    metrics = [k for k in ("conversions", "revenue", "roas", "cpl", "cac", "leads", "spend", "ctr",
                           "lead_to_conversion_rate") if k in table and table[k].notna().any()]
    c1, c2, c3 = st.columns([2, 1, 1])
    metric = c1.selectbox("Rank campaigns by", metrics, key="campaign_rank",
                          format_func=lambda k: KPI_REGISTRY[k].label)
    worst_first = c2.toggle("Show weakest first", key="campaign_worst")
    min_share = c3.number_input("Min. spend share %", 0.0, 50.0, 0.0, 0.5, key="campaign_min_share",
                                help="Hide small campaigns whose ratios are unreliable.")
    better_high = KPI_REGISTRY[metric].higher_is_better
    ascending = (better_high is False) != worst_first if better_high is not None else worst_first
    ranked = rank_campaigns(table, metric, ascending=ascending, min_spend_share=min_share)
    ui.show_chart(charts.bar_chart(ranked.head(10), "campaign", metric, KPI_REGISTRY[metric].fmt,
                                   f"Top 10 campaigns by {KPI_REGISTRY[metric].label}"
                                   + (" (weakest first)" if worst_first else ""),
                                   ascending=ascending), "No campaigns match.")
    cols = ["rank", "campaign", "channel", "spend", "leads", "cpl", "conversions", "cac", "revenue",
            "roas", "trend", "trend_change_pct", "anomalies", "first_date", "last_date"]
    ui.show_table(ranked.assign(trend=ranked["trend"].str.capitalize()) if "trend" in ranked else ranked,
                  cols, height=460)
    st.caption("Trend compares the last 4 weeks of the selected data with the 4 weeks before. "
               "Anomalies counts unusual periods found for the campaign on the full dataset.")


# ---------------------------------------------------------------------------------------------
# Page: Funnel
# ---------------------------------------------------------------------------------------------

def page_funnel(filters: Filters) -> None:
    view = filtered_view(filters)
    _header("Funnel", filters)
    funnel, by_channel = view["funnel"], view["funnel_by_channel"]
    if funnel.empty:
        ui.empty_state("A funnel needs at least two stages, for example Clicks and Leads.")
        return
    scopes = ["All channels"] + (sorted(by_channel["scope"].unique()) if not by_channel.empty else [])
    scope = st.selectbox("Funnel for", scopes, key="funnel_scope")
    data = funnel if scope == "All channels" else by_channel[by_channel["scope"] == scope]
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.show_chart(charts.funnel_chart(data, f"Funnel - {scope}"), "Not enough funnel stages.")
    with right:
        st.markdown("### Stage by stage")
        ui.show_table(data, ["label", "value", "rate_from_previous", "drop_off_pct"])
        step = biggest_drop_off(data)
        if step is not None:
            st.caption(f"Biggest drop after the click stage: only "
                       f"{format_value(step['rate_from_previous'], 'percent')} move on to "
                       f"{step['label'].lower()}.")
        if "revenue" in set(data["stage"]):
            rev = data.loc[data["stage"] == "revenue", "value"].iloc[0]
            st.caption(f"Revenue from these conversions: {format_value(rev, 'money')}.")


# ---------------------------------------------------------------------------------------------
# Page: Segments / Geography / Product
# ---------------------------------------------------------------------------------------------

SEGMENT_KIND_LABELS = {"segment": "Customer segments", "geography": "Geography", "product": "Products"}
_KIND_CAPABILITY = {"segment": "segment_analysis", "geography": "geography_analysis",
                    "product": "product_analysis"}


def page_segments(filters: Filters) -> None:
    output = current_output()
    view = filtered_view(filters)
    _header("Segments", filters, title="Segments, Geography and Products")
    df = view["df"]
    caps = output.analysis.capabilities.capabilities if output.analysis.capabilities else {}
    kinds: dict[str, list[str]] = {}
    for dim in available_dimensions(df):
        kind = SEGMENT_DIMENSIONS[dim]
        if not caps or caps.get(_KIND_CAPABILITY[kind]) is None or caps[_KIND_CAPABILITY[kind]].enabled:
            kinds.setdefault(kind, []).append(dim)
    if not kinds:
        ui.empty_state("No segment, geography or product columns were found in this data.")
        return
    tabs = st.tabs([SEGMENT_KIND_LABELS[k] for k in kinds])
    for tab, (kind, dims) in zip(tabs, kinds.items()):
        with tab:
            dim = st.selectbox("Break down by", dims, key=f"seg_dim_{kind}",
                               format_func=lambda d: FIELD_BY_NAME[d].label) if len(dims) > 1 else dims[0]
            table = segment_table(df, dim)
            metrics = [k for k in ("revenue", "roas", "cac", "aov", "conversions", "cpl", "leads",
                                   "lead_to_conversion_rate", "spend") if k in table and table[k].notna().any()]
            metric = st.selectbox("Metric", metrics, key=f"seg_metric_{kind}",
                                  format_func=lambda k: KPI_REGISTRY[k].label)
            ui.show_chart(charts.bar_chart(table, "segment", metric, KPI_REGISTRY[metric].fmt,
                                           f"{KPI_REGISTRY[metric].label} by {FIELD_BY_NAME[dim].label.lower()}",
                                           ascending=KPI_REGISTRY[metric].higher_is_better is False),
                          "No data for this breakdown.")
            ui.show_table(table, ["segment", "spend", "leads", "conversions", "revenue",
                                  "revenue_share_pct", "cac", "aov", "roas", "growth_pct", "growth_trend"])


# ---------------------------------------------------------------------------------------------
# Page: Anomalies
# ---------------------------------------------------------------------------------------------

def _anomalies_in_view(anomalies: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    if anomalies.empty:
        return anomalies
    keep = pd.Series(True, index=anomalies.index)
    if filters.date_filtered:
        keep &= (anomalies["period_end"] >= filters.start) & (anomalies["period_start"] <= filters.end)
    dims = {"campaign": filters.dimensions.get("campaign_name"), "channel": filters.dimensions.get("channel"),
            "region": filters.dimensions.get("region"), "platform": filters.dimensions.get("platform")}
    for entity_type, selected in dims.items():
        if selected:
            keep &= (anomalies["entity_type"] != entity_type) | anomalies["entity"].isin(selected)
    return anomalies[keep]


def _incidents_in_view(incidents: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    if incidents is None or incidents.empty:
        return incidents
    keep = pd.Series(True, index=incidents.index)
    if filters.date_filtered:
        keep &= (incidents["period_end"] >= filters.start) & (incidents["period_start"] <= filters.end)
    selected = filters.dimensions.get("campaign_name")
    if selected:
        keep &= incidents["campaigns"].map(lambda cs: bool(set(cs) & set(selected)))
    for entity_type in ("channel", "region", "platform"):
        chosen = filters.dimensions.get(entity_type)
        if chosen:
            keep &= (incidents["entity_type"] != entity_type) | incidents["entity"].isin(chosen)
    return incidents[keep]


def _impact_text(value, direction, label) -> str:
    if direction not in ("loss", "gain") or not value:
        return "Not estimated"
    return f"{format_value(value, 'money', compact=True)} {direction} ({label})"


def page_anomalies(filters: Filters) -> None:
    output = current_output()
    _header("Anomalies", filters)
    caps = output.analysis.capabilities
    if caps is not None and not caps.enabled("anomaly_detection"):
        ui.empty_state(caps.capabilities["anomaly_detection"].reason)
        return
    incidents = _incidents_in_view(output.analysis.incidents, filters)
    flags = _anomalies_in_view(output.analysis.anomalies, filters)
    if incidents is None or incidents.empty:
        ui.empty_state("No unusual periods were found for the current filters. That is good news: "
                       "performance moved in line with its recent history and the rest of the business.")
        return
    losses = incidents[incidents["impact_direction"] == "loss"]["impact_inr"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", format_count(len(incidents)), border=True,
              help="Related flags grouped into one business event.")
    c2.metric("High severity", format_count((incidents["severity"] == "high").sum()), border=True)
    c3.metric("Estimated money at stake", format_value(losses, "money", compact=True), border=True,
              help="Sum of the estimated losses (extra cost, lost revenue, spend without results).")
    c4.metric("Individual flags", format_count(len(flags)), border=True)
    st.caption("Incidents are ranked by estimated rupee impact. Flags on the same entity and period are "
               "one incident; campaign flags explained by a platform outage or a broader channel/region "
               "problem are attached as related effects. Impacts are estimates from the data, not "
               "accounting figures.")

    table = pd.DataFrame({
        "Rank": incidents["rank"],
        "Period": [f"{format_date(s)} - {format_date(e)}" for s, e in
                   zip(incidents["period_start"], incidents["period_end"])],
        "Entity": incidents["entity"] + " (" + incidents["entity_type"] + ")",
        "What was unusual": incidents["metrics"],
        "Estimated impact": [_impact_text(v, d, l) for v, d, l in
                             zip(incidents["impact_inr"], incidents["impact_direction"], incidents["impact_label"])],
        "Severity": incidents["severity"].str.capitalize(),
        "Assessment": incidents["sentiment"].map({"negative": "Problem", "positive": "Improvement",
                                                  "check": "Check"}),
        "Related effects": incidents["n_related"],
    })
    st.dataframe(table, hide_index=True, width="stretch", height=320)

    labels = [f"#{rk} {per} | {ent} | {what}" for rk, per, ent, what in
              zip(table["Rank"], table["Period"], table["Entity"], table["What was unusual"])]
    choice = st.selectbox("Show details for incident", range(len(labels)), format_func=lambda i: labels[i],
                          key="incident_pick")
    inc = incidents.iloc[choice]
    st.markdown(f"**{inc['headline']}**")

    def flag_table(items):
        return pd.DataFrame([{
            "Entity": f"{f['entity']} ({f['entity_type']})", "Metric": f["metric_label"],
            "Period": f"{format_date(pd.Timestamp(f['period_start']))} - {format_date(pd.Timestamp(f['period_end']))}",
            "Expected": f["expected"], "Observed": f["observed"],
            "Change": "N/A" if f["change_pct"] is None else f"{f['change_pct']:+.0f}%",
            "Assessment": {"negative": "Problem", "positive": "Improvement", "check": "Check"}[f["sentiment"]],
            "Estimated impact": _impact_text(f["impact_inr"], f.get("impact_direction", "none"), f["impact_label"]),
            "Note": f.get("note", ""),
        } for f in items])

    st.markdown("### Flags in this incident")
    st.dataframe(flag_table(inc["flags"]), hide_index=True, width="stretch")
    if inc["related"]:
        st.markdown("### Related effects")
        st.caption("Flags elsewhere that are explained by this incident (same period, same campaigns).")
        st.dataframe(flag_table(inc["related"]), hide_index=True, width="stretch")

    main = max(inc["flags"], key=lambda f: f["impact_inr"])
    metric = main["metric"]
    start, end = pd.Timestamp(main["period_start"]), pd.Timestamp(main["period_end"])
    source = output.analysis.anomalies
    expected = source[(source["entity"] == main["entity"]) & (source["metric"] == metric)
                      & (source["period_start"] == start)]["baseline"]
    expected = float(expected.iloc[0]) if len(expected) else None
    fig = None
    if main["method"] == "tracking_outage_rule":
        daily = _entity_daily(output.clean.clean_df, main["entity_type"], main["entity"], start)
        if daily is not None:
            fig = charts.anomaly_chart(daily, metric, KPI_REGISTRY[metric].fmt, start, end, expected,
                                       f"{KPI_REGISTRY[metric].label} per day - {main['entity']}")
    else:
        weekly = _entity_weekly(output.clean.clean_df, main["entity_type"], main["entity"])
        if weekly is not None:
            fig = charts.anomaly_chart(weekly, metric, KPI_REGISTRY[metric].fmt, start, end, expected,
                                       f"{KPI_REGISTRY[metric].label} per week - {main['entity']}")
    ui.show_chart(fig, "No chart is available for this incident.")

    with st.expander(f"All individual flags ({len(flags)})"):
        flat = flags.assign(
            Period=[f"{format_date(s)} - {format_date(e)}" for s, e in zip(flags["period_start"], flags["period_end"])],
            Entity=flags["entity"] + " (" + flags["entity_type"] + ")",
            Metric=flags["metric"].map(lambda m: KPI_REGISTRY[m].label if m in KPI_REGISTRY else m),
            Expected=[format_value(v, KPI_REGISTRY[m].fmt) for v, m in zip(flags["baseline"], flags["metric"])],
            Observed=[format_value(v, KPI_REGISTRY[m].fmt) for v, m in zip(flags["observed"], flags["metric"])],
            Change=flags["pct_change"].map(lambda v: "N/A" if pd.isna(v) else f"{v:+.0f}%"),
            Severity=flags["severity"].str.capitalize())
        st.dataframe(flat[["Period", "Entity", "Metric", "Expected", "Observed", "Change", "Severity"]],
                     hide_index=True, width="stretch")


def _entity_rows(df, entity_type, entity):
    column = campaign_key(df) if entity_type == "campaign" else entity_type
    if column is None or column not in df:
        return None
    return df[df[column].astype(str) == str(entity)]


def _entity_weekly(df, entity_type, entity):
    rows = _entity_rows(df, entity_type, entity)
    if rows is None or rows.empty:
        return None
    return group_kpis(rows, "week_start").rename(columns={"week_start": "period"})


def _entity_daily(df, entity_type, entity, around):
    rows = _entity_rows(df, entity_type, entity)
    if rows is None or rows.empty:
        return None
    window = rows[(rows["date"] >= around - pd.Timedelta(days=28)) & (rows["date"] <= around + pd.Timedelta(days=14))]
    return group_kpis(window, "date").rename(columns={"date": "period"})


# ---------------------------------------------------------------------------------------------
# Pages: AI Insights, Data Quality, Reports
# ---------------------------------------------------------------------------------------------

AI_BANNER = "AI-generated interpretation of verified metrics. Hypotheses require validation."


def page_ai() -> None:
    import ai.client as ai_client
    import config.settings as config_settings
    from ai.cache import budget_status, get_insights
    from ai.client import AIError
    from ai.context_builder import build_evidence
    from ai.schemas import SECTIONS
    from ai.service import evidence_lookup

    output = current_output()
    settings = config_settings.settings
    _header("AI Insights")
    state = _state()
    # Without calls (AI off or not configured) saved insights are still shown, e.g. the demo seed.
    read_only = not (settings.ai_enabled and settings.has_gemini_key and settings.gemini_model)
    if read_only:
        saved = (get_insights(output.analysis, settings, None, run_id=output.run_id, allow_call=False)
                 if settings.ai_cache_enabled else None)
        if saved is None:
            if not settings.ai_enabled:
                st.info("AI is turned off in this app. The dashboard, findings, anomalies and reports "
                        "are complete without AI.")
            else:
                st.warning("AI is switched on but not fully set up: the Gemini key or model name is "
                           "missing from the app settings. Add both, then restart the app.")
            return
        state["ai_run"] = saved

    st.info(AI_BANNER)
    pack = build_evidence(output.analysis, model=settings.gemini_model)
    st.caption(f"Gemini receives {len(pack.items)} evidence items ({format_count(pack.size)} characters) "
               "from the full dataset - no raw rows. Filters do not change the AI input.")
    with st.expander("See the evidence pack sent to Gemini"):
        st.json(pack.json_text, expanded=False)

    # Saved insights for exactly this evidence are shown straight away: no API call.
    if state.get("ai_run") is None and settings.ai_cache_enabled:
        cached = get_insights(output.analysis, settings, None, run_id=output.run_id, allow_call=False)
        if cached is not None:
            state["ai_run"] = cached
    budget = budget_status(settings, output.run_id, state.get("ai_session_calls", 0))
    session_text = (f" · this session: {budget.calls_this_session} of {budget.session_limit}"
                    if budget.session_limit is not None else "")
    st.caption(f"AI calls used today: {budget.calls_today} of {budget.day_limit} · this analysis: "
               f"{budget.calls_this_run} of {budget.run_limit}{session_text} · saved results are "
               f"{'reused automatically' if settings.ai_cache_enabled else 'not reused'}.")

    def request(force: bool) -> None:
        try:
            with st.spinner("Asking Gemini to interpret the evidence..."):
                result = get_insights(output.analysis, settings,
                                      lambda: ai_client.client_from_settings(settings),
                                      run_id=output.run_id, force=force,
                                      session_calls=state.get("ai_session_calls", 0))
            if not result.from_cache:
                state["ai_session_calls"] = state.get("ai_session_calls", 0) + result.calls_made
            state["ai_run"] = result
        except AIError as exc:
            ui.friendly_error(exc)

    notice = state.pop("ai_notice", None)
    if notice:
        st.error(f"The new call did not succeed, so the saved insights are still shown. {notice}")
    run = state.get("ai_run")
    if read_only:
        st.success(f"Loaded saved insights from {_format_timestamp(run.cached_at)}. No API call used.")
        st.caption("New AI calls are switched off in this app, so the saved insights are shown.")
    elif run is None or not run.ok:
        if budget.remaining <= 0:
            st.warning(budget.reason)
        if st.button("Generate AI insights", key="generate_ai", type="primary",
                     disabled=budget.remaining <= 0):
            request(force=False)
    else:
        if run.from_cache:
            st.success(f"Loaded saved insights from {_format_timestamp(run.cached_at)}. No API call used.")
        if not state.get("ai_confirm"):
            if st.button("Regenerate insights", key="regenerate_ai"):
                state["ai_confirm"] = True
                st.rerun()
        else:
            st.warning(f"Regenerating makes a new Gemini call and uses the budget ({budget.remaining} "
                       "call(s) left). The saved insights stay available if it fails.")
            if budget.remaining <= 0:
                st.caption(budget.reason)
            c_yes, c_no = st.columns([1, 4])
            if c_yes.button("Yes, make a new call", key="confirm_regenerate", type="primary",
                            disabled=budget.remaining <= 0):
                state["ai_confirm"] = False
                previous = run
                request(force=True)
                fresh = state.get("ai_run")
                if fresh is not None and not fresh.ok:
                    state["ai_notice"] = fresh.error.user_message
                    state["ai_run"] = previous          # keep showing the saved insights
                st.rerun()                              # redraw with the new (or kept) insights
            if c_no.button("Cancel", key="cancel_regenerate"):
                state["ai_confirm"] = False
                st.rerun()
    run = state.get("ai_run")
    if run is None:
        ui.empty_state("No AI insights yet. Press 'Generate AI insights' to make one Gemini call.")
        return
    if run.error is not None:
        ui.friendly_error(run.error)
        return

    ev = run.evaluation
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Statements kept", format_count(ev["kept"]), border=True)
    c2.metric("All figures verified", format_count(ev["verified"]), border=True)
    c3.metric("Unverified figures", format_count(ev["unverified"]), border=True)
    c4.metric("Weak", format_count(ev.get("weak", 0)), border=True,
              help="Kept but weak: restates a single finding, or leaves out the rupee impact of the "
                   "incident it relies on.")
    c5.metric("Dropped (no evidence)", format_count(ev["dropped"]), border=True)
    usage = ("saved result, no call used" if run.from_cache else
             f"{run.calls_made} call(s), about {format_count(run.input_tokens)} input and "
             f"{format_count(run.output_tokens)} output tokens")
    st.caption(f"Written by {run.model} · {usage}.")
    for key, title, label in SECTIONS:
        raw = run.insights.get(key)
        items = raw if isinstance(raw, list) else ([raw] if raw else [])
        if not items:
            continue
        st.markdown(f"## {title}")
        for i, item in enumerate(items):
            _ai_item(item, label, run.pack, evidence_lookup, f"{key}_{i}")
    if ev["dropped_items"]:
        with st.expander(f"{ev['dropped']} statement(s) removed because they cited no valid evidence"):
            for d in ev["dropped_items"]:
                st.markdown(f"- *{d['section']}*: {d['text']} ({d['reason']})")


def _ai_item(item: dict, label: str, pack, lookup, key: str) -> None:
    import html
    ids = ", ".join(item.get("evidence_ids", []))
    weak = ("<span class='mi-tag mi-tag-weak'>Weak</span>"
            if item.get("weak") else "")
    st.markdown(f"<div class='mi-finding'><span class='mi-tag'>{html.escape(label)}</span>{weak}"
                f"{html.escape(item.get('text', ''))}</div>", unsafe_allow_html=True)
    details = []
    if item.get("validation_step"):
        details.append(f"How to validate: {item['validation_step']}")
    if item.get("metric_to_watch"):
        details.append(f"Priority: {item.get('priority', '')} | Metric to watch: {item['metric_to_watch']}")
    if item.get("test_shift_pct"):
        details.append(f"Suggested test shift: {item['test_shift_pct']:.0f}% of the source budget "
                       "(a proposal, not a measured figure)")
    if item.get("at_stake_text"):
        details.append(item["at_stake_text"] + " - calculated by the app")
    for d in details:
        st.caption(d)
    for w in item.get("warnings", []):
        st.warning(w)
    with st.expander(f"Evidence {ids}"):
        for ev_item in lookup(pack, item.get("evidence_ids", [])):
            st.markdown(f"**{ev_item['id']} - {ev_item['title']}**")
            st.markdown(chr(10).join(f"- {k}: {v}" for k, v in ev_item["facts"].items()))


def page_quality() -> None:
    output = current_output()
    _header("Data Quality")
    if "raw_df" in _state():
        st.button("Edit mapping and re-run", key="edit_rerun_quality", on_click=_open_mapping_editor,
                  help="Opens Upload & Profile with your current choices filled in.")
    else:
        st.caption("To change the mapping of a reopened analysis, upload the file again.")
    summary = output.clean.quality_summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows uploaded", format_count(summary.rows_in), border=True)
    c2.metric("Rows analysed", format_count(summary.rows_out), border=True)
    c3.metric("Duplicates removed", format_count(summary.removed_duplicates), border=True)
    c4.metric("Rows with a quality flag", format_count(summary.flagged_rows), border=True)
    st.markdown("## What this means for the analysis")
    if summary.limitations:
        for text in summary.limitations:
            st.markdown(f"- {text}")
    else:
        st.success("No limitations: the data could be used as supplied.")

    st.markdown("## Change log")
    log = output.clean.quality_log_df
    if log.empty:
        ui.empty_state("No changes were needed.")
        return
    f1, f2 = st.columns(2)
    severities = f1.multiselect("Severity", ["error", "warning", "info"], key="dq_sev", placeholder="All")
    actions = f2.multiselect("Type of change", sorted(log["action"].unique()), key="dq_action",
                             placeholder="All", format_func=lambda a: a.replace("_", " "))
    shown = log
    if severities:
        shown = shown[shown["severity"].isin(severities)]
    if actions:
        shown = shown[shown["action"].isin(actions)]
    display = shown.rename(columns={"row": "Row in file (0 = first data row)", "column": "Column",
                                    "original_value": "Original value", "cleaned_value": "Cleaned value",
                                    "action": "Action", "reason": "Reason", "severity": "Severity"})
    display["Action"] = display["Action"].str.replace("_", " ")
    st.dataframe(display, hide_index=True, width="stretch", height=360)
    buffer = io.StringIO()
    log.to_csv(buffer, index=False)
    st.download_button("Download the full change log (CSV)", buffer.getvalue(),
                       file_name="data_quality_log.csv", mime="text/csv", key="dq_download")
    _ai_usage_panel()


def _ai_usage_panel() -> None:
    """Calls today, cache hits and remaining budget (free-tier awareness, SPEC 14.6)."""
    import config.settings as config_settings
    from ai.cache import budget_status, today_start_utc
    from database import repository
    from database.connection import DatabaseError
    settings = config_settings.settings
    st.markdown("## AI usage")
    if not settings.ai_enabled:
        st.caption("AI is turned off in this app, so no AI calls are made.")
        return
    try:
        usage = repository.ai_usage_since(today_start_utc())
    except DatabaseError as exc:
        ui.friendly_error(exc)
        return
    output = current_output()
    budget = budget_status(settings, output.run_id if output else None,
                           _state().get("ai_session_calls", 0))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI calls today", f"{usage['calls']} of {settings.ai_max_calls_per_day}", border=True)
    c2.metric("Failed calls today", format_count(usage["failed"]), border=True)
    c3.metric("Saved results reused today", format_count(usage["cache_hits"]), border=True,
              help="Cache hits: no API call and no budget used.")
    c4.metric("Calls left now", format_count(budget.remaining), border=True,
              help="The lower of the daily limit and the per-run limit.")
    st.caption(f"Text processed today: about {format_count(usage['input_tokens'])} tokens sent and "
               f"{format_count(usage['output_tokens'])} received. The daily and per-analysis limits "
               "are set in the app settings; Google's free-tier quotas can change. The AI day "
               "restarts at 5:30 AM India time.")


def _format_timestamp(iso: str | None) -> str:
    """India time, e.g. '27 Sep 2026, 5:34 PM'."""
    if not iso:
        return "an earlier session"
    return format_datetime_ist(iso)


def _ai_sections() -> dict | None:
    """AI output for the report: this session's result, else saved insights for the same evidence.
    Exports never trigger an AI call."""
    run = _state().get("ai_run")
    if run is not None and run.ok:
        return run.insights
    import config.settings as config_settings
    from ai.cache import get_insights
    settings = config_settings.settings
    output = current_output()
    if output is None or not settings.gemini_model:
        return None
    cached = get_insights(output.analysis, settings, None, run_id=output.run_id, allow_call=False)
    return cached.insights if cached is not None else None


def page_reports() -> None:
    output = current_output()
    _header("Reports")
    pdf_col, excel_col = st.columns(2, gap="medium")
    with pdf_col, st.container(border=True):
        ui.card_title("Executive PDF report", "A consulting-style report with KPIs, channel, campaign, "
                      "funnel and segment analysis, findings, performance concerns, data quality, "
                      "methodology and limitations.")
        if st.button("Generate PDF", key="generate_pdf", type="primary", icon=":material/picture_as_pdf:"):
            try:
                with st.spinner("Building the PDF report..."):
                    path = export_pdf(output.analysis, ai=_ai_sections())
                _state()["pdf_path"] = str(path)
            except ReportError as exc:
                ui.friendly_error(exc)
        pdf_path = _state().get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            st.success(f"Report ready: {Path(pdf_path).name}")
            st.download_button("Download PDF", Path(pdf_path).read_bytes(), file_name=Path(pdf_path).name,
                               mime="application/pdf", key="download_pdf", icon=":material/download:")
    with excel_col, st.container(border=True):
        ui.card_title("Analytical Excel workbook", "Supporting evidence: KPIs, clean data, campaign, "
                      "channel, segment, funnel and trend tables, anomalies, the data-quality log and "
                      "methodology, as real numbers you can sort and filter.")
        if st.button("Generate Excel", key="generate_excel", type="primary", icon=":material/table_view:"):
            try:
                with st.spinner("Building the Excel workbook..."):
                    path = export_workbook(output.analysis, output.clean.clean_df,
                                           output.clean.quality_log_df, ai=_ai_sections())
                _state()["excel_path"] = str(path)
            except WorkbookError as exc:
                ui.friendly_error(exc)
        excel_path = _state().get("excel_path")
        if excel_path and Path(excel_path).exists():
            st.success(f"Workbook ready: {Path(excel_path).name}")
            st.download_button("Download Excel", Path(excel_path).read_bytes(),
                               file_name=Path(excel_path).name, key="download_excel", icon=":material/download:",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if output.run_id:
        st.caption(f"This is saved as analysis #{output.run_id}; generated reports are recorded "
                   "against it.")


def not_ready() -> None:
    ui.empty_state(NOT_READY)


# ---------------------------------------------------------------------------------------------
# About (static: never reads the loaded data, the database or AI settings)
# ---------------------------------------------------------------------------------------------

ABOUT_FEATURES = [
    ("Upload any marketing export", "CSV or Excel; columns are recognised automatically and you "
     "can change any mapping."),
    ("Clean and check the data", "duplicates, bad dates and impossible values are fixed or "
     "flagged, and every change is logged."),
    ("Calculate the KPIs", "CTR, CPC, CPL, CAC, ROAS, AOV and funnel rates, always recomputed "
     "from totals."),
    ("Find what changed", "trends, channel and campaign rankings, segments and weekly anomaly "
     "detection."),
    ("Explain it in plain language", "optional AI insights that cite the numbers they are based on."),
    ("Share the results", "a PDF report and an analytical Excel workbook with the same numbers "
     "as the dashboard."),
]
ABOUT_STACK = ("Python · pandas · Streamlit · Plotly · SQLite · Google Gemini · ReportLab · "
               "matplotlib · XlsxWriter")


def page_about() -> None:
    _header("About")
    what, who = st.columns([3, 2], gap="large")
    with what, st.container(border=True):
        ui.card_title("What this platform does")
        items = "".join(f"<li><b>{html.escape(title)}</b> - {html.escape(text)}</li>"
                        for title, text in ABOUT_FEATURES)
        st.markdown(f"<ul class='mi-about-list'>{items}</ul>", unsafe_allow_html=True)
        st.caption(f"Built with {ABOUT_STACK}. Every number is calculated in Python; the AI only "
                   "interprets a summary of those numbers.")
    with who, st.container(border=True):
        ui.creator_card()
