"""The dashboard pages. Every number shown comes from the AnalysisResult (full dataset) or from
the same analytics functions re-run on the filtered rows."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.campaign import campaign_table, rank_campaigns
from analytics.channel import channel_table
from analytics.common import campaign_key
from analytics.funnel import biggest_drop_off, funnel_by_channel, funnel_table
from analytics.kpis import KPI_REGISTRY, compute_kpis, group_kpis, period_comparison
from analytics.segments import SEGMENT_DIMENSIONS, available_dimensions, segment_table
from analytics.trends import time_series
from config.fields import FIELD_BY_NAME
from dashboard import charts, components as ui
from dashboard.filters import (DATE_PRESETS, Filters, apply_dimension_filters, apply_filters,
                               available_filters, describe, preset_range)
from dashboard.pipeline import (DEFAULT_SAMPLE, SAMPLE_DATASETS, load_raw, prepare,
                                run_pipeline, sample_path)
from ingestion.loader import IngestionError
from processing.cleaner import CleaningError
from reports.pdf_report import ReportError, export_pdf
from utils.formatting import format_count, format_date, format_value

PAGES = ["Upload & Profile", "Executive Overview", "Performance Trends", "Channels", "Campaigns",
         "Funnel", "Segments", "Anomalies", "AI Insights", "Data Quality", "Reports"]
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
    for key in ("output", "prep", "prep_key", "view", "view_key", "pdf_path"):
        s.pop(key, None)


def current_prep():
    """Profile + mapping for the loaded file, recomputed only when the file or the user's
    mapping choices change."""
    s = _state()
    if "raw_df" not in s:
        return None
    key = (s["report"].file_hash, tuple(sorted(s.get("overrides", {}).items(), key=str)))
    if s.get("prep_key") != key:
        s["prep"] = prepare(s["raw_df"], s["report"], s.get("overrides") or None)
        s["prep_key"] = key
    return s["prep"]


def current_output():
    return _state().get("output")


# ---------------------------------------------------------------------------------------------
# Sidebar: navigation + filters
# ---------------------------------------------------------------------------------------------

def sidebar() -> tuple[str, Filters | None]:
    with st.sidebar:
        st.markdown("### Marketing Intelligence Platform")
        page = st.radio("Section", PAGES, key="page", label_visibility="collapsed")
        output = current_output()
        if output is None:
            return page, None
        df = output.clean.clean_df
        st.divider()
        st.markdown("**Filters**")
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
                   + (f" | Run #{output.run_id}" if output.run_id else ""))
        return page, filters


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
    ui.page_header("Upload & Profile", "Upload a marketing export (CSV or Excel) or load a sample. "
                   "The file is profiled, cleaned and analysed automatically.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        uploaded = st.file_uploader("Marketing data file", type=["csv", "xlsx"], key="uploader",
                                    help="One row per day and campaign works best. Maximum size is "
                                         "set by MAX_UPLOAD_MB (default 50 MB).")
        if uploaded is not None and _state().get("uploaded_id") != uploaded.file_id:
            _state()["uploaded_id"] = uploaded.file_id
            try:
                raw_df, report = load_raw(uploaded)
                _reset_for_new_file(raw_df, report, uploaded.name)
            except IngestionError as exc:
                ui.friendly_error(exc)
    with right:
        label = st.selectbox("Or use a sample dataset", list(SAMPLE_DATASETS),
                             index=list(SAMPLE_DATASETS).index(DEFAULT_SAMPLE), key="sample_choice")
        if st.button("Load sample dataset", key="load_sample", type="secondary"):
            try:
                raw_df, report = load_raw(sample_path(label))
                _reset_for_new_file(raw_df, report, label)
            except IngestionError as exc:
                ui.friendly_error(exc)

    prep = current_prep()
    if prep is None:
        ui.empty_state("No data loaded yet. Upload a file or load a sample dataset to begin.")
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
    run = st.button("Run analysis", type="primary", key="run_analysis",
                    disabled=not prep.validation.can_analyse)
    if run:
        try:
            with st.status("Running analysis...", expanded=True) as status:
                output = run_pipeline(prep, progress=lambda msg: status.write(msg))
                status.update(label="Analysis complete", state="complete", expanded=False)
            _state()["output"] = output
            for key in ("view", "view_key", "pdf_path"):
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
        if output.is_reupload:
            st.info("This exact file was uploaded before; a new analysis run was recorded for it.")
        if output.save_error:
            st.warning(f"The results are shown but were not saved: {output.save_error}")


def _mapping_editor(prep) -> None:
    table = prep.mapping.to_frame().rename(columns={
        "column": "Column in file", "maps_to": "Used as", "status": "Confidence",
        "score": "Match score", "suggestions": "Suggestions", "note": "Note"})
    table["Confidence"] = table["Confidence"].map({
        "confirmed": "Confirmed", "high_confidence": "High confidence (inferred)",
        "uncertain": "Uncertain - please confirm", "unmapped": "Not recognised",
        "derived": "Ratio column (recalculated)", "ignored": "Not used"}).fillna(table["Confidence"])
    st.dataframe(table, hide_index=True, width="stretch")

    overrides = _state().setdefault("overrides", {})
    to_confirm = [m for m in prep.mapping.columns if m.status == "uncertain" or m.source == "user"]
    if not to_confirm:
        st.caption("Every column that matters was recognised with confidence.")
        return
    st.markdown("**Confirm what these columns mean**")
    st.caption("Business-critical fields (Date, Spend, Revenue, Leads, Conversions) are never "
               "guessed; analysis starts once you confirm them.")
    for m in to_confirm:
        options = [f for f, _ in m.candidates] if m.candidates else []
        if m.source == "user" and m.field and m.field not in options:
            options = [m.field] + options
        choices = ["(choose)"] + options + ["Not used"]
        current = overrides.get(m.column, "(choose)")
        current = "Not used" if m.column in overrides and overrides[m.column] is None else current
        picked = st.selectbox(
            f"'{m.column}' contains", choices, index=choices.index(current) if current in choices else 0,
            format_func=lambda f: FIELD_BY_NAME[f].label if f in FIELD_BY_NAME else f,
            key=f"confirm_{m.column}")
        if picked == "(choose)":
            overrides.pop(m.column, None)
        else:
            overrides[m.column] = None if picked == "Not used" else picked
    if st.button("Apply choices", key="apply_mapping"):
        _state().pop("prep_key", None)
        st.rerun()


# ---------------------------------------------------------------------------------------------
# Page: Executive Overview
# ---------------------------------------------------------------------------------------------

def page_overview(filters: Filters) -> None:
    output = current_output()
    view = filtered_view(filters)
    ui.page_header("Executive Overview", _filter_caption(filters))
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
    ui.page_header("Performance Trends", _filter_caption(filters))
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
    ui.page_header("Channels", _filter_caption(filters))
    ch = view["channels"]
    if ch.empty:
        ui.empty_state("Channel analysis needs a channel or platform column.")
        return
    metrics = [k for k in ("roas", "cpl", "cac", "ctr", "cpc", "lead_to_conversion_rate",
                           "revenue", "spend", "leads", "conversions") if k in ch and ch[k].notna().any()]
    left, right = st.columns(2, gap="large")
    with left:
        metric = st.selectbox("Compare channels on", metrics, key="channel_metric",
                              format_func=lambda k: KPI_REGISTRY[k].label)
        overall = view["kpis"].get(metric)
        better_high = KPI_REGISTRY[metric].higher_is_better
        ui.show_chart(charts.bar_chart(ch, "channel", metric, KPI_REGISTRY[metric].fmt,
                                       f"{KPI_REGISTRY[metric].label} by channel",
                                       ascending=better_high is False,
                                       reference=overall.value if overall and metric not in
                                       ("revenue", "spend", "leads", "conversions") else None),
                      "No data for this metric.")
    with right:
        index_cols = [c for c in ch.columns if c.endswith("_index")]
        if index_cols:
            idx = st.selectbox("Efficiency index (average = 100)", index_cols, key="channel_index",
                               format_func=lambda c: KPI_REGISTRY[c.removesuffix("_index")].label)
            base = idx.removesuffix("_index")
            ui.show_chart(charts.index_chart(ch, "channel", idx, f"{KPI_REGISTRY[base].label} index vs "
                                             "average", lower_is_better=KPI_REGISTRY[base].higher_is_better is False),
                          "No index available.")
    st.markdown("## Channel table")
    cols = ["channel", "spend", "spend_share_pct", "impressions", "clicks", "ctr", "cpc", "leads", "cpl",
            "conversions", "lead_to_conversion_rate", "cac", "revenue", "revenue_share_pct", "roas", "roi"]
    ui.show_table(ch.sort_values("spend", ascending=False) if "spend" in ch else ch, cols)


# ---------------------------------------------------------------------------------------------
# Page: Campaigns
# ---------------------------------------------------------------------------------------------

def page_campaigns(filters: Filters) -> None:
    output = current_output()
    view = filtered_view(filters)
    ui.page_header("Campaigns", _filter_caption(filters))
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
    ui.page_header("Funnel", _filter_caption(filters))
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
    ui.page_header("Segments, Geography and Products", _filter_caption(filters))
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


def page_anomalies(filters: Filters) -> None:
    output = current_output()
    ui.page_header("Anomalies", _filter_caption(filters))
    caps = output.analysis.capabilities
    if caps is not None and not caps.enabled("anomaly_detection"):
        ui.empty_state(caps.capabilities["anomaly_detection"].reason)
        return
    found = _anomalies_in_view(output.analysis.anomalies, filters)
    if found.empty:
        ui.empty_state("No unusual periods were found for the current filters. That is good news: "
                       "performance moved in line with its recent history and the rest of the business.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Unusual periods", format_count(len(found)), border=True)
    c2.metric("High severity", format_count((found["severity"] == "high").sum()), border=True)
    c3.metric("Positive changes", format_count((found["sentiment"] == "positive").sum()), border=True)
    st.caption("Each week is compared with the median of the previous 8 weeks, after removing the "
               "change that all campaigns shared that week (seasonality). A separate daily rule "
               "looks for tracking outages.")

    table = found.assign(
        Period=[f"{format_date(s)} - {format_date(e)}" for s, e in zip(found["period_start"], found["period_end"])],
        Entity=found["entity"] + " (" + found["entity_type"] + ")",
        Metric=found["metric"].map(lambda m: KPI_REGISTRY[m].label if m in KPI_REGISTRY else m),
        Expected=[format_value(v, KPI_REGISTRY[m].fmt) for v, m in zip(found["baseline"], found["metric"])],
        Observed=[format_value(v, KPI_REGISTRY[m].fmt) for v, m in zip(found["observed"], found["metric"])],
        Change=found["pct_change"].map(lambda v: "N/A" if pd.isna(v) else f"{v:+.0f}%"),
        Severity=found["severity"].str.capitalize(),
        Assessment=found["sentiment"].map({"negative": "Problem", "positive": "Improvement",
                                           "check": "Check"}))
    st.dataframe(table[["Period", "Entity", "Metric", "Expected", "Observed", "Change", "Severity",
                        "Assessment"]], hide_index=True, width="stretch", height=320)

    labels = [f"{r.Period} | {r.Entity} | {r.Metric} {r.Change}" for r in table.itertuples()]
    choice = st.selectbox("Show details for", range(len(labels)), format_func=lambda i: labels[i],
                          key="anomaly_pick")
    row = found.iloc[choice]
    st.markdown(f"**{row['description']}**")
    weekly = _entity_weekly(output.clean.clean_df, row["entity_type"], row["entity"])
    metric = row["metric"]
    fig = None
    if weekly is not None and row["grain"] == "week":
        fig = charts.anomaly_chart(weekly, metric, KPI_REGISTRY[metric].fmt, row["period_start"],
                                   row["period_end"] + pd.Timedelta(days=0), row["baseline"],
                                   f"{KPI_REGISTRY[metric].label} per week - {row['entity']}")
    elif row["grain"] == "day":
        daily = _entity_daily(output.clean.clean_df, row["entity_type"], row["entity"], row["period_start"])
        if daily is not None:
            fig = charts.anomaly_chart(daily, metric, KPI_REGISTRY[metric].fmt, row["period_start"],
                                       row["period_end"], row["baseline"],
                                       f"{KPI_REGISTRY[metric].label} per day - {row['entity']}")
    ui.show_chart(fig, "No chart is available for this item.")


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

def page_ai() -> None:
    ui.page_header("AI Insights", "Interpretation of the verified findings by Google Gemini, clearly "
                   "labelled as AI-generated.")
    ui.empty_state("AI insights are not enabled yet. The dashboard, findings and anomalies above are "
                   "calculated without AI and are complete on their own.")


def page_quality() -> None:
    output = current_output()
    ui.page_header("Data Quality", "What was fixed, flagged or excluded while cleaning, and what it "
                   "means for the numbers.")
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


def page_reports() -> None:
    output = current_output()
    ui.page_header("Reports", "Executive PDF and analytical Excel workbook for the full dataset "
                   "(dashboard filters do not apply to exports).")
    st.markdown("## Executive PDF report")
    st.caption("A consulting-style report with KPIs, channel, campaign, funnel and segment analysis, "
               "findings, performance concerns, data quality, methodology and limitations.")
    if st.button("Generate PDF", key="generate_pdf", type="primary"):
        try:
            with st.spinner("Building the PDF report..."):
                path = export_pdf(output.analysis)
            _state()["pdf_path"] = str(path)
        except ReportError as exc:
            ui.friendly_error(exc)
    pdf_path = _state().get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        st.success(f"Report ready: {Path(pdf_path).name}")
        st.download_button("Download PDF", Path(pdf_path).read_bytes(), file_name=Path(pdf_path).name,
                           mime="application/pdf", key="download_pdf")
    st.markdown("## Excel workbook")
    ui.empty_state("The Excel workbook export is not available yet.")
    if output.run_id:
        st.caption(f"This analysis is saved as run #{output.run_id}; generated reports are recorded "
                   "against it.")


def not_ready() -> None:
    ui.empty_state(NOT_READY)
