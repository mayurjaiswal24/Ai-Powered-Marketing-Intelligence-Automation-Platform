"""Analytical Excel workbook, generated from the AnalysisResult (supporting evidence for the
dashboard and the PDF).

Cells hold REAL numbers (not text) with Excel number formats, so users can sort, filter and
calculate. Money uses a custom Indian-grouping format (₹12,34,567); percentages keep the same
0-100 values as the dashboard with a "%" format; ROAS shows as 3.45x. Sheets are created only
when the data supports them.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analytics.kpis import KPI_REGISTRY, RATIO_KPIS
from config import settings as cfg
from config.fields import FIELD_BY_NAME
from config.settings import PROJECT_ROOT, settings
from dashboard import theme
from dashboard.tables import column_format, column_label
from reports import export_filename
from utils.formatting import (as_sentence, format_count, format_date, format_datetime_ist,
                              pct_decimals, title_case)

EXPORTS_DIR = (Path(settings.exports_dir) if Path(settings.exports_dir).is_absolute()
               else PROJECT_ROOT / settings.exports_dir)
WORKBOOK_TITLE = "Marketing Intelligence Workbook"

# Excel custom formats. Indian grouping needs two conditional sections (crore, lakh) plus a
# default; Excel allows exactly that many, so these formats are for non-negative values.
INR_0 = '[>=10000000]"₹"##\\,##\\,##\\,##0;[>=100000]"₹"##\\,##\\,##0;"₹"##,##0'
INR_2 = '[>=10000000]"₹"##\\,##\\,##\\,##0.00;[>=100000]"₹"##\\,##\\,##0.00;"₹"##,##0.00'
COUNT = '[>=10000000]##\\,##\\,##\\,##0;[>=100000]##\\,##\\,##0;##,##0'
PERCENT = '0.0"%"'          # values are already on the 0-100 scale, exactly as in the dashboard
SIGNED_PERCENT = '+0.0"%";-0.0"%";0.0"%"'
# Small non-zero percentages get more decimals (same rule as utils.formatting.pct_decimals), so
# 0.04% never shows as 0.0%. The format is chosen per cell; the stored number is unchanged.
PERCENT_BY_DECIMALS = {2: '0.00"%"', 3: '0.000"%"'}
SIGNED_PERCENT_BY_DECIMALS = {2: '+0.00"%";-0.00"%";0.00"%"', 3: '+0.000"%";-0.000"%";0.000"%"'}
RATIO = '0.00"x"'
INDEX = "0"
DATE = "d mmm yyyy"

MAX_COL_WIDTH = 48


class WorkbookError(Exception):
    """Workbook could not be produced. `user_message` is safe to show in the UI."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def _is_blank(value) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        return isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    except TypeError:
        return False


class _Book:
    def __init__(self, target):
        self.writer = pd.ExcelWriter(target, engine="xlsxwriter",
                                     engine_kwargs={"options": {"remove_timezone": True}})
        self.wb = self.writer.book
        base = {"font_name": "Calibri", "font_size": 10, "valign": "top"}
        self.fmt = {
            "title": self.wb.add_format({"bold": True, "font_size": 16, "font_color": theme.INK}),
            "subtitle": self.wb.add_format({"font_size": 10, "font_color": theme.INK_SECONDARY}),
            "label": self.wb.add_format({**base, "font_color": theme.INK_SECONDARY}),
            "section": self.wb.add_format({"bold": True, "font_size": 12, "font_color": theme.INK}),
            "header": self.wb.add_format({**base, "bold": True, "font_color": "#ffffff",
                                          "bg_color": theme.PRIMARY, "text_wrap": True,
                                          "border": 1, "border_color": theme.PRIMARY}),
            "text": self.wb.add_format(base),
            "wrap": self.wb.add_format({**base, "text_wrap": True}),
            "money0": self.wb.add_format({**base, "num_format": INR_0}),
            "money2": self.wb.add_format({**base, "num_format": INR_2}),
            "count": self.wb.add_format({**base, "num_format": COUNT}),
            "percent": self.wb.add_format({**base, "num_format": PERCENT}),
            "signed_percent": self.wb.add_format({**base, "num_format": SIGNED_PERCENT}),
            **{f"percent_{d}": self.wb.add_format({**base, "num_format": f})
               for d, f in PERCENT_BY_DECIMALS.items()},
            **{f"signed_percent_{d}": self.wb.add_format({**base, "num_format": f})
               for d, f in SIGNED_PERCENT_BY_DECIMALS.items()},
            "ratio": self.wb.add_format({**base, "num_format": RATIO}),
            "index": self.wb.add_format({**base, "num_format": INDEX}),
            "date": self.wb.add_format({**base, "num_format": DATE}),
            "number": self.wb.add_format({**base, "num_format": "0.00"}),
        }

    def sheet(self, name: str):
        ws = self.wb.add_worksheet(name)
        ws.hide_gridlines(2)
        return ws

    def number_format(self, kind: str, values: pd.Series | None = None):
        if kind == "money":
            small = values is not None and values.dropna().abs().median() < 100 if values is not None \
                and values.notna().any() else False
            return self.fmt["money2" if small else "money0"]
        return self.fmt.get(kind, self.fmt["text"])

    def _small_percent(self, fmt, value: float):
        """Swap a percent format for one with more decimals when the value is small."""
        for kind in ("percent", "signed_percent"):
            if fmt is self.fmt[kind]:
                decimals = pct_decimals(value)
                return fmt if decimals == 1 else self.fmt[f"{kind}_{decimals}"]
        return fmt

    def write_value(self, ws, row: int, col: int, value, fmt) -> None:
        if _is_blank(value):
            ws.write_blank(row, col, None, fmt)
        elif isinstance(value, (pd.Timestamp, datetime)):
            ws.write_datetime(row, col, pd.Timestamp(value).to_pydatetime(), self.fmt["date"])
        elif isinstance(value, bool):
            ws.write_boolean(row, col, value, fmt)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            ws.write_number(row, col, float(value), self._small_percent(fmt, float(value)))
        elif hasattr(value, "item"):                           # numpy scalars
            self.write_value(ws, row, col, value.item(), fmt)
        else:
            ws.write_string(row, col, str(value), fmt)

    def table(self, ws, df: pd.DataFrame, start_row: int = 0, columns: list[str] | None = None,
              formats: dict[str, str] | None = None, freeze: bool = True, filter_: bool = True,
              labels: bool = True) -> int:
        """Write a formatted table; returns the next free row."""
        columns = [c for c in (columns or list(df.columns)) if c in df]
        if not columns:
            return start_row
        data = df[columns].reset_index(drop=True)
        formats = formats or {}
        for j, col in enumerate(columns):
            header = column_label(col) if labels else col
            ws.write_string(start_row, j, header, self.fmt["header"])
            kind = formats.get(col) or column_format(col, data[col])
            if kind == "percent" and col.endswith(("_change_pct", "growth_pct", "_wow_pct", "_mom_pct")):
                kind = "signed_percent"
            cell_fmt = self.number_format(kind, pd.to_numeric(data[col], errors="coerce")
                                          if kind == "money" else None)
            if kind == "text":
                cell_fmt = self.fmt["text"]
            values = data[col].tolist()
            for i, value in enumerate(values, start=start_row + 1):
                self.write_value(ws, i, j, value, cell_fmt)
            width = max([len(str(header))] + [len(_display(v)) for v in values[:300]]) + 2
            ws.set_column(j, j, min(max(width, 9), MAX_COL_WIDTH))
        end = start_row + len(data)
        if filter_ and len(data):
            ws.autofilter(start_row, 0, end, len(columns) - 1)
        if freeze:
            ws.freeze_panes(start_row + 1, 1)
        return end + 2

    def close(self) -> None:
        self.writer.close()


def _display(value) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, (pd.Timestamp, datetime)):
        return "30 Sep 2026"
    return str(value)


# ---------------------------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------------------------

def _kpi_format_name(key: str, value) -> str:
    fmt = KPI_REGISTRY[key].fmt
    if fmt == "money":
        return "money2" if value is not None and abs(value) < 100 else "money0"
    return fmt


def _executive_kpis(book: _Book, result, generated: str) -> None:
    ws = book.sheet("Executive_KPIs")
    meta = result.metadata
    ws.write_string(0, 0, WORKBOOK_TITLE, book.fmt["title"])
    ws.write_string(1, 0, "Supporting evidence for the Marketing Intelligence dashboard and "
                    "executive PDF. All figures cover the full dataset.", book.fmt["subtitle"])
    ws.write_string(2, 0, cfg.REPORT_SIGNATURE, book.fmt["subtitle"])
    info = [("Dataset", str(meta.get("dataset_name") or "N/A")),
            ("Period Covered", f"{format_date(meta.get('date_min'))} to {format_date(meta.get('date_max'))}"),
            ("Rows Analysed", format_count(meta.get("rows"))),
            ("Analysis", f"#{meta['run_id']}" if meta.get("run_id") else "Not saved"),
            ("Generated", generated)]
    for i, (label, value) in enumerate(info, start=3):
        ws.write_string(i, 0, label, book.fmt["label"])
        ws.write_string(i, 1, value, book.fmt["text"])

    start = 10
    ws.write_string(start - 1, 0, "Key Performance Indicators (Verified Metrics)", book.fmt["section"])
    headers = ["KPI", "Value", "How It Is Calculated", "Note"]
    for j, h in enumerate(headers):
        ws.write_string(start, j, h, book.fmt["header"])
    row = start + 1
    for k in result.kpis.values():
        if not k.available:
            continue
        ws.write_string(row, 0, title_case(k.label), book.fmt["text"])
        book.write_value(ws, row, 1, k.value, book.fmt[_kpi_format_name(k.key, k.value)])
        ws.write_string(row, 2, k.formula_text, book.fmt["wrap"])
        ws.write_string(row, 3, k.note or "", book.fmt["wrap"])
        row += 1
    unavailable = [k for k in result.kpis.values() if not k.available]
    if unavailable:
        row += 1
        ws.write_string(row, 0, "Not Available for This Dataset", book.fmt["section"])
        for k in unavailable:
            row += 1
            ws.write_string(row, 0, title_case(k.label), book.fmt["text"])
            ws.write_string(row, 2, k.reason_unavailable, book.fmt["wrap"])
    ws.set_column(0, 0, 26)
    ws.set_column(1, 1, 18)
    ws.set_column(2, 2, 46)
    ws.set_column(3, 3, 52)
    ws.freeze_panes(start + 1, 0)


def _kpi_analysis(book: _Book, result) -> None:
    ws = book.sheet("KPI_Analysis")
    k = result.kpi_table.copy()
    k["available"] = k["available"].map({True: "Yes", False: "No"})
    k["note"] = k["note"].where(k["available"] == "Yes", k["reason_unavailable"])
    ws.write_string(0, 0, "Every KPI with the numbers behind it (audit trail). Ratios = numerator / "
                    "denominator of summed totals.", book.fmt["subtitle"])
    k["kpi_name"] = k["label"].map(title_case)
    table = k[["kpi_name", "value", "numerator", "denominator", "available", "formula", "note"]]
    # One number format per row is needed for "value" (money, %, x...), so write it by hand.
    end = book.table(ws, table.drop(columns=["value"]), start_row=2,
                     formats={"numerator": "number", "denominator": "number", "kpi_name": "text",
                              "available": "text", "formula": "text", "note": "text"})
    ws.write_string(2, len(table.columns) - 1, "Value", book.fmt["header"])
    for i, row in enumerate(k.itertuples(), start=3):
        fmt = book.fmt[_kpi_format_name(row.kpi, row.value)] if row.kpi in KPI_REGISTRY else book.fmt["number"]
        book.write_value(ws, i, len(table.columns) - 1, row.value, fmt)
    ws.set_column(len(table.columns) - 1, len(table.columns) - 1, 18)
    return end


def _clean_data(book: _Book, clean_df: pd.DataFrame) -> None:
    ws = book.sheet("Clean_Data")
    book.table(ws, clean_df, labels=False)


def _simple_sheet(book: _Book, name: str, df: pd.DataFrame, note: str) -> None:
    if df is None or df.empty:
        return
    ws = book.sheet(name)
    ws.write_string(0, 0, note, book.fmt["subtitle"])
    book.table(ws, df, start_row=2)


def _segments(book: _Book, result) -> None:
    parts = [t for t in result.segments.values() if not t.empty]
    if not parts:
        return
    table = pd.concat(parts, ignore_index=True)
    _simple_sheet(book, "Segment_Analysis", table,
                  "One block per dimension (customer segment, region, product ...). Growth = last 4 "
                  "weeks vs the 4 weeks before.")


def _funnel(book: _Book, result) -> None:
    parts = [f for f in (result.funnel, result.funnel_by_channel) if not f.empty]
    if not parts:
        return
    table = pd.concat(parts, ignore_index=True).rename(columns={"scope": "channel_scope"})
    _simple_sheet(book, "Funnel_Analysis", table,
                  "Stage totals and stage-to-stage conversion. A Channel Scope of 'all' means all "
                  "channels together.")


def _trends(book: _Book, result) -> None:
    parts = []
    for grain, t in result.trends.items():
        t = t.copy()
        if grain == "month":
            t["period"] = pd.to_datetime(t["period"] + "-01")
        parts.append(t.rename(columns={"period": "period_start"}))
    if not parts:
        return
    table = pd.concat(parts, ignore_index=True)
    _simple_sheet(book, "Trend_Analysis", table,
                  "Daily, weekly (starting on Monday) and monthly totals and KPIs. Days is the number "
                  "of days with data in the period (partial periods have fewer). Filter on the Grain "
                  "column.")


def _channels(book: _Book, result) -> None:
    ch = result.channels
    if ch is None or ch.empty:
        return
    from analytics.channel import owned_channels, paid_channels
    paid, owned = paid_channels(ch), owned_channels(ch)
    ws = book.sheet("Channel_Analysis")
    ws.write_string(0, 0, "Paid channels. Each Index column = paid channel value / paid-media average "
                    "× 100 (100 = average). Owned channels are listed separately below and not ranked.",
                    book.fmt["subtitle"])
    end = book.table(ws, paid, start_row=2)
    if not owned.empty:
        ws.write_string(end, 0, "Owned Channels (Not Ranked: Own Audience, Mostly Fixed Costs, ROAS Not "
                        "Comparable with Paid Media)", book.fmt["section"])
        book.table(ws, owned.drop(columns=[c for c in owned.columns if c.endswith("_index")]),
                   start_row=end + 1, freeze=False, filter_=False)


def _incidents(book: _Book, result) -> None:
    inc = getattr(result, "incidents", None)
    if inc is None or inc.empty:
        return

    def summary(flags):
        return "; ".join(f"{f['entity']} {f['metric_label']} "
                         + ("" if f["change_pct"] is None else f"{f['change_pct']:+.0f}%")
                         + (" (not an improvement)" if f.get("note") else "") for f in flags)

    table = pd.DataFrame({
        "rank": inc["rank"], "incident_id": inc["incident_id"], "entity_type": inc["entity_type"],
        "entity": inc["entity"], "period_start": inc["period_start"], "period_end": inc["period_end"],
        "metrics": inc["metrics"], "estimated_impact_inr": inc["impact_inr"],
        "impact_direction": inc["impact_direction"], "impact_basis": inc["impact_label"],
        "severity": inc["severity"], "assessment": inc["sentiment"],
        "flags": inc["flags"].map(summary), "related_effects": inc["related"].map(summary),
    })
    _simple_sheet(book, "Incidents", table,
                  "Anomaly flags grouped into business incidents, ranked by their estimated ₹ impact "
                  "(an estimate from the data, not an accounting figure). Related effects are flags on "
                  "other campaigns or channels explained by the same incident. Individual flags are on "
                  "the Anomalies sheet.")


def _anomalies(book: _Book, result) -> None:
    a = result.anomalies
    if a.empty:
        return
    table = a.drop(columns=["details_json"], errors="ignore").rename(
        columns={"baseline": "expected", "pct_change": "change_pct"})
    _simple_sheet(book, "Anomalies", table,
                  "Unusual periods (weekly robust score or daily tracking-outage rule). Expected is "
                  "the level expected from recent weeks and the rest of the business. Values use the "
                  "metric's own unit (see the Metric column).")


def _ai(book: _Book, ai: dict | None) -> None:
    if not ai:
        return
    from ai.schemas import SECTIONS
    rows = []
    for key, title, label in SECTIONS:
        raw = ai.get(key)
        for item in raw if isinstance(raw, list) else ([raw] if raw else []):
            rows.append({"section": title_case(title), "type": label, "text": item.get("text", ""),
                         "evidence_ids": ", ".join(item.get("evidence_ids", [])),
                         "validation_step": item.get("validation_step", ""),
                         "priority": item.get("priority", ""),
                         "metric_to_watch": item.get("metric_to_watch", ""),
                         "test_shift_pct": item.get("test_shift_pct"),
                         "rupees_at_stake": item.get("rupees_at_stake") if key == "recommendations" else None,
                         "at_stake_basis": item.get("at_stake_text", "") if key == "recommendations" else "",
                         "check": item.get("status", ""),
                         "quality": "weak: " + "; ".join(item.get("weak_reasons", [])) if item.get("weak") else "ok"})
    if rows:
        _simple_sheet(book, "AI_Insights", pd.DataFrame(rows),
                      "AI-GENERATED content (Google Gemini): an interpretation of the verified evidence. "
                      "Hypotheses need to be validated before you act on them. Check shows whether every "
                      "figure matched the cited evidence.")


def _data_quality(book: _Book, result, log: pd.DataFrame | None) -> None:
    q = result.quality_summary
    if q is None:
        return
    ws = book.sheet("Data_Quality")
    ws.write_string(0, 0, "Data Quality", book.fmt["title"])
    items = [("Rows in Uploaded File", q.rows_in), ("Rows Analysed", q.rows_out),
             ("Duplicate Rows Removed", q.removed_duplicates),
             ("Summary Rows Removed", q.removed_summary_rows), ("Rows with a Quality Flag", q.flagged_rows)]
    for i, (label, value) in enumerate(items, start=2):
        ws.write_string(i, 0, label, book.fmt["label"])
        ws.write_number(i, 1, value, book.fmt["count"])
    row = 2 + len(items) + 1
    ws.write_string(row, 0, "What This Means for the Analysis", book.fmt["section"])
    for text in q.limitations or ["No limitations: the data could be used as supplied."]:
        row += 1
        ws.write_string(row, 0, f"- {text}", book.fmt["text"])
    row += 2
    if log is not None and not log.empty:
        ws.write_string(row, 0, "Change Log (Data Row 1 = the First Row under the Column Headings)",
                        book.fmt["section"])
        table = log.rename(columns={"row": "data_row", "column": "column_name"})
        table["data_row"] = table["data_row"] + 1
        table["column_name"] = table["column_name"].map(
            lambda c: title_case(FIELD_BY_NAME[c].label) if c in FIELD_BY_NAME else c)
        table["action"] = table["action"].str.replace("_", " ").str.capitalize()
        table["reason"] = table["reason"].map(as_sentence)
        table["severity"] = table["severity"].str.capitalize()
        book.table(ws, table, start_row=row + 1, formats={"data_row": "index"}, freeze=False,
                   labels=True)
    ws.set_column(0, 0, 30)
    ws.set_column(1, 1, 16)


def _methodology(book: _Book, result) -> None:
    ws = book.sheet("Methodology")
    ws.write_string(0, 0, "Methodology", book.fmt["title"])
    rows = [{"kpi": title_case(KPI_REGISTRY[k].label), "formula": KPI_REGISTRY[k].formula_text,
             "meaning": KPI_REGISTRY[k].description, "used_in_this_dataset":
             "Yes" if result.kpis.get(k) is not None and result.kpis[k].available else "No"}
            for k in RATIO_KPIS]
    end = book.table(ws, pd.DataFrame(rows), start_row=2, freeze=False, filter_=False,
                     formats={c: "text" for c in ("kpi", "formula", "meaning", "used_in_this_dataset")})
    rules = [
        "Additive metrics (spend, impressions, clicks, leads, conversions, revenue) are summed; ratios "
        "are recalculated from those sums, never averaged across rows.",
        "A ratio uses only rows where both inputs are known; missing values are never treated as zero.",
        "Reach is not additive and is never summed.",
        "CAC = Spend / Customers when a Customers field exists, otherwise Spend / Conversions.",
        "ROI is only calculated from a gross profit or margin column; no margin is ever assumed.",
        f"Anomalies: weekly robust score vs the median of the previous {cfg.ANOMALY_BASELINE_WEEKS} full "
        f"weeks, with seasonality removed (typical change across entities). Flag if score >= "
        f"{cfg.ANOMALY_SCORE_THRESHOLD} and change >= {cfg.ANOMALY_MIN_PCT_CHANGE:.0f}%, with minimum "
        "volume and a small-numbers chance test. Daily rule for tracking outages.",
        "Money: INR. Percentages are shown on a 0-100 scale (12.3 = 12.3%). ROAS is a multiple (3.45 = 3.45x).",
    ]
    ws.write_string(end, 0, "Calculation rules", book.fmt["section"])
    for i, text in enumerate(rules, start=end + 1):
        ws.write_string(i, 0, f"- {text}", book.fmt["text"])
    ws.set_column(0, 0, 26)
    ws.set_column(1, 1, 48)
    ws.set_column(2, 2, 60)
    ws.set_column(3, 3, 20)


def _profitability(book: _Book, prof) -> None:
    """U4: one sheet with the summary, then paid campaigns, paid channels, owned channels (not
    ranked) and months. Figures based on an assumed margin are labelled as such at the top."""
    if prof is None or not getattr(prof, "available", False):
        return
    from analytics.profitability import TABLE_COLUMNS, assumed_margin_label, summary_lines
    ws = book.sheet("Profitability")
    row = 0
    if prof.assumed:
        ws.write_string(row, 0, f"{assumed_margin_label(prof.assumed_margin_pct)}: every figure on this "
                        "sheet is an estimate (the data has no gross profit or margin column).", book.fmt["section"])
        row += 1
    band = cfg.PROFIT_NEAR_BAND
    ws.write_string(row, 0, f"{prof.basis_note} Status: ROAS vs the campaign's own break-even ROAS (1 / gross "
                    f"margin); Profitable at {1 + band:.2f}× or more, Near break-even within ±{band * 100:.0f}%, "
                    f"Loss-making below {1 - band:.2f}×. Owned channels are not rated.", book.fmt["subtitle"])
    row += 1
    for line in summary_lines(prof):
        ws.write_string(row, 0, line, book.fmt["subtitle"])
        row += 1
    row += 1
    monthly = prof.monthly.rename(columns={"period": "month"}) if not prof.monthly.empty else prof.monthly
    blocks = [("Paid Campaigns", prof.campaigns, ["campaign", "channel"] + TABLE_COLUMNS),
              ("Paid Channels", prof.channels, ["channel"] + TABLE_COLUMNS),
              ("Owned Channels (Not Ranked: Mostly Fixed Costs)", prof.owned_channels,
               ["channel"] + TABLE_COLUMNS[:-1]),
              ("Owned-Channel Campaigns (Not Ranked)", prof.owned_campaigns,
               ["campaign", "channel"] + TABLE_COLUMNS[:-1]),
              ("Months", monthly, ["month"] + TABLE_COLUMNS)]
    for title, table, cols in blocks:
        if table is None or table.empty:
            continue
        ws.write_string(row, 0, title, book.fmt["section"])
        row = book.table(ws, table, start_row=row + 1, columns=cols, freeze=False, filter_=False)


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------

def generate_workbook(result, output, clean_df: pd.DataFrame | None = None,
                      quality_log: pd.DataFrame | None = None, ai: dict | None = None,
                      profitability=None) -> list[str]:
    """Write the workbook to a path or binary buffer. Returns the sheet names written.
    `profitability` (optional) replaces the result's own, e.g. one based on an assumed margin."""
    generated = format_datetime_ist(datetime.now(timezone.utc)) + " IST"
    try:
        book = _Book(str(output) if isinstance(output, Path) else output)
        _executive_kpis(book, result, generated)
        if clean_df is not None and not clean_df.empty:
            _clean_data(book, clean_df)
        _kpi_analysis(book, result)
        _simple_sheet(book, "Campaign_Analysis", result.campaigns,
                      "One row per campaign. Ratios are recomputed from each campaign's totals.")
        _channels(book, result)
        _profitability(book, profitability if profitability is not None else getattr(result, "profitability", None))
        _segments(book, result)
        _funnel(book, result)
        _trends(book, result)
        _incidents(book, result)
        _anomalies(book, result)
        _ai(book, ai)
        _data_quality(book, result, quality_log)
        _methodology(book, result)
        names = [ws.get_name() for ws in book.wb.worksheets()]
        book.close()
        return names
    except Exception as exc:  # noqa: BLE001
        raise WorkbookError("The Excel workbook could not be created. Please try again; the "
                            "analysis itself is still available in the dashboard.") from exc


def export_workbook(result, clean_df: pd.DataFrame | None = None, quality_log: pd.DataFrame | None = None,
                    exports_dir: str | Path | None = None, db_path=None, ai: dict | None = None,
                    profitability=None) -> Path:
    """Save under exports/<run_id>/ and record it in the database (if the run was saved)."""
    run_id = result.metadata.get("run_id")
    folder = Path(exports_dir or EXPORTS_DIR) / (str(run_id) if run_id else "unsaved")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / export_filename("Marketing_Intelligence_Workbook", run_id, "xlsx")
    generate_workbook(result, path, clean_df, quality_log, ai, profitability=profitability)
    if run_id:
        from database import repository
        from database.connection import DatabaseError
        try:
            repository.save_report(run_id, "excel", str(path), db_path=db_path)
        except DatabaseError:
            pass
    return path
