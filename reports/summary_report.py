"""Basic "Summary Report" PDF (U8): 2-3 pages in plain words, with the creator signature.

The same sentences as the Basic view (utils/plain_language.py), built from the same AnalysisResult,
so every number matches the dashboard. It reuses the executive report's fonts, styles, page
decoration and chart builder; the full Executive PDF and the Excel workbook stay available.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import settings as cfg
from dashboard import chart_standard as cs
from dashboard import theme
from reports import export_filename, pdf_charts
from reports.pdf_report import (EXPORTS_DIR, MARGIN, TEXT_W, ReportError, _decorate, _NumberedCanvas,
                                _register_fonts, _Report, tag)
from utils import plain_language as pl
from utils.formatting import format_count, format_date, format_datetime_ist

SUMMARY_TITLE = "Marketing Summary Report"
# Key-number status -> (background, text colour), the same colours as the Basic view cards.
STATUS_COLOURS = {"good": ("#e3f4e8", theme.GOOD_TEXT), "watch": ("#fff3d1", "#8a5d00"),
                  "bad": ("#fde2e1", "#a32121"), "neutral": (theme.PANEL, theme.INK_SECONDARY)}


def _key_numbers_table(rep: _Report, numbers: list[pl.KeyNumber]) -> None:
    s = rep.s
    rows = [[Paragraph("<b>What</b>", s["cell_head"]), Paragraph("<b>Number</b>", s["cell_head"]),
             Paragraph("<b>In Plain Words</b>", s["cell_head"]), Paragraph("<b>How It Looks</b>", s["cell_head"])]]
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(theme.AXIS)),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(theme.GRID)),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for i, k in enumerate(numbers, start=1):
        bg, fg = STATUS_COLOURS[k.status]
        status = f"<font color='{fg}'><b>{escape(k.status_text)}</b></font>" + (
            f"<br/>{escape(k.basis)}" if k.basis else "")
        rows.append([Paragraph(escape(k.title), s["cell"]), Paragraph(f"<b>{escape(k.value)}</b>", s["body"]),
                     Paragraph(escape(k.sentence), s["cell"]), Paragraph(status, s["cell"])])
        style.append(("BACKGROUND", (3, i), (3, i), colors.HexColor(bg)))
    t = Table(rows, colWidths=[34 * mm, 24 * mm, 64 * mm, TEXT_W - 122 * mm], repeatRows=1)
    t.setStyle(TableStyle(style))
    rep.story += [t, Spacer(1, 4)]
    rep.p(f"Colours: a target's status when you set one; money spent against this month's budget; otherwise "
          f"the last {cfg.BASIC_COMPARE_DAYS} days compared with the {cfg.BASIC_COMPARE_DAYS} days before. "
          "The numbers cover the full dataset.", "caption")


def _heading(rep: _Report, title: str) -> None:
    rep.story.append(Paragraph(escape(title), rep.s["h1"]))


def generate_summary_pdf(result, output: str | Path | io.BytesIO, df: pd.DataFrame | None = None,
                         ai: dict | None = None, profitability=None, targets=None, pacing=None) -> None:
    """Write the Basic summary for `result`. `df` = the cleaned data (for the recent comparison);
    `profitability` / `targets` / `pacing` as on the Reports page (they may carry user assumptions)."""
    _register_fonts()
    generated = format_datetime_ist(datetime.now(timezone.utc)) + " IST"
    rep = _Report(result, ai)
    prof = profitability if profitability is not None else rep.prof
    pacing = pacing if pacing is not None else rep.pacing
    meta = result.metadata
    try:
        rep.story.append(Paragraph(SUMMARY_TITLE, rep.s["cover_title"]))
        rep.story.append(Spacer(1, 2 * mm))
        rep.p(f"{meta.get('dataset_name') or 'Your data'} · {format_date(meta.get('date_min'))} to "
              f"{format_date(meta.get('date_max'))} · {format_count(meta.get('rows'))} rows · Generated {generated}",
              "small")
        rep.p(escape(cfg.REPORT_SIGNATURE), "small", raw=True)
        rep.story.append(Spacer(1, 3 * mm))

        _heading(rep, "How Are We Doing?")
        _key_numbers_table(rep, pl.key_numbers(result, df, targets=getattr(targets, "targets", None),
                                               pacing=pacing))
        rep.bullets(pl.overview_lines(result, df) + pl.profit_lines(prof) + pl.target_lines(targets)
                    + pl.pacing_lines(pacing))

        ch = result.channels
        if ch is not None and not ch.empty and "spend_share_pct" in ch:
            _heading(rep, "Where the Money Goes")
            series = {"Share of Spend": "spend_share_pct"}
            if "revenue_share_pct" in ch:
                series["Share of Revenue"] = "revenue_share_pct"
            elif "leads_share_pct" in ch:
                series["Share of Leads"] = "leads_share_pct"
            result_col = list(series.values())[-1]
            span = cs.date_span(result.trends.get("day"), "period")
            rep.chart(pdf_charts.share_png(ch, "channel", series,
                                           cs.share_title(ch, "channel", result_col,
                                                          list(series)[-1].removeprefix("Share of "),
                                                          "Where the Money Goes and What It Brings Back"),
                                           subtitle=cs.subtitle(None, None, *span, note="Share of total (%)"),
                                           owned=cs.owned_names(ch["channel"])))
            rep.bullets(pl.money_lines(result))

        best, weakest, metric = pl.best_and_weakest(result)
        if metric is not None:
            _heading(rep, "What's Working and What's Not")
            rep.story.append(Paragraph("Working well", rep.s["h2"]))
            rep.bullets([pl.campaign_line(r, metric) for _, r in best.iterrows()])
            if not weakest.empty:
                rep.story.append(Paragraph("Not working as well", rep.s["h2"]))
                rep.bullets([pl.campaign_line(r, metric) for _, r in weakest.iterrows()])

        alerts = pl.top_alerts(result)
        _heading(rep, "Top Alerts")
        if alerts:
            rep.bullets([f"<b>{escape(a['title'])}, {escape(a['period'])}</b>: {escape(a['what'])} "
                         f"{escape(a['impact'])}" for a in alerts], raw=True)
        else:
            rep.p("No problems stood out in this data.")

        summary, recs = pl.ai_summary(ai)
        if summary or recs:
            _heading(rep, "AI Summary")
            rep.p(tag("AI interpretation") + " Written by Google Gemini from the verified numbers above. "
                  "Check before acting on it.", "small", raw=True)
            for text in summary:
                rep.p(text)
            if recs:
                rep.story.append(Paragraph("Top recommendations", rep.s["h2"]))
                rep.bullets(recs)
        rep.story.append(Spacer(1, 3 * mm))
        rep.p("For every detail, see the Executive PDF Report and the Excel workbook (Reports page).", "caption")

        target = str(output) if isinstance(output, Path) else output
        doc = SimpleDocTemplate(target, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=20 * mm, bottomMargin=20 * mm, title=SUMMARY_TITLE,
                                author="Marketing Intelligence Platform",
                                subject=str(meta.get("dataset_name") or ""))
        decorate = _decorate(meta, generated, title=SUMMARY_TITLE, first_page=True)
        doc.build(rep.story, onFirstPage=decorate, onLaterPages=decorate, canvasmaker=_NumberedCanvas)
    except ReportError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReportError("The summary report could not be created. Please try again; the full "
                          "report and the dashboard are still available.") from exc


def export_summary_pdf(result, exports_dir: str | Path | None = None, **kwargs) -> Path:
    """Save the summary under exports/<run_id>/ (next to the full report)."""
    run_id = result.metadata.get("run_id")
    folder = Path(exports_dir or EXPORTS_DIR) / (str(run_id) if run_id else "unsaved")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / export_filename("Marketing_Summary_Report", run_id, "pdf")
    generate_summary_pdf(result, path, **kwargs)
    return path
