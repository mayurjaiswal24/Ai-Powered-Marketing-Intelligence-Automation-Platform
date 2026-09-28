"""Executive PDF report, generated automatically from the AnalysisResult.

Consulting-style layout (ReportLab): cover page, running header/footer with page numbers,
numbered sections, content labels that separate verified metrics, analytical findings and AI
content (SPEC §16). Every number comes from the AnalysisResult and is formatted with
utils/formatting.py, so it matches the dashboard exactly. Sections without data are left out.
AI content is never invented: without AI output the AI section says so plainly.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from analytics.kpis import KPI_REGISTRY, RATIO_KPIS
from config import settings as cfg
from config.settings import PROJECT_ROOT, settings
from dashboard import theme
from dashboard.tables import format_table
from reports import export_filename, pdf_charts
from utils.formatting import format_count, format_date, format_datetime_ist, format_value

FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
EXPORTS_DIR = (Path(settings.exports_dir) if Path(settings.exports_dir).is_absolute()
               else PROJECT_ROOT / settings.exports_dir)
REPORT_TITLE = "Marketing Intelligence Report"
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
TEXT_W = PAGE_W - 2 * MARGIN

AI_NOT_GENERATED = "AI insights were not generated for this run."

# Content labels (SPEC §16): background tint + text colour, so they stay readable in print.
TAGS = {
    "Verified metric": ("#e3eefb", "#1c5cab"),
    "Analytical finding": ("#e9e6f8", "#4a3aa7"),
    "Analytical summary": ("#e9e6f8", "#4a3aa7"),
    "AI interpretation": ("#fde9e0", "#a8431c"),
    "Hypothesis": ("#fdf1d6", "#8a5d00"),
    "Recommendation": ("#e0f1e0", "#006300"),
}


class ReportError(Exception):
    """PDF could not be produced. `user_message` is safe to show in the UI."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


# ---------------------------------------------------------------------------------------------
# Fonts and styles
# ---------------------------------------------------------------------------------------------

def _register_fonts() -> None:
    """DejaVu Sans is embedded because ReportLab's built-in fonts have no ₹ glyph."""
    if "DejaVu" in pdfmetrics.getRegisteredFontNames():
        return
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    except Exception as exc:  # noqa: BLE001
        raise ReportError("The report font is missing (assets/fonts). Reinstall the project files "
                          "and try again.") from exc
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                  italic="DejaVu", boldItalic="DejaVu-Bold")


def _styles() -> dict[str, ParagraphStyle]:
    ink, ink2, muted = (colors.HexColor(c) for c in (theme.INK, theme.INK_SECONDARY, theme.INK_MUTED))
    base = dict(fontName="DejaVu", textColor=ink, alignment=TA_LEFT)
    return {
        "cover_title": ParagraphStyle("cover_title", fontName="DejaVu-Bold", fontSize=26, leading=32,
                                      textColor=ink),
        "cover_sub": ParagraphStyle("cover_sub", fontSize=12.5, leading=17, textColor=ink2, **{k: v for k, v in base.items() if k != "textColor"}),
        "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15, leading=19, spaceBefore=4,
                             spaceAfter=8, textColor=ink),
        "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11, leading=14, spaceBefore=8,
                             spaceAfter=4, textColor=ink),
        "body": ParagraphStyle("body", fontSize=9.5, leading=13.5, spaceAfter=5, **base),
        "bullet": ParagraphStyle("bullet", fontSize=9.5, leading=13.5, spaceAfter=3, leftIndent=12,
                                 bulletIndent=2, **base),
        "small": ParagraphStyle("small", fontSize=8, leading=10.5, textColor=ink2,
                                **{k: v for k, v in base.items() if k != "textColor"}),
        "caption": ParagraphStyle("caption", fontSize=8, leading=10.5, textColor=muted, spaceAfter=8,
                                  **{k: v for k, v in base.items() if k != "textColor"}),
        "cell": ParagraphStyle("cell", fontSize=8, leading=10, **base),
        "cell_head": ParagraphStyle("cell_head", fontName="DejaVu-Bold", fontSize=8, leading=10,
                                    textColor=ink2),
        "kpi_label": ParagraphStyle("kpi_label", fontSize=8, leading=10, textColor=ink2,
                                    **{k: v for k, v in base.items() if k != "textColor"}),
        "kpi_value": ParagraphStyle("kpi_value", fontName="DejaVu-Bold", fontSize=14, leading=18,
                                    textColor=ink),
        "notice": ParagraphStyle("notice", fontSize=9.5, leading=13.5, textColor=ink2,
                                 **{k: v for k, v in base.items() if k != "textColor"}),
    }


def tag(label: str) -> str:
    bg, fg = TAGS[label]
    return f'<font backColor="{bg}" color="{fg}" size="7.5">&nbsp;{escape(label.upper())}&nbsp;</font>'


# ---------------------------------------------------------------------------------------------
# Page decoration: header, footer, "Page x of y"
# ---------------------------------------------------------------------------------------------

class _NumberedCanvas(rl_canvas.Canvas):
    """Draws 'Page x of y' after the total number of pages is known."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):  # noqa: N802 (ReportLab API name)
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.setFont("DejaVu", 7.5)
                self.setFillColor(colors.HexColor(theme.INK_MUTED))
                self.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _decorate(meta: dict, generated: str):
    def on_page(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor(theme.GRID))
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)
        canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canvas.setFont("DejaVu-Bold", 8)
        canvas.setFillColor(colors.HexColor(theme.INK))
        canvas.drawString(MARGIN, PAGE_H - 11 * mm, REPORT_TITLE)
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor(theme.INK_SECONDARY))
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 11 * mm, str(meta.get("dataset_name") or ""))
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor(theme.INK_MUTED))
        canvas.drawString(MARGIN, 10 * mm, cfg.REPORT_SIGNATURE)
        canvas.drawString(MARGIN, 6.5 * mm, f"Generated {generated} | Confidential - internal use")
        canvas.restoreState()
    return on_page


# ---------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------

class _Report:
    def __init__(self, result, ai: dict | None):
        self.r = result
        self.ai = ai
        self.s = _styles()
        self.story: list = []
        self.section_no = 0

    # text
    def h1(self, title: str) -> None:
        self.section_no += 1
        self.story.append(CondPageBreak(60 * mm))
        self.story.append(Paragraph(f"{self.section_no}. {escape(title)}", self.s["h1"]))

    def h2(self, title: str) -> None:
        self.story.append(Paragraph(escape(title), self.s["h2"]))

    def p(self, text: str, style: str = "body", raw: bool = False) -> None:
        self.story.append(Paragraph(text if raw else escape(text), self.s[style]))

    def bullets(self, items: list[str], raw: bool = False) -> None:
        for item in items:
            self.story.append(Paragraph(item if raw else escape(item), self.s["bullet"], bulletText="•"))

    def notice(self, text: str) -> None:
        t = Table([[Paragraph(escape(text), self.s["notice"])]], colWidths=[TEXT_W])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(theme.PANEL)),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(theme.AXIS)),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        self.story += [t, Spacer(1, 6)]

    def chart(self, png: bytes | None, caption: str | None = None) -> None:
        if png is None:
            return
        img = Image(io.BytesIO(png))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth, img.drawHeight = TEXT_W, TEXT_W * ratio
        parts = [img] + ([Paragraph(escape(caption), self.s["caption"])] if caption else [Spacer(1, 6)])
        self.story.append(KeepTogether(parts))

    def table(self, df: pd.DataFrame, columns: list[str], widths: list[float] | None = None,
              max_rows: int | None = None) -> None:
        if df is None or df.empty:
            return
        data = df.head(max_rows) if max_rows else df
        # Compact money (₹3.3 Cr) keeps dense tables readable; exact values are in the Excel file.
        shown = format_table(data, [c for c in columns if c in data], compact_money=True)
        header = [Paragraph(escape(c), self.s["cell_head"]) for c in shown.columns]
        rows = [[Paragraph(escape(str(v)), self.s["cell"]) for v in row] for row in shown.itertuples(index=False)]
        if widths is None:
            widths = _auto_widths(shown)
        t = Table([header] + rows, colWidths=widths, repeatRows=1)
        style = [
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(theme.AXIS)),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(theme.GRID)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]
        t.setStyle(TableStyle(style))
        self.story += [t, Spacer(1, 8)]
        if max_rows and len(df) > max_rows:
            self.p(f"Showing {max_rows} of {len(df)} rows; the full table is in the appendix or the "
                   "Excel workbook.", "caption")

    def kpi_grid(self, keys: list[str]) -> None:
        kpis = self.r.kpis
        cells = []
        for key in keys:
            k = kpis.get(key)
            if k is None or not k.available:
                continue
            cells.append([Paragraph(escape(k.label), self.s["kpi_label"]),
                          Paragraph(escape(k.formatted_compact), self.s["kpi_value"])])  # same as dashboard cards
        if not cells:
            return
        per_row = 4
        rows = []
        for i in range(0, len(cells), per_row):
            chunk = cells[i:i + per_row] + [[""]] * (per_row - len(cells[i:i + per_row]))
            rows.append([Table([[c[0]], [c[1]]] if len(c) == 2 else [[""]], colWidths=[TEXT_W / per_row - 6])
                         for c in chunk])
        t = Table(rows, colWidths=[TEXT_W / per_row] * per_row)
        t.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(theme.GRID)),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor(theme.GRID)),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(theme.SURFACE)),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        self.story += [t, Spacer(1, 6)]


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------

def _cover(rep: _Report, generated: str) -> None:
    meta = rep.r.metadata
    s = rep.s
    rep.story.append(Spacer(1, 55 * mm))
    rep.story.append(Paragraph(REPORT_TITLE, s["cover_title"]))
    rep.story.append(Spacer(1, 4 * mm))
    rep.story.append(Paragraph("Performance, efficiency and data-quality review generated "
                               "automatically from verified analytics", s["cover_sub"]))
    rep.story.append(Spacer(1, 14 * mm))
    rows = [("Dataset", meta.get("dataset_name") or "N/A"),
            ("Period covered", f"{format_date(meta.get('date_min'))} to {format_date(meta.get('date_max'))}"),
            ("Rows analysed", format_count(meta.get("rows"))),
            ("Analysis run", f"#{meta['run_id']}" if meta.get("run_id") else "Not saved"),
            ("Generated", generated)]
    t = Table([[Paragraph(escape(a), s["kpi_label"]), Paragraph(escape(str(b)), s["body"])] for a, b in rows],
              colWidths=[40 * mm, TEXT_W - 40 * mm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(theme.GRID)),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    rep.story.append(t)
    rep.story.append(Spacer(1, 5 * mm))
    rep.story.append(Paragraph(escape(cfg.REPORT_SIGNATURE), s["small"]))
    rep.story.append(Spacer(1, 12 * mm))
    rep.story.append(Paragraph("How to read this report", s["h2"]))
    rep.story.append(Paragraph(
        "Every statement carries a label: " + " ".join(tag(t) for t in
        ("Verified metric", "Analytical finding", "AI interpretation", "Hypothesis", "Recommendation"))
        + ". Verified metrics are calculated directly from the data; analytical findings come from "
        "deterministic rules and statistics; AI content, when present, interprets that evidence and "
        "must be validated before acting on it.", s["body"]))
    rep.story.append(PageBreak())


def _executive_summary(rep: _Report) -> None:
    r, k = rep.r, rep.r.kpis
    rep.h1("Executive summary")
    rep.p(tag("Analytical summary") + " Written automatically from the verified metrics and findings "
          "below; it contains no AI-generated text.", "small", raw=True)
    meta = r.metadata
    channels = len(r.channels) if not r.channels.empty else 0
    campaigns = len(r.campaigns) if not r.campaigns.empty else 0
    parts = [f"Between {format_date(meta.get('date_min'))} and {format_date(meta.get('date_max'))}, "
             f"{format_count(meta.get('rows'))} rows of marketing data were analysed"
             + (f" across {channels} channels and {campaigns} campaigns." if channels else ".")]
    if k["spend"].available:
        text = f"Spend totalled {k['spend'].formatted_compact}"
        if k["revenue"].available and k["roas"].available:
            text += (f", generating {k['revenue'].formatted_compact} of revenue, a ROAS of "
                     f"{k['roas'].formatted}.")
        else:
            text += "."
        parts.append(text)
    if k["leads"].available:
        text = f"The campaigns produced {k['leads'].formatted} leads"
        if k["cpl"].available:
            text += f" at a CPL of {k['cpl'].formatted}"
        parts.append(text + ".")
    if k["conversions"].available:
        text = f"{k['conversions'].formatted} conversions were recorded"
        if k["cac"].available and k["cac"].value is not None:
            text += f", an acquisition cost (CAC) of {k['cac'].formatted} each"
        parts.append(text + ".")
    rep.p(" ".join(parts))
    rep.kpi_grid(["revenue", "spend", "roas", "roi", "leads", "cpl", "conversions", "cac"])

    if r.findings:
        rep.h2("Headline findings")
        rep.bullets([f"{tag('Analytical finding')} {escape(f.text)}" for f in r.findings[:3]], raw=True)
    inc = r.incidents
    if inc is not None and not inc.empty:
        problems = inc[inc["sentiment"] == "negative"]
        at_stake = problems[problems["impact_direction"] == "loss"]["impact_inr"].sum()
        rep.p(f"Anomaly detection found {len(inc)} incidents ({len(r.anomalies)} individual flags); "
              f"{len(problems)} are performance problems with an estimated "
              f"{format_value(at_stake, 'money', compact=True)} at stake. See Performance concerns.")
    q = r.quality_summary
    if q is not None:
        rep.p(q.limitations[0] if q.limitations else "No data-quality limitations affect these results.")
    rep.notice(AI_NOT_GENERATED if not rep.ai else "AI interpretation is included in the AI insights section.")


def _kpi_overview(rep: _Report) -> None:
    rep.h1("KPI overview")
    rep.p(tag("Verified metric") + " Calculated directly from the cleaned data. Ratios are "
          "recalculated from totals, never averaged.", "small", raw=True)
    rows = []
    for k in rep.r.kpis.values():
        if k.available:
            rows.append({"KPI": k.label, "Value": k.formatted, "How it is calculated": k.formula_text,
                         "Note": k.note})
    df = pd.DataFrame(rows)
    _plain_table(rep, df, [38 * mm, 32 * mm, 55 * mm, TEXT_W - 125 * mm])
    unavailable = [k for k in rep.r.kpis.values() if not k.available]
    if unavailable:
        rep.h2("Not available for this dataset")
        rep.bullets([k.reason_unavailable for k in unavailable])


def _auto_widths(shown: pd.DataFrame, pad: float = 7.0) -> list[float]:
    """Column widths from the content: every value (and every header word) fits on one line;
    the first (label) column absorbs whatever space is left, and wraps if it must."""
    need = []
    for col in shown.columns:
        values = [pdfmetrics.stringWidth(str(v), "DejaVu", 8) for v in shown[col]]
        words = [pdfmetrics.stringWidth(w, "DejaVu-Bold", 8) for w in str(col).split()]
        need.append(max(values + words + [18]) + pad)
    rest = sum(need[1:])
    first = TEXT_W - rest
    if first < 32 * mm:                       # too many columns: shrink the others evenly
        scale = (TEXT_W - 32 * mm) / rest
        return [32 * mm] + [w * scale for w in need[1:]]
    return [first] + need[1:]


def _plain_table(rep: _Report, df: pd.DataFrame, widths: list[float]) -> None:
    """A table of already-formatted text."""
    if df.empty:
        return
    s = rep.s
    header = [Paragraph(escape(c), s["cell_head"]) for c in df.columns]
    body = [[Paragraph(escape(str(v)), s["cell"]) for v in row] for row in df.itertuples(index=False)]
    t = Table([header] + body, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(theme.AXIS)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(theme.GRID)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    rep.story += [t, Spacer(1, 8)]


def _overall_performance(rep: _Report) -> None:
    monthly = rep.r.trends.get("month")
    if monthly is None or monthly.empty:
        return
    rep.h1("Overall performance")
    rep.p(tag("Verified metric") + " Monthly totals. Partial first/last months are marked by their "
          "day count in the table.", "small", raw=True)
    outcome = next((m for m in ("revenue", "conversions", "leads") if m in monthly), None)
    for metric in [m for m in (outcome, "spend") if m]:
        rep.chart(pdf_charts.line_png(monthly, "period", metric, KPI_REGISTRY[metric].fmt,
                                      f"{KPI_REGISTRY[metric].label} by month", date_axis=False))
    cols = ["period", "days", "spend", "leads", "cpl", "conversions", "revenue", "roas"]
    rep.table(monthly.rename(columns={"period": "month"}), ["month"] + cols[1:])


def _channels(rep: _Report) -> None:
    ch = rep.r.channels
    if ch.empty:
        return
    from analytics.channel import owned_channels, paid_channels
    paid, owned = paid_channels(ch), owned_channels(ch)
    rep.h1("Channel analysis")
    rep.p(tag("Verified metric") + " Shares of spend and results by channel. Efficiency index = paid "
          "channel value / paid-media average x 100 (100 = average). Owned channels are reported "
          "separately and not ranked.", "small", raw=True)
    series = {"Share of spend": "spend_share_pct"}
    if "revenue_share_pct" in ch:
        series["Share of revenue"] = "revenue_share_pct"
    elif "leads_share_pct" in ch:
        series["Share of leads"] = "leads_share_pct"
    rep.chart(pdf_charts.share_png(ch, "channel", series, "Where the money goes and what it returns"))
    metric = "roas" if "roas" in paid and paid["roas"].notna().any() else "cpl"
    if metric in paid and len(paid):
        rep.chart(pdf_charts.hbar_png(paid, "channel", metric, KPI_REGISTRY[metric].fmt,
                                      f"{KPI_REGISTRY[metric].label} by paid channel",
                                      ascending=KPI_REGISTRY[metric].higher_is_better is False),
                  f"Paid channels only. Overall {KPI_REGISTRY[metric].label} (all channels): "
                  f"{rep.r.kpis[metric].formatted}.")
    cols = ["channel", "spend", "spend_share_pct", "leads", "cpl", "conversions", "cac", "revenue",
            "roas", "cpl_index"]
    rep.h2("Paid channels")
    rep.table(paid.sort_values("spend", ascending=False) if "spend" in paid else paid, cols)
    if not owned.empty:
        rep.h2("Owned channels (not ranked)")
        for _, row in owned.iterrows():
            parts = [f"spend {format_value(row['spend'], 'money', compact=True)}"]
            for m in ("revenue", "conversions", "roas"):
                if m in owned and pd.notna(row.get(m)):
                    parts.append(f"{KPI_REGISTRY[m].label} {format_value(row[m], KPI_REGISTRY[m].fmt, compact=True)}")
            rep.p(f"{tag('Verified metric')} <b>{escape(str(row['channel']))}</b>: {escape(', '.join(parts))}. "
                  "Owned channels reach the company's own audience with mostly fixed costs, so their ROAS "
                  "is not comparable with paid media.", raw=True)


def _campaigns(rep: _Report) -> None:
    c = rep.r.campaigns
    if c.empty:
        return
    rep.h1("Campaign analysis")
    outcome = next((m for m in ("conversions", "leads") if m in c), None)
    rep.p(tag("Verified metric") + f" Top campaigns by {outcome}. Trend compares the last 4 weeks "
          "of data with the 4 weeks before.", "small", raw=True)
    ranked = c.sort_values(outcome, ascending=False) if outcome else c
    metric = "roas" if "roas" in c and c["roas"].notna().any() else "cpl"
    rep.chart(pdf_charts.hbar_png(ranked, "campaign", outcome, "count", f"Campaigns by {outcome}", top_n=10))
    rep.table(ranked, ["campaign", "channel", "spend", "leads", "cpl", "conversions", "cac", "revenue",
                       metric, "trend"], max_rows=10)


def _funnel(rep: _Report) -> None:
    f = rep.r.funnel
    if f.empty:
        return
    rep.h1("Funnel analysis")
    rep.p(tag("Verified metric") + " Only the stages present in the data are shown. The chart uses "
          "a logarithmic scale so the smaller lower stages stay visible.", "small", raw=True)
    rep.chart(pdf_charts.funnel_png(f, "Marketing funnel"))
    rep.table(f, ["label", "value", "rate_from_previous", "drop_off_pct"])


def _segments(rep: _Report) -> None:
    segs = {d: t for d, t in rep.r.segments.items() if not t.empty}
    if not segs:
        return
    rep.h1("Segment, geography and product analysis")
    rep.p(tag("Verified metric") + " Growth compares the last 4 weeks with the 4 weeks before.",
          "small", raw=True)
    from config.fields import FIELD_BY_NAME
    for dim, table in segs.items():
        rep.h2(f"By {FIELD_BY_NAME[dim].label.lower()}")
        cols = ["segment", "spend", "leads", "conversions", "revenue", "revenue_share_pct", "cac", "aov",
                "roas", "growth_pct"]
        rep.table(table.sort_values("spend", ascending=False) if "spend" in table else table, cols)


def _trends(rep: _Report) -> None:
    weekly = rep.r.trends.get("week")
    if weekly is None or weekly.empty:
        return
    rep.h1("Trends")
    rep.p(tag("Verified metric") + " Weekly view with efficiency over time.", "small", raw=True)
    eff = "roas" if "roas" in weekly and weekly["roas"].notna().any() else "cpl"
    outcome = next((m for m in ("revenue", "conversions", "leads") if m in weekly), None)
    if outcome:
        rep.chart(pdf_charts.line_png(weekly, "period", outcome, KPI_REGISTRY[outcome].fmt,
                                      f"Weekly {KPI_REGISTRY[outcome].label.lower()}"))
    if eff in weekly:
        rep.chart(pdf_charts.line_png(weekly, "period", eff, KPI_REGISTRY[eff].fmt,
                                      f"Weekly {KPI_REGISTRY[eff].label}"))
    if rep.r.seasonality is not None:
        rep.h2("Seasonality index (100 = an average month)")
        season = rep.r.seasonality.copy()
        season["month"] = season["month_of_year"].map(lambda m: pd.Timestamp(2000, int(m), 1).strftime("%b"))
        cols = ["month"] + [c for c in season.columns if c.endswith("_index")]
        _plain_table(rep, season[cols].assign(**{c: season[c].round(0).astype(int).astype(str)
                                                 for c in cols[1:]}).rename(
            columns={c: c.replace("_index", "").capitalize() for c in cols}),
            [TEXT_W / len(cols)] * len(cols))
        rep.p(rep.r.seasonality_note, "caption")
    else:
        rep.p(rep.r.seasonality_note, "caption")


def _key_findings(rep: _Report) -> None:
    if not rep.r.findings:
        return
    rep.h1("Key findings")
    for f in rep.r.findings:
        rep.p(f"{tag('Analytical finding')} <b>{escape(f.title)}</b><br/>{escape(f.text)}", raw=True)


def _impact(value, direction, label) -> str:
    if direction not in ("loss", "gain") or not value:
        return "impact not estimated"
    return f"estimated {format_value(value, 'money', compact=True)} {direction} ({label})"


def _performance_concerns(rep: _Report) -> None:
    inc = rep.r.incidents
    if inc is None or inc.empty:
        return
    rep.h1("Performance concerns")
    rep.p(tag("Analytical finding") + " Incidents from the anomaly detector (method in Methodology), "
          "ranked by estimated rupee impact. Flags on the same entity and period form one incident; "
          "effects on other campaigns/channels in the same event are listed as related effects. They "
          "describe what changed and by how much, not why; impacts are estimates.", "small", raw=True)
    problems = inc[inc["sentiment"] != "positive"]
    if problems.empty:
        rep.p("No problem incidents were detected.")
    for r in problems.head(8).itertuples():
        lines = [f"<b>#{r.rank} {escape(r.entity)} ({escape(r.entity_type)}), {format_date(r.period_start)} to "
                 f"{format_date(r.period_end)}</b> - {escape(r.metrics)}; "
                 f"{escape(_impact(r.impact_inr, r.impact_direction, r.impact_label))}. Severity {r.severity}."]
        for f in r.flags[:3]:
            change = "" if f["change_pct"] is None else f" ({f['change_pct']:+.0f}%)"
            note = f" {f['note']}" if f.get("note") else ""
            lines.append(f"&nbsp;&nbsp;- {escape(f['metric_label'])}: expected {escape(f['expected'])}, observed "
                         f"{escape(f['observed'])}{change}.{escape(note)}")
        if r.related:
            shown = "; ".join(f"{f['entity']} {f['metric_label']} "
                              + ("" if f["change_pct"] is None else f"{f['change_pct']:+.0f}%")
                              + (" (not an improvement)" if f.get("note") else "")
                              for f in r.related[:4])
            more = f" and {len(r.related) - 4} more" if len(r.related) > 4 else ""
            lines.append(f"&nbsp;&nbsp;- Related effects: {escape(shown + more)}.")
        rep.p("<br/>".join(lines), raw=True)
    if len(problems) > 8:
        rep.p(f"{len(problems) - 8} further incidents are listed in the appendix.", "caption")
    good = inc[inc["sentiment"] == "positive"]
    if not good.empty:
        rep.h2("Notable improvements")
        rep.bullets([f"{r.entity} ({r.entity_type}), {format_date(r.period_start)} to {format_date(r.period_end)}: "
                     f"{r.metrics}; {_impact(r.impact_inr, r.impact_direction, r.impact_label)}."
                     for r in good.head(5).itertuples()])


def _ai_section(rep: _Report) -> None:
    rep.h1("AI insights, investigation areas and recommendations")
    if not rep.ai:
        rep.notice(AI_NOT_GENERATED + " This section would contain Gemini's interpretation of the "
                   "verified evidence, possible explanations (hypotheses), areas to investigate and "
                   "recommendations, each clearly labelled. The rest of this report is complete "
                   "without it.")
        return
    rep.notice("AI-generated interpretation of verified metrics (Google Gemini). Hypotheses require "
               "validation. Evidence IDs refer to the evidence pack stored with this analysis.")
    from ai.schemas import SECTIONS
    for key, title, label in SECTIONS:
        raw = rep.ai.get(key)
        items = raw if isinstance(raw, list) else ([raw] if raw else [])
        if not items:
            continue
        rep.h2(title)
        for item in items:
            text = escape(item.get("text", ""))
            extra = []
            if item.get("validation_step"):
                extra.append(f"How to validate: {escape(item['validation_step'])}")
            if item.get("metric_to_watch"):
                extra.append(f"Priority: {escape(str(item.get('priority', '')))}; metric to watch: "
                             f"{escape(item['metric_to_watch'])}")
            if item.get("test_shift_pct"):
                extra.append(f"Suggested test shift: {item['test_shift_pct']:.0f}% of the source budget "
                             "(proposal)")
            if item.get("at_stake_text"):
                extra.append(escape(item["at_stake_text"]) + " - calculated by the app")
            ids = ", ".join(item.get("evidence_ids", []))
            status = " <i>(contains unverified figures)</i>" if item.get("status") == "unverified" else ""
            if item.get("weak"):
                status += " <i>(weak: " + escape("; ".join(item.get("weak_reasons", []))) + ")</i>"
            rep.p(f"{tag(label)} {text}{status}" + "".join(f"<br/>{e}" for e in extra)
                  + (f"<br/><font color='{theme.INK_MUTED}' size='7.5'>Evidence: {ids}</font>" if ids else ""),
                  raw=True)


def _data_quality(rep: _Report) -> None:
    q = rep.r.quality_summary
    if q is None:
        return
    rep.h1("Data quality")
    rows = [("Rows in uploaded file", format_count(q.rows_in)), ("Rows analysed", format_count(q.rows_out)),
            ("Duplicate rows removed", format_count(q.removed_duplicates)),
            ("Summary rows removed", format_count(q.removed_summary_rows)),
            ("Rows with a quality flag", format_count(q.flagged_rows))]
    _plain_table(rep, pd.DataFrame(rows, columns=["Measure", "Value"]), [70 * mm, 40 * mm])
    if q.counts_by_action:
        rep.h2("Changes made during cleaning")
        changes = pd.DataFrame(sorted(q.counts_by_action.items(), key=lambda kv: -kv[1]),
                               columns=["Change", "Count"])
        changes["Change"] = changes["Change"].str.replace("_", " ").str.capitalize()
        changes["Count"] = changes["Count"].map(format_count)
        _plain_table(rep, changes, [70 * mm, 40 * mm])
    rep.p("Every individual change (row, original value, new value, reason) is in the data-quality "
          "log in the dashboard and the Excel workbook.", "caption")


def _methodology(rep: _Report) -> None:
    rep.h1("Methodology")
    rep.h2("KPI definitions")
    rows = [{"KPI": KPI_REGISTRY[k].label, "Formula": KPI_REGISTRY[k].formula_text,
             "Meaning": KPI_REGISTRY[k].description}
            for k in RATIO_KPIS if k in rep.r.kpis and rep.r.kpis[k].available]
    _plain_table(rep, pd.DataFrame(rows), [34 * mm, 58 * mm, TEXT_W - 92 * mm])
    cac = rep.r.kpis.get("cac")
    rules = [
        "Additive metrics (spend, impressions, clicks, leads, conversions, revenue) are summed. "
        "Ratios are recalculated from those sums, never averaged across rows.",
        "A ratio uses only rows where both of its inputs are known; missing values are never "
        "treated as zero.",
        "Reach is not additive (the same person is reached on many days) and is never summed.",
    ]
    if cac is not None and cac.available:
        rules.append(f"CAC: {cac.note or cac.formula_text}")
    roi = rep.r.kpis.get("roi")
    rules.append("ROI is only calculated from a gross profit or margin column; no margin is ever "
                 "assumed." + (f" {roi.note}" if roi is not None and roi.available else " This dataset "
                                                                                          "has neither, so ROI is not reported."))
    rep.h2("Calculation rules")
    rep.bullets(rules)
    rep.h2("Anomaly detection")
    rep.bullets([
        "Weekly check per campaign, channel and region on spend, revenue, clicks, CPC, CPL, lead "
        "qualification rate and lead-to-conversion rate.",
        f"Baseline: the median of the previous {cfg.ANOMALY_BASELINE_WEEKS} full weeks (at least "
        f"{cfg.ANOMALY_MIN_HISTORY_WEEKS}). Seasonality is removed by comparing each entity's change "
        "with the typical (median) change of all entities in the same week.",
        f"A period is flagged only if its robust score (distance from normal in MAD units) is at least "
        f"{cfg.ANOMALY_SCORE_THRESHOLD}, the change is at least {cfg.ANOMALY_MIN_PCT_CHANGE:.0f}%, "
        "the volume is large enough, and the change is bigger than chance with small counts would "
        "produce. Sustained 3-week changes are checked as well as single weeks.",
        "A daily rule flags possible tracking outages: spend continues but clicks fall below "
        f"{cfg.OUTAGE_CLICK_SHARE:.0%} of normal.",
    ])


def _limitations(rep: _Report) -> None:
    items = []
    caps = rep.r.capabilities
    if caps is not None:
        items += caps.limitations
    q = rep.r.quality_summary
    if q is not None:
        items += q.limitations
    items += [n for n in rep.r.notes if n not in items]
    items.append("Findings and anomalies show what changed and by how much. They do not prove why; "
                 "causes need to be confirmed with the teams involved.")
    rep.h1("Limitations")
    rep.bullets(list(dict.fromkeys(items)))


def _appendix(rep: _Report) -> None:
    rep.h1("Appendix: supporting metrics")
    c = rep.r.campaigns
    if not c.empty:
        rep.h2("All campaigns")
        metric = "roas" if "roas" in c and c["roas"].notna().any() else "cpl"
        rep.table(c.sort_values("spend", ascending=False) if "spend" in c else c,
                  ["campaign", "channel", "spend", "leads", "conversions", "revenue", metric,
                   "active_days"])
    inc = rep.r.incidents
    if inc is not None and not inc.empty:
        rep.h2("All incidents (ranked by estimated impact)")
        rows = pd.DataFrame({
            "#": inc["rank"].astype(str),
            "Period": [f"{format_date(s)} - {format_date(e)}" for s, e in zip(inc["period_start"], inc["period_end"])],
            "Entity": inc["entity"] + " (" + inc["entity_type"] + ")",
            "Metrics": inc["metrics"],
            "Impact": [_impact(v, d, l).replace("estimated ", "") for v, d, l in
                       zip(inc["impact_inr"], inc["impact_direction"], inc["impact_label"])],
            "Related": inc["n_related"].astype(str),
        })
        _plain_table(rep, rows, [8 * mm, 32 * mm, 42 * mm, 34 * mm, TEXT_W - 130 * mm, 14 * mm])
    a = rep.r.anomalies
    if not a.empty:
        rep.h2("All individual flags")
        rows = pd.DataFrame({
            "Period": [f"{format_date(s)} - {format_date(e)}" for s, e in zip(a["period_start"], a["period_end"])],
            "Entity": a["entity"] + " (" + a["entity_type"] + ")",
            "Metric": a["metric"].map(lambda m: KPI_REGISTRY[m].label if m in KPI_REGISTRY else m),
            "Expected": [format_value(v, KPI_REGISTRY[m].fmt) for v, m in zip(a["baseline"], a["metric"])],
            "Observed": [format_value(v, KPI_REGISTRY[m].fmt) for v, m in zip(a["observed"], a["metric"])],
            "Change": a["pct_change"].map(lambda v: "N/A" if pd.isna(v) else f"{v:+.0f}%"),
        })
        _plain_table(rep, rows, [34 * mm, 48 * mm, 36 * mm, 20 * mm, 20 * mm, TEXT_W - 158 * mm])


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------

def generate_pdf(result, output: str | Path | io.BytesIO, ai: dict | None = None) -> None:
    """Write the executive report for `result` to a file path or a binary buffer."""
    _register_fonts()
    generated = format_datetime_ist(datetime.now(timezone.utc)) + " IST"
    rep = _Report(result, ai)
    try:
        _cover(rep, generated)
        for section in (_executive_summary, _kpi_overview, _overall_performance, _channels,
                        _campaigns, _funnel, _segments, _trends, _key_findings,
                        _performance_concerns, _ai_section, _data_quality, _methodology,
                        _limitations, _appendix):
            section(rep)
        target = str(output) if isinstance(output, Path) else output
        doc = SimpleDocTemplate(target, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=20 * mm, bottomMargin=20 * mm, title=REPORT_TITLE,
                                author="Marketing Intelligence Platform",
                                subject=str(result.metadata.get("dataset_name") or ""))
        decorate = _decorate(result.metadata, generated)
        doc.build(rep.story, onFirstPage=decorate, onLaterPages=decorate, canvasmaker=_NumberedCanvas)
    except ReportError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReportError("The PDF report could not be created. Please try again; if it keeps "
                          "failing, the analysis itself is still available in the dashboard.") from exc


def export_pdf(result, exports_dir: str | Path | None = None, db_path=None,
               ai: dict | None = None) -> Path:
    """Save the report under exports/<run_id>/ and record it in the database (if the run was saved)."""
    run_id = result.metadata.get("run_id")
    folder = Path(exports_dir or EXPORTS_DIR) / (str(run_id) if run_id else "unsaved")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / export_filename("Marketing_Intelligence_Report", run_id, "pdf")
    generate_pdf(result, path, ai=ai)
    if run_id:
        from database import repository
        from database.connection import DatabaseError
        try:
            repository.save_report(run_id, "pdf", str(path), db_path=db_path)
        except DatabaseError:
            pass   # the file exists; failing to record it must not lose the report
    return path
