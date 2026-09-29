"""Plain-English sentences for the Basic view and the Summary Report (U8).

Every number comes from the AnalysisResult (or the stored profitability / pacing / targets results);
this module only chooses words. Nothing is recalculated here except the "last 30 days vs the 30 days
before" comparison, which calls the same analytics/kpis.py function the dashboard filters use.

Rules for Basic text: no unexplained abbreviations (CTR, CPC, CPL, CAC, ROAS, AOV, KPI, ROI).
Our own sentences use plain words; AI text (which may use them) goes through `plain_ai_text`, which
turns each abbreviation into plain words and removes evidence and incident IDs. A glossary
(`GLOSSARY`) explains the few everyday terms that remain (lead, customer won, alert...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

import config.settings as cfg
from utils.formatting import format_change, format_date, format_date_range, format_value

# Abbreviations and the plain words that replace them in Basic text (AI text included).
ABBREVIATION_WORDS = {
    "ROAS": "return on ad spend", "ROI": "return on investment", "CPL": "cost per lead",
    "CAC": "cost per customer", "CPC": "cost per click", "CTR": "click-through rate",
    "AOV": "average sale value", "KPIs": "headline numbers", "KPI": "headline number",
}
ABBREVIATIONS = ("CTR", "CPC", "CPL", "CAC", "ROAS", "AOV", "KPI", "ROI")

# The Basic glossary ("What Do These Words Mean?" on every Basic page). A term the Professional
# view shows as an abbreviation is named with its meaning in brackets.
GLOSSARY = [
    ("Lead", "A person who showed interest, for example by filling in a form."),
    ("Customer won", "A lead who went on to buy or sign up (the data calls this a conversion)."),
    ("Money earned", "Revenue: the value of the sales that marketing brought in."),
    ("Money back per ₹1 spent", "Money earned divided by money spent. ₹4 back means ₹4 of sales for every ₹1 "
                                "of advertising. The Professional view calls this ROAS (return on ad spend)."),
    ("Cost to get one lead", "Money spent divided by leads. The Professional view calls this CPL (cost per lead)."),
    ("Cost to win one customer", "Money spent divided by customers won. The Professional view calls this "
                                 "CAC (customer acquisition cost)."),
    ("Profit after the cost of what was sold", "Gross profit minus marketing spend: what marketing really "
                                               "earned after paying for the product and the ads."),
    ("Alert", "A period when a number moved far from its usual level. The ₹ amount is an estimate of what "
              "the change cost or earned."),
    ("Paid and own channels", "Paid channels (search, social, video ads) are bought per click or view. Own "
                              "channels, like your email list, use your own audience and are not compared "
                              "with paid advertising."),
]

# KPI keys -> plain names (used in sentences, target lines and alerts).
PLAIN_NAMES = {
    "spend": "money spent", "revenue": "money earned", "roas": "money back per ₹1 spent",
    "roi": "profit per ₹1 spent", "cpl": "cost per lead", "cac": "cost per customer",
    "cpc": "cost per click", "ctr": "share of people who clicked", "aov": "average sale value",
    "leads": "leads", "conversions": "customers won", "clicks": "clicks", "impressions": "ad views",
    "qualified_leads": "qualified leads", "gross_profit": "gross profit",
    "click_to_lead_rate": "share of clicks that became leads",
    "lead_qualification_rate": "share of leads that were qualified",
    "lead_to_conversion_rate": "share of leads that became customers",
    "budget_utilisation": "share of budget used",
    "roas_headroom": "money back above break-even", "monthly_spend": "monthly spend",
    "monthly_leads": "monthly leads", "monthly_conversions": "monthly customers won",
    "monthly_revenue": "monthly revenue",
}

STATUS_TEXT = {"good": "Looking good", "watch": "Worth watching", "bad": "Needs attention",
               "neutral": "For information"}
_TARGET_TO_STATUS = {"On Target": "good", "Within 10%": "watch", "Off Target": "bad", "Over Target": "bad",
                     "Under Target": "bad"}
_PACING_TO_STATUS = {"On Pace": "good", "Underspending": "watch", "Overspending": "bad"}
_ENTITY_WORDS = {"campaign": "campaign", "channel": "channel", "region": "region", "platform": "platform",
                 "segment": "customer segment", "product": "product"}


# ---------------------------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------------------------

def plain_name(key: str) -> str:
    return PLAIN_NAMES.get(key, key.replace("_", " "))


def per_rupee(roas) -> str:
    """ROAS 4.57 -> '₹4.57' (what ₹1 of spend brought back)."""
    return "N/A" if roas is None or pd.isna(roas) else f"₹{roas:,.2f}"


def money(value) -> str:
    return format_value(value, "money", compact=True)


_ID_PATTERNS = [
    (re.compile(r"\s*\((?:evidence\s+)?[EI]\d{1,3}(?:\s*,\s*[EI]\d{1,3})*\)", re.I), ""),   # "(E01, E04)"
    (re.compile(r"\bfrom incident [EI]\d{1,3}\b", re.I), "from an earlier incident"),
    (re.compile(r"\bincident [EI]\d{1,3}\b", re.I), "an earlier incident"),
    (re.compile(r"\b(?:evidence\s+)?E\d{1,3}\b"), ""),
    (re.compile(r"\bI\d{2}\b"), "an earlier incident"),
]


def strip_ids(text: str) -> str:
    """Remove evidence IDs (E01) and incident IDs (I09): Basic readers never see them."""
    out = text or ""
    for pattern, repl in _ID_PATTERNS:
        out = pattern.sub(repl, out)
    return re.sub(r"\s{2,}", " ", out).replace(" .", ".").replace(" ,", ",").strip()


_NUMBER = r"(\d+(?:\.\d+)?)"


def plain_words(text: str) -> str:
    """Abbreviations -> plain words: 'ROAS 1.55x' -> '₹1.55 back per ₹1 spent', 'CPL' -> 'cost per lead'."""
    out = text or ""
    out = re.sub(rf"\bROAS\s*(?:of\s+)?{_NUMBER}x", r"₹\1 back per ₹1 spent", out)
    out = re.sub(rf"{_NUMBER}x\s+ROAS\b", r"₹\1 back per ₹1 spent", out)
    for abbr, words in ABBREVIATION_WORDS.items():
        out = re.sub(rf"\b{abbr}\b", words, out)
    return out


def plain_ai_text(text: str) -> str:
    return plain_words(strip_ids(text))


def unexplained_terms(text: str) -> list[str]:
    """Abbreviations used without their meaning next to them (for tests). Explained = 'ROAS (return
    on ad spend)' or 'key numbers (KPIs)'."""
    found = []
    for abbr in ABBREVIATIONS:
        for match in re.finditer(rf"\b{abbr}s?\b", text or ""):
            bracketed = text[match.start() - 1:match.start()] == "(" and text[match.end():match.end() + 1] == ")"
            if not text[match.end():].startswith(" (") and not bracketed:
                found.append(abbr)
                break
    return found


# ---------------------------------------------------------------------------------------------
# Key numbers (the cards on "How Are We Doing?" and the Summary Report)
# ---------------------------------------------------------------------------------------------

@dataclass
class KeyNumber:
    key: str
    title: str
    value: str             # as on the Professional KPI cards (same formatting)
    sentence: str
    status: str            # good / watch / bad / neutral (green / yellow / red / grey)
    basis: str             # why that colour, in plain words

    @property
    def status_text(self) -> str:
        return STATUS_TEXT[self.status]


_CARD_ORDER = ["spend", "revenue", "roas", "cpl", "cac", "leads", "conversions"]
_TITLES = {"spend": "Money Spent", "revenue": "Money Earned", "roas": "Money Back per ₹1 Spent",
           "cpl": "Cost to Get One Lead", "cac": "Cost to Win One Customer", "leads": "Leads",
           "conversions": "Customers Won"}


def recent_comparison(df: pd.DataFrame | None, capabilities=None):
    """The last BASIC_COMPARE_DAYS days vs the same number of days before (analytics/kpis.py)."""
    if df is None or df.empty or "date" not in df or df["date"].notna().sum() == 0:
        return None
    from analytics.kpis import period_comparison
    end = df["date"].max()
    start = end - pd.Timedelta(days=cfg.BASIC_COMPARE_DAYS - 1)
    comparison = period_comparison(df, start, end, capabilities=capabilities)
    return None if comparison.note else comparison


def _trend_status(delta) -> tuple[str, str]:
    days = cfg.BASIC_COMPARE_DAYS
    if delta is None or delta.pct_change is None or delta.is_good is None:
        return "neutral", ""
    change = abs(delta.pct_change)
    word = {"up": "higher", "down": "lower"}.get(delta.direction, "about the same")
    basis = (f"{format_change(delta.pct_change).lstrip('+−-')} {word} than the previous {days} days"
             if delta.direction in ("up", "down") else f"About the same as the previous {days} days")
    if delta.is_good:
        return "good", basis
    if change < cfg.BASIC_WATCH_PCT:                      # slightly worse: still green, said honestly
        return "good", f"Only {basis[0].lower()}{basis[1:]}"
    return ("watch" if change < cfg.BASIC_BAD_PCT else "bad"), basis


def _sentence(key: str, k, kpis) -> str:
    v = k.formatted_compact
    if key == "spend":
        return f"You spent {v} on marketing."
    if key == "revenue":
        return f"Your marketing brought in {v} of revenue."
    if key == "roas":
        return f"For every ₹1 spent, you got {per_rupee(k.value)} back."
    if key == "cpl":
        return f"Getting one lead (a person who showed interest) cost {k.formatted} on average."
    if key == "cac":
        counted = " (counted as one conversion)" if "Conversions" in (k.note or "") else ""
        return f"Winning one customer{counted} cost {k.formatted} in marketing on average."
    if key == "leads":
        return f"{k.formatted} people showed interest (leads)."
    return f"{k.formatted} leads became customers."


def key_numbers(analysis, df: pd.DataFrame | None = None, targets: dict | None = None,
                pacing=None, limit: int = 5) -> list[KeyNumber]:
    """Up to `limit` headline numbers in plain words, with a colour and the reason for it:
    a target's status when one is set; spend -> budget pacing; otherwise the recent trend."""
    from analytics import targets as tg
    kpis = analysis.kpis
    comparison = recent_comparison(df, analysis.capabilities)
    deltas = comparison.deltas if comparison is not None else {}
    prof = getattr(analysis, "profitability", None)
    targets = tg.clean_targets(targets or {})
    out = []
    for key in _CARD_ORDER:
        k = kpis.get(key)
        if k is None or not k.available or k.value is None or len(out) >= limit:
            continue
        status, basis = _trend_status(deltas.get(key))
        if key in targets and key in tg.RATIO_TARGETS:
            label = tg.target_status(k.value, targets[key], key)
            if label in _TARGET_TO_STATUS:
                status = _TARGET_TO_STATUS[label]
                basis = f"Your target is {format_value(targets[key], k.fmt)}: {label.lower()}"
        elif key == "spend" and pacing is not None and getattr(pacing, "available", False):
            label = pacing.overall.get("status")
            if label in _PACING_TO_STATUS:
                status = _PACING_TO_STATUS[label]
                basis = f"This month's spending: {label.lower()} against the budget"
        elif key == "roas" and prof is not None and prof.available:
            breakeven = prof.totals.get("break_even_roas")
            if breakeven is not None and not pd.isna(breakeven) and k.value < breakeven:
                status = "bad"
                basis = f"Below the {per_rupee(breakeven)} needed to cover the cost of what was sold"
        value = per_rupee(k.value) if key == "roas" else k.formatted_compact   # as on the KPI cards
        out.append(KeyNumber(key, _TITLES[key], value, _sentence(key, k, kpis), status, basis))
    return out


# ---------------------------------------------------------------------------------------------
# Page sentences
# ---------------------------------------------------------------------------------------------

def overview_lines(analysis, df: pd.DataFrame | None = None) -> list[str]:
    """'How Are We Doing?': the period, spend, revenue, return, and the recent change."""
    k, meta = analysis.kpis, analysis.metadata
    lines = []
    period = f"Between {format_date(meta.get('date_min'))} and {format_date(meta.get('date_max'))}"
    if k["spend"].available and k["revenue"].available:
        lines.append(f"{period}, you spent {k['spend'].formatted_compact} on marketing and it brought in "
                     f"{k['revenue'].formatted_compact} of revenue.")
    elif k["spend"].available and k["leads"].available:
        lines.append(f"{period}, you spent {k['spend'].formatted_compact} on marketing and got "
                     f"{k['leads'].formatted} leads.")
    if k["roas"].available and k["roas"].value is not None:
        lines.append(f"For every ₹1 spent, you got {per_rupee(k['roas'].value)} back.")
    comparison = recent_comparison(df, analysis.capabilities)
    if comparison is not None:
        parts = []
        for key in ("revenue", "leads", "cpl", "roas"):
            d = comparison.deltas.get(key)
            if d is None or d.pct_change is None or d.direction not in ("up", "down"):
                continue
            verb = "rose" if d.direction == "up" else "fell"
            judge = "" if d.is_good is None else (" (good)" if d.is_good else " (not good)")
            parts.append(f"{plain_name(key)} {verb} {format_change(d.pct_change).lstrip('+−-')}{judge}")
        if parts:
            lines.append(f"In the last {cfg.BASIC_COMPARE_DAYS} days, compared with the {cfg.BASIC_COMPARE_DAYS} "
                         f"days before: " + "; ".join(parts) + ".")
    return lines


def profit_lines(prof) -> list[str]:
    if prof is None or not prof.available:
        return []
    t = prof.totals
    lines = [f"After the cost of what was sold, marketing made {money(t.get('contribution'))} of profit: "
             f"{format_value(t.get('profit_per_rupee'), 'money')} for every ₹1 spent."]
    counts = prof.status_counts or {}
    losing, total = counts.get("Loss-making", 0), sum(counts.values())
    if total:
        lines.append(f"{losing} of {total} paid campaigns lost money after the cost of what was sold"
                     + (f" ({money(prof.loss_spend)} of spend)." if losing and prof.loss_spend else "."))
    if getattr(prof, "assumed_margin_pct", None) is not None:
        lines.append(f"This uses the margin you entered ({prof.assumed_margin_pct:g}%), not a margin from the data.")
    return lines


def target_lines(result) -> list[str]:
    """Targets in plain words: how many are on track and which is furthest off."""
    if result is None or result.scorecard.empty:
        return []
    from analytics import targets as tg
    sc = result.scorecard
    rated = sc[~sc["status"].isin([tg.NOT_AVAILABLE, tg.PARTIAL])]
    if rated.empty:
        return ["Your targets cannot be checked yet (no full month or no actual value)."]
    lines = [f"{int((rated['status'] == tg.ON).sum())} of {len(rated)} targets are on track."]
    off = rated[rated["status"].isin([tg.OFF, tg.OVER, tg.UNDER])]
    if not off.empty:
        worst = off.loc[off["gap_pct"].abs().idxmax()] if off["gap_pct"].notna().any() else off.iloc[0]
        name = plain_name(worst["metric"]) if worst["metric"] in PLAIN_NAMES else str(worst["label"]).lower()
        lines.append(f"Furthest from target: {name}, {format_value(worst['actual'], worst['fmt'])} against a "
                     f"target of {format_value(worst['target'], worst['fmt'])}.")
    return lines


def pacing_lines(pacing) -> list[str]:
    if pacing is None or not getattr(pacing, "available", False):
        return []
    from analytics.targets import month_label
    p = pacing.overall
    ratio = p.get("pacing_ratio")
    if ratio is None:
        return []
    how = {"On Pace": "on plan", "Underspending": "slower than planned",
           "Overspending": "faster than planned"}.get(p.get("status"), "not rated yet")
    lines = [f"{month_label(p['month'])}: {format_value(ratio * 100, 'percent')} of the budget planned so far "
             f"has been spent, so spending is {how}."]
    if getattr(pacing, "assumed", False):
        lines.append("This uses the monthly budget you entered.")
    return lines


def _paid(table: pd.DataFrame, col: str) -> pd.DataFrame:
    from config.fields import channel_type
    if table is None or table.empty or col not in table:
        return pd.DataFrame()
    source = table[col] if col == "channel" else table.get("channel", table[col])
    return table[source.map(channel_type) != "owned"]


def money_lines(analysis) -> list[str]:
    """'Where the Money Goes': the biggest channel and what each paid channel returns."""
    ch = analysis.channels
    if ch is None or ch.empty or "spend" not in ch:
        return []
    paid = _paid(ch, "channel")
    lines = []
    if not paid.empty:
        top = paid.sort_values("spend", ascending=False).iloc[0]
        text = (f"{top['channel']} gets the most money: {format_value(top['spend_share_pct'], 'percent')} "
                f"of all spend")
        if "revenue_share_pct" in top and not pd.isna(top["revenue_share_pct"]):
            text += f", and it brings in {format_value(top['revenue_share_pct'], 'percent')} of revenue"
        lines.append(text + ".")
        if "roas" in paid and paid["roas"].notna().any():
            for r in paid.dropna(subset=["roas"]).sort_values("roas", ascending=False).itertuples():
                lines.append(f"{r.channel}: every ₹1 spent brought back {per_rupee(r.roas)}.")
        elif "cpl" in paid and paid["cpl"].notna().any():
            for r in paid.dropna(subset=["cpl"]).sort_values("cpl").itertuples():
                lines.append(f"{r.channel}: one lead cost {format_value(r.cpl, 'money')}.")
    owned = ch[~ch.index.isin(paid.index)] if not paid.empty else ch
    if not owned.empty and len(owned) < len(ch):
        names = ", ".join(owned["channel"].astype(str))
        lines.append(f"{names} uses your own audience (for example your email list) with mostly fixed costs, "
                     "so it is not compared with paid advertising.")
    return lines


def campaign_metric(campaigns: pd.DataFrame) -> str | None:
    for key in ("roas", "cpl"):
        if key in campaigns and campaigns[key].notna().any():
            return key
    return None


def best_and_weakest(analysis) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Top and bottom paid campaigns (at least BASIC_MIN_SPEND_SHARE % of spend each)."""
    from analytics.campaign import rank_campaigns
    paid = _paid(analysis.campaigns, "campaign")
    metric = campaign_metric(paid) if not paid.empty else None
    if metric is None:
        return pd.DataFrame(), pd.DataFrame(), None
    n = cfg.BASIC_TOP_CAMPAIGNS
    ranked = rank_campaigns(paid, metric, min_spend_share=cfg.BASIC_MIN_SPEND_SHARE)
    ranked = ranked.dropna(subset=[metric])
    best = ranked.head(n)
    weakest = ranked.iloc[::-1].head(n)
    weakest = weakest[~weakest["campaign"].isin(best["campaign"])]
    return best, weakest, metric


def campaign_line(row, metric: str) -> str:
    if metric == "roas":
        return f"{row['campaign']}: every ₹1 spent brought back {per_rupee(row['roas'])}."
    return f"{row['campaign']}: one lead cost {format_value(row['cpl'], 'money')}."


def top_alerts(analysis, n: int | None = None) -> list[dict]:
    """The biggest problem incidents, in plain words, with their estimated ₹ impact."""
    inc = analysis.incidents
    if inc is None or inc.empty:
        return []
    problems = inc[inc["sentiment"] == "negative"].sort_values("rank")
    out = []
    for r in problems.head(n or cfg.BASIC_TOP_ALERTS).itertuples():
        changes = []
        for f in (r.flags or [])[:3]:
            pct = f.get("change_pct")
            name = plain_name(f.get("metric", "")) if f.get("metric") in PLAIN_NAMES else \
                str(f.get("metric_label", "")).lower()
            if pct is None or pd.isna(pct):
                changes.append(f"{name} changed unusually")
            else:
                changes.append(f"{name} {'rose' if pct > 0 else 'fell'} {abs(pct):.0f}%")
        what = "; ".join(changes) if changes else "an unusual change"
        impact = (f"Estimated impact: about {money(r.impact_inr)} "
                  f"{'lost' if r.impact_direction == 'loss' else 'gained'}."
                  if r.impact_inr and r.impact_direction in ("loss", "gain")
                  else "The ₹ impact could not be estimated.")
        out.append({"title": f"{r.entity} ({_ENTITY_WORDS.get(r.entity_type, r.entity_type)})",
                    "period": format_date_range(r.period_start, r.period_end),
                    "what": what[0].upper() + what[1:] + ".", "impact": impact,
                    "impact_inr": r.impact_inr})
    return out


def ai_summary(insights: dict | None, n: int = 3) -> tuple[list[str], list[str]]:
    """(summary sentences, top recommendations) from saved AI insights, IDs removed, terms explained."""
    if not insights:
        return [], []
    raw = insights.get("executive_summary")
    items = raw if isinstance(raw, list) else ([raw] if raw else [])
    summary = [plain_ai_text(i.get("text", "")) for i in items if i.get("text")]
    recs = insights.get("recommendations") or []
    order = {"high": 0, "medium": 1, "low": 2}
    recs = sorted(recs, key=lambda r: order.get(str(r.get("priority", "")).lower(), 3))
    return summary, [plain_ai_text(r.get("text", "")) for r in recs[:n] if r.get("text")]


__all__ = ["GLOSSARY", "ABBREVIATIONS", "KeyNumber", "key_numbers", "overview_lines", "profit_lines",
           "target_lines", "pacing_lines", "money_lines", "best_and_weakest", "campaign_line", "top_alerts",
           "ai_summary", "plain_ai_text", "plain_words", "strip_ids", "unexplained_terms", "per_rupee",
           "plain_name", "recent_comparison"]
