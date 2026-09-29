"""Build the compact evidence pack Gemini receives (SPEC §13: evidence, not raw data).

Raw data -> deterministic analysis -> compact evidence (this module) -> Gemini.

Every evidence item gets an ID (E01, E02, ...) that the AI must cite. Facts are written as
already-formatted text (₹4.2 Cr, 38.5%, 3.45x) so the AI can quote them exactly and the
evaluator can check every number it writes against the cited evidence. No raw rows are sent.
If the pack is larger than AI_CONTEXT_MAX_CHARS, the lowest-priority items are left out first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from analytics.kpis import KPI_REGISTRY
from config.fields import FIELD_BY_NAME, channel_type
from config.settings import AI_CONTEXT_MAX_CHARS, AI_MAX_INCIDENTS, AI_TOP_CAMPAIGNS
from utils.formatting import format_change, format_count, format_date, format_value

from config.settings import MIN_SPEND_SHARE_FOR_RANKING as MIN_SPEND_SHARE  # noqa: E402


@dataclass
class EvidenceItem:
    type: str
    title: str
    facts: dict[str, str]
    priority: int          # 1 = must keep ... 5 = first to drop
    id: str = ""
    # Exact numbers kept for calculations in code (e.g. spend for "₹ at stake"); not sent to AI.
    values: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "title": self.title, "facts": self.facts}


@dataclass
class EvidencePack:
    dataset: dict
    items: list[EvidenceItem]
    not_available: list[str]
    dropped_items: int = 0
    json_text: str = ""
    fingerprint: str = ""
    by_id: dict[str, EvidenceItem] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.json_text)


def _fmt(metric: str, value, compact: bool = True) -> str:
    fmt = KPI_REGISTRY[metric].fmt if metric in KPI_REGISTRY else "count"
    return format_value(value, fmt, compact=compact)


def _pct(value) -> str:
    return format_value(value, "percent")


def _row_facts(row: pd.Series, metrics: list[str]) -> dict[str, str]:
    facts = {}
    for m in metrics:
        if m in row and pd.notna(row[m]):
            if m.endswith("_share_pct"):
                facts[m.replace("_pct", "").replace("_", " ")] = _pct(row[m])
            elif m.endswith("_index"):
                facts[m.replace("_", " ")] = f"{row[m]:.0f} (100 = paid-media average)"
            elif m in ("trend_change_pct", "growth_pct"):
                facts["last 4 weeks vs previous 4"] = format_change(row[m])
            else:
                facts[KPI_REGISTRY[m].label if m in KPI_REGISTRY else m] = _fmt(m, row[m])
    return facts


def _raw_values(row: pd.Series) -> dict[str, float]:
    """Exact spend / ROAS of a campaign or channel, for "₹ at stake" (not sent to Gemini)."""
    return {m: float(row[m]) for m in ("spend", "roas") if m in row and pd.notna(row[m])}


def _channel_label(channel) -> str:
    return f"{channel} (owned channel)" if channel_type(channel) == "owned" else str(channel)


def _headline(table: pd.DataFrame) -> str | None:
    for m in ("roas", "cpl", "ctr"):
        if m in table and table[m].notna().any():
            return m
    return None


# ---------------------------------------------------------------------------------------------
# Evidence sections
# ---------------------------------------------------------------------------------------------

def _kpi_items(result) -> list[EvidenceItem]:
    facts = {}
    for k in result.kpis.values():
        if k.available and k.value is not None:
            label = "ROAS overall (all channels)" if k.key == "roas" else k.label
            facts[label] = k.formatted_compact
    paid_roas = (getattr(result, "paid_kpis", None) or {}).get("roas")
    if paid_roas is not None and paid_roas.available and paid_roas.value is not None:
        facts["ROAS paid media only (owned channels excluded)"] = paid_roas.formatted_compact
    notes = [k.note for k in result.kpis.values() if k.available and k.key in ("cac", "aov", "roi") and k.note]
    if notes:
        facts["notes"] = " ".join(notes)
    return [EvidenceItem("kpi", "Overall KPIs for the full dataset", facts, 1)]


def _quality_items(result) -> list[EvidenceItem]:
    q = result.quality_summary
    if q is None:
        return []
    facts = {"rows uploaded": format_count(q.rows_in), "rows analysed": format_count(q.rows_out),
             "duplicate rows removed": format_count(q.removed_duplicates)}
    for i, text in enumerate(q.limitations, start=1):
        facts[f"limitation {i}"] = text
    return [EvidenceItem("data_quality", "Data-quality limitations", facts, 1)]


def _not_available(result) -> list[str]:
    caps = result.capabilities
    return list(caps.limitations) if caps is not None else []


def _finding_items(result) -> list[EvidenceItem]:
    return [EvidenceItem("finding", f.title, {"finding": f.text}, 1 if i < 5 else 3)
            for i, f in enumerate(result.findings)]


def _channel_items(result) -> list[EvidenceItem]:
    ch = result.channels
    if ch.empty:
        return []
    metrics = ["spend", "spend_share_pct", "leads", "cpl", "conversions", "cac", "revenue",
               "revenue_share_pct", "roas", "ctr", "lead_to_conversion_rate", "cpl_index", "roas_index"]
    ordered = ch.sort_values("spend", ascending=False) if "spend" in ch else ch
    items = []
    for _, row in ordered.iterrows():
        kind = row.get("channel_type", "paid")
        facts = {"channel type": ("owned - own audience, mostly fixed costs; not comparable with paid "
                                  "media and not ranked" if kind == "owned" else "paid")}
        facts.update(_row_facts(row, metrics))
        items.append(EvidenceItem("channel", f"Channel: {row['channel']}", facts, 2,
                                  values=_raw_values(row)))
    return items


def _funnel_items(result) -> list[EvidenceItem]:
    f = result.funnel
    if f.empty:
        return []
    facts = {}
    for _, row in f.sort_values("stage_order").iterrows():
        value = format_value(row["value"], "money" if row["stage"] == "revenue" else "count", compact=True)
        rate = row["rate_from_previous"]
        facts[row["label"]] = value + ("" if pd.isna(rate) else f" ({_pct(rate)} of previous stage)")
    return [EvidenceItem("funnel", "Funnel (full dataset)", facts, 2)]


def _trend_items(result) -> list[EvidenceItem]:
    items = []
    monthly = result.trends.get("month")
    if monthly is not None and not monthly.empty:
        outcome = next((m for m in ("revenue", "conversions", "leads") if m in monthly), None)
        eff = _headline(monthly)
        facts = {}
        for _, row in monthly.iterrows():
            parts = [f"spend {_fmt('spend', row.get('spend'))}"] if "spend" in monthly else []
            if outcome:
                parts.append(f"{outcome} {_fmt(outcome, row[outcome])}")
            if eff:
                parts.append(f"{KPI_REGISTRY[eff].label} {_fmt(eff, row[eff])}")
            partial = " (partial month)" if row.get("days", 31) < 28 else ""
            facts[pd.Timestamp(f"{row['period']}-01").strftime("%b %Y") + partial] = ", ".join(parts)
        items.append(EvidenceItem("trend", "Monthly performance", facts, 2))
    if result.seasonality is not None:
        col = next((c for c in ("revenue_index", "conversions_index", "leads_index") if c in result.seasonality), None)
        if col:
            s = result.seasonality.sort_values(col, ascending=False)
            name = lambda m: pd.Timestamp(2000, int(m), 1).strftime("%B")  # noqa: E731
            facts = {"strongest months": ", ".join(f"{name(r.month_of_year)} ({getattr(r, col):.0f})" for r in s.head(3).itertuples()),
                     "weakest months": ", ".join(f"{name(r.month_of_year)} ({getattr(r, col):.0f})" for r in s.tail(3).itertuples()),
                     "meaning": f"{col.replace('_index', '')} index, 100 = an average month"}
            items.append(EvidenceItem("trend", "Seasonality", facts, 3))
    return items


def _campaign_items(result) -> list[EvidenceItem]:
    c = result.campaigns
    if c.empty:
        return []
    metrics = ["channel", "spend", "spend_share_pct", "leads", "cpl", "conversions", "cac", "revenue",
               "roas", "trend_change_pct"]
    items, seen = [], set()
    outcome = next((m for m in ("conversions", "leads") if m in c), None)
    if outcome:
        for _, row in c.sort_values(outcome, ascending=False).head(AI_TOP_CAMPAIGNS).iterrows():
            seen.add(row["campaign"])
            facts = {"channel": _channel_label(row.get("channel", ""))} | _row_facts(row, metrics[1:])
            items.append(EvidenceItem("campaign", f"Top campaign by {outcome}: {row['campaign']}", facts, 3,
                                      values=_raw_values(row)))
    metric = _headline(c)
    if metric and "spend_share_pct" in c:
        paid = c["channel"].map(channel_type) == "paid" if "channel" in c else True
        sizeable = c[(c["spend_share_pct"] >= MIN_SPEND_SHARE) & c[metric].notna() & paid]
        worst_first = KPI_REGISTRY[metric].higher_is_better is not False
        for _, row in sizeable.sort_values(metric, ascending=worst_first).head(AI_TOP_CAMPAIGNS).iterrows():
            if row["campaign"] in seen:
                continue
            facts = {"channel": _channel_label(row.get("channel", ""))} | _row_facts(row, metrics[1:])
            items.append(EvidenceItem("campaign", f"Weak campaign by {KPI_REGISTRY[metric].label}: "
                                      f"{row['campaign']}", facts, 3, values=_raw_values(row)))
    return items


def _segment_items(result) -> list[EvidenceItem]:
    items = []
    for i, (dim, table) in enumerate(result.segments.items()):
        if table.empty:
            continue
        metrics = ["spend", "revenue_share_pct", "leads", "cpl", "conversions", "cac", "aov", "roas",
                   "growth_pct"]
        facts = {}
        for _, row in table.iterrows():
            row_facts = _row_facts(row, metrics)
            facts[str(row["segment"])] = "; ".join(f"{k} {v}" for k, v in row_facts.items())
        items.append(EvidenceItem("segment", f"Performance by {FIELD_BY_NAME[dim].label.lower()}",
                                  facts, 3 if i == 0 else 4))
    return items


def _incident_items(result) -> list[EvidenceItem]:
    """Top incidents by estimated rupee impact, with their flags and related effects."""
    inc = getattr(result, "incidents", None)
    if inc is None or inc.empty:
        return []
    items = []
    for r in inc.head(AI_MAX_INCIDENTS).itertuples():
        facts = {
            "entity": f"{r.entity} ({r.entity_type})",
            "period": f"{format_date(r.period_start)} to {format_date(r.period_end)}",
            "metrics flagged": r.metrics,
            "estimated impact": (f"{format_value(r.impact_inr, 'money', compact=True)} "
                                 f"{r.impact_direction} ({r.impact_label})") if r.impact_direction in ("loss", "gain")
                                else "not estimated",
            "assessment": {"negative": "problem", "positive": "improvement", "check": "needs review"}[r.sentiment],
            "severity": r.severity,
        }
        for i, f in enumerate(r.flags[:3], start=1):
            change = "" if f["change_pct"] is None else f", change {format_change(f['change_pct'])}"
            facts[f"flag {i}"] = (f"{f['metric_label']}: expected {f['expected']}, observed {f['observed']}"
                                  f"{change}" + (f" ({f['note']})" if f.get("note") else ""))
        if r.related:
            shown = "; ".join(
                f"{f['entity']} {f['metric_label']} {format_change(f['change_pct'])}"
                + (" (lower cost not an improvement)" if f.get("note") else "")
                for f in r.related[:4])
            more = f" and {len(r.related) - 4} more" if len(r.related) > 4 else ""
            facts["related effects"] = shown + more
        items.append(EvidenceItem("incident", f"Incident {r.incident_id}: {r.entity}", facts,
                                  2 if r.rank <= 5 else 3))
    return items


def _measurement_items(result) -> list[EvidenceItem]:
    notes = list(getattr(result, "notes", []) or [])
    if not notes:
        return []
    return [EvidenceItem("measurement", "Measurement notes and limitations",
                         {f"note {i}": n for i, n in enumerate(notes, start=1)}, 1)]


# ---------------------------------------------------------------------------------------------
# U9 (prompt v6): planning evidence from the U4-U7 results. Only what the DATA says: the stored
# AnalysisResult parts never contain user inputs (targets, assumed margin, a typed monthly budget,
# scenarios), and analyses saved before U4-U7 (parts are None) simply have no such items.
# ---------------------------------------------------------------------------------------------

def _money(value) -> str:
    return format_value(value, "money", compact=True)


def _profitability_items(result) -> list[EvidenceItem]:
    p = getattr(result, "profitability", None)
    if p is None or not p.available or p.assumed_margin_pct is not None:   # an assumed margin: never sent
        return []
    t = p.totals
    facts = {"basis": p.basis_note,
             "total contribution (gross profit minus spend)": _money(t.get("contribution")),
             "profit per ₹1 of spend": format_value(t.get("profit_per_rupee"), "money"),
             "gross margin": _pct(t.get("gross_margin_pct")),
             "break-even ROAS overall": format_value(t.get("break_even_roas"), "ratio")}
    if p.status_counts:
        facts["paid campaigns by status"] = ", ".join(f"{s} {n}" for s, n in p.status_counts.items() if n)
    values = {}
    if p.loss_spend is not None:
        losers = p.loss_makers(3)
        if not losers.empty:
            facts["spend in loss-making paid campaigns"] = (f"{_money(p.loss_spend)} ({_pct(p.loss_spend_share_pct)}"
                                                            " of paid campaign spend)")
            facts["biggest loss-makers (₹ lost)"] = "; ".join(
                f"{r.campaign}: {_money(-r.contribution)} lost" for r in losers.itertuples())
            values = {f"loss:{r.campaign}": float(-r.contribution) for r in losers.itertuples()}
        else:
            near = p.campaigns[p.campaigns["status"] == "Near break-even"].sort_values("contribution").head(3)
            facts["loss-making campaigns"] = "none"
            if not near.empty:
                facts["closest to break-even (contribution)"] = "; ".join(
                    f"{r.campaign}: {_money(r.contribution)}" for r in near.itertuples())
    if not p.channels.empty:
        facts["paid channels: ROAS vs own break-even ROAS"] = "; ".join(
            f"{r.channel} {format_value(r.roas, 'ratio')} vs {format_value(r.break_even_roas, 'ratio')} "
            f"({r.status})" for r in p.channels.itertuples())
    return [EvidenceItem("profitability", "Profitability (gross profit minus spend)", facts, 2, values=values)]


def _pacing_items(result) -> list[EvidenceItem]:
    pr = getattr(result, "pacing", None)
    if pr is None or not pr.available or pr.assumed:        # a typed budget is a user input: never sent
        return []
    from analytics.pacing import PARTIAL
    from analytics.targets import month_label
    o = pr.overall
    ratio = None if o.get("pacing_ratio") is None else o["pacing_ratio"] * 100
    facts = {"budget source": "Budget column in the data",
             "latest month": f"{month_label(o['month'])} (data to {format_date(pr.as_of)}, day {o['day']} "
                             f"of {o['days_in_month']})",
             "status": o["status"],
             "spend to date vs plan to date": f"{_money(o['spend_to_date'])} vs {_money(o['planned_to_date'])} "
                                              f"({_pct(ratio)} of plan)",
             "month budget": _money(o.get("month_budget"))}
    if o.get("remaining_planned", 0) > 0 and o.get("projected_spend") is not None:
        facts["projected month-end spend (last 7 days' pace)"] = _money(o["projected_spend"])
    rated = pr.monthly[pr.monthly["status"] != PARTIAL] if not pr.monthly.empty else pr.monthly
    if not rated.empty:
        facts["full months: share of budget used"] = (f"{_pct(rated['budget_utilisation'].min())} to "
                                                      f"{_pct(rated['budget_utilisation'].max())}")
    ch = pr.channels
    if ch is not None and not ch.empty:
        off = ch[~ch["owned"] & (ch["status"] != "On Pace")].head(4)
        if not off.empty:
            facts["paid channels off pace"] = "; ".join(
                f"{r.channel} {_pct(r.pacing_pct)} of plan ({r.status})" for r in off.itertuples())
    return [EvidenceItem("pacing", "Budget pacing, latest month", facts, 2)]


def _forecast_items(result) -> list[EvidenceItem]:
    fr = getattr(result, "forecast", None)
    if fr is None or not fr.available or not fr.metrics:
        return []
    from analytics.forecast import LIMITS_NOTE
    facts = {}
    for m, mf in fr.metrics.items():
        fmt = KPI_REGISTRY[m].fmt
        fc = mf.forecast
        show = lambda v: format_value(v, fmt, compact=True)  # noqa: E731
        error = ("accuracy not measurable" if mf.wape is None
                 else f"past forecasts were off by about {_pct(mf.wape)}")
        facts[f"{KPI_REGISTRY[m].label}, next {fr.horizon} weeks"] = (
            f"{show(fc['low'].sum())} to {show(fc['high'].sum())} (central estimate {show(fc['forecast'].sum())}); "
            f"{error}")
    facts["range meaning"] = ("sum of each week's 10th-90th percentile range of past forecast errors; "
                              f"method chosen by a backtest on {fr.weeks} full weeks")
    facts["limits"] = LIMITS_NOTE
    return [EvidenceItem("forecast", f"Forecast outlook, next {fr.horizon} weeks (all channels)", facts, 2)]


def _marginal_cost_items(result) -> list[EvidenceItem]:
    rc = getattr(result, "response_curves", None)
    if rc is None or not rc.curves:
        return []
    from analytics.response_curves import PERFORMANCE
    facts, skipped = {}, []
    for name, cv in rc.curves.items():
        if not cv.usable:
            skipped.append(name)
            continue
        now = f"now {_money(cv.base_spend)} a week"
        if cv.status == PERFORMANCE:
            room = max(0.0, (cv.cap_conversions or 0) - cv.base_conversions) * (cv.cpa or 0)
            facts[name] = (f"paid per result: cost per conversion {_money(cv.cpa)} ({now}); capacity about "
                           f"{format_count(cv.cap_conversions)} conversions a week, so at most about {_money(room)} "
                           "more a week buys anything")
        else:
            mc = cv.marginal_cost(cv.base_spend)
            facts[name] = (f"next conversion costs {_money(mc)} ({now}); confidence {cv.confidence}"
                           + ("; returns limited to proportional, treat as an upper limit" if cv.clamped else ""))
    if not facts:
        return []
    if skipped:
        facts["not estimated (not enough spend variation)"] = ", ".join(skipped)
    facts["meaning"] = ("cost of one more conversion at today's weekly spend, from past weekly spend and "
                        "conversions (incident weeks left out); a past pattern, not a guarantee")
    return [EvidenceItem("marginal_cost", "Cost of the next conversion by paid channel", facts, 2)]


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------

def build_evidence(result, max_chars: int = AI_CONTEXT_MAX_CHARS, model: str = "") -> EvidencePack:
    meta = result.metadata
    dataset = {"name": meta.get("dataset_name"), "rows_analysed": format_count(meta.get("rows")),
               "period": f"{format_date(meta.get('date_min'))} to {format_date(meta.get('date_max'))}",
               "currency": "INR"}
    candidates = (_kpi_items(result) + _quality_items(result) + _measurement_items(result)
                  + _finding_items(result) + _channel_items(result) + _funnel_items(result)
                  + _trend_items(result) + _incident_items(result) + _campaign_items(result)
                  + _segment_items(result)
                  # U9: appended last so the IDs of the items above do not move.
                  + _profitability_items(result) + _marginal_cost_items(result)
                  + _pacing_items(result) + _forecast_items(result))
    not_available = _not_available(result)

    # Keep the most important items that fit; drop from the lowest priority upwards.
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i].priority, i))
    kept: set[int] = set()
    for i in order:
        trial = sorted(kept | {i})
        if len(_to_json(dataset, [candidates[j] for j in trial], not_available, assign=True)) <= max_chars:
            kept.add(i)
    items = [candidates[i] for i in sorted(kept)]
    json_text = _to_json(dataset, items, not_available, assign=True)
    pack = EvidencePack(dataset, items, not_available, dropped_items=len(candidates) - len(items),
                        json_text=json_text, by_id={it.id: it for it in items})
    from ai.cache import compute_fingerprint
    pack.fingerprint = compute_fingerprint(json_text, model)
    return pack


def _to_json(dataset: dict, items: list[EvidenceItem], not_available: list[str], assign: bool) -> str:
    if assign:
        for n, item in enumerate(items, start=1):
            item.id = f"E{n:02d}"
    body = {"dataset": dataset, "evidence": [it.as_dict() for it in items],
            "not_available": not_available}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
