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

MIN_SPEND_SHARE = 3.0   # % of spend; smaller campaigns have unreliable ratios


@dataclass
class EvidenceItem:
    type: str
    title: str
    facts: dict[str, str]
    priority: int          # 1 = must keep ... 5 = first to drop
    id: str = ""

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
    facts = {k.label: k.formatted_compact for k in result.kpis.values() if k.available and k.value is not None}
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
        items.append(EvidenceItem("channel", f"Channel: {row['channel']}", facts, 2))
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
            items.append(EvidenceItem("campaign", f"Top campaign by {outcome}: {row['campaign']}", facts, 3))
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
                                      f"{row['campaign']}", facts, 3))
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
                  + _segment_items(result))
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
