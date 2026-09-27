"""Anomaly incidents: turn many individual flags into a few business incidents.

A single real problem usually raises several flags: a broken checkout lowers the conversion
rate AND revenue; a tracking outage on one ad platform shows up on each of its campaigns and on
the channel. This module groups them so management sees each problem once:

1. Same entity + overlapping (or back-to-back) periods -> one incident with several metrics.
2. Related effects:
   - A platform/channel tracking outage (daily rule) explains the flags of its campaigns (and of
     channels made only of those campaigns) in the same period: they attach to it.
   - A channel/region incident with flags on 2+ of its campaigns at the same time is a broad
     issue: the campaign incidents attach to it as related effects.
   - A channel/region incident with a flag on exactly ONE of its campaigns is an echo of that
     campaign: it attaches to the campaign incident.
3. Contradictory labels are fixed: a lower CPC or CPL is NOT an improvement when conversions,
   the conversion rate or revenue collapsed in the same incident; it becomes "check".
4. Each incident gets an estimated rupee impact (money lost, overspent or at risk; or gained),
   computed from the data for its period, and incidents are ranked by it. The largest single
   effect is used, not the sum, because the flags of one incident describe the same money.
"""

from __future__ import annotations

import json


import pandas as pd

from analytics.common import campaign_key
from analytics.kpis import KPI_REGISTRY, safe_divide
from utils.formatting import format_date, format_value

AGGREGATE_TYPES = ("platform", "channel", "region")
COLLAPSE_METRICS = {"lead_to_conversion_rate", "revenue", "conversions"}
COST_METRICS = {"cpc", "cpl"}

INCIDENT_COLUMNS = ["incident_id", "rank", "entity_type", "entity", "period_start", "period_end",
                    "metrics", "n_flags", "n_related", "severity", "sentiment", "impact_inr",
                    "impact_direction", "impact_label", "headline", "campaigns", "flags", "related"]


# ---------------------------------------------------------------------------------------------
# Membership: which campaigns belong to each entity
# ---------------------------------------------------------------------------------------------

def _members(df: pd.DataFrame, key: str | None) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    if key is None:
        return out
    campaigns = df[key].astype(str)
    for c in campaigns.unique():
        out[("campaign", c)] = {c}
    for col in AGGREGATE_TYPES:
        if col in df:
            for value, part in campaigns.groupby(df[col].astype(str)):
                out[(col, value)] = set(part.unique())
    return out


def _entity_rows(df: pd.DataFrame, entity_type: str, entity: str, key: str | None) -> pd.DataFrame:
    col = key if entity_type == "campaign" else entity_type
    if col is None or col not in df:
        return df.iloc[0:0]
    return df[df[col].astype(str) == str(entity)]


# ---------------------------------------------------------------------------------------------
# Rupee impact of one flag
# ---------------------------------------------------------------------------------------------

def _sum(rows: pd.DataFrame, col: str) -> float:
    return float(rows[col].astype(float).sum()) if col in rows else float("nan")


def flag_impact(flag: dict, df: pd.DataFrame, key: str | None) -> tuple[float, str, str]:
    """(rupees, 'loss'|'gain', label) for one flag, estimated from the data in its period."""
    if flag.get("sentiment") == "check" and flag["metric"] in COST_METRICS:
        return 0.0, "none", "not counted (cost fell only because results fell)"
    all_rows = _entity_rows(df, flag["entity_type"], flag["entity"], key)
    start, end = pd.Timestamp(flag["period_start"]), pd.Timestamp(flag["period_end"])
    rows = all_rows[(all_rows["date"] >= start) & (all_rows["date"] <= end)]
    weeks = ((end - start).days + 1) / 7
    m, base, obs = flag["metric"], flag["baseline"], flag["observed"]
    if rows.empty or base is None or pd.isna(base):
        return 0.0, "none", "not estimated"

    if flag.get("method") == "tracking_outage_rule":
        details = json.loads(flag.get("details_json") or "{}")
        value = float(details.get("spend_during_outage", _sum(rows, "spend")))
        return value, "loss", "spend with no tracked results"
    if m == "spend":
        diff = _sum(rows, "spend") - base * weeks
        return abs(diff), "loss", ("spend above expected" if diff > 0 else "spend (and reach) below expected")
    if m == "revenue":
        diff = base * weeks - _sum(rows, "revenue")
        return abs(diff), ("loss" if diff > 0 else "gain"), ("revenue below expected" if diff > 0
                                                             else "revenue above expected")
    if m in COST_METRICS:
        volume_col = "leads" if m == "cpl" else "clicks"
        actual = safe_divide(_sum(rows, "spend"), _sum(rows, volume_col))
        if actual is None:
            return 0.0, "none", "not estimated"
        diff = (actual - base) * _sum(rows, volume_col)
        what = "per lead" if m == "cpl" else "per click"
        return abs(diff), ("loss" if diff > 0 else "gain"), (f"extra cost {what}" if diff > 0
                                                             else f"saving {what}")
    if m == "lead_to_conversion_rate":
        lost = (base - obs) / 100 * _sum(rows, "leads")
        value_per = (safe_divide(_sum(all_rows, "revenue"), _sum(all_rows, "conversions"))
                     if "revenue" in df else None)
        unit_label = "revenue (at average order value)"
        if value_per is None:                                   # no revenue: value at CAC
            value_per = safe_divide(_sum(all_rows, "spend"), _sum(all_rows, "conversions"))
            unit_label = "acquisition cost (CAC)"
        if value_per is None:
            return 0.0, "none", "not estimated"
        return abs(lost * value_per), ("loss" if lost > 0 else "gain"), (
            f"conversions {'lost' if lost > 0 else 'gained'}, valued at {unit_label}")
    if m == "lead_qualification_rate":
        share = 1 - safe_divide(obs, base) if base else 0
        value = _sum(rows, "spend") * share
        return abs(value), ("loss" if value > 0 else "gain"), ("spend on unqualified leads" if value > 0
                                                               else "better lead quality")
    if m == "clicks":
        cpc = safe_divide(_sum(all_rows, "spend"), _sum(all_rows, "clicks"))
        diff = (base * weeks - _sum(rows, "clicks")) * (cpc or 0)
        return abs(diff), ("loss" if diff > 0 else "gain"), ("value of missing clicks (at usual CPC)" if diff > 0
                                                             else "value of extra clicks (at usual CPC)")
    return 0.0, "none", "not estimated"


# ---------------------------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------------------------

def _overlaps(a_start, a_end, b_start, b_end, gap_days: int = 1) -> bool:
    return a_start <= b_end + pd.Timedelta(days=gap_days) and b_start <= a_end + pd.Timedelta(days=gap_days)


def _group_same_entity(flags: list[dict]) -> list[dict]:
    groups = []
    by_entity: dict[tuple, list[dict]] = {}
    for f in flags:
        by_entity.setdefault((f["entity_type"], f["entity"]), []).append(f)
    for (etype, entity), items in by_entity.items():
        items.sort(key=lambda f: f["period_start"])
        current = [items[0]]
        end = items[0]["period_end"]
        for f in items[1:]:
            if f["period_start"] <= end + pd.Timedelta(days=1):
                current.append(f)
                end = max(end, f["period_end"])
            else:
                groups.append(current)
                current, end = [f], f["period_end"]
        groups.append(current)
    return [{"entity_type": g[0]["entity_type"], "entity": g[0]["entity"],
             "start": min(f["period_start"] for f in g), "end": max(f["period_end"] for f in g),
             "flags": g, "related": [], "parent": None} for g in groups]


def _attach(child: dict, parent: dict) -> None:
    child["parent"] = parent
    parent["related"].extend(child["flags"] + child["related"])
    child["related"] = []


def build_incidents(anomalies: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if anomalies is None or anomalies.empty:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)
    key = campaign_key(df)
    members = _members(df, key)
    flags = [dict(r, flag_id=f"F{i:02d}") for i, r in enumerate(anomalies.to_dict("records"), start=1)]
    groups = _group_same_entity(flags)

    def member_set(g):
        return members.get((g["entity_type"], str(g["entity"])), set())

    def overlapping(a, b):
        return _overlaps(a["start"], a["end"], b["start"], b["end"])

    # (a) Tracking outages absorb what they explain.
    outages = [g for g in groups if any(f["method"] == "tracking_outage_rule" for f in g["flags"])]
    for outage in outages:
        scope = member_set(outage)
        for g in groups:
            if g is outage or g["parent"] is not None or g in outages:
                continue
            if overlapping(g, outage) and member_set(g) and member_set(g) <= scope:
                _attach(g, outage)

    # (b) Channel/region incidents: broad issue (2+ member campaigns flagged) or echo (exactly 1).
    aggregates = [g for g in groups if g["entity_type"] in ("channel", "region") and g["parent"] is None
                  and g not in outages]
    for agg in sorted(aggregates, key=lambda g: (g["entity_type"] != "channel", g["start"])):
        if agg["parent"] is not None:
            continue
        scope = member_set(agg)
        campaign_groups = [g for g in groups if g["entity_type"] == "campaign" and g["parent"] is None
                           and str(g["entity"]) in scope and overlapping(g, agg)]
        flagged = {str(g["entity"]) for g in campaign_groups}
        if len(flagged) >= 2:
            for g in campaign_groups:
                _attach(g, agg)
        elif len(flagged) == 1:
            _attach(agg, campaign_groups[0])

    top = [g for g in groups if g["parent"] is None]

    # (c) Contradictory labels and (d) rupee impact.
    rows = []
    for g in top:
        everything = g["flags"] + g["related"]
        collapsed = any(f["metric"] in COLLAPSE_METRICS and f["direction"] == "down" for f in everything) \
            or any(f["method"] == "tracking_outage_rule" for f in everything)
        for f in everything:
            if collapsed and f["metric"] in COST_METRICS and f["direction"] == "down" and f["sentiment"] == "positive":
                f["sentiment"] = "check"
                f["note"] = ("Lower cost is not an improvement here: conversions or revenue fell in the "
                             "same period.")
        impacts = []
        for f in g["flags"]:
            value, direction, label = flag_impact(f, df, key)
            f.update(impact_inr=value, impact_direction=direction, impact_label=label)
            impacts.append((value, direction, label, f))
        for f in g["related"]:
            value, direction, label = flag_impact(f, df, key)
            f.update(impact_inr=value, impact_direction=direction, impact_label=label)
        best = max(impacts, key=lambda t: t[0])
        own = g["flags"]
        sentiments = {f["sentiment"] for f in own}
        sentiment = "negative" if "negative" in sentiments else ("positive" if sentiments == {"positive"} else "check")
        if best[1] == "loss" and sentiment == "positive":
            sentiment = "check"
        scope = set(member_set(g))
        for f in g["related"]:
            scope |= members.get((f["entity_type"], str(f["entity"])), set())
        rows.append({
            "entity_type": g["entity_type"], "entity": g["entity"],
            "period_start": g["start"], "period_end": g["end"],
            "metrics": ", ".join(dict.fromkeys(_metric_label(f["metric"]) for f in own)),
            "n_flags": len(own), "n_related": len(g["related"]),
            "severity": "high" if any(f["severity"] == "high" for f in own + g["related"]) else "medium",
            "sentiment": sentiment,
            "impact_inr": float(best[0]), "impact_direction": best[1], "impact_label": best[2],
            "headline": _headline(g, best),
            "campaigns": sorted(scope),
            "flags": [_flag_summary(f) for f in own],
            "related": [_flag_summary(f) for f in g["related"]],
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["impact_inr", "period_start"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(0, "incident_id", [f"I{i:02d}" for i in range(1, len(out) + 1)])
    return out[INCIDENT_COLUMNS]


def _metric_label(metric: str) -> str:
    return KPI_REGISTRY[metric].label if metric in KPI_REGISTRY else metric


def _flag_summary(f: dict) -> dict:
    fmt = KPI_REGISTRY[f["metric"]].fmt if f["metric"] in KPI_REGISTRY else "count"
    return {"flag_id": f["flag_id"], "entity_type": f["entity_type"], "entity": f["entity"],
            "metric": f["metric"], "metric_label": _metric_label(f["metric"]),
            "period_start": pd.Timestamp(f["period_start"]).strftime("%Y-%m-%d"),
            "period_end": pd.Timestamp(f["period_end"]).strftime("%Y-%m-%d"),
            "expected": format_value(f["baseline"], fmt), "observed": format_value(f["observed"], fmt),
            "change_pct": None if f["pct_change"] is None or pd.isna(f["pct_change"]) else round(float(f["pct_change"]), 1),
            "severity": f["severity"], "sentiment": f["sentiment"], "method": f["method"],
            "impact_inr": round(float(f.get("impact_inr", 0.0)), 0),
            "impact_label": f.get("impact_label", ""), "impact_direction": f.get("impact_direction", "none"),
            "note": f.get("note", ""),
            "description": f["description"]}


def _headline(g: dict, best) -> str:
    value, direction, label, flag = best
    period = f"{format_date(g['start'])} to {format_date(g['end'])}"
    metrics = ", ".join(dict.fromkeys(_metric_label(f["metric"]) for f in g["flags"]))
    text = f"{g['entity']} ({g['entity_type']}), {period}: unusual {metrics}."
    if value > 0 and direction in ("loss", "gain"):
        text += f" Estimated impact {format_value(value, 'money')} ({label})."
    if g["related"]:
        text += f" {len(g['related'])} related effect(s) on other campaigns/channels/regions."
    return text


def incidents_for_campaign(incidents: pd.DataFrame, campaign: str, start, end) -> pd.DataFrame:
    """Incidents that touch a campaign (directly or through its platform/channel/region)
    and overlap [start, end]."""
    if incidents is None or incidents.empty:
        return incidents
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = incidents.apply(lambda r: campaign in r["campaigns"] and r["period_end"] >= start
                           and r["period_start"] <= end, axis=1)
    return incidents[mask]


def match_planted(incident: pd.Series, entity_type: str, entity: str, start, end) -> bool:
    """True if the incident itself, one of its flags or one of its related effects is on this
    entity and overlaps the period (used for recall at incident level)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for f in [dict(entity_type=incident["entity_type"], entity=incident["entity"],
                   period_start=incident["period_start"], period_end=incident["period_end"])] \
            + list(incident["flags"]) + list(incident["related"]):
        if f["entity_type"] == entity_type and str(f["entity"]) == str(entity) and \
                pd.Timestamp(f["period_start"]) <= end and pd.Timestamp(f["period_end"]) >= start:
            return True
    return False


__all__ = ["build_incidents", "flag_impact", "incidents_for_campaign", "match_planted", "INCIDENT_COLUMNS"]
