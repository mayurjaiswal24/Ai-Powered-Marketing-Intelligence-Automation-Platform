"""Deterministic analytical findings: plain-English statements produced by rules, not by AI.

Each finding is labelled "Analytical finding" (SPEC §16: facts and statistical findings are kept
apart from AI interpretation) and carries the exact numbers behind it, each pointing at the
table cell it came from, so any finding can be checked against the dashboard and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analytics.funnel import biggest_drop_off
from analytics.kpis import KPI_REGISTRY
from utils.formatting import format_pct, format_value

FINDING_TYPE = "Analytical finding"
MIN_SPEND_SHARE_FOR_RANKING = 3.0      # % of spend; smaller campaigns have unreliable ratios
MIN_SPEND_SHARE_FOR_MOVERS = 2.0
CONCENTRATION_SHARE = 40.0             # % of spend in one channel worth pointing out


@dataclass
class Finding:
    id: str
    category: str                 # performance / concentration / funnel / trend / segment
    title: str
    text: str
    entity: str | None
    metric: str | None
    priority: int                 # lower = more important
    numbers: list[dict] = field(default_factory=list)   # {name, table, key_column, key, column, value}
    type: str = FINDING_TYPE


def _num(name: str, table: str, key_column: str, key, column: str, frame: pd.DataFrame) -> dict:
    row = frame.loc[frame[key_column] == key]
    value = row[column].iloc[0]
    return {"name": name, "table": table, "key_column": key_column, "key": key,
            "column": column, "value": None if pd.isna(value) else float(value)}


def _fmt(metric: str, value) -> str:
    fmt = KPI_REGISTRY[metric].fmt if metric in KPI_REGISTRY else "count"
    return format_value(value, fmt)


def _headline_metric(table: pd.DataFrame) -> str | None:
    """ROAS when revenue exists; otherwise CPL; otherwise CTR."""
    for metric in ("roas", "cpl", "ctr"):
        if metric in table and table[metric].notna().any():
            return metric
    return None


def generate_findings(tables: dict[str, pd.DataFrame]) -> list[Finding]:
    findings: list[Finding] = []
    kpis = tables.get("kpis", pd.DataFrame())
    channels = tables.get("channels", pd.DataFrame())
    campaigns = tables.get("campaigns", pd.DataFrame())
    funnel = tables.get("funnel", pd.DataFrame())

    def overall(metric):
        if kpis.empty or metric not in set(kpis["kpi"]):
            return None
        return _num(f"overall {metric}", "kpis", "kpi", metric, "value", kpis)

    # 1. Spend concentration in one channel.
    if not channels.empty and "spend_share_pct" in channels and len(channels) > 1:
        top = channels.loc[channels["spend_share_pct"].idxmax()]
        if top["spend_share_pct"] >= CONCENTRATION_SHARE:
            outcome = next((m for m in ("revenue", "conversions", "leads")
                            if f"{m}_share_pct" in channels), None)
            nums = [_num("spend share %", "channels", "channel", top["channel"], "spend_share_pct", channels)]
            text = f"{top['channel']} takes {format_pct(top['spend_share_pct'])} of all spend"
            if outcome:
                nums.append(_num(f"{outcome} share %", "channels", "channel", top["channel"],
                                 f"{outcome}_share_pct", channels))
                text += f" and delivers {format_pct(top[f'{outcome}_share_pct'])} of {outcome}"
            findings.append(Finding("F-concentration", "concentration",
                                    f"Spend is concentrated in {top['channel']}", text + ".",
                                    top["channel"], "spend", 1, nums))

    # 2-3. Best and weakest channel on the headline efficiency metric.
    metric = _headline_metric(channels) if not channels.empty else None
    if metric and len(channels) > 1:
        better_high = KPI_REGISTRY[metric].higher_is_better
        valid = channels[channels[metric].notna()]
        best = valid.loc[valid[metric].idxmax() if better_high else valid[metric].idxmin()]
        worst = valid.loc[valid[metric].idxmin() if better_high else valid[metric].idxmax()]
        ov = overall(metric)
        label = KPI_REGISTRY[metric].label
        for fid, row, word, prio in (("F-best-channel", best, "best", 2),
                                     ("F-weakest-channel", worst, "weakest", 3)):
            nums = [_num(f"channel {metric}", "channels", "channel", row["channel"], metric, channels)]
            text = f"{row['channel']} has the {word} {label} of all channels ({_fmt(metric, row[metric])})"
            if ov:
                nums.append(ov)
                text += f", against {_fmt(metric, ov['value'])} overall"
            findings.append(Finding(fid, "performance", f"{word.capitalize()} channel by {label}: "
                                    f"{row['channel']}", text + ".", row["channel"], metric, prio, nums))

    # 4. Funnel bottleneck (largest drop after the click stage).
    if not funnel.empty:
        step = biggest_drop_off(funnel)
        if step is not None:
            prev_label = funnel.loc[funnel["stage_order"] == step["stage_order"] - 1, "label"].iloc[0]
            nums = [_num("stage conversion %", "funnel", "stage", step["stage"], "rate_from_previous", funnel)]
            findings.append(Finding(
                "F-funnel-bottleneck", "funnel", f"Biggest funnel drop: {prev_label} to {step['label']}",
                f"Only {format_pct(step['rate_from_previous'])} of {prev_label.lower()} become "
                f"{step['label'].lower()}, the largest drop-off after the click stage.",
                None, step["stage"], 4, nums))

    if not campaigns.empty:
        # 5. Top campaign by the main outcome.
        outcome = next((m for m in ("conversions", "leads", "clicks") if m in campaigns), None)
        if outcome:
            top = campaigns.loc[campaigns[outcome].idxmax()]
            nums = [_num(outcome, "campaigns", "campaign", top["campaign"], outcome, campaigns)]
            text = f"{top['campaign']} produced the most {outcome} ({_fmt(outcome, top[outcome])})"
            if f"{outcome}_share_pct" in campaigns:
                nums.append(_num(f"{outcome} share %", "campaigns", "campaign", top["campaign"],
                                 f"{outcome}_share_pct", campaigns))
                text += f", {format_pct(top[f'{outcome}_share_pct'])} of the total"
            findings.append(Finding("F-top-campaign", "performance", f"Top campaign by {outcome}",
                                    text + ".", top["campaign"], outcome, 5, nums))

        # 6. Weakest sizeable campaign on the headline metric.
        metric = _headline_metric(campaigns)
        if metric and "spend_share_pct" in campaigns:
            big = campaigns[(campaigns["spend_share_pct"] >= MIN_SPEND_SHARE_FOR_RANKING)
                            & campaigns[metric].notna()]
            if len(big) > 1:
                better_high = KPI_REGISTRY[metric].higher_is_better
                worst = big.loc[big[metric].idxmin() if better_high else big[metric].idxmax()]
                label = KPI_REGISTRY[metric].label
                nums = [_num(f"campaign {metric}", "campaigns", "campaign", worst["campaign"], metric, campaigns),
                        _num("spend share %", "campaigns", "campaign", worst["campaign"], "spend_share_pct", campaigns)]
                text = (f"{worst['campaign']} has the weakest {label} among campaigns with at least "
                        f"{MIN_SPEND_SHARE_FOR_RANKING:.0f}% of spend ({_fmt(metric, worst[metric])}, "
                        f"{format_pct(worst['spend_share_pct'])} of spend)")
                ov = overall(metric)
                if ov:
                    nums.append(ov)
                    text += f"; overall {label} is {_fmt(metric, ov['value'])}"
                findings.append(Finding("F-weakest-campaign", "performance",
                                        f"Weakest sizeable campaign by {label}", text + ".",
                                        worst["campaign"], metric, 6, nums))

        # 7-8. Biggest recent movers (last 4 weeks vs previous 4).
        if "trend_change_pct" in campaigns and "spend_share_pct" in campaigns:
            movers = campaigns[(campaigns["spend_share_pct"] >= MIN_SPEND_SHARE_FOR_MOVERS)
                               & campaigns["trend_change_pct"].notna()]
            if len(movers):
                tm = movers["trend_metric"].iloc[0]
                up = movers.loc[movers["trend_change_pct"].idxmax()]
                down = movers.loc[movers["trend_change_pct"].idxmin()]
                for fid, row, prio in (("F-riser", up, 7), ("F-faller", down, 8)):
                    change = row["trend_change_pct"]
                    if (fid == "F-riser" and change <= 0) or (fid == "F-faller" and change >= 0):
                        continue
                    nums = [_num(f"{tm} change %", "campaigns", "campaign", row["campaign"],
                                 "trend_change_pct", campaigns)]
                    verb = "grew" if change > 0 else "fell"
                    findings.append(Finding(
                        fid, "trend", f"Biggest {'riser' if change > 0 else 'faller'}: {row['campaign']}",
                        f"{row['campaign']}: {tm} {verb} {abs(change):.1f}% in the last 4 weeks "
                        "compared with the 4 weeks before.", row["campaign"], tm, prio, nums))

    # 9. Segment with the best efficiency (first available segment-type dimension).
    for name, seg in tables.items():
        if not name.startswith("segment_") or seg.empty or len(seg) < 2:
            continue
        metric = _headline_metric(seg)
        if metric is None:
            continue
        better_high = KPI_REGISTRY[metric].higher_is_better
        valid = seg[seg[metric].notna()]
        best = valid.loc[valid[metric].idxmax() if better_high else valid[metric].idxmin()]
        dim = name.removeprefix("segment_").replace("_", " ")
        label = KPI_REGISTRY[metric].label
        nums = [_num(f"segment {metric}", name, "segment", best["segment"], metric, seg)]
        text = f"By {dim}, {best['segment']} has the best {label} ({_fmt(metric, best[metric])})"
        if "cac" in seg and pd.notna(best.get("cac")):
            nums.append(_num("segment cac", name, "segment", best["segment"], "cac", seg))
            text += f" with a CAC of {_fmt('cac', best['cac'])}"
        findings.append(Finding(f"F-best-{name}", "segment", f"Best {dim}: {best['segment']}",
                                text + ".", best["segment"], metric, 9, nums))
        break

    return sorted(findings, key=lambda f: f.priority)
