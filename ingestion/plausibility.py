"""Plausibility warnings: do the mapped totals make marketing sense?

Run after mapping and before analysis. They never block the analysis: real data can be odd.
But a funnel that runs backwards (more leads than clicks) or a 160x ROAS usually means a
column is mapped to the wrong field, or the data is synthetic. Totals are summed from the raw
file with the same number parser the cleaner uses; ratios come from those totals (never an
average of row ratios), like everywhere else in the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.fields import channel_type
from config.settings import PLAUSIBLE_MAX_CTR_PCT, PLAUSIBLE_MAX_ROAS, PLAUSIBLE_MIN_LEAD_TO_CONV_PCT
from ingestion.mapper import MappingResult
from utils.formatting import format_count, format_inr, format_pct, format_ratio
from utils.parsing import parse_numbers

HEADLINE = "These numbers look unusual. Check the mapping; the data may also be synthetic."


@dataclass
class PlausibilityWarning:
    check: str          # short id, e.g. "leads_gt_clicks"
    message: str


def _total(df: pd.DataFrame, column: str) -> float:
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        series = parse_numbers(series)[0]
    return float(series.sum(skipna=True))


def check_plausibility(df: pd.DataFrame, mapping: MappingResult) -> list[PlausibilityWarning]:
    active = mapping.active
    totals = {f: _total(df, col) for f, col in active.items()
              if f in ("spend", "impressions", "reach", "clicks", "leads", "conversions", "revenue")}
    label = {f: f"'{col}'" for f, col in active.items()}
    out: list[PlausibilityWarning] = []

    def more_than(big: str, small: str, check: str, what: str):
        if big in totals and small in totals and totals[big] > totals[small] > 0:
            out.append(PlausibilityWarning(check,
                f"{what}: {label[big]} total {format_count(totals[big])} is more than "
                f"{label[small]} total {format_count(totals[small])}."))

    more_than("leads", "clicks", "leads_gt_clicks", "More leads than clicks")
    more_than("conversions", "leads", "conversions_gt_leads", "More conversions than leads")
    more_than("clicks", "impressions", "clicks_gt_impressions", "More clicks than impressions")
    more_than("reach", "impressions", "reach_gt_impressions",
              "Reach is higher than impressions (a person cannot be reached without an impression)")

    if totals.get("impressions", 0) > 0 and "clicks" in totals:
        ctr = totals["clicks"] / totals["impressions"] * 100
        if ctr > PLAUSIBLE_MAX_CTR_PCT:
            out.append(PlausibilityWarning("ctr_high",
                f"Overall CTR is {format_pct(ctr)} (above {PLAUSIBLE_MAX_CTR_PCT:.0f}% is very unusual)."))
    if totals.get("leads", 0) > 0 and "conversions" in totals:
        rate = totals["conversions"] / totals["leads"] * 100
        if rate < PLAUSIBLE_MIN_LEAD_TO_CONV_PCT:
            out.append(PlausibilityWarning("lead_to_conv_low",
                f"Lead-to-conversion rate is {format_pct(rate)} (below "
                f"{PLAUSIBLE_MIN_LEAD_TO_CONV_PCT}% is very unusual)."))
    if totals.get("spend", 0) > 0 and "revenue" in totals:
        roas = totals["revenue"] / totals["spend"]
        if roas > PLAUSIBLE_MAX_ROAS:
            out.append(PlausibilityWarning("roas_high",
                f"Overall ROAS is {format_ratio(roas)} (above {PLAUSIBLE_MAX_ROAS:.0f}x is rare - "
                "check that Revenue and Spend are the right columns)."))

    # Money spent on channels that are organic by name ("Organic Search", "SEO" ...).
    if "channel" in active and "spend" in active:
        channels = df[active["channel"]].astype(str)
        spend = df[active["spend"]]
        if not pd.api.types.is_numeric_dtype(spend):
            spend = parse_numbers(spend)[0]
        organic = channels.str.lower().str.contains("organic") | channels.map(
            lambda c: " ".join(c.lower().split()) in ("seo", "organic search", "organic social"))
        by_channel = spend[organic].groupby(channels[organic]).sum()
        for name, amount in by_channel[by_channel > 0].items():
            out.append(PlausibilityWarning("organic_spend",
                f"Spend of {format_inr(amount)} is recorded on '{name}', which sounds organic "
                "(organic channels usually have no ad spend)."))
    return out
