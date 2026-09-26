"""Funnel analytics: how many people move from each stage to the next, and where they drop off.

Only stages that exist in the data are shown (SPEC §19: never fabricate missing information).
"""

from __future__ import annotations

import pandas as pd

from analytics.common import channel_key
from analytics.kpis import safe_divide

FUNNEL_STAGES = [("impressions", "Impressions"), ("clicks", "Clicks"), ("leads", "Leads"),
                 ("qualified_leads", "Qualified leads"), ("opportunities", "Opportunities"),
                 ("conversions", "Conversions")]
# Revenue is shown at the end of the funnel as a value, not a count, so it gets no rate.
VALUE_STAGE = ("revenue", "Revenue")


def available_stages(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(c, label) for c, label in FUNNEL_STAGES if c in df and df[c].notna().any()]


def funnel_table(df: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    """Stage totals, stage-to-stage rate (%) and drop-off (%) for the rows in `df`."""
    stages = available_stages(df)
    if len(stages) < 2:
        return pd.DataFrame()
    rows, previous = [], None
    for order, (col, label) in enumerate(stages, start=1):
        value = float(df[col].astype(float).sum())
        rate = safe_divide(value, previous, 100.0) if previous is not None else None
        rows.append({"scope": scope, "stage_order": order, "stage": col, "label": label,
                     "value": value, "rate_from_previous": rate,
                     "drop_off_pct": None if rate is None else 100.0 - rate})
        previous = value
    if "revenue" in df and df["revenue"].notna().any():
        rows.append({"scope": scope, "stage_order": len(rows) + 1, "stage": VALUE_STAGE[0],
                     "label": VALUE_STAGE[1], "value": float(df["revenue"].astype(float).sum()),
                     "rate_from_previous": None, "drop_off_pct": None})
    return pd.DataFrame(rows)


def funnel_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    key = channel_key(df)
    if key is None:
        return pd.DataFrame()
    parts = [funnel_table(part, str(name)) for name, part in df.groupby(key)]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def biggest_drop_off(funnel: pd.DataFrame, skip_first: bool = True) -> pd.Series | None:
    """The stage transition that loses the largest share of people. Impressions -> clicks is
    skipped by default: it is always tiny (a normal CTR is ~1%) and would always 'win'."""
    rates = funnel[funnel["rate_from_previous"].notna()]
    if skip_first and len(rates) and rates.iloc[0]["stage"] == "clicks":
        rates = rates.iloc[1:]
    if rates.empty:
        return None
    return rates.loc[rates["rate_from_previous"].idxmin()]
