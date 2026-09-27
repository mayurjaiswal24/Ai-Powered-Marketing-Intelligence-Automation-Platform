"""Channel analytics: compare channels (or platforms) with each other and with the average."""

from __future__ import annotations

import pandas as pd

from analytics.common import add_shares, channel_key
from analytics.kpis import compute_kpis, group_kpis
from config.fields import channel_type

# Efficiency indices: channel value / overall value x 100. 100 = same as the overall average.
# For cost KPIs (CPC, CPL, CAC) below 100 is better; for the others above 100 is better.
INDEX_KPIS = ["ctr", "cpc", "cpl", "lead_to_conversion_rate", "cac", "roas"]


def channel_table(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """One row per channel with KPIs from sums, shares, channel type (paid / owned) and
    efficiency indices.

    Indices compare each PAID channel with the paid-media average (100 = average). Owned
    channels (e.g. email to the company's own list) get no index: their costs are mostly fixed
    and their audience already knows the brand, so ranking them against paid media would
    mislead. They are reported separately.
    """
    key = by or channel_key(df)
    if key is None:
        return pd.DataFrame()
    table = add_shares(group_kpis(df, key))
    table.insert(1, "channel_type", table[key].map(channel_type))
    paid_rows = df[df[key].map(channel_type) == "paid"]
    overall = compute_kpis(paid_rows) if not paid_rows.empty else {}
    is_paid = table["channel_type"] == "paid"
    for kpi in INDEX_KPIS:
        if kpi in table and overall.get(kpi) and overall[kpi].value:
            table[f"{kpi}_index"] = (table[kpi] / overall[kpi].value * 100).where(is_paid)
    return table.rename(columns={key: "channel"})


def paid_media_kpis(df: pd.DataFrame, by: str | None = None) -> dict:
    """KPIs of the paid channels only (owned channels such as email left out), from sums like
    every other KPI. Empty when there is no channel column or no paid channel."""
    key = by or channel_key(df)
    if key is None:
        return {}
    paid_rows = df[df[key].map(channel_type) == "paid"]
    return compute_kpis(paid_rows) if not paid_rows.empty else {}


def paid_channels(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["channel_type"] == "paid"] if "channel_type" in table else table


def owned_channels(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["channel_type"] == "owned"] if "channel_type" in table else table.iloc[0:0]
