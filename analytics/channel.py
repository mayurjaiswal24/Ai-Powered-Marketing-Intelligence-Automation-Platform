"""Channel analytics: compare channels (or platforms) with each other and with the average."""

from __future__ import annotations

import pandas as pd

from analytics.common import add_shares, channel_key
from analytics.kpis import compute_kpis, group_kpis

# Efficiency indices: channel value / overall value x 100. 100 = same as the overall average.
# For cost KPIs (CPC, CPL, CAC) below 100 is better; for the others above 100 is better.
INDEX_KPIS = ["ctr", "cpc", "cpl", "lead_to_conversion_rate", "cac", "roas"]


def channel_table(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """One row per channel with KPIs from sums, shares and efficiency indices vs overall."""
    key = by or channel_key(df)
    if key is None:
        return pd.DataFrame()
    table = add_shares(group_kpis(df, key))
    overall = compute_kpis(df)
    for kpi in INDEX_KPIS:
        if kpi in table and overall.get(kpi) and overall[kpi].value:
            table[f"{kpi}_index"] = table[kpi] / overall[kpi].value * 100
    return table.rename(columns={key: "channel"})
