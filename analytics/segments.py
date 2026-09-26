"""One generic group analysis, used for customer segments, geography, products and customer type.

For each group: volume, conversion, revenue, CAC, AOV, ROAS (where the data allows),
contribution (share of totals) and growth (last 4 weeks vs the previous 4).
"""

from __future__ import annotations

import pandas as pd

from analytics.common import add_shares, primary_outcome, recent_change
from analytics.kpis import group_kpis

# Dimension -> the kind of analysis it belongs to (for labels in the dashboard/reports).
SEGMENT_DIMENSIONS = {
    "customer_segment": "segment", "customer_type": "segment", "new_returning": "segment",
    "age": "segment", "gender": "segment",
    "region": "geography", "state": "geography", "city": "geography", "country": "geography",
    "market": "geography", "city_tier": "geography",
    "product_category": "product", "product": "product", "sku": "product",
}


def available_dimensions(df: pd.DataFrame) -> list[str]:
    return [d for d in SEGMENT_DIMENSIONS if d in df and df[d].notna().any()]


def segment_table(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """KPIs, contribution and growth for each value of `dimension`."""
    if dimension not in df:
        return pd.DataFrame()
    table = add_shares(group_kpis(df, dimension))
    outcome = primary_outcome(df)
    if outcome and "date" in df:
        growth = recent_change(df, dimension, outcome).set_index(dimension)
        table["growth_metric"] = outcome
        table["growth_pct"] = table[dimension].map(growth["change_pct"])
        table["growth_trend"] = table[dimension].map(growth["trend"]).fillna("n/a")
    table.insert(0, "dimension", dimension)
    return table.rename(columns={dimension: "segment"})


def all_segment_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {d: segment_table(df, d) for d in available_dimensions(df)}
