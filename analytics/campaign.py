"""Campaign analytics: KPIs per campaign, shares, ranking, recent trend and active dates."""

from __future__ import annotations

import pandas as pd

from analytics.common import add_shares, campaign_key, primary_outcome, recent_change
from analytics.kpis import KPI_REGISTRY, group_kpis


def campaign_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per campaign. Columns: campaign, channel (if known), summed metrics, ratio KPIs
    (from sums), shares of spend/revenue/conversions/leads, first/last date, active days,
    and the last-4-weeks vs previous-4-weeks trend of the main outcome metric."""
    key = campaign_key(df)
    if key is None:
        return pd.DataFrame()
    table = group_kpis(df, key)
    table = add_shares(table)

    if "channel" in df:
        # The campaign's channel (most common one if a campaign spans several).
        channel = df.groupby(key)["channel"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
        table.insert(1, "channel", table[key].map(channel))
    if "date" in df:
        dates = df.groupby(key)["date"].agg(["min", "max", "nunique"])
        table["first_date"] = table[key].map(dates["min"])
        table["last_date"] = table[key].map(dates["max"])
        table["active_days"] = table[key].map(dates["nunique"])
        outcome = primary_outcome(df)
        if outcome:
            trend = recent_change(df, key, outcome).set_index(key)
            table["trend_metric"] = outcome
            table["trend_change_pct"] = table[key].map(trend["change_pct"])
            table["trend"] = table[key].map(trend["trend"]).fillna("n/a")
    return table.rename(columns={key: "campaign"})


def rank_campaigns(table: pd.DataFrame, metric: str, top: int | None = None,
                   ascending: bool | None = None, min_spend_share: float = 0.0) -> pd.DataFrame:
    """Rank campaigns by a metric. By default the best come first (using the KPI registry's
    higher-is-better flag, so for CPL the lowest is best). `min_spend_share` (%) ignores tiny
    campaigns whose ratios are unreliable."""
    if metric not in table:
        return table.iloc[0:0]
    if ascending is None:
        better = KPI_REGISTRY[metric].higher_is_better if metric in KPI_REGISTRY else True
        ascending = better is False
    subset = table[table[metric].notna()]
    if min_spend_share and "spend_share_pct" in subset:
        subset = subset[subset["spend_share_pct"] >= min_spend_share]
    ranked = subset.sort_values([metric, "campaign"], ascending=[ascending, True])
    ranked = ranked.assign(rank=range(1, len(ranked) + 1))
    return ranked.head(top) if top else ranked
