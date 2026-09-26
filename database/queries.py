"""Readable analytical SQL on the stored clean data.

These queries return TOTALS only. Ratios (CTR, CPL, ROAS, ...) are deliberately not computed
here: every KPI formula lives in analytics/kpis.py, so a ratio can never be calculated two
different ways. SUM() skips missing (NULL) values, exactly like pandas' sum().
"""

from __future__ import annotations

import pandas as pd

from database.repository import session

CHANNEL_TOTALS_SQL = """
SELECT
    channel,
    COUNT(*)              AS rows,
    SUM(spend)            AS spend,
    SUM(impressions)      AS impressions,
    SUM(clicks)           AS clicks,
    SUM(leads)            AS leads,
    SUM(conversions)      AS conversions,
    SUM(revenue)          AS revenue
FROM marketing_records
WHERE run_id = ?
GROUP BY channel
ORDER BY spend DESC
"""

MONTHLY_SPEND_REVENUE_SQL = """
SELECT
    month,
    SUM(spend)    AS spend,
    SUM(revenue)  AS revenue
FROM marketing_records
WHERE run_id = ? AND month IS NOT NULL
GROUP BY month
ORDER BY month
"""

TOP_CAMPAIGNS_BY_CONVERSIONS_SQL = """
SELECT
    campaign_name,
    channel,
    SUM(conversions)  AS conversions,
    SUM(spend)        AS spend,
    SUM(revenue)      AS revenue
FROM marketing_records
WHERE run_id = ?
GROUP BY campaign_name, channel
ORDER BY conversions DESC, campaign_name
LIMIT ?
"""


def channel_totals(run_id: int, db_path=None) -> pd.DataFrame:
    """Spend, impressions, clicks, leads, conversions and revenue per channel."""
    with session(db_path) as conn:
        return pd.read_sql(CHANNEL_TOTALS_SQL, conn, params=(run_id,))


def monthly_spend_revenue(run_id: int, db_path=None) -> pd.DataFrame:
    """Total spend and revenue per calendar month (YYYY-MM)."""
    with session(db_path) as conn:
        return pd.read_sql(MONTHLY_SPEND_REVENUE_SQL, conn, params=(run_id,))


def top_campaigns_by_conversions(run_id: int, limit: int = 10, db_path=None) -> pd.DataFrame:
    """The campaigns with the most conversions, with their spend and revenue."""
    with session(db_path) as conn:
        return pd.read_sql(TOP_CAMPAIGNS_BY_CONVERSIONS_SQL, conn, params=(run_id, limit))
