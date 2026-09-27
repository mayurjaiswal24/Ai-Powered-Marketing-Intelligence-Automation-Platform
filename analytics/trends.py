"""Time-series analytics: daily / weekly / monthly totals and KPIs, growth and rolling averages."""

from __future__ import annotations

import pandas as pd

from analytics.kpis import group_kpis

GRAIN_COLUMN = {"day": "date", "week": "week_start", "month": "month"}
GROWTH_METRICS = ["spend", "leads", "conversions", "revenue"]
from config.settings import MIN_MONTHS_FOR_SEASONALITY  # noqa: E402
from config.settings import ROLLING_AVERAGE_WEEKS as ROLLING_WEEKS  # noqa: E402


def time_series(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Totals and ratio KPIs per period. `days` = number of distinct dates with data in the
    period, so charts can mark partial weeks/months at the edges of the data."""
    col = GRAIN_COLUMN[grain]
    if col not in df or df[col].isna().all():
        return pd.DataFrame()
    table = group_kpis(df[df[col].notna()], col).rename(columns={col: "period"})
    table["days"] = table["period"].map(df.groupby(col)["date"].nunique())
    table.insert(0, "grain", grain)

    if grain in ("week", "month"):
        growth_name = "wow" if grain == "week" else "mom"
        for m in GROWTH_METRICS:
            if m in table:
                table[f"{m}_{growth_name}_pct"] = table[m].pct_change(fill_method=None) * 100
    if grain == "week":
        for m in GROWTH_METRICS:
            if m in table:
                table[f"{m}_rolling_{ROLLING_WEEKS}w"] = table[m].rolling(ROLLING_WEEKS).mean()
    return table


def all_time_series(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {g: time_series(df, g) for g in GRAIN_COLUMN}
    return {g: t for g, t in out.items() if not t.empty}


def seasonality_index(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Month-of-year index (100 = average month) for the main metrics.

    Only computed with at least 12 full months of data: with less, a "seasonal" pattern cannot
    be told apart from growth or one-off events.
    """
    if "date" not in df or df["date"].isna().all():
        return None, "Seasonality is unavailable because there is no date column."
    monthly = df.groupby(df["date"].dt.to_period("M"))
    days = monthly["date"].nunique()
    full = [p for p, d in days.items() if d == p.days_in_month]
    if len(full) < MIN_MONTHS_FOR_SEASONALITY:
        return None, (f"Insufficient history for a seasonality index: {len(full)} full month(s) "
                      f"of data, at least {MIN_MONTHS_FOR_SEASONALITY} are needed.")
    metrics = [m for m in GROWTH_METRICS if m in df and df[m].notna().any()]
    sums = monthly[metrics].sum(min_count=1).loc[full]
    # With more than a year, average the same calendar month across years.
    by_month = sums.groupby(sums.index.month).mean()
    index = by_month / by_month.mean() * 100
    index.index.name = "month_of_year"
    index.columns = [f"{m}_index" for m in metrics]
    return index.reset_index(), (f"Based on {len(full)} full months; 100 = an average month.")

