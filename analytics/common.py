"""Small helpers shared by the analytics modules."""

from __future__ import annotations

import pandas as pd

from analytics.kpis import safe_divide

# A recent change smaller than this (in %) is described as "flat".
FLAT_BAND_PCT = 5.0
RECENT_WINDOW_DAYS = 28   # "last 4 weeks vs previous 4 weeks"


def campaign_key(df: pd.DataFrame) -> str | None:
    """The column that names a campaign: the name if present, else the ID."""
    for col in ("campaign_name", "campaign_id"):
        if col in df and df[col].notna().any():
            return col
    return None


def channel_key(df: pd.DataFrame) -> str | None:
    for col in ("channel", "platform"):
        if col in df and df[col].notna().any():
            return col
    return None


def primary_outcome(df: pd.DataFrame) -> str | None:
    """The main result metric available: revenue, else conversions, else leads, else clicks."""
    for col in ("revenue", "conversions", "leads", "clicks"):
        if col in df and df[col].notna().any():
            return col
    return None


def direction_of(pct_change: float | None) -> str:
    if pct_change is None or pd.isna(pct_change):
        return "n/a"
    if abs(pct_change) < FLAT_BAND_PCT:
        return "flat"
    return "up" if pct_change > 0 else "down"


def recent_change(df: pd.DataFrame, by: str | None, metric: str,
                  window_days: int = RECENT_WINDOW_DAYS) -> pd.DataFrame:
    """Sum of `metric` in the last `window_days` of the data vs the `window_days` before,
    per group (or overall if `by` is None). Additive metrics only."""
    if "date" not in df or metric not in df or df["date"].isna().all():
        return pd.DataFrame()
    end = df["date"].max()
    recent_start = end - pd.Timedelta(days=window_days - 1)
    prev_start = recent_start - pd.Timedelta(days=window_days)
    recent = df[df["date"] >= recent_start]
    previous = df[(df["date"] >= prev_start) & (df["date"] < recent_start)]
    if by is None:
        r = pd.Series({"all": recent[metric].astype(float).sum(min_count=1)})
        p = pd.Series({"all": previous[metric].astype(float).sum(min_count=1)})
    else:
        r = recent.groupby(by)[metric].sum(min_count=1).astype(float)
        p = previous.groupby(by)[metric].sum(min_count=1).astype(float)
    out = pd.DataFrame({f"{metric}_last_4w": r, f"{metric}_prev_4w": p})
    out["change_pct"] = [safe_divide(a - b, b, 100.0) if pd.notna(a) and pd.notna(b) else None
                         for a, b in zip(out.iloc[:, 0], out.iloc[:, 1])]
    out["trend"] = out["change_pct"].map(direction_of)
    out.index.name = by or "scope"
    return out.reset_index()


def add_shares(table: pd.DataFrame, metrics=("spend", "revenue", "conversions", "leads")) -> pd.DataFrame:
    """Each group's share (%) of the total for the given additive metrics."""
    for m in metrics:
        if m in table:
            total = table[m].sum(min_count=1)
            table[f"{m}_share_pct"] = table[m] / total * 100 if total else None
    return table
