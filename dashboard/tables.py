"""Formatting result tables for display. Shared by the dashboard, the PDF and the Excel workbook
so a table looks the same (and shows the same numbers) everywhere. No Streamlit code here."""

from __future__ import annotations

import pandas as pd

from analytics.kpis import KPI_REGISTRY
from utils.formatting import format_count, format_date, format_value

MONEY_COLUMNS = {"spend", "budget", "revenue", "gross_profit", "baseline_money", "observed_money"}
COUNT_COLUMNS = {"rows", "impressions", "clicks", "leads", "qualified_leads", "opportunities",
                 "conversions", "customers", "orders", "active_days", "value", "days"}
NICE_NAMES = {
    "campaign": "Campaign", "channel": "Channel", "segment": "Segment", "rows": "Rows",
    "first_date": "First day", "last_date": "Last day", "active_days": "Active days",
    "trend": "Trend (4 wks)", "trend_change_pct": "Change (4 wks)", "growth_trend": "Trend (4 wks)",
    "growth_pct": "Growth (4 wks)", "anomalies": "Anomalies", "label": "Stage",
    "rate_from_previous": "Conversion from previous", "drop_off_pct": "Drop-off",
}


def column_label(col: str) -> str:
    if col in NICE_NAMES:
        return NICE_NAMES[col]
    if col in KPI_REGISTRY:
        return KPI_REGISTRY[col].label
    if col.endswith("_share_pct"):
        return f"{col.removesuffix('_share_pct').replace('_', ' ').capitalize()} share"
    if col.endswith("_index"):
        base = col.removesuffix("_index")
        return f"{KPI_REGISTRY[base].label if base in KPI_REGISTRY else base} index"
    return col.replace("_", " ").capitalize()


def column_format(col: str, series: pd.Series) -> str:
    if col in KPI_REGISTRY:
        return KPI_REGISTRY[col].fmt
    if col.endswith("_pct") or col in ("rate_from_previous", "drop_off_pct"):
        return "percent"
    if col.endswith("_index"):
        return "index"
    if col in MONEY_COLUMNS:
        return "money"
    if col in COUNT_COLUMNS:
        return "count"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "text"


def format_table(df: pd.DataFrame, columns: list[str] | None = None,
                 compact_money: bool = False) -> pd.DataFrame:
    """A display copy with every number formatted the same way as the KPI cards and reports.
    Missing values show as N/A. (Sorting happens BEFORE formatting, via the page's controls.)"""
    columns = [c for c in (columns or list(df.columns)) if c in df]
    out = pd.DataFrame(index=df.index)
    for col in columns:
        fmt = column_format(col, df[col])
        if fmt == "date":
            out[column_label(col)] = df[col].map(format_date)
        elif fmt == "index":
            out[column_label(col)] = df[col].map(lambda v: "N/A" if pd.isna(v) else f"{v:.0f}")
        elif fmt == "text":
            out[column_label(col)] = df[col].astype(object).where(df[col].notna(), "N/A").astype(str)
        elif fmt == "count":
            out[column_label(col)] = df[col].map(format_count)
        else:
            out[column_label(col)] = df[col].map(
                lambda v, f=fmt: format_value(v, f, compact=compact_money and f == "money"))
    return out.reset_index(drop=True)
