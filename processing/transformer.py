"""Reshape the table: canonical column names, only the columns analysis uses, strict types,
and calendar helper columns (week, month, quarter, year).
"""

from __future__ import annotations

import pandas as pd

from config.fields import COUNT_FIELDS, FIELD_BY_NAME, FIELDS
from processing.quality import QualityLog

TIME_COLUMNS = ["week_start", "month", "quarter", "year"]


def select_and_rename(df: pd.DataFrame, mapping, log: QualityLog) -> pd.DataFrame:
    """Keep only mapped columns (confirmed or high-confidence), renamed to canonical names.
    Every dropped column is logged once, with the reason."""
    active = mapping.active                       # field -> original column
    reasons = {
        "unmapped": "column did not match any known marketing field",
        "derived": "ratio column; it is recalculated from totals instead",
        "ignored": "you chose not to use this column",
        "uncertain": "meaning of the column was not confirmed",
    }
    for m in mapping.columns:
        if m.status in reasons:
            log.add(None, m.column, "(whole column)", "(dropped)", "column_dropped",
                    reasons[m.status], "info")
    out = df[[active[f.name] for f in FIELDS if f.name in active]].copy()
    out.columns = [f.name for f in FIELDS if f.name in active]
    return out


def add_time_columns(df: pd.DataFrame, log: QualityLog) -> pd.DataFrame:
    """week_start (Monday), month (YYYY-MM), quarter (YYYY-Qn), year, all from `date`.
    If the file already had week/month/quarter/year columns, the date-based values replace
    them so every view uses one consistent calendar."""
    if "date" not in df:
        return df
    for col in TIME_COLUMNS:
        if col in df:
            log.add(None, col, "(whole column)", "(recalculated from date)", "column_replaced",
                    "calendar columns are recalculated from the Date column for consistency", "info")
    dates = df["date"]
    df = df.drop(columns=[c for c in TIME_COLUMNS if c in df])
    df["week_start"] = (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.normalize()
    df["month"] = dates.dt.strftime("%Y-%m").where(dates.notna()).astype("string")
    df["quarter"] = (dates.dt.year.astype("Int64").astype("string") + "-Q"
                     + dates.dt.quarter.astype("Int64").astype("string"))
    df["year"] = dates.dt.year.astype("Int64")
    return df


def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Final column types: dates as datetime, counts as whole numbers, money as decimals,
    categories as text."""
    for col in df.columns:
        spec = FIELD_BY_NAME.get(col)
        if spec is None:
            continue
        if spec.ftype == "date":
            df[col] = pd.to_datetime(df[col])
        elif col in COUNT_FIELDS:
            values = df[col].astype(float)
            whole = values.dropna()
            df[col] = values.astype("Int64") if (whole == whole.round()).all() else values
        elif spec.ftype in ("number", "money"):
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype("string")
    return df


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical = [f.name for f in FIELDS if f.name in df and f.name not in TIME_COLUMNS]
    extras = [c for c in TIME_COLUMNS + ["dq_flags", "source_row"] if c in df]
    return df[canonical + extras]
