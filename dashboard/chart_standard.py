"""The chart standard (docs/CHART_STYLE_GUIDE.md) as small helpers shared by the dashboard and the
PDF: insight titles, subtitles and "Other" grouping.

Insight titles only READ numbers from tables the analytics layer already produced (or from the
same analytics functions, e.g. period_comparison), and format them with utils/formatting.py. They
never calculate a KPI. When the data has no clear takeaway, a neutral title is returned instead.
"""

from __future__ import annotations

import pandas as pd

from analytics.common import direction_of
from analytics.funnel import biggest_drop_off
from analytics.kpis import KPI_REGISTRY, period_comparison
from config.settings import TREND_WINDOW_DAYS
from dashboard import theme
from utils.formatting import format_date_range, format_value, title_case

# A channel "punches above its weight" when its share of results beats its share of spend by at
# least this many percentage points (smaller gaps are noise, so the title stays neutral).
SHARE_GAP_POINTS = 5.0
UNIT_LABELS = {"money": "₹", "percent": "%", "ratio": "x", "count": "Count"}
GRAIN_LABELS = {"day": "Daily", "week": "Weekly", "month": "Monthly"}


def _label(metric: str) -> str:
    return title_case(KPI_REGISTRY[metric].label)


def _fmt(metric: str, value) -> str:
    return format_value(value, KPI_REGISTRY[metric].fmt, compact=True)


# ---------------------------------------------------------------------------------------------
# Subtitle: metric, unit, period
# ---------------------------------------------------------------------------------------------

def subtitle(metric: str | None = None, grain: str | None = None, start=None, end=None,
             note: str | None = None) -> str:
    """'Revenue (₹) · Weekly · 1 Oct 2025 – 30 Sep 2026' (parts that are unknown are left out)."""
    parts = []
    if metric in KPI_REGISTRY:
        parts.append(f"{KPI_REGISTRY[metric].label} ({UNIT_LABELS[KPI_REGISTRY[metric].fmt]})")
    if grain in GRAIN_LABELS:
        parts.append(GRAIN_LABELS[grain])
    if start is not None and end is not None and not pd.isna(start) and not pd.isna(end):
        parts.append(format_date_range(pd.Timestamp(start), pd.Timestamp(end)))
    if note:
        parts.append(note)
    return " · ".join(parts)


def date_span(df: pd.DataFrame | None, col: str = "date") -> tuple:
    if df is None or df.empty or col not in df or df[col].isna().all():
        return None, None
    return df[col].min(), df[col].max()


# ---------------------------------------------------------------------------------------------
# "Other" grouping
# ---------------------------------------------------------------------------------------------

def group_other(df: pd.DataFrame, category: str, additive: list[str], sort_by: str,
                max_categories: int = theme.MAX_CATEGORIES) -> pd.DataFrame:
    """More than `max_categories` rows -> the top (max - 1) by `sort_by` plus one "Other" row.
    Only additive columns (totals and shares) are summed for "Other"; everything else (ratios)
    is left empty, because a ratio must never be added up."""
    if df is None or len(df) <= max_categories or sort_by not in df:
        return df
    data = df.sort_values(sort_by, ascending=False)
    top, rest = data.head(max_categories - 1), data.iloc[max_categories - 1:]
    other = {category: theme.OTHER_LABEL}
    for col in additive:
        if col in rest:
            other[col] = rest[col].astype(float).sum(min_count=1)
    return pd.concat([top, pd.DataFrame([other])], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Insight titles
# ---------------------------------------------------------------------------------------------

def share_title(table: pd.DataFrame, category: str, value_col: str, value_name: str,
                neutral: str, spend_col: str = "spend_share_pct") -> str:
    """'Paid Search Brings 55.3% of Revenue on 44.8% of Spend': the item whose share of results
    beats its share of spend by the most (at least SHARE_GAP_POINTS)."""
    if table is None or table.empty or value_col not in table or spend_col not in table:
        return neutral
    data = table[[category, spend_col, value_col]].dropna()
    data = data[data[category] != theme.OTHER_LABEL]
    if data.empty:
        return neutral
    gap = data[value_col] - data[spend_col]
    best = data.loc[gap.idxmax()]
    if gap.max() < SHARE_GAP_POINTS:
        return neutral
    return (f"{best[category]} Brings {format_value(best[value_col], 'percent')} of {value_name} "
            f"on {format_value(best[spend_col], 'percent')} of Spend")


def ranking_title(table: pd.DataFrame, category: str, metric: str, neutral: str,
                  reference: float | None = None, reference_label: str = "Average",
                  weakest: bool = False) -> str:
    """'Affiliate Has the Best ROAS: 6.67x vs 4.18x Paid Average' (ratios, with a reference), or
    'Google Search - Brand Keywords Leads on Conversions: 1,775' (totals)."""
    if table is None or table.empty or metric not in table or category not in table:
        return neutral
    data = table[[category, metric]].dropna()
    data = data[data[category] != theme.OTHER_LABEL]
    if len(data) < 2:
        return neutral
    better = KPI_REGISTRY[metric].higher_is_better
    ascending = (better is False) != weakest if better is not None else weakest
    data = data.sort_values(metric, ascending=ascending)
    first, second = data.iloc[0], data.iloc[1]
    if first[metric] == second[metric]:
        return neutral
    name, value, label = first[category], _fmt(metric, first[metric]), _label(metric)
    if better is None:
        word = "Lowest" if weakest else "Highest"
        return f"{name} Has the {word} {label}: {value}"
    if weakest:
        head = f"{name} Has the Weakest {label}: {value}"
    elif reference is None:
        return f"{name} Leads on {label}: {value}"
    else:
        head = f"{name} Has the Best {label}: {value}"
    if reference is not None and not pd.isna(reference):
        return f"{head} vs {_fmt(metric, reference)} {reference_label}"
    return head


def index_title(table: pd.DataFrame, category: str, index_col: str, metric: str, neutral: str) -> str:
    """'Paid Search Leads on CTR: Index 639' (the subtitle says Average = 100)."""
    if table is None or table.empty or index_col not in table:
        return neutral
    data = table[[category, index_col]].dropna()
    if len(data) < 2:
        return neutral
    lower_is_better = KPI_REGISTRY[metric].higher_is_better is False
    best = data.loc[data[index_col].idxmin() if lower_is_better else data[index_col].idxmax()]
    return f"{best[category]} Leads on {_label(metric)}: Index {best[index_col]:.0f}"


def trend_title(rows: pd.DataFrame, metric: str, neutral: str, date_col: str = "date",
                capabilities=None) -> str:
    """'Revenue Fell 12.4% in the Last 28 Days' — the last TREND_WINDOW_DAYS of the data vs the
    same number of days before (analytics.kpis.period_comparison). Changes inside the 'flat'
    band stay neutral."""
    if rows is None or rows.empty or date_col not in rows or rows[date_col].isna().all():
        return neutral
    end = pd.Timestamp(rows[date_col].max()).normalize()
    start = end - pd.Timedelta(days=TREND_WINDOW_DAYS - 1)
    comparison = period_comparison(rows, start, end, date_col=date_col, capabilities=capabilities)
    delta = comparison.deltas.get(metric)
    if comparison.note or delta is None or delta.pct_change is None:
        return neutral
    direction = direction_of(delta.pct_change)
    if direction not in ("up", "down"):
        return neutral
    verb = "Rose" if direction == "up" else "Fell"
    return (f"{_label(metric)} {verb} {format_value(abs(delta.pct_change), 'percent')} "
            f"in the Last {TREND_WINDOW_DAYS} Days")


def funnel_title(funnel: pd.DataFrame, neutral: str) -> str:
    """'Biggest Drop: Only 5.6% of Clicks Become Leads' (same step as the funnel finding)."""
    if funnel is None or funnel.empty:
        return neutral
    data = funnel[funnel["stage"] != "revenue"].sort_values("stage_order").reset_index(drop=True)
    step = biggest_drop_off(data)
    if step is None:
        return neutral
    position = data.index[data["stage"] == step["stage"]][0]
    if position == 0:
        return neutral
    previous = title_case(data.loc[position - 1, "label"])
    return (f"Biggest Drop: Only {format_value(step['rate_from_previous'], 'percent')} of "
            f"{previous} Become {title_case(step['label'])}")
