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
from config.fields import channel_type
from config.settings import TREND_WINDOW_DAYS
from dashboard import theme
from utils.formatting import format_change, format_date_range, format_value, title_case

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
    """'Biggest Drop After the Click: Only 5.6% of Clicks Become Leads' (same step as the funnel
    finding). "After the Click" is only said when a Clicks stage comes before the step; otherwise
    the title is the plain 'Biggest Drop: Only X% of A Become B'."""
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
    after_click = "clicks" in set(data["stage"].iloc[:position])
    head = "Biggest Drop After the Click" if after_click else "Biggest Drop"
    return (f"{head}: Only {format_value(step['rate_from_previous'], 'percent')} of "
            f"{previous} Become {title_case(step['label'])}")


# ---------------------------------------------------------------------------------------------
# Partial weeks, owned channels, labels next to a reference line
# ---------------------------------------------------------------------------------------------

PARTIAL_WEEKS_NOTE = "Full weeks only"
FUNNEL_CLICK_NOTE = ("A low impressions-to-clicks rate is normal: most people who see an ad do not "
                     "click it. That is why the biggest drop is looked for after the click.")
OWNED_SUFFIX = " (owned)"


def full_weeks(trend: pd.DataFrame | None) -> tuple[pd.DataFrame | None, bool]:
    """A weekly trend table without an incomplete first or last week (fewer than 7 days of data
    at the edges of the file), so a short edge week never looks like a collapse. Only the chart
    line changes: totals and KPIs are calculated from all rows. Returns (table, dropped?); the
    table is left as it is when fewer than two full weeks would remain."""
    if trend is None or trend.empty or "days" not in trend:
        return trend, False
    keep = pd.Series(True, index=trend.index)
    if trend["days"].iloc[0] < 7:
        keep.iloc[0] = False
    if trend["days"].iloc[-1] < 7:
        keep.iloc[-1] = False
    if keep.all() or keep.sum() < 2:
        return trend, False
    return trend[keep], True


def owned_names(names) -> set[str]:
    """The owned channels (config.fields.channel_type) among `names`, e.g. {'Email'}."""
    return {str(n) for n in names if str(n) != theme.OTHER_LABEL and channel_type(n) == "owned"}


def label_crosses_line(value, reference, axis_max: float, room: float) -> bool:
    """True when a bar's value label, printed just after the bar end, would be crossed by a
    vertical reference line: the bar ends left of the line, closer than the label's width
    (`room`, a share of the axis)."""
    if value is None or reference is None or pd.isna(value) or pd.isna(reference) or not axis_max:
        return False
    return 0 <= reference - value < room * axis_max


# ---------------------------------------------------------------------------------------------
# Profitability titles (U4): read the profitability tables, never calculate
# ---------------------------------------------------------------------------------------------

def profit_channel_title(channels: pd.DataFrame, neutral: str) -> str:
    """'Video Loses ₹4.2 L After Gross Profit' (the biggest loss), else
    'Paid Search Contributes the Most: ₹8.0 Cr'."""
    if channels is None or len(channels) < 2 or "contribution" not in channels:
        return neutral
    data = channels[["channel", "contribution"]].dropna()
    if len(data) < 2:
        return neutral
    worst, best = data.loc[data["contribution"].idxmin()], data.loc[data["contribution"].idxmax()]
    if worst["contribution"] < 0:
        return f"{worst['channel']} Loses {format_value(-worst['contribution'], 'money', compact=True)} After Gross Profit"
    return f"{best['channel']} Contributes the Most: {format_value(best['contribution'], 'money', compact=True)}"


def breakeven_title(campaigns: pd.DataFrame, neutral: str, band: float) -> str:
    """'3 of 15 Campaigns Are Below Break-Even ROAS', or 'All 15 Campaigns Clear Break-Even; 2 Are
    Within 10% of It'."""
    if campaigns is None or campaigns.empty or "status" not in campaigns:
        return neutral
    rated = campaigns[campaigns["status"] != "Not available"]
    if len(rated) < 2:
        return neutral
    below = int(((rated["roas"] < rated["break_even_roas"]) | (rated["status"] == "Loss-making")).sum())
    near = int((rated["status"] == "Near break-even").sum())
    if below:
        return f"{below} of {len(rated)} Campaigns Are Below Break-Even ROAS"
    if near:
        return f"All {len(rated)} Campaigns Clear Break-Even; {near} Are Within {band * 100:.0f}% of It"
    return f"All {len(rated)} Campaigns Are Comfortably Above Break-Even ROAS"


def contribution_trend_title(monthly: pd.DataFrame, neutral: str) -> str:
    """'Contribution Was Negative in 2 of 12 Months', else 'Contribution Stayed Positive in All
    12 Months'."""
    if monthly is None or len(monthly) < 2 or "contribution" not in monthly:
        return neutral
    values = monthly["contribution"].dropna()
    if len(values) < 2:
        return neutral
    negative = int((values < 0).sum())
    if negative:
        return f"Contribution Was Negative in {negative} of {len(values)} Months"
    return f"Contribution Stayed Positive in All {len(values)} Months"


# ---------------------------------------------------------------------------------------------
# Targets title (U5): reads the scorecard, never calculates
# ---------------------------------------------------------------------------------------------

def targets_title(scorecard: pd.DataFrame, neutral: str) -> str:
    """'CPL Is Furthest Off Target: +18.2%; 3 of 5 On Target', 'All 4 Metrics Are On Target' or
    '2 of 4 Metrics On Target; the Rest Are Within 10%'."""
    if scorecard is None or scorecard.empty or "status" not in scorecard:
        return neutral
    rated = scorecard[~scorecard["status"].isin(["Not available", "Partial month"])]
    if rated.empty:
        return neutral
    on = int((rated["status"] == "On Target").sum())
    off = rated[rated["status"].isin(["Off Target", "Over Target", "Under Target"])]
    if not off.empty and off["gap_pct"].notna().any():
        worst = off.loc[off["gap_pct"].abs().idxmax()]
        return (f"{worst['label']} Is Furthest Off Target: {format_change(worst['gap_pct'])}; "
                f"{on} of {len(rated)} On Target")
    if on == len(rated):
        return f"All {len(rated)} Metrics Are On Target" if len(rated) > 1 else f"{rated.iloc[0]['label']} Is On Target"
    return f"{on} of {len(rated)} Metrics On Target; the Rest Are Within 10%"


# ---------------------------------------------------------------------------------------------
# Budget pacing and forecast titles (U6): read the pacing / forecast results, never calculate KPIs
# ---------------------------------------------------------------------------------------------

def pacing_title(monthly: pd.DataFrame, neutral: str) -> str:
    """'Every Full Month Underspent: 80.1%–82.7% of Budget Used', 'All 11 Full Months Were On Pace'
    or '3 of 12 Full Months Overspent' (partial months are not rated)."""
    if monthly is None or monthly.empty or "status" not in monthly:
        return neutral
    rated = monthly[monthly["status"].isin(["On Pace", "Overspending", "Underspending"])]
    if rated.empty:
        return neutral
    counts = rated["status"].value_counts()
    low, high = rated["budget_utilisation"].min(), rated["budget_utilisation"].max()
    span = f"{format_value(low, 'percent')}–{format_value(high, 'percent')} of Budget Used"
    if len(counts) == 1:
        status = counts.index[0]
        if status == "On Pace":
            return f"All {len(rated)} Full Months Were On Pace" if len(rated) > 1 else "The Full Month Was On Pace"
        verb = "Underspent" if status == "Underspending" else "Overspent"
        return f"Every Full Month {verb}: {span}" if len(rated) > 1 else f"The Full Month {verb}: {span}"
    over = int(counts.get("Overspending", 0))
    if over:
        return f"{over} of {len(rated)} Full Months Overspent the Budget"
    return f"{int(counts.get('Underspending', 0))} of {len(rated)} Full Months Underspent the Budget"


def forecast_title(history: pd.DataFrame, forecast: pd.DataFrame, metric: str, neutral: str) -> str:
    """'Weekly Revenue Expected to Hold Near ₹52.0 L' or 'Weekly Leads Expected to Fall 8.2% vs the
    Last 4 Weeks': the forecast's average week against the average of the last 4 actual weeks."""
    if history is None or forecast is None or history.empty or forecast.empty:
        return neutral
    recent = float(history["actual"].tail(4).mean())
    ahead = float(forecast["forecast"].mean())
    label = f"Weekly {_label(metric)}"
    if recent == 0:
        return neutral
    change = (ahead - recent) / abs(recent) * 100          # a display comparison, not a KPI
    direction = direction_of(change)
    if direction == "flat":
        return f"{label} Expected to Hold Near {_fmt(metric, ahead)}"
    if direction not in ("up", "down"):
        return neutral
    verb = "Rise" if direction == "up" else "Fall"
    return f"{label} Expected to {verb} {format_value(abs(change), 'percent')} vs the Last 4 Weeks"
