"""Budget pacing (U6): is spend on track against the planned budget this month?

Budget source, in this order:
  1. A mapped Budget column: each row is the planned budget for its date and entity (campaign,
     region, ...), so a day's plan = the sum of its rows and a month's budget = the sum of the
     month's rows. Pacing by channel and campaign is possible.
  2. Otherwise a "Monthly Budget (₹)" the user enters (session only), spread evenly over the
     calendar days of each month. Everything based on it is labelled. Only overall pacing.
  3. Otherwise the page explains that pacing needs a budget.

Per month: budget, spend, utilisation % (spend / budget, the KPI registry formula) and variance.
For an "as of" date (default: the last date in the data): planned to date, spend to date, pacing
ratio, projected month-end spend and projected over/under, status (PACING_BAND = 5%). The formulas
live in analytics/kpis.py (pacing_metrics, pacing_status); this module only sums the days.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field

import pandas as pd

from analytics.common import channel_key
from analytics.kpis import NOT_AVAILABLE, pacing_metrics, pacing_status, safe_divide
from analytics.targets import month_label
from config.fields import channel_type
from utils.formatting import format_date, format_value

SOURCE_COLUMN, SOURCE_USER = "column", "user"
PARTIAL = "Partial month"
REASON_NO_DATES = "Budget pacing needs a date column and a spend column."
REASON_NO_BUDGET = ("Budget pacing needs a planned budget: map a Budget column, or enter a monthly "
                    "budget above.")


def budget_label(amount: float) -> str:
    """The label used everywhere a figure depends on the user's monthly budget."""
    return f"Based on your monthly budget of {format_value(amount, 'money')}"


def has_budget_column(df: pd.DataFrame) -> bool:
    return df is not None and "budget" in df and pd.to_numeric(df["budget"], errors="coerce").fillna(0).gt(0).any()


def can_pace(df: pd.DataFrame) -> bool:
    return (df is not None and not df.empty and "date" in df and df["date"].notna().any()
            and "spend" in df and df["spend"].notna().any())


def _month_end(day: pd.Timestamp) -> pd.Timestamp:
    return day.replace(day=calendar.monthrange(day.year, day.month)[1])


def daily_plan(df: pd.DataFrame, monthly_budget: float | None = None) -> pd.DataFrame:
    """One row per calendar day from the first day of the first month to the last day of the last
    month: spend and planned budget (summed over rows), and whether the day has data.
    Column source: planned only on days with rows. User source: the monthly budget / days in month
    on every calendar day."""
    rows = df[df["date"].notna()]
    dates = pd.to_datetime(rows["date"]).dt.normalize()
    start, end = dates.min().replace(day=1), _month_end(dates.max())
    days = pd.date_range(start, end, freq="D")
    spend = pd.to_numeric(rows["spend"], errors="coerce").groupby(dates).sum(min_count=1)
    out = pd.DataFrame(index=days)
    out["has_data"] = out.index.isin(dates.unique())
    out["spend"] = spend.reindex(days)
    if monthly_budget is not None:
        out["planned"] = [float(monthly_budget) / calendar.monthrange(d.year, d.month)[1] for d in days]
    else:
        budget = pd.to_numeric(rows["budget"], errors="coerce").groupby(dates).sum(min_count=1)
        out["planned"] = budget.reindex(days)
    out.index.name = "date"
    return out


def pace_as_of(daily: pd.DataFrame, as_of, recent_days: int | None = None) -> dict:
    """Pacing for the month that contains `as_of`. Days after `as_of` that have no data yet are
    planned at the average daily plan of the last `recent_days` days (column source)."""
    if recent_days is None:
        from config.settings import PACING_RECENT_DAYS as recent_days
    as_of = pd.Timestamp(as_of).normalize()
    month_start, month_end = as_of.replace(day=1), _month_end(as_of)
    mtd = daily.loc[month_start:as_of]
    rest = daily.loc[as_of + pd.Timedelta(days=1):month_end]
    recent = daily.loc[as_of - pd.Timedelta(days=recent_days - 1):as_of]
    known = rest[rest["has_data"] | rest["planned"].notna()]
    missing = len(rest) - len(known)
    avg_plan = safe_divide(recent["planned"].sum(), len(recent)) or 0.0
    remaining = float(known["planned"].sum()) + missing * avg_plan
    out = pacing_metrics(mtd["spend"].sum(), mtd["planned"].sum(), remaining,
                         recent["spend"].sum(), recent["planned"].sum())
    out.update(as_of=as_of, month=as_of.strftime("%Y-%m"), day=as_of.day,
               days_in_month=month_end.day, estimated_days=missing,
               status=pacing_status(out["pacing_ratio"]))
    return out


def monthly_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Per month: budget (planned on the days that have data), spend, utilisation %, variance ₹
    and status; a month with fewer days of data than calendar days is 'Partial month'."""
    data = daily[daily["has_data"]]
    if data.empty:
        return pd.DataFrame()
    month = data.index.to_period("M").astype(str)
    g = data.groupby(month).agg(budget=("planned", "sum"), spend=("spend", "sum"), days=("has_data", "size"))
    g["calendar_days"] = [calendar.monthrange(int(p[:4]), int(p[5:7]))[1] for p in g.index]
    g["budget_utilisation"] = [safe_divide(s, b, 100.0) for s, b in zip(g["spend"], g["budget"])]
    g["variance"] = g["spend"] - g["budget"]
    g["status"] = [pacing_status(None if u is None or pd.isna(u) else u / 100) for u in g["budget_utilisation"]]
    g.loc[g["days"] < g["calendar_days"], "status"] = PARTIAL
    g.index.name = "period"
    return g.reset_index()


@dataclass
class PacingResult:
    available: bool
    reason: str = ""
    source: str = ""                        # 'column' or 'user'
    monthly_budget: float | None = None     # user source only
    as_of: pd.Timestamp | None = None
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    overall: dict = field(default_factory=dict)
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    channels: pd.DataFrame = field(default_factory=pd.DataFrame)    # column source only
    campaigns: pd.DataFrame = field(default_factory=pd.DataFrame)   # column source only

    @property
    def assumed(self) -> bool:
        return self.source == SOURCE_USER

    @property
    def label(self) -> str:
        return budget_label(self.monthly_budget) if self.assumed else "Budget from the data's Budget column"


def _entity_table(df: pd.DataFrame, col: str, as_of) -> pd.DataFrame:
    rows = []
    for name, part in df.groupby(df[col].astype(object).where(df[col].notna(), "(none)")):
        p = pace_as_of(daily_plan(part), as_of)
        owned = col in ("channel", "platform") and channel_type(str(name)) == "owned"
        rows.append({col: f"{name} (owned)" if owned else str(name), "owned": owned, **{
            k: p[k] for k in ("month_budget", "planned_to_date", "spend_to_date", "pacing_ratio",
                              "projected_spend", "projected_variance", "status")}})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pacing_pct"] = out["pacing_ratio"].astype(float) * 100
    return out.sort_values(["owned", "month_budget"], ascending=[True, False]).reset_index(drop=True)


def pacing_analysis(df: pd.DataFrame, as_of=None, monthly_budget: float | None = None) -> PacingResult:
    """Pacing for `df` (the full dataset). The Budget column wins over a user budget."""
    if not can_pace(df):
        return PacingResult(False, REASON_NO_DATES)
    use_column = has_budget_column(df)
    if not use_column and (monthly_budget is None or pd.isna(monthly_budget) or float(monthly_budget) <= 0):
        return PacingResult(False, REASON_NO_BUDGET)
    user_budget = None if use_column else float(monthly_budget)
    daily = daily_plan(df, user_budget)
    first, last = daily.index[daily["has_data"]].min(), daily.index[daily["has_data"]].max()
    as_of = last if as_of is None else min(max(pd.Timestamp(as_of).normalize(), first), last)
    result = PacingResult(True, source=SOURCE_COLUMN if use_column else SOURCE_USER, monthly_budget=user_budget,
                          as_of=as_of, first_date=first, last_date=last)
    result.overall = pace_as_of(daily, as_of)
    result.monthly = monthly_table(daily)
    if use_column:
        month = df[pd.to_datetime(df["date"]).dt.to_period("M") == as_of.to_period("M")]
        ch = channel_key(month)
        if ch:
            result.channels = _entity_table(month, ch, as_of)
        camp = next((c for c in ("campaign", "campaign_id") if c in month and month[c].notna().any()), None)
        if camp:
            result.campaigns = _entity_table(month, camp, as_of).rename(columns={camp: "campaign"})
    return result


# ---------------------------------------------------------------------------------------------
# Plain English and display tables (text only; numbers via utils/formatting.py)
# ---------------------------------------------------------------------------------------------

def summary_lines(result: PacingResult | None) -> list[str]:
    if result is None or not result.available:
        return []
    p = result.overall
    month = month_label(p["month"])
    lines = [f"As of {format_date(result.as_of)} (day {p['day']} of {p['days_in_month']}), spend to date is "
             f"{format_value(p['spend_to_date'], 'money', compact=True)} against a plan of "
             f"{format_value(p['planned_to_date'], 'money', compact=True)} "
             f"({format_value(None if p['pacing_ratio'] is None else p['pacing_ratio'] * 100, 'percent')} "
             f"of plan): {p['status']}."]
    if p["projected_spend"] is not None and p["remaining_planned"] > 0:
        var = p["projected_variance"]
        side = "over" if var is not None and var > 0 else "under"
        lines.append(f"At the last 7 days' pace, {month} should end at "
                     f"{format_value(p['projected_spend'], 'money', compact=True)} against a budget of "
                     f"{format_value(p['month_budget'], 'money', compact=True)} "
                     f"({format_value(abs(var or 0), 'money', compact=True)} {side}).")
    elif p["remaining_planned"] == 0:
        lines.append(f"{month} is complete: spend was {format_value(p['spend_to_date'], 'money', compact=True)} "
                     f"against a budget of {format_value(p['month_budget'], 'money', compact=True)}.")
    rated = result.monthly[result.monthly["status"] != PARTIAL] if not result.monthly.empty else result.monthly
    if not rated.empty:
        low, high = rated["budget_utilisation"].min(), rated["budget_utilisation"].max()
        lines.append(f"Full months used {format_value(low, 'percent')} to {format_value(high, 'percent')} "
                     "of their budget.")
    if result.assumed:
        lines.append(f"{result.label} (spread evenly over each month's days).")
    return lines


def display_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly is None or monthly.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Month": [month_label(p) for p in monthly["period"]],
        "Budget": [format_value(v, "money", compact=True) for v in monthly["budget"]],
        "Spend": [format_value(v, "money", compact=True) for v in monthly["spend"]],
        "Utilisation": [format_value(v, "percent") for v in monthly["budget_utilisation"]],
        "Variance": [_signed_money(v) for v in monthly["variance"]],
        "Status": list(monthly["status"])})


def display_entities(table: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        label: list(table[col]),
        "Status": list(table["status"]),              # second, so it stays visible on narrow screens
        "Month Budget": [format_value(v, "money", compact=True) for v in table["month_budget"]],
        "Planned to Date": [format_value(v, "money", compact=True) for v in table["planned_to_date"]],
        "Spend to Date": [format_value(v, "money", compact=True) for v in table["spend_to_date"]],
        "Pacing": [format_value(v, "percent") for v in table["pacing_pct"]],
        "Projected Month-End": [format_value(v, "money", compact=True) for v in table["projected_spend"]],
        "Projected Over/Under": [_signed_money(v) for v in table["projected_variance"]]})


def _signed_money(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    text = format_value(abs(float(value)), "money", compact=True)
    return ("+" if value > 0 else "−" if value < 0 else "") + text


__all__ = ["PacingResult", "pacing_analysis", "pace_as_of", "daily_plan", "monthly_table", "summary_lines",
           "budget_label", "has_budget_column", "NOT_AVAILABLE"]
