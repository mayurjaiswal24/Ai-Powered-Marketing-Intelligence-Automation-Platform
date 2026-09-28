"""Targets vs actual (U5): is each KPI on track against the target the user set?

The user enters a target per metric (blank = no target). Python compares the actual value with
it; nothing is shown for a metric without a target.

Which "actual" is used:
  - Ratio targets (CPL, CAC, ROAS, CTR, lead-to-conversion rate, break-even headroom) use the
    selected period: the same compute_kpis() values as the KPI cards (the full dataset in the
    PDF and Excel). Headroom comes from the profitability result (U4), so it needs gross profit
    data or an assumed margin.
  - Monthly targets (spend, leads, conversions, revenue) are compared month by month. A month
    with fewer days of data than calendar days is "Partial month" and is not rated: half a
    month's spend is not "under target". The scorecard shows the latest full month.

Status (tolerance TARGET_TOLERANCE = 0.10; direction from the KPI registry):
  lower is better (CPL, CAC):  at or below target = On Target; up to 10% above = Within 10%;
                               more = Off Target.
  higher is better (the rest): mirrored (at or above target; up to 10% below; more).
  monthly spend (a plan, neither higher nor lower is better): within +/-5% = On Target,
                               within +/-10% = Within 10%, else Over Target / Under Target.

Targets are stored per column layout (the same key as the mapping cache) in SQLite, so the next
file with the same columns shows them again; in public mode they live in the session only.
No KPI is calculated here: actual values come from analytics/kpis.py.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field

import pandas as pd

from analytics.common import channel_key
from analytics.kpis import KPI_REGISTRY, compute_kpis, group_kpis
from config.fields import channel_type
from utils.formatting import format_change, format_value, title_case

ON, WITHIN, OFF, OVER, UNDER = "On Target", "Within 10%", "Off Target", "Over Target", "Under Target"
NOT_AVAILABLE, PARTIAL = "Not available", "Partial month"
TARGET_STATUSES = [ON, WITHIN, OFF, OVER, UNDER, PARTIAL, NOT_AVAILABLE]

# Ratio targets: compared over the selected period. (key = KPI registry key)
RATIO_TARGETS = ["cpl", "cac", "roas", "ctr", "lead_to_conversion_rate", "roas_headroom"]
# Monthly targets: target key -> the summed metric compared each month.
MONTHLY_TARGETS = {"monthly_spend": "spend", "monthly_leads": "leads",
                   "monthly_conversions": "conversions", "monthly_revenue": "revenue"}
TARGET_KEYS = RATIO_TARGETS + list(MONTHLY_TARGETS)

REASON_NO_METRICS = ("Targets need at least one of these in the data: spend with leads, clicks, "
                     "conversions or revenue, or a date column for monthly targets.")


@dataclass(frozen=True)
class TargetMetric:
    key: str
    label: str            # 'CPL', 'Monthly Spend'
    fmt: str              # money / percent / ratio / count
    unit: str             # shown next to the form field, e.g. '₹ per lead'
    higher_is_better: bool | None
    monthly: bool = False


def _metric(key: str) -> TargetMetric:
    if key in MONTHLY_TARGETS:
        spec = KPI_REGISTRY[MONTHLY_TARGETS[key]]
        unit = "₹ per month" if spec.fmt == "money" else "per month"
        return TargetMetric(key, f"Monthly {title_case(spec.label)}", spec.fmt, unit, spec.higher_is_better, True)
    spec = KPI_REGISTRY[key]
    units = {"cpl": "₹ per lead", "cac": "₹ per customer", "roas": "x (₹ revenue per ₹1 spend)",
             "ctr": "%", "lead_to_conversion_rate": "%", "roas_headroom": "% above break-even ROAS"}
    label = "Break-Even Headroom" if key == "roas_headroom" else title_case(spec.label)
    return TargetMetric(key, label, spec.fmt, units[key], spec.higher_is_better)


TARGET_METRICS: dict[str, TargetMetric] = {k: _metric(k) for k in TARGET_KEYS}


# ---------------------------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------------------------

def gap_pct(actual, target) -> float | None:
    """(Actual - Target) / |Target| x 100: how far the actual is from the target, in %.
    A target of zero (possible for headroom) gives None."""
    if actual is None or target is None or pd.isna(actual) or pd.isna(target) or float(target) == 0:
        return None
    return (float(actual) - float(target)) / abs(float(target)) * 100


def target_status(actual, target, key: str, tolerance: float | None = None,
                  spend_band: float | None = None) -> str:
    """On Target / Within 10% / Off Target (Over / Under Target for monthly spend); a missing
    actual is 'Not available'. Boundaries: exactly 10% worse than target is still 'Within 10%'."""
    if tolerance is None:
        from config.settings import TARGET_TOLERANCE as tolerance
    if spend_band is None:
        from config.settings import SPEND_ON_TARGET_BAND as spend_band
    if actual is None or target is None or pd.isna(actual) or pd.isna(target):
        return NOT_AVAILABLE
    actual, target = float(actual), float(target)
    metric = TARGET_METRICS[key]
    if target == 0:                           # only possible for headroom: no % band to measure
        better = actual >= 0 if metric.higher_is_better else actual <= 0
        return ON if better else OFF
    gap = round(gap_pct(actual, target), 9)   # rounding: 10% must not become 10.0000001%
    tol, band = round(tolerance * 100, 9), round(spend_band * 100, 9)
    if metric.higher_is_better is None:       # monthly spend: a plan to hit, not "more is better"
        if abs(gap) <= band:
            return ON
        if abs(gap) <= tol:
            return WITHIN
        return OVER if gap > 0 else UNDER
    worse_by = -gap if metric.higher_is_better else gap      # > 0 means on the wrong side
    if worse_by <= 0:
        return ON
    return WITHIN if worse_by <= tol else OFF


def is_met(status: str) -> bool:
    return status == ON


# ---------------------------------------------------------------------------------------------
# Which targets can be set for this data
# ---------------------------------------------------------------------------------------------

def available_metrics(df: pd.DataFrame, capabilities=None, profitability=None) -> list[str]:
    """Target keys this data can measure (in form order)."""
    if df is None or df.empty:
        return []
    kpis = compute_kpis(df, capabilities)
    keys = [k for k in RATIO_TARGETS if k != "roas_headroom" and k in kpis and kpis[k].available]
    if profitability is not None and getattr(profitability, "available", False):
        keys.append("roas_headroom")
    if "month" in df and df["month"].notna().any():
        keys += [k for k, m in MONTHLY_TARGETS.items() if m in df and df[m].notna().any()]
    return keys


def clean_targets(values: dict) -> dict[str, float]:
    """Only known metrics with a number (blank / None = no target)."""
    out = {}
    for key, value in (values or {}).items():
        if key in TARGET_METRICS and value is not None and not pd.isna(value):
            out[key] = float(value)
    return out


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def month_label(period: str) -> str:
    """'2026-09' -> 'Sep 2026'."""
    try:
        return pd.Period(str(period), freq="M").strftime("%b %Y")
    except (ValueError, TypeError):
        return str(period)


def monthly_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month: summed spend/leads/conversions/revenue (the KPI engine's group totals),
    days of data and whether the month is complete (data on its first and last calendar day and
    every day in between)."""
    if df is None or df.empty or "month" not in df or df["month"].notna().sum() == 0:
        return pd.DataFrame()
    rows = df[df["month"].notna()]
    out = group_kpis(rows, "month").rename(columns={"month": "period"})
    if "date" in rows:
        days = rows.groupby("month")["date"].nunique()
        out["days"] = out["period"].map(days)
        out["calendar_days"] = out["period"].map(
            lambda p: calendar.monthrange(int(str(p)[:4]), int(str(p)[5:7]))[1])
        out["full"] = out["days"] >= out["calendar_days"]
    else:
        out["full"] = True
    return out


@dataclass
class TargetsResult:
    targets: dict[str, float]
    scorecard: pd.DataFrame = field(default_factory=pd.DataFrame)   # one row per metric with a target
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)     # months x monthly targets (long)
    channels: pd.DataFrame = field(default_factory=pd.DataFrame)    # channels x ratio targets (long)
    period_note: str = ""

    @property
    def counts(self) -> dict[str, int]:
        return {s: int((self.scorecard["status"] == s).sum()) for s in TARGET_STATUSES} \
            if not self.scorecard.empty else {}


def _row(key: str, actual, target: float, basis: str) -> dict:
    m = TARGET_METRICS[key]
    return {"metric": key, "label": m.label, "fmt": m.fmt, "basis": basis, "actual": actual,
            "target": target, "gap": None if actual is None or pd.isna(actual) else float(actual) - target,
            "gap_pct": gap_pct(actual, target), "status": target_status(actual, target, key)}


def targets_analysis(df: pd.DataFrame, targets: dict, capabilities=None, profitability=None,
                     kpis: dict | None = None) -> TargetsResult | None:
    """Actual vs target for the rows in `df` (the filtered view, or the full dataset for exports).
    None when no target is set. `kpis` (the same compute_kpis values the page already has) and
    `profitability` (for headroom) are reused when given."""
    targets = clean_targets(targets)
    if not targets or df is None or df.empty:
        return None
    kpis = kpis if kpis is not None else compute_kpis(df, capabilities)
    result = TargetsResult(targets)
    rows = []
    for key in RATIO_TARGETS:
        if key not in targets:
            continue
        if key == "roas_headroom":
            prof_ok = profitability is not None and getattr(profitability, "available", False)
            actual = profitability.totals.get("roas_headroom") if prof_ok else None
        else:
            k = kpis.get(key)
            actual = k.value if k is not None and k.available else None
        rows.append(_row(key, actual, targets[key], "Selected period"))

    monthly = monthly_actuals(df) if any(k in targets for k in MONTHLY_TARGETS) else pd.DataFrame()
    month_rows = []
    for key, metric in MONTHLY_TARGETS.items():
        if key not in targets:
            continue
        if monthly.empty or metric not in monthly:
            rows.append(_row(key, None, targets[key], "Needs a date column"))
            continue
        for m in monthly.itertuples():
            actual = getattr(m, metric)
            row = _row(key, actual, targets[key], month_label(m.period))
            row["period"] = m.period
            row["days"] = getattr(m, "days", None)
            if not m.full:
                row["status"] = PARTIAL
            month_rows.append(row)
        full = monthly[monthly["full"]]
        if full.empty:
            last = monthly.iloc[-1]
            row = _row(key, last[metric], targets[key], f"{month_label(last['period'])} (partial month)")
            row["status"] = PARTIAL
        else:
            last = full.iloc[-1]
            row = _row(key, last[metric], targets[key], f"{month_label(last['period'])} (latest full month)")
        rows.append(row)
    result.scorecard = pd.DataFrame(rows)
    # Scorecard order = form order.
    result.scorecard = result.scorecard.sort_values(
        "metric", key=lambda s: s.map(TARGET_KEYS.index), kind="stable").reset_index(drop=True)
    result.monthly = pd.DataFrame(month_rows)
    result.channels = _channel_rows(df, targets, profitability)
    if "date" in df and df["date"].notna().any():
        from utils.formatting import format_date_range
        result.period_note = format_date_range(df["date"].min(), df["date"].max())
    return result


def _channel_rows(df: pd.DataFrame, targets: dict, profitability=None) -> pd.DataFrame:
    """Each channel's ratio KPIs (from its own totals) against the same targets. Owned channels are
    labelled '(owned)': their costs are mostly fixed, so read their CPL/CAC with care."""
    ratio_keys = [k for k in RATIO_TARGETS if k in targets]
    ch_key = channel_key(df)
    if not ratio_keys or ch_key is None:
        return pd.DataFrame()
    table = group_kpis(df, ch_key).rename(columns={ch_key: "channel"})
    if "roas_headroom" in ratio_keys and profitability is not None and getattr(profitability, "available", False):
        heads = pd.concat([profitability.channels, profitability.owned_channels])
        if not heads.empty:
            table["roas_headroom"] = table["channel"].map(heads.set_index("channel")["roas_headroom"])
    rows = []
    for r in table.to_dict("records"):
        name = str(r["channel"])
        owned = channel_type(name) == "owned"
        for key in ratio_keys:
            actual = r.get(key)
            actual = None if actual is None or pd.isna(actual) else float(actual)
            row = _row(key, actual, targets[key], "Selected period")
            row.update(channel=f"{name} (owned)" if owned else name, owned=owned, spend=r.get("spend"))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Display helpers (text only; numbers are formatted with utils/formatting.py)
# ---------------------------------------------------------------------------------------------

def target_note(key: str, actual, target: float) -> str:
    """'Target ₹9,000 · On Target' (the small line under an Executive Overview KPI card)."""
    m = TARGET_METRICS[key]
    return f"Target {format_value(target, m.fmt)} · {target_status(actual, target, key)}"


def overview_notes(kpis: dict, targets: dict, keys: list[str]) -> dict[str, str]:
    """Target lines for the Executive Overview cards: ratio KPIs with a target only."""
    targets = clean_targets(targets)
    out = {}
    for key in keys:
        if key in targets and key in RATIO_TARGETS and key in kpis and kpis[key].available:
            out[key] = target_note(key, kpis[key].value, targets[key])
    return out


def display_scorecard(scorecard: pd.DataFrame) -> pd.DataFrame:
    """The scorecard as text columns (each row has its own unit), for the page, PDF and Excel."""
    if scorecard is None or scorecard.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Metric": scorecard["label"],
        "Compared Over": scorecard["basis"],
        "Actual": [format_value(v, f) for v, f in zip(scorecard["actual"], scorecard["fmt"])],
        "Target": [format_value(v, f) for v, f in zip(scorecard["target"], scorecard["fmt"])],
        "Gap": [_signed(v, f) for v, f in zip(scorecard["gap"], scorecard["fmt"])],
        "Gap vs Target": [format_change(v) for v in scorecard["gap_pct"]],
        "Status": scorecard["status"]})


def _signed(value, fmt: str) -> str:
    text = format_value(None if value is None or pd.isna(value) else abs(value), fmt)
    if value is None or pd.isna(value) or text == "N/A":
        return text
    return ("+" if value > 0 else "−" if value < 0 else "") + text


def display_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Months as rows, each monthly target as an 'actual (status)' column."""
    if monthly is None or monthly.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Month": [month_label(p) for p in monthly["period"].unique()]})
    for key in [k for k in MONTHLY_TARGETS if k in set(monthly["metric"])]:
        part = monthly[monthly["metric"] == key]
        m = TARGET_METRICS[key]
        out[m.label] = [format_value(v, m.fmt, compact=m.fmt == "money") for v in part["actual"]]
        out[f"{m.label} Status"] = list(part["status"])
    return out


def display_channels(channels: pd.DataFrame) -> pd.DataFrame:
    """Channels as rows, each ratio target as an 'actual' + status column pair."""
    if channels is None or channels.empty:
        return pd.DataFrame()
    order = channels.drop_duplicates("channel")
    order = order.sort_values(["owned", "spend"], ascending=[True, False], na_position="last")
    out = pd.DataFrame({"Channel": list(order["channel"])})
    for key in [k for k in RATIO_TARGETS if k in set(channels["metric"])]:
        part = channels[channels["metric"] == key].set_index("channel").reindex(order["channel"])
        m = TARGET_METRICS[key]
        out[m.label] = [format_value(v, m.fmt) for v in part["actual"]]
        out[f"{m.label} Status"] = list(part["status"].fillna(NOT_AVAILABLE))
    return out


def summary_lines(result: TargetsResult | None) -> list[str]:
    """Plain English, for the page and the PDF: how many targets are met and the furthest off."""
    if result is None or result.scorecard.empty:
        return []
    sc = result.scorecard
    rated = sc[~sc["status"].isin([NOT_AVAILABLE, PARTIAL])]
    lines = [f"{int((rated['status'] == ON).sum())} of {len(rated)} targets are on target."
             if len(rated) else "No target can be rated yet (no full month or no actual value)."]
    off = rated[rated["status"].isin([OFF, OVER, UNDER])]
    if not off.empty:
        worst = off.loc[off["gap_pct"].abs().idxmax()] if off["gap_pct"].notna().any() else off.iloc[0]
        lines.append(f"Furthest from target: {worst['label']} at {format_value(worst['actual'], worst['fmt'])} "
                     f"vs a target of {format_value(worst['target'], worst['fmt'])} "
                     f"({format_change(worst['gap_pct'])}).")
    return lines


# ---------------------------------------------------------------------------------------------
# Saving and loading (SQLite per column layout; session only in public mode)
# ---------------------------------------------------------------------------------------------

SESSION_KEY = "targets"


def load_targets(state, layout_key: str | None, public: bool, db_path=None) -> dict[str, float]:
    """The targets for this layout: the session's copy first, else (not public) the saved ones.
    `state` is the Streamlit session state (or any dict)."""
    store = state.setdefault(SESSION_KEY, {})
    slot = layout_key or ""
    if slot in store:
        return dict(store[slot])
    saved: dict[str, float] = {}
    if not public and layout_key:
        from database import repository
        from database.connection import DatabaseError
        try:
            saved = repository.load_targets(layout_key, db_path=db_path)
        except DatabaseError:
            saved = {}
    store[slot] = clean_targets(saved)
    return dict(store[slot])


def save_targets(state, layout_key: str | None, values: dict, public: bool, db_path=None) -> dict[str, float]:
    """Keep the targets in the session and (not public, known layout) in SQLite. Returns them."""
    targets = clean_targets(values)
    state.setdefault(SESSION_KEY, {})[layout_key or ""] = targets
    if not public and layout_key:
        from database import repository
        repository.save_targets(layout_key, targets, db_path=db_path)
    return dict(targets)


def clear_targets(state, layout_key: str | None, public: bool, db_path=None) -> None:
    state.setdefault(SESSION_KEY, {})[layout_key or ""] = {}
    if not public and layout_key:
        from database import repository
        repository.clear_targets(layout_key, db_path=db_path)


__all__ = ["TargetsResult", "TargetMetric", "TARGET_METRICS", "TARGET_KEYS", "RATIO_TARGETS",
           "MONTHLY_TARGETS", "target_status", "targets_analysis", "available_metrics", "overview_notes",
           "load_targets", "save_targets", "clear_targets", "summary_lines"]
