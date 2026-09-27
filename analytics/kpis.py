"""The KPI engine: the ONLY place where marketing KPIs are calculated.

Dashboard, PDF, Excel and AI evidence all use these functions, so ROAS (or any KPI) can never
be computed two different ways (SPEC §23).

Three rules behind every number:
  1. Additive metrics (spend, clicks, leads, ...) are summed. Reach is NOT additive (the same
     person can be reached on many days), so it is never summed into a total.
  2. Ratios are recalculated from summed totals, never averaged across rows. Averaging row
     CTRs would give a 10-impression row the same weight as a 1,00,000-impression row.
  3. A ratio only uses rows where BOTH of its inputs are known. If revenue is missing on a row,
     that row's spend is left out of ROAS too; otherwise ROAS would be understated.
     (Alternative: use every row for the denominator. Rejected because it silently biases the
     ratio; instead the note says how many rows were left out.)
Zero or missing denominators give None, which displays as "N/A".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.formatting import format_change, format_value

# Summable base metrics, in display order. Reach is deliberately absent (not additive).
BASE_METRICS = ["spend", "budget", "impressions", "clicks", "leads", "qualified_leads",
                "opportunities", "conversions", "customers", "orders", "revenue", "gross_profit"]


@dataclass(frozen=True)
class KPISpec:
    key: str
    label: str
    formula_text: str
    fmt: str                          # money / percent / ratio / count
    description: str
    higher_is_better: bool | None     # None = neither (e.g. spend)
    required: tuple[str, ...] = ()    # fields that must exist (alternatives handled in code)


KPI_REGISTRY: dict[str, KPISpec] = {k.key: k for k in [
    # --- Totals ---------------------------------------------------------------------------
    KPISpec("spend", "Spend", "Sum of Spend", "money", "Total marketing spend.", None, ("spend",)),
    KPISpec("impressions", "Impressions", "Sum of Impressions", "count",
            "Times ads were shown.", True, ("impressions",)),
    KPISpec("clicks", "Clicks", "Sum of Clicks", "count", "Clicks on ads or links.", True, ("clicks",)),
    KPISpec("leads", "Leads", "Sum of Leads", "count", "Enquiries / form submissions.", True, ("leads",)),
    KPISpec("qualified_leads", "Qualified leads", "Sum of Qualified leads", "count",
            "Leads judged genuine by the sales/counselling team.", True, ("qualified_leads",)),
    KPISpec("conversions", "Conversions", "Sum of Conversions", "count",
            "Purchases / enrolments.", True, ("conversions",)),
    KPISpec("revenue", "Revenue", "Sum of Revenue", "money", "Money received from conversions.",
            True, ("revenue",)),
    KPISpec("gross_profit", "Gross profit", "Sum of Gross profit", "money",
            "Revenue minus the direct cost of what was sold.", True, ("gross_profit",)),
    # --- Ratios -----------------------------------------------------------------------------
    KPISpec("ctr", "CTR", "Clicks / Impressions x 100", "percent",
            "Share of ad views that led to a click.", True, ("clicks", "impressions")),
    KPISpec("cpc", "CPC", "Spend / Clicks", "money", "Average cost of one click.", False,
            ("spend", "clicks")),
    KPISpec("cpl", "CPL", "Spend / Leads", "money", "Average cost of one lead.", False,
            ("spend", "leads")),
    KPISpec("click_to_lead_rate", "Click-to-lead rate", "Leads / Clicks x 100", "percent",
            "Share of clicks that became leads.", True, ("leads", "clicks")),
    KPISpec("lead_qualification_rate", "Lead qualification rate",
            "Qualified leads / Leads x 100", "percent",
            "Share of leads judged genuine (a lead-quality measure).", True,
            ("qualified_leads", "leads")),
    KPISpec("lead_to_conversion_rate", "Lead-to-conversion rate", "Conversions / Leads x 100",
            "percent", "Share of leads that converted.", True, ("conversions", "leads")),
    KPISpec("cac", "CAC", "Spend / Customers (or Spend / Conversions if there is no Customers "
            "field)", "money", "Average cost to acquire one customer.", False, ("spend",)),
    KPISpec("roas", "ROAS", "Revenue / Spend", "ratio",
            "Revenue earned for every rupee of spend.", True, ("revenue", "spend")),
    KPISpec("aov", "AOV", "Revenue / Orders (or Revenue / Conversions if there is no Orders "
            "field)", "money", "Average value of one order.", True, ("revenue",)),
    KPISpec("roi", "ROI", "(Gross profit - Spend) / Spend x 100", "percent",
            "Profit after marketing cost, per rupee of spend. Needs gross profit or margin data; "
            "no margin is ever assumed.", True, ("spend",)),
    KPISpec("budget_utilisation", "Budget utilisation", "Spend / Budget x 100", "percent",
            "Share of the planned budget actually spent.", None, ("spend", "budget")),
]}

TOTAL_KPIS = ["spend", "impressions", "clicks", "leads", "qualified_leads", "conversions",
              "revenue", "gross_profit"]
RATIO_KPIS = ["ctr", "cpc", "cpl", "click_to_lead_rate", "lead_qualification_rate",
              "lead_to_conversion_rate", "cac", "roas", "aov", "roi", "budget_utilisation"]

# KPI -> capability key in ingestion.validator (to reuse its plain-English reasons).
_CAPABILITY_OF = {"ctr": "ctr", "cpc": "cpc", "cpl": "cpl", "roas": "revenue_metrics",
                  "roi": "roi", "aov": "aov", "cac": "cac"}


@dataclass
class KPIResult:
    key: str
    label: str
    value: float | None
    available: bool
    fmt: str
    formula_text: str
    reason_unavailable: str = ""
    note: str = ""
    numerator: float | None = None
    denominator: float | None = None
    rows_used: int | None = None

    @property
    def formatted(self) -> str:
        return format_value(self.value, self.fmt)

    @property
    def formatted_compact(self) -> str:
        return format_value(self.value, self.fmt, compact=True)


# ---------------------------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------------------------

def safe_divide(numerator, denominator, scale: float = 1.0) -> float | None:
    """numerator / denominator x scale, or None when either is missing or the denominator is 0."""
    try:
        if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
            return None
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator) * scale
    except (TypeError, ValueError):
        return None


def _has(df: pd.DataFrame, col: str) -> bool:
    """The field exists and has at least one value."""
    return col in df.columns and df[col].notna().any()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(float)


def cac_denominator(df: pd.DataFrame) -> str | None:
    """Customers if the dataset has them, otherwise Conversions (locked decision)."""
    if _has(df, "customers"):
        return "customers"
    if _has(df, "conversions"):
        return "conversions"
    return None


def aov_denominator(df: pd.DataFrame) -> str | None:
    if _has(df, "orders"):
        return "orders"
    if _has(df, "conversions"):
        return "conversions"
    return None


def _gross_profit_series(df: pd.DataFrame) -> pd.Series | None:
    """Row-level gross profit: the gross_profit column, else revenue x margin.
    Margins above 1 are read as percentages (35 -> 35%)."""
    if _has(df, "gross_profit"):
        return _num(df, "gross_profit")
    if _has(df, "margin") and _has(df, "revenue"):
        margin = _num(df, "margin")
        margin = margin.where(margin <= 1, margin / 100)
        return _num(df, "revenue") * margin
    return None


def ratio_parts(df: pd.DataFrame, key: str) -> tuple[pd.Series, pd.Series, float, str] | None:
    """Row-level (numerator, denominator) for a ratio KPI, with values blanked on rows where
    either input is missing (rule 3), plus the scale (100 for percentages) and a note.
    Returns None if the dataset lacks the fields."""
    note = ""
    if key == "cac":
        den_col = cac_denominator(df)
        if not _has(df, "spend") or den_col is None:
            return None
        num, den = _num(df, "spend"), _num(df, den_col)
        note = ("Denominator: Customers." if den_col == "customers"
                else "Denominator: Conversions (no Customers field in the data).")
        scale = 1.0
    elif key == "aov":
        den_col = aov_denominator(df)
        if not _has(df, "revenue") or den_col is None:
            return None
        num, den = _num(df, "revenue"), _num(df, den_col)
        note = ("Denominator: Orders." if den_col == "orders"
                else "Denominator: Conversions (no Orders field in the data).")
        scale = 1.0
    elif key == "roi":
        gp = _gross_profit_series(df)
        if gp is None or not _has(df, "spend"):
            return None
        spend = _num(df, "spend")
        num, den = gp - spend, spend
        note = ("Based on the gross profit column." if _has(df, "gross_profit")
                else "Gross profit = Revenue x Margin (from the margin column).")
        scale = 100.0
    else:
        pairs = {
            "ctr": ("clicks", "impressions", 100.0), "cpc": ("spend", "clicks", 1.0),
            "cpl": ("spend", "leads", 1.0), "click_to_lead_rate": ("leads", "clicks", 100.0),
            "lead_qualification_rate": ("qualified_leads", "leads", 100.0),
            "lead_to_conversion_rate": ("conversions", "leads", 100.0),
            "roas": ("revenue", "spend", 1.0), "budget_utilisation": ("spend", "budget", 100.0),
        }
        n_col, d_col, scale = pairs[key]
        if not (_has(df, n_col) and _has(df, d_col)):
            return None
        num, den = _num(df, n_col), _num(df, d_col)
    both = num.notna() & den.notna()
    return num.where(both), den.where(both), scale, note


def _unavailable_reason(key: str, df: pd.DataFrame, capabilities) -> str:
    cap_key = _CAPABILITY_OF.get(key)
    if capabilities is not None and cap_key in getattr(capabilities, "capabilities", {}):
        cap = capabilities.capabilities[cap_key]
        if not cap.enabled:
            return cap.reason
    spec = KPI_REGISTRY[key]
    if key == "roi":
        return ("ROI is hidden because it needs gross profit or margin data. ROAS is shown "
                "instead; no margin is assumed.")
    if key == "cac":
        return "CAC is unavailable because Spend or Conversions/Customers is missing."
    if key == "aov":
        return "AOV is unavailable because Revenue or Conversions/Orders is missing."
    missing = [f.replace("_", " ").title() for f in spec.required if not _has(df, f)]
    return f"{spec.label} is unavailable because {' and '.join(missing) or 'required data'} " \
           f"{'is' if len(missing) <= 1 else 'are'} missing from the dataset."


# ---------------------------------------------------------------------------------------------
# Overall KPIs
# ---------------------------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, capabilities=None) -> dict[str, KPIResult]:
    """All KPIs for the rows in `df` (the whole dataset or a filtered view).

    `capabilities` (an ingestion.validator.ValidationResult) is optional; when given, a KPI
    whose capability is disabled is reported as unavailable with the validator's reason.
    """
    results: dict[str, KPIResult] = {}
    for key in TOTAL_KPIS:
        spec = KPI_REGISTRY[key]
        disabled = _capability_disabled(key, capabilities)
        if not _has(df, key) or disabled:
            results[key] = KPIResult(key, spec.label, None, False, spec.fmt, spec.formula_text,
                                     reason_unavailable=_unavailable_reason(key, df, capabilities))
            continue
        values = _num(df, key)
        results[key] = KPIResult(key, spec.label, float(values.sum()), True, spec.fmt,
                                 spec.formula_text, rows_used=int(values.notna().sum()),
                                 note=_missing_note(values, len(df)))

    for key in RATIO_KPIS:
        spec = KPI_REGISTRY[key]
        parts = None if _capability_disabled(key, capabilities) else ratio_parts(df, key)
        if parts is None:
            if key == "budget_utilisation" and not _has(df, "budget"):
                continue   # optional extra: simply omitted when there is no budget column
            results[key] = KPIResult(key, spec.label, None, False, spec.fmt, spec.formula_text,
                                     reason_unavailable=_unavailable_reason(key, df, capabilities))
            continue
        num, den, scale, note = parts
        numerator, denominator = float(num.sum()), float(den.sum())
        value = safe_divide(numerator, denominator, scale)
        rows_used = int(num.notna().sum())
        extra = []
        if note:
            extra.append(note)
        left_out = int((df.index.size - rows_used))
        if left_out and rows_used:
            extra.append(f"{left_out:,} row(s) without both values were left out.")
        if value is None:
            extra.append(f"Cannot be calculated because the {_den_label(key, df)} total is zero "
                         "or missing.")
        results[key] = KPIResult(key, spec.label, value, True, spec.fmt, spec.formula_text,
                                 note=" ".join(extra), numerator=numerator,
                                 denominator=denominator, rows_used=rows_used)
    return results


def _capability_disabled(key: str, capabilities) -> bool:
    cap_key = _CAPABILITY_OF.get(key)
    caps = getattr(capabilities, "capabilities", None) if capabilities is not None else None
    return bool(caps and cap_key in caps and not caps[cap_key].enabled)


def _den_label(key: str, df: pd.DataFrame) -> str:
    den = {"ctr": "impressions", "cpc": "clicks", "cpl": "leads", "click_to_lead_rate": "clicks",
           "lead_qualification_rate": "leads", "lead_to_conversion_rate": "leads",
           "roas": "spend", "roi": "spend", "budget_utilisation": "budget",
           "cac": cac_denominator(df) or "conversions",
           "aov": aov_denominator(df) or "conversions"}[key]
    return den.replace("_", " ")


def _missing_note(values: pd.Series, total_rows: int) -> str:
    missing = total_rows - int(values.notna().sum())
    return f"{missing:,} row(s) have no value and are not included." if missing else ""


# ---------------------------------------------------------------------------------------------
# KPIs by group (channel, campaign, month, ...)
# ---------------------------------------------------------------------------------------------

def group_kpis(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """One row per group: summed base metrics plus every available ratio KPI, recomputed from
    the group's sums (never an average of row ratios). Missing groups labels become '(none)'."""
    by = [by] if isinstance(by, str) else list(by)
    keys = df[by].astype(object).where(df[by].notna(), "(none)")
    grouped_on = [keys[c] for c in by]

    # Every column that needs summing, grouped ONCE (grouping per metric was slow on big files).
    columns: dict[str, pd.Series] = {}
    for metric in BASE_METRICS:
        if _has(df, metric):
            columns[metric] = _num(df, metric)
    ratios = {}
    for key in RATIO_KPIS:
        parts = ratio_parts(df, key)
        if parts is None:
            continue
        num, den, scale, _ = parts
        columns[f"__num_{key}"], columns[f"__den_{key}"] = num, den
        ratios[key] = scale
    grouped = pd.DataFrame(columns, index=df.index).groupby(grouped_on)
    out = grouped.size().rename("rows").to_frame()
    if columns:
        sums = grouped.sum(min_count=1)
        for metric in BASE_METRICS:
            if metric in sums:
                out[metric] = sums[metric]
        for key, scale in ratios.items():
            n, d = sums[f"__num_{key}"], sums[f"__den_{key}"]
            ratio = (n / d.where(d != 0)) * scale
            out[key] = ratio.replace([np.inf, -np.inf], np.nan)

    out = out.sort_index().reset_index()
    return out


# ---------------------------------------------------------------------------------------------
# Period comparison (for KPI card arrows)
# ---------------------------------------------------------------------------------------------

@dataclass
class KPIDelta:
    current: float | None
    previous: float | None
    abs_change: float | None          # for percentages this is in percentage points
    pct_change: float | None          # relative change, %
    direction: str                    # up / down / flat / n/a
    is_good: bool | None              # None when "better" is undefined (e.g. spend)

    @property
    def formatted(self) -> str:
        return format_change(self.pct_change)


@dataclass
class PeriodComparison:
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    previous_start: pd.Timestamp
    previous_end: pd.Timestamp
    current: dict[str, KPIResult]
    previous: dict[str, KPIResult]
    deltas: dict[str, KPIDelta] = field(default_factory=dict)
    note: str = ""


def period_comparison(df: pd.DataFrame, start, end, date_col: str = "date",
                      capabilities=None) -> PeriodComparison:
    """KPIs for [start, end] vs the equally long period just before it."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    length = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=length - 1)
    dates = df[date_col]
    current_df = df[(dates >= start) & (dates <= end)]
    previous_df = df[(dates >= prev_start) & (dates <= prev_end)]

    current = compute_kpis(current_df, capabilities)
    previous = compute_kpis(previous_df, capabilities)
    deltas: dict[str, KPIDelta] = {}
    for key, cur in current.items():
        prev = previous.get(key)
        c = cur.value
        p = prev.value if prev is not None and len(previous_df) else None
        abs_change = None if c is None or p is None else c - p
        pct_change = safe_divide(abs_change, abs(p) if p else None, 100.0) if abs_change is not None else None
        if abs_change is None:
            direction = "n/a"
        elif abs(abs_change) < 1e-12:
            direction = "flat"
        else:
            direction = "up" if abs_change > 0 else "down"
        better = KPI_REGISTRY[key].higher_is_better
        is_good = None if better is None or direction in ("n/a", "flat") else \
            (direction == "up") == better
        deltas[key] = KPIDelta(c, p, abs_change, pct_change, direction, is_good)

    note = "" if len(previous_df) else "No data in the previous period, so there is nothing to compare with."
    return PeriodComparison(start, end, prev_start, prev_end, current, previous, deltas, note)
