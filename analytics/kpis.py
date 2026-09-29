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
    # --- Profitability (U4). Per entity, from its OWN summed totals; see profit_metrics() ----
    KPISpec("gross_margin_pct", "Gross margin", "Gross profit / Revenue x 100", "percent",
            "Share of revenue left after the direct cost of what was sold.", True,
            ("gross_profit", "revenue")),
    KPISpec("break_even_roas", "Break-even ROAS", "1 / Gross margin", "ratio",
            "The ROAS at which gross profit exactly pays for the spend.", False,
            ("gross_profit", "revenue")),
    KPISpec("contribution", "Contribution", "Gross profit - Spend", "money",
            "Money left after the direct cost of sales and the marketing spend.", True,
            ("gross_profit", "spend")),
    KPISpec("profit_per_rupee", "Profit per ₹1", "Contribution / Spend", "money",
            "Contribution earned for every rupee of spend (below zero = a loss).", True,
            ("gross_profit", "spend")),
    KPISpec("roas_headroom", "Headroom", "(ROAS / Break-even ROAS - 1) x 100", "percent",
            "How far ROAS is above (or below) break-even.", True, ("gross_profit", "revenue", "spend")),
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


# ---------------------------------------------------------------------------------------------
# Profitability (U4): gross margin, break-even ROAS, contribution, profit per ₹1, headroom
# ---------------------------------------------------------------------------------------------

PROFIT_KPIS = ["gross_margin_pct", "break_even_roas", "contribution", "profit_per_rupee", "roas_headroom"]
PROFITABLE, NEAR, LOSS, NOT_AVAILABLE = "Profitable", "Near break-even", "Loss-making", "Not available"
PROFIT_STATUSES = [PROFITABLE, NEAR, LOSS, NOT_AVAILABLE]
# Summable row-level parts; every profitability KPI is recomputed from their sums (rule 2).
PROFIT_PARTS = ["spend", "revenue", "gross_profit", "_gp_m", "_rev_m", "_contrib", "_spend_c",
                "_rev_r", "_spend_r"]


def has_profit_data(df: pd.DataFrame) -> bool:
    """True when the data itself has gross profit (a gross profit or a margin column)."""
    return _gross_profit_series(df) is not None


def profit_parts(df: pd.DataFrame, assumed_margin_pct: float | None = None) -> pd.DataFrame | None:
    """Row-level inputs for the profitability KPIs, or None without Revenue + Spend + a gross
    profit source.

    Gross profit comes from the data (gross profit column, else revenue x margin column). Only when
    the data has neither is a user-entered `assumed_margin_pct` used: gross profit = revenue x
    margin. A margin in the data always wins; no margin is ever assumed automatically.
    Each KPI only uses rows where both of its inputs are known (rule 3): margin from rows with
    gross profit AND revenue, contribution from rows with gross profit AND spend (exactly the rows
    ROI uses), ROAS from rows with revenue AND spend (exactly the rows the ROAS KPI uses)."""
    if not (_has(df, "revenue") and _has(df, "spend")):
        return None
    gp = _gross_profit_series(df)
    if gp is None:
        if assumed_margin_pct is None or pd.isna(assumed_margin_pct):
            return None
        gp = _num(df, "revenue") * (float(assumed_margin_pct) / 100)
    rev, spend = _num(df, "revenue"), _num(df, "spend")
    m, c, r = gp.notna() & rev.notna(), gp.notna() & spend.notna(), rev.notna() & spend.notna()
    return pd.DataFrame({
        "spend": spend, "revenue": rev, "gross_profit": gp,
        "_gp_m": gp.where(m), "_rev_m": rev.where(m),
        "_contrib": (gp - spend).where(c), "_spend_c": spend.where(c),
        "_rev_r": rev.where(r), "_spend_r": spend.where(r)}, index=df.index)


def _div(num: pd.Series, den: pd.Series, scale: float = 1.0) -> pd.Series:
    """num / den x scale per row; a zero or missing denominator gives NaN (shown as N/A)."""
    num, den = num.astype(float), den.astype(float)
    return (num / den.where(den != 0) * scale).replace([np.inf, -np.inf], np.nan)


def profit_metrics(sums: pd.DataFrame, band: float | None = None) -> pd.DataFrame:
    """The profitability KPIs from SUMMED parts (one row per entity: campaign, channel, month or
    the whole dataset). Formulas:
      Gross margin %   = Gross profit / Revenue x 100
      Break-even ROAS  = 1 / margin            (ROAS needed for gross profit to pay for the spend)
      Contribution     = Gross profit - Spend  (money left after cost of sales and marketing)
      Profit per ₹1    = Contribution / Spend
      Headroom         = (ROAS / Break-even ROAS - 1) x 100
    A margin of zero or below has no break-even ROAS (no ROAS can pay for the spend)."""
    out = sums.copy()
    margin = _div(out["_gp_m"], out["_rev_m"])
    out["gross_margin_pct"] = margin * 100
    out["break_even_roas"] = _div(pd.Series(1.0, index=out.index), margin.where(margin > 0))
    out["roas"] = _div(out["_rev_r"], out["_spend_r"])
    out["contribution"] = out["_contrib"]
    out["profit_per_rupee"] = _div(out["_contrib"], out["_spend_c"])
    out["roas_headroom"] = (_div(out["roas"], out["break_even_roas"]) - 1) * 100
    out["status"] = profit_status(out["roas"], out["break_even_roas"], out["contribution"], band)
    return out.drop(columns=[c for c in PROFIT_PARTS if c.startswith("_")])


def profit_status(roas: pd.Series, break_even: pd.Series, contribution: pd.Series | None = None,
                  band: float | None = None) -> pd.Series:
    """Profitable: ROAS >= (1 + band) x break-even; Loss-making: ROAS < (1 - band) x break-even;
    Near break-even: in between (both boundaries of the band count as 'near' on the low side and
    'profitable' on the high side). Without a break-even ROAS (margin zero or below, or no
    revenue) an entity that spent money and made no gross profit is Loss-making; anything else
    missing is 'Not available'."""
    if band is None:
        from config.settings import PROFIT_NEAR_BAND as band
    ratio = _div(roas, break_even).round(9)          # rounding: 1.10 must not become 1.0999999
    status = pd.Series(NOT_AVAILABLE, index=roas.index, dtype=object)
    status[ratio >= round(1 + band, 9)] = PROFITABLE
    status[(ratio >= round(1 - band, 9)) & (ratio < round(1 + band, 9))] = NEAR
    status[ratio < round(1 - band, 9)] = LOSS
    if contribution is not None:
        status[ratio.isna() & (contribution.astype(float) < 0)] = LOSS
    return status


# ---------------------------------------------------------------------------------------------
# Budget pacing (U6): pacing ratio, projected month-end spend, over/under
# ---------------------------------------------------------------------------------------------

ON_PACE, OVERSPENDING, UNDERSPENDING = "On Pace", "Overspending", "Underspending"
PACING_STATUSES = [ON_PACE, OVERSPENDING, UNDERSPENDING, NOT_AVAILABLE]


def pacing_metrics(spend_to_date, planned_to_date, remaining_planned, recent_spend,
                   recent_planned) -> dict:
    """Month-to-date pacing from summed daily spend and planned budget. Formulas:
      Pacing ratio          = Spend to date / Planned budget to date
      Recent utilisation    = Spend in the last 7 days / Planned budget in those days
                              (falls back to the pacing ratio when those days had no budget)
      Month budget          = Planned to date + Planned for the rest of the month
      Projected month-end   = Spend to date + Remaining planned x Recent utilisation
                              (assumes the recent pace continues; nothing left = spend to date)
      Projected over/under  = Projected month-end - Month budget (above 0 = overspend)
      Utilisation to date   = Spend to date / Month budget x 100"""
    spend_to_date = float(spend_to_date or 0.0)
    planned_to_date = float(planned_to_date or 0.0)
    remaining_planned = float(remaining_planned or 0.0)
    ratio = safe_divide(spend_to_date, planned_to_date)
    recent = safe_divide(recent_spend, recent_planned)
    recent = ratio if recent is None else recent
    month_budget = planned_to_date + remaining_planned
    if remaining_planned == 0:
        projected = spend_to_date
    else:
        projected = None if recent is None else spend_to_date + remaining_planned * recent
    return {"spend_to_date": spend_to_date, "planned_to_date": planned_to_date,
            "remaining_planned": remaining_planned, "month_budget": month_budget,
            "pacing_ratio": ratio, "recent_utilisation": recent, "projected_spend": projected,
            "projected_variance": None if projected is None or month_budget == 0 else projected - month_budget,
            "utilisation_to_date_pct": safe_divide(spend_to_date, month_budget, 100.0)}


def pacing_status(ratio, band: float | None = None) -> str:
    """On Pace within +/-band of the plan (both boundaries count as on pace); Overspending above,
    Underspending below; no planned budget to date = Not available."""
    if band is None:
        from config.settings import PACING_BAND as band
    if ratio is None or pd.isna(ratio):
        return NOT_AVAILABLE
    ratio = round(float(ratio), 9)
    if ratio > round(1 + band, 9):
        return OVERSPENDING
    if ratio < round(1 - band, 9):
        return UNDERSPENDING
    return ON_PACE


# ---------------------------------------------------------------------------------------------
# Response curves (U7 scenario planner): conversions = a x spend^b per channel
# ---------------------------------------------------------------------------------------------

def response_scale(base_conversions, base_spend, b) -> float | None:
    """Anchor the curve to the recent average week: a = conversions / spend^b, so the curve passes
    exactly through the last 8 weeks' average spend and conversions (the regression gives b, the
    shape; the anchor keeps the level current)."""
    if base_conversions is None or base_spend is None or b is None or float(base_spend) <= 0:
        return None
    return float(base_conversions) / float(base_spend) ** float(b)


def response_conversions(spend, a, b) -> float | None:
    """Weekly conversions expected at a weekly spend: a x spend^b (0 at no spend). With 0 < b <= 1
    each extra rupee buys a little less than the one before (diminishing returns)."""
    if spend is None or a is None or b is None or pd.isna(spend):
        return None
    return 0.0 if float(spend) <= 0 else float(a) * float(spend) ** float(b)


def marginal_cost_per_conversion(spend, a, b) -> float | None:
    """Cost of the NEXT conversion at this weekly spend = 1 / (a x b x spend^(b-1)), the inverse of
    the curve's slope. Rises as spend grows when b < 1. None when it cannot be worked out."""
    if spend is None or a is None or b is None or float(spend) <= 0 or float(a) <= 0 or float(b) <= 0:
        return None
    return safe_divide(1.0, float(a) * float(b) * float(spend) ** (float(b) - 1.0))
