"""Response curves and the scenario planner (U7): "what happens if we move budget?".

Honest and simple on purpose (pandas + numpy only, fully deterministic):
  Step 0 - data check per PAID channel (owned channels such as email are left out), on full
    weeks: at least RESPONSE_MIN_WEEKS weeks with spend, spend variation (CV) >= 0.15 and highest
    week / lowest week >= 1.5. A channel that fails is "Not enough spend variation to estimate" and
    is left out. Fewer than 2 qualifying channels = no planner (the page explains why).
  Pricing - "auction" channels (default) get a curve; "performance" channels (e.g. Affiliate, paid
    per result) are never curve-fitted because their spend follows their results: conversions =
    spend / recent 8-week CPA, capped at 1.2 x their best week's conversions.
  Curve - weekly conversions = a x spend^b, fitted as a straight line on log(spend) vs
    log(conversions), leaving out the weeks of that channel's incidents (outages, budget typos).
    b above 1 is set to 1 (returns never grow faster than spend); b <= 0 = insufficient evidence.
    `a` is anchored to the last 8 weeks' average (spend, conversions), so the curve starts from
    today's level. Confidence from R-squared and weeks. 200 bootstrap refits (fixed seed) give
    the 10th-90th percentile range. Revenue = conversions x recent 8-week AOV.
Limits: these are past patterns. Seasonality can inflate a curve (spend and results both rise in
the busy season), so use the planner to size test budgets, not as a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.common import campaign_key, channel_key
from analytics.forecast import weekly_history
from analytics.kpis import (marginal_cost_per_conversion, response_conversions, response_scale,
                            safe_divide)
from config import settings as cfg
from config.fields import channel_type

AUCTION, PERFORMANCE = "auction", "performance"
FITTED, INSUFFICIENT = "fitted", "insufficient"
FAIL_VARIATION = "Not enough spend variation to estimate"
LAKH = 100_000.0
HONEST_NOTE = ("These curves describe past patterns. Seasonality can inflate them (spend and results "
               "both rise in the busy season), and nothing outside the spend range already tried is "
               "known. Use them to size test budgets, not as a guarantee.")
REASON_NO_DATA = ("The scenario planner needs a date column, a channel column, spend and conversions.")


def channel_pricing(channel) -> str:
    """'performance' for channels paid per result (config CHANNEL_PRICING), else 'auction'."""
    key = " ".join(str(channel).strip().lower().split())
    return cfg.CHANNEL_PRICING.get(key, AUCTION)


@dataclass
class ChannelCurve:
    channel: str
    pricing: str
    status: str                              # fitted / insufficient / performance
    reason: str = ""
    weekly: pd.DataFrame = field(default_factory=pd.DataFrame)   # period, spend, conversions, revenue, excluded
    excluded_incidents: list = field(default_factory=list)       # incident IDs whose weeks were left out
    weeks_used: int = 0
    b_raw: float | None = None
    b: float | None = None
    clamped: bool = False
    a: float | None = None
    r2: float | None = None
    confidence: str = ""
    base_spend: float = 0.0                  # recent 8-week average weekly spend (the baseline)
    base_conversions: float = 0.0
    aov: float | None = None                 # recent 8-week revenue / conversions
    cpa: float | None = None                 # recent 8-week spend / conversions (performance channels)
    cpa_range: tuple = (None, None)          # 10th / 90th percentile of recent weekly CPA
    cap_conversions: float | None = None     # performance channels: 1.2 x best weekly conversions
    min_spend: float = 0.0                   # lowest / highest weekly spend seen
    max_spend: float = 0.0
    boot_b: np.ndarray | None = None

    @property
    def usable(self) -> bool:
        return self.status in (FITTED, PERFORMANCE)

    @property
    def guard_low(self) -> float:
        return cfg.SCENARIO_GUARDRAIL_LOW * self.min_spend

    @property
    def guard_high(self) -> float:
        return cfg.SCENARIO_GUARDRAIL_HIGH * self.max_spend

    def conversions(self, spend: float) -> float | None:
        if self.status == FITTED:
            return response_conversions(spend, self.a, self.b)
        if self.status == PERFORMANCE:
            raw = safe_divide(spend, self.cpa)
            return None if raw is None else min(raw, self.cap_conversions)
        return None

    def conversions_range(self, spend: float) -> tuple[float | None, float | None]:
        """10th-90th percentile of weekly conversions at this spend."""
        lo_p, hi_p = cfg.RESPONSE_RANGE_PERCENTILES
        if self.status == FITTED and self.boot_b is not None and spend > 0:
            samples = self.base_conversions * (float(spend) / self.base_spend) ** self.boot_b
            return float(np.percentile(samples, lo_p)), float(np.percentile(samples, hi_p))
        if self.status == PERFORMANCE:
            lo_cpa, hi_cpa = self.cpa_range
            low = safe_divide(spend, hi_cpa)
            high = safe_divide(spend, lo_cpa)
            cap = self.cap_conversions
            return (None if low is None else min(low, cap)), (None if high is None else min(high, cap))
        point = self.conversions(spend)
        return point, point

    def marginal_cost(self, spend: float) -> float | None:
        """Cost of the next conversion at this weekly spend (None = more spend buys nothing more)."""
        if self.status == FITTED:
            return marginal_cost_per_conversion(spend, self.a, self.b)
        if self.status == PERFORMANCE:
            return None if spend / self.cpa >= self.cap_conversions else self.cpa
        return None

    def per_lakh(self, spend: float) -> float | None:
        """'The next ₹1 L here buys about N conversions' = ₹1 L / marginal cost."""
        mc = self.marginal_cost(spend)
        return None if mc is None else LAKH / mc


@dataclass
class ResponseCurvesResult:
    available: bool                          # True = at least 2 usable channels (planner on)
    reason: str = ""
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    curves: dict = field(default_factory=dict)          # channel -> ChannelCurve (qualifying channels)
    weeks: int = 0

    @property
    def usable(self) -> dict:
        return {c: cv for c, cv in self.curves.items() if cv.usable}


# ---------------------------------------------------------------------------------------------
# Step 0: data check
# ---------------------------------------------------------------------------------------------

def spend_variation(spend) -> dict:
    """Weeks with spend, coefficient of variation (std / mean) and highest / lowest week, on the
    weeks that had spend."""
    s = np.asarray(spend, dtype=float)
    s = s[s > 0]
    cv = float(np.std(s, ddof=1) / np.mean(s)) if len(s) > 1 else None
    ratio = float(s.max() / s.min()) if len(s) else None
    return {"weeks_with_spend": int(len(s)), "cv": cv, "max_min": ratio}


def passes_check(v: dict) -> bool:
    return (v["weeks_with_spend"] >= cfg.RESPONSE_MIN_WEEKS and v["cv"] is not None
            and v["cv"] >= cfg.RESPONSE_MIN_CV and v["max_min"] is not None
            and v["max_min"] >= cfg.RESPONSE_MIN_MAX_RATIO)


# ---------------------------------------------------------------------------------------------
# Incident weeks of a channel
# ---------------------------------------------------------------------------------------------

def _monday(value) -> pd.Timestamp:
    d = pd.Timestamp(value).normalize()
    return d - pd.Timedelta(days=d.weekday())


def incident_weeks(df: pd.DataFrame, incidents: pd.DataFrame | None, channel: str) -> dict:
    """{week start: [incident IDs]} for incidents that belong to this channel: the channel itself,
    or a campaign / platform whose rows are all in this channel. Incidents that span several
    channels (e.g. a region) are not a channel's own and are kept."""
    out: dict = {}
    key = channel_key(df)
    if incidents is None or incidents.empty or key is None:
        return out
    for r in incidents.itertuples():
        column = campaign_key(df) if r.entity_type == "campaign" else r.entity_type
        if column is None or column not in df:
            continue
        channels = set(df.loc[df[column].astype(str) == str(r.entity), key].dropna().astype(str))
        if channels != {str(channel)}:
            continue
        week = _monday(r.period_start)
        while week <= pd.Timestamp(r.period_end):
            out.setdefault(week, []).append(r.incident_id)
            week += pd.Timedelta(days=7)
    return out


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------

def fit_power(spend, conversions) -> tuple[float, float, float]:
    """Least squares on log(conversions) = log(a) + b x log(spend). Returns (b, a, R-squared)."""
    x, y = np.log(np.asarray(spend, dtype=float)), np.log(np.asarray(conversions, dtype=float))
    b, log_a = np.polyfit(x, y, 1)
    residual = y - (log_a + b * x)
    total = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (residual ** 2).sum() / total) if total > 0 else 0.0
    return float(b), float(np.exp(log_a)), r2


def bootstrap_b(spend, conversions, samples: int | None = None, seed: int | None = None) -> np.ndarray:
    """Refit b on weeks drawn with replacement (fixed seed = same answer every time), limited to
    0..1 like the main fit (a flat refit means 'no extra conversions')."""
    samples = samples or cfg.RESPONSE_BOOTSTRAP_SAMPLES
    rng = np.random.default_rng(cfg.RESPONSE_BOOTSTRAP_SEED if seed is None else seed)
    x, y = np.log(np.asarray(spend, dtype=float)), np.log(np.asarray(conversions, dtype=float))
    idx = rng.integers(0, len(x), size=(samples, len(x)))
    xs, ys = x[idx], y[idx]
    xc, yc = xs - xs.mean(axis=1, keepdims=True), ys - ys.mean(axis=1, keepdims=True)
    var = (xc ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(var > 0, (xc * yc).sum(axis=1) / var, np.nan)
    b = np.where(np.isnan(b), np.nanmedian(b) if np.isfinite(b).any() else 0.0, b)
    return np.clip(b, 0.0, 1.0)


def confidence(r2: float | None, weeks: int) -> str:
    hi_r2, hi_w = cfg.RESPONSE_CONFIDENCE_HIGH
    md_r2, md_w = cfg.RESPONSE_CONFIDENCE_MEDIUM
    if r2 is None:
        return "Low"
    if r2 >= hi_r2 and weeks >= hi_w:
        return "High"
    if r2 >= md_r2 and weeks >= md_w:
        return "Medium"
    return "Low"


def fit_channel(weekly: pd.DataFrame, channel: str, excluded: dict | None = None) -> ChannelCurve:
    """The curve (or the performance-pricing rule) for one channel from its full weeks."""
    excluded = excluded or {}
    w = weekly[["period", "spend", "conversions"] + (["revenue"] if "revenue" in weekly else [])].copy()
    w["excluded"] = w["period"].isin(list(excluded))
    pricing = channel_pricing(channel)
    spent = w[w["spend"] > 0]
    recent = w.tail(cfg.RESPONSE_RECENT_WEEKS)
    curve = ChannelCurve(channel, pricing, PERFORMANCE if pricing == PERFORMANCE else FITTED, weekly=w,
                         excluded_incidents=sorted({i for ids in excluded.values() for i in ids}),
                         base_spend=float(recent["spend"].mean()), base_conversions=float(recent["conversions"].mean()),
                         min_spend=float(spent["spend"].min()) if len(spent) else 0.0,
                         max_spend=float(spent["spend"].max()) if len(spent) else 0.0)
    if "revenue" in recent:
        curve.aov = safe_divide(recent["revenue"].sum(), recent["conversions"].sum())
    if curve.base_spend <= 0 or curve.base_conversions <= 0:
        curve.status = INSUFFICIENT
        curve.reason = f"No spend or no conversions in the last {cfg.RESPONSE_RECENT_WEEKS} weeks."
        return curve
    if pricing == PERFORMANCE:
        # Paid per result: spend follows results, so a curve would only show that relationship.
        curve.cpa = safe_divide(recent["spend"].sum(), recent["conversions"].sum())
        weekly_cpa = (recent["spend"] / recent["conversions"]).replace([np.inf, -np.inf], np.nan).dropna()
        lo_p, hi_p = cfg.RESPONSE_RANGE_PERCENTILES
        curve.cpa_range = ((float(np.percentile(weekly_cpa, lo_p)), float(np.percentile(weekly_cpa, hi_p)))
                           if len(weekly_cpa) else (curve.cpa, curve.cpa))
        curve.cap_conversions = cfg.PERFORMANCE_CAP_MULTIPLE * float(w["conversions"].max())
        curve.weeks_used = len(recent)
        return curve
    used = w[(w["spend"] > 0) & (w["conversions"] > 0) & ~w["excluded"]]
    curve.weeks_used = len(used)
    if len(used) < cfg.RESPONSE_MIN_FIT_WEEKS:
        curve.status = INSUFFICIENT
        curve.reason = (f"Only {len(used)} usable weeks after leaving out incident weeks; "
                        f"{cfg.RESPONSE_MIN_FIT_WEEKS} are needed.")
        return curve
    b_raw, _, r2 = fit_power(used["spend"], used["conversions"])
    curve.b_raw, curve.r2 = b_raw, r2
    if b_raw <= 0:
        curve.status = INSUFFICIENT
        curve.reason = "Insufficient evidence: weeks with more spend did not bring more conversions."
        return curve
    curve.clamped = b_raw > 1
    curve.b = min(b_raw, 1.0)
    curve.a = response_scale(curve.base_conversions, curve.base_spend, curve.b)
    # b above 1 usually means spend and results rose together for another reason (for example the
    # busy season), not that returns really grow faster than spend: the shape is doubtful, so Low.
    curve.confidence = "Low" if curve.clamped else confidence(r2, len(used))
    curve.boot_b = bootstrap_b(used["spend"], used["conversions"])
    return curve


# ---------------------------------------------------------------------------------------------
# The whole analysis
# ---------------------------------------------------------------------------------------------

CHECK_COLUMNS = ["channel", "pricing", "weeks_with_spend", "cv", "max_min", "qualifies", "result"]


def response_analysis(df: pd.DataFrame, incidents: pd.DataFrame | None = None) -> ResponseCurvesResult:
    """Data check for every paid channel, then a curve (or the per-result rule) for each channel
    that qualifies. `available` = at least 2 usable channels."""
    key = channel_key(df) if df is not None and not df.empty else None
    if (key is None or "date" not in df or df["date"].notna().sum() == 0
            or "spend" not in df or "conversions" not in df):
        return ResponseCurvesResult(False, REASON_NO_DATA)
    rows, curves, weeks = [], {}, 0
    names = df.groupby(key)["spend"].sum().sort_values(ascending=False, kind="stable").index
    for channel in [str(c) for c in names]:
        if channel_type(channel) == "owned":
            continue
        weekly = weekly_history(df, channel)
        if weekly.empty or "spend" not in weekly or "conversions" not in weekly:
            continue
        weeks = max(weeks, len(weekly))
        v = spend_variation(weekly["spend"])
        ok = passes_check(v)
        pricing = channel_pricing(channel)
        result = "Qualifies" if ok else FAIL_VARIATION
        if ok:
            curve = fit_channel(weekly, channel, incident_weeks(df, incidents, channel))
            curves[channel] = curve
            if curve.status == INSUFFICIENT:
                result = curve.reason
            elif curve.status == PERFORMANCE:
                result = "Qualifies (priced per result: not curve-fitted)"
        rows.append({"channel": channel, "pricing": pricing, **v, "qualifies": ok, "result": result})
    checks = pd.DataFrame(rows, columns=CHECK_COLUMNS)
    out = ResponseCurvesResult(False, "", checks, curves, weeks)
    usable = len(out.usable)
    if usable >= 2:
        out.available = True
    elif checks.empty:
        out.reason = "No paid channel has full weeks of spend and conversions."
    else:
        out.reason = (f"Only {usable} paid channel{'s' if usable != 1 else ''} passed the data check, and the "
                      "planner needs at least 2 to compare. A channel qualifies when it has at least "
                      f"{cfg.RESPONSE_MIN_WEEKS} weeks with spend and its weekly spend varied enough "
                      f"(variation of at least {cfg.RESPONSE_MIN_CV:.2f} and a highest week at least "
                      f"{cfg.RESPONSE_MIN_MAX_RATIO:g} times the lowest). Without that variation the data "
                      "cannot show what more or less spend would do.")
    return out


# ---------------------------------------------------------------------------------------------
# Scenarios (never saved: they live in the browser session only)
# ---------------------------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    table: pd.DataFrame                      # one row per channel
    totals: dict                             # before/after with low/high
    warnings: list = field(default_factory=list)


def baseline(result: ResponseCurvesResult) -> dict:
    return {c: cv.base_spend for c, cv in result.usable.items()}


def apply_guardrails(result: ResponseCurvesResult, spend: dict) -> tuple[dict, list]:
    """Cap each channel's weekly spend at its guardrails; a warning names every cap."""
    from utils.formatting import format_value
    out, warnings = {}, []
    for c, value in spend.items():
        cv = result.curves[c]
        capped = min(max(float(value), cv.guard_low), cv.guard_high)
        if abs(capped - float(value)) > 0.005:
            side = "highest" if capped < value else "lowest"
            warnings.append(f"{c} was capped at {format_value(capped, 'money', compact=True)} a week: "
                            f"{cfg.SCENARIO_GUARDRAIL_HIGH if side == 'highest' else cfg.SCENARIO_GUARDRAIL_LOW:g}"
                            f" × its {side} weekly spend so far. Beyond that the curve would be a guess.")
        out[c] = capped
    return out, warnings


def move_budget(result: ResponseCurvesResult, source: str, target: str, pct: float | None = None) -> tuple[dict, list]:
    """Move pct% of the FROM channel's weekly spend to the TO channel; the total stays the same.
    The amount is reduced when it would take either channel past its guardrail."""
    from utils.formatting import format_value
    pct = cfg.SCENARIO_MOVE_PCT if pct is None else float(pct)
    spend = baseline(result)
    if source == target or source not in spend or target not in spend:
        return spend, []
    wanted = spend[source] * pct / 100
    room = min(spend[source] - result.curves[source].guard_low, result.curves[target].guard_high - spend[target])
    amount = max(0.0, min(wanted, room))
    warnings = []
    if amount < wanted - 0.005:
        warnings.append(f"Only {format_value(amount, 'money', compact=True)} a week could be moved (not "
                        f"{format_value(wanted, 'money', compact=True)}): more would take a channel outside the "
                        "spend range seen so far.")
    spend[source] -= amount
    spend[target] += amount
    return spend, warnings


def adjust_channels(result: ResponseCurvesResult, changes_pct: dict) -> tuple[dict, list]:
    """Each channel's weekly spend changed by its own % (advanced sliders), within guardrails."""
    spend = {c: s * (1 + float(changes_pct.get(c, 0.0)) / 100) for c, s in baseline(result).items()}
    return apply_guardrails(result, spend)


def suggest_allocation(result: ResponseCurvesResult, steps: int = 400) -> dict:
    """Move budget step by step from the channel where the last rupee buys the fewest conversions
    to the one where the next rupee buys the most, while that gains conversions. Each channel stays
    within +/-30% of its current spend and inside its guardrails; the total never changes. A labelled
    estimate, deterministic."""
    base = baseline(result)
    if len(base) < 2:
        return base
    limit = cfg.SCENARIO_SUGGEST_MAX_CHANGE
    curves = result.usable
    low = {c: max(curves[c].guard_low, s * (1 - limit)) for c, s in base.items()}
    high = {c: min(curves[c].guard_high, s * (1 + limit)) for c, s in base.items()}
    step = sum(base.values()) / steps
    spend = dict(base)
    order = list(base)                                   # fixed order: ties always break the same way
    for _ in range(steps * 2):
        gain = {c: curves[c].conversions(spend[c] + step) - curves[c].conversions(spend[c])
                for c in order if spend[c] + step <= high[c] + 1e-9}
        loss = {c: curves[c].conversions(spend[c]) - curves[c].conversions(spend[c] - step)
                for c in order if spend[c] - step >= low[c] - 1e-9}
        if not gain or not loss:
            break
        to = max(gain, key=lambda c: (gain[c], -order.index(c)))
        frm = min(loss, key=lambda c: (loss[c], order.index(c)))
        if to == frm or gain[to] <= loss[frm] + 1e-12:
            break
        spend[to] += step
        spend[frm] -= step
    return spend


def evaluate(result: ResponseCurvesResult, new_spend: dict, name: str = "Scenario",
             warnings: list | None = None) -> Scenario:
    """Before (the recent 8-week average week) vs after (the new weekly spend) per channel and in
    total, with 10th-90th percentile ranges. Totals cover the channels in the planner only.
    Total ranges add up each channel's low and high (a cautious, wide range)."""
    rows = []
    for c, cv in result.usable.items():
        before, after = cv.base_spend, float(new_spend.get(c, cv.base_spend))
        conv_b, conv_a = cv.conversions(before), cv.conversions(after)
        low, high = cv.conversions_range(after)
        aov = cv.aov
        rows.append({"channel": c, "pricing": cv.pricing, "spend_before": before, "spend_after": after,
                     "spend_change": after - before, "conversions_before": conv_b, "conversions_after": conv_a,
                     "conversions_low": low, "conversions_high": high,
                     "revenue_before": None if aov is None else conv_b * aov,
                     "revenue_after": None if aov is None else conv_a * aov,
                     "revenue_low": None if aov is None or low is None else low * aov,
                     "revenue_high": None if aov is None or high is None else high * aov,
                     "marginal_cost_after": cv.marginal_cost(after), "per_lakh_after": cv.per_lakh(after)})
    table = pd.DataFrame(rows)
    has_rev = not table.empty and table["revenue_before"].notna().all()

    def total(col):
        return float(table[col].sum()) if not table.empty and table[col].notna().all() else None

    spend_b, spend_a = total("spend_before"), total("spend_after")
    t = {"spend_before": spend_b, "spend_after": spend_a,
         "conversions_before": total("conversions_before"), "conversions_after": total("conversions_after"),
         "conversions_low": total("conversions_low"), "conversions_high": total("conversions_high")}
    for k in ("before", "after", "low", "high"):
        t[f"revenue_{k}"] = total(f"revenue_{k}") if has_rev else None
    # CAC = spend / conversions and ROAS = revenue / spend, recomputed from the scenario's totals.
    t["cac_before"] = safe_divide(spend_b, t["conversions_before"])
    t["cac_after"] = safe_divide(spend_a, t["conversions_after"])
    t["cac_low"] = safe_divide(spend_a, t["conversions_high"])       # more conversions = lower CAC
    t["cac_high"] = safe_divide(spend_a, t["conversions_low"])
    for k in ("before", "after", "low", "high"):
        t[f"roas_{k}"] = safe_divide(t[f"revenue_{k}"], spend_b if k == "before" else spend_a)
    return Scenario(name, table, t, list(warnings or []))


# ---------------------------------------------------------------------------------------------
# Plain English and display tables
# ---------------------------------------------------------------------------------------------

def clamp_note(curve: ChannelCurve) -> str:
    """Plain-English warning for a curve whose fitted b was above 1 and was limited to 1."""
    if not curve.clamped:
        return ""
    return (f"{curve.channel}: in the past, weeks with more spend brought even more than proportionally more "
            f"conversions (fitted b = {curve.b_raw:.2f}). That usually means spend and results rose together for "
            "another reason, such as the busy season, so returns were limited to proportional (b = 1) and "
            "confidence is Low. Treat its estimate as an upper limit.")


def per_lakh_sentence(curve: ChannelCurve) -> str:
    from utils.formatting import format_value
    n = curve.per_lakh(curve.base_spend)
    if n is None:
        return f"{curve.channel}: more spend buys no more conversions (it is at its cap)."
    return (f"{curve.channel}: the next ₹1 L here buys about {format_value(n, 'count')} conversions "
            f"(about {format_value(curve.marginal_cost(curve.base_spend), 'money', compact=True)} each).")


def checks_table(result: ResponseCurvesResult) -> pd.DataFrame:
    from utils.formatting import format_count, format_value
    c = result.checks
    if c is None or c.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Channel": c["channel"], "Pricing": c["pricing"].map({AUCTION: "Auction", PERFORMANCE: "Per result"}),
        "Weeks With Spend": [format_count(v) for v in c["weeks_with_spend"]],
        "Spend Variation (CV)": ["N/A" if v is None or pd.isna(v) else f"{v:.2f}" for v in c["cv"]],
        "Highest ÷ Lowest Week": [format_value(v, "ratio") for v in c["max_min"]],
        "Result": c["result"]})


def efficiency_table(result: ResponseCurvesResult) -> pd.DataFrame:
    """One row per channel in the planner: current weekly spend, curve, marginal cost."""
    from utils.formatting import format_date, format_value
    rows = []
    for c, cv in result.curves.items():
        if not cv.usable:
            continue
        weeks = cv.weekly.loc[cv.weekly["excluded"], "period"]
        if cv.status == PERFORMANCE:
            shape = f"Per result: CPA {format_value(cv.cpa, 'money', compact=True)}, cap {format_value(cv.cap_conversions, 'count')}/week"
        else:
            shape = f"b = {cv.b:.2f}" + (" (limited to 1)" if cv.clamped else "")
        rows.append({"Channel": c, "Weekly Spend Now": format_value(cv.base_spend, "money", compact=True),
                     "Weekly Conversions Now": format_value(cv.base_conversions, "count"),
                     "Response": shape, "Confidence": cv.confidence or "N/A (not fitted)",
                     "Next Conversion Costs": format_value(cv.marginal_cost(cv.base_spend), "money", compact=True),
                     "Next ₹1 L Buys": format_value(cv.per_lakh(cv.base_spend), "count"),
                     "Incident Weeks Left Out": "Not applicable (not fitted)" if cv.status == PERFORMANCE else
                     (", ".join(format_date(p) for p in weeks) + f" ({', '.join(cv.excluded_incidents)})")
                     if len(weeks) else "None"})
    return pd.DataFrame(rows)


def summary_frame(result: ResponseCurvesResult) -> pd.DataFrame:
    """Real numbers per qualifying channel for the Excel sheet and the PDF table."""
    rows = []
    for c, cv in result.curves.items():
        rows.append({"channel": c, "pricing": cv.pricing, "status": cv.status, "weeks_used": cv.weeks_used,
                     "b": cv.b, "b_fitted": cv.b_raw, "r_squared": cv.r2, "confidence": cv.confidence,
                     "weekly_spend_now": cv.base_spend, "weekly_conversions_now": cv.base_conversions,
                     "aov": cv.aov, "cpa": cv.cpa,
                     "marginal_cost_per_conversion": cv.marginal_cost(cv.base_spend) if cv.usable else None,
                     "conversions_per_lakh": cv.per_lakh(cv.base_spend) if cv.usable else None,
                     "guardrail_low": cv.guard_low, "guardrail_high": cv.guard_high,
                     "incidents_left_out": ", ".join(cv.excluded_incidents)})
    return pd.DataFrame(rows)


def scenario_display(s: Scenario) -> pd.DataFrame:
    from utils.formatting import format_value
    t = s.table
    if t.empty:
        return pd.DataFrame()

    def rng(lo, hi, fmt):
        return f"{format_value(lo, fmt, compact=True)} – {format_value(hi, fmt, compact=True)}"

    return pd.DataFrame({
        "Channel": t["channel"],
        "Weekly Spend Now": [format_value(v, "money", compact=True) for v in t["spend_before"]],
        "Weekly Spend in Scenario": [format_value(v, "money", compact=True) for v in t["spend_after"]],
        "Change": [("+" if v > 0.5 else "") + format_value(v, "money", compact=True) if abs(v) > 0.5 else "No change"
                   for v in t["spend_change"]],
        "Conversions Now": [format_value(v, "count") for v in t["conversions_before"]],
        "Conversions in Scenario": [format_value(v, "count") for v in t["conversions_after"]],
        "Likely Range": [rng(lo, hi, "count") for lo, hi in zip(t["conversions_low"], t["conversions_high"])],
        "Next Conversion Costs": [format_value(v, "money", compact=True) for v in t["marginal_cost_after"]]})


__all__ = ["ChannelCurve", "ResponseCurvesResult", "Scenario", "response_analysis", "fit_channel", "fit_power",
           "bootstrap_b", "spend_variation", "passes_check", "incident_weeks", "move_budget", "adjust_channels",
           "suggest_allocation", "evaluate", "apply_guardrails", "baseline", "channel_pricing", "HONEST_NOTE"]
