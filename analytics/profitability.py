"""Profitability (U4): which campaigns and channels make money after the cost of what was sold.

ROAS alone cannot say this: a 3x ROAS is a profit at a 50% gross margin but a loss at a 25%
margin. So every entity gets its OWN margin (from its own summed gross profit and revenue), its
own break-even ROAS (1 / margin) and a status against it. The formulas live in analytics/kpis.py
(profit_metrics); this module only groups rows and builds the tables.

Gross profit comes from the data (gross profit or margin column). Without it, the user may enter
an "Assumed gross margin (%)" for this session only; the result then says "Based on your assumed
margin of X%" everywhere it appears. No margin is ever assumed automatically.

Owned channels (e.g. email to the company's own list) are reported on a separate, unranked line:
their costs are mostly fixed, so a break-even ROAS test would mislead. They are left out of the
status counts, the loss-making spend and the loss-maker lists, but the overall totals include
them, so total contribution = total gross profit - total spend (the same rows as ROI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analytics.common import campaign_key, channel_key
from analytics.kpis import (LOSS, NEAR, PROFIT_PARTS, PROFIT_STATUSES, has_profit_data, profit_metrics,
                            profit_parts)
from config.fields import channel_type
from utils.formatting import format_value

TABLE_COLUMNS = ["spend", "revenue", "gross_profit", "gross_margin_pct", "roas", "break_even_roas",
                 "contribution", "profit_per_rupee", "roas_headroom", "status"]

REASON_NO_REVENUE = ("Profitability needs a Revenue column: gross margin, break-even ROAS and "
                     "contribution are all based on revenue. This data has no revenue (for example, "
                     "a lead-generation export), so profitability cannot be shown.")
REASON_NO_SPEND = "Profitability needs a Spend column to compare gross profit with marketing cost."
REASON_NO_MARGIN = ("This data has no gross profit or margin column, so profit cannot be calculated. "
                    "Enter an assumed gross margin to see an estimate; no margin is ever assumed "
                    "automatically.")


@dataclass
class ProfitabilityResult:
    available: bool
    reason: str = ""
    needs_margin: bool = False            # only the margin is missing (an assumed margin would help)
    assumed_margin_pct: float | None = None
    basis_note: str = ""                  # where gross profit comes from, shown with every result
    totals: dict = field(default_factory=dict)            # all rows (paid + owned)
    campaigns: pd.DataFrame = field(default_factory=pd.DataFrame)        # paid campaigns
    owned_campaigns: pd.DataFrame = field(default_factory=pd.DataFrame)
    channels: pd.DataFrame = field(default_factory=pd.DataFrame)         # paid channels
    owned_channels: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    status_counts: dict = field(default_factory=dict)     # paid campaigns per status
    loss_spend: float | None = None                       # spend in loss-making paid campaigns
    loss_spend_share_pct: float | None = None             # ... as % of paid campaign spend

    @property
    def assumed(self) -> bool:
        return self.assumed_margin_pct is not None

    def loss_makers(self, n: int | None = None) -> pd.DataFrame:
        """Loss-making paid campaigns, biggest ₹ loss (most negative contribution) first."""
        c = self.campaigns
        if c.empty or "status" not in c:
            return c
        rows = c[c["status"] == LOSS].sort_values("contribution")
        return rows.head(n) if n else rows


def assumed_margin_label(pct: float) -> str:
    """'Based on your assumed margin of 35%' (whole numbers without decimals)."""
    text = f"{pct:.0f}" if float(pct).is_integer() else f"{pct:.1f}"
    return f"Based on your assumed margin of {text}%"


def _grouped(parts: pd.DataFrame, keys: pd.Series, name: str) -> pd.DataFrame:
    keys = keys.astype(object).where(keys.notna(), "(none)")
    sums = parts.groupby(keys).sum(min_count=1)
    sums.index.name = name
    return profit_metrics(sums).reset_index()


def profitability_analysis(df: pd.DataFrame, assumed_margin_pct: float | None = None) -> ProfitabilityResult:
    """Profitability for the rows in `df` (the full dataset or a filtered view)."""
    if df is None or df.empty or "revenue" not in df or df["revenue"].notna().sum() == 0:
        return ProfitabilityResult(False, REASON_NO_REVENUE)
    if "spend" not in df or df["spend"].notna().sum() == 0:
        return ProfitabilityResult(False, REASON_NO_SPEND)
    from_data = has_profit_data(df)
    if not from_data and assumed_margin_pct is None:
        return ProfitabilityResult(False, REASON_NO_MARGIN, needs_margin=True)
    parts = profit_parts(df, None if from_data else assumed_margin_pct)
    if parts is None:
        return ProfitabilityResult(False, REASON_NO_MARGIN, needs_margin=not from_data)

    assumed = None if from_data else float(assumed_margin_pct)
    if assumed is not None:
        basis = assumed_margin_label(assumed) + " (gross profit = revenue x margin)."
    elif "gross_profit" in df and df["gross_profit"].notna().any():
        basis = "Based on the gross profit column in the data."
    else:
        basis = "Based on the margin column in the data (gross profit = revenue x margin)."
    result = ProfitabilityResult(True, assumed_margin_pct=assumed, basis_note=basis)

    totals = profit_metrics(parts.sum(min_count=1).to_frame().T)
    result.totals = totals.iloc[0].to_dict()

    ch_key = channel_key(df)
    if ch_key is not None:
        channels = _grouped(parts, df[ch_key], "channel")
        owned = channels["channel"].map(channel_type) == "owned"
        result.channels = channels[~owned].sort_values("contribution", ascending=False).reset_index(drop=True)
        result.owned_channels = channels[owned].reset_index(drop=True)

    camp_key = campaign_key(df)
    if camp_key is not None:
        campaigns = _grouped(parts, df[camp_key], "campaign")
        if ch_key is not None:
            # A campaign's channel = its most common channel (as in the campaign table).
            main = df.groupby(camp_key)[ch_key].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
            campaigns.insert(1, "channel", campaigns["campaign"].map(main))
            owned = campaigns["channel"].map(lambda c: c is not None and channel_type(c) == "owned")
        else:
            owned = pd.Series(False, index=campaigns.index)
        result.campaigns = campaigns[~owned].sort_values("contribution").reset_index(drop=True)
        result.owned_campaigns = campaigns[owned].reset_index(drop=True)
        paid = result.campaigns
        result.status_counts = {s: int((paid["status"] == s).sum()) for s in PROFIT_STATUSES}
        paid_spend = paid["spend"].sum(min_count=1)
        loss = paid.loc[paid["status"] == LOSS, "spend"].sum()
        result.loss_spend = float(loss)
        result.loss_spend_share_pct = float(loss / paid_spend * 100) if paid_spend else None

    if "month" in df and df["month"].notna().any():
        monthly = _grouped(parts[df["month"].notna()], df.loc[df["month"].notna(), "month"], "period")
        if "date" in df:
            monthly["days"] = monthly["period"].map(df.groupby("month")["date"].nunique())
        result.monthly = monthly.sort_values("period").reset_index(drop=True)
    return result


def summary_lines(result: ProfitabilityResult, top: int = 3) -> list[str]:
    """Plain-English summary (page, PDF): total contribution, spend in loss-makers, top loss-makers."""
    if not result.available:
        return [result.reason]
    t = result.totals
    lines = [f"Total contribution (gross profit minus spend): "
             f"{format_value(t.get('contribution'), 'money', compact=True)}, "
             f"{format_value(t.get('profit_per_rupee'), 'money')} per ₹1 of spend."]
    if result.loss_spend is not None:
        lines.append(f"Spend in loss-making campaigns: {format_value(result.loss_spend, 'money', compact=True)} "
                     f"({format_value(result.loss_spend_share_pct, 'percent')} of paid campaign spend).")
    worst = result.loss_makers(top)
    if not worst.empty:
        names = "; ".join(f"{r.campaign} ({format_value(r.contribution, 'money', compact=True)})"
                          for r in worst.itertuples())
        lines.append(f"Biggest loss-makers: {names}.")
    elif result.loss_spend is not None:
        near = result.campaigns[result.campaigns["status"] == NEAR].sort_values("contribution")
        text = "No paid campaign is loss-making."
        if not near.empty:
            names = "; ".join(f"{r.campaign} ({format_value(r.contribution, 'money', compact=True)})"
                              for r in near.head(top).itertuples())
            text += f" Closest to break-even: {names}."
        lines.append(text)
    return lines


__all__ = ["ProfitabilityResult", "profitability_analysis", "summary_lines", "assumed_margin_label",
           "TABLE_COLUMNS", "PROFIT_PARTS"]
