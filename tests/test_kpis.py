"""Phase 06 tests: every KPI checked against hand calculations, plus number formatting."""

import math

import numpy as np
import pandas as pd
import pytest

from analytics.kpis import (KPI_REGISTRY, RATIO_KPIS, compute_kpis, group_kpis,
                            period_comparison, safe_divide)
from ingestion.mapper import map_fields
from ingestion.validator import validate
from utils.formatting import (format_change, format_count, format_inr, format_pct,
                              format_ratio, format_value)


@pytest.fixture
def six_rows():
    """Hand-made data. Totals (worked out by hand):
         spend 5,500 | impressions 88,000 | clicks 800 | leads 50 | qualified 20
         conversions 6 | revenue 63,000 | gross profit 35,000 | budget 6,000
    """
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",
                                "2026-01-04", "2026-01-05", "2026-01-06"]),
        "channel": ["A", "A", "A", "B", "B", "B"],
        "spend":           [1000.0, 500.0, 300.0, 2000.0, 1500.0, 200.0],
        "budget":          [1000.0, 600.0, 400.0, 2000.0, 1700.0, 300.0],
        "impressions":     [10000, 5000, 2000, 40000, 30000, 1000],
        "clicks":          [200, 50, 0, 400, 100, 50],
        "leads":           [20, 5, 0, 10, 5, 10],
        "qualified_leads": [8, 2, 0, 4, 2, 4],
        "conversions":     [2, 1, 0, 1, 0, 2],
        "revenue":         [20000.0, 5000.0, 0.0, 30000.0, 0.0, 8000.0],
        "gross_profit":    [12000.0, 3000.0, 0.0, 15000.0, 0.0, 5000.0],
    })


def test_every_kpi_by_hand(six_rows):
    k = compute_kpis(six_rows)
    assert k["spend"].value == 5500
    assert k["revenue"].value == 63000
    # CTR = 800 / 88,000 x 100 = 0.90909...%
    assert k["ctr"].value == pytest.approx(800 / 88000 * 100)
    # CPC = 5,500 / 800 = 6.875
    assert k["cpc"].value == pytest.approx(6.875)
    # CPL = 5,500 / 50 = 110
    assert k["cpl"].value == pytest.approx(110)
    # Click-to-lead = 50 / 800 x 100 = 6.25%
    assert k["click_to_lead_rate"].value == pytest.approx(6.25)
    # Lead qualification = 20 / 50 x 100 = 40%
    assert k["lead_qualification_rate"].value == pytest.approx(40)
    # Lead-to-conversion = 6 / 50 x 100 = 12%
    assert k["lead_to_conversion_rate"].value == pytest.approx(12)
    # CAC = 5,500 / 6 conversions = 916.67 (no customers column)
    assert k["cac"].value == pytest.approx(5500 / 6)
    # ROAS = 63,000 / 5,500 = 11.4545x
    assert k["roas"].value == pytest.approx(63000 / 5500)
    # AOV = 63,000 / 6 = 10,500
    assert k["aov"].value == pytest.approx(10500)
    # ROI = (35,000 - 5,500) / 5,500 x 100 = 536.36%
    assert k["roi"].value == pytest.approx((35000 - 5500) / 5500 * 100)
    # Budget utilisation = 5,500 / 6,000 x 100 = 91.67%
    assert k["budget_utilisation"].value == pytest.approx(5500 / 6000 * 100)
    for key in RATIO_KPIS:
        assert k[key].available and k[key].formula_text == KPI_REGISTRY[key].formula_text


def test_ratio_of_sums_not_mean_of_ratios(six_rows):
    # Row CTRs are 2%, 1%, 0%, 1%, 0.333%, 5% -> their mean is 1.56%.
    # The true CTR is 800 / 88,000 = 0.91%. The engine must return the latter.
    row_ctr = six_rows["clicks"] / six_rows["impressions"] * 100
    mean_of_ratios = row_ctr.mean()
    engine = compute_kpis(six_rows)["ctr"].value
    assert mean_of_ratios == pytest.approx(1.5556, abs=1e-3)
    assert engine == pytest.approx(0.9091, abs=1e-3)
    assert abs(engine - mean_of_ratios) > 0.5


def test_group_kpis_recomputed_from_group_sums(six_rows):
    g = group_kpis(six_rows, "channel").set_index("channel")
    # Channel A: spend 1,800, clicks 250, impressions 17,000
    assert g.loc["A", "spend"] == 1800
    assert g.loc["A", "cpc"] == pytest.approx(1800 / 250)
    assert g.loc["A", "ctr"] == pytest.approx(250 / 17000 * 100)
    # Channel B: spend 3,700, clicks 550, revenue 38,000
    assert g.loc["B", "cpc"] == pytest.approx(3700 / 550)
    assert g.loc["B", "roas"] == pytest.approx(38000 / 3700)
    assert g["rows"].tolist() == [3, 3]
    # Group sums add back up to the overall totals.
    assert g["spend"].sum() == compute_kpis(six_rows)["spend"].value


def test_zero_clicks_gives_na(six_rows):
    only_zero = six_rows.iloc[[2]]            # the row with 0 clicks and 0 leads
    k = compute_kpis(only_zero)
    assert k["cpc"].value is None and k["cpc"].available
    assert k["cpc"].formatted == "N/A"
    assert "zero" in k["cpc"].note
    assert k["cpl"].formatted == "N/A"
    g = group_kpis(six_rows.assign(channel=["A", "A", "Z", "B", "B", "B"]), "channel")
    assert pd.isna(g.set_index("channel").loc["Z", "cpc"])


def test_cac_uses_customers_when_present(six_rows):
    with_customers = six_rows.assign(customers=[1, 1, 0, 1, 0, 1])   # 4 customers
    k = compute_kpis(with_customers)
    assert k["cac"].value == pytest.approx(5500 / 4)
    assert "Customers" in k["cac"].note
    without = compute_kpis(six_rows)["cac"]
    assert without.value == pytest.approx(5500 / 6)
    assert "Conversions" in without.note


def test_aov_uses_orders_when_present(six_rows):
    k = compute_kpis(six_rows.assign(orders=[3, 1, 0, 2, 0, 3]))    # 9 orders
    assert k["aov"].value == pytest.approx(63000 / 9)
    assert "Orders" in k["aov"].note


def test_roi_needs_profit_or_margin(six_rows):
    no_margin = compute_kpis(six_rows.drop(columns="gross_profit"))
    assert not no_margin["roi"].available and no_margin["roi"].value is None
    assert "no margin is assumed" in no_margin["roi"].reason_unavailable
    assert no_margin["roas"].available                     # ROAS still shown
    with_margin = compute_kpis(six_rows.drop(columns="gross_profit").assign(margin=[50] * 6))
    # Gross profit = 63,000 x 50% = 31,500 -> ROI = (31,500 - 5,500) / 5,500 x 100
    assert with_margin["roi"].value == pytest.approx((31500 - 5500) / 5500 * 100)
    assert "margin" in with_margin["roi"].note


def test_missing_revenue_rows_left_out_of_roas(six_rows):
    df = six_rows.copy()
    df.loc[0, "revenue"] = np.nan      # revenue unknown for the 1,000-spend row
    k = compute_kpis(df)
    assert k["revenue"].value == 43000
    # ROAS uses only rows where both revenue and spend are known: 43,000 / 4,500.
    assert k["roas"].value == pytest.approx(43000 / 4500)
    assert "left out" in k["roas"].note
    assert "not included" in k["revenue"].note


def test_unavailable_kpis_follow_capabilities(six_rows):
    lead_gen = six_rows.drop(columns=["revenue", "gross_profit"])
    caps = validate(map_fields(list(lead_gen.columns),
                               column_types={c: ("date" if c == "date" else "text" if c == "channel"
                                                 else "number") for c in lead_gen.columns}))
    k = compute_kpis(lead_gen, caps)
    assert not k["revenue"].available and not k["roas"].available and not k["roi"].available
    assert "no revenue field was detected" in k["roas"].reason_unavailable
    assert k["cpl"].available and k["cpl"].value == pytest.approx(110)


def test_reach_is_never_summed(six_rows):
    k = compute_kpis(six_rows.assign(reach=[900, 400, 100, 3000, 2000, 80]))
    assert "reach" not in k
    assert "reach" not in group_kpis(six_rows.assign(reach=1), "channel").columns


def test_period_comparison(six_rows):
    pc = period_comparison(six_rows, "2026-01-04", "2026-01-06")
    assert (pc.previous_start, pc.previous_end) == (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"))
    spend = pc.deltas["spend"]           # current 3,700 vs previous 1,800
    assert (spend.current, spend.previous) == (3700, 1800)
    assert spend.pct_change == pytest.approx((3700 - 1800) / 1800 * 100)
    assert spend.direction == "up" and spend.is_good is None       # spend: neither good nor bad
    cpc = pc.deltas["cpc"]               # 3,700/550 = 6.73 vs 1,800/250 = 7.20 -> down = good
    assert cpc.direction == "down" and cpc.is_good is True
    assert pc.deltas["roas"].is_good is (pc.deltas["roas"].direction == "up")
    empty = period_comparison(six_rows, "2026-01-01", "2026-01-03")
    assert empty.deltas["spend"].direction == "n/a" and empty.note


def test_safe_divide():
    assert safe_divide(10, 4) == 2.5
    assert safe_divide(1, 4, 100) == 25
    for bad in [(1, 0), (1, None), (None, 5), (1, float("nan")), (float("nan"), 2)]:
        assert safe_divide(*bad) is None


# --- Formatting --------------------------------------------------------------------------------

def test_inr_formatting():
    assert format_inr(1234567) == "₹12,34,567"
    assert format_inr(42000000, compact=True) == "₹4.2 Cr"
    assert format_inr(850000, compact=True) == "₹8.5 L"
    assert format_inr(99999, compact=True) == "₹99,999"
    assert format_inr(32.456) == "₹32.46"
    assert format_inr(-1500) == "-₹1,500"
    assert format_inr(0) == "₹0" and format_inr(5) == "₹5"
    for missing in [None, float("nan"), pd.NA, float("inf")]:
        assert format_inr(missing) == "N/A"


def test_other_formats():
    assert format_pct(12.345) == "12.3%"
    assert format_ratio(3.4512) == "3.45x"
    assert format_count(1234567) == "12,34,567"
    assert format_change(12.34) == "+12.3%" and format_change(-5) == "-5.0%"
    assert format_value(None, "percent") == "N/A" and format_ratio(None) == "N/A"
    assert format_value(5500, "money") == "₹5,500"
    assert not math.isnan(0.0) and format_pct(0.0) == "0.0%"   # a real zero is not N/A
