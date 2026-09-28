"""Mapping must work on ANY marketing file, not one company's layout.

Six small files with different shapes (tests/fixtures/build_mapping_fixtures.py): a Google Ads
export, a Meta Ads export, an e-commerce file, a lead-gen/CRM file, an agency report and the
column layout of a real retail test file. For each: the expected mapping, the cross-check with
the file's own calculated columns, and the plausibility warnings. Plus the clean Kalpa sample
(no warnings) and the real test file when it is present on this computer.
"""

from pathlib import Path

import pandas as pd
import pytest

from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare
from ingestion.mapper import map_fields
from ingestion.plausibility import check_plausibility
from tests.fixtures.build_mapping_fixtures import OUT, build

REAL_FILE = Path(r"real_test_file.xlsx")


@pytest.fixture(scope="module", autouse=True)
def fixture_files():
    build()                          # seeded: always the same files


def prep_of(name, overrides=None):
    raw, report = load_raw(OUT / f"{name}.csv")
    return prepare(raw, report, overrides)


def statuses(prep):
    return {m.column: m.status for m in prep.mapping.columns}


def checks(prep):
    return {c.column: c.status for c in prep.crosscheck.checks}


# --- Google Ads --------------------------------------------------------------------------------

def test_google_ads_export():
    prep = prep_of("google_ads")
    assert prep.mapping.active == {
        "date": "Day", "campaign_name": "Campaign", "campaign_type": "Campaign type",
        "spend": "Cost", "impressions": "Impr.", "clicks": "Clicks", "conversions": "Conversions",
        "revenue": "Conv. value"}
    for col in ("CTR", "Avg. CPC", "Cost / conv.", "Conv. rate"):
        assert statuses(prep)[col] == "derived"
    assert checks(prep) == {"CTR": "verified", "Avg. CPC": "verified", "Cost / conv.": "verified",
                            "Conv. rate": "verified"}
    conv_rate = next(c for c in prep.crosscheck.checks if c.column == "Conv. rate")
    assert conv_rate.definition == "Conversions ÷ Clicks × 100"      # says which definition matched
    assert prep.warnings == [] and prep.validation.can_analyse


# --- Meta Ads ----------------------------------------------------------------------------------

def test_meta_ads_export():
    prep = prep_of("meta_ads")
    active = prep.mapping.active
    assert (active["spend"], active["clicks"], active["reach"], active["date"]) == (
        "Amount spent (INR)", "Link clicks", "Reach", "Reporting starts")
    assert set(checks(prep).values()) == {"verified"} and len(checks(prep)) == 4   # CTR, CPC, CPL, CPM
    assert prep.crosscheck.verified_fields >= {"spend", "clicks", "impressions", "leads"}
    assert "verified by the file's own calculations" in prep.mapping.by_column("Link clicks").reason
    assert prep.warnings == []


# --- E-commerce --------------------------------------------------------------------------------

def test_ecommerce_file():
    prep = prep_of("ecommerce")
    m = prep.mapping
    assert m.active["spend"] == "Ad Spend"
    assert m.active["revenue"] == "Net Revenue"                    # net preferred over gross
    assert m.by_column("Gross Revenue").status == "not_used"
    assert "preferred" in m.by_column("Gross Revenue").reason
    assert m.active["conversions"] == "Orders"                     # no Conversions column
    assert "Orders used as Conversions" in m.by_column("Orders").reason
    for col in ("Product Cost", "Total Cost"):                     # never Spend
        assert m.by_column(col).status == "not_used" and "not advertising spend" in m.by_column(col).reason
    profit = m.by_column("Profit")                                 # never gross profit automatically
    assert profit.status == "uncertain" and profit.field is None
    assert profit.candidates == [("gross_profit", 0.0)] and "twice" in profit.reason
    assert "gross_profit" not in m.active
    assert profit not in m.needs_confirmation                      # a warning, not a blocker
    assert statuses(prep)["AOV"] == statuses(prep)["ROAS"] == statuses(prep)["CPC"] == "derived"
    assert checks(prep) == {"CPC": "verified", "ROAS": "verified", "AOV": "verified"}
    assert prep.warnings == [] and prep.validation.can_analyse


# --- Lead generation / CRM ---------------------------------------------------------------------

def test_leadgen_crm_marketing_cost_needs_confirmation():
    prep = prep_of("leadgen_crm")
    m = prep.mapping
    cost = m.by_column("Marketing Cost")                           # the only cost column
    assert cost.status == "uncertain" and cost.candidates == [("spend", 0.0)]
    assert cost in m.needs_confirmation and not prep.validation.can_analyse
    assert (m.active["conversions"], m.active["revenue"], m.active["channel"]) == (
        "Deals Won", "Deal Value", "Lead Source")
    cpl = next(c for c in prep.crosscheck.checks if c.column == "CPL")
    assert cpl.status == "alternative" and "Marketing Cost ÷ Leads" in cpl.message
    lead_to_deal = next(c for c in prep.crosscheck.checks if c.column == "Lead-to-Deal %")
    assert lead_to_deal.status == "verified" and lead_to_deal.definition == "Deals Won ÷ Leads × 100"

    confirmed = prep_of("leadgen_crm", {"Marketing Cost": "spend"})
    assert confirmed.validation.can_analyse
    assert checks(confirmed)["CPL"] == "verified"                  # now it matches the mapping
    assert confirmed.warnings == []


# --- Agency report -----------------------------------------------------------------------------

def test_agency_report_mismatch_is_reported():
    prep = prep_of("agency_report")
    assert prep.mapping.active["spend"] == "Media Spend"
    assert prep.mapping.active["conversions"] == "Purchases"
    assert statuses(prep)["CTR %"] == "derived"                    # "2.35%" text, still derived
    assert checks(prep)["CTR %"] == "verified"                     # "%" text parsed for the check
    mismatch = next(c for c in prep.crosscheck.checks if c.column == "CPC")
    assert mismatch.status == "mismatch"
    assert mismatch.message == ("Your file's CPC doesn't match Media Spend ÷ Clicks - please check "
                                "these mappings.")
    assert "spend" not in {f for c in prep.crosscheck.checks if c.column == "CPC" for f in c.fields}


# --- The retail layout --------------------------------------------------------------------

RETAIL_EXPECTED_ACTIVE = {"spend": "Spend", "revenue": "Net_Revenue", "conversions": "Orders",
                       "clicks": "Clicks", "leads": "Leads", "impressions": "Impressions",
                       "region": "Sales_Region"}


def _assert_retail(prep):
    m = prep.mapping
    for fname, col in RETAIL_EXPECTED_ACTIVE.items():
        assert m.active[fname] == col
    s = statuses(prep)
    for col in ("CTR_%", "CPC", "CPL", "Conversion_Rate_%", "ROAS", "ROI_%", "Average_Order_Value"):
        assert s[col] == "derived", col
    for col in ("Gross_Revenue", "Product_Cost", "Total_Cost", "Marketing_Cost"):
        assert s[col] == "not_used", col
    assert s["Profit"] == "uncertain" and "gross_profit" not in m.active
    for col in ("Discount", "Ad_Format", "Campaign_Status", "Device"):   # no nonsense suggestions
        # U2: known-but-not-analysed columns are "Recognised ... (not used)", never a field.
        assert s[col] == "not_used" and m.by_column(col).source == "recognised", col
        assert m.by_column(col).field is None and not m.by_column(col).candidates, col
    assert m.needs_confirmation == [] and prep.validation.can_analyse
    c = checks(prep)
    for col in ("CTR_%", "CPC", "CPL", "Conversion_Rate_%", "ROAS"):
        assert c[col] == "verified", col
    conv = next(x for x in prep.crosscheck.checks if x.column == "Conversion_Rate_%")
    assert conv.definition == "Orders ÷ Clicks × 100"
    assert c["Average_Order_Value"] == "alternative"               # the file uses gross revenue
    assert {w.check for w in prep.warnings} == {"roas_high", "organic_spend"}


def test_retail_column_layout():
    _assert_retail(prep_of("retail_columns"))


@pytest.mark.skipif(not REAL_FILE.exists(), reason="the real test file is not on this computer")
def test_real_file():
    raw, report = load_raw(REAL_FILE)
    _assert_retail(prepare(raw, report))


# --- The clean sample stays clean ---------------------------------------------------------------

def test_kalpa_clean_sample_maps_fully_with_no_warnings():
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    prep = prepare(raw, report)
    assert all(m.status == "confirmed" for m in prep.mapping.columns)
    assert prep.mapping.needs_confirmation == [] and prep.validation.can_analyse
    assert prep.warnings == []
    assert prep.crosscheck.mismatches == []


# --- General rules -----------------------------------------------------------------------------

def test_derived_by_name_or_values():
    df = pd.DataFrame({"Spend": [100.0, 200.0, 300.0, 400.0], "Avg basket": [1.0, 2.0, 3.0, 4.0],
                       "Share of voice": [12.5, 13.1, 14.7, 9.9],
                       "Engagement score": ["12.1%", "13.4%", "9.8%", "11.2%"]})
    from ingestion.profiler import profile_dataset
    m = map_fields(profile_dataset(df))
    assert m.by_column("Share of voice").status == "derived"          # 'share' is a derived word
    assert m.by_column("Engagement score").status == "derived"        # values look like percentages
    assert "percent" in m.by_column("Engagement score").reason
    assert m.active == {"spend": "Spend"}


def test_marketing_cost_is_not_used_when_spend_exists():
    m = map_fields(["Spend", "Marketing cost", "COGS", "Operating cost"])
    assert m.active == {"spend": "Spend"}
    assert {m.by_column(c).status for c in ("Marketing cost", "COGS", "Operating cost")} == {"not_used"}


def test_every_mapping_has_a_reason():
    for name in ("google_ads", "meta_ads", "ecommerce", "leadgen_crm", "agency_report", "retail_columns"):
        prep = prep_of(name)
        assert all(m.reason for m in prep.mapping.columns), name


@pytest.mark.parametrize("totals, expected", [
    ({"Clicks": 10, "Leads": 20}, "leads_gt_clicks"),
    ({"Leads": 10, "Conversions": 20}, "conversions_gt_leads"),
    ({"Impressions": 10, "Clicks": 20}, "clicks_gt_impressions"),
    ({"Impressions": 10, "Reach": 20}, "reach_gt_impressions"),
    ({"Impressions": 100, "Clicks": 60}, "ctr_high"),
    ({"Leads": 10000, "Conversions": 5}, "lead_to_conv_low"),
    ({"Spend": 10, "Revenue": 600}, "roas_high"),
])
def test_plausibility_checks(totals, expected):
    df = pd.DataFrame({k: [v / 2, v / 2] for k, v in totals.items()})
    found = {w.check for w in check_plausibility(df, map_fields(df))}
    assert expected in found


def test_plausibility_organic_spend_and_no_false_alarm():
    df = pd.DataFrame({"Channel": ["Organic Social", "Paid Search", "SEO"], "Spend": [500, 1000, 0],
                       "Impressions": [10000, 20000, 5000], "Clicks": [200, 400, 100],
                       "Leads": [20, 40, 10], "Conversions": [2, 4, 1], "Revenue": [2000, 4000, 1000]})
    warnings = check_plausibility(df, map_fields(df))
    assert [w.check for w in warnings] == ["organic_spend"]          # SEO had no spend: no warning
    assert "Organic Social" in warnings[0].message
