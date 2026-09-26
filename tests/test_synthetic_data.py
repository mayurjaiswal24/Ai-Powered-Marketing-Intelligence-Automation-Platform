"""Phase 01 tests: the synthetic dataset obeys business rules and contains the planted anomalies."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import generate_data as gen  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
RENAMED = {"spend": "Amount Spent (INR)", "conversions": "Enrollments"}


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("generated")
    gen.write_all(folder)
    return folder


@pytest.fixture(scope="module")
def clean(out_dir):
    return pd.read_csv(out_dir / "marketing_clean.csv")


# --- Shape --------------------------------------------------------------------------------------

def test_columns_and_row_count(clean):
    assert list(clean.columns) == gen.CLEAN_COLUMNS
    assert 6000 <= len(clean) <= 10000


def test_date_range(clean):
    dates = pd.to_datetime(clean["date"])
    assert dates.min() == pd.Timestamp("2025-10-01")
    assert dates.max() == pd.Timestamp("2026-09-30")


def test_one_row_per_date_campaign_region(clean):
    assert not clean.duplicated(["date", "campaign_id", "region"]).any()


def test_category_values(clean):
    assert set(clean["channel"]) == {"Paid Search", "Paid Social", "Video",
                                     "Professional Network", "Email", "Affiliate"}
    assert set(clean["platform"]) == {"Google Ads", "Meta", "YouTube", "LinkedIn",
                                      "Email (in-house)", "Affiliate Network"}
    assert set(clean["region"]) == set(gen.REGIONS)
    assert set(clean["city_tier"]) == {"Tier 1", "Tier 2", "Tier 3"}
    assert set(clean["customer_segment"]) == {"Students", "Early-Career Professionals",
                                              "Mid-Career Professionals"}
    assert set(clean["campaign_type"]) == {"Prospecting", "Retargeting", "Brand", "Promotion"}
    assert set(clean["objective"]) == {"Awareness", "Lead Generation", "Conversion"}
    assert set(clean["course_category"]) == set(gen.CATEGORY_PRICE_RANGE)


# --- Hard consistency rules --------------------------------------------------------------------

def test_funnel_consistency(clean):
    assert (clean["reach"] <= clean["impressions"]).all()
    assert (clean["clicks"] <= clean["impressions"]).all()
    assert (clean["leads"] <= clean["clicks"]).all()
    assert (clean["qualified_leads"] <= clean["leads"]).all()
    assert (clean["conversions"] <= clean["qualified_leads"]).all()


def test_no_negatives_and_no_missing(clean):
    numeric = clean.select_dtypes("number")
    assert (numeric >= 0).all().all()
    assert not clean.isna().any().any()


def test_spend_within_budget(clean):
    assert (clean["spend"] <= clean["budget"] * 1.1 + 0.01).all()


def test_zero_spend_only_on_email_weekends(clean):
    zero = clean[clean["spend"] == 0]
    assert len(zero) > 0
    assert set(zero["channel"]) == {"Email"}
    assert (pd.to_datetime(zero["date"]).dt.dayofweek >= 5).all()


def test_revenue_and_profit_consistent(clean):
    assert (clean["gross_profit"] <= clean["revenue"]).all()
    assert (clean.loc[clean["conversions"] == 0, "revenue"] == 0).all()


def test_business_realism(clean):
    """Relative patterns the spec asks for (not exact values)."""
    by_seg = clean.groupby("customer_segment")[["spend", "conversions", "revenue"]].sum()
    aov = by_seg["revenue"] / by_seg["conversions"]
    cac = by_seg["spend"] / by_seg["conversions"]
    assert aov["Students"] < aov["Early-Career Professionals"] < aov["Mid-Career Professionals"]
    assert cac["Mid-Career Professionals"] > cac["Early-Career Professionals"]

    by_ch = clean.groupby("channel")[["impressions", "clicks", "spend", "leads", "qualified_leads"]].sum()
    ctr = by_ch["clicks"] / by_ch["impressions"]
    cpc = by_ch["spend"] / by_ch["clicks"]
    qual = by_ch["qualified_leads"] / by_ch["leads"]
    assert ctr.idxmin() == "Video"
    assert ctr["Paid Search"] > ctr[["Paid Social", "Video", "Professional Network"]].max()
    assert cpc.idxmax() == "Professional Network"
    assert qual.idxmax() == "Professional Network" and qual.idxmin() == "Paid Social"

    tier = clean[clean["channel"] == "Paid Search"].groupby("city_tier")[["spend", "clicks"]].sum()
    tier_cpc = tier["spend"] / tier["clicks"]
    assert tier_cpc["Tier 1"] > tier_cpc["Tier 2"]


def test_seasonal_peaks(clean):
    monthly = clean.groupby(clean["date"].str[:7])["revenue"].sum()
    normal = monthly[["2026-02", "2026-03", "2026-04", "2026-08"]].mean()
    for peak in ["2026-01", "2026-06"]:
        assert monthly[peak] > normal * 1.2


# --- Planted anomalies -------------------------------------------------------------------------

def metric_value(rows: pd.DataFrame, metric: str, days: int) -> float:
    s = rows[["spend", "clicks", "leads", "qualified_leads", "conversions", "revenue",
              "impressions"]].sum()
    ratios = {
        "cpl": ("spend", "leads"),
        "cpc": ("spend", "clicks"),
        "lead_to_conversion_rate": ("conversions", "leads"),
        "qualified_lead_rate": ("qualified_leads", "leads"),
    }
    if metric in ratios:
        numerator, denominator = ratios[metric]
        return s[numerator] / s[denominator]
    # Additive metrics are compared per day so windows of different length are comparable.
    return s[metric] / days


@pytest.mark.parametrize("anomaly", gen.ANOMALIES, ids=lambda a: a.anomaly_id)
def test_planted_anomaly_is_visible(clean, anomaly):
    mask = pd.Series(True, index=clean.index)
    if anomaly.campaign_ids:
        mask &= clean["campaign_id"].isin(anomaly.campaign_ids)
    if anomaly.platform:
        mask &= clean["platform"] == anomaly.platform
    if anomaly.region:
        mask &= clean["region"] == anomaly.region
    dates = pd.to_datetime(clean["date"])
    start, end = pd.Timestamp(anomaly.start), pd.Timestamp(anomaly.end)
    window = clean[mask & (dates >= start) & (dates <= end)]
    baseline = clean[mask & (dates >= start - pd.Timedelta(days=28)) & (dates < start)]

    window_days = (end - start).days + 1
    assert window["date"].nunique() == window_days, "anomaly window must be fully present"
    assert baseline["date"].nunique() == 28, "needs 4 weeks of normal history before it"

    observed = metric_value(window, anomaly.metric, window_days)
    reference = metric_value(baseline, anomaly.metric, 28)
    change_pct = (observed / reference - 1) * 100
    if anomaly.direction == "up":
        assert change_pct > 0
    else:
        assert change_pct < 0
    # At least half of the documented magnitude must show up in the data.
    assert abs(change_pct) >= abs(anomaly.approx_magnitude_pct) * 0.5


def test_anomaly_and_pattern_files(out_dir):
    data = json.loads((out_dir / "ground_truth_anomalies.json").read_text(encoding="utf-8"))
    records = data["anomalies"]
    assert 8 <= len(records) <= 12
    required = {"id", "metric", "channel", "platform", "campaign_ids", "region", "start_date",
                "end_date", "expected_direction", "approx_magnitude_pct", "story"}
    for r in records:
        assert required <= set(r)
    patterns = json.loads((out_dir / "expected_patterns.json").read_text(encoding="utf-8"))
    names = " ".join(p["name"] for p in patterns["patterns"])
    assert "Diwali" in names and "January" in names and "May-July" in names


# --- Variants ----------------------------------------------------------------------------------

def test_revenue_variants(out_dir, clean):
    no_rev = pd.read_csv(out_dir / "marketing_no_revenue.csv")
    no_margin = pd.read_csv(out_dir / "marketing_no_margin.csv")
    assert "revenue" not in no_rev and "gross_profit" not in no_rev
    assert "revenue" in no_margin and "gross_profit" not in no_margin
    assert len(no_rev) == len(no_margin) == len(clean)


def test_meta_export_style(out_dir, clean):
    meta = pd.read_csv(out_dir / "meta_ads_export_style.csv")
    assert list(meta.columns) == ["Day", "Campaign name", "Amount spent (INR)", "Impressions",
                                  "Link clicks", "Results"]
    meta_clean = clean[clean["platform"] == "Meta"]
    assert meta["Results"].sum() == meta_clean["leads"].sum()
    assert meta["Amount spent (INR)"].sum() == pytest.approx(meta_clean["spend"].sum(), rel=1e-6)


def test_test_slices(out_dir):
    sizes = {"test_a_small.csv": 40, "test_b_medium.csv": 300, "test_c_large.csv": 1500}
    for name, n in sizes.items():
        part = pd.read_csv(out_dir / name)
        assert len(part) == n
        assert list(part.columns) == gen.CLEAN_COLUMNS


# --- Messy variant -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def messy(out_dir):
    # Only truly empty cells count as missing, so the literal text "N/A" stays visible.
    return pd.read_excel(out_dir / "marketing_messy.xlsx", dtype=object,
                         keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def quality_truth(out_dir):
    return json.loads((out_dir / "ground_truth_quality_issues.json").read_text(encoding="utf-8"))


def issues_of(truth, issue_type):
    return [i for i in truth["issues"] if i["type"] == issue_type]


def test_messy_structure(messy, quality_truth, clean):
    assert set(RENAMED.values()) <= set(messy.columns)
    assert "spend" not in messy.columns
    assert "Column1" in messy.columns
    assert messy.iloc[-1]["campaign_id"] == "Grand Total"
    assert quality_truth["messy_row_count"] == len(messy)
    dup_count = issues_of(quality_truth, "exact_duplicate_row")[0]["count"]
    assert len(messy) == len(clean) + dup_count + 1  # + duplicates + total row


def test_messy_duplicates_match_ground_truth(messy, quality_truth):
    body = messy.iloc[:-1]
    record = issues_of(quality_truth, "exact_duplicate_row")[0]
    assert body.duplicated().sum() == record["count"]
    for first, second in record["row_index_pairs"]:
        assert body.iloc[first].equals(body.iloc[second])


def test_messy_cells_match_ground_truth(messy, quality_truth):
    for issue in issues_of(quality_truth, "inconsistent_label") + issues_of(quality_truth, "whitespace_or_case"):
        assert (messy.loc[issue["row_index"], issue["column"]] == issue["messy_value"]).all()
    for issue in issues_of(quality_truth, "money_as_text"):
        values = messy.loc[issue["row_index"], RENAMED.get(issue["column"], issue["column"])]
        assert all(isinstance(v, str) for v in values)
    missing = {i["messy_value"]: i for i in issues_of(quality_truth, "missing_value")}
    assert (messy.loc[missing["N/A"]["row_index"], "revenue"] == "N/A").all()
    assert messy.loc[missing["(blank)"]["row_index"], "revenue"].isna().all()
    for issue in issues_of(quality_truth, "impossible_value"):
        rows = messy.loc[issue["row_index"]]
        if issue["column"] == "spend":
            assert (rows["Amount Spent (INR)"].astype(float) < 0).all()
        else:
            assert (rows["clicks"].astype(int) > rows["impressions"].astype(int)).all()
    date_issues = issues_of(quality_truth, "mixed_date_format")
    assert {i["format"] for i in date_issues} == {"iso_text", "day_first_text", "d_mon_yyyy_text"}
    for issue in date_issues:
        assert all(isinstance(v, str) for v in messy.loc[issue["row_index"], "date"])


# --- Determinism -------------------------------------------------------------------------------

def test_generator_is_deterministic(out_dir, tmp_path):
    gen.write_all(tmp_path)
    for name in ["marketing_clean.csv", "marketing_no_revenue.csv", "meta_ads_export_style.csv",
                 "test_a_small.csv", "ground_truth_anomalies.json",
                 "ground_truth_quality_issues.json"]:
        assert (tmp_path / name).read_bytes() == (out_dir / name).read_bytes(), name
    a = pd.read_excel(tmp_path / "marketing_messy.xlsx", dtype=object)
    b = pd.read_excel(out_dir / "marketing_messy.xlsx", dtype=object)
    pd.testing.assert_frame_equal(a, b)


def test_committed_sample_matches_generator(out_dir):
    """The files in data/sample must be exactly what the generator produces."""
    for name in ["marketing_clean.csv", "ground_truth_anomalies.json",
                 "ground_truth_quality_issues.json", "expected_patterns.json"]:
        assert (SAMPLE_DIR / name).read_bytes() == (out_dir / name).read_bytes(), name


def test_different_seed_changes_data():
    a = gen.generate_clean(seed=1, row_target=2000)
    b = gen.generate_clean(seed=2, row_target=2000)
    assert not np.array_equal(a["spend"].to_numpy(), b["spend"].to_numpy())
