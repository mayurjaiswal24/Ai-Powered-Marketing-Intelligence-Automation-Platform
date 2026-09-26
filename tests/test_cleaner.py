"""Phase 04 tests: cleaning turns the messy file back into the clean data, and logs every change."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ingestion.loader import load_file
from ingestion.mapper import map_fields
from ingestion.profiler import profile_dataset
from processing.cleaner import CleaningError, clean_dataset

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"
KEYS = ["date", "campaign_id", "region"]


def clean_file(name, **kwargs):
    raw, _ = load_file(SAMPLE / name)
    return clean_dataset(raw, map_fields(profile_dataset(raw)), **kwargs)


@pytest.fixture(scope="module")
def messy_result():
    return clean_file("marketing_messy.xlsx")


@pytest.fixture(scope="module")
def reference():
    df = pd.read_csv(SAMPLE / "marketing_clean.csv", parse_dates=["date"])
    return df.rename(columns={"course_category": "product_category"})


@pytest.fixture(scope="module")
def truth():
    return json.loads((SAMPLE / "ground_truth_quality_issues.json").read_text(encoding="utf-8"))


def truth_issues(truth, issue_type, column=None):
    return [i for i in truth["issues"] if i["type"] == issue_type
            and (column is None or i.get("column") == column)]


# --- Messy file vs clean original --------------------------------------------------------------

def test_row_count_and_keys(messy_result, reference):
    df = messy_result.clean_df
    assert len(df) == len(reference)
    assert not df.duplicated(KEYS).any()
    merged = df.merge(reference, on=KEYS, suffixes=("", "_ref"), how="outer", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_categories_restored_exactly(messy_result, reference):
    merged = messy_result.clean_df.merge(reference, on=KEYS, suffixes=("", "_ref"))
    for col in ["campaign_name", "campaign_type", "objective", "channel", "platform",
                "city_tier", "customer_segment", "product_category"]:
        mismatch = merged[col].astype(str) != merged[f"{col}_ref"].astype(str)
        assert not mismatch.any(), f"{col}: {merged.loc[mismatch, [col, col + '_ref']].head()}"


def test_totals_match_within_explained_difference(messy_result, reference, truth):
    df = messy_result.clean_df

    # Spend: negatives are excluded (set to missing); "₹1,23,456" and "INR 5000" text was
    # written rounded to whole rupees, so up to 0.5 rupee per such cell differs.
    negative = sum(truth_issues(truth, "impossible_value", "spend")[0]["clean_values"])
    rounding = sum(v - round(v) for i in truth_issues(truth, "money_as_text", "spend")
                   if i["format"] in ("rupee_symbol_indian_grouping", "inr_prefix")
                   for v in i["clean_values"])
    expected_spend = reference["spend"].sum() - negative - rounding
    assert df["spend"].sum() == pytest.approx(expected_spend, abs=0.01)

    # Revenue: blank and "N/A" cells stay missing (not zero), so their true values are absent.
    missing_revenue = sum(v for i in truth_issues(truth, "missing_value", "revenue")
                          for v in i["clean_values"])
    assert df["revenue"].sum() == pytest.approx(reference["revenue"].sum() - missing_revenue, abs=0.01)

    # Clicks: the impossible "clicks > impressions" values are excluded.
    bad_clicks = sum(truth_issues(truth, "impossible_value", "clicks")[0]["clean_values"])
    assert df["clicks"].sum() == reference["clicks"].sum() - bad_clicks

    # Untouched by the messy injections: must match exactly.
    for col in ["leads", "conversions", "impressions", "qualified_leads", "gross_profit", "budget"]:
        assert df[col].sum() == pytest.approx(reference[col].sum()), col


def test_every_value_matches_except_logged_ones(messy_result, reference):
    """Cell by cell: any difference from the clean original must be explained by the log."""
    df = messy_result.clean_df
    merged = df.merge(reference, on=KEYS, suffixes=("", "_ref"))
    log = messy_result.quality_log_df
    for col in ["spend", "revenue", "clicks", "impressions", "leads", "conversions"]:
        a, b = merged[col].astype(float), merged[f"{col}_ref"].astype(float)
        differs = ~np.isclose(a.fillna(-1), b, atol=0.5)
        explained = set(log.loc[log["column"] == col, "row"].dropna().astype(int))
        unexplained = set(merged.loc[differs, "source_row"]) - explained
        assert not unexplained, f"{col}: unexplained differences in rows {sorted(unexplained)[:5]}"


def test_log_matches_ground_truth(messy_result, truth):
    log = messy_result.quality_log_df
    counts = log.groupby("action").size()
    assert counts["duplicate_removed"] == 25
    assert counts["summary_row_removed"] == 1
    assert counts["date_standardized"] == 300        # the text-formatted dates (ISO text included)
    assert counts["text_to_number"] == 180
    assert counts["missing_value"] == 80
    assert counts["invalid_value_removed"] == 24
    assert counts["column_dropped"] == 1
    assert (log.loc[log["action"] == "column_dropped", "column"] == "Column1").all()

    negative_rows = set(truth_issues(truth, "impossible_value", "spend")[0]["row_index"])
    logged = set(log.loc[(log["action"] == "invalid_value_removed") & (log["column"] == "spend"), "row"])
    assert logged == negative_rows
    # The original bad value is kept in the log, the cleaned value is shown as missing.
    entry = log[(log["action"] == "invalid_value_removed") & (log["column"] == "spend")].iloc[0]
    assert entry["original_value"].startswith("-") and entry["cleaned_value"] == "(missing)"
    assert entry["severity"] == "error"


def test_invalid_values_flagged_not_dropped(messy_result):
    df = messy_result.clean_df
    negative = df["dq_flags"].str.contains("negative_spend")
    assert negative.sum() == 12 and df.loc[negative, "spend"].isna().all()
    assert df.loc[negative, "leads"].notna().all()   # the rest of the row is still usable
    assert df["dq_flags"].str.contains("clicks_gt_impressions").sum() == 12


def test_missing_revenue_stays_missing_not_zero(messy_result):
    df = messy_result.clean_df
    assert df["revenue"].isna().sum() == 80


def test_types_and_time_columns(messy_result):
    df = messy_result.clean_df
    assert str(df["date"].dtype).startswith("datetime64")
    assert str(df["clicks"].dtype) == "Int64" and str(df["spend"].dtype) == "float64"
    row = df[df["date"] == pd.Timestamp("2026-01-07")].iloc[0]   # a Wednesday
    assert row["week_start"] == pd.Timestamp("2026-01-05")
    assert (row["month"], row["quarter"], row["year"]) == ("2026-01", "2026-Q1", 2026)
    assert (df["week_start"].dt.dayofweek == 0).all()


def test_summary_text(messy_result):
    text = messy_result.quality_summary.to_text()
    for phrase in ["Rows in uploaded file: 8,497", "Rows after cleaning:   8,471",
                   "Duplicate rows removed: 25", "80 rows have no usable revenue",
                   "revenue totals may be understated", "read day-first"]:
        assert phrase in text, phrase
    print("\n" + text)


# --- Unit-level cases --------------------------------------------------------------------------

def tiny(rows):
    raw = pd.DataFrame(rows, dtype=str)
    return clean_dataset(raw, map_fields(profile_dataset(raw)))


def test_money_and_missing_cases():
    result = tiny({"date": ["2026-01-05"] * 5, "channel": ["Email"] * 5,
                   "spend": ["₹1,23,456", "1,23,456.00", "INR 5000", "N/A", ""],
                   "leads": ["1", "2", "3", "4", "5"]})
    spend = result.clean_df["spend"].tolist()
    assert spend[:3] == [123456.0, 123456.0, 5000.0]
    assert np.isnan(spend[3]) and np.isnan(spend[4])       # missing, NOT zero


def test_date_cases_day_first():
    result = tiny({"date": ["05/01/2026", "2026-01-05", "5-Jan-2026", "25/12/2025"],
                   "spend": ["1", "2", "3", "4"]})
    dates = result.clean_df["date"].tolist()
    assert dates[:3] == [pd.Timestamp("2026-01-05")] * 3
    assert dates[3] == pd.Timestamp("2025-12-25")
    log = result.quality_log_df
    ambiguous = log[log["action"] == "ambiguous_date"]
    assert ambiguous["original_value"].tolist() == ["05/01/2026"]   # 25/12 cannot be ambiguous


def test_label_standardization():
    result = tiny({"platform": ["Meta"] * 3 + ["FB", "facebook", " meta ", "Meta Ads", "Google Ads", "adwords"],
                   "channel": ["Paid Social"] * 7 + ["Paid Search", "paid search"],
                   "spend": [str(i) for i in range(1, 10)]})   # distinct rows, not duplicates
    df = result.clean_df
    assert df["platform"].tolist() == ["Meta"] * 7 + ["Google Ads"] * 2
    assert set(df["channel"]) == {"Paid Social", "Paid Search"}


def test_flag_only_duplicate_policy():
    result = clean_file("test_a_small.csv")
    raw, _ = load_file(SAMPLE / "test_a_small.csv")
    doubled = pd.concat([raw, raw.iloc[:3]], ignore_index=True)
    flagged = clean_dataset(doubled, map_fields(profile_dataset(doubled)), duplicate_policy="flag")
    assert len(flagged.clean_df) == len(result.clean_df) + 3
    assert flagged.clean_df["dq_flags"].str.contains("duplicate").sum() == 3
    removed = clean_dataset(doubled, map_fields(profile_dataset(doubled)))
    assert len(removed.clean_df) == len(result.clean_df)


def test_unconfirmed_critical_mapping_refused():
    raw, _ = load_file(SAMPLE / "meta_ads_export_style.csv")
    with pytest.raises(CleaningError) as info:
        clean_dataset(raw, map_fields(profile_dataset(raw)))
    assert "Results" in info.value.user_message
    ok = clean_dataset(raw, map_fields(profile_dataset(raw), overrides={"Results": "leads"}))
    assert ok.clean_df["leads"].sum() == raw["Results"].astype(int).sum()


# --- Stability ---------------------------------------------------------------------------------

def test_cleaning_twice_gives_identical_output(messy_result):
    again = clean_file("marketing_messy.xlsx")
    pd.testing.assert_frame_equal(again.clean_df, messy_result.clean_df)
    pd.testing.assert_frame_equal(again.quality_log_df, messy_result.quality_log_df)


def test_clean_dataset_passes_through():
    result = clean_file("marketing_clean.csv")
    log = result.quality_log_df
    assert len(result.clean_df) == 8471
    assert set(log["severity"]) <= {"info"}                       # no warnings, no errors
    assert set(log["action"]) <= {"outlier_flagged"}              # only review flags
    assert len(log) < 0.01 * len(result.clean_df)                 # near-zero entries
    reference = pd.read_csv(SAMPLE / "marketing_clean.csv")
    for col in ["spend", "revenue", "leads", "conversions", "clicks"]:
        assert result.clean_df[col].sum() == pytest.approx(reference[col].sum())
