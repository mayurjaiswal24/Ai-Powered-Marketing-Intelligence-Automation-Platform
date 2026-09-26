"""Phase 03 tests: profiling finds the problems we planted in the messy file, and nothing
in the clean file."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ingestion.loader import load_file
from ingestion.mapper import map_fields
from ingestion.profiler import cluster_labels, profile_dataset

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def analyse(name):
    df, _ = load_file(SAMPLE / name)
    profile = profile_dataset(df)
    mapping = map_fields(profile)
    profile.apply_mapping(df, mapping)
    return df, profile, mapping


@pytest.fixture(scope="module")
def messy():
    return analyse("marketing_messy.xlsx")


@pytest.fixture(scope="module")
def truth():
    return json.loads((SAMPLE / "ground_truth_quality_issues.json").read_text(encoding="utf-8"))


def rows_of(profile, kind, column=None):
    return {r for f in profile.finding(kind, column) for r in f.rows}


def truth_rows(truth, issue_type, **match):
    rows = set()
    for issue in truth["issues"]:
        if issue["type"] == issue_type and all(issue.get(k) == v for k, v in match.items()):
            rows.update(issue["row_index"])
    return rows


def detection_report(profile, truth) -> dict[str, tuple[int, int, int]]:
    """issue -> (planted, detected correctly, false alarms)."""
    spend_col = "Amount Spent (INR)"
    checks = {
        "duplicate rows": ({p[1] for i in truth["issues"] if i["type"] == "exact_duplicate_row"
                            for p in i["row_index_pairs"]}, set(profile.duplicate_rows)),
        "platform labels": (truth_rows(truth, "inconsistent_label"),
                            rows_of(profile, "inconsistent_labels", "platform")),
        "channel whitespace/case": (truth_rows(truth, "whitespace_or_case", column="channel"),
                                    rows_of(profile, "inconsistent_labels", "channel")),
        "region whitespace/case": (truth_rows(truth, "whitespace_or_case", column="region"),
                                   rows_of(profile, "inconsistent_labels", "region")),
        # "iso_text" dates are identical text to real dates, so only the other two are findable.
        "mixed date formats": (truth_rows(truth, "mixed_date_format", format="day_first_text")
                               | truth_rows(truth, "mixed_date_format", format="d_mon_yyyy_text"),
                               rows_of(profile, "mixed_date_formats", "date")),
        "spend as text": (truth_rows(truth, "money_as_text", column="spend"),
                          rows_of(profile, "money_as_text", spend_col)),
        "revenue as text": (truth_rows(truth, "money_as_text", column="revenue"),
                            rows_of(profile, "money_as_text", "revenue")),
        "missing revenue": (truth_rows(truth, "missing_value", column="revenue"),
                            rows_of(profile, "missing_values", "revenue")),
        "negative spend": (truth_rows(truth, "impossible_value", column="spend"),
                           rows_of(profile, "negative_values", spend_col)),
        "clicks > impressions": (truth_rows(truth, "impossible_value", column="clicks"),
                                 rows_of(profile, "clicks_exceed_impressions")),
        "total row": (truth_rows(truth, "total_row"), set(profile.summary_rows)),
    }
    return {name: (len(planted), len(planted & found), len(found - planted))
            for name, (planted, found) in checks.items()}


def test_messy_detection_rate(messy, truth):
    _, profile, _ = messy
    report = detection_report(profile, truth)
    print("\nDetection rate per issue type (planted / found / false alarms):")
    for name, (planted, found, false_alarms) in report.items():
        print(f"  {name:<26} {planted:>4} {found:>4} {false_alarms:>4}   {found / planted:.0%}")
    for name, (planted, found, false_alarms) in report.items():
        assert planted > 0, name
        assert found == planted, f"{name}: missed {planted - found}"
        assert false_alarms == 0, f"{name}: {false_alarms} false alarms"


def test_messy_junk_column_and_renamed_headers(messy):
    _, profile, mapping = messy
    assert [f.column for f in profile.finding("mostly_empty_column")] == ["Column1"]
    assert mapping.active["spend"] == "Amount Spent (INR)"
    assert mapping.active["conversions"] == "Enrollments"
    assert mapping.by_column("Column1").status == "unmapped"


def test_messy_label_suggestions(messy):
    _, profile, _ = messy
    platform = [c for c in profile.label_clusters if c.column == "platform"]
    assert len(platform) == 1
    assert platform[0].suggested_label == "Meta"
    assert set(platform[0].inconsistent_variants) == {"facebook", "FB", "Meta Ads", "meta "}
    suggestions = {c.suggested_label for c in profile.label_clusters if c.column == "region"}
    assert suggestions == {"North", "South", "West", "Central"}


def test_messy_types_and_ranges(messy):
    _, profile, _ = messy
    assert profile.columns["date"].inferred_type == "date"
    assert profile.columns["Amount Spent (INR)"].inferred_type == "money"
    assert profile.columns["revenue"].inferred_type == "money"
    assert profile.columns["channel"].inferred_type == "text"
    assert profile.date_min == pd.Timestamp("2025-10-01")
    assert profile.date_max == pd.Timestamp("2026-09-30")
    assert profile.columns["Amount Spent (INR)"].min_value < 0   # the planted negatives


def test_messy_text_summary(messy):
    _, profile, _ = messy
    text = profile.to_text()
    assert text.startswith("DATASET PROFILE")
    for phrase in ["Rows: 8,497", "Columns: 22", "Date range: 1 Oct 2025 - 30 Sep 2026",
                   "25 duplicate rows", "Grand Total", "80 missing values in 'revenue'",
                   "'facebook', 'FB', 'Meta Ads', 'meta ' look like 'Meta'",
                   "Spend", "<- 'Amount Spent (INR)'"]:
        assert phrase in text, phrase
    assert all(ord(ch) < 0x2000 or ch == "₹" for ch in text), "no emojis/symbols in UI text"


def test_clean_file_has_no_findings():
    _, profile, _ = analyse("marketing_clean.csv")
    assert profile.findings == []
    assert profile.label_clusters == []
    assert profile.duplicate_rows == [] and profile.summary_rows == []
    assert "No problems found." in profile.to_text()


def test_cluster_labels_does_not_merge_different_values():
    s = pd.Series(["Tier 1", "Tier 2", "KL-GS-001", "KL-GS-002", "Early-Career Professionals",
                   "Mid-Career Professionals", "Paid Search", "Paid Social", "North", "South"])
    assert cluster_labels(s) == []


def test_cluster_labels_catches_typos_and_aliases():
    s = pd.Series(["LinkedIn"] * 5 + ["Linkdin"] + ["Google Ads"] * 3 + ["adwords", "google"])
    clusters = {c.suggested_label: set(c.inconsistent_variants) for c in cluster_labels(s)}
    assert clusters == {"LinkedIn": {"Linkdin"}, "Google Ads": {"adwords", "google"}}
