"""Phase 03 tests: capability detection switches analyses on/off based on available fields."""

from pathlib import Path

import pandas as pd
import pytest

from ingestion.loader import load_file
from ingestion.mapper import map_fields
from ingestion.profiler import profile_dataset
from ingestion.validator import validate

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"
ALL_CAPABILITIES = {"time_series", "ctr", "cpc", "cpl", "revenue_metrics", "roi", "aov", "cac",
                    "funnel", "campaign_analysis", "channel_analysis", "segment_analysis",
                    "geography_analysis", "product_analysis", "anomaly_detection"}


def validate_file(name, overrides=None):
    df, _ = load_file(SAMPLE / name)
    profile = profile_dataset(df)
    mapping = map_fields(profile, overrides=overrides)
    return validate(mapping, profile)


def disabled(result):
    return {k for k, c in result.capabilities.items() if not c.enabled}


def test_clean_dataset_enables_everything():
    result = validate_file("marketing_clean.csv")
    assert set(result.capabilities) == ALL_CAPABILITIES
    assert disabled(result) == set()
    assert result.can_analyse
    assert result.cac_denominator == "conversions"
    assert any("Spend / Conversions" in n for n in result.notes)
    assert result.capabilities["roi"].details["basis"] == "gross_profit"
    assert result.funnel_stages == ["impressions", "clicks", "leads", "qualified_leads", "conversions"]


def test_no_revenue_disables_revenue_roas_roi():
    result = validate_file("marketing_no_revenue.csv")
    assert disabled(result) == {"revenue_metrics", "roi", "aov"}
    assert "no revenue field was detected" in result.capabilities["revenue_metrics"].reason
    assert result.can_analyse


def test_no_margin_disables_only_roi():
    result = validate_file("marketing_no_margin.csv")
    assert disabled(result) == {"roi"}
    assert "no margin is assumed" in result.capabilities["roi"].reason


def test_meta_export_blocks_until_results_confirmed():
    result = validate_file("meta_ads_export_style.csv")
    assert not result.can_analyse
    assert "Results" in result.blocking[0]
    assert not result.enabled("cpl")

    confirmed = validate_file("meta_ads_export_style.csv", overrides={"Results": "leads"})
    assert confirmed.can_analyse
    assert confirmed.enabled("cpl") and confirmed.enabled("ctr") and confirmed.enabled("time_series")
    assert not confirmed.enabled("revenue_metrics")
    assert not confirmed.enabled("channel_analysis")


def test_no_core_metrics_is_blocking():
    df = pd.DataFrame({"date": ["2026-01-01"], "campaign": ["A"], "impressions": ["100"]}, dtype=str)
    profile = profile_dataset(df)
    result = validate(map_fields(profile), profile)
    assert not result.can_analyse
    assert "Spend, Revenue, Leads or Conversions" in result.blocking[0]


def test_unreadable_date_disables_trends():
    df = pd.DataFrame({"date": ["soon", "later", "someday", "2026-01-01"],
                       "spend": ["1", "2", "3", "4"]}, dtype=str)
    profile = profile_dataset(df)
    result = validate(map_fields(profile), profile)
    assert not result.enabled("time_series")
    assert not result.enabled("anomaly_detection")
    assert "could not be read as dates" in result.capabilities["time_series"].reason


def test_cac_prefers_customers_and_aov_prefers_orders():
    mapping = map_fields(["date", "spend", "revenue", "conversions", "customers", "orders"],
                         column_types={"date": "date", "spend": "number", "revenue": "number",
                                       "conversions": "number", "customers": "number",
                                       "orders": "number"})
    result = validate(mapping)
    assert result.cac_denominator == "customers"
    assert result.aov_denominator == "orders"


@pytest.mark.parametrize("fields, roi_on", [
    (["spend", "revenue", "gross_profit"], True),
    (["spend", "revenue", "margin"], True),
    (["spend", "revenue"], False),
])
def test_roi_needs_profit_or_margin(fields, roi_on):
    result = validate(map_fields(fields))
    assert result.enabled("roi") is roi_on
