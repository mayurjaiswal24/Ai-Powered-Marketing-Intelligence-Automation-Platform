"""Phase 03 tests: mapping uploaded column names to canonical marketing fields."""

from pathlib import Path

import pandas as pd
import pytest

from config.fields import CRITICAL_FIELDS, FIELDS
from ingestion.loader import load_file
from ingestion.mapper import map_fields, normalize_header
from ingestion.profiler import profile_dataset

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def profile_of(name):
    df, _ = load_file(SAMPLE / name)
    return profile_dataset(df)


@pytest.fixture(scope="module")
def meta_profile():
    return profile_of("meta_ads_export_style.csv")


@pytest.mark.parametrize("raw, normalized", [
    ("Amount Spent (INR)", "amount spent"),
    ("Conv. value", "conv value"),
    ("campaign_name", "campaign name"),
    ("  Link   Clicks ", "link clicks"),
    ("Spend INR", "spend"),
    ("Cost / conv.", "cost conv"),
    ("CTR (all)", "ctr"),
])
def test_normalize_header(raw, normalized):
    assert normalize_header(raw) == normalized


def test_registry_is_consistent():
    assert set(CRITICAL_FIELDS) == {"date", "spend", "revenue", "leads", "conversions"}
    names = [f.name for f in FIELDS]
    assert len(names) == len(set(names))
    for f in FIELDS:
        assert f.ftype in {"date", "number", "money", "category", "id", "text"}


def test_meta_export_mapping(meta_profile):
    mapping = map_fields(meta_profile)
    print("\n" + mapping.to_frame().to_string())
    active = mapping.active
    assert active["spend"] == "Amount spent (INR)"
    assert active["clicks"] == "Link clicks"
    assert active["date"] == "Day"
    assert active["impressions"] == "Impressions"
    assert active["campaign_name"] == "Campaign name"

    results = mapping.by_column("Results")
    assert results.status == "uncertain"          # never silently mapped
    assert results.field is None
    assert {f for f, _ in results.candidates} == {"leads", "conversions"}
    assert "leads" not in active and "conversions" not in active
    assert [m.column for m in mapping.needs_confirmation] == ["Results"]


def test_user_override_resolves_uncertain(meta_profile):
    mapping = map_fields(meta_profile, overrides={"Results": "leads"})
    m = mapping.by_column("Results")
    assert (m.status, m.field, m.source) == ("confirmed", "leads", "user")
    assert mapping.needs_confirmation == []
    ignored = map_fields(meta_profile, overrides={"Results": None})
    assert ignored.by_column("Results").status == "ignored"
    assert ignored.needs_confirmation == []


def test_user_override_wins_a_conflict(meta_profile):
    mapping = map_fields(meta_profile, overrides={"Impressions": "clicks"})
    assert mapping.active["clicks"] == "Impressions"
    assert mapping.by_column("Link clicks").status == "uncertain"
    assert mapping.conflicts


def test_bad_override_rejected(meta_profile):
    with pytest.raises(ValueError):
        map_fields(meta_profile, overrides={"Results": "not_a_field"})
    with pytest.raises(ValueError):
        map_fields(meta_profile, overrides={"No such column": "leads"})


def test_clean_dataset_maps_fully_confirmed():
    mapping = map_fields(profile_of("marketing_clean.csv"))
    assert all(m.status == "confirmed" for m in mapping.columns), mapping.to_frame()
    assert len(mapping.active) == 21
    assert mapping.active["product_category"] == "course_category"
    assert mapping.needs_confirmation == []


def test_fuzzy_match_needs_right_data_type():
    headers = ["Impresions", "Revnue", "Dte"]
    numeric = map_fields(headers, column_types={"Impresions": "number", "Revnue": "money",
                                                "Dte": "text"})
    assert numeric.by_column("Impresions").status == "high_confidence"
    assert numeric.active["impressions"] == "Impresions"
    assert numeric.by_column("Revnue").status == "high_confidence"
    # Same typo but the values are text, not numbers -> only suggested.
    text = map_fields(["Impresions"], column_types={"Impresions": "text"})
    assert text.by_column("Impresions").status == "uncertain"
    # 'Dte' is a weak match for Date -> uncertain and, being critical, needs confirmation.
    assert numeric.by_column("Dte").status in ("uncertain", "unmapped")


def test_ambiguous_derived_and_unknown_headers():
    mapping = map_fields(["Sales", "CTR", "Cost per result", "Zodiac sign", "ROAS"])
    assert mapping.by_column("Sales").status == "uncertain"
    assert {f for f, _ in mapping.by_column("Sales").candidates} == {"revenue", "conversions", "orders"}
    assert mapping.by_column("CTR").status == "derived"
    assert mapping.by_column("Cost per result").status == "derived"
    assert mapping.by_column("ROAS").status == "derived"
    assert mapping.by_column("Zodiac sign").status == "unmapped"
    assert mapping.active == {}


def test_one_column_per_field():
    df = pd.DataFrame({"Cost": ["1"], "Spend": ["2"], "Clicks (all)": ["3"], "Link clicks": ["4"]},
                      dtype=str)
    mapping = map_fields(df)
    assert mapping.active["spend"] == "Spend"          # preferred synonym wins the tie
    assert mapping.by_column("Cost").status == "uncertain"
    assert mapping.active["clicks"] == "Clicks (all)"  # "clicks" is the canonical name
    assert len(mapping.conflicts) == 2
    # Two possible Spend columns: the user must choose (Spend is business-critical).
    assert "Cost" in [m.column for m in mapping.needs_confirmation]
