"""Tests for the shared text -> number/date parsing and display formatting helpers."""

import pandas as pd
import pytest

from utils.formatting import format_count, format_date, group_indian
from utils.parsing import missing_mask, parse_dates, parse_numbers


@pytest.mark.parametrize("text, value, formatted", [
    ("1200", 1200.0, False),
    ("-3.5", -3.5, False),
    ("₹1,23,456", 123456.0, True),
    ("1,23,456.00", 123456.0, True),
    ("123,456", 123456.0, True),
    ("INR 5000", 5000.0, True),
    ("Rs. 500", 500.0, True),
    ("(1,200)", -1200.0, True),
    ("1200,50", 1200.5, True),       # European decimal comma
    ("1.234,56", 1234.56, True),     # European thousands dot
])
def test_parse_numbers_valid(text, value, formatted):
    numbers, is_formatted = parse_numbers(pd.Series([text]))
    assert numbers[0] == pytest.approx(value)
    assert bool(is_formatted[0]) is formatted


@pytest.mark.parametrize("text", ["", "N/A", "na", "null", "-", "none", "abc", "12%", "a,b", "1,2,3"])
def test_parse_numbers_invalid_or_missing(text):
    numbers, formatted = parse_numbers(pd.Series([text]))
    assert pd.isna(numbers[0])
    assert not formatted[0]


def test_missing_tokens():
    s = pd.Series(["", " N/A ", "NULL", "-", "None", "#N/A", "0", "x", None])
    assert missing_mask(s).tolist() == [True, True, True, True, True, True, False, False, True]


@pytest.mark.parametrize("text, expected, label", [
    ("2026-01-05", "2026-01-05", "YYYY-MM-DD"),
    ("05/01/2026", "2026-01-05", "DD/MM/YYYY"),      # Indian day-first reading
    ("5-Jan-2026", "2026-01-05", "D-Mon-YYYY"),
    ("15 March 2026", "2026-03-15", "D Month YYYY"),
    ("12/25/2026", "2026-12-25", "MM/DD/YYYY (US)"),  # only possible as month-first
    ("2026-05-10 00:00:00", "2026-05-10", "YYYY-MM-DD hh:mm:ss"),
])
def test_parse_dates(text, expected, label):
    dates, labels = parse_dates(pd.Series([text]))
    assert dates[0] == pd.Timestamp(expected)
    assert labels[0] == label


def test_parse_dates_rejects_junk():
    dates, labels = parse_dates(pd.Series(["junk", "", "N/A", "2026-13-45"]))
    assert dates.isna().all() and labels.isna().all()


def test_indian_grouping():
    assert group_indian(12345678) == "1,23,45,678"
    assert group_indian(999) == "999"
    assert group_indian(-1234567) == "-12,34,567"
    assert format_count(8497) == "8,497"
    assert format_count(float("nan")) == "N/A"
    assert format_date(pd.Timestamp("2025-10-01")) == "1 Oct 2025"
    assert format_date(pd.NaT) == "N/A"
