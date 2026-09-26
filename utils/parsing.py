"""Turning messy text cells into numbers and dates.

Shared by profiling (Phase 03: "how many values would parse?") and cleaning (Phase 04:
"parse them"). Keeping one implementation means the profile never promises something the
cleaner cannot deliver.
"""

from __future__ import annotations

import pandas as pd

# Cell values that mean "no value". Compared after trimming and lower-casing.
MISSING_TOKENS = {"", "n/a", "na", "null", "-", "none", "nan", "#n/a"}

# Currency markers that may appear in money cells: "₹1,23,456", "INR 5000", "Rs. 500".
_CURRENCY_PATTERN = r"(?i)(?:₹|rs\.?|inr|usd|\$|€|£)"

# Date formats we accept, in priority order. Indian convention is day-first, so 05/01/2026 is
# read as 5 January 2026. US-style month-first is tried last and only catches dates that
# cannot be day-first (e.g. 12/25/2026).
DATE_FORMATS = [
    ("YYYY-MM-DD", "%Y-%m-%d"),
    ("YYYY-MM-DD hh:mm:ss", "%Y-%m-%d %H:%M:%S"),
    ("DD/MM/YYYY", "%d/%m/%Y"),
    ("DD-MM-YYYY", "%d-%m-%Y"),
    ("DD.MM.YYYY", "%d.%m.%Y"),
    ("D-Mon-YYYY", "%d-%b-%Y"),
    ("D Mon YYYY", "%d %b %Y"),
    ("D-Month-YYYY", "%d-%B-%Y"),
    ("D Month YYYY", "%d %B %Y"),
    ("Mon D, YYYY", "%b %d, %Y"),
    ("Month D, YYYY", "%B %d, %Y"),
    ("YYYY/MM/DD", "%Y/%m/%d"),
    ("MM/DD/YYYY (US)", "%m/%d/%Y"),
]


def missing_mask(series: pd.Series) -> pd.Series:
    """True where the cell is empty or a 'no value' token such as N/A."""
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin(MISSING_TOKENS)


def parse_numbers(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse text into numbers.

    Handles currency symbols/words, Indian or Western thousand separators ("1,23,456" and
    "123,456"), accounting negatives "(1,200)", and European decimals ("1200,50", "1.234,56").

    Returns (numbers as float with NaN where unparseable or missing,
             mask of cells written as formatted text, i.e. with a currency marker or separators).
    """
    text = series.fillna("").astype(str).str.strip()
    missing = text.str.lower().isin(MISSING_TOKENS)

    # Fast path: plain numbers like "1200" or "-3.5" (the vast majority of cells).
    plain = text.str.fullmatch(_PLAIN_NUMBER) & ~missing
    numbers = pd.Series(float("nan"), index=series.index, dtype=float)
    numbers[plain] = pd.to_numeric(text[plain], errors="coerce")
    formatted = pd.Series(False, index=series.index)

    # Slow path, once per distinct leftover value: "₹1,23,456", "INR 5000", "(1,200)" ...
    todo = ~plain & ~missing
    if todo.any():
        unique = pd.Series(text[todo].unique())
        values, is_formatted = _parse_formatted(unique)
        lookup_value = dict(zip(unique, values))
        lookup_formatted = dict(zip(unique, is_formatted))
        numbers[todo] = text[todo].map(lookup_value).astype(float)
        formatted[todo] = text[todo].map(lookup_formatted).astype(bool)
    return numbers, formatted


_PLAIN_NUMBER = r"-?\d+(\.\d+)?|-?\.\d+"


def _parse_formatted(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    has_currency = text.str.contains(_CURRENCY_PATTERN, regex=True)
    work = text.str.replace(_CURRENCY_PATTERN, "", regex=True)
    work = work.str.replace(r"[\s\u00a0]", "", regex=True)

    negative_paren = work.str.fullmatch(r"\(.*\)")
    work = work.str.replace(r"^\((.*)\)$", r"\1", regex=True)

    # European style: "1.234,56" -> "1234.56" and "1200,50" -> "1200.50". A trailing comma
    # followed by 1-2 digits can never be Indian/Western grouping (those groups end in 3 digits).
    euro_full = work.str.fullmatch(r"-?\d{1,3}(\.\d{3})+,\d+")
    work = work.mask(euro_full, work.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    euro_decimal = work.str.fullmatch(r"-?\d+,\d{1,2}")
    work = work.mask(euro_decimal, work.str.replace(",", ".", regex=False))

    # Thousand separators must sit between digits: "1,23,456" yes, "a,b" no.
    has_separator = work.str.fullmatch(r"-?\d{1,3}(,\d{2,3})+(\.\d+)?")
    work = work.mask(has_separator, work.str.replace(",", "", regex=False))

    numbers = pd.to_numeric(work.where(work.str.fullmatch(_PLAIN_NUMBER), None), errors="coerce")
    numbers = numbers.where(~negative_paren, -numbers)
    formatted = (has_currency | has_separator | euro_full | euro_decimal) & numbers.notna()
    return numbers.astype(float), formatted


def parse_dates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse text dates written in any of DATE_FORMATS.

    Returns (datetimes with NaT where unparseable or missing,
             the format label that matched each cell, or None).
    """
    text = series.fillna("").astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    labels = pd.Series(None, index=series.index, dtype=object)
    remaining = ~missing_mask(series)
    for label, fmt in DATE_FORMATS:
        if not remaining.any():
            break
        parsed = pd.to_datetime(text[remaining], format=fmt, errors="coerce")
        hit = parsed.notna()
        if hit.any():
            idx = parsed[hit].index
            result.loc[idx] = parsed[hit].astype("datetime64[ns]")
            labels.loc[idx] = label
            remaining.loc[idx] = False
    return result, labels
