"""All number and date formatting for display (dashboard, PDF, Excel, text summaries).

Indian conventions (locked decisions):
  - money in INR with ₹ and Indian grouping: ₹12,34,567
  - compact money for KPI cards in lakh (L) and crore (Cr): ₹8.5 L, ₹4.2 Cr
  - percentages to 1 decimal: 12.3%
  - ROAS to 2 decimals with "x": 3.45x
  - anything missing or impossible shows "N/A", never 0
"""

from __future__ import annotations

import datetime as dt
import math

NA = "N/A"
LAKH = 1_00_000
CRORE = 1_00_00_000


def _is_missing(value) -> bool:
    """None, NaN, pd.NA and infinity all count as 'no value'."""
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(number) or math.isinf(number)


def group_indian(whole: int) -> str:
    """12345678 -> '1,23,45,678' (last 3 digits, then groups of 2)."""
    sign = "-" if whole < 0 else ""
    digits = str(abs(whole))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return sign + ",".join(groups + [tail])


def format_count(value) -> str:
    """Whole-number counts (rows, clicks, leads) with Indian grouping: 1,23,456."""
    if _is_missing(value):
        return NA
    return group_indian(int(round(float(value))))


def format_inr(value, compact: bool = False, decimals: int | None = None) -> str:
    """Rupees.

    Full:    1234567 -> '₹12,34,567'; small amounts keep paise: 32.456 -> '₹32.46'
    Compact: 42000000 -> '₹4.2 Cr'; 850000 -> '₹8.5 L'; below one lakh shown in full.
    `decimals` forces the number of decimals in full mode (default: 0 from ₹100 up, else 2).
    """
    if _is_missing(value):
        return NA
    number = float(value)
    sign = "-" if number < 0 else ""
    number = abs(number)
    if compact and number >= CRORE:
        return f"{sign}₹{number / CRORE:.1f} Cr"
    if compact and number >= LAKH:
        return f"{sign}₹{number / LAKH:.1f} L"
    if decimals is None:
        decimals = 0 if number >= 100 or number.is_integer() else 2
    rounded = round(number, decimals)
    whole = int(rounded)
    text = group_indian(whole)
    if decimals > 0:
        fraction = f"{rounded - whole:.{decimals}f}"[1:]   # ".46"
        text += fraction
    return f"{sign}₹{text}"


def pct_decimals(value, decimals: int = 1) -> int:
    """Decimals needed so a small non-zero percentage never shows as 0.0%:
    2 decimals below 1%, 3 below 0.1% (0.04 -> '0.040%'), otherwise `decimals`."""
    size = abs(float(value))
    if size == 0 or size >= 1:
        return decimals
    return max(decimals, 3 if size < 0.1 else 2)


def format_pct(value, decimals: int = 1) -> str:
    """A percentage already on the 0-100 scale: 12.345 -> '12.3%', 0.456 -> '0.46%'."""
    if _is_missing(value):
        return NA
    return f"{float(value):.{pct_decimals(value, decimals)}f}%"


def format_ratio(value, decimals: int = 2) -> str:
    """A multiple such as ROAS: 3.4512 -> '3.45x'."""
    if _is_missing(value):
        return NA
    return f"{float(value):.{decimals}f}x"


def format_value(value, fmt: str, compact: bool = False) -> str:
    """Format by KPI format type: 'money', 'percent', 'ratio' or 'count'."""
    if fmt == "money":
        return format_inr(value, compact=compact)
    if fmt == "percent":
        return format_pct(value)
    if fmt == "ratio":
        return format_ratio(value)
    if fmt == "count":
        return format_count(value)
    raise ValueError(f"Unknown format type: {fmt}")


def format_change(pct_change, decimals: int = 1) -> str:
    """Relative change with a sign, for KPI card arrows: 12.34 -> '+12.3%'."""
    if _is_missing(pct_change):
        return NA
    return f"{float(pct_change):+.{pct_decimals(pct_change, decimals)}f}%"


def format_date(value) -> str:
    """A date as '5 Jan 2026'; missing -> N/A."""
    if value is None:
        return NA
    try:
        if value != value:  # NaT / NaN
            return NA
    except TypeError:
        pass
    if isinstance(value, dt.datetime):
        value = value.date()
    return f"{value.day} {value.strftime('%b %Y')}"
