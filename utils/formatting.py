"""All number and date formatting for display (dashboard, PDF, Excel, text summaries).

Indian conventions: digits grouped as 12,34,567. More formatters (₹, L/Cr, %, ROAS) are
added in the KPI and dashboard phases.
"""

from __future__ import annotations

import datetime as dt
import math

NA = "N/A"


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
    """Whole-number counts (rows, clicks, leads) with Indian grouping; missing -> N/A."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA
    return group_indian(int(round(value)))


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
