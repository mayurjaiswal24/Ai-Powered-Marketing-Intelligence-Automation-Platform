"""Dashboard filters: which ones exist for a dataset, applying them, and describing them.

Filters only select rows. All metrics for the filtered view are then recomputed by the same
analytics functions (analytics/kpis.py and friends), never calculated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from utils.formatting import format_date, format_date_range

# (column, label) in display order. A filter is offered only if the column has values.
DIMENSION_FILTERS = [
    ("channel", "Channel"), ("platform", "Platform"), ("campaign_name", "Campaign"),
    ("region", "Region"), ("product_category", "Product / Course Category"),
    ("product", "Product / Course"), ("customer_segment", "Customer Segment"),
    ("customer_type", "Customer Type"),
]

DATE_PRESETS = ["All dates", "Last 30 days", "Last 90 days", "Custom range"]


@dataclass
class Filters:
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    dimensions: dict[str, list[str]] = field(default_factory=dict)   # empty list = all values
    data_start: pd.Timestamp | None = None
    data_end: pd.Timestamp | None = None

    @property
    def date_filtered(self) -> bool:
        return self.start is not None and (self.start != self.data_start or self.end != self.data_end)

    @property
    def active(self) -> bool:
        return self.date_filtered or any(self.dimensions.values())

    def key(self) -> tuple:
        """Hashable form, for caching."""
        return (self.start, self.end, tuple(sorted((k, tuple(v)) for k, v in self.dimensions.items())))


def available_filters(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(c, label) for c, label in DIMENSION_FILTERS if c in df and df[c].notna().any()]


def preset_range(preset: str, data_start: pd.Timestamp, data_end: pd.Timestamp):
    if preset == "Last 30 days":
        return max(data_start, data_end - pd.Timedelta(days=29)), data_end
    if preset == "Last 90 days":
        return max(data_start, data_end - pd.Timedelta(days=89)), data_end
    return data_start, data_end


def apply_dimension_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, values in filters.dimensions.items():
        if values and col in df:
            mask &= df[col].astype(str).isin(values)
    return df[mask]


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    out = apply_dimension_filters(df, filters)
    if filters.start is not None and "date" in out:
        out = out[(out["date"] >= filters.start) & (out["date"] <= filters.end)]
    return out


def describe(filters: Filters) -> str:
    parts = []
    if filters.date_filtered:
        parts.append(f"{format_date(filters.start)} to {format_date(filters.end)}")
    labels = dict(DIMENSION_FILTERS)
    for col, values in filters.dimensions.items():
        if values:
            shown = ", ".join(values[:3]) + (f" +{len(values) - 3} more" if len(values) > 3 else "")
            parts.append(f"{labels.get(col, col)}: {shown}")
    return "Filtered view: " + "; ".join(parts) if parts else "Showing all data"


def describe_chips(filters: Filters) -> list[str]:
    """The active filters as short chips for the page header ("1 Jul 2026 – 30 Sep 2026",
    "Channel: Paid Search, Video"); one "Showing all data" chip when nothing is filtered."""
    chips = []
    if filters.date_filtered:
        chips.append(format_date_range(filters.start, filters.end))
    labels = dict(DIMENSION_FILTERS)
    for col, values in filters.dimensions.items():
        if values:
            shown = ", ".join(values[:3]) + (f" +{len(values) - 3} more" if len(values) > 3 else "")
            chips.append(f"{labels.get(col, col)}: {shown}")
    return chips or ["Showing all data"]
