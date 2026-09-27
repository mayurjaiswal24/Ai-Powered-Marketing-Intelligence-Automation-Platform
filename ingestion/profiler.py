"""Data profiling: describe an unknown dataset before anything is changed.

The profile answers "what is in this file and how clean is it?": types, missing values,
duplicates, ranges, inconsistent labels, summary rows and (once fields are mapped) impossible
values. It never modifies the data; cleaning happens in Phase 04.

Usage:
    profile = profile_dataset(raw_df)          # structure + quality, no field knowledge
    mapping = map_fields(profile)              # ingestion/mapper.py
    profile.apply_mapping(raw_df, mapping)     # adds checks that need to know which column is what
    print(profile.to_text())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from rapidfuzz import fuzz

from config.fields import FIELD_BY_NAME, LABEL_ALIASES
from config.settings import (LABEL_SIMILARITY, MAX_LABELS_FOR_CLUSTERING, MOSTLY_EMPTY_SHARE,
                             TYPE_DETECTION_SHARE)
from utils.formatting import format_count, format_date
from utils.parsing import missing_mask, parse_dates, parse_numbers

if TYPE_CHECKING:  # avoid a circular import at runtime
    from ingestion.mapper import MappingResult

SUMMARY_ROW_PATTERN = re.compile(r"^\s*(grand\s*total|sub\s*-?\s*total|total)\b", re.IGNORECASE)


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str          # date / number / money / mixed / text / empty
    non_missing: int
    missing: int
    missing_breakdown: dict[str, int]   # e.g. {"(blank)": 40, "N/A": 40}
    unique: int
    number_share: float         # share of non-missing values that parse as numbers
    date_share: float           # share of non-missing values that parse as dates
    formatted_number_count: int  # numbers written as text with ₹ / INR / separators
    min_value: float | None = None
    max_value: float | None = None
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    date_formats: dict[str, int] = field(default_factory=dict)
    top_values: dict[str, int] = field(default_factory=dict)
    # True when the values look like percentages or ratios ("4.5%", or mostly decimals 0-100):
    # such a column is a derived metric, never a count or money field.
    ratio_like: bool = False


@dataclass
class LabelCluster:
    column: str
    suggested_label: str
    variants: dict[str, int]      # every raw spelling in the cluster -> row count

    @property
    def inconsistent_variants(self) -> dict[str, int]:
        return {k: v for k, v in self.variants.items() if k != self.suggested_label}


@dataclass
class QualityFinding:
    kind: str                    # machine-readable type, e.g. "duplicate_rows"
    message: str                 # plain-English sentence for the UI
    count: int
    column: str | None = None
    rows: list[int] = field(default_factory=list)   # 0-based row positions in the raw table
    details: dict = field(default_factory=dict)


@dataclass
class DatasetProfile:
    rows: int
    column_names: list[str]
    columns: dict[str, ColumnProfile]
    duplicate_rows: list[int]              # positions of the extra copies
    summary_rows: list[int]
    label_clusters: list[LabelCluster]
    findings: list[QualityFinding]
    date_column: str | None = None
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    mapping: "MappingResult | None" = None

    def finding(self, kind: str, column: str | None = None) -> list[QualityFinding]:
        return [f for f in self.findings if f.kind == kind and (column is None or f.column == column)]

    # -- checks that need the field mapping ------------------------------------------------------
    def apply_mapping(self, df: pd.DataFrame, mapping: "MappingResult") -> None:
        """Add findings that require knowing which column is Spend, Clicks, etc."""
        self.mapping = mapping
        self.findings = [f for f in self.findings if not f.details.get("needs_mapping")]
        active = mapping.active  # canonical field -> column name

        numbers: dict[str, pd.Series] = {}
        for fname, column in active.items():
            spec = FIELD_BY_NAME[fname]
            if spec.ftype in ("number", "money"):
                numbers[fname] = parse_numbers(df[column])[0]

        # Money and counts can never be negative.
        for fname, values in numbers.items():
            negative = values < 0
            if negative.any():
                label = FIELD_BY_NAME[fname].label
                self.findings.append(QualityFinding(
                    "negative_values", f"{format_count(negative.sum())} negative values in "
                    f"{label} (column '{active[fname]}'). {label} cannot be negative.",
                    int(negative.sum()), active[fname], _positions(negative),
                    {"field": fname, "needs_mapping": True}))

        # Funnel stages that cannot exceed the stage above them.
        pairs = [("clicks", "impressions", "clicks are higher than impressions"),
                 ("reach", "impressions", "reach is higher than impressions"),
                 ("qualified_leads", "leads", "qualified leads are higher than leads")]
        for lower, upper, text in pairs:
            if lower in numbers and upper in numbers:
                bad = numbers[lower] > numbers[upper]
                if bad.any():
                    self.findings.append(QualityFinding(
                        f"{lower}_exceed_{upper}", f"{format_count(bad.sum())} rows where {text}, "
                        "which is not possible.", int(bad.sum()), active[lower], _positions(bad),
                        {"field": lower, "compared_with": upper, "needs_mapping": True}))

        # Prefer the mapped Date column for the dataset's date range.
        if "date" in active and active["date"] in self.columns:
            col = self.columns[active["date"]]
            self.date_column, self.date_min, self.date_max = col.name, col.date_min, col.date_max

    # -- text summary ----------------------------------------------------------------------------
    def to_text(self) -> str:
        lines = ["DATASET PROFILE", "",
                 f"Rows: {format_count(self.rows)}",
                 f"Columns: {format_count(len(self.column_names))}"]
        if self.date_min is not None:
            lines.append(f"Date range: {format_date(self.date_min)} - {format_date(self.date_max)}"
                         f" (column '{self.date_column}')")
        else:
            lines.append("Date range: no date column recognised")

        if self.mapping is not None:
            lines += ["", "Recognised fields:"]
            marker = {"confirmed": "[OK]   ", "high_confidence": "[OK~]  "}
            for m in self.mapping.columns:
                if m.status in marker:
                    spec = FIELD_BY_NAME[m.field]
                    how = "" if m.status == "confirmed" else "  (inferred, please check)"
                    lines.append(f"  {marker[m.status]}{spec.label:<26} <- '{m.column}'{how}")
            uncertain = [m for m in self.mapping.columns if m.status == "uncertain"]
            if uncertain:
                lines += ["", "Needs your confirmation:"]
                for m in uncertain:
                    options = " or ".join(FIELD_BY_NAME[c].label for c, _ in m.candidates) or "no suggestion"
                    lines.append(f"  [?]    '{m.column}' could be: {options}")
            other = [m for m in self.mapping.columns if m.status in ("unmapped", "derived", "not_used", "ignored")]
            if other:
                lines += ["", "Not used in analysis:"]
                for m in other:
                    lines.append(f"  [-]    '{m.column}' ({m.reason or m.note or m.status})")

        lines += ["", "Data-quality findings:"]
        if not self.findings:
            lines.append("  No problems found.")
        for f in self.findings:
            lines.append(f"  [!]    {f.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Building the profile
# ---------------------------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Profile a raw DataFrame (all cells text, as returned by ingestion.loader)."""
    # Parse each column once and reuse the results for every check below.
    columns: dict[str, ColumnProfile] = {}
    parsed: dict[str, _Parsed] = {}
    for name in df.columns:
        columns[name], parsed[name] = _profile_column(df[name])
    findings: list[QualityFinding] = []

    # Exact duplicates: rows identical in every column (the first copy is kept as original).
    dup_mask = df.duplicated(keep="first")
    duplicate_rows = _positions(dup_mask)
    if duplicate_rows:
        findings.append(QualityFinding(
            "duplicate_rows", f"{format_count(len(duplicate_rows))} duplicate rows (exact copies "
            "of another row).", len(duplicate_rows), rows=duplicate_rows))

    # Summary rows like "Grand Total" that would double-count every metric.
    summary_rows = _find_summary_rows(df, columns)
    if summary_rows:
        labels = sorted({_summary_label(df.iloc[r]) for r in summary_rows})
        findings.append(QualityFinding(
            "summary_rows", f"{len(summary_rows)} summary row(s) ({', '.join(labels)}) that would "
            "double-count totals if kept.", len(summary_rows), rows=summary_rows))

    # Summary rows are reported once above; their blank cells are not "missing values".
    not_summary = pd.Series(True, index=df.index)
    not_summary.iloc[summary_rows] = False

    for name, col in columns.items():
        if col.inferred_type == "empty" or (col.missing / max(len(df), 1)) >= MOSTLY_EMPTY_SHARE:
            findings.append(QualityFinding(
                "mostly_empty_column", f"Column '{name}' is almost entirely empty "
                f"({format_count(col.non_missing)} filled cells); it is probably not useful.",
                1, name))
            continue
        missing = missing_mask(df[name]) & not_summary
        if missing.any():
            raw = df[name][missing].fillna("").astype(str).str.strip()
            breakdown = {("(blank)" if k == "" else k): int(v) for k, v in raw.value_counts().items()}
            parts = ", ".join(f"{format_count(v)} {k}" for k, v in breakdown.items())
            findings.append(QualityFinding(
                "missing_values", f"{format_count(missing.sum())} missing values in '{name}' ({parts}).",
                int(missing.sum()), name, _positions(missing), {"breakdown": breakdown}))
        if col.inferred_type == "date" and len(col.date_formats) > 1:
            formats = ", ".join(f"{k} ({format_count(v)})" for k, v in col.date_formats.items())
            labels = parsed[name].date_labels
            main = max(col.date_formats, key=col.date_formats.get)
            odd = labels.notna() & (labels != main)
            findings.append(QualityFinding(
                "mixed_date_formats", f"Mixed date formats in '{name}': {formats}. Dates like "
                "05/01/2026 are read day-first (5 Jan 2026).", int(odd.sum()), name,
                _positions(odd), {"formats": col.date_formats, "main_format": main}))
        if col.inferred_type in ("number", "money") and col.formatted_number_count:
            formatted = parsed[name].formatted
            findings.append(QualityFinding(
                "money_as_text", f"{format_count(col.formatted_number_count)} values in '{name}' are "
                "numbers written as text (with ₹, INR or commas); they can be converted.",
                col.formatted_number_count, name, _positions(formatted)))
        if col.inferred_type in ("number", "money", "date"):
            readable = (parsed[name].dates.notna() if col.inferred_type == "date"
                        else parsed[name].numbers.notna())
            bad = ~readable & ~missing_mask(df[name])
            bad.iloc[summary_rows] = False
            if bad.any():
                findings.append(QualityFinding(
                    "unreadable_values", f"{format_count(bad.sum())} values in '{name}' could not be "
                    f"read as {'dates' if col.inferred_type == 'date' else 'numbers'}.",
                    int(bad.sum()), name, _positions(bad)))
        if col.inferred_type == "mixed":
            findings.append(QualityFinding(
                "mixed_types", f"Column '{name}' mixes numbers and text "
                f"({col.number_share:.0%} numeric).", 1, name))

    clusters: list[LabelCluster] = []
    for name, col in columns.items():
        if col.inferred_type == "text" and 1 < col.unique <= MAX_LABELS_FOR_CLUSTERING:
            for cluster in cluster_labels(df[name], name):
                clusters.append(cluster)
                variants = ", ".join(f"'{v}'" for v in cluster.inconsistent_variants)
                affected = sum(cluster.inconsistent_variants.values())
                mask = df[name].isin(list(cluster.inconsistent_variants))
                verb = "looks" if len(cluster.inconsistent_variants) == 1 else "look"
                findings.append(QualityFinding(
                    "inconsistent_labels", f"Inconsistent labels in '{name}': {variants} {verb} like "
                    f"'{cluster.suggested_label}' ({format_count(affected)} rows).", affected, name,
                    _positions(mask), {"suggested": cluster.suggested_label,
                                       "variants": cluster.variants}))

    # Dataset date range: first date-typed column until a mapping says otherwise.
    date_col = next((c for c in columns.values() if c.inferred_type == "date"), None)
    return DatasetProfile(
        rows=len(df), column_names=list(df.columns), columns=columns,
        duplicate_rows=duplicate_rows, summary_rows=summary_rows, label_clusters=clusters,
        findings=findings, date_column=date_col.name if date_col else None,
        date_min=date_col.date_min if date_col else None,
        date_max=date_col.date_max if date_col else None)


@dataclass
class _Parsed:
    numbers: pd.Series
    formatted: pd.Series
    dates: pd.Series
    date_labels: pd.Series


def _profile_column(series: pd.Series) -> tuple[ColumnProfile, _Parsed]:
    missing = missing_mask(series)
    present = series[~missing]
    n = len(present)

    raw_missing = series[missing].fillna("").astype(str).str.strip()
    breakdown = {("(blank)" if k == "" else k): int(v)
                 for k, v in raw_missing.value_counts().items()}

    numbers, formatted = parse_numbers(series)
    # Trying every date format on every column is slow; only do it if a sample looks like dates.
    if n and parse_dates(present.head(200))[0].notna().any():
        dates, date_labels = parse_dates(series)
    else:
        dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        date_labels = pd.Series(None, index=series.index, dtype=object)
    number_share = numbers.notna().sum() / n if n else 0.0
    date_share = dates.notna().sum() / n if n else 0.0

    if n == 0:
        kind = "empty"
    elif date_share >= TYPE_DETECTION_SHARE:
        kind = "date"
    elif number_share >= TYPE_DETECTION_SHARE:
        kind = "money" if formatted.any() else "number"
    elif number_share >= 0.5:
        kind = "mixed"
    else:
        kind = "text"

    profile = ColumnProfile(
        name=str(series.name), inferred_type=kind, non_missing=n, missing=int(missing.sum()),
        missing_breakdown=breakdown, unique=int(present.nunique()),
        number_share=float(number_share), date_share=float(date_share),
        formatted_number_count=int(formatted.sum()) if kind in ("number", "money") else 0)
    if kind in ("number", "money", "mixed") and numbers.notna().any():
        profile.min_value, profile.max_value = float(numbers.min()), float(numbers.max())
    if kind == "date":
        profile.date_min, profile.date_max = dates.min(), dates.max()
        profile.date_formats = {k: int(v) for k, v in date_labels.value_counts().items()}
    if kind == "text":
        profile.top_values = {str(k): int(v) for k, v in present.value_counts().head(5).items()}
    profile.ratio_like = _looks_like_ratio(present, numbers)
    return profile, _Parsed(numbers, formatted, dates, date_labels)


def _looks_like_ratio(present: pd.Series, numbers: pd.Series) -> bool:
    if present.empty:
        return False
    if present.astype(str).str.strip().str.endswith("%").mean() >= 0.5:
        return True
    values = numbers.dropna()
    if len(values) < max(3, 0.9 * len(present)):
        return False
    in_range = ((values >= 0) & (values <= 100)).mean()
    decimals = (values != values.round()).mean()
    return bool(in_range >= 0.95 and decimals >= 0.5)


# ---------------------------------------------------------------------------------------------
# Inconsistent category labels
# ---------------------------------------------------------------------------------------------

def _label_key(label: str) -> str:
    key = re.sub(r"\s+", " ", str(label).strip().lower())
    return LABEL_ALIASES.get(key, key)


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def cluster_labels(series: pd.Series, column: str = "") -> list[LabelCluster]:
    """Group spellings of the same label ("FB", "facebook", "Meta Ads", "meta ") and suggest
    one. Step 1: same text after trimming/lower-casing or a known alias. Step 2: very similar
    spellings (typos) via rapidfuzz, but never across different numbers ("Tier 1" vs "Tier 2",
    "KL-GS-001" vs "KL-GS-002")."""
    counts = series[~missing_mask(series)].astype(str).value_counts()
    groups: dict[str, dict[str, int]] = {}
    for label, count in counts.items():
        groups.setdefault(_label_key(label), {})[label] = int(count)

    # Merge near-identical keys into the bigger group.
    ordered = sorted(groups, key=lambda k: -sum(groups[k].values()))
    merged: dict[str, dict[str, int]] = {}
    for key in ordered:
        target = None
        for existing in merged:
            if (len(key) >= 4 and _digits(key) == _digits(existing)
                    and fuzz.ratio(key, existing) >= LABEL_SIMILARITY):
                target = existing
                break
        if target is None:
            merged[key] = dict(groups[key])
        else:
            merged[target].update(groups[key])

    clusters = []
    for variants in merged.values():
        if len(variants) < 2:
            continue
        # Suggest the most common spelling that has no stray spaces.
        tidy = {k: v for k, v in variants.items() if k == k.strip() and "  " not in k}
        pool = tidy or variants
        suggested = max(pool, key=lambda k: (pool[k], k))
        clusters.append(LabelCluster(column, suggested.strip(), variants))
    return clusters


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def _positions(mask: pd.Series) -> list[int]:
    return [int(i) for i in mask.to_numpy().nonzero()[0]]


def _find_summary_rows(df: pd.DataFrame, columns: dict[str, ColumnProfile]) -> list[int]:
    text_cols = [c for c, p in columns.items() if p.inferred_type in ("text", "mixed")]
    hits = pd.Series(False, index=df.index)
    for c in text_cols:
        hits |= df[c].fillna("").astype(str).str.match(SUMMARY_ROW_PATTERN)
    return _positions(hits)


def _summary_label(row: pd.Series) -> str:
    for value in row:
        if isinstance(value, str) and SUMMARY_ROW_PATTERN.match(value):
            return value.strip()
    return "Total"
