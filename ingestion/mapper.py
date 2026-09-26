"""Field mapping: decide which uploaded column is Spend, Leads, Date, and so on.

Deterministic on purpose (no AI): a synonym dictionary plus fuzzy matching (rapidfuzz),
checked against the column's actual data type. Three confidence levels:

    confirmed         header is a known synonym ("Amount spent (INR)" -> Spend), or the user chose it
    high_confidence   header is a close spelling of a synonym ("Impresions") AND the data type fits
    uncertain         anything weaker, or a known-ambiguous header like "Results": suggested only,
                      never used until the user confirms it

Other outcomes: "derived" (ratio columns such as CTR or ROAS, which we recompute from totals),
"unmapped" (nothing matched), "ignored" (the user said to skip it).

Alternative considered: ask Gemini to map columns. Rejected (locked decision): mappings drive
every number in the product, so they must be repeatable and explainable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

from config.fields import (AMBIGUOUS_HEADERS, DERIVED_METRIC_HEADERS, FIELD_BY_NAME, FIELDS)
from config.settings import MAPPING_HIGH_CONFIDENCE, MAPPING_MIN_CANDIDATE

ACTIVE_STATUSES = ("confirmed", "high_confidence")
_STATUS_RANK = {"confirmed": 0, "high_confidence": 1}


@dataclass
class ColumnMapping:
    column: str
    normalized: str
    field: str | None                 # the canonical field in use (None unless confirmed/high)
    status: str                       # confirmed / high_confidence / uncertain / derived / unmapped / ignored
    score: float = 0.0                # 0-100 match score behind the decision
    source: str = ""                  # synonym / fuzzy / user / ambiguous / conflict
    candidates: list[tuple[str, float]] = field(default_factory=list)  # suggestions (field, score)
    note: str = ""


@dataclass
class MappingResult:
    columns: list[ColumnMapping]
    conflicts: list[str] = field(default_factory=list)

    @property
    def active(self) -> dict[str, str]:
        """Canonical field -> column name, for mappings that analysis may use."""
        return {m.field: m.column for m in self.columns if m.status in ACTIVE_STATUSES and m.field}

    def by_column(self, column: str) -> ColumnMapping:
        return next(m for m in self.columns if m.column == column)

    @property
    def uncertain(self) -> list[ColumnMapping]:
        return [m for m in self.columns if m.status == "uncertain"]

    @property
    def needs_confirmation(self) -> list[ColumnMapping]:
        """Uncertain columns that might be a business-critical field (Date, Spend, Revenue,
        Leads, Conversions). Analysis must wait until the user decides."""
        return [m for m in self.uncertain
                if any(FIELD_BY_NAME[f].critical for f, _ in m.candidates)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "column": m.column,
            "maps_to": FIELD_BY_NAME[m.field].label if m.field else "",
            "status": m.status,
            "score": round(m.score),
            "suggestions": ", ".join(FIELD_BY_NAME[f].label + (f" ({s:.0f})" if s else "")
                                     for f, s in m.candidates),
            "note": m.note,
        } for m in self.columns])


# ---------------------------------------------------------------------------------------------
# Header normalization and synonym lookup
# ---------------------------------------------------------------------------------------------

_UNIT_WORDS = r"\b(inr|rs|usd|eur|gbp|in rupees|rupees)\b"


def normalize_header(header: str) -> str:
    """'Amount Spent (INR)' -> 'amount spent'; 'Conv. value' -> 'conv value';
    'campaign_name' -> 'campaign name'."""
    text = str(header).lower().strip()
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)   # drop bracketed units/notes
    text = text.replace("₹", " ").replace("%", " percent ")
    text = re.sub(r"[_\-./\\:#,;|&+]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(_UNIT_WORDS, " ", text)
    return re.sub(r"\s+", " ", text).strip()


# normalized synonym -> (field, preference rank)
_SYNONYMS: dict[str, tuple[str, int]] = {}
for _spec in FIELDS:
    for _rank, _syn in enumerate((_spec.name,) + _spec.synonyms):
        _SYNONYMS.setdefault(normalize_header(_syn), (_spec.name, _rank))

_AMBIGUOUS = {normalize_header(k): v for k, v in AMBIGUOUS_HEADERS.items()}
_DERIVED = {normalize_header(h) for h in DERIVED_METRIC_HEADERS}


def _type_fits(field_name: str, column_type: str | None) -> bool:
    """Does the column's data look like what this field needs?"""
    if column_type is None:
        return True           # no data profile available: judge on the header alone
    ftype = FIELD_BY_NAME[field_name].ftype
    if column_type == "empty":
        return False
    if ftype == "date":
        return column_type == "date"
    if ftype in ("number", "money"):
        return column_type in ("number", "money")
    return True               # categories, ids and text can hold anything


def _fuzzy_scores(normalized: str) -> dict[str, float]:
    """Best fuzzy score per field across all its synonyms."""
    best: dict[str, float] = {}
    for syn, (fname, _) in _SYNONYMS.items():
        score = fuzz.token_sort_ratio(normalized, syn)
        if score > best.get(fname, 0):
            best[fname] = score
    return best


# ---------------------------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------------------------

def map_fields(source, column_types: dict[str, str] | None = None,
               overrides: dict[str, str | None] | None = None) -> MappingResult:
    """Map columns to canonical fields.

    `source` is a DatasetProfile (preferred: data types are then checked), a DataFrame, or a
    list of column names. `overrides` = {column: field name, or None to ignore the column};
    user choices always win and are marked confirmed.
    """
    columns, column_types = _columns_and_types(source, column_types)
    overrides = dict(overrides or {})
    for col, fname in overrides.items():
        if col not in columns:
            raise ValueError(f"Unknown column in overrides: {col!r}")
        if fname is not None and fname not in FIELD_BY_NAME:
            raise ValueError(f"Unknown field in overrides: {fname!r}")

    # 1. A first opinion for every column, independently.
    proposals = [_propose(col, column_types.get(col), overrides) for col in columns]

    # 2. One field per column and one column per field. User choices first, then confirmed,
    #    then high-confidence; ties go to the higher score, then the preferred synonym.
    taken: dict[str, str] = {}
    conflicts: list[str] = []
    order = sorted(
        (p for p in proposals if p.status in ACTIVE_STATUSES),
        key=lambda p: (p.source != "user", _STATUS_RANK[p.status], -p.score,
                       _SYNONYMS.get(p.normalized, ("", 99))[1], columns.index(p.column)))
    for p in order:
        if p.field in taken:
            winner = taken[p.field]
            label = FIELD_BY_NAME[p.field].label
            conflicts.append(f"'{p.column}' and '{winner}' both look like {label}; "
                             f"using '{winner}'.")
            p.note = f"'{winner}' is already used for {label}"
            p.candidates = [(p.field, p.score)]
            p.status, p.source, p.field = "uncertain", "conflict", None
        else:
            taken[p.field] = p.column

    # 3. Suggestions for uncertain columns must not point at fields already in use.
    for p in proposals:
        if p.status == "uncertain" and p.source != "conflict":
            p.candidates = [(f, s) for f, s in p.candidates if f not in taken]
            if not p.candidates:
                p.status = "unmapped"
                p.note = p.note or "no unused field matches this column"
    return MappingResult(proposals, conflicts)


def _columns_and_types(source, column_types):
    # Imported here to avoid a circular import (the profiler imports this module's types).
    from ingestion.profiler import DatasetProfile
    if isinstance(source, DatasetProfile):
        types = {name: col.inferred_type for name, col in source.columns.items()}
        return list(source.column_names), {**types, **(column_types or {})}
    if isinstance(source, pd.DataFrame):
        return [str(c) for c in source.columns], dict(column_types or {})
    return [str(c) for c in source], dict(column_types or {})


def _propose(column: str, column_type: str | None, overrides: dict) -> ColumnMapping:
    normalized = normalize_header(column)

    if column in overrides:
        fname = overrides[column]
        if fname is None:
            return ColumnMapping(column, normalized, None, "ignored", 100, "user",
                                 note="skipped by user")
        return ColumnMapping(column, normalized, fname, "confirmed", 100, "user",
                             note="chosen by user")

    if normalized in _DERIVED:
        return ColumnMapping(column, normalized, None, "derived", 100, "synonym",
                             note="ratio metric; recalculated from totals instead")

    if normalized in _AMBIGUOUS:
        options = [f for f in _AMBIGUOUS[normalized] if _type_fits(f, column_type)]
        return ColumnMapping(column, normalized, None, "uncertain", 0, "ambiguous",
                             candidates=[(f, 0.0) for f in options],
                             note="ambiguous name; please choose what it means")

    if normalized in _SYNONYMS:
        fname = _SYNONYMS[normalized][0]
        mapping = ColumnMapping(column, normalized, fname, "confirmed", 100, "synonym")
        if not _type_fits(fname, column_type):
            mapping.note = f"values do not look like {FIELD_BY_NAME[fname].ftype} data"
        return mapping

    scores = _fuzzy_scores(normalized) if normalized else {}
    ranked = sorted(((f, s) for f, s in scores.items() if s >= MAPPING_MIN_CANDIDATE),
                    key=lambda fs: -fs[1])[:3]
    if not ranked:
        return ColumnMapping(column, normalized, None, "unmapped", 0, "",
                             note="no known field matches this name")
    best_field, best_score = ranked[0]
    if best_score >= MAPPING_HIGH_CONFIDENCE and _type_fits(best_field, column_type):
        return ColumnMapping(column, normalized, best_field, "high_confidence", best_score,
                             "fuzzy", candidates=ranked)
    note = ("name is similar but the values have a different type"
            if best_score >= MAPPING_HIGH_CONFIDENCE else "name is only partly similar")
    return ColumnMapping(column, normalized, None, "uncertain", best_score, "fuzzy",
                         candidates=ranked, note=note)
