"""Field mapping: decide which uploaded column is Spend, Leads, Date, and so on.

Deterministic on purpose (no AI): a synonym dictionary plus fuzzy matching (rapidfuzz),
checked against the column's actual data type. Three confidence levels:

    confirmed         header is a known synonym ("Amount spent (INR)" -> Spend), or the user chose it
    high_confidence   header is a close spelling of a synonym ("Impresions") AND the data type fits
    uncertain         anything weaker, or a known-ambiguous header like "Results": suggested only,
                      never used until the user confirms it

Other outcomes: "derived" (ratios, rates and averages such as CTR, ROAS or AOV, which the app
recalculates from totals), "not_used" (recognised but deliberately not used, e.g. product cost,
gross revenue when net revenue exists, marketing cost when a clear spend column exists),
"unmapped" (nothing matched), "ignored" (the user said to skip it).

Safety rules for any marketing file (config/fields.py):
  - derived columns (name or values look like a rate/ratio/average) are never mapped to counts
    or money;
  - total/product/operating costs are never Spend; "marketing cost" is only an uncertain
    fallback when there is no spend column;
  - a plain "Profit" column is never taken as gross profit automatically (it may already
    subtract marketing cost, so ROI would count spend twice);
  - net revenue is preferred over gross; Orders stand in for Conversions when there are none.
Every mapping carries a short plain-English reason.

Alternative considered: ask Gemini to map columns. Rejected (locked decision): mappings drive
every number in the product, so they must be repeatable and explainable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

from config.fields import (AMBIGUOUS_HEADERS, DERIVED_METRIC_HEADERS, DERIVED_WORDS, FIELD_BY_NAME,
                           FIELDS, MARKETING_COSTS, NOT_SPEND_COSTS, PROFIT_HEADERS, RATIO_FIELDS)
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
    reason: str = ""                  # why the app decided this, in plain English


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
            "reason": m.reason,
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
    # Synonyms first (their order is the preference order), then the canonical name.
    for _rank, _syn in enumerate(_spec.synonyms + (_spec.name,)):
        _SYNONYMS.setdefault(normalize_header(_syn), (_spec.name, _rank))

_AMBIGUOUS = {normalize_header(k): v for k, v in AMBIGUOUS_HEADERS.items()}
_DERIVED = {normalize_header(h) for h in DERIVED_METRIC_HEADERS}
_NOT_SPEND = {normalize_header(h) for h in NOT_SPEND_COSTS}
_MARKETING_COSTS = {normalize_header(h) for h in MARKETING_COSTS}
_PROFIT = {normalize_header(h) for h in PROFIT_HEADERS}
_COUNT_OR_MONEY = {f.name for f in FIELDS if f.ftype in ("number", "money")}


def _suggestable(field_name: str, column_type: str | None) -> bool:
    """May this field be offered as a suggestion? A number field may be suggested for a text
    column (numbers stored as text happen), but a text/category field never for a number column
    (that produced suggestions like Discount -> Country)."""
    if _type_fits(field_name, column_type):
        return True
    return FIELD_BY_NAME[field_name].ftype in ("number", "money") and column_type in ("text", "mixed")


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
    if ftype == "id":
        return column_type in ("text", "number", "mixed")    # IDs can be numeric, never money/dates
    return column_type in ("text", "mixed")                  # categories and labels are text


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

    `source` is a DatasetProfile (preferred: data types and value shapes are then checked), a
    DataFrame, or a list of column names. `overrides` = {column: field name, or None to ignore
    the column}; user choices always win and are marked confirmed.
    """
    columns, column_types, ratio_like = _columns_and_types(source, column_types)
    overrides = dict(overrides or {})
    for col, fname in overrides.items():
        if col not in columns:
            raise ValueError(f"Unknown column in overrides: {col!r}")
        if fname is not None and fname not in FIELD_BY_NAME:
            raise ValueError(f"Unknown field in overrides: {fname!r}")

    # 1. A first opinion for every column, independently.
    proposals = [_propose(col, column_types.get(col), ratio_like.get(col, False), overrides)
                 for col in columns]

    # 2. One field per column and one column per field. User choices first, then confirmed,
    #    then high-confidence; ties go to the higher score, then the preferred synonym.
    taken: dict[str, str] = {}
    conflicts: list[str] = []
    rank = lambda p: _SYNONYMS.get(p.normalized, ("", 99))[1]  # noqa: E731
    order = sorted(
        (p for p in proposals if p.status in ACTIVE_STATUSES),
        key=lambda p: (p.source != "user", _STATUS_RANK[p.status], -p.score, rank(p),
                       columns.index(p.column)))
    winners: dict[str, ColumnMapping] = {}
    for p in order:
        if p.field in taken:
            winner = winners[p.field]
            label = FIELD_BY_NAME[p.field].label
            if p.source == "synonym" and winner.source == "synonym" and rank(p) > rank(winner):
                # A clearly preferred name exists (e.g. Net revenue over Gross revenue, Spend
                # over Cost): this column is simply not used, no question needed.
                p.status, p.field, p.source = "not_used", None, "preference"
                p.reason = f"'{winner.column}' is used for {label} (preferred name)"
                p.note = p.reason
                continue
            conflicts.append(f"'{p.column}' and '{winner.column}' both look like {label}; "
                             f"using '{winner.column}'.")
            p.note = f"'{winner.column}' is already used for {label}"
            p.reason = "same meaning as another column; please choose"
            p.candidates = [(p.field, p.score)]
            p.status, p.source, p.field = "uncertain", "conflict", None
        else:
            taken[p.field] = p.column
            winners[p.field] = p

    # 3. Orders / purchases / transactions stand in for Conversions when there are none.
    if "conversions" not in taken and "orders" in taken:
        m = winners["orders"]
        m.field = "conversions"
        m.reason = "Orders used as Conversions (the file has no Conversions column)"
        taken["conversions"] = taken.pop("orders")

    # 4. "Marketing cost" is a Spend fallback only when there is no clear spend column.
    for p in proposals:
        if p.source == "marketing_cost":
            if "spend" in taken:
                p.status = "not_used"
                p.reason = (f"a clear spend column ('{taken['spend']}') exists; marketing cost can "
                            "include agency fees or tools")
                p.note = p.reason
            else:
                p.candidates = [("spend", 0.0)]

    # 5. Suggestions for uncertain columns must not point at fields already in use.
    for p in proposals:
        if p.status == "uncertain" and p.source not in ("conflict", "marketing_cost"):
            p.candidates = [(f, s) for f, s in p.candidates if f not in taken]
            if not p.candidates:
                p.status = "unmapped"
                p.note = p.note or "no unused field matches this column"
                p.reason = "no unused field matches this column"
    return MappingResult(proposals, conflicts)


def _columns_and_types(source, column_types):
    # Imported here to avoid a circular import (the profiler imports this module's types).
    from ingestion.profiler import DatasetProfile
    if isinstance(source, DatasetProfile):
        types = {name: col.inferred_type for name, col in source.columns.items()}
        ratio = {name: col.ratio_like for name, col in source.columns.items()}
        return list(source.column_names), {**types, **(column_types or {})}, ratio
    if isinstance(source, pd.DataFrame):
        return [str(c) for c in source.columns], dict(column_types or {}), {}
    return [str(c) for c in source], dict(column_types or {}), {}


def _is_derived_name(normalized: str) -> bool:
    words = set(normalized.split())
    return normalized in _DERIVED or bool(words & DERIVED_WORDS)


def _propose(column: str, column_type: str | None, ratio_like: bool, overrides: dict) -> ColumnMapping:
    normalized = normalize_header(column)

    if column in overrides:
        fname = overrides[column]
        if fname is None:
            return ColumnMapping(column, normalized, None, "ignored", 100, "user",
                                 note="skipped by user", reason="you chose not to use it")
        return ColumnMapping(column, normalized, fname, "confirmed", 100, "user",
                             note="chosen by user", reason="chosen by you")

    exact = _SYNONYMS.get(normalized)
    ratio_field = exact is not None and exact[0] in RATIO_FIELDS
    # Ratios, rates and averages (by name, or because the values look like percentages).
    if not ratio_field and _is_derived_name(normalized):
        return ColumnMapping(column, normalized, None, "derived", 100, "derived-name",
                             note="derived metric - recalculated by the app, not used",
                             reason="derived metric (a rate, ratio or average) - recalculated by "
                                    "the app, not used")

    if normalized in _NOT_SPEND:
        return ColumnMapping(column, normalized, None, "not_used", 100, "not-spend",
                             note="not ad spend - not used",
                             reason="product/total/operating cost, not advertising spend - not used")

    if normalized in _MARKETING_COSTS:
        return ColumnMapping(column, normalized, None, "uncertain", 0, "marketing_cost",
                             candidates=[("spend", 0.0)],
                             note="may include non-media costs; confirm if it is ad spend",
                             reason="marketing cost may include agency fees or tools; confirm "
                                    "whether it is ad spend")

    if normalized in _PROFIT:
        return ColumnMapping(column, normalized, None, "uncertain", 0, "profit",
                             candidates=[("gross_profit", 0.0)],
                             note="may already subtract marketing cost",
                             reason="this profit may already subtract marketing cost; using it as "
                                    "gross profit would count spend twice in ROI")

    if normalized in _AMBIGUOUS:
        options = [f for f in _AMBIGUOUS[normalized] if _type_fits(f, column_type)]
        return ColumnMapping(column, normalized, None, "uncertain", 0, "ambiguous",
                             candidates=[(f, 0.0) for f in options],
                             note="ambiguous name; please choose what it means",
                             reason="ambiguous name - it can mean several things")

    if exact is not None:
        fname = exact[0]
        mapping = ColumnMapping(column, normalized, fname, "confirmed", 100, "synonym",
                                reason="exact name match")
        if not _type_fits(fname, column_type):
            mapping.note = f"values do not look like {FIELD_BY_NAME[fname].ftype} data"
        return mapping

    # Values that look like percentages/ratios are derived metrics, whatever the name says.
    if ratio_like:
        return ColumnMapping(column, normalized, None, "derived", 100, "derived-values",
                             note="values look like percentages or ratios - not used",
                             reason="values look like percentages or ratios - recalculated by the "
                                    "app, not used")

    scores = _fuzzy_scores(normalized) if normalized else {}
    ranked = sorted(((f, s) for f, s in scores.items()
                     if s >= MAPPING_MIN_CANDIDATE and _suggestable(f, column_type)),
                    key=lambda fs: -fs[1])[:3]
    if not ranked:
        return ColumnMapping(column, normalized, None, "unmapped", 0, "",
                             note="no known field matches this name",
                             reason="no known field matches this name")
    best_field, best_score = ranked[0]
    if best_score >= MAPPING_HIGH_CONFIDENCE and _type_fits(best_field, column_type):
        return ColumnMapping(column, normalized, best_field, "high_confidence", best_score,
                             "fuzzy", candidates=ranked,
                             reason=f"similar name ({best_score:.0f}% match) and matching values")
    return ColumnMapping(column, normalized, None, "uncertain", best_score, "fuzzy",
                         candidates=ranked, note="name is only partly similar",
                         reason=f"partly similar name ({best_score:.0f}% match)")
