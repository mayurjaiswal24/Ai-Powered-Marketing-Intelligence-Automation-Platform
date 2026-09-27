"""Cross-check the mapping against the file's OWN calculated columns.

Many exports already contain CPC, CTR, CPL, ROAS, conversion rate or AOV columns. The app never
uses them (it recalculates every ratio from totals), but they are a free test of the mapping:
if the file's CPC equals Spend / Clicks on almost every row, the Spend and Clicks mappings are
very likely right. If it does not, something is probably mapped to the wrong column.

For each such column we try the common definitions (e.g. conversion rate = orders / clicks, or
orders / leads, as a percentage or a fraction) on a sample of rows and report which one matched.
Only numbers are compared; nothing here changes the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import (CROSSCHECK_MIN_MATCH_SHARE, CROSSCHECK_MIN_ROWS, CROSSCHECK_REL_TOL,
                             CROSSCHECK_ABS_TOL, CROSSCHECK_SAMPLE_ROWS)
from ingestion.mapper import MappingResult, normalize_header
from utils.parsing import parse_numbers

# Which of the file's own columns we recognise, and how each metric can be defined.
# A definition is (numerator role, denominator role, scales to try). Roles are resolved to the
# mapped column first, then to other numeric columns whose names fit the role.
_METRICS: dict[str, dict] = {
    "CPC":  {"names": {"cpc", "cost per click", "avg cpc", "average cpc"},
             "defs": [("spend", "clicks", (1,))]},
    "CTR":  {"names": {"ctr", "ctr percent", "click through rate", "clickthrough rate", "ctr all"},
             "defs": [("clicks", "impressions", (100, 1))]},
    "CPL":  {"names": {"cpl", "cost per lead"},
             "defs": [("spend", "leads", (1,))]},
    "CPM":  {"names": {"cpm", "cost per mille", "cost per 1000 impressions"},
             "defs": [("spend", "impressions", (1000,))]},
    "CPA":  {"names": {"cpa", "cac", "cost per acquisition", "cost per conversion",
                       "cost per purchase", "cost per order", "cost conv"},
             "defs": [("spend", "conversions", (1,)), ("spend", "orders", (1,)),
                      ("spend", "customers", (1,))]},
    # Meta's "Cost per result": a result is a lead or a conversion, depending on the campaign.
    "Cost per result": {"names": {"cost per result", "cost per results"},
                        "defs": [("spend", "leads", (1,)), ("spend", "conversions", (1,))]},
    "ROAS": {"names": {"roas", "roas x", "return on ad spend", "purchase roas"},
             "defs": [("revenue", "spend", (1,))]},
    "ROI":  {"names": {"roi", "roi percent", "return on investment"},
             "defs": [("profit", "spend", (100, 1)), ("net_return", "spend", (100, 1))]},
    "Conversion rate": {"names": {"conversion rate", "conversion rate percent", "conv rate", "cvr",
                                  "cvr percent", "conversion percent", "lead to conversion rate",
                                  "click to conversion rate", "lead to deal percent",
                                  "lead to deal rate", "lead to conversion percent",
                                  "lead to sale rate", "win rate"},
                        "defs": [("conversions", "clicks", (100, 1)), ("conversions", "leads", (100, 1)),
                                 ("orders", "clicks", (100, 1)), ("orders", "leads", (100, 1)),
                                 ("conversions", "impressions", (100, 1))]},
    "AOV":  {"names": {"aov", "average order value", "avg order value", "average basket value"},
             "defs": [("revenue", "orders", (1,)), ("revenue", "conversions", (1,))]},
}

# Canonical field behind each role (for "verified" marks) and header words for alternatives.
_ROLE_FIELD = {"spend": "spend", "clicks": "clicks", "impressions": "impressions", "leads": "leads",
               "conversions": "conversions", "orders": "orders", "customers": "customers",
               "revenue": "revenue", "profit": "gross_profit"}
_ROLE_WORDS = {"spend": ("spend", "spent", "cost"), "revenue": ("revenue", "sales", "gmv", "value"),
               "profit": ("profit", "margin amount", "contribution"),
               "orders": ("orders", "purchases", "transactions"), "customers": ("customers",),
               "conversions": ("conversions", "results"), "leads": ("leads",),
               "clicks": ("clicks",), "impressions": ("impressions",)}


@dataclass
class CrossCheck:
    column: str                 # the file's own calculated column, e.g. "CPC"
    metric: str                 # "CPC"
    status: str                 # verified | alternative | mismatch | unchecked
    message: str
    definition: str = ""        # e.g. "Spend ÷ Clicks × 100"
    fields: tuple[str, ...] = ()   # canonical fields confirmed by a verified match
    contradicted: tuple[str, ...] = ()   # fields of the mapped definition a mismatch disproves


@dataclass
class CrossCheckResult:
    checks: list[CrossCheck] = field(default_factory=list)

    @property
    def verified_fields(self) -> set[str]:
        return {f for c in self.checks if c.status == "verified" for f in c.fields}

    @property
    def mismatches(self) -> list[CrossCheck]:
        return [c for c in self.checks if c.status == "mismatch"]


def _numbers(series: pd.Series) -> pd.Series:
    """Numbers from a column, accepting '12.5%' and formatted text like '₹1,200'."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.replace("%", "", regex=False).str.replace("x", "", regex=False)
    return parse_numbers(text)[0]


def _role_columns(role: str, df: pd.DataFrame, mapping: MappingResult) -> list[tuple[str, bool]]:
    """Columns that could play `role`: (column, is_the_mapped_one). Mapped column first."""
    out: list[tuple[str, bool]] = []
    active = mapping.active
    fname = _ROLE_FIELD.get(role)
    if fname and fname in active:
        out.append((active[fname], True))
    for col in df.columns:
        if any(col == c for c, _ in out):
            continue
        norm = normalize_header(col)
        words = set(norm.split())
        if {"rate", "percent", "per", "ratio", "average", "avg", "roas", "roi", "cpc", "ctr"} & words:
            continue                            # a derived column is never an input
        if any(w in norm for w in _ROLE_WORDS.get(role, ())):
            out.append((col, False))
    return out


def _share_matching(own: pd.Series, calc: pd.Series) -> tuple[float, int]:
    ok = own.notna() & calc.notna() & (calc.abs() != float("inf"))
    if ok.sum() == 0:
        return 0.0, 0
    diff = (own[ok] - calc[ok]).abs()
    tol = (own[ok].abs() * CROSSCHECK_REL_TOL).clip(lower=CROSSCHECK_ABS_TOL)
    return float((diff <= tol).mean()), int(ok.sum())


def _describe(num: str, den: str, scale: int) -> str:
    return f"{num} ÷ {den}" + (f" × {scale}" if scale != 1 else "")


def cross_check(df: pd.DataFrame, mapping: MappingResult) -> CrossCheckResult:
    """Compare every recognised own-calculation column with the mapped fields."""
    result = CrossCheckResult()
    sample = df.head(CROSSCHECK_SAMPLE_ROWS)
    cache: dict[str, pd.Series] = {}

    def nums(col: str) -> pd.Series:
        if col not in cache:
            cache[col] = _numbers(sample[col])
        return cache[col]

    for col in df.columns:
        norm = normalize_header(col)
        metric = next((m for m, spec in _METRICS.items() if norm in spec["names"]), None)
        if metric is None:
            continue
        own = nums(col)
        if own.notna().sum() < CROSSCHECK_MIN_ROWS:
            continue
        # Every (numerator, denominator) pair the definitions allow. Pairs made only of mapped
        # columns are tried first, so a match there always wins over an alternative column.
        trials = []
        for num_role, den_role, scales in _METRICS[metric]["defs"]:
            num_cols = (_net_return_columns(sample, mapping) if num_role == "net_return"
                        else [(c, is_mapped, nums(c)) for c, is_mapped in _role_columns(num_role, sample, mapping)])
            for num_name, num_mapped, num_values in num_cols:
                for den_col, den_mapped in _role_columns(den_role, sample, mapping):
                    if num_name != den_col:
                        trials.append((num_mapped and den_mapped, num_name, num_values, den_col, scales))
        trials.sort(key=lambda trial: not trial[0])            # stable: keeps definition order
        tried_mapped = [_describe(n, d, sc[0]) for mapped, n, _v, d, sc in trials if mapped]
        first_mapped = next(((n, d) for mapped, n, _v, d, _sc in trials if mapped), None)

        hit = None                           # (definition text, all mapped?, columns used)
        for mapped, num_name, num_values, den_col, scales in trials:
            den = nums(den_col)
            ratio = num_values / den.where(den != 0)
            for scale in scales:
                share, n = _share_matching(own, ratio * scale)
                if n >= CROSSCHECK_MIN_ROWS and share >= CROSSCHECK_MIN_MATCH_SHARE:
                    hit = (_describe(num_name, den_col, scale), mapped, (num_name, den_col))
                    break
            if hit:
                break

        if hit and hit[1]:
            fields = _fields_of(hit[2], mapping)
            result.checks.append(CrossCheck(col, metric, "verified",
                                            f"Your file's {col} matches {hit[0]} - these mappings "
                                            "are verified by the file's own calculations.",
                                            hit[0], fields))
        elif hit:
            used = f" (the mapped fields give {tried_mapped[0]})" if tried_mapped else ""
            result.checks.append(CrossCheck(col, metric, "alternative",
                                            f"Your file's {col} is calculated as {hit[0]}{used}. "
                                            "This is fine if intended; otherwise check the mapping.", hit[0]))
        elif tried_mapped:
            result.checks.append(CrossCheck(col, metric, "mismatch",
                                            f"Your file's {col} doesn't match {tried_mapped[0]} - "
                                            "please check these mappings.", tried_mapped[0],
                                            contradicted=_fields_of(first_mapped, mapping)))
        else:
            result.checks.append(CrossCheck(col, metric, "unchecked",
                                            f"Your file's {col} could not be checked (the columns "
                                            "it needs are not mapped)."))
    return result


_NET_RETURN = "(Revenue − Spend)"


def _fields_of(columns: tuple[str, str], mapping: MappingResult) -> tuple[str, ...]:
    """Canonical fields behind a (numerator, denominator) pair of mapped columns."""
    if columns[0] == _NET_RETURN:
        return ("revenue", "spend") + tuple(f for f, c in mapping.active.items() if c == columns[1]
                                            and f not in ("revenue", "spend"))
    by_column = {c: f for f, c in mapping.active.items()}
    return tuple(by_column[c] for c in columns if c in by_column)


def _net_return_columns(df: pd.DataFrame, mapping: MappingResult):
    """Revenue − Spend, the other common ROI numerator (only when both are mapped)."""
    active = mapping.active
    if "revenue" in active and "spend" in active:
        values = _numbers(df[active["revenue"]]) - _numbers(df[active["spend"]])
        return [(_NET_RETURN, True, values)]
    return []


def apply_to_mapping(mapping: MappingResult, checks: CrossCheckResult) -> None:
    """Add 'verified by the file's own calculations' to the reason of verified mappings."""
    verified = checks.verified_fields
    for m in mapping.columns:
        if m.field in verified and "verified" not in m.reason:
            m.reason = (m.reason + "; " if m.reason else "") + "verified by the file's own calculations"
