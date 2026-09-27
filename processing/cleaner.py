"""Cleaning: raw text table + confirmed field mapping -> typed, standardized, canonical table.

Steps, in order (each change is written to the data-quality log):
  1. Remove summary rows such as "Grand Total" (they would double-count every metric).
  2. Remove exact duplicate rows (or only flag them, if configured).
  3. Keep mapped columns and rename them to canonical names; log dropped columns.
  4. Tidy category labels: trim spaces, fix case, merge spellings ("FB" -> "Meta").
  5. Parse dates (day-first for India) and numbers ("₹1,23,456" -> 123456).
     Blank / "N/A" become MISSING, never zero: a zero would silently distort totals and ratios.
  6. Impossible values (negative spend, clicks > impressions, ...): the row is kept, the bad
     value becomes missing so it is excluded from calculations, and the row gets a dq_flag.
  7. Flag outliers (extreme spend or CPL within a channel). Outliers are never removed:
     an unusual day may be real (a sale, a budget error) and is exactly what analysis should see.
  8. Enforce types and add calendar columns (week_start, month, quarter, year).

Alternative considered: dropping rows with any problem. Rejected: it hides problems and loses
the good values on those rows (a row with bad clicks still has valid spend and leads).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.fields import CANONICAL_LABELS, FIELD_BY_NAME
from config.settings import (DATE_DAYFIRST, DUPLICATE_POLICY, OUTLIER_IQR_MULTIPLIER,
                             OUTLIER_MIN_LEADS)
from ingestion.profiler import SUMMARY_ROW_PATTERN, _label_key, cluster_labels
from processing import transformer
from processing.quality import QualityLog, QualitySummary, summarise
from utils.formatting import format_count
from utils.parsing import DAY_FIRST_LABELS, missing_mask, parse_dates, parse_numbers


class CleaningError(Exception):
    """Cleaning cannot start. `user_message` is safe to show in the UI."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class CleanResult:
    clean_df: pd.DataFrame
    quality_log_df: pd.DataFrame
    quality_summary: QualitySummary


# Pairs (lower stage, upper stage): the lower one can never exceed the upper one in a row.
# The lower value is the suspect one (e.g. clicks recorded wrongly), so it is set to missing.
FUNNEL_RULES = [
    ("clicks", "impressions", "clicks cannot be higher than impressions"),
    ("reach", "impressions", "reach cannot be higher than impressions"),
    ("qualified_leads", "leads", "qualified leads cannot be higher than leads"),
    ("conversions", "leads", "conversions cannot be higher than leads"),
]


def clean_dataset(raw_df: pd.DataFrame, mapping, duplicate_policy: str = DUPLICATE_POLICY,
                  dayfirst: bool = DATE_DAYFIRST) -> CleanResult:
    """Clean a raw (all-text) table using a mapping from ingestion.mapper.map_fields."""
    if mapping.needs_confirmation:
        names = ", ".join(f"'{m.column}'" for m in mapping.needs_confirmation)
        raise CleaningError(f"Please confirm what these columns mean before cleaning: {names}.")
    if duplicate_policy not in ("remove", "flag"):
        raise ValueError("duplicate_policy must be 'remove' or 'flag'")

    log = QualityLog()
    rows_in = len(raw_df)
    df = raw_df.reset_index(drop=True).copy()
    df["source_row"] = np.arange(len(df))
    flags = pd.Series("", index=df.index, dtype=object)

    df = _remove_summary_rows(df, log)
    df, flags = _handle_duplicates(df, flags.loc[df.index], log, duplicate_policy)

    source_row = df["source_row"]
    df = transformer.select_and_rename(df.drop(columns="source_row"), mapping, log)
    df["source_row"] = source_row

    _standardize_text(df, log)
    _parse_dates(df, flags, log, dayfirst)
    _parse_numbers(df, flags, log)
    _invalid_values(df, flags, log)
    _flag_outliers(df, flags, log)

    df = transformer.enforce_types(df)
    df = transformer.add_time_columns(df, log)
    df["dq_flags"] = flags.loc[df.index].str.strip(";")
    df = transformer.order_columns(df).reset_index(drop=True)

    log_df = log.to_frame()
    summary = summarise(log_df, rows_in, df, _limitations(df, log_df))
    return CleanResult(df, log_df, summary)


# ---------------------------------------------------------------------------------------------
# Row removal
# ---------------------------------------------------------------------------------------------

def _remove_summary_rows(df: pd.DataFrame, log: QualityLog) -> pd.DataFrame:
    text = df.drop(columns="source_row").astype(str)
    is_summary = text.apply(lambda col: col.str.match(SUMMARY_ROW_PATTERN)).any(axis=1)
    for i in df.index[is_summary]:
        label = next(v for v in text.loc[i] if SUMMARY_ROW_PATTERN.match(v)).strip()
        log.add(int(df.at[i, "source_row"]), "(row)", label, "(removed)", "summary_row_removed",
                "summary/total row; keeping it would double-count every metric", "info")
    return df[~is_summary]


def _handle_duplicates(df, flags, log, policy):
    body = df.drop(columns="source_row")
    dup = body.duplicated(keep="first")
    if not dup.any():
        return df, flags
    # For each duplicate, find the first row it copies (for a useful log message).
    first_of = body.reset_index().groupby(list(body.columns), dropna=False)["index"].transform("first")
    first_of.index = body.index
    rows = df.loc[dup, "source_row"].tolist()
    originals = [f"copy of row {int(df.at[first_of[i], 'source_row'])}" for i in df.index[dup]]
    if policy == "remove":
        log.add(rows, "(row)", originals, "(removed)", "duplicate_removed",
                "exact duplicate of an earlier row", "info")
        return df[~dup], flags[~dup]
    log.add(rows, "(row)", originals, "(kept, flagged)", "duplicate_flagged",
            "exact duplicate of an earlier row", "warning")
    flags[dup] += "duplicate;"
    return df, flags


# ---------------------------------------------------------------------------------------------
# Text / categories
# ---------------------------------------------------------------------------------------------

def _standardize_text(df: pd.DataFrame, log: QualityLog) -> None:
    for col in df.columns:
        spec = FIELD_BY_NAME.get(col)
        if spec is None or spec.ftype not in ("category", "id", "text"):
            continue
        original = df[col].fillna("").astype(str)
        missing = missing_mask(original)
        tidy = original.str.strip().str.replace(r"\s+", " ", regex=True)

        new = tidy
        if spec.ftype == "category":
            # Known channel/platform spellings first ("FB", "facebook" -> "Meta") ...
            if col in ("channel", "platform"):
                known = {label: CANONICAL_LABELS[_label_key(label)]
                         for label in tidy[~missing].unique()
                         if _label_key(label) in CANONICAL_LABELS}
                new = new.replace(known)
            # ... then clusters of near-identical spellings (the same logic the profiler reports).
            merge = {variant: cluster.suggested_label
                     for cluster in cluster_labels(new[~missing], col)
                     for variant in cluster.variants}
            new = new.replace(merge)

        new = new.where(~missing, None)
        changed = (~missing) & (new != original)
        if changed.any():
            label_changed = changed & (new.str.lower() != tidy.str.lower())
            spacing_only = changed & ~label_changed & (new == tidy)
            case_only = changed & ~label_changed & ~spacing_only
            for mask, action, reason in [
                (label_changed, "label_standardized", "different spelling of the same label"),
                (case_only, "label_standardized", "same label in different capitalisation"),
                (spacing_only, "whitespace_trimmed", "extra spaces removed"),
            ]:
                if mask.any():
                    log.add(df.loc[mask, "source_row"].tolist(), col, original[mask].tolist(),
                            new[mask].tolist(), action, reason, "info")
        if missing.any() and spec.ftype == "category":
            _log_missing(df, col, original, missing, log, "warning")
        df[col] = new


# ---------------------------------------------------------------------------------------------
# Dates and numbers
# ---------------------------------------------------------------------------------------------

def _parse_dates(df, flags, log, dayfirst) -> None:
    for col in [c for c in df.columns if FIELD_BY_NAME.get(c) and FIELD_BY_NAME[c].ftype == "date"]:
        original = df[col].fillna("").astype(str)
        missing = missing_mask(original)
        dates, labels = parse_dates(original, dayfirst=dayfirst)
        main = labels.value_counts().idxmax() if labels.notna().any() else None

        other_format = labels.notna() & (labels != main)
        if other_format.any():
            for label in labels[other_format].unique():
                m = other_format & (labels == label)
                log.add(df.loc[m, "source_row"].tolist(), col, original[m].tolist(),
                        dates[m].dt.strftime("%Y-%m-%d").tolist(), "date_standardized",
                        f"date written as {label}; converted to YYYY-MM-DD", "info")

        # 05/01/2026 could be 5 Jan or 1 May: flag the reading we chose.
        ambiguous = labels.isin(DAY_FIRST_LABELS) & (dates.dt.day <= 12) & (dates.dt.day != dates.dt.month)
        if ambiguous.any():
            reading = "day-first (Indian convention)" if dayfirst else "month-first"
            log.add(df.loc[ambiguous, "source_row"].tolist(), col, original[ambiguous].tolist(),
                    dates[ambiguous].dt.strftime("%Y-%m-%d").tolist(), "ambiguous_date",
                    f"day and month could be swapped; read as {reading}", "warning")

        bad = ~missing & dates.isna()
        if bad.any():
            log.add(df.loc[bad, "source_row"].tolist(), col, original[bad].tolist(), None,
                    "unreadable_value", "not a recognisable date; set to missing", "error")
            flags[bad] += f"unreadable_{col};"
        if missing.any():
            _log_missing(df, col, original, missing, log, "warning")
            flags[missing] += f"missing_{col};"
        df[col] = dates


def _parse_numbers(df, flags, log) -> None:
    numeric = [c for c in df.columns
               if FIELD_BY_NAME.get(c) and FIELD_BY_NAME[c].ftype in ("number", "money")]
    for col in numeric:
        original = df[col].fillna("").astype(str)
        missing = missing_mask(original)
        values, formatted = parse_numbers(original)

        if formatted.any():
            log.add(df.loc[formatted, "source_row"].tolist(), col, original[formatted].tolist(),
                    values[formatted].tolist(), "text_to_number",
                    "number written as text (currency sign, INR or separators)", "info")
        bad = ~missing & values.isna()
        if bad.any():
            log.add(df.loc[bad, "source_row"].tolist(), col, original[bad].tolist(), None,
                    "unreadable_value", "not a recognisable number; set to missing", "error")
            flags[bad] += f"unreadable_{col};"
        if missing.any():
            _log_missing(df, col, original, missing, log, "warning")
        df[col] = values


def _log_missing(df, col, original, missing, log, severity) -> None:
    shown = original[missing].str.strip().replace("", "(blank)")
    log.add(df.loc[missing, "source_row"].tolist(), col, shown.tolist(), None, "missing_value",
            "no value in the file; kept as missing (not zero)", severity)


# ---------------------------------------------------------------------------------------------
# Impossible values and outliers
# ---------------------------------------------------------------------------------------------

def _invalid_values(df, flags, log) -> None:
    numeric = [c for c in df.columns
               if FIELD_BY_NAME.get(c) and FIELD_BY_NAME[c].ftype in ("number", "money")
               and c != "margin"]
    for col in numeric:
        negative = df[col] < 0
        if negative.any():
            label = FIELD_BY_NAME[col].label
            log.add(df.loc[negative, "source_row"].tolist(), col, df.loc[negative, col].tolist(),
                    None, "invalid_value_removed",
                    f"{label} cannot be negative; value excluded from calculations", "error")
            flags[negative] += f"negative_{col};"
            df.loc[negative, col] = np.nan

    for lower, upper, reason in FUNNEL_RULES:
        if lower in df and upper in df:
            bad = df[lower] > df[upper]
            if bad.any():
                log.add(df.loc[bad, "source_row"].tolist(), lower, df.loc[bad, lower].tolist(), None,
                        "invalid_value_removed", f"{reason}; value excluded from calculations",
                        "error")
                flags[bad] += f"{lower}_gt_{upper};"
                df.loc[bad, lower] = np.nan


def _flag_outliers(df, flags, log) -> None:
    """Flag (never remove) extreme spend and CPL within each channel."""
    if "spend" not in df:
        return
    groups = df["channel"] if "channel" in df else pd.Series("all", index=df.index)
    checks = {"spend": df["spend"]}
    if "leads" in df:
        enough = df["leads"] >= OUTLIER_MIN_LEADS
        checks["cpl"] = (df["spend"] / df["leads"]).where(enough)
    k = OUTLIER_IQR_MULTIPLIER
    for metric, values in checks.items():
        is_outlier = pd.Series(False, index=df.index)
        for _, idx in values.groupby(groups.fillna("(none)")).groups.items():
            v = values.loc[idx].dropna()
            if len(v) < 20:
                continue  # too few values to judge what is "normal"
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            out = (values.loc[idx] > q3 + k * iqr) | (values.loc[idx] < q1 - k * iqr)
            is_outlier.loc[idx] = out.fillna(False)
        if is_outlier.any():
            shown = values[is_outlier].round(2)
            name = "CPL" if metric == "cpl" else metric
            log.add(df.loc[is_outlier, "source_row"].tolist(), metric, shown.tolist(), shown.tolist(),
                    "outlier_flagged", f"unusually high or low {name} for its channel; kept, only "
                    "flagged", "info")
            flags[is_outlier] += f"outlier_{metric};"


# ---------------------------------------------------------------------------------------------
# Plain-English limitations
# ---------------------------------------------------------------------------------------------

def _limitations(df: pd.DataFrame, log: pd.DataFrame) -> list[str]:
    notes: list[str] = []

    def count(action, column=None):
        m = log["action"] == action
        if column is not None:
            m &= log["column"] == column
        return int(m.sum())

    for col in ["revenue", "spend", "leads", "conversions", "clicks", "impressions", "date"]:
        if col not in df:
            continue
        label = FIELD_BY_NAME[col].label
        missing = count("missing_value", col) + count("unreadable_value", col)
        invalid = count("invalid_value_removed", col)
        if missing:
            effect = ("trends and anomaly checks will skip these rows" if col == "date"
                      else f"{label.lower()} totals may be understated")
            notes.append(f"{format_count(missing)} rows have no usable {label.lower()}; {effect}.")
        if invalid:
            notes.append(f"{format_count(invalid)} impossible {label.lower()} values were excluded "
                         f"from calculations; {label.lower()} totals may be understated.")
    for col in ["qualified_leads", "reach"]:
        n = count("invalid_value_removed", col)
        if n:
            notes.append(f"{format_count(n)} impossible {FIELD_BY_NAME[col].label.lower()} values "
                         "were excluded from calculations.")
    if count("ambiguous_date"):
        notes.append(f"{format_count(count('ambiguous_date'))} dates could be read two ways "
                     "(e.g. 05/01/2026); they were read day-first.")
    if count("duplicate_flagged"):
        notes.append(f"{format_count(count('duplicate_flagged'))} duplicate rows were kept (flag-only "
                     "mode); totals may be overstated.")
    if count("outlier_flagged"):
        notes.append(f"{format_count(count('outlier_flagged'))} unusual values were flagged for "
                     "review but kept in the analysis.")
    return notes
