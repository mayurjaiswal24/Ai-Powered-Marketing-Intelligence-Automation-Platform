"""The data-quality log: every change the cleaner makes, and a plain-English summary.

"Do not silently overwrite questionable source information" (SPEC §7). Each entry keeps the
row reference, the column, the original value, the cleaned value, what was done and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from utils.formatting import format_count

LOG_COLUMNS = ["row", "column", "original_value", "cleaned_value", "action", "reason", "severity"]
SEVERITIES = ("info", "warning", "error")


class QualityLog:
    """Collects log entries in batches (one batch per cleaning step) for speed."""

    def __init__(self) -> None:
        self._parts: list[pd.DataFrame] = []

    def add(self, rows, column, original, cleaned, action: str, reason: str, severity: str) -> None:
        """Add one entry per row. `rows`, `original` and `cleaned` may be scalars or lists of
        the same length; `row` is the 0-based row of the uploaded file (None = whole column)."""
        assert severity in SEVERITIES
        rows = list(rows) if isinstance(rows, (list, tuple, pd.Index, pd.Series)) else [rows]
        n = len(rows)
        if n == 0:
            return

        def as_list(value):
            if isinstance(value, (list, tuple, pd.Series, pd.Index)):
                return [_display(v) for v in value]
            return [_display(value)] * n

        self._parts.append(pd.DataFrame({
            "row": pd.array(rows, dtype="Int64"),
            "column": [column] * n,
            "original_value": as_list(original),
            "cleaned_value": as_list(cleaned),
            "action": [action] * n,
            "reason": [reason] * n,
            "severity": [severity] * n,
        }))

    def to_frame(self) -> pd.DataFrame:
        if not self._parts:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})
        return pd.concat(self._parts, ignore_index=True)[LOG_COLUMNS]


def _display(value) -> str:
    """Log values are stored as text so the log can show exactly what was there."""
    if value is None:
        return "(missing)"
    try:
        if pd.isna(value):
            return "(missing)"
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class QualitySummary:
    rows_in: int
    rows_out: int
    removed_duplicates: int
    removed_summary_rows: int
    flagged_rows: int
    counts_by_action: dict[str, int]
    counts_by_severity: dict[str, int]
    limitations: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = ["DATA QUALITY SUMMARY", "",
                 f"Rows in uploaded file: {format_count(self.rows_in)}",
                 f"Rows after cleaning:   {format_count(self.rows_out)}",
                 f"Duplicate rows removed: {format_count(self.removed_duplicates)}",
                 f"Summary rows removed:   {format_count(self.removed_summary_rows)}",
                 f"Rows with a quality flag: {format_count(self.flagged_rows)}",
                 "", "Changes by type:"]
        if not self.counts_by_action:
            lines.append("  No changes were needed.")
        for action, n in sorted(self.counts_by_action.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {action.replace('_', ' '):<32} {format_count(n):>8}")
        sev = ", ".join(f"{format_count(self.counts_by_severity.get(s, 0))} {s}" for s in SEVERITIES)
        lines += ["", f"Log entries by severity: {sev}", "", "What this means for the analysis:"]
        if not self.limitations:
            lines.append("  No limitations: the data could be used as supplied.")
        for text in self.limitations:
            lines.append(f"  - {text}")
        return "\n".join(lines)


def summarise(log: pd.DataFrame, rows_in: int, clean_df: pd.DataFrame,
              limitations: list[str]) -> QualitySummary:
    flagged = int((clean_df["dq_flags"] != "").sum()) if "dq_flags" in clean_df else 0
    actions = log["action"].value_counts()
    return QualitySummary(
        rows_in=rows_in,
        rows_out=len(clean_df),
        removed_duplicates=int(actions.get("duplicate_removed", 0)),
        removed_summary_rows=int(actions.get("summary_row_removed", 0)),
        flagged_rows=flagged,
        counts_by_action={str(k): int(v) for k, v in actions.items()},
        counts_by_severity={str(k): int(v) for k, v in log["severity"].value_counts().items()},
        limitations=limitations,
    )
