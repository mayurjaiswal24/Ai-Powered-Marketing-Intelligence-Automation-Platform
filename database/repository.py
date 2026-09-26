"""Saving and loading everything the platform keeps in SQLite.

Every function takes an optional `db_path` (tests use a temporary file); otherwise the
DATABASE_PATH setting is used. Tables are created automatically on first use, so the app works
from an empty database (important on Streamlit Community Cloud, which wipes files on restart).

Queries with user-supplied values always use parameters (`?`), never string formatting, which
prevents SQL injection. Table names are only ever taken from our own fixed list.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.fields import COUNT_FIELDS, FIELD_BY_NAME
from database.connection import DatabaseError, connect, get_db_path
from database.schema import (ANALYSIS_TABLES, RECORD_COLUMNS, create_schema, table_columns)

_initialised: set[Path] = set()

RUN_STATUSES = ("started", "cleaned", "analysed", "complete", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def session(db_path=None):
    """Connect, make sure the tables exist, then commit and close (see database.connection)."""
    path = get_db_path(db_path)
    is_new_file = not path.exists()
    with connect(path) as conn:
        if is_new_file or path not in _initialised:
            create_schema(conn)
            _initialised.add(path)
        yield conn


def init_db(db_path=None) -> Path:
    """Create the database file and all tables (safe to call repeatedly)."""
    path = get_db_path(db_path)
    _initialised.discard(path)
    with session(path):
        pass
    return path


# ---------------------------------------------------------------------------------------------
# Datasets and runs
# ---------------------------------------------------------------------------------------------

@dataclass
class DatasetRecord:
    id: int
    file_name: str
    file_hash: str
    uploaded_at: str
    is_reupload: bool          # True if this exact file was uploaded before
    row_count: int | None = None
    column_count: int | None = None


def register_dataset(file_name: str, file_hash: str, row_count: int | None = None,
                     column_count: int | None = None, file_type: str | None = None,
                     db_path=None) -> DatasetRecord:
    """Record an uploaded file. If a file with the same content (hash) exists, return that
    record with is_reupload=True instead of creating a duplicate."""
    with session(db_path) as conn:
        row = conn.execute(
            "SELECT id, file_name, file_hash, uploaded_at, row_count, column_count "
            "FROM datasets WHERE file_hash = ?", (file_hash,)).fetchone()
        if row:
            return DatasetRecord(row[0], row[1], row[2], row[3], True, row[4], row[5])
        uploaded_at = _now()
        cur = conn.execute(
            "INSERT INTO datasets (file_name, file_type, file_hash, uploaded_at, row_count, "
            "column_count) VALUES (?, ?, ?, ?, ?, ?)",
            (file_name, file_type, file_hash, uploaded_at, row_count, column_count))
        return DatasetRecord(cur.lastrowid, file_name, file_hash, uploaded_at, False,
                             row_count, column_count)


def find_dataset_by_hash(file_hash: str, db_path=None) -> DatasetRecord | None:
    with session(db_path) as conn:
        row = conn.execute("SELECT id, file_name, file_hash, uploaded_at, row_count, column_count "
                           "FROM datasets WHERE file_hash = ?", (file_hash,)).fetchone()
    return DatasetRecord(row[0], row[1], row[2], row[3], True, row[4], row[5]) if row else None


def create_run(dataset_id: int, settings: dict | None = None, db_path=None) -> int:
    with session(db_path) as conn:
        cur = conn.execute("INSERT INTO runs (dataset_id, created_at, settings_json) VALUES (?, ?, ?)",
                           (dataset_id, _now(), json.dumps(settings or {}, sort_keys=True)))
        return cur.lastrowid


def update_run_status(run_id: int, status: str, db_path=None) -> None:
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status: {status}")
    with session(db_path) as conn:
        conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))


def get_run(run_id: int, db_path=None) -> dict | None:
    with session(db_path) as conn:
        df = pd.read_sql("SELECT r.*, d.file_name, d.file_hash FROM runs r "
                         "JOIN datasets d ON d.id = r.dataset_id WHERE r.id = ?", conn,
                         params=(run_id,))
    if df.empty:
        return None
    run = df.iloc[0].to_dict()
    run["settings"] = json.loads(run.pop("settings_json") or "{}")
    return run


def list_runs(dataset_id: int | None = None, db_path=None) -> pd.DataFrame:
    sql = ("SELECT r.id, r.dataset_id, d.file_name, r.created_at, r.status FROM runs r "
           "JOIN datasets d ON d.id = r.dataset_id")
    params: tuple = ()
    if dataset_id is not None:
        sql += " WHERE r.dataset_id = ?"
        params = (dataset_id,)
    with session(db_path) as conn:
        return pd.read_sql(sql + " ORDER BY r.id DESC", conn, params=params)


# ---------------------------------------------------------------------------------------------
# Field mappings and quality log
# ---------------------------------------------------------------------------------------------

def save_field_mappings(run_id: int, mapping, db_path=None) -> None:
    rows = [(run_id, m.column, m.field or (m.candidates[0][0] if m.candidates else None),
             float(m.score), m.status, 1 if m.source == "user" else 0) for m in mapping.columns]
    with session(db_path) as conn:
        conn.execute("DELETE FROM field_mappings WHERE run_id = ?", (run_id,))
        conn.executemany("INSERT INTO field_mappings VALUES (?, ?, ?, ?, ?, ?)", rows)


def load_field_mappings(run_id: int, db_path=None) -> pd.DataFrame:
    with session(db_path) as conn:
        return pd.read_sql("SELECT source_column, canonical_field, confidence, level, "
                           "confirmed_by_user FROM field_mappings WHERE run_id = ? "
                           "ORDER BY rowid", conn, params=(run_id,))


_LOG_RENAME = {"row": "row_ref", "column": "column_name"}


def save_quality_log(run_id: int, log_df: pd.DataFrame, db_path=None) -> None:
    out = log_df.rename(columns=_LOG_RENAME).copy()
    out.insert(0, "run_id", run_id)
    with session(db_path) as conn:
        conn.execute("DELETE FROM data_quality_logs WHERE run_id = ?", (run_id,))
        out.to_sql("data_quality_logs", conn, if_exists="append", index=False)


def load_quality_log(run_id: int, db_path=None) -> pd.DataFrame:
    with session(db_path) as conn:
        df = pd.read_sql("SELECT row_ref, column_name, original_value, cleaned_value, action, "
                         "reason, severity FROM data_quality_logs WHERE run_id = ? ORDER BY id",
                         conn, params=(run_id,))
    df = df.rename(columns={v: k for k, v in _LOG_RENAME.items()})
    df["row"] = df["row"].astype("Int64")
    return df


# ---------------------------------------------------------------------------------------------
# Clean marketing records
# ---------------------------------------------------------------------------------------------

def save_clean_records(run_id: int, clean_df: pd.DataFrame, db_path=None) -> None:
    """Store the cleaned dataset. Dates are stored as ISO text; the list of columns this run
    actually has is remembered so loading returns exactly the same shape."""
    unknown = [c for c in clean_df.columns if c not in RECORD_COLUMNS]
    if unknown:
        raise ValueError(f"Columns not in marketing_records: {unknown}")
    out = clean_df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out.insert(0, "run_id", run_id)
    with session(db_path) as conn:
        conn.execute("DELETE FROM marketing_records WHERE run_id = ?", (run_id,))
        out.to_sql("marketing_records", conn, if_exists="append", index=False, chunksize=2000)
        conn.execute("UPDATE runs SET record_columns = ? WHERE id = ?",
                     (json.dumps(list(clean_df.columns)), run_id))


def load_clean_records(run_id: int, db_path=None) -> pd.DataFrame:
    with session(db_path) as conn:
        row = conn.execute("SELECT record_columns FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None or row[0] is None:
            raise DatabaseError("No cleaned data was found for this analysis run.")
        columns = json.loads(row[0])
        quoted = ", ".join(f'"{c}"' for c in columns)
        df = pd.read_sql(f"SELECT {quoted} FROM marketing_records WHERE run_id = ? "
                         "ORDER BY record_id", conn, params=(run_id,))
    return _restore_types(df)


def _restore_types(df: pd.DataFrame) -> pd.DataFrame:
    """SQLite has no date or nullable-integer types; restore the pandas types used in cleaning."""
    for col in df.columns:
        spec = FIELD_BY_NAME.get(col)
        if col in ("date", "week_start"):
            df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")  # same precision as cleaning
        elif col in COUNT_FIELDS or col == "year":
            df[col] = df[col].astype("Float64").astype("Int64")
        elif col == "source_row":
            df[col] = df[col].astype("int64")
        elif col == "dq_flags":
            df[col] = df[col].fillna("").astype(object)
        elif spec is not None and spec.ftype in ("number", "money"):
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype("string")
    return df


# ---------------------------------------------------------------------------------------------
# Analysis results (tidy tables)
# ---------------------------------------------------------------------------------------------

def save_analysis_table(run_id: int, table: str, df: pd.DataFrame, db_path=None) -> None:
    """Replace this run's rows in one of the analysis tables."""
    if table not in ANALYSIS_TABLES:
        raise ValueError(f"Not an analysis table: {table}")
    with session(db_path) as conn:
        allowed = set(table_columns(conn, table)) - {"id", "run_id"}
        extra = [c for c in df.columns if c not in allowed]
        if extra:
            raise ValueError(f"Columns not in {table}: {extra}")
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d")
        out.insert(0, "run_id", run_id)
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        out.to_sql(table, conn, if_exists="append", index=False)


def load_analysis_table(run_id: int, table: str, db_path=None) -> pd.DataFrame:
    if table not in ANALYSIS_TABLES:
        raise ValueError(f"Not an analysis table: {table}")
    with session(db_path) as conn:
        cols = [c for c in table_columns(conn, table) if c not in ("id", "run_id")]
        quoted = ", ".join(f'"{c}"' for c in cols)
        return pd.read_sql(f"SELECT {quoted} FROM {table} WHERE run_id = ? ORDER BY rowid",
                           conn, params=(run_id,))


# ---------------------------------------------------------------------------------------------
# AI records and reports
# ---------------------------------------------------------------------------------------------

def save_ai_run(run_id: int | None, *, fingerprint: str | None, model: str | None,
                prompt_version: str | None, success: bool, error_type: str | None = None,
                input_tokens: int | None = None, output_tokens: int | None = None,
                latency_ms: int | None = None, cache_hit: bool = False, db_path=None) -> int:
    with session(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO ai_runs (run_id, created_at, fingerprint, model, prompt_version, success, "
            "error_type, input_tokens, output_tokens, latency_ms, cache_hit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, _now(), fingerprint, model, prompt_version, int(success), error_type,
             input_tokens, output_tokens, latency_ms, int(cache_hit)))
        return cur.lastrowid


def count_ai_calls_since(since_iso: str, db_path=None) -> int:
    """Real (non-cached) Gemini calls since a UTC timestamp, for the daily quota."""
    with session(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM ai_runs WHERE cache_hit = 0 AND created_at >= ?",
                            (since_iso,)).fetchone()[0]


def save_ai_insights(run_id: int | None, fingerprint: str, insights: dict,
                     evaluation: dict | None = None, db_path=None) -> int:
    with session(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO ai_insights (run_id, fingerprint, created_at, insights_json, "
            "evaluation_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, fingerprint, _now(), json.dumps(insights, ensure_ascii=False),
             json.dumps(evaluation, ensure_ascii=False) if evaluation is not None else None))
        return cur.lastrowid


def load_latest_ai_insights(fingerprint: str, db_path=None) -> dict | None:
    """Most recent stored insights for an evidence fingerprint (the AI cache lookup)."""
    with session(db_path) as conn:
        row = conn.execute("SELECT insights_json, evaluation_json, created_at FROM ai_insights "
                           "WHERE fingerprint = ? ORDER BY id DESC LIMIT 1",
                           (fingerprint,)).fetchone()
    if row is None:
        return None
    return {"insights": json.loads(row[0]),
            "evaluation": json.loads(row[1]) if row[1] else None,
            "created_at": row[2]}


def save_report(run_id: int, report_type: str, file_path: str, db_path=None) -> int:
    with session(db_path) as conn:
        cur = conn.execute("INSERT INTO reports (run_id, report_type, file_path, created_at) "
                           "VALUES (?, ?, ?, ?)", (run_id, report_type, str(file_path), _now()))
        return cur.lastrowid


def list_reports(run_id: int, db_path=None) -> pd.DataFrame:
    with session(db_path) as conn:
        return pd.read_sql("SELECT id, report_type, file_path, created_at FROM reports "
                           "WHERE run_id = ? ORDER BY id", conn, params=(run_id,))


def delete_run(run_id: int, db_path=None) -> None:
    """Delete a run and (through ON DELETE CASCADE) everything that belongs to it."""
    with session(db_path) as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
