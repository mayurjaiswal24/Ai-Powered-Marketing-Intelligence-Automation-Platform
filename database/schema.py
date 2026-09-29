"""Database tables. Creating them is safe to repeat (CREATE TABLE IF NOT EXISTS).

The tables are kept separate by purpose (SPEC §8): what was uploaded, what was done to it,
the clean data, the analytics results, the AI outputs and the reports. Every result table
carries `run_id`, so each analysis run is self-contained and old runs can be compared or
deleted without touching anything else.

Analytics tables use a "tidy" long layout (one row = one entity x one metric). That keeps
the schema stable while analyses evolve, and SQL like "all metrics for campaign X" stays simple.
"""

from __future__ import annotations

import sqlite3

from config.fields import COUNT_FIELDS, FIELDS

SCHEMA_VERSION = 6   # 2: run_snapshots; 3: mapping cache + mapping-call log, runs.session_id;
                     # 4: learned_mappings (U2); 5: targets (U5); 6: learned_mappings removed


def _record_column_type(name: str, ftype: str) -> str:
    if name in COUNT_FIELDS or name == "year":
        return "INTEGER"
    if ftype in ("number", "money"):
        return "REAL"
    return "TEXT"   # dates are stored as ISO text (YYYY-MM-DD), which sorts correctly


# Clean marketing data: one column per canonical field, whether or not a dataset has it.
RECORD_COLUMNS: dict[str, str] = {f.name: _record_column_type(f.name, f.ftype) for f in FIELDS}
RECORD_COLUMNS.update({"week_start": "TEXT", "dq_flags": "TEXT", "source_row": "INTEGER"})

_record_cols_sql = ",\n    ".join(f'"{name}" {sqltype}' for name, sqltype in RECORD_COLUMNS.items())

TABLES = f"""
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per uploaded file. The SHA-256 hash lets us recognise a re-upload of the same file.
CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name     TEXT NOT NULL,
    file_type     TEXT,
    file_hash     TEXT NOT NULL UNIQUE,
    uploaded_at   TEXT NOT NULL,
    row_count     INTEGER,
    column_count  INTEGER
);

-- One analysis of a dataset (a dataset can be analysed several times, e.g. new mappings).
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'started'
                    CHECK (status IN ('started', 'cleaned', 'analysed', 'complete', 'failed')),
    settings_json   TEXT,
    record_columns  TEXT   -- JSON list of the clean-data columns this run actually has
);

CREATE TABLE IF NOT EXISTS field_mappings (
    run_id             INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    source_column      TEXT NOT NULL,
    canonical_field    TEXT,
    confidence         REAL,
    level              TEXT NOT NULL,
    confirmed_by_user  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, source_column)
);

CREATE TABLE IF NOT EXISTS data_quality_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    row_ref         INTEGER,
    column_name     TEXT,
    original_value  TEXT,
    cleaned_value   TEXT,
    action          TEXT NOT NULL,
    reason          TEXT,
    severity        TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error'))
);
CREATE INDEX IF NOT EXISTS ix_dql_run ON data_quality_logs(run_id);

CREATE TABLE IF NOT EXISTS marketing_records (
    record_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    {_record_cols_sql}
);
CREATE INDEX IF NOT EXISTS ix_records_run ON marketing_records(run_id);
CREATE INDEX IF NOT EXISTS ix_records_run_date ON marketing_records(run_id, date);

-- Headline KPIs for the whole run, with the numbers behind each one.
CREATE TABLE IF NOT EXISTS kpi_results (
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kpi          TEXT NOT NULL,
    value        REAL,
    numerator    REAL,
    denominator  REAL,
    formula      TEXT,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS campaign_analysis (
    run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    campaign  TEXT NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL
);

CREATE TABLE IF NOT EXISTS channel_analysis (
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    channel  TEXT NOT NULL,
    metric   TEXT NOT NULL,
    value    REAL
);

-- Segments in the broad sense: customer segment, region, city tier, product category ...
CREATE TABLE IF NOT EXISTS segment_analysis (
    run_id     INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    dimension  TEXT NOT NULL,
    segment    TEXT NOT NULL,
    metric     TEXT NOT NULL,
    value      REAL
);

CREATE TABLE IF NOT EXISTS funnel_analysis (
    run_id              INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scope               TEXT NOT NULL DEFAULT 'all',
    stage_order         INTEGER NOT NULL,
    stage               TEXT NOT NULL,
    value               REAL,
    rate_from_previous  REAL
);

CREATE TABLE IF NOT EXISTS trend_analysis (
    run_id           INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    grain            TEXT NOT NULL CHECK (grain IN ('day', 'week', 'month')),
    period           TEXT NOT NULL,
    dimension        TEXT NOT NULL DEFAULT 'all',
    dimension_value  TEXT NOT NULL DEFAULT 'all',
    metric           TEXT NOT NULL,
    value            REAL
);

CREATE TABLE IF NOT EXISTS anomalies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    metric        TEXT NOT NULL,
    entity_type   TEXT,
    entity        TEXT,
    grain         TEXT,
    period_start  TEXT,
    period_end    TEXT,
    baseline      REAL,
    observed      REAL,
    pct_change    REAL,
    score         REAL,
    direction     TEXT,
    sentiment     TEXT,   -- negative / positive / check (is the change bad, good or unclear?)
    severity      TEXT,
    method        TEXT,
    description   TEXT,
    details_json  TEXT
);

-- Every Gemini call (or cache hit), for quota tracking and troubleshooting.
CREATE TABLE IF NOT EXISTS ai_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    fingerprint     TEXT,
    model           TEXT,
    prompt_version  TEXT,
    success         INTEGER NOT NULL,
    error_type      TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      INTEGER,
    cache_hit       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_ai_runs_created ON ai_runs(created_at);

-- Validated AI output, stored separately from the facts it interprets.
CREATE TABLE IF NOT EXISTS ai_insights (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    fingerprint      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    insights_json    TEXT NOT NULL,
    evaluation_json  TEXT
);
CREATE INDEX IF NOT EXISTS ix_ai_insights_fp ON ai_insights(fingerprint);

-- Snapshot of a finished run's results, so "Recent analyses" can reopen it without recomputing.
-- Written and read only by this application (Python pickle of the cleaned data + AnalysisResult).
CREATE TABLE IF NOT EXISTS run_snapshots (
    run_id          INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL,
    payload         BLOB NOT NULL
);

-- Field-mapping memory per column layout (hash of the file's column names).
-- source 'ai' = Gemini's answer (reused: same layout, no new call); source 'user' = the user's
-- choice, which wins over AI and rules next time. field NULL = "not used".
CREATE TABLE IF NOT EXISTS mapping_cache (
    layout_key   TEXT NOT NULL,
    column_name  TEXT NOT NULL,
    source       TEXT NOT NULL CHECK (source IN ('ai', 'user')),
    field        TEXT,
    reason       TEXT,
    session_id   TEXT NOT NULL DEFAULT '',   -- public mode: a user's choices stay in their session
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (layout_key, column_name, source, session_id)
);

-- Every Gemini field-mapping call (not cache hits), for the daily / per-session mapping budget.
CREATE TABLE IF NOT EXISTS ai_mapping_calls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    layout_key     TEXT,
    session_id     TEXT,
    model          TEXT,
    success        INTEGER NOT NULL,
    error_type     TEXT,
    input_tokens   INTEGER,
    output_tokens  INTEGER
);
CREATE INDEX IF NOT EXISTS ix_ai_mapping_calls_created ON ai_mapping_calls(created_at);

-- Targets (U5): the user's target per metric for a column layout (the same layout key as
-- mapping_cache), so the next file with the same columns shows the same targets. Never written in
-- public mode (targets then live in the browser session only).
CREATE TABLE IF NOT EXISTS targets (
    layout_key  TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (layout_key, metric)
);

CREATE TABLE IF NOT EXISTS reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    report_type  TEXT NOT NULL CHECK (report_type IN ('pdf', 'excel')),
    file_path    TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""

# Tables that hold per-run analysis results in tidy form (saved/loaded generically).
ANALYSIS_TABLES = ("kpi_results", "campaign_analysis", "channel_analysis", "segment_analysis",
                   "funnel_analysis", "trend_analysis", "anomalies")


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist yet, and record the schema version once."""
    conn.executescript(TABLES)
    # Version 3: runs remember the browser session that made them (public mode isolation).
    if "session_id" not in table_columns(conn, "runs"):
        conn.execute("ALTER TABLE runs ADD COLUMN session_id TEXT")
    # Version 6: no cross-file mapping learning any more (see docs/DECISIONS.md).
    conn.execute("DROP TABLE IF EXISTS learned_mappings")
    current = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    if current is None or current < SCHEMA_VERSION:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
