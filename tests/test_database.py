"""Phase 05 tests: SQLite schema, save/load round trips and SQL consistency (temporary DBs only)."""

import os
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from database import queries, repository as repo
from database.connection import DatabaseError, connect
from database.schema import SCHEMA_VERSION, create_schema
from ingestion.loader import load_file
from ingestion.mapper import map_fields
from ingestion.profiler import profile_dataset
from processing.cleaner import clean_dataset

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture(scope="module")
def cleaned():
    raw, report = load_file(SAMPLE / "test_c_large.csv")
    mapping = map_fields(profile_dataset(raw))
    result = clean_dataset(raw, mapping)
    df = result.clean_df.copy()
    # Add realistic gaps so NULL handling is exercised: missing revenue, a removed click value.
    df.loc[[3, 10, 50], "revenue"] = np.nan
    df.loc[[7], "clicks"] = pd.NA
    df.loc[[7], "dq_flags"] = "clicks_gt_impressions"
    return report, mapping, result, df


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def run_id(db, cleaned):
    report, _, _, _ = cleaned
    ds = repo.register_dataset(report.filename, report.file_hash, report.rows, report.columns,
                               report.file_type, db_path=db)
    return repo.create_run(ds.id, {"duplicate_policy": "remove"}, db_path=db)


# --- Schema ------------------------------------------------------------------------------------

def test_schema_creation_is_idempotent(db):
    repo.init_db(db)
    repo.init_db(db)
    with connect(db) as conn:
        create_schema(conn)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        versions = conn.execute("SELECT version FROM schema_version").fetchall()
    expected = {"schema_version", "datasets", "runs", "field_mappings", "data_quality_logs",
                "marketing_records", "kpi_results", "campaign_analysis", "channel_analysis",
                "segment_analysis", "funnel_analysis", "trend_analysis", "anomalies", "ai_runs",
                "ai_insights", "reports"}
    assert expected <= tables
    assert versions == [(SCHEMA_VERSION,)]


def test_works_from_empty_database(tmp_path):
    fresh = tmp_path / "sub" / ".." / "fresh.db"
    assert repo.list_runs(db_path=fresh).empty     # tables are created on first use


# --- Datasets, runs, re-uploads ------------------------------------------------------------------

def test_reupload_is_detected(db):
    first = repo.register_dataset("a.csv", "abc123", 10, 3, "csv", db_path=db)
    again = repo.register_dataset("a copy.csv", "abc123", 10, 3, "csv", db_path=db)
    other = repo.register_dataset("b.csv", "def456", 5, 3, "csv", db_path=db)
    assert not first.is_reupload and again.is_reupload
    assert again.id == first.id and again.file_name == "a.csv"
    assert other.id != first.id
    assert repo.find_dataset_by_hash("abc123", db_path=db).id == first.id
    assert repo.find_dataset_by_hash("nope", db_path=db) is None


def test_run_lifecycle(db, run_id):
    run = repo.get_run(run_id, db_path=db)
    assert run["status"] == "started" and run["settings"] == {"duplicate_policy": "remove"}
    repo.update_run_status(run_id, "cleaned", db_path=db)
    assert repo.get_run(run_id, db_path=db)["status"] == "cleaned"
    with pytest.raises(ValueError):
        repo.update_run_status(run_id, "exploded", db_path=db)
    assert len(repo.list_runs(db_path=db)) == 1


# --- Round trips -------------------------------------------------------------------------------

def test_clean_records_round_trip(db, run_id, cleaned):
    df = cleaned[3]
    repo.save_clean_records(run_id, df, db_path=db)
    loaded = repo.load_clean_records(run_id, db_path=db)
    pd.testing.assert_frame_equal(loaded, df.reset_index(drop=True), check_dtype=True)


def test_saving_twice_replaces_not_duplicates(db, run_id, cleaned):
    df = cleaned[3]
    repo.save_clean_records(run_id, df, db_path=db)
    repo.save_clean_records(run_id, df, db_path=db)
    assert len(repo.load_clean_records(run_id, db_path=db)) == len(df)


def test_quality_log_round_trip(db, run_id):
    raw, _ = load_file(SAMPLE / "test_a_small.csv")
    doubled = pd.concat([raw, raw.iloc[:2]], ignore_index=True)
    doubled.loc[5, "spend"] = "-100"
    doubled.loc[6, "revenue"] = "N/A"
    log = clean_dataset(doubled, map_fields(profile_dataset(doubled))).quality_log_df
    assert len(log) >= 4
    repo.save_quality_log(run_id, log, db_path=db)
    loaded = repo.load_quality_log(run_id, db_path=db)
    pd.testing.assert_frame_equal(loaded, log.reset_index(drop=True), check_dtype=False)
    assert loaded["row"].dtype == "Int64"


def test_field_mappings_round_trip(db, run_id):
    raw, _ = load_file(SAMPLE / "meta_ads_export_style.csv")
    mapping = map_fields(profile_dataset(raw), overrides={"Results": "leads"})
    repo.save_field_mappings(run_id, mapping, db_path=db)
    loaded = repo.load_field_mappings(run_id, db_path=db)
    assert loaded["source_column"].tolist() == [m.column for m in mapping.columns]
    results = loaded[loaded["source_column"] == "Results"].iloc[0]
    assert (results["canonical_field"], results["level"], results["confirmed_by_user"]) == \
        ("leads", "confirmed", 1)


def test_analysis_table_round_trip(db, run_id):
    kpis = pd.DataFrame({"kpi": ["ctr", "roas"], "value": [2.5, np.nan],
                         "numerator": [25.0, 0.0], "denominator": [1000.0, 0.0],
                         "formula": ["Clicks / Impressions x 100", "Revenue / Spend"],
                         "note": [None, "Spend is zero"]})
    repo.save_analysis_table(run_id, "kpi_results", kpis, db_path=db)
    pd.testing.assert_frame_equal(repo.load_analysis_table(run_id, "kpi_results", db_path=db), kpis)
    with pytest.raises(ValueError):
        repo.save_analysis_table(run_id, "datasets", kpis, db_path=db)
    with pytest.raises(ValueError):
        repo.save_analysis_table(run_id, "kpi_results", kpis.assign(bogus=1), db_path=db)


def test_ai_records_and_reports(db, run_id):
    repo.save_ai_run(run_id, fingerprint="fp1", model="test-model", prompt_version="v1",
                     success=True, input_tokens=100, output_tokens=50, latency_ms=1200, db_path=db)
    repo.save_ai_run(run_id, fingerprint="fp1", model="test-model", prompt_version="v1",
                     success=True, cache_hit=True, db_path=db)
    assert repo.count_ai_calls_since("2000-01-01", db_path=db) == 1   # cache hits don't count
    repo.save_ai_insights(run_id, "fp1", {"insights": [{"text": "₹ ok"}]}, {"score": 1}, db_path=db)
    stored = repo.load_latest_ai_insights("fp1", db_path=db)
    assert stored["insights"]["insights"][0]["text"] == "₹ ok"
    assert repo.load_latest_ai_insights("missing", db_path=db) is None
    repo.save_report(run_id, "pdf", "exports/report.pdf", db_path=db)
    assert repo.list_reports(run_id, db_path=db)["report_type"].tolist() == ["pdf"]


def test_deleting_a_run_removes_its_data(db, run_id, cleaned):
    repo.save_clean_records(run_id, cleaned[3], db_path=db)
    repo.delete_run(run_id, db_path=db)
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM marketing_records").fetchone()[0] == 0


# --- SQL vs pandas -----------------------------------------------------------------------------

def test_sql_channel_totals_equal_pandas(db, run_id, cleaned):
    df = cleaned[3]
    repo.save_clean_records(run_id, df, db_path=db)
    sql = queries.channel_totals(run_id, db_path=db).set_index("channel").sort_index()
    pandas_totals = (df.groupby("channel")[["spend", "impressions", "clicks", "leads",
                                            "conversions", "revenue"]].sum().sort_index())
    for col in pandas_totals.columns:
        np.testing.assert_allclose(sql[col].to_numpy(float), pandas_totals[col].to_numpy(float),
                                   rtol=1e-9, err_msg=col)
    assert sql["rows"].sum() == len(df)


def test_monthly_and_top_campaign_queries(db, run_id, cleaned):
    df = cleaned[3]
    repo.save_clean_records(run_id, df, db_path=db)
    monthly = queries.monthly_spend_revenue(run_id, db_path=db)
    expected = df.groupby("month")[["spend", "revenue"]].sum()
    assert monthly["month"].tolist() == expected.index.tolist()
    np.testing.assert_allclose(monthly["revenue"], expected["revenue"])
    top = queries.top_campaigns_by_conversions(run_id, limit=3, db_path=db)
    by_campaign = df.groupby("campaign_name")["conversions"].sum().sort_values(ascending=False)
    assert len(top) == 3
    assert top["conversions"].iloc[0] == by_campaign.iloc[0]


# --- Friendly errors ---------------------------------------------------------------------------

def test_unopenable_path_gives_friendly_error(tmp_path):
    with pytest.raises(DatabaseError) as info:
        repo.init_db(tmp_path)                       # a folder, not a file
    assert "could not be opened" in info.value.user_message
    with pytest.raises(DatabaseError):
        repo.init_db(tmp_path / "no_such_folder" / "x.db")


def test_read_only_database_gives_friendly_error(tmp_path):
    path = tmp_path / "ro.db"
    repo.init_db(path)
    os.chmod(path, stat.S_IREAD)
    try:
        with pytest.raises(DatabaseError) as info:
            repo.register_dataset("a.csv", "hash", db_path=path)
        assert "read-only" in info.value.user_message
        assert "sqlite" not in info.value.user_message.lower()
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_corrupt_database_file(tmp_path):
    path = tmp_path / "bad.db"
    path.write_bytes(b"this is not a database" * 100)
    with pytest.raises(DatabaseError) as info:
        repo.init_db(path)
    assert "damaged" in info.value.user_message


def test_error_rolls_back_the_transaction(db):
    repo.init_db(db)
    with pytest.raises(DatabaseError):
        with connect(db) as conn:
            conn.execute("INSERT INTO datasets (file_name, file_hash, uploaded_at) VALUES ('x', 'h1', 'now')")
            conn.execute("INSERT INTO datasets (file_name, file_hash, uploaded_at) VALUES ('y', 'h1', 'now')")
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0


def test_foreign_keys_enforced(db):
    with pytest.raises(DatabaseError):
        repo.create_run(999, db_path=db)
    with pytest.raises(sqlite3.Error):
        raw = sqlite3.connect(db)
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute("INSERT INTO runs (dataset_id, created_at) VALUES (999, 'now')")
