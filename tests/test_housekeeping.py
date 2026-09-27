"""Storage housekeeping: delete one analysis, keep only the newest KEEP_LAST_RUNS."""

from pathlib import Path

import pytest

import database.connection as dbc
import config.settings as config_settings
from config.settings import Settings, load_settings
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from database import housekeeping, repository
from database.connection import get_db_path

TABLES_WITH_RUN = ("field_mappings", "data_quality_logs", "marketing_records", "kpi_results",
                   "run_snapshots", "reports", "ai_insights")


def make_run(db, exports, name="file.csv", file_hash=None, with_files=True) -> int:
    """A run with a snapshot, a report file, cached AI insights and an AI call log entry."""
    ds = repository.register_dataset(name, file_hash or f"hash-{name}", 10, 3, "csv", db_path=db)
    run_id = repository.create_run(ds.id, db_path=db)
    repository.save_run_snapshot(run_id, b"snapshot", db_path=db)
    with repository.session(db) as conn:
        conn.execute("INSERT INTO kpi_results (run_id, kpi, value) VALUES (?, 'spend', 1.0)", (run_id,))
    repository.save_ai_insights(run_id, f"fp{run_id}", {"x": 1}, db_path=db)
    repository.save_ai_run(run_id, fingerprint=f"fp{run_id}", model="m", prompt_version="v4",
                           success=True, db_path=db)
    if with_files:
        folder = exports / str(run_id)
        folder.mkdir(parents=True)
        for kind, suffix in (("pdf", "pdf"), ("excel", "xlsx")):
            path = folder / f"report_run{run_id}.{suffix}"
            path.write_bytes(b"x")
            repository.save_report(run_id, kind, str(path), db_path=db)
    return run_id


def count(db, table, run_id=None):
    with repository.session(db) as conn:
        if run_id is None:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()[0]


@pytest.fixture
def store(tmp_path):
    return tmp_path / "app.db", tmp_path / "exports"


def test_default_location_and_setting():
    s = load_settings({})
    assert s.database_path == "data/app/marketing_intelligence.db" and s.keep_last_runs == 20
    assert load_settings({"KEEP_LAST_RUNS": "5"}).keep_last_runs == 5
    assert load_settings({"KEEP_LAST_RUNS": "0"}).keep_last_runs == 20     # below minimum -> default


def test_default_database_folder_is_created(tmp_path, monkeypatch):
    target = tmp_path / "data" / "app" / "m.db"
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(target)))
    repository.init_db()
    assert get_db_path() == target and target.exists()


def test_delete_analysis_removes_everything_of_that_run(store):
    db, exports = store
    keep, gone = make_run(db, exports, "a.csv"), make_run(db, exports, "b.csv")
    assert housekeeping.delete_analysis(gone, db_path=db, exports_dir=exports)
    assert repository.get_run(gone, db_path=db) is None
    for table in TABLES_WITH_RUN:
        assert count(db, table, gone) == 0, table
        if table != "field_mappings" and table != "data_quality_logs" and table != "marketing_records":
            assert count(db, table, keep) >= 1, table                 # the other run is untouched
    assert not (exports / str(gone)).exists()                           # files and empty folder
    assert (exports / str(keep)).exists()
    assert repository.find_dataset_by_hash("hash-b.csv", db_path=db) is None   # unused dataset
    assert repository.find_dataset_by_hash("hash-a.csv", db_path=db) is not None
    # The AI call log survives (daily quota), just unlinked from the deleted run.
    assert count(db, "ai_runs") == 2
    with repository.session(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ai_runs WHERE run_id IS NULL").fetchone()[0] == 1
    assert not housekeeping.delete_analysis(gone, db_path=db, exports_dir=exports)   # already gone


def test_dataset_kept_while_another_run_uses_it(store):
    db, exports = store
    first = make_run(db, exports, "same.csv", file_hash="h")
    make_run(db, exports, "same.csv", file_hash="h")                    # re-upload, second run
    housekeeping.delete_analysis(first, db_path=db, exports_dir=exports)
    assert repository.find_dataset_by_hash("h", db_path=db) is not None


def test_never_deletes_files_outside_the_exports_folder(store, tmp_path):
    db, exports = store
    run_id = make_run(db, exports, with_files=False)
    outside = tmp_path / "precious.pdf"
    outside.write_bytes(b"keep me")
    repository.save_report(run_id, "pdf", str(outside), db_path=db)
    housekeeping.delete_analysis(run_id, db_path=db, exports_dir=exports)
    assert outside.exists()


def test_folder_with_other_files_is_kept(store):
    db, exports = store
    run_id = make_run(db, exports)
    (exports / str(run_id) / "my_notes.txt").write_text("not from the app")
    housekeeping.delete_analysis(run_id, db_path=db, exports_dir=exports)
    assert list((exports / str(run_id)).iterdir()) == [exports / str(run_id) / "my_notes.txt"]


def test_retention_keeps_the_newest_runs(store):
    db, exports = store
    runs = [make_run(db, exports, f"f{i}.csv") for i in range(5)]
    deleted = housekeeping.apply_retention(keep=3, db_path=db, exports_dir=exports)
    assert deleted == runs[:2]                                          # the two oldest
    assert list(repository.list_recent_runs(limit=10, db_path=db)["run_id"]) == runs[:1:-1]
    for run_id in runs[:2]:
        assert not (exports / str(run_id)).exists()
        assert count(db, "ai_insights", run_id) == 0 and count(db, "run_snapshots", run_id) == 0
    for run_id in runs[2:]:
        assert (exports / str(run_id)).exists()
    assert housekeeping.apply_retention(keep=3, db_path=db, exports_dir=exports) == []   # stable


def test_new_ids_are_never_reused_after_deletion(store):
    db, exports = store
    a = make_run(db, exports, "a.csv")
    housekeeping.delete_analysis(a, db_path=db, exports_dir=exports)
    assert make_run(db, exports, "b.csv") > a          # no clash with an old export folder


def test_pipeline_applies_retention(tmp_path, monkeypatch):
    monkeypatch.setattr(config_settings, "settings", Settings(keep_last_runs=2))
    db = tmp_path / "app.db"
    raw, report = load_raw(SAMPLE_DIR / "test_a_small.csv")
    ids = [run_pipeline(prepare(raw, report), db_path=db).run_id for _ in range(3)]
    assert list(repository.list_recent_runs(limit=10, db_path=db)["run_id"]) == ids[:0:-1]
    assert repository.get_run(ids[0], db_path=db) is None


# --- The Delete button (streamlit.testing) -----------------------------------------------------

def test_delete_button_needs_confirmation(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    db = tmp_path / "app.db"
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(db)))
    raw, report = load_raw(SAMPLE_DIR / "test_a_small.csv")
    first = run_pipeline(prepare(raw, report), db_path=db).run_id
    second = run_pipeline(prepare(raw, report), db_path=db).run_id

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=240)
    at.run()
    at.button(key=f"delete_run_{first}").click().run()
    assert any("cannot be undone" in w.value for w in at.warning)
    at.button(key=f"cancel_delete_{first}").click().run()             # cancel keeps it
    assert repository.get_run(first, db_path=db) is not None
    at.button(key=f"delete_run_{first}").click().run()
    at.button(key=f"confirm_delete_{first}").click().run()
    assert not at.exception
    assert repository.get_run(first, db_path=db) is None
    assert repository.get_run(second, db_path=db) is not None
    assert any(f"#{first} was deleted" in s.value for s in at.success)
    assert not any(b.key == f"open_run_{first}" for b in at.button)
