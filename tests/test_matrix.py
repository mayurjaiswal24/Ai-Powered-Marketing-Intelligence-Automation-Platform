"""Phase 14 test matrix: every sample and mapping test file end to end, plus a locked database.

Each file goes through the whole product: load -> profile -> map (with the user's choice where
the app must ask) -> clean -> store -> analyse -> PDF -> Excel. Other rows of the matrix (bad
files, AI failures, public mode, demo seed) are covered by their own test files; see
docs/ACCEPTANCE_CHECKLIST.md.
"""

import sqlite3
from pathlib import Path

import pytest

from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from database import repository
from database.connection import DatabaseError
from reports.excel_report import export_workbook
from reports.pdf_report import export_pdf
from tests.fixtures.build_mapping_fixtures import OUT, build

REAL_FILE = Path(r"real_test_file.xlsx")

# file -> (choices the user makes when the app asks, KPIs that must be available)
SAMPLES = {
    "test_a_small.csv": ({}, ["spend", "leads", "cpl"]),
    "test_b_medium.csv": ({}, ["spend", "leads", "cpl"]),
    "test_c_large.csv": ({}, ["spend", "roas", "cac"]),
    "marketing_clean.csv": ({}, ["spend", "revenue", "roas", "roi"]),
    "marketing_messy.xlsx": ({}, ["spend", "revenue", "roas"]),
    "marketing_no_revenue.csv": ({}, ["spend", "leads", "cpl"]),
    "marketing_no_margin.csv": ({}, ["spend", "revenue", "roas"]),
    "meta_ads_export_style.csv": ({"Results": "leads"}, ["spend", "leads", "cpl", "ctr"]),
}
MAPPING_FILES = {
    "google_ads.csv": ({}, ["spend", "clicks", "conversions", "roas"]),
    "meta_ads.csv": ({}, ["spend", "leads", "cpl", "ctr"]),
    "ecommerce.csv": ({}, ["spend", "revenue", "roas", "aov"]),
    "leadgen_crm.csv": ({"Marketing Cost": "spend"}, ["spend", "leads", "cpl", "cac"]),
    "agency_report.csv": ({}, ["spend", "conversions", "roas"]),
    "retail_columns.csv": ({}, ["spend", "revenue", "roas"]),
}


@pytest.fixture(scope="module", autouse=True)
def mapping_files():
    build()


def _end_to_end(path: Path, choices: dict, kpis: list[str], tmp_path: Path) -> None:
    raw, report = load_raw(path)
    prep = prepare(raw, report, choices or None)
    assert prep.validation.can_analyse, prep.validation.blocking
    db = tmp_path / "app.db"
    output = run_pipeline(prep, db_path=db)
    assert output.run_id is not None and output.save_error is None
    for key in kpis:
        assert output.analysis.kpis[key].available, key
    # ROI never appears without gross profit or margin data (no margin is ever assumed).
    if not {"gross_profit", "margin"} & set(output.clean.clean_df.columns):
        assert not output.analysis.kpis["roi"].available
    pdf = export_pdf(output.analysis, exports_dir=tmp_path / "exports", db_path=db)
    xlsx = export_workbook(output.analysis, exports_dir=tmp_path / "exports", db_path=db)
    assert pdf.stat().st_size > 10_000 and xlsx.stat().st_size > 5_000
    assert repository.list_recent_runs(db_path=db)["run_id"].tolist() == [output.run_id]


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_file_end_to_end(name, tmp_path):
    choices, kpis = SAMPLES[name]
    _end_to_end(SAMPLE_DIR / name, choices, kpis, tmp_path)


@pytest.mark.parametrize("name", list(MAPPING_FILES))
def test_mapping_test_file_end_to_end(name, tmp_path):
    choices, kpis = MAPPING_FILES[name]
    _end_to_end(OUT / name, choices, kpis, tmp_path)


@pytest.mark.skipif(not REAL_FILE.exists(), reason="the real test file is not on this computer")
def test_real_file_end_to_end(tmp_path):
    _end_to_end(REAL_FILE, {}, ["spend", "revenue", "roas", "cac"], tmp_path)


# --- Database missing or locked ------------------------------------------------------------------

def test_database_locked_gives_a_plain_message(tmp_path):
    db = tmp_path / "locked.db"
    repository.init_db(db)
    holder = sqlite3.connect(db)
    holder.execute("BEGIN EXCLUSIVE")                  # another process holds the database
    try:
        with pytest.raises(DatabaseError) as caught:
            repository.create_run(repository.register_dataset("f.csv", "h", db_path=db).id, db_path=db)
        assert "busy" in caught.value.user_message
    finally:
        holder.rollback()
        holder.close()
    # The analysis itself still works when saving is impossible (results shown, not saved).
    raw, report = load_raw(SAMPLE_DIR / "test_a_small.csv")
    lock = sqlite3.connect(db)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        output = run_pipeline(prepare(raw, report), db_path=db)
    finally:
        lock.rollback()
        lock.close()
    assert output.run_id is None and "busy" in output.save_error
    assert output.analysis.kpis["spend"].available


def test_database_folder_missing_is_created_or_explained(tmp_path):
    missing = tmp_path / "no" / "such" / "folder" / "app.db"
    with pytest.raises(DatabaseError) as caught:           # an explicit path is never invented
        repository.list_recent_runs(db_path=missing)
    assert "could not be opened" in caught.value.user_message
