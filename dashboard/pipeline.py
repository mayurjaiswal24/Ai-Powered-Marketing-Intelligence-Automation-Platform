"""The end-to-end pipeline behind the app, with no Streamlit code (so it can be tested alone).

    prepare()       load -> profile -> map fields -> check capabilities   (before the user confirms)
    run_pipeline()  clean -> save to SQLite -> analyse -> save results      (after "Run analysis")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from analytics.engine import AnalysisResult, run_analysis, save_analysis
from config.settings import PROJECT_ROOT
from database import repository
from database.connection import DatabaseError
from ingestion.loader import LoadReport, load_file
from ingestion.mapper import MappingResult, map_fields
from ingestion.profiler import DatasetProfile, profile_dataset
from ingestion.validator import ValidationResult, validate
from processing.cleaner import CleanResult, clean_dataset

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_DATASETS = {
    "Kalpa Learning - clean sample (CSV)": "marketing_clean.csv",
    "Kalpa Learning - messy sample (Excel)": "marketing_messy.xlsx",
    "Lead generation only - no revenue (CSV)": "marketing_no_revenue.csv",
    "Meta Ads export - needs a mapping decision (CSV)": "meta_ads_export_style.csv",
}
DEFAULT_SAMPLE = "Kalpa Learning - clean sample (CSV)"


@dataclass
class PreparedUpload:
    raw_df: pd.DataFrame
    report: LoadReport
    profile: DatasetProfile
    mapping: MappingResult
    validation: ValidationResult


@dataclass
class PipelineOutput:
    clean: CleanResult
    analysis: AnalysisResult
    run_id: int | None
    is_reupload: bool
    save_error: str | None      # plain-English message if saving to the database failed


def sample_path(label: str) -> Path:
    return SAMPLE_DIR / SAMPLE_DATASETS[label]


def load_raw(source, filename: str | None = None) -> tuple[pd.DataFrame, LoadReport]:
    return load_file(source, filename)


def prepare(raw_df: pd.DataFrame, report: LoadReport,
            overrides: dict[str, str | None] | None = None) -> PreparedUpload:
    """Profile the raw table, map its columns (with the user's choices) and check what the data
    supports. Cheap enough to repeat whenever the user changes a mapping choice."""
    profile = profile_dataset(raw_df)
    mapping = map_fields(profile, overrides=overrides)
    profile.apply_mapping(raw_df, mapping)
    validation = validate(mapping, profile)
    return PreparedUpload(raw_df, report, profile, mapping, validation)


def run_pipeline(prep: PreparedUpload, progress: Callable[[str], None] = lambda _msg: None,
                 db_path=None) -> PipelineOutput:
    """Clean, store and analyse. Database problems do not stop the analysis: the results are
    still shown, with a note that they were not saved."""
    progress("Cleaning and standardising the data")
    cleaned = clean_dataset(prep.raw_df, prep.mapping)

    run_id, is_reupload, save_error = None, False, None
    progress("Saving the cleaned data")
    try:
        ds = repository.register_dataset(prep.report.filename, prep.report.file_hash,
                                         prep.report.rows, prep.report.columns,
                                         prep.report.file_type, db_path=db_path)
        is_reupload = ds.is_reupload
        run_id = repository.create_run(ds.id, {"overrides": {m.column: m.field for m in prep.mapping.columns
                                                             if m.source == "user"}}, db_path=db_path)
        repository.save_field_mappings(run_id, prep.mapping, db_path=db_path)
        repository.save_quality_log(run_id, cleaned.quality_log_df, db_path=db_path)
        repository.save_clean_records(run_id, cleaned.clean_df, db_path=db_path)
        repository.update_run_status(run_id, "cleaned", db_path=db_path)
    except DatabaseError as exc:
        save_error = exc.user_message
        run_id = None

    progress("Calculating KPIs, trends, funnel, segments and anomalies")
    analysis = run_analysis(cleaned.clean_df, prep.validation, cleaned.quality_summary,
                            dataset_name=prep.report.filename, run_id=run_id)

    output = PipelineOutput(cleaned, analysis, run_id, is_reupload, save_error)
    if run_id is not None:
        progress("Saving the analysis results")
        try:
            save_analysis(analysis, run_id, db_path=db_path)
            repository.save_run_snapshot(run_id, pickle.dumps(output), db_path=db_path)
            repository.update_run_status(run_id, "complete", db_path=db_path)
        except DatabaseError as exc:
            output.save_error = exc.user_message
    return output


def reopen_run(run_id: int, db_path=None) -> PipelineOutput:
    """Load a previous run's results from its snapshot (no recomputation).
    Raises DatabaseError with a plain message if it cannot be reopened."""
    payload = repository.load_run_snapshot(run_id, db_path=db_path)
    if payload is None:
        raise DatabaseError("This analysis has no saved results to reopen. Please upload the file again.")
    try:
        output = pickle.loads(payload)   # written by this app only (see database/schema.py)
    except Exception as exc:  # noqa: BLE001 - e.g. saved by an older version of the app
        raise DatabaseError("This analysis was saved by an older version of the app and cannot be "
                            "reopened. Please upload the file again.") from exc
    output.is_reupload = False
    output.save_error = None
    return output
