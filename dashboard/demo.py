"""Instant demo: the finished Kalpa sample analysis, shipped with the app.

On a fresh deployment the database is empty (Streamlit Community Cloud wipes files on restart),
and a full analysis of the 8,471-row sample takes about 10 seconds. So the finished analysis
(`data/demo/kalpa_demo_snapshot.pkl.gz`, written by scripts/refresh_demo_seed.py) is restored
when "Load sample dataset" is pressed for the Kalpa clean sample: it is saved as a normal run
(so reports, Recent analyses and the AI budget work as usual) and the demo seed's AI insights
are preloaded. Real uploads always run the full pipeline.

The snapshot is only used when it matches the sample file exactly (same SHA-256 hash) and was
written in the current snapshot format; otherwise the normal pipeline runs.
"""

from __future__ import annotations

import copy
import gzip
import pickle
from pathlib import Path

from config.settings import PROJECT_ROOT

DEMO_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "demo" / "kalpa_demo_snapshot.pkl.gz"
# Raise when the pickled classes change shape; the refresh script then rewrites the snapshot.
DEMO_SNAPSHOT_FORMAT = 1


def write_demo_snapshot(prep, output, path: Path | None = None) -> Path:
    """Save the prepared upload and the finished analysis (written by this app only)."""
    from database.schema import SCHEMA_VERSION
    path = Path(path or DEMO_SNAPSHOT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": DEMO_SNAPSHOT_FORMAT, "schema_version": SCHEMA_VERSION,
               "file_hash": prep.report.file_hash, "prep": prep, "output": output}
    with gzip.open(path, "wb", compresslevel=9) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_demo_snapshot(file_hash: str, path: Path | None = None) -> dict | None:
    """The snapshot for this exact file, or None (missing, other file, old format, unreadable)."""
    path = Path(path or DEMO_SNAPSHOT_PATH)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)          # our own file, shipped with the app
    except Exception:  # noqa: BLE001 - e.g. written by an older version of the app
        return None
    if payload.get("format") != DEMO_SNAPSHOT_FORMAT or payload.get("file_hash") != file_hash:
        return None
    return payload


def restore_demo_run(payload: dict, settings, db_path=None, session_id: str | None = None):
    """Save the snapshot as a new run in this database and preload the demo AI insights.
    Returns (prep, output). A database problem still returns the analysis (not saved)."""
    from ai.demo_seed import preload_demo_insights
    from analytics.engine import save_analysis
    from database import repository
    from database.connection import DatabaseError
    from database.housekeeping import housekeep

    prep = payload["prep"]
    output = copy.deepcopy(payload["output"])
    report = prep.report
    if getattr(output, "layout_key", None) is None and getattr(prep, "raw_df", None) is not None:
        from ai.mapping import layout_key     # U5: saved targets belong to the column layout
        output.layout_key = layout_key(prep.raw_df.columns)
    try:
        ds = repository.register_dataset(report.filename, report.file_hash, report.rows,
                                         report.columns, report.file_type, db_path=db_path)
        run_id = repository.create_run(ds.id, {"overrides": {}, "demo_snapshot": True},
                                       db_path=db_path, session_id=session_id)
        output.run_id, output.is_reupload, output.save_error = run_id, ds.is_reupload, None
        output.analysis.metadata["run_id"] = run_id
        repository.save_field_mappings(run_id, prep.mapping, db_path=db_path)
        repository.save_quality_log(run_id, output.clean.quality_log_df, db_path=db_path)
        repository.save_clean_records(run_id, output.clean.clean_df, db_path=db_path)
        save_analysis(output.analysis, run_id, db_path=db_path)
        repository.save_run_snapshot(run_id, pickle.dumps(output), db_path=db_path)
        repository.update_run_status(run_id, "complete", db_path=db_path)
        housekeep(db_path=db_path)
    except DatabaseError as exc:
        output.run_id, output.save_error = None, exc.user_message
        output.analysis.metadata["run_id"] = None
    try:
        preload_demo_insights(output, report, settings, db_path=db_path)
    except Exception:  # noqa: BLE001 - the saved insights are a convenience only
        pass
    return prep, output
