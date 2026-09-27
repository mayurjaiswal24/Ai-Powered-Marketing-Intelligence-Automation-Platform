"""Storage housekeeping: delete one analysis, or keep only the newest KEEP_LAST_RUNS.

Deleting a run removes everything that belongs to it:
  - its database records (clean rows, quality log, mappings, analysis tables, snapshot, report
    records and cached AI insights - all linked to the run with ON DELETE CASCADE),
  - its PDF and Excel files (as recorded in the database) and then its export folder
    `<EXPORTS_DIR>/<run_id>/` once it is empty,
  - the dataset row (file name + hash) once no run uses it any more.

One thing is deliberately kept: the AI call log (`ai_runs`: time, model, tokens - no content).
Its link to the run is cleared instead, because the daily AI budget counts those calls; deleting
them would let the app spend more of the free-tier quota than allowed.
"""

from __future__ import annotations

from pathlib import Path

from config.settings import PROJECT_ROOT, settings
from database.repository import session


def exports_root(exports_dir: str | Path | None = None) -> Path:
    p = Path(exports_dir if exports_dir is not None else settings.exports_dir)
    return p if p.is_absolute() else PROJECT_ROOT / p


def delete_analysis(run_id: int, db_path=None, exports_dir: str | Path | None = None) -> bool:
    """Delete one analysis run and its files. Returns False if the run did not exist."""
    run_id = int(run_id)
    with session(db_path) as conn:
        if conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
            return False
        report_files = [row[0] for row in conn.execute(
            "SELECT file_path FROM reports WHERE run_id = ?", (run_id,))]
        conn.execute("UPDATE ai_runs SET run_id = NULL WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))      # cascades to the rest
        conn.execute("DELETE FROM datasets WHERE id NOT IN (SELECT dataset_id FROM runs)")
    _delete_files(report_files, exports_root(exports_dir))
    return True


def apply_retention(keep: int | None = None, db_path=None,
                    exports_dir: str | Path | None = None) -> list[int]:
    """Keep the newest `keep` runs (default KEEP_LAST_RUNS); delete older ones. Returns the
    deleted run ids, oldest first."""
    keep = max(1, int(keep if keep is not None else settings.keep_last_runs))
    with session(db_path) as conn:
        old = [row[0] for row in conn.execute(
            "SELECT id FROM runs ORDER BY id DESC LIMIT -1 OFFSET ?", (keep,))]
    for run_id in sorted(old):
        delete_analysis(run_id, db_path=db_path, exports_dir=exports_dir)
    return sorted(old)


def _delete_files(report_files: list[str], root: Path) -> None:
    """Delete the run's recorded PDF/Excel files, then its folder once it is empty.
    Only files the database recorded are touched, and never anything outside the exports
    folder (a safety check against a wrong path)."""
    root = root.resolve()
    folders = set()
    for name in report_files:
        try:
            target = Path(name).resolve()
        except OSError:
            continue
        if root not in target.parents:
            continue
        target.unlink(missing_ok=True)
        folders.add(target.parent)
    for folder in folders:
        if folder != root and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
