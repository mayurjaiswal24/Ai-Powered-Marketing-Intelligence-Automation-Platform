"""Time each pipeline step on the largest files (Phase 14 performance check).

    .venv\\Scripts\\python scripts\\benchmark.py [file ...]

Uses a temporary database and export folder (the app's own data is untouched). No AI calls.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.engine import run_analysis, save_analysis  # noqa: E402
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare  # noqa: E402
from database import repository  # noqa: E402
from processing.cleaner import clean_dataset  # noqa: E402
from reports.excel_report import export_workbook  # noqa: E402
from reports.pdf_report import export_pdf  # noqa: E402

DEFAULT_FILES = [SAMPLE_DIR / "marketing_clean.csv", SAMPLE_DIR / "marketing_messy.xlsx",
                 Path(r"real_test_file.xlsx")]


def bench(path: Path) -> dict[str, float]:
    times: dict[str, float] = {}

    def step(name, fn):
        start = time.perf_counter()
        value = fn()
        times[name] = time.perf_counter() - start
        return value

    with tempfile.TemporaryDirectory() as folder:
        db, exports = Path(folder) / "b.db", Path(folder) / "exports"
        raw, report = step("load file", lambda: load_raw(path))
        prep = step("profile + map + checks", lambda: prepare(raw, report))
        cleaned = step("clean", lambda: clean_dataset(raw, prep.mapping))

        def store():
            ds = repository.register_dataset(report.filename, report.file_hash, report.rows,
                                              report.columns, report.file_type, db_path=db)
            run_id = repository.create_run(ds.id, db_path=db)
            repository.save_clean_records(run_id, cleaned.clean_df, db_path=db)
            repository.save_quality_log(run_id, cleaned.quality_log_df, db_path=db)
            return run_id
        run_id = step("save to SQLite", store)
        analysis = step("analyse (KPIs, trends, anomalies)",
                        lambda: run_analysis(cleaned.clean_df, prep.validation, cleaned.quality_summary,
                                             dataset_name=report.filename, run_id=run_id))
        step("save analysis", lambda: save_analysis(analysis, run_id, db_path=db))
        step("PDF report", lambda: export_pdf(analysis, exports_dir=exports, db_path=db))
        step("Excel workbook", lambda: export_workbook(analysis, exports_dir=exports, db_path=db))
    times["rows"] = report.rows
    return times


def main(files: list[Path]) -> None:
    for path in files:
        if not path.exists():
            print(f"(skipped, not found: {path})")
            continue
        t = bench(path)
        rows = int(t.pop("rows"))
        analysis_total = sum(v for k, v in t.items() if k not in ("PDF report", "Excel workbook"))
        print(f"\n{path.name} ({rows:,} rows)")
        for name, seconds in t.items():
            print(f"  {name:<36}{seconds:6.2f} s")
        print(f"  {'upload -> analysis ready':<36}{analysis_total:6.2f} s")
        print(f"  {'total incl. PDF + Excel':<36}{sum(t.values()):6.2f} s")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]] or DEFAULT_FILES)
