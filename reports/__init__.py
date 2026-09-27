"""Report builders (PDF and Excel)."""

from datetime import datetime


def export_filename(stem: str, run_id, extension: str) -> str:
    """File name that ties an export to its analysis run and time, e.g.
    Marketing_Intelligence_Report_run12_2026-09-27_1015.pdf (no overwriting within a day)."""
    run = f"run{run_id}" if run_id else "unsaved"
    return f"{stem}_{run}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.{extension}"
