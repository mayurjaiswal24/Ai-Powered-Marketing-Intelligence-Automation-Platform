"""Opening the SQLite database safely.

SQLite needs no server: the whole database is one file (DATABASE_PATH, default
`data/app/marketing_intelligence.db` inside the project folder; the folder is created on first
use). On Streamlit Community Cloud this file is wiped on restart, so the app must always work
from an empty database. Old runs are pruned by database/housekeeping.py (KEEP_LAST_RUNS).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from config.settings import PROJECT_ROOT, settings


class DatabaseError(Exception):
    """A database problem. `user_message` is safe to show in the UI."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def get_db_path(path: str | Path | None = None) -> Path:
    """The database file: an explicit path, else DATABASE_PATH (relative to the project folder)."""
    p = Path(path) if path is not None else Path(settings.database_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def friendly_message(error: Exception) -> str:
    """Translate SQLite's technical errors into plain English."""
    text = str(error).lower()
    if "unable to open" in text:
        return ("The app's database could not be opened, so results cannot be saved or loaded. "
                "Please try again later, or ask the app owner to check the storage folder.")
    if "readonly" in text or "read-only" in text:
        return ("The app's database is read-only, so results could not be saved. Please ask the "
                "app owner to check the storage permissions.")
    if "locked" in text or "busy" in text:
        return ("The database is busy (another process is using it). Please wait a moment "
                "and try again.")
    if "malformed" in text or "not a database" in text:
        return ("The app's database file is damaged. The app owner can delete it, and it will be "
                "recreated empty.")
    if "disk" in text and "full" in text:
        return "There is not enough disk space to save the results."
    return "A database error stopped the results from being saved or loaded. Please try again."


@contextmanager
def connect(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the database, commit on success, roll back on failure, always close.

    Any SQLite error is re-raised as DatabaseError with a friendly message (the technical
    error is kept as the cause for debugging).
    """
    db_path = get_db_path(path)
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")   # enforce links between tables
        yield conn
        conn.commit()
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        if conn is not None:
            conn.rollback()
        raise DatabaseError(friendly_message(exc)) from exc
    finally:
        if conn is not None:
            conn.close()
