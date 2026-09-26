"""Application settings.

Every setting comes from an environment variable. Locally these are read from a `.env` file
(see `.env.example`); in deployment they come from Streamlit secrets / the host environment.
If a variable is missing or unreadable, a safe default is used so the app always starts.

Alternative considered: a pydantic-settings class. Plain dataclass + helpers keeps the
dependency list short and is easier to read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load `.env` from the project root if it exists. Variables already set in the real
# environment win (override=False), so deployment settings are never replaced by a stray file.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# --- Profiling & field-mapping thresholds (fixed business rules, not per-deployment) -------------
# A header whose fuzzy similarity to a known synonym reaches this score (0-100), AND whose data
# has the right type, is mapped as "high-confidence". Below it, the mapping is only suggested.
MAPPING_HIGH_CONFIDENCE = 88
# Candidates scoring below this are not even suggested.
MAPPING_MIN_CANDIDATE = 65
# Two category labels this similar (0-100) are treated as spelling variants ("Linkdin" ~ "LinkedIn").
LABEL_SIMILARITY = 90
# Share of non-missing values that must parse as a number/date for a column to get that type.
TYPE_DETECTION_SHARE = 0.9
# Columns with more distinct labels than this are not checked for inconsistent spellings.
MAX_LABELS_FOR_CLUSTERING = 200
# A column that is at least this share empty is flagged as a probable junk column.
MOSTLY_EMPTY_SHARE = 0.95

_TRUE_WORDS = {"1", "true", "yes", "y", "on"}
_FALSE_WORDS = {"0", "false", "no", "n", "off"}


def _parse_bool(raw: str | None, default: bool) -> bool:
    """Read 'true'/'false'-style text. Anything unrecognised falls back to the default."""
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return default


def _parse_int(raw: str | None, default: int, minimum: int = 0) -> int:
    """Read a whole number. Blank, non-numeric or below-minimum values use the default."""
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value >= minimum else default


def _parse_str(raw: str | None, default: str) -> str:
    if raw is None or not raw.strip():
        return default
    return raw.strip()


@dataclass(frozen=True)
class Settings:
    # repr=False keeps the key out of logs and error messages if the object is ever printed.
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = ""
    ai_enabled: bool = False
    ai_max_calls_per_run: int = 3
    ai_max_calls_per_day: int = 15
    ai_cache_enabled: bool = True
    database_path: str = "marketing_intelligence.db"
    # Uploads above this size are refused with a friendly message.
    max_upload_mb: int = 50
    # Above this many rows we warn that analysis may be slow (the file still loads).
    large_row_warning: int = 200_000

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from an environment mapping (defaults to the real environment)."""
    env = os.environ if environ is None else environ
    defaults = Settings()
    return Settings(
        gemini_api_key=_parse_str(env.get("GEMINI_API_KEY"), defaults.gemini_api_key),
        gemini_model=_parse_str(env.get("GEMINI_MODEL"), defaults.gemini_model),
        ai_enabled=_parse_bool(env.get("AI_ENABLED"), defaults.ai_enabled),
        ai_max_calls_per_run=_parse_int(env.get("AI_MAX_CALLS_PER_RUN"), defaults.ai_max_calls_per_run),
        ai_max_calls_per_day=_parse_int(env.get("AI_MAX_CALLS_PER_DAY"), defaults.ai_max_calls_per_day),
        ai_cache_enabled=_parse_bool(env.get("AI_CACHE_ENABLED"), defaults.ai_cache_enabled),
        database_path=_parse_str(env.get("DATABASE_PATH"), defaults.database_path),
        max_upload_mb=_parse_int(env.get("MAX_UPLOAD_MB"), defaults.max_upload_mb, minimum=1),
        large_row_warning=_parse_int(env.get("LARGE_ROW_WARNING"), defaults.large_row_warning, minimum=1),
    )


settings = load_settings()
