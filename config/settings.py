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


def _load_streamlit_secrets() -> None:
    """Deployment (Streamlit Community Cloud): the app's Secrets are a TOML file whose
    root-level keys Streamlit turns into environment variables when it loads them. Load them
    now, before the settings below are read. Locally there is no secrets file: nothing happens."""
    try:
        from streamlit.runtime.secrets import secrets_singleton
        if not secrets_singleton.load_if_toml_exists():
            return
        # Streamlit copies only text and numbers into the environment; a TOML true/false
        # (e.g. AI_ENABLED = true) would be skipped, so copy those too.
        for key, value in secrets_singleton.items():
            if isinstance(value, bool):
                os.environ.setdefault(key, "true" if value else "false")
    except Exception:  # noqa: BLE001 - a broken secrets file must not stop the app starting
        pass


_load_streamlit_secrets()

# --- Product and creator details (shown in the sidebar, About page, PDF and Excel) --------------
APP_NAME = "Marketing Intelligence Platform"
APP_VERSION = "2.0"
CREATOR_NAME = "Mayur Jaiswal"
CREATOR_LINKEDIN = "https://www.linkedin.com/in/mayur-jaiswal-679501213"
CREATOR_TITLE = "PGDM (Marketing & AI), Dr. D. Y. Patil B-School, Pune"
CREATOR_EDUCATION_PREVIOUS = "BBA (Marketing & HR), Jagran Lakecity University"
CREATOR_FOCUS = "Growth & Strategy Analysis · Marketing Analytics · Applied AI · Power BI"
CREATOR_OPEN_TO = ("Open to Growth & Strategy Analyst and Marketing Analytics Roles · "
                   "On-site, Hybrid or Remote")
# One signature line, identical in the PDF (cover + every footer) and the Excel title block.
REPORT_SIGNATURE = f"Prepared with {APP_NAME} v{APP_VERSION} · Created by {CREATOR_NAME}"

# --- Profiling & field-mapping thresholds (fixed business rules, not per-deployment) -------------
# A header whose fuzzy similarity to a known synonym reaches this score (0-100), AND whose data
# has the right type, is mapped as "high-confidence". Below it, the mapping is only suggested.
MAPPING_HIGH_CONFIDENCE = 88
# Candidates scoring below this are not even suggested.
MAPPING_MIN_CANDIDATE = 75   # raised from 65: weaker matches produced nonsense (e.g. Discount -> Country)
# Value pattern (U2): a column with an unknown name is recognised from its values (e.g. "Meta",
# "Google Ads" -> Platform) only when at least this share of its most common values is known
# vocabulary (config/knowledge/value_vocabulary.csv), with at least 2 different known values.
VALUE_PATTERN_MIN_SHARE = 0.8
# Two category labels this similar (0-100) are treated as spelling variants ("Linkdin" ~ "LinkedIn").
LABEL_SIMILARITY = 90

# --- Cross-check with the file's own calculated columns (ingestion/crosscheck.py) ---
CROSSCHECK_SAMPLE_ROWS = 500       # rows compared: a random sample spread over the whole file
CROSSCHECK_SAMPLE_SEED = 14        # fixed seed, so the same file always gives the same result
CROSSCHECK_MIN_ROWS = 5            # fewer comparable rows than this -> no verdict
CROSSCHECK_MIN_MATCH_SHARE = 0.90  # share of rows that must agree (files round their own ratios)
CROSSCHECK_REL_TOL = 0.02          # agree within 2% ...
CROSSCHECK_ABS_TOL = 0.011         # ... or within 0.011 (a value rounded to 2 decimals)

# --- Plausibility warnings before analysis (ingestion/plausibility.py). Warn, never block. ---
PLAUSIBLE_MAX_CTR_PCT = 50.0            # overall CTR above this is very unusual for ads
PLAUSIBLE_MIN_LEAD_TO_CONV_PCT = 0.1    # fewer than 1 in 1,000 leads converting is suspicious
PLAUSIBLE_MAX_ROAS = 50.0               # overall ROAS above 50x is rare outside synthetic data
# Share of non-missing values that must parse as a number/date for a column to get that type.
TYPE_DETECTION_SHARE = 0.9
# Columns with more distinct labels than this are not checked for inconsistent spellings.
MAX_LABELS_FOR_CLUSTERING = 200
# A column that is at least this share empty is flagged as a probable junk column.
MOSTLY_EMPTY_SHARE = 0.95

# --- Cleaning rules ------------------------------------------------------------------------------
# Indian convention: 05/01/2026 means 5 January 2026. Set False for US-style files.
DATE_DAYFIRST = True
# "remove" drops exact duplicate rows (logged); "flag" keeps them and marks dq_flags instead.
DUPLICATE_POLICY = "remove"
# Outliers are only FLAGGED, never removed. A value is flagged if it lies beyond
# Q3 + k x IQR (or below Q1 - k x IQR) within its channel. k = 3 flags only extreme values.
OUTLIER_IQR_MULTIPLIER = 3.0
# Row-level CPL is too noisy with very few leads (1 lead -> CPL = whole day's spend), so CPL
# outliers are only checked on rows with at least this many leads.
OUTLIER_MIN_LEADS = 5

# --- Anomaly detection (weekly, robust) -----------------------------------------------------------
# Each entity's week is compared with the median of its previous ANOMALY_BASELINE_WEEKS full weeks,
# after removing the movement shared by the rest of the business that week (seasonality).
ANOMALY_BASELINE_WEEKS = 8
# At least this many full weeks of history are needed before a week can be judged.
ANOMALY_MIN_HISTORY_WEEKS = 4
# Robust z-score (distance from the median in MAD units) needed to call a week unusual.
# 3.5 is the common rule of thumb; tuned on the sample data (Phase 07): 4.0 found the same
# 10/10 planted anomalies as 3.5 but halved the unexplained flags (22 -> 12).
ANOMALY_SCORE_THRESHOLD = 4.0
# ... AND the change must be at least this big (%), so tiny-but-steady metrics are not flagged
# for small wobbles ("do not call something an anomaly merely because it is numerically large").
ANOMALY_MIN_PCT_CHANGE = 30.0
# The MAD is floored at this share of the baseline: a very steady history must not turn a
# 3% change into a "huge" score.
ANOMALY_SCALE_FLOOR = 0.05
# Minimum weekly volume (baseline median) before a metric is checked, to avoid tiny-number noise.
# For ratio metrics the volume is the denominator (e.g. CPL needs enough leads).
ANOMALY_MIN_VOLUME = {
    "spend": 10_000,           # rupees per week
    "revenue": 50_000,         # rupees per week
    "clicks": 200,
    "cpc": 200,                # clicks per week
    "cpl": 20,                 # leads per week
    "lead_qualification_rate": 30,   # leads per week
    "lead_to_conversion_rate": 50,   # leads per week
}
# Scores at or above this (and changes of at least 50%) are labelled "high" severity.
ANOMALY_HIGH_SCORE = 6.0

# Daily tracking-outage rule: spend continues but clicks collapse.
OUTAGE_BASELINE_DAYS = 28          # compare with the median of the previous 28 days
OUTAGE_CLICK_SHARE = 0.05          # clicks below 5% of normal ...
OUTAGE_MIN_SPEND_SHARE = 0.25      # ... while spend is at least 25% of normal
OUTAGE_MIN_BASELINE_CLICKS = 20    # only for entities that normally get >= 20 clicks a day

# --- Analytics rules ----------------------------------------------------------------------------
TREND_WINDOW_DAYS = 28             # "last 4 weeks vs the previous 4 weeks" for trends and growth
TREND_FLAT_BAND_PCT = 5.0          # a change smaller than this (%) is described as "flat"
ROLLING_AVERAGE_WEEKS = 4          # rolling averages on weekly trends
MIN_MONTHS_FOR_SEASONALITY = 12    # full months needed before a month-of-year index is shown
MIN_SPEND_SHARE_FOR_RANKING = 3.0  # % of spend; smaller campaigns have unreliable ratios
MIN_SPEND_SHARE_FOR_MOVERS = 2.0   # % of spend a campaign needs to be a "biggest riser/faller"
CONCENTRATION_SHARE = 40.0         # % of spend in one channel worth pointing out
MIN_USABLE_DATE_SHARE = 0.5        # below this share of readable dates, trends are switched off
HEADER_SEARCH_ROWS = 30            # how far down an upload we look for the real header row
ANOMALY_WINDOWS = (1, 3)           # single weeks and sustained 3-week stretches

# --- Profitability (U4, analytics/profitability.py) ----------------------------------------------
# ROAS within +/-10% of the break-even ROAS is "Near break-even"; at least 1.10x break-even is
# "Profitable"; below 0.90x is "Loss-making".
PROFIT_NEAR_BAND = 0.10
# Allowed range for the optional, session-only "Assumed gross margin (%)" (blank by default).
ASSUMED_MARGIN_MIN_PCT = 1.0
ASSUMED_MARGIN_MAX_PCT = 99.0

# --- Targets vs actual (U5, analytics/targets.py) --------------------------------------------------
# A metric within 10% of its target on the wrong side is "Within 10%"; further away is "Off Target".
TARGET_TOLERANCE = 0.10
# Monthly spend is a plan to hit, not "more is better": within +/-5% of the target is "On Target",
# within +/-10% is "Within 10%", beyond that "Over Target" or "Under Target".
SPEND_ON_TARGET_BAND = 0.05

# --- Budget pacing and forecast (U6, analytics/pacing.py, analytics/forecast.py) ----------------
# Spend to date within +/-5% of the planned budget to date is "On Pace"; above = "Overspending",
# below = "Underspending".
PACING_BAND = 0.05
PACING_RECENT_DAYS = 7             # "recent utilisation" = last 7 days' spend / planned budget
FORECAST_MIN_WEEKS = 26            # full weeks of history needed before a forecast is shown
FORECAST_HORIZON_WEEKS = 8         # default weeks ahead (the page allows 4-8)
FORECAST_MIN_HORIZON_WEEKS = 4
FORECAST_MAX_HORIZON_WEEKS = 8
FORECAST_BACKTEST_WEEKS = 12       # methods are compared on the last 12 weeks (rolling origin)
FORECAST_RANGE_PERCENTILES = (10, 90)   # shaded range = 10th-90th percentile of past errors
FORECAST_COVERAGE_WEEKS = 26      # the range check replays the last 26 weeks
ANOMALY_MIN_ENTITIES_FOR_COMMON_MOVE = 3   # entities needed to estimate "the typical change"

# --- Scenario planner (U7, analytics/response_curves.py) --------------------------------------
# Data check per paid channel (full weeks): enough weeks with spend, and spend that varied enough
# (coefficient of variation, and highest week / lowest week) to see how results respond to it.
RESPONSE_MIN_WEEKS = 20
RESPONSE_MIN_CV = 0.15
RESPONSE_MIN_MAX_RATIO = 1.5
RESPONSE_MIN_FIT_WEEKS = 12        # weeks left after incident weeks are removed, to fit a curve
RESPONSE_RECENT_WEEKS = 8          # baseline spend, curve anchor, AOV and CPA = the last 8 full weeks
# How each channel is priced. "auction" (the default) gets a fitted curve; "performance" channels
# are paid per result, so their spend follows results and is never curve-fitted (keys lower-case).
CHANNEL_PRICING = {"affiliate": "performance", "affiliates": "performance",
                   "affiliate network": "performance"}
PERFORMANCE_CAP_MULTIPLE = 1.2     # performance channel: at most 1.2 x its best week's conversions
RESPONSE_BOOTSTRAP_SAMPLES = 200   # curve refits on resampled weeks, for the 10th-90th percentile range
RESPONSE_BOOTSTRAP_SEED = 7        # fixed seed: the same data always gives the same range
RESPONSE_RANGE_PERCENTILES = (10, 90)
# Confidence in a curve: High needs R-squared >= 0.5 on >= 30 weeks; Medium >= 0.25 on >= 20 weeks.
RESPONSE_CONFIDENCE_HIGH = (0.5, 30)
RESPONSE_CONFIDENCE_MEDIUM = (0.25, 20)
# Guardrails: a scenario's weekly spend stays within 0.5 x the lowest and 1.3 x the highest weekly
# spend seen for that channel (outside that the curve is a guess).
SCENARIO_GUARDRAIL_LOW = 0.5
SCENARIO_GUARDRAIL_HIGH = 1.3
SCENARIO_MOVE_PCT = 15             # "Move Budget" default: 15% of the FROM channel's weekly spend
SCENARIO_SLIDER_PCT = 50           # advanced sliders: each channel -50% to +50%
SCENARIO_SUGGEST_MAX_CHANGE = 0.30 # "Suggest Allocation": at most +/-30% per channel

# --- Basic view (U8, utils/plain_language.py) ----------------------------------------------------
# A key number's colour: its target status when a target is set; otherwise the last N days compared
# with the N days before. Worse by less than WATCH % = green, WATCH to BAD % = yellow, more = red.
BASIC_COMPARE_DAYS = 30
BASIC_WATCH_PCT = 5.0
BASIC_BAD_PCT = 15.0
BASIC_TOP_ALERTS = 3               # problem incidents on "Top Alerts" and in the Summary Report
BASIC_TOP_CAMPAIGNS = 3            # best and weakest campaigns on "What's Working and What's Not"
BASIC_MIN_SPEND_SHARE = 1.0        # campaigns under 1% of spend are too small to call best or weakest

# --- AI (Gemini) ---------------------------------------------------------------------------------
# The evidence pack sent to Gemini is capped at this many characters; lowest-priority evidence is
# trimmed first. Keeps each call small (free-tier friendly) and focused.
AI_CONTEXT_MAX_CHARS = 21_000    # 15,000 -> 18,000 for incidents (Phase 12 review) -> 21,000 for U9 planning items
AI_TOP_CAMPAIGNS = 5               # best and weakest campaigns included as evidence
AI_MAX_INCIDENTS = 10              # top incidents (by estimated rupee impact) included as evidence
AI_TIMEOUT_SECONDS = 90            # give up on a Gemini call after this long
AI_TEMPERATURE = 0.2               # low: we want careful, repeatable interpretation
AI_MAX_CALLS_PER_REQUEST = 2       # one request may use a 2nd call: fallback model or unreadable answer
AI_NUMBER_TOLERANCE = 0.01         # relative tolerance when checking AI numbers against evidence
# Prompt v7: an incident this short (days) is never the primary cause of a trend over the 28-day
# comparison window; the evaluator marks such a claim as weak.
AI_SHORT_INCIDENT_DAYS = 14
# AI field mapping (ai/mapping.py): what Gemini may see about a problem column.
AI_MAPPING_SAMPLE_VALUES = 3       # at most this many example values per problem column
AI_MAPPING_SAMPLE_CHARS = 40       # each example value is cut to this length
AI_MAPPING_DICTIONARY_MAX_CHARS = 4_000   # data-dictionary sheet text sent with the columns
# Public (deployed demo) mode: runs of a browser session are deleted after this many hours.
PUBLIC_RUN_MAX_AGE_HOURS = 24

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
    # Used automatically (once) when the main model answers "high demand" (HTTP 503).
    gemini_fallback_model: str = ""
    ai_enabled: bool = False
    ai_max_calls_per_run: int = 3
    ai_max_calls_per_day: int = 15
    ai_cache_enabled: bool = True
    # App data lives in data/app/ so the project folder stays clean (git-ignored).
    database_path: str = "data/app/marketing_intelligence.db"
    # Housekeeping: only the newest N analysis runs are kept (records, snapshots, cached AI
    # insights and export folders of older runs are deleted when a new run is saved).
    keep_last_runs: int = 20
    # Public demo mode (true in deployment): each browser session sees only its own runs,
    # smaller uploads, per-session AI limits, and runs are deleted after 24 hours.
    public_mode: bool = False
    max_upload_mb_public: int = 10
    ai_max_calls_per_session: int = 2          # insight calls per browser session (public mode)
    ai_max_mapping_calls_per_day: int = 10     # Gemini field-mapping calls per UTC day
    ai_max_mapping_calls_per_session: int = 2  # field-mapping calls per browser session (public)
    # Folder for generated PDF/Excel files (relative paths are inside the project folder).
    exports_dir: str = "exports"
    # Uploads above this size are refused with a friendly message.
    max_upload_mb: int = 50
    # Above this many rows we warn that analysis may be slow (the file still loads).
    large_row_warning: int = 200_000
    # U8: the view a new session starts in, "Professional" (everything) or "Basic" (plain words).
    default_view: str = "Professional"

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def upload_limit_mb(self) -> int:
        """The upload size limit that applies now (smaller in public mode)."""
        return min(self.max_upload_mb, self.max_upload_mb_public) if self.public_mode else self.max_upload_mb


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from an environment mapping (defaults to the real environment)."""
    env = os.environ if environ is None else environ
    defaults = Settings()
    return Settings(
        gemini_api_key=_parse_str(env.get("GEMINI_API_KEY"), defaults.gemini_api_key),
        gemini_model=_parse_str(env.get("GEMINI_MODEL"), defaults.gemini_model),
        gemini_fallback_model=_parse_str(env.get("GEMINI_FALLBACK_MODEL"), defaults.gemini_fallback_model),
        ai_enabled=_parse_bool(env.get("AI_ENABLED"), defaults.ai_enabled),
        ai_max_calls_per_run=_parse_int(env.get("AI_MAX_CALLS_PER_RUN"), defaults.ai_max_calls_per_run),
        ai_max_calls_per_day=_parse_int(env.get("AI_MAX_CALLS_PER_DAY"), defaults.ai_max_calls_per_day),
        ai_cache_enabled=_parse_bool(env.get("AI_CACHE_ENABLED"), defaults.ai_cache_enabled),
        database_path=_parse_str(env.get("DATABASE_PATH"), defaults.database_path),
        exports_dir=_parse_str(env.get("EXPORTS_DIR"), defaults.exports_dir),
        keep_last_runs=_parse_int(env.get("KEEP_LAST_RUNS"), defaults.keep_last_runs, minimum=1),
        public_mode=_parse_bool(env.get("PUBLIC_MODE"), defaults.public_mode),
        max_upload_mb_public=_parse_int(env.get("MAX_UPLOAD_MB_PUBLIC"), defaults.max_upload_mb_public,
                                        minimum=1),
        ai_max_calls_per_session=_parse_int(env.get("AI_MAX_CALLS_PER_SESSION"),
                                            defaults.ai_max_calls_per_session),
        ai_max_mapping_calls_per_day=_parse_int(env.get("AI_MAX_MAPPING_CALLS_PER_DAY"),
                                                defaults.ai_max_mapping_calls_per_day),
        ai_max_mapping_calls_per_session=_parse_int(env.get("AI_MAX_MAPPING_CALLS_PER_SESSION"),
                                                    defaults.ai_max_mapping_calls_per_session),
        max_upload_mb=_parse_int(env.get("MAX_UPLOAD_MB"), defaults.max_upload_mb, minimum=1),
        large_row_warning=_parse_int(env.get("LARGE_ROW_WARNING"), defaults.large_row_warning, minimum=1),
        default_view=_parse_view(env.get("DEFAULT_VIEW"), defaults.default_view),
    )


VIEWS = ("Basic", "Professional")


def _parse_view(value: str | None, default: str) -> str:
    """'basic' / 'Professional' (any case) -> the view name; anything else -> the default."""
    text = (value or "").strip().capitalize()
    return text if text in VIEWS else default


settings = load_settings()
