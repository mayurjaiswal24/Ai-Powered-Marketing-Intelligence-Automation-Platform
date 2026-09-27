"""Free-tier protection: fingerprinting, caching and call budgets (SPEC §14).

- Fingerprint = SHA-256 of canonical JSON (sorted keys) of the evidence pack + PROMPT_VERSION +
  model. Same analysis, same prompt, same model -> same fingerprint -> the saved answer is reused
  and NO API call is made.
- Budgets: AI_MAX_CALLS_PER_RUN (per analysis run) and AI_MAX_CALLS_PER_DAY (per UTC day),
  both counted from the ai_runs log. Cache hits are free and never count.
- Every attempt, including cache hits, is logged in ai_runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from ai.client import AIError, ai_error
from ai.prompts import PROMPT_VERSION

# A regenerate/retry can use at most this many calls (the second only for unreadable JSON).
from config.settings import AI_MAX_CALLS_PER_REQUEST as MAX_CALLS_PER_REQUEST  # noqa: E402


def compute_fingerprint(pack_json: str, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    canonical = json.dumps({"evidence": json.loads(pack_json), "prompt_version": prompt_version,
                            "model": model}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def today_start_utc() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")


@dataclass
class BudgetStatus:
    calls_today: int
    day_limit: int
    calls_this_run: int
    run_limit: int
    # Public mode only: calls made in this browser session and the per-session limit.
    calls_this_session: int = 0
    session_limit: int | None = None

    @property
    def remaining(self) -> int:
        left = min(self.day_limit - self.calls_today, self.run_limit - self.calls_this_run)
        if self.session_limit is not None:
            left = min(left, self.session_limit - self.calls_this_session)
        return max(0, left)

    @property
    def reason(self) -> str:
        """Why no call is allowed (empty when calls remain)."""
        if self.calls_today >= self.day_limit:
            return (f"The daily AI limit ({self.day_limit} calls, AI_MAX_CALLS_PER_DAY) has been "
                    "reached. Saved insights still work; new calls are possible tomorrow (UTC).")
        if self.session_limit is not None and self.calls_this_session >= self.session_limit:
            return (f"This demo allows {self.session_limit} AI calls per session "
                    "(AI_MAX_CALLS_PER_SESSION). Saved insights still work.")
        if self.calls_this_run >= self.run_limit:
            return (f"The AI limit for this analysis run ({self.run_limit} calls, "
                    "AI_MAX_CALLS_PER_RUN) has been reached. Saved insights still work.")
        return ""


def budget_status(settings, run_id: int | None, session_calls: int = 0, db_path=None,
                  session_total: int | None = None) -> BudgetStatus:
    """Calls used today and in this run, from the ai_runs log. If the run was not saved
    (run_id None), the calls made in this browser session are used instead.
    In public mode the calls of the whole browser session (`session_total`, defaulting to
    `session_calls`) are also limited by AI_MAX_CALLS_PER_SESSION."""
    from database import repository
    from database.connection import DatabaseError
    try:
        today = repository.count_ai_calls_since(today_start_utc(), db_path=db_path)
        this_run = (repository.count_ai_calls_for_run(run_id, db_path=db_path)
                    if run_id is not None else session_calls)
    except DatabaseError:
        today, this_run = 0, session_calls
    status = BudgetStatus(today, settings.ai_max_calls_per_day, this_run, settings.ai_max_calls_per_run)
    if getattr(settings, "public_mode", False):
        status.calls_this_session = session_calls if session_total is None else session_total
        status.session_limit = settings.ai_max_calls_per_session
    return status


def load_cached(fingerprint: str, db_path=None) -> dict | None:
    from database import repository
    from database.connection import DatabaseError
    try:
        return repository.load_latest_ai_insights(fingerprint, db_path=db_path)
    except DatabaseError:
        return None


def get_insights(result, settings, client_factory, *, run_id: int | None = None, force: bool = False,
                 session_calls: int = 0, db_path=None, allow_call: bool = True):
    """Saved insights if this exact evidence was interpreted before; otherwise (if allowed and
    within budget) one new Gemini call.

    `client_factory()` builds the Gemini client only when a call is really needed.
    `force=True` (the confirmed "Regenerate" button) skips the cache but still respects the budget.
    `allow_call=False` only looks in the cache (used when a page opens, and by the reports).
    Returns an AIRunResult (see ai/service.py).
    """
    from ai.context_builder import build_evidence
    from ai.service import AIRunResult, generate_ai_insights, log_cache_hit

    if not settings.ai_enabled and allow_call:
        raise ai_error("disabled")
    model = settings.gemini_model or ""
    pack = build_evidence(result, model=model)

    if settings.ai_cache_enabled and not force:
        cached = load_cached(pack.fingerprint, db_path=db_path)
        if cached is not None:
            if allow_call:
                log_cache_hit(run_id, pack, model, db_path=db_path)
            stored = cached["insights"]
            return AIRunResult(pack=pack, insights=stored.get("sections"), evaluation=cached["evaluation"],
                               model=stored.get("model", model),
                               prompt_version=stored.get("prompt_version", PROMPT_VERSION),
                               from_cache=True, cached_at=cached["created_at"])
    if not allow_call:
        return None

    budget = budget_status(settings, run_id, session_calls, db_path=db_path)
    if budget.remaining <= 0:
        return AIRunResult(pack=pack, model=model, error=AIError("budget", budget.reason))
    client = client_factory()
    return generate_ai_insights(result, client, run_id=run_id, db_path=db_path,
                                max_calls=min(MAX_CALLS_PER_REQUEST, budget.remaining))
