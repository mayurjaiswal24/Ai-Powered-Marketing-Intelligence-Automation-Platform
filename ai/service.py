"""One AI run: evidence -> Gemini -> validation -> evaluation -> storage.

Called only when the user presses "Generate AI insights" (never automatically). Every call is
logged in `ai_runs` (time, fingerprint, model, success, tokens, latency). Phase 12 adds caching
and daily quota checks around this function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from ai.client import AIError, ai_error
from ai.context_builder import EvidencePack, build_evidence
from ai.evaluator import evaluate
from ai.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS, build_user_prompt
from ai.schemas import AIInsights


@dataclass
class AIRunResult:
    pack: EvidencePack
    insights: dict | None = None          # evaluated sections, ready to display / export
    evaluation: dict | None = None
    error: AIError | None = None
    calls_made: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    notes: list[str] = field(default_factory=list)
    from_cache: bool = False
    cached_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.insights is not None and self.error is None


def _log(run_id, pack, client, success, error_type=None, response=None, db_path=None) -> None:
    """Record the call; a database problem must never break the AI feature itself."""
    from database import repository
    from database.connection import DatabaseError
    try:
        repository.save_ai_run(run_id, fingerprint=pack.fingerprint, model=getattr(client, "model", ""),
                               prompt_version=PROMPT_VERSION, success=success, error_type=error_type,
                               input_tokens=getattr(response, "input_tokens", None),
                               output_tokens=getattr(response, "output_tokens", None),
                               latency_ms=getattr(response, "latency_ms", None), db_path=db_path)
    except DatabaseError:
        pass


def log_cache_hit(run_id, pack, model: str, db_path=None) -> None:
    """A reuse of saved insights: logged for transparency, never counted against the budget."""
    from database import repository
    from database.connection import DatabaseError
    try:
        repository.save_ai_run(run_id, fingerprint=pack.fingerprint, model=model,
                               prompt_version=PROMPT_VERSION, success=True, cache_hit=True,
                               db_path=db_path)
    except DatabaseError:
        pass


def generate_ai_insights(result, client, run_id: int | None = None, max_calls: int = 2,
                         db_path=None, log: bool = True) -> AIRunResult:
    """Ask Gemini to interpret the evidence. At most one extra call is made, and only when the
    answer was unreadable AND max_calls allows it. Quota and other errors are never retried."""
    pack = build_evidence(result, model=getattr(client, "model", ""))
    run = AIRunResult(pack=pack, model=getattr(client, "model", ""))
    prompt = build_user_prompt(pack.json_text)

    while True:
        if run.calls_made >= max_calls:
            run.error = run.error or ai_error("budget")
            return run
        run.calls_made += 1
        try:
            response = client.generate(SYSTEM_INSTRUCTIONS, prompt, AIInsights)
        except AIError as exc:
            if log:
                _log(run_id, pack, client, False, exc.kind, db_path=db_path)
            run.error = exc
            # "High demand" (503) on the main model: try the configured fallback model once,
            # if the call budget allows. Quota errors (429) are never retried.
            if exc.kind == "server" and run.calls_made < max_calls and                     getattr(client, "use_fallback", lambda: False)():
                run.notes.append(f"The main model was overloaded; the fallback model "
                                 f"{client.model} answered instead.")
                run.model = client.model
                continue
            return run
        run.input_tokens += response.input_tokens or 0
        run.output_tokens += response.output_tokens or 0
        try:
            parsed = AIInsights.model_validate_json(response.text)
        except (ValidationError, ValueError):
            if log:
                _log(run_id, pack, client, False, "invalid_response", response, db_path=db_path)
            run.error = ai_error("invalid_response")
            if run.calls_made < max_calls:
                run.notes.append("The first answer was not valid JSON; asked once more.")
                continue
            return run
        if log:
            _log(run_id, pack, client, True, response=response, db_path=db_path)
        run.error = None
        break

    checked, evaluation = evaluate(parsed.model_dump(), pack)
    run.insights, run.evaluation = checked, evaluation.as_dict()
    if log:
        from database import repository
        from database.connection import DatabaseError
        try:
            repository.save_ai_insights(run_id, pack.fingerprint,
                                        {"model": run.model, "prompt_version": PROMPT_VERSION,
                                         "sections": checked}, run.evaluation, db_path=db_path)
        except DatabaseError:
            run.notes.append("The AI result could not be saved to the database.")
    return run


def evidence_lookup(pack: EvidencePack, ids: list[str]) -> list[dict]:
    """The evidence items behind a statement, for 'show evidence' in the UI and reports."""
    return [pack.by_id[i].as_dict() for i in ids if i in pack.by_id]
