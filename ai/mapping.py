"""Gemini field mapping for the columns Python's rules could not map with confidence.

Order of authority: the user's choice > Gemini > Python rules. Python maps first; only the
remaining problem columns (uncertain or unmapped) go to Gemini, all in ONE call. Gemini never
calculates anything: it only says what a column means.

What is sent (never the data itself): the column names, up to AI_MAPPING_SAMPLE_VALUES short
example values per problem column, the mappings Python already made, the list of fields the
app understands, and the text of the file's data-dictionary sheet if it has one.

Reliability: same main model + Flash-Lite fallback on "high demand" as the insights; answers are
cached per column layout (same column names -> no new call); a daily limit
(AI_MAX_MAPPING_CALLS_PER_DAY) and, in public mode, a per-session limit. When AI is off, busy
or over budget, the app falls back to asking the user, with a short note saying why.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from ai.client import AIError
from config.fields import FIELD_BY_NAME, FIELDS
from config.settings import AI_MAPPING_SAMPLE_CHARS, AI_MAPPING_SAMPLE_VALUES

MAPPING_PROMPT_VERSION = "m1"
NOT_USED = "not_used"
PRIVACY_NOTE = "Only column names and a few sample values are sent to Google Gemini for mapping."


# --- What Gemini must return --------------------------------------------------------------------

class AIColumnMapping(BaseModel):
    column: str = Field(description="The problem column's name, exactly as given.")
    field: str = Field(description="One canonical field name from the list, or 'not_used'.")
    reason: str = Field(max_length=200, description="One short line: why this field.")


class AIMappingAnswer(BaseModel):
    mappings: list[AIColumnMapping]


SYSTEM_INSTRUCTIONS = """You map the columns of a marketing data file to the fields of an analytics app.
Python rules have already mapped the columns they were sure about. You receive only the columns
they could not map, with a few example values, plus the fields already used.

Rules:
- Return exactly one entry for EVERY problem column: a field name from canonical_fields, or "not_used".
- Never use a field that appears in already_mapped, and never use one field twice.
- Rates, ratios, percentages and averages (CTR, CPC, CPM, CPL, CPA, ROAS, ROI, AOV, conversion
  rate, "avg ...", "... per ...") are always "not_used": the app recalculates them from totals.
- Advertising spend is only media/ad spend. Total cost, product cost, COGS or operating cost are
  never spend ("not_used").
- A "Profit" or "Net profit" column may already subtract marketing cost: use "not_used" unless it
  is clearly gross profit before marketing costs.
- Use the data dictionary (if given) to understand what a column means.
- Columns that describe things the app has no field for (e.g. device, landing page, ad format,
  status) are "not_used".
- reason: one short line in plain English. Do not repeat the example values.
"""


def build_prompt(request: dict) -> str:
    return ("Map every problem column. Answer in the required JSON format.\n\n"
            + json.dumps(request, ensure_ascii=False, indent=1))


# --- Outcome ------------------------------------------------------------------------------------

@dataclass
class AIMappingOutcome:
    decisions: dict[str, tuple[str | None, str]] = field(default_factory=dict)  # column -> (field|None, reason)
    source: str = "none"          # "call", "cache" or "none"
    note: str = ""                # why AI was not used, or what happened (shown small in the UI)
    calls_made: int = 0
    model: str = ""
    rejected_items: list[str] = field(default_factory=list)   # answers that failed validation

    @property
    def used(self) -> bool:
        return bool(self.decisions)


def layout_key(columns) -> str:
    """Same column names (in any order) -> same key -> cached answers reused."""
    names = sorted(str(c) for c in columns)
    return hashlib.sha256(json.dumps([MAPPING_PROMPT_VERSION, names]).encode("utf-8")).hexdigest()


def problem_columns(mapping) -> list:
    """Columns the rules left open (and the user has not decided)."""
    return [m for m in mapping.columns if m.status in ("uncertain", "unmapped") and m.source != "user"]


def _samples(series: pd.Series) -> list[str]:
    values = []
    for value in series.dropna().astype(str).str.strip():
        if value and value not in values:
            values.append(value[:AI_MAPPING_SAMPLE_CHARS])
        if len(values) >= AI_MAPPING_SAMPLE_VALUES:
            break
    return values


def build_request(raw_df: pd.DataFrame, mapping, problems: list, dictionary: str | None) -> dict:
    """The complete, compact input for Gemini: no rows beyond 3 examples per problem column."""
    request = {
        "canonical_fields": [{"name": f.name, "label": f.label, "type": f.ftype} for f in FIELDS],
        "already_mapped": {c: f for f, c in mapping.active.items()},
        "problem_columns": [{"column": m.column, "examples": _samples(raw_df[m.column])}
                            for m in problems],
    }
    if dictionary:
        request["data_dictionary"] = dictionary
    return request


def validate_answer(text: str, problems: list, mapping) -> tuple[dict, list[str]]:
    """Parse and check Gemini's answer. Unknown columns or field names, fields already used and
    duplicates are rejected item by item (those columns are left for the user)."""
    answer = AIMappingAnswer.model_validate_json(text)        # raises ValidationError / ValueError
    wanted = {m.column for m in problems}
    taken = set(mapping.active)
    decisions: dict[str, tuple[str | None, str]] = {}
    rejected: list[str] = []
    for item in answer.mappings:
        name = item.field.strip()
        if item.column not in wanted or item.column in decisions:
            rejected.append(f"{item.column}: not a problem column")
        elif name == NOT_USED:
            decisions[item.column] = (None, item.reason.strip())
        elif name not in FIELD_BY_NAME:
            rejected.append(f"{item.column}: unknown field '{name}'")
        elif name in taken:
            rejected.append(f"{item.column}: '{name}' is already used")
        else:
            taken.add(name)
            decisions[item.column] = (name, item.reason.strip())
    return decisions, rejected


# --- The assistant used by the pipeline ------------------------------------------------------

class MappingAssistant:
    """Remembers mapping decisions per column layout and asks Gemini when needed.

    `client_factory()` builds the Gemini client only when a call is really needed (tests pass a
    fake). `session_id` is the browser session (public mode); `session_calls` how many mapping
    calls this session already made.
    """

    def __init__(self, settings, client_factory=None, *, session_id: str = "", session_calls: int = 0,
                 db_path=None, allow_call: bool = True, blocked_note: str = ""):
        self.settings = settings
        self.client_factory = client_factory
        self.session_id = session_id
        self.session_calls = session_calls
        self.db_path = db_path
        self.calls_made = 0          # calls made through this assistant (for the session counter)
        # False after one attempt for this upload failed: changing a dropdown must not trigger
        # another (budgeted) call. `blocked_note` repeats why AI was not used.
        self.allow_call = allow_call
        self.blocked_note = blocked_note

    # -- memory ---------------------------------------------------------------------------------
    def _memory(self, key: str) -> dict:
        from database import repository
        from database.connection import DatabaseError
        try:
            return repository.load_mapping_cache(key, self._user_scope(), db_path=self.db_path)
        except DatabaseError:
            return {"ai": {}, "user": {}}

    def _user_scope(self) -> str:
        return self.session_id if self.settings.public_mode else ""

    def remembered_choices(self, columns) -> dict[str, str | None]:
        """The user's earlier choices for this column layout (they win over AI and rules)."""
        return dict(self._memory(layout_key(columns))["user"])

    def remember_choices(self, columns, choices: dict[str, str | None]) -> None:
        from database import repository
        from database.connection import DatabaseError
        if not choices:
            return
        try:
            repository.save_mapping_cache(layout_key(columns), "user",
                                          {c: (f, "chosen by you") for c, f in choices.items()},
                                          session_id=self._user_scope(), db_path=self.db_path)
        except DatabaseError:
            pass

    # -- Gemini ---------------------------------------------------------------------------------
    def suggest(self, raw_df: pd.DataFrame, report, mapping) -> AIMappingOutcome:
        problems = problem_columns(mapping)
        if not problems:
            return AIMappingOutcome()
        key = layout_key(raw_df.columns)
        cached = self._memory(key)["ai"]
        if cached:
            wanted = {m.column for m in problems}
            return AIMappingOutcome(decisions={c: d for c, d in cached.items() if c in wanted},
                                    source="cache", note="AI mapping reused for this column layout "
                                    "(no new Gemini call).")
        if not self.allow_call:
            return AIMappingOutcome(note=self.blocked_note)
        reason = self._not_available()
        if reason:
            return AIMappingOutcome(note=reason)
        try:
            client = self.client_factory()
        except AIError as exc:
            return AIMappingOutcome(note=f"AI mapping not used: {exc.user_message}")
        request = build_request(raw_df, mapping, problems,
                                getattr(report, "data_dictionary", None))
        return self._call(client, request, problems, mapping, key)

    def _not_available(self) -> str:
        s = self.settings
        if self.client_factory is None or not s.ai_enabled:
            return "AI mapping is off, so the columns below need your choice."
        if not s.has_gemini_key or not s.gemini_model:
            return "AI mapping is not configured (no Gemini key or model), so the columns below need your choice."
        from ai.cache import today_start_utc
        from database import repository
        from database.connection import DatabaseError
        try:
            today = repository.count_mapping_calls(today_start_utc(), db_path=self.db_path)
        except DatabaseError:
            today = 0
        if today >= s.ai_max_mapping_calls_per_day:
            return (f"The daily AI mapping limit ({s.ai_max_mapping_calls_per_day} calls) has been "
                    "reached, so the columns below need your choice.")
        if s.public_mode and self.session_calls + self.calls_made >= s.ai_max_mapping_calls_per_session:
            return ("The AI mapping limit for this session has been reached, so the columns below "
                    "need your choice.")
        return ""

    def _log(self, key, client, success, error_type=None, response=None) -> None:
        from database import repository
        from database.connection import DatabaseError
        try:
            repository.save_mapping_call(layout_key=key, session_id=self.session_id or None,
                                         model=getattr(client, "model", ""), success=success,
                                         error_type=error_type,
                                         input_tokens=getattr(response, "input_tokens", None),
                                         output_tokens=getattr(response, "output_tokens", None),
                                         db_path=self.db_path)
        except DatabaseError:
            pass

    def _call(self, client, request, problems, mapping, key) -> AIMappingOutcome:
        outcome = AIMappingOutcome(model=getattr(client, "model", ""))
        prompt = build_prompt(request)
        for _attempt in range(2):          # the 2nd attempt only for the fallback model on 503
            outcome.calls_made += 1
            self.calls_made += 1
            try:
                response = client.generate(SYSTEM_INSTRUCTIONS, prompt, AIMappingAnswer)
            except AIError as exc:
                self._log(key, client, False, exc.kind)
                if exc.kind == "server" and getattr(client, "use_fallback", lambda: False)():
                    outcome.model = client.model
                    continue
                outcome.note = f"AI mapping not available ({exc.user_message}) The columns below need your choice."
                return outcome
            try:
                decisions, rejected = validate_answer(response.text, problems, mapping)
            except (ValidationError, ValueError):
                self._log(key, client, False, "invalid_response", response)
                outcome.note = ("Gemini's mapping answer could not be read, so the columns below "
                                "need your choice.")
                return outcome
            self._log(key, client, True, response=response)
            outcome.decisions, outcome.rejected_items, outcome.source = decisions, rejected, "call"
            outcome.model = getattr(client, "model", outcome.model)
            self._save(key, decisions)
            return outcome
        outcome.note = "AI mapping not available (Gemini is busy). The columns below need your choice."
        return outcome

    def _save(self, key: str, decisions: dict) -> None:
        from database import repository
        from database.connection import DatabaseError
        try:
            repository.save_mapping_cache(key, "ai", decisions, db_path=self.db_path)
        except DatabaseError:
            pass


# --- Badges -------------------------------------------------------------------------------------

BADGES = {
    "verified": "AI · verified",
    "check": "AI · not verified, please check",
    "failed": "AI mapping failed a check — please choose",
}


def contradicted_ai_columns(mapping, crosscheck) -> dict[str, str]:
    """AI-mapped columns that the file's own calculations prove wrong: {column: field}."""
    ai_cols = {m.field: m.column for m in mapping.columns if m.source == "ai" and m.field}
    out = {}
    for check in crosscheck.checks:
        for fname in check.contradicted:
            if fname in ai_cols:
                out[ai_cols[fname]] = fname
    return out


def badges(mapping, crosscheck) -> dict[str, str]:
    """Badge key per AI-involved column: verified / check / failed."""
    out = {}
    verified = crosscheck.verified_fields
    for m in mapping.columns:
        if m.source == "ai_rejected":
            out[m.column] = "failed"
        elif m.source == "ai" and m.field:
            if m.field in verified:
                out[m.column] = "verified"
            elif FIELD_BY_NAME[m.field].critical:
                out[m.column] = "check"
    return out

