"""Prompt v3 tests: fallback model, list sizes, 'weak' checks and impact ranking. FAKE client only."""

import json

import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError

from ai.client import AIResponse, classify_error
from ai.context_builder import build_evidence
from ai.evaluator import evaluate
from ai.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS
from ai.schemas import AIInsights
from ai.service import generate_ai_insights
from config.settings import load_settings
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from tests.test_ai import good_answer, quota_error


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    return run_pipeline(prepare(raw, report), db_path=tmp_path_factory.mktemp("v3") / "a.db").analysis


@pytest.fixture(scope="module")
def pack(analysis):
    return build_evidence(analysis, model="main-model")


class FallbackFake:
    """Main model overloaded (503) on the first call; the fallback answers."""

    def __init__(self, responses, model="main-model", fallback_model="fallback-model"):
        self.responses, self.model, self.fallback_model = list(responses), model, fallback_model
        self.calls = []

    def use_fallback(self):
        if not self.fallback_model:
            return False
        self.model, self.fallback_model = self.fallback_model, ""
        return True

    def generate(self, system, prompt, schema):
        self.calls.append(self.model)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return AIResponse(text=nxt, model=self.model, input_tokens=10, output_tokens=5)


def overloaded():
    return classify_error(genai_errors.ServerError(503, {"error": {
        "code": 503, "status": "UNAVAILABLE", "message": "This model is currently experiencing high demand."}}))


# --- Model fallback ----------------------------------------------------------------------------

def test_settings_read_fallback_model():
    s = load_settings({"GEMINI_MODEL": "gemini-3.5-flash", "GEMINI_FALLBACK_MODEL": "gemini-3.1-flash-lite"})
    assert (s.gemini_model, s.gemini_fallback_model) == ("gemini-3.5-flash", "gemini-3.1-flash-lite")


def test_fallback_answers_on_high_demand(analysis, pack, tmp_path):
    client = FallbackFake([overloaded(), json.dumps(good_answer(pack))])
    run = generate_ai_insights(analysis, client, max_calls=2, db_path=tmp_path / "f.db")
    assert run.ok and client.calls == ["main-model", "fallback-model"]
    assert run.model == "fallback-model" and "fallback" in run.notes[0]


def test_fallback_needs_budget_and_never_follows_quota(analysis, pack, tmp_path):
    one = FallbackFake([overloaded(), json.dumps(good_answer(pack))])
    run = generate_ai_insights(analysis, one, max_calls=1, db_path=tmp_path / "b.db")
    assert run.error.kind == "server" and one.calls == ["main-model"]
    quota = FallbackFake([classify_error(quota_error()), json.dumps(good_answer(pack))])
    run = generate_ai_insights(analysis, quota, max_calls=2, db_path=tmp_path / "q.db")
    assert run.error.kind == "quota" and quota.calls == ["main-model"]


# --- Schema ------------------------------------------------------------------------------------

def test_three_to_five_key_findings_and_recommendations(pack):
    answer = good_answer(pack)
    AIInsights.model_validate(answer)
    for key in ("key_findings", "recommendations"):
        too_few = dict(answer, **{key: answer[key][:2]})
        with pytest.raises(ValidationError):
            AIInsights.model_validate(too_few)
        too_many = dict(answer, **{key: answer[key] * 2})
        with pytest.raises(ValidationError):
            AIInsights.model_validate(too_many)


# --- Weak checks and ranking -------------------------------------------------------------------

def _incidents(pack):
    return [i for i in pack.items if i.type == "incident"]


def _impact_text(item):
    return item.facts["estimated impact"].split(" (")[0].rsplit(" ", 1)[0]     # e.g. "₹11.8 L"


def test_restated_key_finding_is_weak(pack):
    checked, ev = evaluate(AIInsights.model_validate(good_answer(pack)).model_dump(), pack)
    restated = checked["key_findings"][0]             # cites one "finding" item only
    assert restated["weak"] and "Restates" in restated["weak_reasons"][0]
    assert restated["status"] == "verified"          # still kept, just marked
    assert ev.weak >= 1


def test_recommendation_must_quote_incident_impact(pack):
    inc = _incidents(pack)[0]
    answer = good_answer(pack)
    answer["recommendations"][0] = {"text": f"Fix the issue behind {inc.title}.", "evidence_ids": [inc.id],
                                    "priority": "high", "metric_to_watch": "Revenue"}
    answer["recommendations"][1] = {"text": f"Fix it now: {_impact_text(inc)} is at stake.",
                                    "evidence_ids": [inc.id], "priority": "high", "metric_to_watch": "Revenue"}
    checked, _ = evaluate(AIInsights.model_validate(answer).model_dump(), pack)
    by_text = {r["text"]: r for r in checked["recommendations"]}
    assert by_text[f"Fix the issue behind {inc.title}."]["weak"]
    assert "rupee impact" in by_text[f"Fix the issue behind {inc.title}."]["weak_reasons"][0]
    assert not by_text[f"Fix it now: {_impact_text(inc)} is at stake."]["weak"]


def test_recommendations_ranked_by_incident_impact(pack):
    small, big = _incidents(pack)[-1], _incidents(pack)[0]            # incidents come ranked
    answer = good_answer(pack)
    answer["recommendations"] = [
        {"text": "No incident behind this.", "evidence_ids": [pack.items[0].id], "priority": "high",
         "metric_to_watch": "CPL"},
        {"text": f"Small one: {_impact_text(small)}.", "evidence_ids": [small.id], "priority": "low",
         "metric_to_watch": "CPL"},
        {"text": f"Big one: {_impact_text(big)}.", "evidence_ids": [big.id], "priority": "medium",
         "metric_to_watch": "Revenue"},
    ]
    checked, _ = evaluate(AIInsights.model_validate(answer).model_dump(), pack)
    order = [r["text"].split(":")[0] for r in checked["recommendations"]]
    assert order == ["Big one", "Small one", "No incident behind this."]


# --- Prompt v3 ---------------------------------------------------------------------------------

def test_prompt_v3_rules():
    assert PROMPT_VERSION >= "v3"            # the v3 rules stay in later versions
    text = " ".join(SYSTEM_INSTRUCTIONS.lower().split())
    for rule in ("rank recommendations by the rupee impact", "test shift", "between 10 and 20",
                 "decide whether to continue", "quote that incident's estimated impact",
                 "never cite a single \"finding\" item", "3-5 key findings"):
        assert rule in text, rule
