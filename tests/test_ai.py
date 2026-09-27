"""Phase 11 tests: evidence pack, schema, evaluator and client errors. FAKE client only - the
real Gemini API is never called in tests."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError

import config.settings as config_settings
import database.connection as dbc
from ai.client import AIError, AIResponse, GeminiClient, classify_error, client_from_settings
from ai.context_builder import build_evidence
from ai.evaluator import evaluate, extract_numbers
from ai.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS
from ai.schemas import AIInsights
from ai.service import generate_ai_insights
from config.settings import AI_CONTEXT_MAX_CHARS, Settings
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from database import repository as repo

ROOT = Path(__file__).resolve().parent.parent


class FakeClient:
    """Stands in for GeminiClient. Each call returns (or raises) the next scripted response."""

    def __init__(self, responses, model="fake-model"):
        self.responses = list(responses)
        self.model = model
        self.calls = []

    def generate(self, system, prompt, schema):
        self.calls.append((system, prompt, schema))
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return AIResponse(text=nxt, model=self.model, input_tokens=1000, output_tokens=400, latency_ms=5)


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    return run_pipeline(prepare(raw, report), db_path=tmp_path_factory.mktemp("db") / "a.db").analysis


@pytest.fixture(scope="module")
def pack(analysis):
    return build_evidence(analysis)


def good_answer(pack) -> dict:
    kpi = next(i for i in pack.items if i.type == "kpi")
    finding = next(i for i in pack.items if i.type == "finding")
    spend, roas = kpi.facts["Spend"], kpi.facts["ROAS"]
    return {
        "executive_summary": {"text": f"Spend was {spend} with a ROAS of {roas}.", "evidence_ids": [kpi.id]},
        "key_findings": [
            {"text": finding.facts["finding"], "evidence_ids": [finding.id]},
            {"text": "ROAS was 9.99x across the period.", "evidence_ids": [kpi.id]},     # wrong number
            {"text": "Something unsupported.", "evidence_ids": ["E99"]},                  # fake ID
        ],
        "performance_concerns": [{"text": f"CPL stands at {kpi.facts['CPL']}.", "evidence_ids": [kpi.id]}],
        "hypotheses": [{"text": "The mix may favour cheaper channels.", "evidence_ids": [kpi.id],
                        "validation_step": "Compare channel CPL month by month."}],
        "investigation_areas": [{"text": "Review the weakest channel.", "evidence_ids": [finding.id]}],
        "recommendations": [{"text": "Shift budget to the best channel.", "evidence_ids": [finding.id],
                             "priority": "high", "metric_to_watch": "ROAS"},
                            {"text": "Review the weakest campaign's landing page.", "evidence_ids": [kpi.id],
                             "priority": "medium", "metric_to_watch": "Click-to-lead rate"},
                            {"text": "Check lead quality with the sales team.", "evidence_ids": [kpi.id],
                             "priority": "low", "metric_to_watch": "Lead-to-conversion rate"}],
        "limitations": [{"text": "Outliers were flagged.", "evidence_ids": []}],       # no evidence
    }


# --- Evidence pack -----------------------------------------------------------------------------

def test_evidence_pack_is_compact_and_identified(pack, analysis):
    assert pack.size <= AI_CONTEXT_MAX_CHARS
    ids = [i.id for i in pack.items]
    assert ids == [f"E{n:02d}" for n in range(1, len(ids) + 1)]
    types = {i.type for i in pack.items}
    assert {"kpi", "data_quality", "measurement", "finding", "channel", "funnel", "trend", "incident",
            "campaign", "segment"} <= types
    email = next(i for i in pack.items if i.title == "Channel: Email")
    assert email.facts["channel type"].startswith("owned")
    measurement = next(i for i in pack.items if i.type == "measurement")
    assert any("upper-funnel" in v for v in measurement.facts.values())
    incident = next(i for i in pack.items if i.type == "incident")
    assert "estimated impact" in incident.facts
    body = json.loads(pack.json_text)
    assert set(body) == {"dataset", "evidence", "not_available"}
    assert "2025-10-01" not in pack.json_text            # no raw rows / raw dates
    assert pack.by_id["E01"].facts["ROAS"] == analysis.kpis["roas"].formatted_compact


def test_evidence_trimmed_by_priority(analysis):
    small = build_evidence(analysis, max_chars=3000)
    assert small.size <= 3000 and small.dropped_items > 0
    assert small.items[0].type == "kpi"                  # highest priority survives
    assert "incident" not in {i.type for i in small.items[:3]}


def test_fingerprint_changes_with_model_not_with_rebuild(analysis):
    a, b = build_evidence(analysis, model="m1"), build_evidence(analysis, model="m1")
    assert a.fingerprint == b.fingerprint
    assert build_evidence(analysis, model="m2").fingerprint != a.fingerprint


def test_context_size_on_10k_dataset(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_data
    df = generate_data.generate_clean(seed=7, row_target=10_000)
    assert 9_000 <= len(df) <= 10_000
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    raw, report = load_raw(path)
    result = run_pipeline(prepare(raw, report), db_path=tmp_path / "b.db").analysis
    big = build_evidence(result)
    assert big.size <= AI_CONTEXT_MAX_CHARS


def test_no_revenue_pack_says_what_is_missing(tmp_path):
    raw, report = load_raw(SAMPLE_DIR / "marketing_no_revenue.csv")
    result = run_pipeline(prepare(raw, report), db_path=tmp_path / "n.db").analysis
    p = build_evidence(result)
    assert "Revenue" not in p.by_id["E01"].facts and "ROAS" not in p.by_id["E01"].facts
    assert any("no revenue field" in n for n in p.not_available)


# --- Prompt and schema -------------------------------------------------------------------------

def test_prompt_is_dataset_agnostic():
    assert PROMPT_VERSION
    for word in ("kalpa", "course", "enrol", "learning", "student"):   # no dataset-specific words
        assert word not in SYSTEM_INSTRUCTIONS.lower()
    assert "evidence" in SYSTEM_INSTRUCTIONS and "hypotheses" in SYSTEM_INSTRUCTIONS
    assert "optimise targeting" in SYSTEM_INSTRUCTIONS            # named as NOT allowed


def test_schema_validation(pack):
    ok = AIInsights.model_validate(good_answer(pack))
    assert ok.recommendations[0].priority == "high"
    bad = good_answer(pack)
    bad["recommendations"][0]["priority"] = "urgent"
    with pytest.raises(ValidationError):
        AIInsights.model_validate(bad)
    missing = good_answer(pack)
    del missing["hypotheses"][0]["validation_step"]
    with pytest.raises(ValidationError):
        AIInsights.model_validate(missing)


# --- Evaluator ---------------------------------------------------------------------------------

@pytest.mark.parametrize("text, value, kind", [
    ("₹4.2 Cr", 4.2e7, "money"), ("₹8.5 L", 8.5e5, "money"), ("₹12,34,567", 1234567, "money"),
    ("38.5%", 38.5, "percent"), ("3.45x", 3.45, "ratio"), ("2,77,378 leads", 277378, "plain"),
    ("-64%", 64, "percent"),
])
def test_extract_numbers(text, value, kind):
    n = extract_numbers(text)[0]
    assert n.value == pytest.approx(value) and n.kind == kind


def test_extract_numbers_skips_non_metrics():
    assert extract_numbers("over 4 weeks in 2026 from 9 Feb, see E03") == []


def test_evaluator_drops_fake_ids_and_flags_wrong_numbers(pack):
    checked, ev = evaluate(AIInsights.model_validate(good_answer(pack)).model_dump(), pack)
    assert ev.dropped == 2                                   # E99 item + item with no evidence
    reasons = " ".join(d["reason"] for d in ev.dropped_items)
    assert "E99" in reasons and "no valid evidence ID" in reasons
    findings = checked["key_findings"]
    assert len(findings) == 2
    assert findings[0]["status"] == "verified"
    assert findings[1]["status"] == "unverified" and "9.99x" in findings[1]["warnings"][0]
    assert checked["executive_summary"]["status"] == "verified"     # ₹7.3 Cr and 4.65x match E01
    assert checked["limitations"] == []
    assert ev.unverified == 1 and ev.kept == ev.verified + ev.unverified


def test_evaluator_tolerates_rounding(pack, analysis):
    kpi = pack.by_id["E01"]
    spend = analysis.kpis["spend"].value / 1e7          # e.g. 7.44 (crore), shown as "₹7.4 Cr"
    ctr = analysis.kpis["ctr"].value
    answer = good_answer(pack)
    answer["key_findings"][0] = {"text": f"Spend was about ₹{spend:.2f} Cr and CTR {ctr:.2f}%.",
                                 "evidence_ids": ["E01"]}
    checked, _ = evaluate(AIInsights.model_validate(answer).model_dump(), pack)
    assert checked["key_findings"][0]["status"] == "verified", kpi.facts


# --- Client and service ------------------------------------------------------------------------

def quota_error():
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": "Resource has been exhausted",
                                                    "status": "RESOURCE_EXHAUSTED"}})


@pytest.mark.parametrize("exc, kind", [
    (quota_error(), "quota"),
    (genai_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid.",
                                              "status": "INVALID_ARGUMENT"}}), "invalid_key"),
    (genai_errors.ClientError(403, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}), "invalid_key"),
    (genai_errors.ClientError(404, {"error": {"code": 404, "message": "models/x is not found", "status": "NOT_FOUND"}}), "model_not_found"),
    (genai_errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}), "server"),
    (httpx.ConnectTimeout("timed out"), "network"),
    (RuntimeError("boom"), "unknown"),
])
def test_error_classification(exc, kind):
    err = classify_error(exc)
    assert err.kind == kind
    assert "Traceback" not in err.user_message and "RESOURCE_EXHAUSTED" not in err.user_message


def test_quota_error_is_friendly_and_not_retried(analysis, tmp_path):
    db = tmp_path / "q.db"
    client = FakeClient([classify_error(quota_error()), json.dumps({})])
    run = generate_ai_insights(analysis, client, max_calls=3, db_path=db)
    assert run.error is not None and run.error.kind == "quota"
    assert "usage limit" in run.error.user_message
    assert len(client.calls) == 1 and run.insights is None           # no retry on quota
    assert repo.count_ai_calls_since("2000-01-01", db_path=db) == 1   # the failed call is logged


def test_invalid_json_retried_once_within_budget(analysis, pack, tmp_path):
    client = FakeClient(["not json at all", json.dumps(good_answer(pack))])
    run = generate_ai_insights(analysis, client, max_calls=2, db_path=tmp_path / "r.db")
    assert run.ok and len(client.calls) == 2 and run.notes
    once = FakeClient(["{broken", json.dumps(good_answer(pack))])
    run1 = generate_ai_insights(analysis, once, max_calls=1, db_path=tmp_path / "r.db")
    assert run1.error.kind == "invalid_response" and len(once.calls) == 1


def test_successful_run_is_evaluated_and_stored(analysis, pack, tmp_path):
    db = tmp_path / "s.db"
    client = FakeClient([json.dumps(good_answer(pack))])
    run = generate_ai_insights(analysis, client, run_id=None, db_path=db)
    assert run.ok and run.evaluation["dropped"] == 2 and run.input_tokens == 1000
    stored = repo.load_latest_ai_insights(run.pack.fingerprint, db_path=db)
    assert stored["insights"]["prompt_version"] == PROMPT_VERSION
    assert stored["evaluation"]["unverified"] == 1
    system, prompt, schema = client.calls[0]
    assert schema is AIInsights and run.pack.json_text in prompt


def test_client_configuration_errors():
    with pytest.raises(AIError) as info:
        GeminiClient("", "some-model")
    assert info.value.kind == "missing_key"
    with pytest.raises(AIError) as info:
        GeminiClient("key", "")
    assert info.value.kind == "missing_model"
    with pytest.raises(AIError) as info:
        client_from_settings(Settings(ai_enabled=False))
    assert info.value.kind == "disabled"


def test_key_never_in_repr():
    s = Settings(gemini_api_key="secret-xyz", ai_enabled=True, gemini_model="m")
    assert "secret-xyz" not in repr(s)


# --- Exports and UI ----------------------------------------------------------------------------

def test_pdf_includes_labelled_ai_output(analysis, pack, tmp_path):
    from pypdf import PdfReader
    from reports.pdf_report import AI_NOT_GENERATED, generate_pdf
    run = generate_ai_insights(analysis, FakeClient([json.dumps(good_answer(pack))]), log=False)
    path = tmp_path / "ai.pdf"
    generate_pdf(analysis, path, ai=run.insights)
    text = "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)
    assert AI_NOT_GENERATED not in text
    assert "HYPOTHESIS" in text and "How to validate" in text and "Evidence: E" in text
    assert "contains unverified figures" in text


@pytest.fixture
def app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)


def _run_to_ai_page(at):
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    at.sidebar.radio(key="page").set_value("AI Insights").run()
    return at


def test_ai_page_when_disabled(app, monkeypatch):
    monkeypatch.setattr(config_settings, "settings", Settings(ai_enabled=False))
    at = _run_to_ai_page(app)
    assert not at.exception
    assert any("turned off" in i.value for i in at.info)


def test_ai_page_with_fake_client(app, monkeypatch, pack):
    import ai.client
    monkeypatch.setattr(config_settings, "settings",
                        Settings(ai_enabled=True, gemini_api_key="test-key-123", gemini_model="fake-model"))
    fake = FakeClient([json.dumps(good_answer(pack))])
    monkeypatch.setattr(ai.client, "client_from_settings", lambda settings: fake)
    at = _run_to_ai_page(app)
    assert any("AI-generated interpretation" in i.value for i in at.info)
    at.button(key="generate_ai").click().run()
    assert not at.exception and not at.error
    assert len(fake.calls) == 1
    labels = {m.label: m.value for m in at.metric}
    assert labels["Dropped (no evidence)"] == "2" and labels["Unverified figures"] == "1"
    rendered = " ".join(str(getattr(e, "value", "")) for e in at.main)
    assert "test-key-123" not in rendered                  # the key is never shown


def test_ai_page_quota_error_keeps_app_working(app, monkeypatch):
    import ai.client
    monkeypatch.setattr(config_settings, "settings",
                        Settings(ai_enabled=True, gemini_api_key="k", gemini_model="fake-model"))
    monkeypatch.setattr(ai.client, "client_from_settings",
                        lambda settings: FakeClient([classify_error(quota_error())]))
    at = _run_to_ai_page(app)
    at.button(key="generate_ai").click().run()
    assert not at.exception
    assert any("usage limit" in e.value for e in at.error)
    at.sidebar.radio(key="page").set_value("Executive Overview").run()
    assert not at.exception and len(at.metric) >= 4        # the rest of the app is unaffected
