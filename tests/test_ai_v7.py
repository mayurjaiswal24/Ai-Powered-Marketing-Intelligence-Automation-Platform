"""Prompt v7 (owner review of the v6 answer): pacing must reach a recommendation, a short incident is
never the primary cause of a longer trend, and causal-certainty wording is banned. FAKE AI client only."""

import json

import pytest

from ai.context_builder import build_evidence
from ai.evaluator import evaluate
from ai.prompts import BANNED_CAUSAL_PHRASES, PROMPT_VERSION, SYSTEM_INSTRUCTIONS
from ai.service import generate_ai_insights
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from tests.test_ai import FakeClient, good_answer


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    return run_pipeline(prepare(raw, report), db_path=tmp_path_factory.mktemp("v7") / "a.db").analysis


@pytest.fixture(scope="module")
def pack(analysis):
    return build_evidence(analysis, model="m")


def _of_type(pack, kind):
    return next(i for i in pack.items if i.type == kind)


def _short_incident(pack):
    """The Google Ads tracking outage: 8-9 Sep 2026 (2 days)."""
    return next(i for i in pack.items if i.type == "incident" and i.facts["period"].startswith("8 Sep 2026"))


def _recs(*texts_ids):
    return [{"text": t, "evidence_ids": ids, "priority": "medium", "metric_to_watch": "CAC"} for t, ids in texts_ids]


def test_prompt_v7_rules():
    assert PROMPT_VERSION >= "v7"             # the v7 rules stay in later versions
    text = " ".join(SYSTEM_INSTRUCTIONS.lower().split())
    for rule in ("at least one recommendation must refer to it directly", "primary or main cause",
                 "contributed to part of the decline; the remaining drop is unexplained",
                 "consistent with", "may indicate", "one contributing factor"):
        assert rule in text, rule
    for phrase in BANNED_CAUSAL_PHRASES:
        assert f'"{phrase}"' in text, phrase
    assert "optimise targeting" in text and "leverage social media" in text      # earlier list kept


# --- 1. Pacing must reach a recommendation ---------------------------------------------------------

def test_no_recommendation_on_pacing_is_weak(pack):
    kpi = _of_type(pack, "kpi")
    assert _of_type(pack, "pacing").facts["status"] == "Underspending"
    checked, ev = evaluate({"recommendations": _recs(("Review CPL each week.", [kpi.id]),
                                                     ("Check lead quality with sales.", [kpi.id]))}, pack)
    top = checked["recommendations"][0]
    assert top["weak"] and any("budget pacing" in r for r in top["weak_reasons"]) and ev.weak == 1
    assert not checked["recommendations"][1]["weak"]


@pytest.mark.parametrize("text, cite_pacing", [
    ("Move the unused budget from underspending into Paid Search, where the next conversion costs least.", False),
    ("Plan the next month's budget from the latest month's figures.", True),
])
def test_recommendation_on_pacing_passes(pack, text, cite_pacing):
    kpi, pacing = _of_type(pack, "kpi"), _of_type(pack, "pacing")
    ids = [pacing.id] if cite_pacing else [kpi.id]
    checked, ev = evaluate({"recommendations": _recs(("Review CPL each week.", [kpi.id]), (text, ids))}, pack)
    assert ev.weak == 0 and not any(r["weak"] for r in checked["recommendations"])


def test_on_pace_needs_no_pacing_recommendation(pack):
    pacing = _of_type(pack, "pacing")
    original = pacing.facts["status"]
    pacing.facts["status"] = "On Pace"
    try:
        _, ev = evaluate({"recommendations": _recs(("Review CPL each week.", [_of_type(pack, "kpi").id]))}, pack)
    finally:
        pacing.facts["status"] = original
    assert ev.weak == 0


# --- 2. A short incident is never the primary cause of a longer trend -------------------------------

def test_short_incident_as_primary_cause_is_weak(pack):
    inc = _short_incident(pack)
    item = {"text": "The four-week revenue fall was primarily driven by the 2-day Google Ads outage.",
            "evidence_ids": [inc.id], "validation_step": "Compare daily revenue inside and outside the outage."}
    checked, _ = evaluate({"hypotheses": [item]}, pack)
    h = checked["hypotheses"][0]
    assert h["weak"] and any("2-day incident" in r for r in h["weak_reasons"])


def test_short_incident_as_partial_factor_passes(pack):
    inc = _short_incident(pack)
    item = {"text": "The 2-day Google Ads outage contributed to part of the decline; the remaining drop is "
                    "unexplained.", "evidence_ids": [inc.id],
            "validation_step": "Compare daily revenue inside and outside the outage."}
    checked, _ = evaluate({"hypotheses": [item]}, pack)
    assert not checked["hypotheses"][0]["weak"]


def test_long_incident_may_be_called_the_main_factor(pack):
    long = next(i for i in pack.items if i.type == "incident" and i.facts["period"].startswith("3 Aug 2026"))
    item = {"text": "The East region's August drop was mainly in lead-to-conversion.", "evidence_ids": [long.id]}
    checked, _ = evaluate({"key_findings": [item]}, pack)
    assert not checked["key_findings"][0]["weak"]                  # 28 days: not a "short" incident


# --- 3. Causal-certainty wording --------------------------------------------------------------------

@pytest.mark.parametrize("phrase", BANNED_CAUSAL_PHRASES)
def test_causal_certainty_wording_is_weak(pack, phrase):
    kpi = _of_type(pack, "kpi")
    item = {"text": f"The spend mix {phrase} the Video weakness.", "evidence_ids": [kpi.id]}
    checked, _ = evaluate({"performance_concerns": [item]}, pack)
    c = checked["performance_concerns"][0]
    assert c["weak"] and any(phrase in r for r in c["weak_reasons"])


def test_hedged_wording_and_validation_steps_pass(pack):
    kpi = _of_type(pack, "kpi")
    item = {"text": "The low click-to-lead rate is consistent with form friction and may indicate one "
                    "contributing factor.", "evidence_ids": [kpi.id],
            "validation_step": "Run an A/B test; a higher rate for the short form confirms the idea."}
    checked, _ = evaluate({"hypotheses": [item]}, pack)
    assert not checked["hypotheses"][0]["weak"]                     # only the statement text is checked


def test_fake_client_answer_following_v7_has_no_weak_items(analysis, pack, tmp_path):
    pacing, inc = _of_type(pack, "pacing"), _short_incident(pack)
    ans = good_answer(pack)
    ans["key_findings"] = ans["key_findings"][1:3] + [
        {"text": "The 2-day Google Ads outage contributed to part of the decline; the remaining drop is unexplained.",
         "evidence_ids": [inc.id, _of_type(pack, "kpi").id]}]
    ans["recommendations"][1] = {"text": "Put the unused budget from the latest month's underspending into a small "
                                         "Paid Search test.", "evidence_ids": [pacing.id], "priority": "medium",
                                 "metric_to_watch": "Budget utilisation"}
    client = FakeClient([json.dumps(ans)])
    run = generate_ai_insights(analysis, client, db_path=tmp_path / "f.db")
    assert run.ok and len(client.calls) == 1
    assert "as evidenced by" in client.calls[0][0]                  # the rule is in the system prompt
    recs = run.insights["recommendations"]
    assert not any("budget pacing" in r for rec in recs for r in rec.get("weak_reasons", []))
    outage = next(i for i in run.insights["key_findings"] if inc.id in i["evidence_ids"])
    assert not outage["weak"]
