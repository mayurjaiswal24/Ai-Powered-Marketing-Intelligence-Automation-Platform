"""Prompt v4: structured test shift (a proposal, not verified), and the new marketing rules."""

import pytest
from pydantic import ValidationError

from ai.context_builder import build_evidence
from ai.evaluator import evaluate
from ai.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS
from ai.schemas import AIInsights
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from tests.test_ai import good_answer


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    result = run_pipeline(prepare(raw, report), db_path=tmp_path_factory.mktemp("v4") / "a.db").analysis
    return build_evidence(result, model="m")


def test_test_shift_pct_range(pack):
    answer = good_answer(pack)
    answer["recommendations"][0]["test_shift_pct"] = 15
    assert AIInsights.model_validate(answer).recommendations[0].test_shift_pct == 15
    answer["recommendations"][1].pop("test_shift_pct", None)
    assert AIInsights.model_validate(answer).recommendations[1].test_shift_pct is None
    for bad in (5, 25):
        answer["recommendations"][0]["test_shift_pct"] = bad
        with pytest.raises(ValidationError):
            AIInsights.model_validate(answer)


def test_test_shift_is_a_proposal_but_text_percentages_are_checked(pack):
    answer = good_answer(pack)
    rec = answer["recommendations"][0]
    rec["test_shift_pct"] = 15
    rec["text"] = "Move a small test share of budget from the weakest to the strongest campaign."
    checked, _ = evaluate(AIInsights.model_validate(answer).model_dump(), pack)
    first = next(r for r in checked["recommendations"] if r["text"] == rec["text"])
    assert first["status"] == "verified" and first["test_shift_pct"] == 15

    answer["recommendations"][0]["text"] = "Move 15% of budget from the weakest campaign."   # in text
    checked, _ = evaluate(AIInsights.model_validate(answer).model_dump(), pack)
    flagged = next(r for r in checked["recommendations"] if r["text"].startswith("Move 15%"))
    assert flagged["status"] == "unverified" and "15%" in flagged["warnings"][0]


def test_prompt_v4_rules():
    assert PROMPT_VERSION == "v4"
    text = " ".join(SYSTEM_INSTRUCTIONS.lower().split())
    for rule in ("already leads, so they can improve lead-to-conversion, not click-to-lead",
                 "compare the incident's length with the comparison window",
                 "earlier period", "prevent a repeat", "not how to \"recover\"", "test_shift_pct"):
        assert rule in text, rule
