"""Prompt v8 (owner review of the v7 answer): a reallocation linked to an underspent month must deploy the
unused budget where the next conversion costs least, or say it addresses efficiency, not the underspend.
Plus the display fix: inline evidence tags ("[E47]") are removed from AI text on screen, in the PDF and in
Excel, while the stored answer keeps them. FAKE AI client only."""

import copy
import json
import re
from unittest.mock import MagicMock

import pytest
from openpyxl import load_workbook
from pypdf import PdfReader

from ai.context_builder import build_evidence
from ai.evaluator import evaluate
from ai.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS
from ai.service import generate_ai_insights
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from reports.excel_report import generate_workbook
from reports.pdf_report import generate_pdf
from tests.test_ai import FakeClient, good_answer
from utils.formatting import strip_evidence_tags

BRACKET_TAG = re.compile(r"[\[(]\s*E\d{2,3}")


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    return run_pipeline(prepare(raw, report), db_path=tmp_path_factory.mktemp("v8") / "a.db").analysis


@pytest.fixture(scope="module")
def pack(analysis):
    return build_evidence(analysis, model="m")


def _of_type(pack, kind):
    return next(i for i in pack.items if i.type == kind)


def _rec(text, ids):
    return {"text": text, "evidence_ids": ids, "priority": "high", "metric_to_watch": "CAC"}


def _underspend_reasons(checked):
    return [r for rec in checked["recommendations"] for r in rec.get("weak_reasons", []) if "underspend" in r]


def test_prompt_v8_rule():
    assert PROMPT_VERSION == "v8"
    text = " ".join(SYSTEM_INSTRUCTIONS.lower().split())
    assert "does not fix an underspend" in text
    assert "where the next conversion costs least" in text and "raising total spend" in text
    assert "addresses efficiency, not the underspend itself" in text


# --- Underspend logic --------------------------------------------------------------------------------

def test_pacing_linked_reallocation_only_is_weak(pack):
    """The v7 answer's top recommendation: tied to the underspend, but only moves money Video -> Paid Search."""
    pacing, cost = _of_type(pack, "pacing"), _of_type(pack, "marginal_cost")
    assert pacing.facts["status"] == "Underspending"
    rec = _rec("Address the September underspending of 80.7% of plan by reallocating budget. Shift a small test "
               "budget FROM Video, where the next conversion costs ₹33,046, TO Paid Search, where it costs ₹9,521.",
               [pacing.id, cost.id])
    checked, ev = evaluate({"recommendations": [rec]}, pack)
    assert checked["recommendations"][0]["weak"] and _underspend_reasons(checked) and ev.weak == 1


@pytest.mark.parametrize("text", [
    # (a) the unused budget goes to the channel with the lowest cost of the next conversion
    "Deploy the unused ₹12.2 L of September budget (80.7% of plan) into Paid Search, where the next "
    "conversion costs least (₹9,521), and shift a small test budget away from Video.",
    # (b) the reallocation is said to address efficiency, not the underspend
    "September spend was 80.7% of plan. Shift a small test budget from Video to Paid Search; this move "
    "improves efficiency, not the underspend itself.",
    "Reallocate a small test budget from Video to Paid Search (80.7% of plan used); the move does not "
    "address the underspend, only efficiency.",
])
def test_underspend_rule_met_passes(pack, text):
    pacing, cost = _of_type(pack, "pacing"), _of_type(pack, "marginal_cost")
    checked, _ = evaluate({"recommendations": [_rec(text, [pacing.id, cost.id])]}, pack)
    assert not _underspend_reasons(checked)


def test_unused_budget_without_next_conversion_cost_is_weak(pack):
    """(a) needs the lowest cost of the next conversion, not just any channel."""
    pacing = _of_type(pack, "pacing")
    rec = _rec("Move the unused budget from the underspending month into Video.", [pacing.id])
    checked, _ = evaluate({"recommendations": [rec]}, pack)
    assert _underspend_reasons(checked)


def test_reallocation_not_linked_to_pacing_is_not_checked(pack):
    cost = _of_type(pack, "marginal_cost")
    rec = _rec("Shift a small test budget from Video, where the next conversion costs ₹33,046, to Paid Search.",
               [cost.id])
    checked, _ = evaluate({"recommendations": [rec, _rec("Plan the next month from 80.7% of plan.",
                                                          [_of_type(pack, "pacing").id])]}, pack)
    assert not _underspend_reasons(checked)


def test_rule_applies_only_when_underspending(pack):
    pacing, cost = _of_type(pack, "pacing"), _of_type(pack, "marginal_cost")
    original = pacing.facts["status"]
    pacing.facts["status"] = "Overspending"
    try:
        checked, _ = evaluate({"recommendations": [_rec("Pacing is off plan: shift budget from Video to Paid "
                                                        "Search.", [pacing.id, cost.id])]}, pack)
    finally:
        pacing.facts["status"] = original
    assert not _underspend_reasons(checked)


def test_fake_client_answer_following_v8(analysis, pack, tmp_path):
    pacing, cost = _of_type(pack, "pacing"), _of_type(pack, "marginal_cost")
    ans = good_answer(pack)
    ans["recommendations"][0] = _rec("Deploy the unused September budget (80.7% of plan) into Paid Search, where "
                                     "the next conversion costs least, raising total spend.", [pacing.id, cost.id])
    client = FakeClient([json.dumps(ans)])
    run = generate_ai_insights(analysis, client, db_path=tmp_path / "f.db")
    assert run.ok and len(client.calls) == 1
    assert "does not fix an underspend" in " ".join(client.calls[0][0].lower().split())
    assert not _underspend_reasons(run.insights)


# --- Display: no inline evidence tags on screen, in the PDF or in Excel --------------------------------

TAGGED = {
    "key_findings": [{"text": "Paid Search spend rose 12% [E16], while CPL held (E03).",
                      "evidence_ids": ["E03", "E16"]}],
    "hypotheses": [{"text": "The fall may indicate form friction [E01, E08, E20].", "evidence_ids": ["E01", "E08", "E20"],
                    "validation_step": "Compare the form's completion rate by week [E08]."}],
    "recommendations": [{"text": "Deploy the unused budget into Paid Search [E47].", "evidence_ids": ["E46", "E47"],
                         "priority": "high", "metric_to_watch": "CAC"}],
}


def test_strip_evidence_tags():
    assert strip_evidence_tags("Spend fell 12% [E47].") == "Spend fell 12%."
    assert strip_evidence_tags("CPL rose (E16), while [E01, E08, E20] ROAS held.") == "CPL rose, while ROAS held."
    assert strip_evidence_tags("Keep (see note) and [1] as written.") == "Keep (see note) and [1] as written."
    assert strip_evidence_tags(None) is None


def test_rendered_text_has_no_evidence_tags_but_stored_data_does(analysis, tmp_path, monkeypatch):
    ai = copy.deepcopy(TAGGED)
    # PDF
    generate_pdf(analysis, tmp_path / "ai.pdf", ai=ai)
    pdf_text = "\n".join(p.extract_text() for p in PdfReader(str(tmp_path / "ai.pdf")).pages)
    assert "Deploy the unused budget into Paid Search." in " ".join(pdf_text.split())
    assert not BRACKET_TAG.search(pdf_text)
    # Excel: text and validation step clean; the Evidence IDs column still lists the IDs
    generate_workbook(analysis, tmp_path / "ai.xlsx", ai=ai)
    cells = [c.value for row in load_workbook(tmp_path / "ai.xlsx")["AI_Insights"].iter_rows() for c in row
             if isinstance(c.value, str)]
    assert "Deploy the unused budget into Paid Search." in cells and "E46, E47" in cells
    assert not any(BRACKET_TAG.search(c) for c in cells)
    # Screen (AI Insights page item)
    import dashboard.layout as layout
    fake_st = MagicMock()
    monkeypatch.setattr(layout, "st", fake_st)
    for key, items in ai.items():
        for i, item in enumerate(items):
            layout._ai_item(item, "Label", None, lambda pack, ids: [], f"{key}_{i}")
    shown = [str(c.args[0]) for name in ("markdown", "caption") for c in getattr(fake_st, name).call_args_list]
    assert any("Deploy the unused budget into Paid Search." in s for s in shown)
    assert not any(BRACKET_TAG.search(s) for s in shown)
    # The stored answer is unchanged: the tags are still there for the evaluator's citation checks
    assert ai == TAGGED and "[E47]" in ai["recommendations"][0]["text"]


# --- Capacity of channels paid per result (Affiliate: "at most about ₹69,290 more a week") ------------

def _capacity_reasons(item):
    return [r for r in item.get("weak_reasons", []) if "stated capacity" in r]


def test_capacity_check_with_the_fake_client(analysis, pack, tmp_path):
    """(a) within capacity - not flagged; (b) above it - weak with both amounts; (c) an auction channel
    has no capacity, so the check never fires on it."""
    pacing, cost, video = _of_type(pack, "pacing"), _of_type(pack, "marginal_cost"), "E16"
    assert "at most about ₹69,290 more a week" in cost.facts["Affiliate"]
    assert "at most" not in cost.facts["Paid Search"]
    ans = good_answer(pack)
    ans["recommendations"] = [
        _rec("Put up to ₹60,000 more a week into Affiliate, within its stated room of ₹69,290 more a week.",
             [cost.id]),                                                                        # (a)
        _rec("Deploy the unused September budget (80.7% of plan) into Affiliate, where the next conversion "
             "costs ₹8,688, the lowest among paid channels.", [pacing.id, cost.id]),              # (b)
        _rec("Deploy the unused September budget (80.7% of plan) into Paid Search, where the next conversion "
             "costs least (₹9,521).", [pacing.id, cost.id, video]),                               # (c)
    ]
    client = FakeClient([json.dumps(ans)])
    run = generate_ai_insights(analysis, client, db_path=tmp_path / "c.db")
    assert run.ok and len(client.calls) == 1
    recs = run.insights["recommendations"]
    within = next(r for r in recs if r["text"].startswith("Put up to"))
    over = _capacity_reasons(next(r for r in recs if "into Affiliate" in r["text"] and r is not within))
    assert not _capacity_reasons(within)
    assert over and "₹12.2 L over the month" in over[0] and "₹69,290 more a week" in over[0]
    paid_search = next(r for r in recs if "into Paid Search" in r["text"])
    assert not _capacity_reasons(paid_search)


@pytest.mark.parametrize("text, extra, flagged", [
    ("Move ₹2 L a week into Affiliate.", {}, True),                                   # weekly, above ₹69,290
    ("Move ₹50,000 a week into Affiliate.", {}, False),                               # weekly, within
    ("Deploy ₹2.5 L of the unused budget into Affiliate this month.", {}, False),     # month room ~₹2.97 L
    ("Deploy ₹5 L of the unused budget into Affiliate this month.", {}, True),
    ("Shift a small test budget from Video to Affiliate.", {"test_shift_pct": 10}, False),  # 10% × ₹2.6 L a week
    ("Shift a test budget from Video to Affiliate.", {"test_shift_pct": 40}, True),         # 40% × ₹2.6 L = ₹1.0 L
    ("Shift budget from Affiliate to Paid Search.", {"test_shift_pct": 40}, False),   # money leaves Affiliate
    ("Scale Paid Search by ₹5 L a week.", {}, False),                                 # auction: no capacity
])
def test_capacity_amounts(pack, text, extra, flagged):
    cost = _of_type(pack, "marginal_cost")
    item = {**_rec(text, [cost.id, "E16", "E19"]), **extra, "source_evidence_id": "E16" if "from Video" in text else "E19"}
    checked, _ = evaluate({"recommendations": [item]}, pack)
    assert bool(_capacity_reasons(checked["recommendations"][0])) is flagged
