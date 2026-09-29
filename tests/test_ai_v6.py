"""U9 (AI prompt v6): planning evidence (profitability, next-conversion cost, pacing, forecast), the new
prompt rules and the evaluator's forecast and loss-maker checks. FAKE AI client only."""

import copy
import json

import pytest

from ai.context_builder import EvidenceItem, EvidencePack, build_evidence
from ai.evaluator import evaluate, extract_numbers
from ai.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS
from ai.service import generate_ai_insights
from analytics.pacing import pacing_analysis
from analytics.profitability import profitability_analysis
from config.settings import AI_CONTEXT_MAX_CHARS
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from tests.test_ai import FakeClient, good_answer

NEW_TYPES = ["profitability", "marginal_cost", "pacing", "forecast"]


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    return run_pipeline(prepare(raw, report), db_path=tmp_path_factory.mktemp("v6") / "a.db")


@pytest.fixture(scope="module")
def analysis(output):
    return output.analysis


@pytest.fixture(scope="module")
def pack(analysis):
    return build_evidence(analysis, model="m")


def _item(pack, kind):
    return next(i for i in pack.items if i.type == kind)


# --- Evidence pack ---------------------------------------------------------------------------------

def test_new_items_present_last_with_valid_ids_and_within_size(pack):
    assert [i.type for i in pack.items[-4:]] == NEW_TYPES           # appended: older IDs do not move
    ids = [i.id for i in pack.items]
    assert ids == [f"E{n:02d}" for n in range(1, len(ids) + 1)]
    assert pack.size <= AI_CONTEXT_MAX_CHARS and pack.dropped_items == 1    # only a low-priority segment
    assert all(t in pack.json_text for t in ('"type":"profitability"', '"type":"forecast"'))


def test_new_item_contents_on_kalpa(pack):
    prof = _item(pack, "profitability").facts
    assert prof["total contribution (gross profit minus spend)"] == "₹12.9 Cr"
    assert prof["loss-making campaigns"] == "none"
    assert "Meta - Digital Marketing Prospecting: -₹6.2 L" in prof["closest to break-even (contribution)"]
    assert "Paid Search 5.65x vs 1.67x (Profitable)" in prof["paid channels: ROAS vs own break-even ROAS"]
    mc = _item(pack, "marginal_cost").facts
    assert mc["Paid Search"].startswith("next conversion costs ₹9,521")
    assert "paid per result" in mc["Affiliate"] and "at most about" in mc["Affiliate"]
    assert "Email" not in mc                                         # owned channels are never in the planner
    pace = _item(pack, "pacing").facts
    assert pace["status"] == "Underspending" and pace["budget source"] == "Budget column in the data"
    fc = _item(pack, "forecast").facts
    assert all("past forecasts were off by about" in fc[k] and " to " in fc[k] for k in fc if "next 8 weeks" in k)


def test_user_inputs_are_never_in_the_evidence(analysis, output):
    """Targets, an assumed margin, a typed monthly budget and scenarios never reach Gemini."""
    fake = copy.copy(analysis)
    clean = output.clean.clean_df
    df = clean.drop(columns=[c for c in ("gross_profit", "margin", "budget") if c in clean])
    fake.profitability = profitability_analysis(df, assumed_margin_pct=40)
    fake.pacing = pacing_analysis(df, monthly_budget=5_000_000)
    assert fake.profitability.available and fake.pacing.available and fake.pacing.assumed
    types = {i.type for i in build_evidence(fake, model="m").items}
    assert "profitability" not in types and "pacing" not in types


def test_older_analyses_without_planning_parts_still_work(analysis):
    old = copy.copy(analysis)
    old.profitability = old.pacing = old.forecast = old.response_curves = None
    pack = build_evidence(old, model="m")
    assert not {i.type for i in pack.items} & set(NEW_TYPES)


def test_no_revenue_sample_has_no_profitability_item(tmp_path):
    raw, report = load_raw(SAMPLE_DIR / "marketing_no_revenue.csv")
    result = run_pipeline(prepare(raw, report), db_path=tmp_path / "n.db").analysis
    assert "profitability" not in {i.type for i in build_evidence(result, model="m").items}


def test_size_limit_trims_lowest_priority_first(analysis):
    small = build_evidence(analysis, max_chars=12_000, model="m")
    assert small.size <= 12_000 and small.dropped_items > 1
    kept = {i.type for i in small.items}
    assert {"kpi", "profitability", "marginal_cost"} <= kept and "segment" not in kept


def test_fingerprint_changes_with_the_new_prompt_version(pack):
    from ai.cache import compute_fingerprint
    assert PROMPT_VERSION == "v6"
    assert pack.fingerprint != compute_fingerprint(pack.json_text, "m", prompt_version="v5")


# --- Prompt ----------------------------------------------------------------------------------------

def test_prompt_v6_rules():
    text = " ".join(SYSTEM_INSTRUCTIONS.lower().split())
    for rule in ("cost of the next conversion", "not on average roas", "paid per result",
                 "own break-even roas", "pacing risk", "range together with its past accuracy",
                 "never introduce a number that is not in the evidence"):
        assert rule in text, rule
    for kept in ("source_evidence_id", "roas paid media only", "owned channels"):     # v2-v5 rules stay
        assert kept in text, kept


# --- Evaluator ---------------------------------------------------------------------------------------

def test_money_range_with_one_unit_is_two_amounts():
    assert [n.value for n in extract_numbers("₹2.7–5.9 Cr next")] == [2.7e7, 5.9e7]
    assert [n.value for n in extract_numbers("₹75.5 L to ₹1.1 Cr")] == [7.55e6, 1.1e7]


def _answer_with(pack, extra_findings: list[dict]) -> dict:
    ans = good_answer(pack)
    ans["key_findings"] = extra_findings + ans["key_findings"][:1]
    return ans


def test_forecast_statements_need_range_and_past_accuracy(pack):
    fc = _item(pack, "forecast")
    rev = fc.facts["Revenue, next 8 weeks"]              # "₹X to ₹Y (central estimate ₹Z); past ... N%"
    low, rest = rev.split(" to ", 1)
    high = rest.split(" (")[0]
    err = rev.rsplit("about ", 1)[1]
    central = rev.split("central estimate ")[1].split(")")[0]
    items = [
        {"text": f"Revenue should be {low}–{high} over the next 8 weeks; past forecasts were off by about {err}.",
         "evidence_ids": [fc.id]},
        {"text": f"Revenue will reach {central} over the next 8 weeks.", "evidence_ids": [fc.id]},
        {"text": f"Revenue should be {low} to {high} over the next 8 weeks.", "evidence_ids": [fc.id]},
        {"text": "Revenue will be ₹9.9 Cr to ₹12.3 Cr next quarter.", "evidence_ids": [fc.id]},
    ]
    checked, _ = evaluate(_answer_with(pack, items), pack)
    good, single, no_error, invented = checked["key_findings"][:4]
    assert good["status"] == "verified" and not good["weak"]
    assert single["weak"] and any("single number" in w for w in single["weak_reasons"])
    assert no_error["weak"] and any("past accuracy" in w for w in no_error["weak_reasons"])
    assert invented["status"] == "unverified"                       # numbers not in the evidence


def test_short_range_form_is_verified(pack):
    fc = _item(pack, "forecast")
    spend = fc.facts["Spend, next 8 weeks"]
    low, high = spend.split(" to ")[0], spend.split(" to ")[1].split(" (")[0]
    err = spend.rsplit("about ", 1)[1]
    assert low.endswith(" L") and high.endswith(" Cr")                # mixed units: long form only
    item = {"text": f"Spend is expected at {low} to {high}; past forecasts were off by about {err}.",
            "evidence_ids": [fc.id]}
    checked, _ = evaluate(_answer_with(pack, [item]), pack)
    assert checked["key_findings"][0]["status"] == "verified" and not checked["key_findings"][0]["weak"]


def test_loss_maker_recommendation_gets_rupees_at_stake():
    prof = EvidenceItem("profitability", "Profitability", {"biggest loss-makers (₹ lost)": "Camp A: ₹4.5 L lost"},
                        2, id="E01", values={"loss:Camp A": 450_000.0})
    kpi = EvidenceItem("kpi", "Overall KPIs", {"Spend": "₹1.0 Cr"}, 1, id="E02")
    pack = EvidencePack({}, [prof, kpi], [], by_id={"E01": prof, "E02": kpi})
    recs = [{"text": "Review spend.", "evidence_ids": ["E02"], "priority": "high", "metric_to_watch": "ROAS"},
            {"text": "Pause Camp A, which lost ₹4.5 L.", "evidence_ids": ["E01"], "priority": "high",
             "metric_to_watch": "Contribution"}]
    checked, _ = evaluate({"recommendations": recs}, pack)
    top = checked["recommendations"][0]
    assert top["text"].startswith("Pause Camp A") and top["rupees_at_stake"] == 450_000.0
    assert "loss-making campaign Camp A" in top["at_stake_text"]


def test_fake_client_run_with_v6_evidence(analysis, pack, tmp_path):
    fc, mc = _item(pack, "forecast"), _item(pack, "marginal_cost")
    ans = good_answer(pack)
    rev = fc.facts["Revenue, next 8 weeks"]
    ans["key_findings"][0] = {"text": f"Revenue should be {rev.split(' (')[0]} over the next 8 weeks; "
                                      f"past forecasts were off by about {rev.rsplit('about ', 1)[1]}.",
                              "evidence_ids": [fc.id]}
    ans["recommendations"][0] = {"text": "Test moving budget from Professional Network, where the next conversion "
                                         "costs ₹43,186, to Paid Search, where it costs ₹9,521.",
                                 "evidence_ids": [mc.id], "priority": "high", "metric_to_watch": "CAC"}
    client = FakeClient([json.dumps(ans)])
    run = generate_ai_insights(analysis, client, db_path=tmp_path / "f.db")
    assert run.ok and len(client.calls) == 1
    system, prompt, _ = client.calls[0]
    assert '"type":"marginal_cost"' in prompt and "pacing risk" in system
    texts = {i["text"]: i for i in run.insights["key_findings"] + run.insights["recommendations"]}
    assert all(texts[a["text"]]["status"] == "verified" for a in (ans["key_findings"][0], ans["recommendations"][0]))
