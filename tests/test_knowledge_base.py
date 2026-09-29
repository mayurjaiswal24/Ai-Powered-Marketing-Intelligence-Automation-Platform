"""U2 - Mapping knowledge base: header corpus proof, context rule, value patterns, no learning.

No real AI anywhere: AI mappings come from the fake mapper in test_ai_mapping.
"""

import csv
import dataclasses
from pathlib import Path

import pandas as pd

from config import knowledge
from config.fields import CRITICAL_FIELDS
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare
from database import repository
from ingestion.mapper import map_fields, normalize_header
from ingestion.profiler import profile_dataset
from tests.test_ai_mapping import FakeMapper, assistant, csv_prep, outlay_frame

CORPUS = Path(__file__).resolve().parent / "fixtures" / "header_corpus.csv"
NEW_SOURCES = {"recognised", "context", "values"}


def outcome(m) -> str:
    """One word for what the rules decided about a single column."""
    if m.status in ("confirmed", "high_confidence"):
        # A lone Orders column stands in for Conversions: it was still recognised as Orders.
        return "orders" if m.reason.startswith("Orders used as Conversions") else m.field
    if m.status == "derived":
        return "derived"
    if m.status == "not_used":
        return "recognised" if m.source == "recognised" else "not_used"
    return "ask" if m.status == "uncertain" else m.status


def corpus_results():
    rows = list(csv.DictReader(open(CORPUS, encoding="utf-8")))
    return [(r, outcome(map_fields([r["header"]]).columns[0])) for r in rows]


# --- Proof: the header corpus ----------------------------------------------------------------

def test_header_corpus_coverage_and_critical_accuracy():
    results = corpus_results()
    assert len(results) >= 300
    assert len({r["source"] for r, _ in results}) >= 16          # every source in the brief
    wrong = [(r["header"], r["expected"], got) for r, got in results if got != r["expected"]]
    assert wrong == []
    # Resolved without AI: a final answer from Python (not a question, not unknown).
    resolved = sum(1 for r, got in results if got == r["expected"] and got not in ("ask", "unmapped"))
    coverage = resolved / len(results)
    # Critical fields: every header that is, or was taken for, Date/Spend/Revenue/Leads/Conversions.
    critical = [(r, got) for r, got in results
                if r["expected"] in CRITICAL_FIELDS or got in CRITICAL_FIELDS]
    accuracy = sum(1 for r, got in critical if got == r["expected"]) / len(critical)
    print(f"\nHeader corpus: {len(results)} headers, {coverage:.1%} resolved without AI, "
          f"critical-field accuracy {accuracy:.0%} ({len(critical)} critical headers)")
    assert coverage >= 0.95
    assert accuracy == 1.0


def test_knowledge_files_are_consistent():
    # Every extra synonym really maps to its own field (no clash with a built-in name or rule).
    for fname, header, source in knowledge.field_synonyms():
        m = map_fields([header]).columns[0]
        assert m.field == fname or outcome(m) == "orders", (header, fname, m.status, m.field)
        assert source
    # Recognised columns are never mapped to a field and never sent to Gemini.
    for header in knowledge.recognised_columns():
        m = map_fields([header]).columns[0]
        assert m.status in ("not_used", "derived") and m.field is None, header
    for header in knowledge.derived_metrics():
        assert map_fields([header]).columns[0].status == "derived", header


def test_header_normalisation_units_case_and_short_forms():
    assert normalize_header("Amount Spent (INR)") == "amount spent"      # unchanged behaviour
    for header in ("Spend (₹)", "Spend in INR", "Ad Spend (USD)", "Cost (Rs.)", "SPEND_INR",
                   "Amt spnd", "Sum of Spend", "AdSpend"):
        assert map_fields([header]).active == {"spend": header}, header
    for header in ("Rev", "Tot Rev", "TotalRevenue", "Revenue ₹"):
        assert map_fields([header]).active == {"revenue": header}, header


def test_recognised_columns_are_labelled_and_not_sent_to_gemini(tmp_path):
    df = pd.DataFrame({"Date": ["2026-03-01", "2026-03-02"], "Spend": [100, 120],
                       "Sessions": [50, 60], "Bounce rate": ["40%", "42%"], "Saves": [3, 4]})
    fake = FakeMapper()
    prep = csv_prep(tmp_path, df, fake)
    m = prep.mapping.by_column("Sessions")
    assert m.status == "not_used" and m.source == "recognised"
    assert m.reason == "Recognised: Sessions (not used in this analysis)"
    assert prep.mapping.by_column("Bounce rate").status == "derived"
    assert fake.requests == []                                  # nothing left for Gemini


# --- Context rule: Meta "Results" -------------------------------------------------------------

def _meta(objectives):
    n = len(objectives)
    return pd.DataFrame({"Day": [f"2026-03-{i + 1:02d}" for i in range(n)],
                         "Campaign name": [f"C{i % 2}" for i in range(n)],
                         "Objective": objectives, "Amount spent (INR)": [100.0 + i for i in range(n)],
                         "Results": [5 + i for i in range(n)]}).astype(str)


def test_results_follow_the_objective_column():
    leads = map_fields(profile_dataset(_meta(["Lead generation", "OUTCOME_LEADS"] * 3)))
    assert leads.active["leads"] == "Results"
    assert leads.by_column("Results").source == "context" and "Objective" in leads.by_column("Results").reason
    sales = map_fields(profile_dataset(_meta(["OUTCOME_SALES", "Sales"] * 3)))
    assert sales.active["conversions"] == "Results"
    # Mixed or non-conversion objectives: still the user's decision.
    for objectives in (["Lead generation", "Sales"] * 3, ["Traffic", "Awareness"] * 3):
        mixed = map_fields(profile_dataset(_meta(objectives)))
        assert mixed.by_column("Results").status == "uncertain", objectives
        assert mixed.by_column("Results") in mixed.needs_confirmation


def test_result_indicator_column_also_decides():
    df = _meta(["x"] * 4).drop(columns=["Objective"])
    df["Result indicator"] = "actions:onsite_conversion.lead_grouped"
    m = map_fields(profile_dataset(df))
    assert m.active["leads"] == "Results"
    df["Result indicator"] = "actions:offsite_conversion.fb_pixel_view_content"
    assert map_fields(profile_dataset(df)).by_column("Results").status == "uncertain"


# --- Value patterns ---------------------------------------------------------------------------

def test_value_patterns_recognise_categories():
    df = pd.DataFrame({"Src": ["FB Ads", "Google Ads", "LinkedIn", "facebook", "Google Ads"],
                       "Where": ["Mumbai", "Pune", "Bangalore", "Delhi", "Pune"],
                       "Kanal": ["Paid Search", "Email", "Meta", "SEO", "Email"],
                       "Gadget": ["Mobile", "Desktop", "Tablet", "Mobile", "Mobile"],
                       "Mystery": ["alpha", "beta", "gamma", "delta", "alpha"]})
    m = map_fields(profile_dataset(df))
    assert m.active["platform"] == "Src" and m.by_column("Src").source == "values"
    assert m.active["city"] == "Where"
    assert m.active["channel"] == "Kanal"                        # channels and platforms mixed
    assert m.by_column("Gadget").reason == "Recognised: Device (not used in this analysis)"
    assert m.by_column("Mystery").status == "unmapped"


def test_date_values_are_suggested_never_guessed():
    df = pd.DataFrame({"Tag": ["2026-03-01", "2026-03-02", "2026-03-03"], "Spend": ["1", "2", "3"]})
    m = map_fields(profile_dataset(df))
    tag = m.by_column("Tag")
    assert tag.status == "uncertain" and tag.candidates[0][0] == "date"
    assert tag in m.needs_confirmation                            # critical: the user confirms


def test_sample_files_do_not_use_the_new_stages():
    """The sample datasets map exactly as before: no column is decided by the new stages."""
    for name in ("marketing_clean.csv", "marketing_messy.xlsx", "marketing_no_revenue.csv",
                 "meta_ads_export_style.csv", "marketing_no_margin.csv"):
        raw, _ = load_raw(SAMPLE_DIR / name)
        mapping = map_fields(profile_dataset(raw))
        assert not {m.source for m in mapping.columns} & NEW_SOURCES, name


# --- No cross-file learning ------------------------------------------------------------------

def test_mapper_has_no_learned_step():
    """An unknown header stays open for Gemini; nothing remembered can fill it."""
    m = map_fields(["Outlay", "Clicks"])
    assert m.by_column("Outlay").status not in ("confirmed", "high_confidence")
    assert "learned" not in {c.source for c in m.columns}
    assert not hasattr(repository, "load_learned_mappings")


def test_new_layout_asks_gemini_fresh_same_layout_reuses_cache(tmp_path):
    db = tmp_path / "m.db"
    first = outlay_frame(cpc_factor=1.0)
    path = tmp_path / "a.csv"
    first.to_csv(path, index=False)
    raw, report = load_raw(path)
    fake = FakeMapper({"Outlay": "spend"})
    done = prepare(raw, report, assistant=assistant(fake, db))
    assert done.badges["Outlay"] == "verified" and len(fake.requests) == 1

    # Same column layout again -> the layout cache answers, no new call.
    again = prepare(raw, report, assistant=assistant(fake, db))
    assert again.ai.source == "cache" and len(fake.requests) == 1

    # A different file (new layout) with the same "Outlay" header -> Gemini is asked fresh,
    # and its new answer is used: a verified answer from the other file is never carried over.
    other = outlay_frame().assign(Region="North")
    path2 = tmp_path / "b.csv"
    other.to_csv(path2, index=False)
    raw2, report2 = load_raw(path2)
    fresh = FakeMapper({"Outlay": "budget"})
    new = prepare(raw2, report2, assistant=assistant(fresh, db))
    assert new.ai.source == "call" and len(fresh.requests) == 1
    assert "Outlay" in {p["column"] for p in fresh.requests[0]["problem_columns"]}
    assert new.mapping.by_column("Outlay").field == "budget"
    # Without AI the header stays for the user to choose.
    assert prepare(raw2, report2).mapping.by_column("Outlay").source != "ai"


def test_old_learned_mappings_table_is_dropped(tmp_path):
    db = tmp_path / "old.db"
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE learned_mappings (header TEXT PRIMARY KEY, field TEXT)")
    with repository.session(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "learned_mappings" not in tables
