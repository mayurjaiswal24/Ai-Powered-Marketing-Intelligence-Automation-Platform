"""Gemini field mapping, user control, public mode and the demo seed. FAKE Gemini only."""

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import ai.client
import ai.demo_seed
import config.settings as config_settings
import database.connection as dbc
from ai.cache import budget_status
from ai.client import AIResponse, ai_error
from ai.mapping import BADGES, MappingAssistant, PRIVACY_NOTE, layout_key
from config.settings import Settings
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from database import housekeeping, repository
from ingestion.loader import IngestionError, load_file
from tests.fixtures.build_mapping_fixtures import OUT, build

ROOT = Path(__file__).resolve().parents[1]
REAL_SEED = ROOT / "data" / "demo" / "kalpa_demo_seed.json"
AI_ON = Settings(ai_enabled=True, gemini_api_key="fake-key", gemini_model="main-model")


class FakeMapper:
    """Answers mapping requests from a {column: field} table ('not_used' for everything else)."""

    def __init__(self, answers=None, errors=(), raw_text=None):
        self.answers, self.errors, self.raw_text = dict(answers or {}), list(errors), raw_text
        self.requests, self.model, self.fallback_model = [], "main-model", "fallback-model"

    def use_fallback(self):
        if not self.fallback_model:
            return False
        self.model, self.fallback_model = self.fallback_model, ""
        return True

    def generate(self, system, prompt, schema):
        request = json.loads(prompt.split("\n\n", 1)[1])
        self.requests.append(request)
        if self.errors:
            raise self.errors.pop(0)
        if self.raw_text is not None:
            return AIResponse(text=self.raw_text, model=self.model)
        items = [{"column": p["column"], "field": self.answers.get(p["column"], "not_used"),
                  "reason": f"test reason for {p['column']}"} for p in request["problem_columns"]]
        return AIResponse(text=json.dumps({"mappings": items}), model=self.model,
                          input_tokens=100, output_tokens=20)


def assistant(fake, db, settings=AI_ON, **kw):
    return MappingAssistant(settings, lambda: fake, db_path=db, **kw)


def csv_prep(tmp_path, df, fake, name="f.csv", **kw):
    path = tmp_path / name
    df.to_csv(path, index=False)
    raw, report = load_raw(path)
    return prepare(raw, report, assistant=assistant(fake, tmp_path / "m.db", **kw))


def outlay_frame(cpc_factor=None):
    rows = []
    for i in range(20):
        outlay, clicks = 1000.0 + 37 * i, 200 + 5 * i
        row = {"Date": f"2026-03-{i + 1:02d}", "Campaign": "Brand", "Outlay": outlay,
               "Impressions": 20000 + 100 * i, "Clicks": clicks}
        if cpc_factor is not None:
            row["CPC"] = round(outlay * cpc_factor / clicks, 2)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module", autouse=True)
def fixture_files():
    build()


# --- Meta export: "Results" -> Leads ------------------------------------------------------------

def test_meta_export_results_mapped_by_ai(tmp_path):
    fake = FakeMapper({"Results": "leads"})
    raw, report = load_raw(SAMPLE_DIR / "meta_ads_export_style.csv")
    prep = prepare(raw, report, assistant=assistant(fake, tmp_path / "m.db"))
    m = prep.mapping.by_column("Results")
    assert (m.field, m.source) == ("leads", "ai") and m.reason.startswith("AI: ")
    assert prep.mapping.active["leads"] == "Results"
    assert prep.badges == {"Results": "check"}                   # critical, nothing to check it with
    assert BADGES["check"] == "AI · not verified, please check"
    assert prep.validation.can_analyse                             # applied, not blocking
    assert prep.ai_mapped_columns == ["Results"]
    # Only names + at most 3 examples per problem column are sent - never the data.
    request = fake.requests[0]
    assert [p["column"] for p in request["problem_columns"]] == ["Results"]
    assert len(request["problem_columns"][0]["examples"]) <= 3
    assert request["already_mapped"]["Amount spent (INR)"] == "spend"
    assert "data_dictionary" not in request
    assert len(json.dumps(request)) < 6000


# --- retail layout with its data dictionary ---------------------------------------------------------

def test_retail_layout_with_data_dictionary(tmp_path):
    frame = pd.read_csv(OUT / "retail_columns.csv")
    path = tmp_path / "retail.xlsx"
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="Marketing_Data", index=False)
        pd.DataFrame({"Column": ["Device", "Profit"],
                      "Description": ["Customer device", "Profit after all costs"]}).to_excel(
            writer, sheet_name="Data_Dictionary", index=False)
    raw, report = load_raw(path)
    assert "Customer device" in report.data_dictionary
    fake = FakeMapper({"Profit": "gross_profit", "Add_to_Cart": "engagements"})
    prep = prepare(raw, report, assistant=assistant(fake, tmp_path / "m.db"))
    request = fake.requests[0]
    assert "Customer device" in request["data_dictionary"]
    sent = {p["column"] for p in request["problem_columns"]}
    assert "Profit" in sent
    assert "Spend" not in sent and "CTR_%" not in sent             # rules handled those
    # U2: recognised-but-not-analysed columns are never sent to Gemini.
    assert not {"Device", "Ad_Format", "Discount", "Add_to_Cart"} & sent
    m = prep.mapping
    assert m.by_column("Device").status == "not_used" and m.by_column("Device").source == "recognised"
    # AI may not turn a plain Profit column into gross profit (it could double-count spend).
    assert m.by_column("Profit").status == "uncertain" and "gross_profit" not in m.active
    assert m.active["revenue"] == "Net_Revenue" and m.active["spend"] == "Spend"   # rules kept
    assert "engagements" in m.active and m.active["engagements"] == "Engagements"  # already used
    assert m.by_column("Add_to_Cart").field is None


# --- Badges: green, yellow, red ------------------------------------------------------------------

def test_green_badge_verified_by_own_cpc(tmp_path):
    prep = csv_prep(tmp_path, outlay_frame(cpc_factor=1.0), FakeMapper({"Outlay": "spend"}))
    assert prep.mapping.active["spend"] == "Outlay"
    assert prep.badges["Outlay"] == "verified" and BADGES["verified"] == "AI · verified"
    assert prep.validation.can_analyse


def test_yellow_badge_when_nothing_can_check_it(tmp_path):
    prep = csv_prep(tmp_path, outlay_frame(), FakeMapper({"Outlay": "spend"}))
    assert prep.mapping.active["spend"] == "Outlay" and prep.badges["Outlay"] == "check"
    assert prep.validation.can_analyse


def test_red_badge_not_applied_and_blocks(tmp_path):
    prep = csv_prep(tmp_path, outlay_frame(cpc_factor=1.5), FakeMapper({"Outlay": "spend"}))
    m = prep.mapping.by_column("Outlay")
    assert m.source == "ai_rejected" and m.field is None and "spend" not in prep.mapping.active
    assert prep.badges["Outlay"] == "failed"
    assert m in prep.mapping.needs_confirmation and not prep.validation.can_analyse
    # The user decides: their choice wins and unblocks the run.
    raw, report = load_raw(tmp_path / "f.csv")
    chosen = prepare(raw, report, {"Outlay": "spend"}, assistant=assistant(FakeMapper(), tmp_path / "m.db"))
    assert chosen.mapping.by_column("Outlay").source == "user" and chosen.validation.can_analyse


# --- AI unavailable: the old flow -----------------------------------------------------------------

@pytest.mark.parametrize("settings, errors, expected", [
    (Settings(ai_enabled=False), [], "AI mapping is off"),
    (AI_ON, [ai_error("quota")], "AI mapping is not available"),
    (dataclasses.replace(AI_ON, ai_max_mapping_calls_per_day=0), [], "daily AI mapping limit"),
])
def test_ai_unavailable_falls_back_to_user_choice(tmp_path, settings, errors, expected):
    fake = FakeMapper({"Results": "leads"}, errors=errors)
    raw, report = load_raw(SAMPLE_DIR / "meta_ads_export_style.csv")
    prep = prepare(raw, report, assistant=assistant(fake, tmp_path / "m.db", settings=settings))
    assert expected in prep.ai.note
    assert prep.mapping.by_column("Results").status == "uncertain"
    assert not prep.validation.can_analyse                         # user must choose, as before


def test_busy_main_model_uses_fallback_once(tmp_path):
    busy = ai_error("server")
    fake = FakeMapper({"Results": "leads"}, errors=[busy])
    raw, report = load_raw(SAMPLE_DIR / "meta_ads_export_style.csv")
    prep = prepare(raw, report, assistant=assistant(fake, tmp_path / "m.db"))
    assert prep.mapping.active["leads"] == "Results" and prep.ai.calls_made == 2
    assert prep.ai.model == "fallback-model"


def test_invalid_answers_are_rejected(tmp_path):
    raw, report = load_raw(SAMPLE_DIR / "meta_ads_export_style.csv")
    bad = FakeMapper({"Results": "sales_magic"})                   # unknown field name
    prep = prepare(raw, report, assistant=assistant(bad, tmp_path / "a.db"))
    assert prep.mapping.by_column("Results").status == "uncertain"
    assert "unknown field" in prep.ai.rejected_items[0]
    garbage = FakeMapper(raw_text="not json")
    prep = prepare(raw, report, assistant=assistant(garbage, tmp_path / "b.db"))
    assert "could not be read" in prep.ai.note and not prep.validation.can_analyse


# --- Cache, user choices, priority ---------------------------------------------------------------

def test_mapping_cache_hit_makes_no_new_call(tmp_path):
    fake = FakeMapper({"Results": "leads"})
    raw, report = load_raw(SAMPLE_DIR / "meta_ads_export_style.csv")
    db = tmp_path / "m.db"
    prepare(raw, report, assistant=assistant(fake, db))
    again = prepare(raw, report, assistant=assistant(fake, db))
    assert len(fake.requests) == 1 and again.ai.source == "cache"
    assert again.mapping.active["leads"] == "Results"
    assert repository.count_mapping_calls("2000-01-01", db_path=db) == 1
    # Same column names in another order -> same layout key.
    assert layout_key(["b", "a"]) == layout_key(["a", "b"])


def test_user_change_is_saved_and_wins_next_time(tmp_path):
    db = tmp_path / "m.db"
    raw, report = load_raw(SAMPLE_DIR / "meta_ads_export_style.csv")
    first = assistant(FakeMapper({"Results": "leads"}), db)
    prepare(raw, report, assistant=first)
    first.remember_choices(raw.columns, {"Results": "conversions", "Link clicks": None})
    later = prepare(raw, report, assistant=assistant(FakeMapper({"Results": "leads"}), db))
    m = later.mapping
    assert m.by_column("Results").field == "conversions" and m.by_column("Results").source == "user"
    assert m.by_column("Link clicks").status == "ignored"          # user > rules as well
    assert later.remembered == {"Results": "conversions", "Link clicks": None}


# --- Edit mapping and re-run (streamlit.testing) ---------------------------------------------------

@pytest.fixture
def ai_app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    fake = FakeMapper({"Results": "leads"})
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    monkeypatch.setattr(config_settings, "settings", AI_ON)
    monkeypatch.setattr(ai.client, "client_from_settings", lambda _s: fake)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240), fake, tmp_path / "app.db"


def test_app_ai_mapping_banner_and_edit_rerun(ai_app):
    at, fake, db = ai_app
    at.run()
    at.selectbox(key="sample_choice").set_value("Meta Ads export - needs a mapping decision (CSV)").run()
    at.button(key="load_sample").click().run()
    assert any("Gemini mapped 1 column (highlighted)" in i.value for i in at.info)
    assert any(PRIVACY_NOTE in c.value for c in at.caption)
    assert any("AI · not verified, please check" in c.value for c in at.caption)   # summary
    assert not at.button(key="run_analysis").disabled
    at.button(key="run_analysis").click().run()
    first_run = at.session_state["output"].run_id
    assert not at.exception and len(fake.requests) == 1

    at.button(key="edit_rerun").click().run()                       # previous choices filled in
    raw = at.session_state["raw_df"]
    layout = layout_key(raw.columns)[:10]
    box = at.selectbox(key=f"map_{layout}_Results")
    assert box.value == "leads"
    box.set_value("conversions")
    at.button(key=f"apply_all_{layout}").click().run()
    assert any("mapping has changed since the last analysis" in i.value for i in at.info)
    at.button(key="run_analysis").click().run()
    output = at.session_state["output"]
    assert output.run_id != first_run and not at.exception
    assert output.analysis.kpis["conversions"].available
    assert len(fake.requests) == 1                                   # no new AI call
    saved = repository.load_mapping_cache(layout_key(raw.columns), db_path=db)["user"]
    assert saved == {"Results": "conversions"}


def test_edit_from_data_quality_page(ai_app):
    at, _fake, _db = ai_app
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="run_analysis").click().run()
    at.sidebar.radio(key="page").set_value("Data Quality").run()
    at.button(key="edit_rerun_quality").click().run()
    assert at.sidebar.radio(key="page").value == "Upload & Profile" and not at.exception
    assert at.session_state["edit_mapping"]


# --- Public mode -----------------------------------------------------------------------------------

def test_public_mode_settings_and_upload_limit(tmp_path):
    public = Settings(public_mode=True, max_upload_mb=50, max_upload_mb_public=1)
    assert public.upload_limit_mb == 1 and Settings().upload_limit_mb == 50
    big = tmp_path / "big.csv"
    big.write_text("Date,Spend\n" + "2026-01-01,1000\n" * 90_000)       # about 1.5 MB
    with pytest.raises(IngestionError):
        load_file(big, max_file_mb=public.upload_limit_mb)


def test_public_mode_per_session_limits(tmp_path):
    s = Settings(public_mode=True, ai_max_calls_per_session=2, ai_max_calls_per_day=15,
                 ai_max_calls_per_run=3)
    status = budget_status(s, None, session_calls=2, db_path=tmp_path / "b.db")
    assert status.remaining == 0 and "per session" in status.reason
    assert budget_status(Settings(), None, session_calls=2, db_path=tmp_path / "b.db").remaining > 0
    public_ai = dataclasses.replace(AI_ON, public_mode=True, ai_max_mapping_calls_per_session=2)
    raw, report = load_raw(SAMPLE_DIR / "meta_ads_export_style.csv")
    fake = FakeMapper({"Results": "leads"})
    prep = prepare(raw, report, assistant=assistant(fake, tmp_path / "m.db", settings=public_ai,
                                                    session_calls=2))
    assert "limit for this session" in prep.ai.note and not fake.requests


def test_public_mode_session_isolation(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    monkeypatch.setattr(config_settings, "settings", Settings(public_mode=True))
    visitor_a = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    visitor_a.run()
    assert any("kept only for this session" in i.value for i in visitor_a.info)
    visitor_a.button(key="load_sample").click().run()
    visitor_a.button(key="run_analysis").click().run()
    run_a = visitor_a.session_state["output"].run_id
    assert repository.run_session_id(run_a) == visitor_a.session_state["session_id"]
    visitor_a.run()
    assert any(b.key == f"open_run_{run_a}" for b in visitor_a.button)

    visitor_b = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    visitor_b.run()
    assert not any(b.key == f"open_run_{run_a}" for b in visitor_b.button)    # cannot see it
    assert not any(b.key == f"delete_run_{run_a}" for b in visitor_b.button)


def test_public_mode_24_hour_cleanup(tmp_path, monkeypatch):
    db = tmp_path / "app.db"
    raw, report = load_raw(SAMPLE_DIR / "test_a_small.csv")
    old = run_pipeline(prepare(raw, report), db_path=db, session_id="visitor-1").run_id
    stale = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(timespec="seconds")
    with repository.session(db) as conn:
        conn.execute("UPDATE runs SET created_at = ? WHERE id = ?", (stale, old))
    monkeypatch.setattr(config_settings, "settings", Settings(public_mode=True))
    fresh = run_pipeline(prepare(raw, report), db_path=db, session_id="visitor-2").run_id   # next save
    assert repository.get_run(old, db_path=db) is None
    assert repository.get_run(fresh, db_path=db) is not None
    assert housekeeping.delete_expired_runs(db_path=db) == []        # nothing else is old


# --- Demo seed -----------------------------------------------------------------------------------

def test_demo_seed_file_is_valid():
    seed = json.loads(REAL_SEED.read_text(encoding="utf-8"))
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    assert seed["dataset"]["file_hash"] == report.file_hash
    assert seed["mapping"] == {m.column: m.field for m in prepare(raw, report).mapping.columns}
    from ai.schemas import AIInsights
    AIInsights.model_validate(seed["ai"]["insights"])


def test_demo_seed_loads_on_empty_database(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(ai.demo_seed, "SEED_PATH", REAL_SEED)
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)   # AI calls are off
    at.run()
    at.button(key="load_sample").click().run()                         # Kalpa clean sample
    at.button(key="run_analysis").click().run()
    at.sidebar.radio(key="page").set_value("AI Insights").run()
    assert not at.exception
    assert any("No AI call was used" in s.value for s in at.success)
    with repository.session(tmp_path / "app.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM ai_insights").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM ai_runs WHERE cache_hit = 0").fetchone()[0] == 0
