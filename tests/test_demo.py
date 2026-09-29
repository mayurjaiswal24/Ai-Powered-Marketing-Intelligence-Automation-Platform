"""Instant demo: the shipped Kalpa snapshot opens the finished analysis with its AI insights."""

from pathlib import Path

import pandas as pd
import pytest

import ai.demo_seed
import dashboard.demo
import database.connection as dbc
from config.settings import Settings
from dashboard.demo import load_demo_snapshot, restore_demo_run
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from database import repository

ROOT = Path(__file__).resolve().parents[1]
REAL_SNAPSHOT = ROOT / "data" / "demo" / "kalpa_demo_snapshot.pkl.gz"
REAL_SEED = ROOT / "data" / "demo" / "kalpa_demo_seed.json"


@pytest.fixture
def demo_on(monkeypatch):
    monkeypatch.setattr(dashboard.demo, "DEMO_SNAPSHOT_PATH", REAL_SNAPSHOT)
    monkeypatch.setattr(ai.demo_seed, "SEED_PATH", REAL_SEED)


def test_snapshot_matches_a_fresh_analysis(demo_on, tmp_path):
    """Guards against a stale snapshot: it must equal what the current code calculates."""
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    payload = load_demo_snapshot(report.file_hash)
    assert payload is not None
    fresh = run_pipeline(prepare(raw, report), db_path=tmp_path / "f.db").analysis
    saved = payload["output"].analysis
    for key, kpi in fresh.kpis.items():
        assert saved.kpis[key].value == pytest.approx(kpi.value, nan_ok=True), key
    for name in ("channels", "campaigns", "funnel", "incidents"):
        pd.testing.assert_frame_equal(saved.tables()[name].reset_index(drop=True),
                                      fresh.tables()[name].reset_index(drop=True), check_dtype=False)
    assert [f.text for f in saved.findings] == [f.text for f in fresh.findings]
    assert saved.paid_kpis["roas"].value == pytest.approx(fresh.paid_kpis["roas"].value)


def test_snapshot_only_for_the_exact_sample_file(demo_on):
    raw, report = load_raw(SAMPLE_DIR / "marketing_no_margin.csv")
    assert load_demo_snapshot(report.file_hash) is None
    assert load_demo_snapshot("0" * 64) is None


def test_restore_saves_a_normal_run_with_insights(demo_on, tmp_path):
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    db = tmp_path / "d.db"
    prep, output = restore_demo_run(load_demo_snapshot(report.file_hash), Settings(), db_path=db,
                                    session_id="visitor")
    assert output.run_id is not None and output.save_error is None
    assert output.analysis.metadata["run_id"] == output.run_id
    assert repository.run_session_id(output.run_id, db_path=db) == "visitor"
    assert repository.list_recent_runs(db_path=db)["run_id"].tolist() == [output.run_id]
    with repository.session(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ai_insights").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM ai_runs").fetchone()[0] == 0          # no call
    again_prep, again = restore_demo_run(load_demo_snapshot(report.file_hash), Settings(), db_path=db)
    assert again.run_id != output.run_id                    # each load is its own run
    assert prep.validation.can_analyse


def test_load_sample_opens_the_finished_analysis_instantly(demo_on, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    at.button(key="load_sample").click().run()            # Kalpa clean sample (default)
    assert not at.exception
    assert at.session_state["output"] is not None          # no "Run analysis" click needed
    assert any("Analysis ready" in s.value for s in at.success)
    assert any("opened instantly" in c.value for c in at.caption)
    at.sidebar.radio(key="page").set_value("AI Insights").run()
    assert any("No AI call was used" in s.value for s in at.success)
    at.sidebar.radio(key="page").set_value("Executive Overview").run()
    assert not at.exception


def test_other_samples_and_uploads_run_the_full_pipeline(demo_on, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(dbc, "settings", Settings(database_path=str(tmp_path / "app.db")))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    at.selectbox(key="sample_choice").set_value("Lead generation only - no revenue (CSV)").run()
    at.button(key="load_sample").click().run()
    assert "output" not in at.session_state or at.session_state["output"] is None
    at.button(key="run_analysis").click().run()
    assert at.session_state["output"].run_id is not None and not at.exception


# Upgrade v2.0 guard: the AI evidence pack for the Kalpa sample must not change by accident, so cached
# insights stay valid. U1-U8 kept 037d98c5...; U9 (prompt v6, planning evidence) changed it on purpose.
KALPA_EVIDENCE_FINGERPRINT = "826ecf58fe8334fe567739cb403cf4f657f0628980021db5799311ea5a75e59e"


def test_kalpa_evidence_pack_fingerprint_unchanged(tmp_path):
    from ai.context_builder import build_evidence
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    analysis = run_pipeline(prepare(raw, report), db_path=tmp_path / "f.db").analysis
    assert build_evidence(analysis, model="test-model").fingerprint == KALPA_EVIDENCE_FINGERPRINT
