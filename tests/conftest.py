"""Test-wide safety: the real Gemini API can never be called from a test.

The owner's .env switches AI on and holds the API key, and config/settings.py loads it at
import time. Every test therefore starts with AI switched off and no key, and the SDK methods
that would contact Google raise an error. Tests that need AI patch in a fake client themselves.

The demo seed (saved Kalpa insights) and the demo snapshot (finished Kalpa analysis) are also
off by default, so tests of the "Generate" flow
start from an empty cache; the demo-seed tests switch it on explicitly.
"""

import dataclasses

import pytest
from google.genai import _api_client as genai_api_client
from google.genai import models as genai_models

import ai.demo_seed
import dashboard.demo
import config.settings as config_settings


class RealGeminiCallInTest(RuntimeError):
    pass


def _refuse(*_args, **_kwargs):
    raise RealGeminiCallInTest("A test tried to contact the real Gemini API. Use a fake client.")


@pytest.fixture(autouse=True)
def no_real_gemini(monkeypatch, tmp_path_factory):
    monkeypatch.setattr(config_settings, "settings",
                        dataclasses.replace(config_settings.settings, ai_enabled=False,
                                            gemini_api_key="", public_mode=False))
    for cls in (genai_models.Models, genai_models.AsyncModels):
        monkeypatch.setattr(cls, "generate_content", _refuse)
        monkeypatch.setattr(cls, "list", _refuse)
    # The SDK's single door to the network: every request goes through these methods.
    for name in ("request", "request_streamed", "async_request", "async_request_streamed",
                 "_request", "_async_request"):
        monkeypatch.setattr(genai_api_client.BaseApiClient, name, _refuse)
    monkeypatch.setattr(ai.demo_seed, "SEED_PATH", tmp_path_factory.getbasetemp() / "no_seed.json")
    monkeypatch.setattr(dashboard.demo, "DEMO_SNAPSHOT_PATH",
                        tmp_path_factory.getbasetemp() / "no_snapshot.pkl.gz")


# Sample files analysed by the PDF and Excel report tests: name -> (file, mapping overrides).
REPORT_DATASETS = {
    "clean": ("marketing_clean.csv", None),
    "messy": ("marketing_messy.xlsx", None),
    "no_revenue": ("marketing_no_revenue.csv", None),
    "no_margin": ("marketing_no_margin.csv", None),
    "meta": ("meta_ads_export_style.csv", {"Results": "leads"}),
}


@pytest.fixture(scope="session")
def report_outputs(tmp_path_factory):
    """Each report sample analysed once per test run, shared by the PDF and Excel tests (read only)."""
    from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
    folder = tmp_path_factory.mktemp("report_samples")
    out = {}
    for name, (file, overrides) in REPORT_DATASETS.items():
        raw, report = load_raw(SAMPLE_DIR / file)
        out[name] = run_pipeline(prepare(raw, report, overrides), db_path=folder / "t.db")
    return out
