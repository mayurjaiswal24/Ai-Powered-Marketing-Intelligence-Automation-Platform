"""Deployment readiness: settings from Streamlit secrets, complete example files, pinned
requirements without test tools, and a rehearsal of a fresh public deployment."""

import re
import tomllib
from pathlib import Path

import pytest

import config.settings as config_settings
from config.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
SETTING_NAMES = set(re.findall(r'env\.get\("([A-Z_]+)"\)', (ROOT / "config" / "settings.py").read_text()))


@pytest.fixture
def secrets_file(tmp_path, monkeypatch):
    """Point Streamlit at a temporary secrets.toml and clean up what it puts in os.environ."""
    from streamlit import config as st_config
    from streamlit.runtime.secrets import secrets_singleton
    path = tmp_path / "secrets.toml"
    monkeypatch.setattr(st_config, "get_option",
                        lambda key, _orig=st_config.get_option: [str(path)] if key == "secrets.files" else _orig(key))
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    secrets_singleton._reset()
    yield path
    secrets_singleton._reset()
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_come_from_streamlit_secrets(secrets_file):
    secrets_file.write_text('GEMINI_API_KEY = "hosted-key"\nGEMINI_MODEL = "hosted-model"\n'
                            'AI_ENABLED = true\nPUBLIC_MODE = "true"\nAI_MAX_CALLS_PER_DAY = 7\n'
                            'KEEP_LAST_RUNS = "12"\n', encoding="utf-8")
    config_settings._load_streamlit_secrets()
    s = load_settings()
    assert (s.gemini_api_key, s.gemini_model) == ("hosted-key", "hosted-model")
    assert s.ai_enabled is True                        # a TOML boolean reaches the app too
    assert s.public_mode is True and s.ai_max_calls_per_day == 7 and s.keep_last_runs == 12
    assert "hosted-key" not in repr(s)


def test_no_secrets_file_locally_is_fine(secrets_file):
    config_settings._load_streamlit_secrets()              # file does not exist: nothing happens
    assert load_settings().public_mode is False


def test_secrets_example_lists_every_setting_without_real_values(secrets_file):
    example = ROOT / ".streamlit" / "secrets.toml.example"
    values = tomllib.loads(example.read_text(encoding="utf-8"))
    assert set(values) == SETTING_NAMES
    assert values["PUBLIC_MODE"] == "true" and values["AI_ENABLED"] == "true"
    assert not values["GEMINI_API_KEY"].startswith("AIza")
    secrets_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    config_settings._load_streamlit_secrets()
    s = load_settings()
    assert s.public_mode and s.upload_limit_mb == 10 and s.ai_max_calls_per_session == 2
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert SETTING_NAMES <= set(re.findall(r"^([A-Z_]+)=", env_example, re.M))


def test_requirements_are_pinned_and_split():
    runtime = [l for l in (ROOT / "requirements.txt").read_text().splitlines() if l and not l.startswith("#")]
    dev = [l for l in (ROOT / "requirements-dev.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[0-9][0-9A-Za-z.]*", l) for l in runtime), runtime
    names = {l.split("==")[0].lower() for l in runtime}
    assert {"streamlit", "pandas", "google-genai", "reportlab", "xlsxwriter", "plotly"} <= names
    assert not names & {"pytest", "pytest-cov", "coverage", "pypdf"}       # test tools stay out
    assert dev[0] == "-r requirements.txt" and any(l.startswith("pytest==") for l in dev)


def test_demo_files_are_tracked_and_present():
    for name in ("kalpa_demo_seed.json", "kalpa_demo_snapshot.pkl.gz"):
        assert (ROOT / "data" / "demo" / name).exists(), name
    gitignore = (ROOT / ".gitignore").read_text()
    assert "data/app/" in gitignore and ".env" in gitignore and ".streamlit/secrets.toml" in gitignore
