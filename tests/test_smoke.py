"""Phase 00 smoke tests: the settings module imports and gives safe defaults."""

from config import settings as settings_module
from config.settings import Settings, load_settings


def test_settings_module_imports():
    assert isinstance(settings_module.settings, Settings)


def test_defaults_with_empty_environment():
    s = load_settings({})
    assert s.gemini_api_key == ""
    assert s.gemini_model == ""
    assert s.ai_enabled is False
    assert s.ai_max_calls_per_run == 3
    assert s.ai_max_calls_per_day == 15
    assert s.ai_cache_enabled is True
    assert s.database_path == "marketing_intelligence.db"
    assert s.has_gemini_key is False
    assert s.max_upload_mb == 50
    assert s.large_row_warning == 200_000


def test_values_are_parsed_safely():
    s = load_settings({
        "AI_ENABLED": "Yes",
        "AI_MAX_CALLS_PER_RUN": "5",
        "AI_MAX_CALLS_PER_DAY": "not-a-number",
        "AI_CACHE_ENABLED": "maybe",
        "DATABASE_PATH": "  ",
    })
    assert s.ai_enabled is True
    assert s.ai_max_calls_per_run == 5
    assert s.ai_max_calls_per_day == 15      # bad number -> default
    assert s.ai_cache_enabled is True        # unrecognised word -> default
    assert s.database_path == "marketing_intelligence.db"  # blank -> default


def test_negative_limits_fall_back_to_default():
    s = load_settings({"AI_MAX_CALLS_PER_RUN": "-1"})
    assert s.ai_max_calls_per_run == 3


def test_api_key_not_shown_in_repr():
    s = load_settings({"GEMINI_API_KEY": "secret-value-123"})
    assert s.has_gemini_key is True
    assert "secret-value-123" not in repr(s)
