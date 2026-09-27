"""The conftest guard applies to every test file: AI is off, and the Gemini SDK cannot reach
Google. This file checks the guard itself and that no test file sits outside its reach."""

from pathlib import Path

import pytest

import config.settings as config_settings
from ai.client import GeminiClient
from tests.conftest import RealGeminiCallInTest

TESTS = Path(__file__).resolve().parent


def test_ai_is_off_and_keyless_in_tests():
    s = config_settings.settings
    assert not s.ai_enabled and not s.gemini_api_key and not s.public_mode


def test_real_client_cannot_reach_google():
    client = GeminiClient("not-a-real-key", "any-model")      # building it is harmless ...
    with pytest.raises(Exception) as caught:                    # ... sending is blocked
        client.generate("system", "prompt", None)
    assert isinstance(caught.value.__cause__ or caught.value, RealGeminiCallInTest)


def test_every_test_file_is_covered_by_the_guard():
    """conftest.py in tests/ applies to every test below it; pytest.ini only collects tests/."""
    ini = (TESTS.parent / "pytest.ini").read_text()
    assert "testpaths = tests" in ini
    assert (TESTS / "conftest.py").exists()
    files = sorted(TESTS.rglob("test_*.py"))
    assert files and all(TESTS in f.parents for f in files)
    # No test may switch the guard off.
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "no_real_gemini" not in text or f.name == "test_zz_guard.py", f.name
