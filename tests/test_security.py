"""The Gemini API key never appears in code, git, settings output, errors, prompts or exports."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ai.client import GeminiClient, classify_error
from ai.context_builder import build_evidence
from config.settings import Settings, load_settings
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline
from reports.excel_report import export_workbook
from reports.pdf_report import export_pdf

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "AIzaSyFAKE-test-key-0123456789abcdefghij"
KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_-]{20,}")


def test_settings_never_print_the_key():
    s = load_settings({"GEMINI_API_KEY": FAKE_KEY, "GEMINI_MODEL": "m"})
    assert s.has_gemini_key
    assert FAKE_KEY not in repr(s) and FAKE_KEY not in str(s)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_no_key_like_strings_in_tracked_files_or_history():
    tracked = subprocess.run(["git", "grep", "-nIE", KEY_PATTERN.pattern], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    # The only allowed match is this test's obviously fake key.
    hits = [line for line in tracked.stdout.splitlines() if "FAKE" not in line]
    assert hits == []
    ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert ignored.stdout.strip() == ".env"
    committed = subprocess.run(["git", "log", "--all", "--oneline", "--", ".env"], cwd=ROOT,
                               capture_output=True, text=True)
    assert committed.stdout.strip() == ""
    assert (ROOT / ".env.example").exists()
    assert "GEMINI_API_KEY=\n" in (ROOT / ".env.example").read_text(encoding="utf-8")   # empty


def test_errors_never_contain_the_key():
    from google.genai import errors as genai_errors
    for code, status in ((400, "INVALID_ARGUMENT"), (403, "PERMISSION_DENIED"), (429, "RESOURCE_EXHAUSTED"),
                         (503, "UNAVAILABLE")):
        exc = genai_errors.APIError(code, {"error": {"code": code, "status": status,
                                                     "message": f"bad key {FAKE_KEY}"}})
        err = classify_error(exc)
        assert FAKE_KEY not in err.user_message and FAKE_KEY not in str(err)
    client = GeminiClient(FAKE_KEY, "model")
    assert FAKE_KEY not in repr(client)


def test_prompt_and_exports_never_contain_the_key(tmp_path):
    raw, report = load_raw(SAMPLE_DIR / "test_a_small.csv")
    output = run_pipeline(prepare(raw, report), db_path=tmp_path / "a.db")
    pack = build_evidence(output.analysis, model="m")
    assert FAKE_KEY not in pack.json_text
    pdf = export_pdf(output.analysis, exports_dir=tmp_path / "x", db_path=tmp_path / "a.db")
    xlsx = export_workbook(output.analysis, exports_dir=tmp_path / "x", db_path=tmp_path / "a.db")
    for path in (pdf, xlsx, tmp_path / "a.db"):
        assert FAKE_KEY.encode() not in path.read_bytes()
    assert Settings().gemini_api_key == ""
