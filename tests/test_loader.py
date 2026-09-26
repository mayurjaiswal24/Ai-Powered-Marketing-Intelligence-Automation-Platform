"""Phase 02 tests: loading CSV/XLSX files into raw text tables, and friendly failures."""

import hashlib
import io
from pathlib import Path

import pytest

from ingestion.loader import IngestionError, load_file

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
FIXTURES = ROOT / "tests" / "fixtures"


# --- Sample dataset files ---------------------------------------------------------------------

@pytest.mark.parametrize("name, rows, cols", [
    ("marketing_clean.csv", 8471, 21),
    ("marketing_no_revenue.csv", 8471, 19),
    ("marketing_no_margin.csv", 8471, 20),
    ("meta_ads_export_style.csv", 1070, 6),
    ("test_a_small.csv", 40, 21),
    ("test_b_medium.csv", 300, 21),
    ("test_c_large.csv", 1500, 21),
    ("marketing_messy.xlsx", 8497, 22),
])
def test_sample_files_load(name, rows, cols):
    df, report = load_file(SAMPLE / name)
    assert df.shape == (rows, cols)
    assert (report.rows, report.columns) == (rows, cols)
    assert report.filename == name
    assert report.file_hash == hashlib.sha256((SAMPLE / name).read_bytes()).hexdigest()


def test_every_sample_file_is_covered():
    tested = {"marketing_clean.csv", "marketing_no_revenue.csv", "marketing_no_margin.csv",
              "meta_ads_export_style.csv", "test_a_small.csv", "test_b_medium.csv",
              "test_c_large.csv", "marketing_messy.xlsx"}
    on_disk = {p.name for p in SAMPLE.iterdir() if p.suffix in (".csv", ".xlsx")}
    assert on_disk == tested


def test_values_stay_as_original_text():
    df, report = load_file(SAMPLE / "marketing_messy.xlsx")
    assert report.sheet_used == "Campaign Data"
    assert all(isinstance(v, str) for v in df["revenue"])
    assert (df["revenue"] == "N/A").sum() == 40          # text "N/A" preserved
    assert (df["revenue"] == "").sum() == 40             # blank cells become ""
    assert df["Amount Spent (INR)"].str.startswith("₹").sum() == 30
    assert df["date"].str.fullmatch(r"\d{2}/\d{2}/\d{4}").sum() == 150  # day-first text kept
    assert df.iloc[-1]["campaign_id"] == "Grand Total"    # removing it is cleaning's job
    # Real Excel dates come out as plain YYYY-MM-DD, not "2025-10-01 00:00:00".
    assert df.iloc[0]["date"] == "2025-10-01"


def test_clean_csv_matches_file_text():
    df, report = load_file(SAMPLE / "marketing_clean.csv")
    first_line = (SAMPLE / "marketing_clean.csv").read_text(encoding="utf-8").splitlines()[1]
    assert ",".join(df.iloc[0]) == first_line
    assert report.encoding == "utf-8" and report.delimiter == "comma"
    assert report.warnings == []


# --- Uploaded-file objects (what Streamlit passes) --------------------------------------------

class FakeUpload:
    """Mimics streamlit's UploadedFile: has .name, .size and .getvalue()."""

    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()
        self.size = len(self._data)

    def getvalue(self):
        return self._data


def test_uploaded_file_object():
    df, report = load_file(FakeUpload(SAMPLE / "test_a_small.csv"))
    assert df.shape == (40, 21)
    assert report.filename == "test_a_small.csv"


def test_bytesio_with_explicit_filename():
    data = (SAMPLE / "test_a_small.csv").read_bytes()
    df, _ = load_file(io.BytesIO(data), "upload.csv")
    assert len(df) == 40


# --- Format variations ------------------------------------------------------------------------

def test_semicolon_csv():
    df, report = load_file(FIXTURES / "semicolon.csv")
    assert report.delimiter == "semicolon"
    assert df.shape == (3, 6)
    assert df.loc[0, "spend"] == "1200,50"


def test_tab_delimited_csv():
    df, report = load_file(FIXTURES / "tab_delimited.csv")
    assert report.delimiter == "tab"
    assert df.shape == (3, 6)


def test_latin1_csv():
    df, report = load_file(FIXTURES / "latin1.csv")
    assert report.encoding == "latin-1"
    assert df.loc[0, "campaign_name"] == "Café Learners Campaign"
    assert any("Latin-1" in w for w in report.warnings)


def test_utf8_bom_csv():
    df, report = load_file(FIXTURES / "utf8_bom.csv")
    assert report.encoding == "utf-8-sig"
    assert list(df.columns) == ["date", "campaign_name", "spend"]   # no stray BOM in "date"
    assert df.loc[0, "spend"] == "₹1,23,456"


def test_two_sheet_workbook_picks_data_sheet():
    df, report = load_file(FIXTURES / "two_sheets.xlsx")
    assert report.sheet_used == "Campaign Data"
    assert report.other_sheets == ["Notes"]
    assert df.shape == (3, 6)
    assert df.loc[0, "spend"] == "1200.5"
    assert any("Notes" in w for w in report.warnings)


def test_title_row_above_header():
    df, report = load_file(FIXTURES / "title_row.xlsx")
    assert report.header_row == 3
    assert list(df.columns) == ["date", "campaign_name", "channel", "spend", "clicks", "leads"]
    assert len(df) == 3
    assert any("row 3" in w for w in report.warnings)


def test_odd_structure_is_repaired_with_warnings():
    df, report = load_file(FIXTURES / "odd_structure.csv")
    assert list(df.columns) == ["date", "spend", "spend (2)", "Unnamed column 4", "leads"]
    assert len(df) == 2
    text = " ".join(report.warnings)
    assert "more than once" in text and "no heading" in text and "empty row" in text


def test_large_row_warning():
    _, report = load_file(SAMPLE / "test_b_medium.csv", large_row_warning=100)
    assert any("300 rows" in w for w in report.warnings)


# --- Friendly failures ------------------------------------------------------------------------

@pytest.mark.parametrize("name, code, phrase", [
    ("empty.csv", "empty", "empty"),
    ("header_only.csv", "header_only", "no data rows"),
    ("fake.xlsx", "corrupt", "damaged or not a real Excel file"),
    ("not_a_workbook.xlsx", "corrupt", "damaged or not a real Excel file"),
    ("empty_workbook.xlsx", "empty", "no data"),
    ("password_protected.xlsx", "password_protected", "password-protected"),
    ("report.pdf", "unsupported_type", "not supported"),
    ("legacy.xls", "unsupported_type", "old Excel format"),
])
def test_bad_files_raise_friendly_errors(name, code, phrase):
    with pytest.raises(IngestionError) as info:
        load_file(FIXTURES / name)
    err = info.value
    assert err.code == code
    assert phrase in err.user_message
    # Plain English: no Python jargon or tracebacks leak into the message.
    for jargon in ("Traceback", "Error:", "KeyError", "BadZipFile", "openpyxl", "\\x"):
        assert jargon not in err.user_message


def test_file_too_large():
    big = io.BytesIO(b"a,b\n" + b"1,2\n" * 400_000)  # about 1.6 MB
    with pytest.raises(IngestionError) as info:
        load_file(big, "big.csv", max_file_mb=1)
    assert info.value.code == "too_large"
    assert "1 MB limit" in info.value.user_message


def test_binary_data_named_csv():
    with pytest.raises(IngestionError) as info:
        load_file(io.BytesIO(b"\x00\x01\x02binary\x00stuff"), "data.csv")
    assert info.value.code == "corrupt"


def test_missing_path():
    with pytest.raises(IngestionError) as info:
        load_file(FIXTURES / "does_not_exist.csv")
    assert info.value.code == "not_found"
