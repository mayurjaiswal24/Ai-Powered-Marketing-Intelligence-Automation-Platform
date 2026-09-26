"""Phase 07 tests: analytics tables, anomaly recall on planted anomalies, findings consistency."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.anomalies import detect_anomalies
from analytics.campaign import rank_campaigns
from analytics.engine import run_analysis, save_analysis, to_storage_tables
from analytics.kpis import compute_kpis
from analytics.trends import seasonality_index
from database import repository as repo
from ingestion.loader import load_file
from ingestion.mapper import map_fields
from ingestion.profiler import profile_dataset
from ingestion.validator import validate
from processing.cleaner import clean_dataset

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def pipeline(name):
    raw, report = load_file(SAMPLE / name)
    profile = profile_dataset(raw)
    mapping = map_fields(profile)
    caps = validate(mapping, profile)
    cleaned = clean_dataset(raw, mapping)
    return cleaned.clean_df, run_analysis(cleaned.clean_df, caps, cleaned.quality_summary,
                                          dataset_name=name)


@pytest.fixture(scope="module")
def clean():
    return pipeline("marketing_clean.csv")


@pytest.fixture(scope="module")
def truth():
    return json.loads((SAMPLE / "ground_truth_anomalies.json").read_text(encoding="utf-8"))["anomalies"]


@pytest.fixture(scope="module")
def patterns():
    return json.loads((SAMPLE / "expected_patterns.json").read_text(encoding="utf-8"))["patterns"]


# --- Anomaly recall ----------------------------------------------------------------------------

def _overlaps(row, start, end):
    return not (row.period_end < pd.Timestamp(start) or row.period_start > pd.Timestamp(end))


def _matches(row, g) -> bool:
    """A detection matches a planted anomaly if the periods overlap and the entity is the one
    the anomaly was planted on (campaign, region or platform)."""
    if not _overlaps(row, g["start_date"], g["end_date"]):
        return False
    if row.entity_type == "campaign":
        return g["campaign_names"] != "all" and row.entity in g["campaign_names"]
    if row.entity_type == "region":
        return g["region"] != "all" and row.entity == g["region"]
    if row.entity_type == "platform":
        platforms = g["platform"] if isinstance(g["platform"], list) else [g["platform"]]
        return row.entity in platforms and g["campaign_ids"] == "all"
    return False


def test_anomaly_recall(clean, truth):
    anomalies = clean[1].anomalies
    detected, report = 0, []
    for g in truth:
        hits = [r for r in anomalies.itertuples() if _matches(r, g)]
        detected += bool(hits)
        how = ", ".join(sorted({f"{r.entity_type}:{r.metric} {r.pct_change:+.0f}%" for r in hits}))
        report.append(f"  {g['id']} {'DETECTED' if hits else 'MISSED  '} {g['metric']:<24} {how}")
    print(f"\nAnomaly recall: {detected}/{len(truth)} planted anomalies detected "
          f"({len(anomalies)} anomalies reported in total)")
    print("\n".join(report))
    assert detected / len(truth) >= 0.8


def test_seasonal_peaks_mostly_not_flagged(clean, truth, patterns):
    anomalies = clean[1].anomalies
    planted = [r for r in anomalies.itertuples() if any(_matches(r, g) for g in truth)]
    peaks = [p for p in patterns if p["start_date"]]
    lines, in_peaks = [], 0
    for p in peaks:
        inside = [r for r in anomalies.itertuples() if _overlaps(r, p["start_date"], p["end_date"])
                  and r not in planted]
        in_peaks += len(inside)
        lines.append(f"  {p['id']} {p['name']:<32} {len(inside)} flag(s)")
    print(f"\nFlags inside normal seasonal windows (not planted anomalies): {in_peaks} of "
          f"{len(anomalies)}")
    print("\n".join(lines))
    # The admissions season alone covers 3 of the 12 months; far fewer than a quarter of all
    # flags should land in the seasonal windows if seasonality is being removed.
    assert in_peaks <= 0.25 * len(anomalies)
    # The Diwali, year-end and January shifts hit every campaign together: almost never flagged.
    for p in peaks:
        if p["id"] in ("P01", "P02", "P03"):
            inside = [r for r in anomalies.itertuples()
                      if _overlaps(r, p["start_date"], p["end_date"]) and r not in planted]
            assert len(inside) <= 2, p["name"]


def test_anomaly_record_fields(clean):
    a = clean[1].anomalies
    required = {"metric", "entity_type", "entity", "period_start", "period_end", "baseline",
                "observed", "pct_change", "score", "direction", "description"}
    assert required <= set(a.columns)
    weekly = a[a["method"] == "weekly_robust_score"]
    assert weekly["score"].abs().min() >= 4.0 and weekly["pct_change"].abs().min() >= 30
    outage = a[a["method"] == "tracking_outage_rule"]
    assert len(outage) >= 1 and outage.iloc[0]["entity"] == "Google Ads"
    assert "tracking outage" in outage.iloc[0]["description"]
    assert set(a["sentiment"]) <= {"positive", "negative", "check"}


def test_detector_on_steady_data_with_one_spike():
    """Four campaigns with perfectly steady weeks, and one real spike: exactly that is found."""
    dates = pd.date_range("2026-01-05", periods=16 * 7, freq="D")
    rows = []
    for c in ["A", "B", "C", "D"]:
        for d in dates:
            spend = 2000.0 * (3 if c == "B" and pd.Timestamp("2026-03-16") <= d <= pd.Timestamp("2026-03-22") else 1)
            rows.append({"date": d, "campaign_name": c, "spend": spend})
    df = pd.DataFrame(rows)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    found = detect_anomalies(df)
    assert len(found) == 1
    row = found.iloc[0]
    assert (row["entity"], row["metric"], row["direction"]) == ("B", "spend", "up")
    assert row["period_start"] == pd.Timestamp("2026-03-16")
    assert row["pct_change"] == pytest.approx(200, abs=1)


# --- Tables ------------------------------------------------------------------------------------

def test_funnel_never_increases(clean):
    result = clean[1]
    for scope, part in pd.concat([result.funnel, result.funnel_by_channel]).groupby("scope"):
        counts = part[part["stage"] != "revenue"].sort_values("stage_order")["value"].to_numpy()
        assert (np.diff(counts) <= 0).all(), scope
    assert result.funnel["stage"].tolist() == ["impressions", "clicks", "leads", "qualified_leads",
                                               "conversions", "revenue"]


def test_campaign_and_channel_tables(clean):
    df, result = clean
    c, ch = result.campaigns, result.channels
    assert len(c) == df["campaign_name"].nunique()
    assert c["spend_share_pct"].sum() == pytest.approx(100)
    assert c["spend"].sum() == pytest.approx(df["spend"].sum())
    assert {"first_date", "last_date", "active_days", "trend"} <= set(c.columns)
    # Index 100 = overall average; ROAS index recomputed from table and KPI values.
    overall_roas = result.kpis["roas"].value
    assert ch["roas_index"].tolist() == pytest.approx((ch["roas"] / overall_roas * 100).tolist())
    ranked = rank_campaigns(c, "cpl", top=3)
    assert ranked["cpl"].is_monotonic_increasing     # lowest CPL is best
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_segments_and_trends(clean):
    df, result = clean
    assert {"customer_segment", "region", "city_tier", "product_category"} <= set(result.segments)
    seg = result.segments["customer_segment"]
    assert seg["conversions"].sum() == df["conversions"].sum()
    assert {"growth_pct", "revenue_share_pct", "aov", "cac", "roas"} <= set(seg.columns)
    weekly = result.trends["week"]
    assert {"spend_wow_pct", "revenue_rolling_4w", "days"} <= set(weekly.columns)
    assert weekly["days"].iloc[0] < 7                      # first week is partial and marked
    monthly = result.trends["month"]
    assert len(monthly) == 12 and "revenue_mom_pct" in monthly
    assert result.seasonality is not None and len(result.seasonality) == 12
    short, note = seasonality_index(df[df["date"] < "2026-04-01"])
    assert short is None and "Insufficient history" in note


# --- Findings --------------------------------------------------------------------------------

def test_every_finding_number_matches_its_table(clean):
    result = clean[1]
    tables = result.tables()
    assert len(result.findings) >= 5
    for f in result.findings:
        assert f.type == "Analytical finding" and f.numbers
        for n in f.numbers:
            table = tables[n["table"]]
            cell = table.loc[table[n["key_column"]] == n["key"], n["column"]].iloc[0]
            assert n["value"] == pytest.approx(float(cell)), (f.id, n)


def test_findings_quote_the_right_values(clean):
    result = clean[1]
    by_id = {f.id: f for f in result.findings}
    best = by_id["F-best-channel"]
    channels = result.channels.set_index("channel")
    assert best.entity == channels["roas"].idxmax()
    assert f"{channels.loc[best.entity, 'roas']:.2f}x" in best.text


# --- Missing revenue ---------------------------------------------------------------------------

def test_no_revenue_dataset_runs_without_revenue_analytics():
    df, result = pipeline("marketing_no_revenue.csv")
    assert not result.kpis["revenue"].available and result.kpis["revenue"].value is None
    assert not result.kpis["roas"].available and not result.kpis["roi"].available
    for table in (result.campaigns, result.channels, *result.segments.values(), *result.trends.values()):
        assert not {"revenue", "roas", "aov", "roi", "gross_profit"} & set(table.columns)
    assert "revenue" not in set(result.funnel["stage"])
    assert "revenue" not in set(result.anomalies["metric"])
    assert all(f.metric not in ("roas", "revenue") for f in result.findings)
    assert result.kpis["cpl"].available and result.findings
    stored = to_storage_tables(result)["kpi_results"].set_index("kpi")
    assert pd.isna(stored.loc["roas", "value"])                      # absent, not zero
    assert "no revenue field" in stored.loc["roas", "note"]


def test_engine_respects_disabled_capabilities(clean):
    df, _ = clean
    lead_gen = df.drop(columns=["revenue", "gross_profit"])
    caps = validate(map_fields(list(lead_gen.columns), column_types={
        c: ("date" if c in ("date", "week_start") else "number" if pd.api.types.is_numeric_dtype(lead_gen[c])
            else "text") for c in lead_gen.columns}))
    result = run_analysis(lead_gen, caps)
    assert compute_kpis(lead_gen, caps)["roas"].available is False
    assert result.kpis["roas"].available is False


# --- Saving ------------------------------------------------------------------------------------

def test_save_analysis_to_sqlite(clean, tmp_path):
    result = clean[1]
    db = tmp_path / "a.db"
    ds = repo.register_dataset("clean.csv", "h", db_path=db)
    run = repo.create_run(ds.id, db_path=db)
    save_analysis(result, run, db_path=db)
    assert repo.get_run(run, db_path=db)["status"] == "analysed"
    kpis = repo.load_analysis_table(run, "kpi_results", db_path=db).set_index("kpi")
    assert kpis.loc["roas", "value"] == pytest.approx(result.kpis["roas"].value)
    anomalies = repo.load_analysis_table(run, "anomalies", db_path=db)
    assert len(anomalies) == len(result.anomalies)
    ch = repo.load_analysis_table(run, "channel_analysis", db_path=db)
    spend = ch[(ch["channel"] == "Email") & (ch["metric"] == "spend")]["value"].iloc[0]
    assert spend == pytest.approx(result.channels.set_index("channel").loc["Email", "spend"])
