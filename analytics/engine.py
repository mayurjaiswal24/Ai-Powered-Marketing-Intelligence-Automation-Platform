"""The analytics engine: one call produces ONE AnalysisResult that every output reads from.

Dashboard, PDF, Excel and the AI evidence summary all use this object (or call the same
analytics/kpis.py functions for filtered views), so a number can never differ between them
(SPEC §23).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from analytics.anomalies import ANOMALY_COLUMNS, detect_anomalies
from analytics.campaign import campaign_table
from analytics.channel import channel_table
from analytics.findings import Finding, generate_findings
from analytics.funnel import funnel_by_channel, funnel_table
from analytics.incidents import INCIDENT_COLUMNS, build_incidents
from analytics.kpis import KPIResult, compute_kpis
from analytics.segments import SEGMENT_DIMENSIONS, available_dimensions, segment_table
from analytics.trends import all_time_series, seasonality_index

# Capability that must be enabled for each kind of segment dimension.
_SEGMENT_CAPABILITY = {"segment": "segment_analysis", "geography": "geography_analysis",
                       "product": "product_analysis"}


@dataclass
class AnalysisResult:
    metadata: dict
    kpis: dict[str, KPIResult]
    kpi_table: pd.DataFrame
    campaigns: pd.DataFrame
    channels: pd.DataFrame
    funnel: pd.DataFrame
    funnel_by_channel: pd.DataFrame
    segments: dict[str, pd.DataFrame]
    trends: dict[str, pd.DataFrame]
    seasonality: pd.DataFrame | None
    seasonality_note: str
    anomalies: pd.DataFrame
    findings: list[Finding]
    incidents: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=INCIDENT_COLUMNS))
    capabilities: object = None          # ingestion.validator.ValidationResult
    quality_summary: object = None       # processing.quality.QualitySummary
    notes: list[str] = field(default_factory=list)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Every result table by name (the names findings refer to)."""
        out = {"kpis": self.kpi_table, "campaigns": self.campaigns, "channels": self.channels,
               "funnel": self.funnel, "funnel_by_channel": self.funnel_by_channel,
               "anomalies": self.anomalies, "incidents": self.incidents}
        out.update({f"segment_{k}": v for k, v in self.segments.items()})
        out.update({f"trend_{k}": v for k, v in self.trends.items()})
        if self.seasonality is not None:
            out["seasonality"] = self.seasonality
        return out


def _enabled(capabilities, key: str) -> bool:
    """True if no capability information was given, or the capability is switched on."""
    caps = getattr(capabilities, "capabilities", None)
    return True if not caps or key not in caps else caps[key].enabled


def kpi_frame(kpis: dict[str, KPIResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "kpi": k.key, "label": k.label, "value": k.value, "formatted": k.formatted,
        "available": k.available, "fmt": k.fmt, "formula": k.formula_text, "note": k.note,
        "reason_unavailable": k.reason_unavailable, "numerator": k.numerator,
        "denominator": k.denominator} for k in kpis.values()])


def run_analysis(clean_df: pd.DataFrame, capabilities=None, quality_summary=None, *,
                 dataset_name: str | None = None, run_id: int | None = None) -> AnalysisResult:
    """Run every analysis the data supports on the cleaned dataset."""
    df = clean_df
    kpis = compute_kpis(df, capabilities)
    has_dates = "date" in df and df["date"].notna().any() and _enabled(capabilities, "time_series")

    campaigns = campaign_table(df) if _enabled(capabilities, "campaign_analysis") else pd.DataFrame()
    channels = channel_table(df) if _enabled(capabilities, "channel_analysis") else pd.DataFrame()
    funnel = funnel_table(df) if _enabled(capabilities, "funnel") else pd.DataFrame()
    funnel_ch = funnel_by_channel(df) if _enabled(capabilities, "funnel") else pd.DataFrame()

    segments = {}
    for dim in available_dimensions(df):
        if _enabled(capabilities, _SEGMENT_CAPABILITY[SEGMENT_DIMENSIONS[dim]]):
            segments[dim] = segment_table(df, dim)

    trends = all_time_series(df) if has_dates else {}
    if has_dates:
        seasonality, season_note = seasonality_index(df)
    else:
        seasonality, season_note = None, "Seasonality is unavailable because there is no usable date column."
    anomalies = (detect_anomalies(df) if has_dates and _enabled(capabilities, "anomaly_detection")
                 else pd.DataFrame(columns=ANOMALY_COLUMNS))

    result = AnalysisResult(
        metadata={
            "dataset_name": dataset_name, "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(df),
            "date_min": df["date"].min() if "date" in df else None,
            "date_max": df["date"].max() if "date" in df else None,
        },
        kpis=kpis, kpi_table=kpi_frame(kpis), campaigns=campaigns, channels=channels,
        funnel=funnel, funnel_by_channel=funnel_ch, segments=segments, trends=trends,
        seasonality=seasonality, seasonality_note=season_note, anomalies=anomalies,
        findings=[], capabilities=capabilities, quality_summary=quality_summary,
        notes=list(getattr(capabilities, "notes", []) or []))
    result.incidents = build_incidents(anomalies, df)
    result.findings = generate_findings(result.tables(), data_end=result.metadata.get("date_max"))
    return result


# ---------------------------------------------------------------------------------------------
# Saving to SQLite (tidy tables, see database/schema.py)
# ---------------------------------------------------------------------------------------------

def _melt(table: pd.DataFrame, key: str, key_name: str) -> pd.DataFrame:
    numeric = [c for c in table.columns if c != key and pd.api.types.is_numeric_dtype(table[c])]
    long = table.melt(id_vars=[key], value_vars=numeric, var_name="metric", value_name="value")
    long["value"] = long["value"].astype(float)
    return long.rename(columns={key: key_name})


def to_storage_tables(result: AnalysisResult) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    k = result.kpi_table
    out["kpi_results"] = pd.DataFrame({
        "kpi": k["kpi"], "value": k["value"].astype(float), "numerator": k["numerator"].astype(float),
        "denominator": k["denominator"].astype(float), "formula": k["formula"],
        "note": k["note"].where(k["available"], k["reason_unavailable"])})
    if not result.campaigns.empty:
        out["campaign_analysis"] = _melt(result.campaigns, "campaign", "campaign")
    if not result.channels.empty:
        out["channel_analysis"] = _melt(result.channels, "channel", "channel")
    seg_parts = [_melt(t.drop(columns="dimension"), "segment", "segment").assign(dimension=d)
                 for d, t in result.segments.items() if not t.empty]
    if seg_parts:
        out["segment_analysis"] = pd.concat(seg_parts, ignore_index=True)[
            ["dimension", "segment", "metric", "value"]]
    funnels = [f for f in (result.funnel, result.funnel_by_channel) if not f.empty]
    if funnels:
        out["funnel_analysis"] = pd.concat(funnels, ignore_index=True)[
            ["scope", "stage_order", "stage", "value", "rate_from_previous"]]
    trend_parts = []
    for grain, t in result.trends.items():
        period = t["period"].astype(str) if grain == "month" else t["period"].dt.strftime("%Y-%m-%d")
        long = _melt(t.drop(columns="grain").assign(period=period), "period", "period")
        trend_parts.append(long.assign(grain=grain))
    if trend_parts:
        out["trend_analysis"] = pd.concat(trend_parts, ignore_index=True)[
            ["grain", "period", "metric", "value"]]
    if not result.anomalies.empty:
        out["anomalies"] = result.anomalies[ANOMALY_COLUMNS]
    return out


def save_analysis(result: AnalysisResult, run_id: int, db_path=None) -> None:
    """Store every result table for this run (replacing earlier results of the same run)."""
    from database import repository
    for table, frame in to_storage_tables(result).items():
        repository.save_analysis_table(run_id, table, frame, db_path=db_path)
    repository.update_run_status(run_id, "analysed", db_path=db_path)
