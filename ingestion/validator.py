"""Capability detection: given the field mapping, which analyses can this dataset support?

"Never fabricate unavailable business information" (SPEC §19). Every analysis is switched on
only when its required fields are mapped; otherwise it is switched off with a plain-English
reason that the dashboard and reports can show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config.fields import FIELD_BY_NAME

if TYPE_CHECKING:
    from ingestion.mapper import MappingResult
    from ingestion.profiler import DatasetProfile

FUNNEL_ORDER = ["impressions", "clicks", "leads", "qualified_leads", "opportunities",
                "conversions", "customers"]
GEOGRAPHY_FIELDS = ["region", "state", "city", "country", "market", "city_tier"]
SEGMENT_FIELDS = ["customer_segment", "customer_type", "new_returning", "age", "gender"]
PRODUCT_FIELDS = ["product", "product_category", "sku"]
# Below this share of readable dates, the Date column is not usable for trends.
MIN_USABLE_DATE_SHARE = 0.5


@dataclass
class Capability:
    key: str
    label: str
    enabled: bool
    reason: str                 # why it is on, or why it is off (plain English)
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    capabilities: dict[str, Capability]
    blocking: list[str]         # problems that stop analysis altogether
    notes: list[str]            # useful facts, e.g. which CAC denominator is used
    cac_denominator: str | None = None
    aov_denominator: str | None = None
    funnel_stages: list[str] = field(default_factory=list)

    @property
    def can_analyse(self) -> bool:
        return not self.blocking

    def enabled(self, key: str) -> bool:
        return self.capabilities[key].enabled

    @property
    def limitations(self) -> list[str]:
        return [c.reason for c in self.capabilities.values() if not c.enabled]


def _label(fname: str) -> str:
    return FIELD_BY_NAME[fname].label


def validate(mapping: "MappingResult", profile: "DatasetProfile | None" = None) -> ValidationResult:
    has = set(mapping.active)
    caps: dict[str, Capability] = {}
    notes: list[str] = []
    blocking: list[str] = []

    def add(key, label, enabled, on_reason, off_reason, **details):
        caps[key] = Capability(key, label, enabled, on_reason if enabled else off_reason, details)

    # --- Time series -----------------------------------------------------------------------------
    date_ok, date_reason = "date" in has, "Trends are unavailable because no date column was found."
    if date_ok and profile is not None:
        col = profile.columns.get(mapping.active["date"])
        if col is not None and col.date_share < MIN_USABLE_DATE_SHARE:
            date_ok = False
            date_reason = (f"Trends are unavailable because most values in '{col.name}' could not "
                           "be read as dates.")
    add("time_series", "Trends over time", date_ok, "Daily, weekly and monthly trends are available.",
        date_reason)

    # --- Efficiency KPIs ----------------------------------------------------------------------
    add("ctr", "Click-through rate (CTR)", {"clicks", "impressions"} <= has,
        "CTR = Clicks / Impressions.", "CTR is unavailable because Clicks or Impressions is missing.")
    add("cpc", "Cost per click (CPC)", {"spend", "clicks"} <= has,
        "CPC = Spend / Clicks.", "CPC is unavailable because Spend or Clicks is missing.")
    add("cpl", "Cost per lead (CPL)", {"spend", "leads"} <= has,
        "CPL = Spend / Leads.", "CPL is unavailable because Spend or Leads is missing.")

    # --- Revenue, ROAS, ROI, AOV, CAC ---------------------------------------------------------
    add("revenue_metrics", "Revenue and ROAS", {"revenue", "spend"} <= has,
        "Revenue and ROAS (Revenue / Spend) are available.",
        "Revenue and ROAS are unavailable because " +
        ("no revenue field was detected in the uploaded dataset." if "revenue" not in has
         else "no spend field was detected."))

    if {"gross_profit", "spend"} <= has:
        add("roi", "ROI", True, "ROI = (Gross profit - Spend) / Spend.", "", basis="gross_profit")
    elif {"margin", "revenue", "spend"} <= has:
        add("roi", "ROI", True, "ROI = (Revenue x Margin - Spend) / Spend, using the margin column.",
            "", basis="margin")
    else:
        add("roi", "ROI", False, "", "ROI is hidden because it needs gross profit or margin data. "
            "ROAS is shown instead; no margin is assumed.")

    aov_den = "orders" if "orders" in has else ("conversions" if "conversions" in has else None)
    add("aov", "Average order value (AOV)", "revenue" in has and aov_den is not None,
        f"AOV = Revenue / {_label(aov_den) if aov_den else ''}.",
        "AOV is unavailable because Revenue or Conversions/Orders is missing.", denominator=aov_den)

    cac_den = "customers" if "customers" in has else ("conversions" if "conversions" in has else None)
    add("cac", "Customer acquisition cost (CAC)", "spend" in has and cac_den is not None,
        f"CAC = Spend / {_label(cac_den) if cac_den else ''}.",
        "CAC is unavailable because Spend or Conversions/Customers is missing.",
        denominator=cac_den)
    if caps["cac"].enabled:
        notes.append(f"CAC is calculated as Spend / {_label(cac_den)}"
                     + (" because no Customers field exists." if cac_den == "conversions" else "."))

    # --- Funnel -----------------------------------------------------------------------------
    stages = [f for f in FUNNEL_ORDER if f in has]
    add("funnel", "Funnel analysis", len(stages) >= 2,
        "Funnel stages: " + " > ".join(_label(s) for s in stages) + ".",
        "Funnel analysis needs at least two funnel stages (e.g. Clicks and Leads).", stages=stages)

    # --- Dimensions -------------------------------------------------------------------------
    for key, label, fields_, off in [
        ("campaign_analysis", "Campaign analysis", ["campaign_name", "campaign_id"],
         "Campaign analysis is unavailable because no campaign column was found."),
        ("channel_analysis", "Channel analysis", ["channel", "platform"],
         "Channel analysis is unavailable because no channel or platform column was found."),
        ("segment_analysis", "Segment analysis", SEGMENT_FIELDS,
         "Segment analysis is hidden because no customer segment column was found."),
        ("geography_analysis", "Geography analysis", GEOGRAPHY_FIELDS,
         "Geography analysis is hidden because no region, state or city column was found."),
        ("product_analysis", "Product analysis", PRODUCT_FIELDS,
         "Product analysis is hidden because no product or course column was found."),
    ]:
        present = [f for f in fields_ if f in has]
        add(key, label, bool(present), "Available by: " + ", ".join(_label(f) for f in present) + ".",
            off, fields=present)

    # --- Anomaly detection ------------------------------------------------------------------
    metrics = [f for f in ("spend", "clicks", "leads", "conversions", "revenue") if f in has]
    add("anomaly_detection", "Anomaly detection", caps["time_series"].enabled and bool(metrics),
        "Anomaly checks run on: " + ", ".join(_label(m) for m in metrics) + ".",
        "Anomaly detection needs a usable date column and at least one metric such as Spend or Leads.",
        metrics=metrics)

    if "reach" in has:
        notes.append("Reach is shown only at the level it was reported; it cannot be added up "
                     "across days or campaigns.")

    # --- Blocking problems ------------------------------------------------------------------
    if not ({"spend", "revenue", "leads", "conversions"} & has):
        blocking.append("No core marketing numbers were found. The file needs at least one of: "
                        "Spend, Revenue, Leads or Conversions.")
    for m in mapping.needs_confirmation:
        options = " or ".join(_label(f) for f, _ in m.candidates)
        blocking.append(f"Please confirm what the column '{m.column}' contains ({options}), "
                        "or mark it as not used, before analysis starts.")

    return ValidationResult(caps, blocking, notes, cac_denominator=cac_den,
                            aov_denominator=aov_den, funnel_stages=stages)
