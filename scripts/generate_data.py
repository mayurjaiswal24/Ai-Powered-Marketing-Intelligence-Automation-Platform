"""Synthetic marketing data generator for "Kalpa Learning" (a fictional Indian ed-tech company).

Kalpa Learning sells paid online courses in four categories and runs paid + owned marketing
across India. Marketing produces leads, a counselling team qualifies them, and some enroll.

What this script writes (all into the output folder, default `data/sample/`):
    marketing_clean.csv                 main dataset (one row = date x campaign x region)
    marketing_messy.xlsx                same data with deliberate data-quality problems
    marketing_no_revenue.csv            revenue + gross_profit removed (lead-gen only)
    marketing_no_margin.csv             only gross_profit removed
    meta_ads_export_style.csv           Meta-only extract with ad-platform style headers
    test_a_small.csv / test_b_medium.csv / test_c_large.csv   small slices for testing
    ground_truth_anomalies.json         the anomalies we planted (the "answer key")
    ground_truth_quality_issues.json    every problem injected into the messy file
    expected_patterns.json              normal seasonal patterns (NOT anomalies)

The output is fully seeded: the same seed and row target always give identical files.

Usage:
    python scripts/generate_data.py                       # defaults
    python scripts/generate_data.py --rows 3000 --output-dir some/folder --seed 7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sample"
DEFAULT_SEED = 42
DEFAULT_ROW_TARGET = 8500

START_DATE = pd.Timestamp("2025-10-01")
END_DATE = pd.Timestamp("2026-09-30")

CLEAN_COLUMNS = [
    "date", "campaign_id", "campaign_name", "campaign_type", "objective", "channel",
    "platform", "region", "city_tier", "customer_segment", "course_category", "budget",
    "spend", "impressions", "reach", "clicks", "leads", "qualified_leads", "conversions",
    "revenue", "gross_profit",
]
REGIONS = ["North", "South", "East", "West", "Central"]

# ---------------------------------------------------------------------------------------------
# Business assumptions
# ---------------------------------------------------------------------------------------------

# Channel economics (relative, not exact). Paid channels: spend buys clicks at a CPC, and
# impressions follow from CTR. Rates are "per click" (lead_rate), "per lead" (qual_rate) and
# "per qualified lead" (conv_rate).
CHANNEL_ECONOMICS = {
    # High intent: people are actively searching for a course. Higher CTR and CPC.
    "Paid Search": dict(cpc=30.0, ctr=0.045, lead_rate=0.085, qual_rate=0.40, conv_rate=0.08,
                        freq=(1.1, 1.3), weekend_spend=0.85, weekend_lead=0.95),
    # Cheap reach, but many casual sign-ups -> weak lead quality.
    "Paid Social": dict(cpc=9.0, ctr=0.011, lead_rate=0.05, qual_rate=0.20, conv_rate=0.055,
                        freq=(1.6, 2.6), weekend_spend=1.10, weekend_lead=1.05),
    # Huge impressions, very low CTR: mostly an awareness channel.
    "Video": dict(cpc=12.0, ctr=0.0035, lead_rate=0.025, qual_rate=0.28, conv_rate=0.07,
                  freq=(1.8, 3.0), weekend_spend=1.10, weekend_lead=1.0),
    # Expensive clicks, but the best-quality leads for working professionals.
    "Professional Network": dict(cpc=70.0, ctr=0.006, lead_rate=0.08, qual_rate=0.55,
                                 conv_rate=0.12, freq=(1.3, 1.8), weekend_spend=0.6,
                                 weekend_lead=0.7),
    # In-house email to existing leads/alumni: near-zero cost, high conversion.
    # "impressions" = email opens. Sends happen on weekdays only (weekend rows have zero spend).
    "Email": dict(cost_per_open=0.30, daily_opens=3000, ctr=0.03, lead_rate=0.10,
                  qual_rate=0.45, conv_rate=0.10, freq=(1.05, 1.2), weekend_open_share=0.25),
    # Affiliates are paid per result: spend = payout per enrolment + fee per lead + a small
    # daily platform fee, so spend scales with conversions.
    "Affiliate": dict(daily_clicks=180, ctr=0.018, lead_rate=0.07, qual_rate=0.40,
                      conv_rate=0.10, payout_per_conversion=5000.0, fee_per_lead=50.0,
                      platform_fee=250.0, freq=(1.2, 1.5)),
}

# Daily budget per campaign per region, in INR, before seasonality.
BASE_DAILY_BUDGET = {
    "Paid Search": 12000, "Paid Social": 9000, "Video": 10000,
    "Professional Network": 7000, "Email": 500, "Affiliate": 20000,
}

# Metro (Tier 1) auctions are more competitive -> higher CPC.
TIER_CPC = {"Tier 1": 1.30, "Tier 2": 1.00, "Tier 3": 0.75}
REGION_CPC = {"North": 1.05, "South": 1.00, "East": 0.90, "West": 1.10, "Central": 0.85}
REGION_DEMAND = {"North": 1.00, "South": 1.05, "East": 0.90, "West": 1.05, "Central": 0.90}

# How campaign type / objective / segment shift the funnel rates.
TYPE_LEAD = {"Prospecting": 1.0, "Retargeting": 1.6, "Brand": 1.3, "Promotion": 1.2}
TYPE_CONV = {"Prospecting": 1.0, "Retargeting": 1.4, "Brand": 1.2, "Promotion": 1.35}
OBJECTIVE_LEAD = {"Awareness": 0.6, "Lead Generation": 1.0, "Conversion": 1.0}
# Brand keywords (people searching "Kalpa Learning") are cheap: little competition.
BRAND_SEARCH_CPC = 0.6
OBJECTIVE_CPC = {"Awareness": 0.8, "Lead Generation": 1.0, "Conversion": 1.0}
OBJECTIVE_CONV = {"Awareness": 1.0, "Lead Generation": 1.0, "Conversion": 1.1}
# Students sign up easily and buy cheaper plans; mid-career professionals are choosier
# (lower conversion -> higher CAC) but buy premium tracks (higher AOV).
SEGMENT_LEAD = {"Students": 1.10, "Early-Career Professionals": 1.00, "Mid-Career Professionals": 0.85}
SEGMENT_QUAL = {"Students": 0.85, "Early-Career Professionals": 1.00, "Mid-Career Professionals": 1.15}
SEGMENT_CONV = {"Students": 1.15, "Early-Career Professionals": 1.00, "Mid-Career Professionals": 0.75}

# List prices (INR) per course category. The segment decides where in the range a buyer lands.
CATEGORY_PRICE_RANGE = {
    "Data & Analytics": (30000, 75000),
    "Digital Marketing": (18000, 45000),
    "Software Development": (40000, 100000),
    "Finance": (25000, 60000),
}
SEGMENT_PRICE_POSITION = {
    "Students": 0.10, "Early-Career Professionals": 0.35, "Mid-Career Professionals": 1.00,
}
# Mid-career buyers mostly choose advanced / executive tracks priced above the standard range.
SEGMENT_PRICE_PREMIUM = {
    "Students": 1.0, "Early-Career Professionals": 1.0, "Mid-Career Professionals": 1.20,
}
# Gross margin at LIST price (after instructor fees, platform and delivery costs).
# Discounts come straight out of margin because delivery cost per enrolment is fixed.
CATEGORY_MARGIN = {
    "Data & Analytics": 0.65, "Digital Marketing": 0.60,
    "Software Development": 0.55, "Finance": 0.62,
}
PROMOTION_DISCOUNT = 0.20
DIWALI_EXTRA_DISCOUNT = 0.05
MAX_COUPON_DISCOUNT = 0.08  # regular (non-promotion) enrolments get a small random coupon

DIWALI_DAY = pd.Timestamp("2025-10-20")
DIWALI_PROMO_WINDOW = (pd.Timestamp("2025-10-10"), pd.Timestamp("2025-10-26"))
LATE_DECEMBER_DIP = (pd.Timestamp("2025-12-22"), pd.Timestamp("2025-12-31"))

# Demand level by calendar month (1.0 = normal). January = New-Year resolutions;
# May-July = admissions / post-exam-results season.
MONTH_DEMAND = {10: 1.00, 11: 0.92, 12: 0.88, 1: 1.30, 2: 1.02, 3: 0.95,
                4: 1.00, 5: 1.20, 6: 1.35, 7: 1.25, 8: 1.00, 9: 0.95}


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    campaign_name: str
    channel: str
    platform: str
    campaign_type: str
    objective: str
    course_category: str
    customer_segment: str
    city_tier: str
    start: str
    end: str
    regions: tuple[str, ...]  # priority order: earlier regions are kept when rows are limited


CAMPAIGNS = [
    Campaign("KL-GS-001", "Google Search - Data Analytics Courses", "Paid Search", "Google Ads",
             "Prospecting", "Lead Generation", "Data & Analytics", "Early-Career Professionals",
             "Tier 1", "2025-10-01", "2026-09-30", ("North", "Central", "West", "South", "East")),
    Campaign("KL-GS-002", "Google Search - Software Development Courses", "Paid Search",
             "Google Ads", "Prospecting", "Lead Generation", "Software Development",
             "Early-Career Professionals", "Tier 2", "2025-10-01", "2026-09-30",
             ("South", "North", "West", "Central")),
    Campaign("KL-GS-003", "Google Search - Brand Keywords", "Paid Search", "Google Ads", "Brand",
             "Conversion", "Data & Analytics", "Early-Career Professionals", "Tier 1", "2025-10-01", "2026-09-30",
             ("West", "South", "North", "East", "Central")),
    Campaign("KL-GS-004", "Google Search - Finance Courses", "Paid Search", "Google Ads",
             "Prospecting", "Lead Generation", "Finance", "Mid-Career Professionals", "Tier 1",
             "2025-11-01", "2026-09-30", ("West", "South", "North")),
    Campaign("KL-META-001", "Meta - Digital Marketing Prospecting", "Paid Social", "Meta",
             "Prospecting", "Lead Generation", "Digital Marketing", "Students", "Tier 2",
             "2025-10-01", "2026-09-30", ("North", "Central", "East", "South", "West")),
    Campaign("KL-META-002", "Meta - Website Retargeting", "Paid Social", "Meta", "Retargeting",
             "Conversion", "Data & Analytics", "Early-Career Professionals", "Tier 1",
             "2025-10-01", "2026-09-30", ("West", "North", "South")),
    Campaign("KL-META-003", "Meta - Finance Basics Awareness", "Paid Social", "Meta",
             "Prospecting", "Awareness", "Finance", "Students", "Tier 3", "2025-11-01",
             "2026-08-31", ("East", "Central", "North")),
    Campaign("KL-YT-001", "YouTube - Career Stories", "Video", "YouTube", "Brand", "Awareness",
             "Software Development", "Students", "Tier 2", "2025-10-01", "2026-09-30",
             ("South", "Central", "North", "West", "East")),
    Campaign("KL-YT-002", "YouTube - Data Analytics Explainers", "Video", "YouTube",
             "Prospecting", "Lead Generation", "Data & Analytics", "Early-Career Professionals",
             "Tier 1", "2025-12-01", "2026-09-30", ("South", "West", "North")),
    Campaign("KL-LI-001", "LinkedIn - Mid-Career Data Upskilling", "Professional Network",
             "LinkedIn", "Prospecting", "Lead Generation", "Data & Analytics",
             "Mid-Career Professionals", "Tier 1", "2025-10-01", "2026-09-30",
             ("West", "South", "North")),
    Campaign("KL-LI-002", "LinkedIn - Finance Leadership Programme", "Professional Network",
             "LinkedIn", "Prospecting", "Lead Generation", "Finance", "Mid-Career Professionals",
             "Tier 1", "2026-01-01", "2026-09-30", ("North", "West")),
    Campaign("KL-EM-001", "Email - Lead Nurture Series", "Email", "Email (in-house)",
             "Retargeting", "Conversion", "Software Development", "Early-Career Professionals",
             "Tier 2", "2025-10-01", "2026-09-30", ("East", "North", "South", "West", "Central")),
    Campaign("KL-EM-002", "Email - Alumni Upsell", "Email", "Email (in-house)", "Retargeting",
             "Conversion", "Finance", "Mid-Career Professionals", "Tier 1", "2025-10-01",
             "2026-09-30", ("North", "West")),
    Campaign("KL-AFF-001", "Affiliate - Education Portals", "Affiliate", "Affiliate Network",
             "Prospecting", "Conversion", "Software Development", "Students", "Tier 3",
             "2025-10-01", "2026-09-30", ("East", "Central", "North", "South")),
    Campaign("KL-PROMO-001", "Diwali Festive Sale", "Paid Social", "Meta", "Promotion",
             "Conversion", "Digital Marketing", "Early-Career Professionals", "Tier 2",
             "2025-10-01", "2025-11-05", ("West", "North", "South", "East", "Central")),
    Campaign("KL-PROMO-002", "New Year Career Reset", "Paid Search", "Google Ads", "Promotion",
             "Conversion", "Software Development", "Early-Career Professionals", "Tier 1",
             "2025-12-26", "2026-01-31", ("South", "North", "West", "East", "Central")),
    Campaign("KL-PROMO-003", "Admissions Season Push", "Paid Search", "Google Ads", "Promotion",
             "Lead Generation", "Data & Analytics", "Students", "Tier 2", "2026-05-01",
             "2026-07-31", ("Central", "North", "South", "West", "East")),
]


@dataclass(frozen=True)
class PlantedAnomaly:
    """One deliberate anomaly. `effects` are multipliers applied inside the generator; the
    rest is written to ground_truth_anomalies.json so later phases can be tested against it."""
    anomaly_id: str
    metric: str
    direction: str             # "up" or "down" for the metric
    sentiment: str             # "negative" (a problem) or "positive" (an improvement)
    approx_magnitude_pct: float
    start: str
    end: str
    story: str
    effects: dict = field(default_factory=dict)
    campaign_ids: tuple[str, ...] | None = None   # None = any campaign
    platform: str | None = None                   # None = any platform
    region: str | None = None                     # None = all regions
    secondary_metrics: tuple[str, ...] = ()


# Effect keys: budget, util (share of budget spent), cpc, ctr, opens (email impressions),
# click_tracking (share of real clicks that get recorded), lead_rate, qual_rate, conv_rate.
# Windows avoid the seasonal peaks so a detector can separate them from normal seasonality.
ANOMALIES = [
    PlantedAnomaly(
        "A01", "cpl", "up", "negative", 120, "2026-02-09", "2026-02-22",
        "Creative fatigue: the same ad creative ran for months, so people stopped responding. "
        "Click-to-lead rate fell by more than half and cost per lead roughly doubled for two weeks.",
        effects={"lead_rate": 0.45}, campaign_ids=("KL-META-001",),
        secondary_metrics=("leads", "click_to_lead_rate")),
    PlantedAnomaly(
        "A02", "lead_to_conversion_rate", "down", "negative", -70, "2026-03-09", "2026-03-29",
        "Landing page problem: a site release broke the course checkout page, so leads kept "
        "coming but far fewer enrolled for three weeks.",
        effects={"conv_rate": 0.30}, campaign_ids=("KL-GS-001",),
        secondary_metrics=("conversions", "revenue", "cac")),
    PlantedAnomaly(
        "A03", "spend", "up", "negative", 350, "2026-04-14", "2026-04-16",
        "Budget misconfiguration: someone typed a daily budget with an extra digit, so spend "
        "jumped about 4.5x for three days before it was caught.",
        effects={"budget": 4.5, "util": 1.05}, campaign_ids=("KL-YT-001",),
        secondary_metrics=("budget", "impressions")),
    PlantedAnomaly(
        "A04", "revenue", "down", "negative", -55, "2026-08-03", "2026-08-23",
        "Regional payment issue: the payment gateway used for East-region customers failed for "
        "many transactions, so enrolments and revenue in East fell by about half for three weeks.",
        effects={"conv_rate": 0.45}, region="East",
        secondary_metrics=("conversions", "roas")),
    PlantedAnomaly(
        "A05", "clicks", "down", "negative", -99, "2026-09-08", "2026-09-09",
        "Tracking outage: a broken tag on Google Ads landing pages meant clicks, leads and "
        "enrolments were not recorded for two days, while spend continued as normal.",
        effects={"click_tracking": 0.01, "lead_rate": 0.10, "conv_rate": 0.0},
        platform="Google Ads", secondary_metrics=("leads", "conversions")),
    PlantedAnomaly(
        "A06", "conversions", "up", "positive", 180, "2026-03-23", "2026-04-12",
        "Positive change: a new webinar-based lead form on LinkedIn attracted far better "
        "prospects, so leads and enrolments rose sharply and CPL fell for three weeks.",
        effects={"lead_rate": 1.8, "conv_rate": 1.6}, campaign_ids=("KL-LI-001",),
        secondary_metrics=("cpl", "leads")),
    PlantedAnomaly(
        "A07", "qualified_lead_rate", "down", "negative", -75, "2026-02-23", "2026-03-06",
        "Low-quality affiliate traffic: one affiliate partner pushed incentivised sign-ups. "
        "Leads tripled but very few were genuine, so the qualified-lead rate collapsed.",
        effects={"lead_rate": 3.0, "qual_rate": 0.25}, campaign_ids=("KL-AFF-001",),
        secondary_metrics=("leads",)),
    PlantedAnomaly(
        "A08", "impressions", "down", "negative", -80, "2025-12-01", "2025-12-07",
        "Email deliverability problem: a sender-reputation issue sent most nurture emails to "
        "spam for a week, so opens (impressions) and clicks fell by about 80%.",
        effects={"opens": 0.20}, campaign_ids=("KL-EM-001",),
        secondary_metrics=("clicks", "leads")),
    PlantedAnomaly(
        "A09", "cpc", "up", "negative", 90, "2026-08-17", "2026-08-30",
        "Competitor bidding: a rival ed-tech brand bid aggressively on software-course "
        "keywords in the South, so CPC nearly doubled and clicks fell at the same spend.",
        effects={"cpc": 1.9}, campaign_ids=("KL-GS-002",), region="South",
        secondary_metrics=("clicks", "cpl")),
    PlantedAnomaly(
        "A10", "spend", "down", "negative", -95, "2026-02-02", "2026-02-06",
        "Accidental pause: the West-region retargeting ad set was paused by mistake during an "
        "account clean-up, so spend and results nearly stopped for five days.",
        effects={"util": 0.05}, campaign_ids=("KL-META-002",), region="West",
        secondary_metrics=("clicks", "conversions")),
]


# ---------------------------------------------------------------------------------------------
# Seasonality and anomaly helpers
# ---------------------------------------------------------------------------------------------

def seasonal_demand(dates: pd.DatetimeIndex) -> np.ndarray:
    """Demand multiplier per day: monthly levels smoothed between mid-months, plus a Diwali
    bump and a late-December holiday dip."""
    # Anchor each month's level on the 15th, then interpolate so changes are gradual.
    anchors = pd.date_range("2025-09-15", "2026-10-15", freq="MS") + pd.Timedelta(days=14)
    anchor_values = np.array([MONTH_DEMAND[d.month] for d in anchors])
    x = dates.values.astype("datetime64[D]").astype(np.int64)
    xp = anchors.values.astype("datetime64[D]").astype(np.int64)
    demand = np.interp(x, xp, anchor_values)

    # Diwali: festive buying + promotions -> a peak around the festival day.
    days_from_diwali = (dates - DIWALI_DAY).days.to_numpy()
    demand = demand * (1 + 0.35 * np.exp(-0.5 * (days_from_diwali / 5.0) ** 2))

    # Christmas / year-end holidays: people are away, activity drops.
    in_dip = (dates >= LATE_DECEMBER_DIP[0]) & (dates <= LATE_DECEMBER_DIP[1])
    demand = np.where(in_dip, demand * 0.70, demand)
    return demand


def anomaly_multipliers(campaign: Campaign, region: str, dates: pd.DatetimeIndex) -> dict:
    """Combine the effects of every planted anomaly that touches this campaign x region."""
    keys = ["budget", "util", "cpc", "ctr", "opens", "click_tracking",
            "lead_rate", "qual_rate", "conv_rate"]
    mult = {k: np.ones(len(dates)) for k in keys}
    for a in ANOMALIES:
        if a.campaign_ids is not None and campaign.campaign_id not in a.campaign_ids:
            continue
        if a.platform is not None and campaign.platform != a.platform:
            continue
        if a.region is not None and region != a.region:
            continue
        in_window = (dates >= pd.Timestamp(a.start)) & (dates <= pd.Timestamp(a.end))
        for key, value in a.effects.items():
            mult[key] = np.where(in_window, mult[key] * value, mult[key])
    return mult


# ---------------------------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------------------------

def generate_pair(campaign: Campaign, region: str, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the daily rows for one campaign in one region."""
    dates = pd.date_range(campaign.start, campaign.end, freq="D")
    n = len(dates)
    econ = CHANNEL_ECONOMICS[campaign.channel]
    channel = campaign.channel
    demand = seasonal_demand(dates) * REGION_DEMAND[region]
    is_weekend = dates.dayofweek.to_numpy() >= 5
    anom = anomaly_multipliers(campaign, region, dates)

    # --- Budget: planned per month; marketers raise budgets in high-demand months. ---------
    monthly_plan = pd.Series(seasonal_demand(dates), index=dates).groupby(dates.month).transform("mean").to_numpy()
    budget = np.round(BASE_DAILY_BUDGET[channel] * monthly_plan * anom["budget"])

    # --- Funnel rates (before per-day noise). -----------------------------------------------
    # Demand affects conversion more gently than budgets: sqrt keeps rates believable.
    demand_rate = np.sqrt(demand)
    lead_p = (econ["lead_rate"] * TYPE_LEAD[campaign.campaign_type]
              * OBJECTIVE_LEAD[campaign.objective] * SEGMENT_LEAD[campaign.customer_segment]
              * demand_rate * anom["lead_rate"])
    qual_p = econ["qual_rate"] * SEGMENT_QUAL[campaign.customer_segment] * anom["qual_rate"]
    conv_p = (econ["conv_rate"] * TYPE_CONV[campaign.campaign_type]
              * OBJECTIVE_CONV[campaign.objective] * SEGMENT_CONV[campaign.customer_segment]
              * demand_rate * anom["conv_rate"])
    if "weekend_lead" in econ:
        lead_p = np.where(is_weekend, lead_p * econ["weekend_lead"], lead_p)
    lead_p = np.clip(lead_p * rng.lognormal(0, 0.08, n), 0, 0.6)
    qual_p = np.clip(qual_p * rng.lognormal(0, 0.08, n), 0, 0.95)
    conv_p = np.clip(conv_p * rng.lognormal(0, 0.10, n), 0, 0.9)
    ctr = econ["ctr"] * anom["ctr"] * rng.lognormal(0, 0.10, n)
    if campaign.objective == "Awareness":
        ctr = ctr * 0.8

    # --- Spend, clicks and impressions: logic differs by channel type. ----------------------
    if channel == "Email":
        # Opens drive everything; spend is the email tool's cost per open, weekdays only.
        opens = econ["daily_opens"] * demand * anom["opens"] * rng.lognormal(0, 0.10, n)
        opens = np.where(is_weekend, opens * econ["weekend_open_share"], opens)
        impressions = rng.poisson(opens)
        spend = np.where(is_weekend, 0.0,
                         np.minimum(impressions * econ["cost_per_open"], budget * 1.05))
        true_clicks = rng.binomial(impressions, np.clip(ctr, 0, 1))
    elif channel == "Affiliate":
        # Traffic comes from partner sites; Kalpa pays per result (see spend below).
        expected_clicks = econ["daily_clicks"] * demand * rng.lognormal(0, 0.12, n)
        true_clicks = rng.poisson(expected_clicks)
        impressions = np.ceil(true_clicks / np.clip(ctr, 1e-4, 1)).astype(np.int64)
        spend = None  # decided after conversions are known
    else:
        # Paid media: spend a share of budget, buy clicks at the day's CPC.
        weekend_factor = np.where(is_weekend, econ["weekend_spend"], 1.0)
        util = np.clip(0.90 * weekend_factor * rng.lognormal(0, 0.06, n), 0.30, 1.05)
        util = np.clip(util * anom["util"], 0.0, 1.08)
        spend = budget * util
        cpc = (econ["cpc"] * TIER_CPC[campaign.city_tier] * REGION_CPC[region]
               * OBJECTIVE_CPC[campaign.objective] * anom["cpc"] * rng.lognormal(0, 0.08, n))
        if channel == "Paid Search" and campaign.campaign_type == "Brand":
            cpc = cpc * BRAND_SEARCH_CPC
        true_clicks = rng.poisson(spend / cpc)
        impressions = np.maximum(np.ceil(true_clicks / np.clip(ctr, 1e-4, 1)), true_clicks).astype(np.int64)

    # Tracking outage: real clicks happened (and were paid for) but most were not recorded.
    clicks = rng.binomial(true_clicks, np.clip(anom["click_tracking"], 0, 1))
    leads = rng.binomial(clicks, lead_p)
    qualified = rng.binomial(leads, qual_p)
    conversions = rng.binomial(qualified, conv_p)

    if channel == "Affiliate":
        spend = (conversions * econ["payout_per_conversion"] + leads * econ["fee_per_lead"]
                 + econ["platform_fee"])
    spend = np.round(np.asarray(spend, dtype=float), 2)

    lo, hi = econ["freq"]
    frequency = rng.uniform(lo, hi, n)
    reach = np.floor(impressions / frequency).astype(np.int64)

    # --- Revenue and gross profit. -----------------------------------------------------------
    price_lo, price_hi = CATEGORY_PRICE_RANGE[campaign.course_category]
    list_price = ((price_lo + (price_hi - price_lo) * SEGMENT_PRICE_POSITION[campaign.customer_segment])
                  * SEGMENT_PRICE_PREMIUM[campaign.customer_segment])
    if campaign.campaign_type == "Promotion":
        in_diwali = (dates >= DIWALI_PROMO_WINDOW[0]) & (dates <= DIWALI_PROMO_WINDOW[1])
        discount = np.where(in_diwali, PROMOTION_DISCOUNT + DIWALI_EXTRA_DISCOUNT, PROMOTION_DISCOUNT)
    else:
        discount = rng.uniform(0, MAX_COUPON_DISCOUNT, n)
    # Several enrolments on one row average out, so the noise shrinks with volume.
    noise = np.exp(rng.normal(0, 0.10, n) / np.sqrt(np.maximum(conversions, 1)))
    revenue = np.round(conversions * list_price * (1 - discount) * noise)
    # Delivery cost per enrolment is fixed at list_price x (1 - margin), so discounts eat margin.
    delivery_cost = conversions * list_price * (1 - CATEGORY_MARGIN[campaign.course_category])
    gross_profit = np.round(np.maximum(revenue - delivery_cost, 0))

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "campaign_id": campaign.campaign_id,
        "campaign_name": campaign.campaign_name,
        "campaign_type": campaign.campaign_type,
        "objective": campaign.objective,
        "channel": channel,
        "platform": campaign.platform,
        "region": region,
        "city_tier": campaign.city_tier,
        "customer_segment": campaign.customer_segment,
        "course_category": campaign.course_category,
        "budget": budget,
        "spend": spend,
        "impressions": impressions,
        "reach": reach,
        "clicks": clicks,
        "leads": leads,
        "qualified_leads": qualified,
        "conversions": conversions,
        "revenue": revenue,
        "gross_profit": gross_profit,
    })


def select_pairs(row_target: int) -> list[tuple[int, int]]:
    """Choose which (campaign, region) pairs to generate.

    Each campaign lists its regions in priority order. We always keep every campaign's first
    region, then add second regions, third regions, ... while the row total stays within the
    target. This keeps all campaigns and channels present even for small targets.
    """
    candidates = []
    for c_idx, c in enumerate(CAMPAIGNS):
        days = len(pd.date_range(c.start, c.end, freq="D"))
        for rank, region in enumerate(c.regions):
            candidates.append((rank, c_idx, REGIONS.index(region), days))
    candidates.sort()

    chosen, total = [], 0
    for rank, c_idx, r_idx, days in candidates:
        if rank == 0 or total + days <= row_target:
            chosen.append((c_idx, r_idx))
            total += days
    return chosen


def generate_clean(seed: int = DEFAULT_SEED, row_target: int = DEFAULT_ROW_TARGET) -> pd.DataFrame:
    frames = []
    for c_idx, r_idx in select_pairs(row_target):
        # One independent random stream per pair: adding/removing a pair never changes others.
        rng = np.random.default_rng([seed, c_idx, r_idx])
        frames.append(generate_pair(CAMPAIGNS[c_idx], REGIONS[r_idx], rng))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "campaign_id", "region"], kind="stable").reset_index(drop=True)
    for col in ["budget", "impressions", "reach", "clicks", "leads", "qualified_leads",
                "conversions", "revenue", "gross_profit"]:
        df[col] = df[col].astype(np.int64)
    return df[CLEAN_COLUMNS]


# ---------------------------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------------------------

def indian_grouping(value: float, decimals: int = 0) -> str:
    """Write a number with Indian digit grouping (12,34,567). Used only to create messy text
    values here; the app's display formatting will live in utils/formatting.py."""
    text = f"{abs(value):.{decimals}f}"
    whole, _, frac = text.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    sign = "-" if value < 0 else ""
    return sign + whole + (f".{frac}" if frac else "")


def make_messy(clean: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Copy the clean data and inject realistic data-quality problems.

    Every problem uses its own set of rows (no row gets two problems), so each can be checked
    independently. Returns the messy frame and a ground-truth record of what was injected.
    `row_index` in the ground truth is the 0-based data row as pandas reads it
    (Excel row number = row_index + 2 because of the header row).
    """
    rng = np.random.default_rng([seed, 999])
    messy = clean.copy().astype(object)
    messy["date"] = pd.to_datetime(clean["date"]).astype(object)  # real Excel dates by default
    messy["_rid"] = np.arange(len(clean))
    used: set[int] = set()
    issues: list[dict] = []

    def pick(count: int, mask: pd.Series | None = None) -> list[int]:
        eligible = np.arange(len(clean)) if mask is None else np.flatnonzero(mask.to_numpy())
        eligible = np.array([i for i in eligible if i not in used])
        chosen = sorted(rng.choice(eligible, size=count, replace=False).tolist())
        used.update(chosen)
        return chosen

    # 1. Inconsistent platform labels on Meta rows.
    for variant in ["facebook", "FB", "Meta Ads", "meta "]:
        rows = pick(40, clean["platform"] == "Meta")
        messy.loc[rows, "platform"] = variant
        issues.append(dict(type="inconsistent_label", column="platform", clean_value="Meta",
                           messy_value=variant, rids=rows))

    # 2. Whitespace and mixed case in categories.
    channel_variants = {"Paid Search": "paid search", "Paid Social": " Paid social",
                        "Video": "VIDEO ", "Email": "email  "}
    for clean_value, variant in channel_variants.items():
        rows = pick(15, clean["channel"] == clean_value)
        messy.loc[rows, "channel"] = variant
        issues.append(dict(type="whitespace_or_case", column="channel", clean_value=clean_value,
                           messy_value=variant, rids=rows))
    region_variants = {"North": " north", "South": "SOUTH ", "West": "West  ", "Central": "central"}
    for clean_value, variant in region_variants.items():
        rows = pick(15, clean["region"] == clean_value)
        messy.loc[rows, "region"] = variant
        issues.append(dict(type="whitespace_or_case", column="region", clean_value=clean_value,
                           messy_value=variant, rids=rows))

    # 3. Mixed date formats stored as text.
    dates = pd.to_datetime(clean["date"])
    formats = {
        "iso_text": lambda d: d.strftime("%Y-%m-%d"),
        "day_first_text": lambda d: d.strftime("%d/%m/%Y"),
        "d_mon_yyyy_text": lambda d: f"{d.day}-{d.strftime('%b-%Y')}",
    }
    for name, fmt in formats.items():
        rows = pick(150)
        messy.loc[rows, "date"] = [fmt(dates[i]) for i in rows]
        issues.append(dict(type="mixed_date_format", column="date", format=name, rids=rows,
                           note="day_first_text means DD/MM/YYYY" if name == "day_first_text" else None))

    # 4. Money stored as text.
    money_formats = {
        "rupee_symbol_indian_grouping": lambda v: "₹" + indian_grouping(round(v)),
        "indian_grouping_2dp": lambda v: indian_grouping(v, 2),
        "inr_prefix": lambda v: f"INR {round(v)}",
    }
    for column in ["spend", "revenue"]:
        positive = clean[column] > 1000
        for name, fmt in money_formats.items():
            rows = pick(30, positive)
            messy.loc[rows, column] = [fmt(float(clean.at[i, column])) for i in rows]
            issues.append(dict(type="money_as_text", column=column, format=name, rids=rows,
                               clean_values={i: float(clean.at[i, column]) for i in rows}))

    # 5. Missing revenue (blank and "N/A").
    rows = pick(40)
    messy.loc[rows, "revenue"] = None
    issues.append(dict(type="missing_value", column="revenue", messy_value="(blank)", rids=rows,
                       clean_values={i: float(clean.at[i, "revenue"]) for i in rows}))
    rows = pick(40)
    messy.loc[rows, "revenue"] = "N/A"
    issues.append(dict(type="missing_value", column="revenue", messy_value="N/A", rids=rows,
                       clean_values={i: float(clean.at[i, "revenue"]) for i in rows}))

    # 6. Impossible values.
    rows = pick(12, clean["spend"] > 0)
    messy.loc[rows, "spend"] = [-float(clean.at[i, "spend"]) for i in rows]
    issues.append(dict(type="impossible_value", column="spend", rule="negative spend", rids=rows,
                       clean_values={i: float(clean.at[i, "spend"]) for i in rows}))
    rows = pick(12, clean["impressions"] > 10)
    messy.loc[rows, "clicks"] = [int(clean.at[i, "impressions"]) * 3 for i in rows]
    issues.append(dict(type="impossible_value", column="clicks", rule="clicks > impressions",
                       rids=rows, clean_values={i: int(clean.at[i, "clicks"]) for i in rows}))

    # 7. Exact duplicate rows: copies of untouched rows inserted at random positions.
    dup_sources = pick(25)
    copies = messy.iloc[dup_sources].copy()
    order = list(range(len(messy)))
    for src in dup_sources:
        order.insert(int(rng.integers(0, len(order) + 1)), src)
    messy = messy.iloc[order].reset_index(drop=True)

    # 8. Junk column (mostly empty, a few stray notes).
    junk = np.full(len(messy), None, dtype=object)
    junk_rows = rng.choice(len(messy), size=20, replace=False)
    junk[junk_rows] = rng.choice(["check", "??", "old", "x"], size=20)
    messy["Column1"] = junk

    # Map original row ids -> final positions so ground truth points at the messy file.
    positions: dict[int, list[int]] = {}
    for pos, rid in enumerate(messy["_rid"].tolist()):
        positions.setdefault(rid, []).append(pos)
    messy = messy.drop(columns="_rid")

    ground_truth_issues = []
    for issue in issues:
        record = {k: v for k, v in issue.items() if k not in ("rids", "clean_values") and v is not None}
        record["count"] = len(issue["rids"])
        record["row_index"] = [positions[r][0] for r in issue["rids"]]
        if "clean_values" in issue:
            record["clean_values"] = [issue["clean_values"][r] for r in issue["rids"]]
        ground_truth_issues.append(record)
    ground_truth_issues.append(dict(
        type="exact_duplicate_row", count=len(dup_sources),
        row_index_pairs=[positions[r][:2] for r in dup_sources],
        note="Each pair is [first occurrence, duplicate]. Keep one, drop the other."))
    ground_truth_issues.append(dict(
        type="junk_column", column="Column1", count=1,
        note="Mostly empty column with stray notes; should be dropped."))

    # 9. Renamed headers.
    renamed = {"spend": "Amount Spent (INR)", "conversions": "Enrollments"}
    messy = messy.rename(columns=renamed)
    ground_truth_issues.append(dict(type="renamed_header", count=len(renamed),
                                    renames=[{"clean": k, "messy": v} for k, v in renamed.items()]))

    # 10. "Grand Total" row at the bottom (sums of the clean numbers).
    total_row = {col: None for col in messy.columns}
    total_row["campaign_id"] = "Grand Total"
    for col in ["budget", "spend", "impressions", "clicks", "leads", "qualified_leads",
                "conversions", "revenue", "gross_profit"]:
        total_row[renamed.get(col, col)] = float(clean[col].sum())
    messy = pd.concat([messy, pd.DataFrame([total_row])], ignore_index=True)
    ground_truth_issues.append(dict(type="total_row", count=1, row_index=[len(messy) - 1],
                                    note="Summary row labelled 'Grand Total'; must be removed."))

    summary = {
        "file": "marketing_messy.xlsx",
        "source": "marketing_clean.csv",
        "clean_row_count": len(clean),
        "messy_row_count": len(messy),
        "row_index_note": "0-based data row as read by pandas; Excel row = row_index + 2.",
        "issues": ground_truth_issues,
    }
    return messy, summary


def make_meta_export(clean: pd.DataFrame) -> pd.DataFrame:
    """Meta-only extract in the style of an Ads Manager export (summed across regions).
    'Results' is deliberately ambiguous: here it means leads."""
    meta = clean[clean["platform"] == "Meta"]
    grouped = (meta.groupby(["date", "campaign_name"], as_index=False)
               [["spend", "impressions", "clicks", "leads"]].sum())
    grouped["spend"] = grouped["spend"].round(2)
    return grouped.rename(columns={
        "date": "Day", "campaign_name": "Campaign name", "spend": "Amount spent (INR)",
        "impressions": "Impressions", "clicks": "Link clicks", "leads": "Results",
    })


def make_test_slices(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Small, predictable slices: all rows of a region over a date range, cut to size."""
    def slice_(region_list, start, end, n):
        part = clean[clean["region"].isin(region_list) & (clean["date"] >= start) & (clean["date"] <= end)]
        return part.head(n).reset_index(drop=True)
    return {
        "test_a_small.csv": slice_(["North"], "2026-02-02", "2026-02-10", 40),
        "test_b_medium.csv": slice_(["North"], "2026-02-01", "2026-03-31", 300),
        "test_c_large.csv": slice_(["North", "South"], "2026-01-01", "2026-06-30", 1500),
    }


def anomalies_json() -> dict:
    records = []
    for a in ANOMALIES:
        campaigns = [c for c in CAMPAIGNS if a.campaign_ids is None or c.campaign_id in a.campaign_ids]
        if a.platform:
            campaigns = [c for c in campaigns if c.platform == a.platform]
        records.append({
            "id": a.anomaly_id,
            "metric": a.metric,
            "secondary_metrics": list(a.secondary_metrics),
            "channel": sorted({c.channel for c in campaigns}) if a.region is None or a.campaign_ids else "all",
            "platform": a.platform or (sorted({c.platform for c in campaigns}) if a.campaign_ids else "all"),
            "campaign_ids": list(a.campaign_ids) if a.campaign_ids else "all",
            "campaign_names": [c.campaign_name for c in campaigns] if a.campaign_ids else "all",
            "region": a.region or "all",
            "start_date": a.start,
            "end_date": a.end,
            "expected_direction": a.direction,
            "sentiment": a.sentiment,
            "approx_magnitude_pct": a.approx_magnitude_pct,
            "story": a.story,
        })
    return {
        "dataset": "marketing_clean.csv",
        "note": ("Magnitudes are approximate: compare the window against the few weeks before it "
                 "for the same campaign/region. Rows outside these windows follow normal "
                 "patterns (see expected_patterns.json)."),
        "anomalies": records,
    }


def patterns_json() -> dict:
    return {
        "note": "These are normal, expected patterns. A good detector should NOT flag them as anomalies.",
        "patterns": [
            {"id": "P01", "name": "Diwali festive peak", "start_date": "2025-10-10",
             "end_date": "2025-10-26", "effect": "Demand up to ~35% above normal around 20 Oct 2025; "
             "the 'Diwali Festive Sale' promotion runs 1 Oct - 5 Nov 2025 with 25% discounts in this window."},
            {"id": "P02", "name": "Late-December holiday dip", "start_date": "2025-12-22",
             "end_date": "2025-12-31", "effect": "Demand about 30% lower than early December."},
            {"id": "P03", "name": "January New-Year peak", "start_date": "2026-01-01",
             "end_date": "2026-01-31", "effect": "Demand ~30% above normal; budgets raised; "
             "'New Year Career Reset' promotion runs 26 Dec 2025 - 31 Jan 2026."},
            {"id": "P04", "name": "Admissions season (May-July)", "start_date": "2026-05-01",
             "end_date": "2026-07-31", "effect": "Demand 20-35% above normal, peaking in June; "
             "'Admissions Season Push' promotion runs in this window."},
            {"id": "P05", "name": "Weekend effect", "start_date": None, "end_date": None,
             "effect": "Search and LinkedIn spend lower on weekends; social/video slightly higher. "
             "Email has no sends on weekends (zero spend)."},
            {"id": "P06", "name": "Campaign flights", "start_date": None, "end_date": None,
             "effect": "Some campaigns start or end mid-year (see DATA_DICTIONARY.md). A campaign "
             "starting or stopping is not an anomaly."},
        ],
    }


# ---------------------------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------------------------

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_all(output_dir: Path, seed: int = DEFAULT_SEED, row_target: int = DEFAULT_ROW_TARGET) -> dict[str, int]:
    """Generate every file. Returns {file name: data row count}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = generate_clean(seed, row_target)
    counts: dict[str, int] = {}

    def to_csv(df: pd.DataFrame, name: str) -> None:
        df.to_csv(output_dir / name, index=False, lineterminator="\n")
        counts[name] = len(df)

    to_csv(clean, "marketing_clean.csv")
    to_csv(clean.drop(columns=["revenue", "gross_profit"]), "marketing_no_revenue.csv")
    to_csv(clean.drop(columns=["gross_profit"]), "marketing_no_margin.csv")
    to_csv(make_meta_export(clean), "meta_ads_export_style.csv")
    for name, part in make_test_slices(clean).items():
        to_csv(part, name)

    messy, quality_truth = make_messy(clean, seed)
    with pd.ExcelWriter(output_dir / "marketing_messy.xlsx", engine="openpyxl",
                        date_format="YYYY-MM-DD", datetime_format="YYYY-MM-DD") as writer:
        messy.to_excel(writer, index=False, sheet_name="Campaign Data")
    counts["marketing_messy.xlsx"] = len(messy)

    write_json(output_dir / "ground_truth_anomalies.json", anomalies_json())
    write_json(output_dir / "ground_truth_quality_issues.json", quality_truth)
    write_json(output_dir / "expected_patterns.json", patterns_json())
    return counts


def channel_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Totals and headline ratios by channel (ratios recomputed from totals, never averaged).
    Used only to print a realism check; the app's KPI logic will live in analytics/kpis.py."""
    t = clean.groupby("channel")[["spend", "leads", "conversions", "revenue"]].sum()
    t.loc["TOTAL"] = t.sum()
    t["CPL"] = t["spend"] / t["leads"]
    t["CAC"] = t["spend"] / t["conversions"]
    t["ROAS"] = t["revenue"] / t["spend"]
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kalpa Learning synthetic marketing data.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_TARGET,
                        help=f"approximate row target for the main dataset (default {DEFAULT_ROW_TARGET})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="folder to write files into (default data/sample)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default 42)")
    args = parser.parse_args()

    counts = write_all(args.output_dir, args.seed, args.rows)
    print(f"Wrote files to {args.output_dir}:")
    for name, n in counts.items():
        print(f"  {name:32s} {n:>6,d} rows")
    clean = pd.read_csv(args.output_dir / "marketing_clean.csv")
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 120):
        print("\nBy channel:")
        print(channel_summary(clean).to_string())


if __name__ == "__main__":
    main()
