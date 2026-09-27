"""Build small, realistic marketing files with DIFFERENT shapes, to test field mapping on any
upload (not just one company's file). Seeded, so the files are identical on every run.

    python tests/fixtures/build_mapping_fixtures.py

Writes tests/fixtures/mapping/:
    google_ads.csv      Google Ads export: Day, Cost, Impr., Conv. value, own CTR/CPC/conv. rate
    meta_ads.csv        Meta Ads export: Amount spent (INR), Reach, Link clicks, own CTR/CPC/CPL/CPM
    ecommerce.csv       Shop data: Orders, Gross/Net revenue, Discount, Product/Total cost, Profit,
                        own CPC/ROAS/AOV columns
    leadgen_crm.csv     Lead-gen / CRM: "Marketing Cost" is the only cost (no Spend column),
                        MQLs, Deals Won, Deal Value, own CPL and lead-to-deal %
    agency_report.csv   Agency weekly report: percentages written as text ("2.35%"), and a CPC
                        column that includes a 15% agency fee (so it must NOT match Spend / Clicks)
    retail_columns.csv     The column layout of a real retail test file, with its formulas
                        (ROAS far above 50x and spend on "Organic Search" -> plausibility warnings)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "mapping"
DAYS = pd.date_range("2026-03-02", periods=28, freq="D")


def _base(rng, campaigns, spend_range, ctr, click_to_lead=None, impressions_per_rupee=(8, 14)):
    rows = []
    for day in DAYS:
        for name, channel in campaigns:
            spend = rng.uniform(*spend_range)
            impressions = int(spend * rng.uniform(*impressions_per_rupee))   # default CPM ~₹70-125
            clicks = max(1, int(impressions * ctr * rng.uniform(0.8, 1.2)))
            row = {"date": day, "campaign": name, "channel": channel, "spend": round(spend, 2),
                   "impressions": impressions, "clicks": clicks}
            if click_to_lead:
                row["leads"] = max(1, int(clicks * click_to_lead * rng.uniform(0.8, 1.2)))
            rows.append(row)
    return pd.DataFrame(rows)


def google_ads(rng) -> pd.DataFrame:
    b = _base(rng, [("Search - Brand", "Search"), ("Search - Generic", "Search"),
                    ("PMax - Catalogue", "Performance Max")], (3000, 9000), 0.045)
    conv = (b["clicks"] * rng.uniform(0.02, 0.04, len(b))).round().clip(lower=1)
    value = (conv * rng.uniform(1200, 1800, len(b))).round(2)
    return pd.DataFrame({
        "Day": b["date"].dt.strftime("%Y-%m-%d"), "Campaign": b["campaign"],
        "Campaign type": b["channel"], "Cost": b["spend"], "Impr.": b["impressions"],
        "Clicks": b["clicks"], "CTR": (b["clicks"] / b["impressions"] * 100).round(2).astype(str) + "%",
        "Avg. CPC": (b["spend"] / b["clicks"]).round(2), "Conversions": conv,
        "Conv. value": value, "Cost / conv.": (b["spend"] / conv).round(2),
        "Conv. rate": (conv / b["clicks"] * 100).round(2).astype(str) + "%"})


def meta_ads(rng) -> pd.DataFrame:
    b = _base(rng, [("Prospecting - Lookalike", "Facebook"), ("Retargeting - Site", "Instagram")],
              (2000, 6000), 0.012, click_to_lead=0.08)
    reach = (b["impressions"] * rng.uniform(0.55, 0.8, len(b))).astype(int)
    return pd.DataFrame({
        "Reporting starts": b["date"].dt.strftime("%Y-%m-%d"), "Campaign name": b["campaign"],
        "Platform": b["channel"], "Amount spent (INR)": b["spend"], "Impressions": b["impressions"],
        "Reach": reach, "Link clicks": b["clicks"],
        "CTR (link click-through rate)": (b["clicks"] / b["impressions"] * 100).round(2),
        "CPC (cost per link click) (INR)": (b["spend"] / b["clicks"]).round(2),
        "Leads": b["leads"], "Cost per lead (INR)": (b["spend"] / b["leads"]).round(2),
        "CPM (cost per 1,000 impressions) (INR)": (b["spend"] / b["impressions"] * 1000).round(2)})


def ecommerce(rng) -> pd.DataFrame:
    b = _base(rng, [("Summer Sale", "Google Ads"), ("New Arrivals", "Meta"), ("Loyalty", "Email")],
              (4000, 12000), 0.02)
    b.loc[b["channel"] == "Email", "spend"] = (b["spend"] * 0.1).round(2)     # owned: low cost
    orders = (b["clicks"] * rng.uniform(0.02, 0.04, len(b))).round().clip(lower=1)
    gross = (orders * rng.uniform(1500, 2500, len(b))).round(2)       # ROAS of a few x
    discount = (gross * rng.uniform(0.05, 0.15, len(b))).round(2)
    net = (gross - discount).round(2)
    product_cost = (net * 0.55).round(2)
    total_cost = (product_cost + b["spend"]).round(2)
    return pd.DataFrame({
        "Order Date": b["date"].dt.strftime("%d/%m/%Y"), "Channel": b["channel"],
        "Campaign": b["campaign"], "Ad Spend": b["spend"], "Impressions": b["impressions"],
        "Clicks": b["clicks"], "Orders": orders, "Gross Revenue": gross, "Discount": discount,
        "Net Revenue": net, "Product Cost": product_cost, "Total Cost": total_cost,
        "Profit": (net - total_cost).round(2), "CPC": (b["spend"] / b["clicks"]).round(2),
        "ROAS": (net / b["spend"]).round(2), "AOV": (net / orders).round(2)})


def leadgen_crm(rng) -> pd.DataFrame:
    b = _base(rng, [("Webinar Series", "LinkedIn"), ("Search - Demo", "Google Ads"),
                    ("Partner Referrals", "Referral")], (1500, 5000), 0.015, click_to_lead=0.12)
    mqls = (b["leads"] * rng.uniform(0.3, 0.5, len(b))).round()
    won = (b["leads"] * rng.uniform(0.05, 0.12, len(b))).round()
    return pd.DataFrame({
        "Date": b["date"].dt.strftime("%d-%b-%Y"), "Lead Source": b["channel"],
        "Campaign": b["campaign"], "Marketing Cost": b["spend"], "Clicks": b["clicks"],
        "Leads": b["leads"], "MQLs": mqls, "Deals Won": won,
        "Deal Value": (won * rng.uniform(5000, 12000, len(b))).round(2),
        "CPL": (b["spend"] / b["leads"]).round(2),
        "Lead-to-Deal %": (won / b["leads"] * 100).round(2)})


def agency_report(rng) -> pd.DataFrame:
    b = _base(rng, [("Diwali Push", "YouTube"), ("Always On", "Display")], (5000, 15000), 0.008)
    purchases = (b["clicks"] * rng.uniform(0.01, 0.03, len(b))).round().clip(lower=1)
    revenue = (purchases * rng.uniform(3000, 5000, len(b))).round(2)
    return pd.DataFrame({
        "Date": b["date"].dt.strftime("%Y-%m-%d"), "Platform": b["channel"],
        "Campaign": b["campaign"], "Media Spend": b["spend"], "Impressions": b["impressions"],
        "Clicks": b["clicks"], "CTR %": (b["clicks"] / b["impressions"] * 100).round(2).astype(str) + "%",
        # The agency adds its 15% fee into CPC, so it does NOT equal Media Spend / Clicks.
        "CPC": (b["spend"] * 1.15 / b["clicks"]).round(2),
        "Purchases": purchases, "Revenue": revenue, "ROAS": (revenue / b["spend"]).round(2)})


def retail_columns(rng) -> pd.DataFrame:
    """Same columns and formulas as the real retail file (verified by inspecting it)."""
    b = _base(rng, [("Campaign_001", "Meta"), ("Campaign_002", "Organic Search")],
              (500, 2000), 0.035, click_to_lead=0.18, impressions_per_rupee=(60, 90))  # like the real file: ROAS far above 50x
    n = len(b)
    orders = (b["leads"] * rng.uniform(0.15, 0.2, n)).round().clip(lower=1)
    gross = (orders * rng.uniform(3500, 4500, n)).round(2)
    discount = (gross * rng.uniform(0.02, 0.08, n)).round(2)
    net = (gross - discount).round(2)
    product_cost = (net * 0.5).round(2)
    marketing_cost = (b["spend"] * 1.1).round(2)
    total_cost = (product_cost + marketing_cost).round(2)
    profit = (net - total_cost).round(2)
    return pd.DataFrame({
        "Date": b["date"].dt.strftime("%Y-%m-%d"), "Campaign_ID": [f"C{i % 2 + 1:03d}" for i in range(n)],
        "Campaign_Name": b["campaign"], "Campaign_Type": "Conversion", "Channel": b["channel"],
        "Ad_Format": "Video", "City": "Pune", "Product": "Smartwatch", "Audience": "25-34",
        "Gender": "Female", "Campaign_Status": "Active", "Spend": b["spend"],
        "Impressions": b["impressions"], "Reach": (b["impressions"] * 0.7).astype(int),
        "Clicks": b["clicks"], "Engagements": (b["clicks"] * 1.5).astype(int), "Leads": b["leads"],
        "Add_to_Cart": (b["leads"] * 0.5).astype(int), "Orders": orders, "Units_Sold": orders * 2,
        "Returns": (orders * 0.05).round(), "Average_Order_Value": (gross / orders).round(2),
        "Gross_Revenue": gross, "Discount": discount, "Net_Revenue": net,
        "Product_Cost": product_cost, "Marketing_Cost": marketing_cost, "Total_Cost": total_cost,
        "Profit": profit, "CTR_%": (b["clicks"] / b["impressions"] * 100).round(2),
        "CPC": (b["spend"] / b["clicks"]).round(2), "CPL": (b["spend"] / b["leads"]).round(2),
        "Conversion_Rate_%": (orders / b["clicks"] * 100).round(2),
        "ROAS": (net / b["spend"]).round(2), "ROI_%": (profit / marketing_cost * 100).round(2),
        "Sales_Region": "West", "Device": "Mobile", "Landing_Page": "/offer",
        "Customer_Type": "New"})


BUILDERS = {"google_ads": google_ads, "meta_ads": meta_ads, "ecommerce": ecommerce,
            "leadgen_crm": leadgen_crm, "agency_report": agency_report, "retail_columns": retail_columns}


def build(out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (name, builder) in enumerate(BUILDERS.items()):
        path = out / f"{name}.csv"
        builder(np.random.default_rng(100 + i)).to_csv(path, index=False, lineterminator="\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
