"""Canonical field registry: the marketing fields the platform understands.

Each uploaded column is matched to one of these fields (see ingestion/mapper.py). Synonyms are
generous and include the header names used by Google Ads, Meta Ads Manager and LinkedIn
Campaign Manager exports. Synonyms are written in plain words; the mapper normalizes both
sides (lower-case, units in brackets removed, punctuation removed) before comparing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str                  # canonical column name used everywhere after mapping
    label: str                 # human-readable name for the UI and reports
    group: str                 # campaign / time / geography / customer / advertising / funnel / revenue / product
    ftype: str                 # date / number / money / category / id / text
    synonyms: tuple[str, ...]  # in order of preference (earlier wins a tie)
    critical: bool = False     # business-critical: an uncertain mapping must be confirmed by the user


FIELDS: tuple[FieldSpec, ...] = (
    # --- Campaign ------------------------------------------------------------------------------
    FieldSpec("campaign_name", "Campaign", "campaign", "category", (
        "campaign name", "campaign", "campaign title", "campaign group name", "campaign group",
        "ad campaign", "campaign_name")),
    FieldSpec("campaign_id", "Campaign ID", "campaign", "id", (
        "campaign id", "campaign code", "campaign key", "campaign number")),
    FieldSpec("campaign_type", "Campaign type", "campaign", "category", (
        "campaign type", "campaign category", "campaign subtype", "advertising channel sub type")),
    FieldSpec("objective", "Objective", "campaign", "category", (
        "objective", "campaign objective", "goal", "marketing objective", "campaign goal")),
    FieldSpec("channel", "Channel", "campaign", "category", (
        "channel", "marketing channel", "media channel", "channel name", "medium",
        "channel group", "default channel group", "advertising channel type", "source medium",
        "lead source")),
    FieldSpec("platform", "Platform", "campaign", "category", (
        "platform", "ad platform", "publisher", "network", "ad network", "publisher platform",
        "source", "site")),

    # --- Time ----------------------------------------------------------------------------------
    FieldSpec("date", "Date", "time", "date", (
        "date", "day", "reporting starts", "report date", "reporting date", "start date",
        "activity date", "transaction date", "order date", "event date", "day date"),
        critical=True),
    FieldSpec("week", "Week", "time", "text", ("week", "week number", "week of", "week start", "iso week")),
    FieldSpec("month", "Month", "time", "text", ("month", "month name", "month of year", "reporting month")),
    FieldSpec("quarter", "Quarter", "time", "text", ("quarter", "fiscal quarter", "qtr")),
    FieldSpec("year", "Year", "time", "number", ("year", "fiscal year", "fy")),

    # --- Geography -----------------------------------------------------------------------------
    FieldSpec("country", "Country", "geography", "category", ("country", "country territory", "nation")),
    FieldSpec("state", "State", "geography", "category", ("state", "province", "state province", "region state")),
    FieldSpec("city", "City", "geography", "category", ("city", "town", "location", "city name")),
    FieldSpec("region", "Region", "geography", "category", ("region", "zone", "sales region", "territory", "area")),
    FieldSpec("market", "Market", "geography", "category", ("market", "geo", "geography", "market name")),
    FieldSpec("city_tier", "City tier", "geography", "category", ("city tier", "tier", "city class", "town tier")),

    # --- Customer ------------------------------------------------------------------------------
    FieldSpec("customer_id", "Customer ID", "customer", "id", ("customer id", "client id", "user id", "customer code")),
    FieldSpec("customer_segment", "Customer segment", "customer", "category", (
        "customer segment", "segment", "audience segment", "audience", "target audience",
        "customer group", "persona")),
    FieldSpec("age", "Age", "customer", "category", ("age", "age group", "age band", "age range")),
    FieldSpec("gender", "Gender", "customer", "category", ("gender", "sex")),
    FieldSpec("customer_type", "Customer type", "customer", "category", ("customer type", "client type", "account type")),
    FieldSpec("new_returning", "New / returning", "customer", "category", (
        "new returning", "new vs returning", "new or returning", "returning customer", "customer status")),

    # --- Advertising ---------------------------------------------------------------------------
    FieldSpec("spend", "Spend", "advertising", "money", (
        # "Cost" alone is how Google Ads names spend. Total/product/operating costs are NOT
        # spend (see NOT_SPEND_COSTS); "marketing cost" is only a fallback (MARKETING_COSTS).
        "spend", "cost", "amount spent", "ad spend", "media spend", "total spend", "total spent",
        "spent", "media cost", "ad cost", "advertising cost", "advertising spend", "ad costs",
        "spend inr", "cost inr", "amount spent inr"),
        critical=True),
    FieldSpec("budget", "Budget", "advertising", "money", (
        "budget", "daily budget", "campaign budget", "planned budget", "budget amount", "allocated budget")),
    FieldSpec("impressions", "Impressions", "advertising", "number", (
        "impressions", "impr", "impression", "total impressions", "ad impressions")),
    FieldSpec("reach", "Reach", "advertising", "number", (
        "reach", "unique reach", "people reached", "accounts reached", "unique users")),
    FieldSpec("clicks", "Clicks", "advertising", "number", (
        "clicks", "link clicks", "total clicks", "clicks all", "outbound clicks", "click",
        "website clicks", "landing page clicks")),
    FieldSpec("engagements", "Engagements", "advertising", "number", (
        "engagements", "engagement", "post engagement", "post engagements", "interactions", "total engagements")),
    FieldSpec("views", "Views", "advertising", "number", (
        "views", "video views", "thruplays", "video plays", "3 second video plays", "views count")),

    # --- Funnel --------------------------------------------------------------------------------
    FieldSpec("leads", "Leads", "funnel", "number", (
        "leads", "lead", "total leads", "form submissions", "lead form submissions", "form fills",
        "enquiries", "inquiries", "lead count", "on facebook leads", "leads form",
        "lead form completions", "enquiry count"),
        critical=True),
    FieldSpec("qualified_leads", "Qualified leads", "funnel", "number", (
        "qualified leads", "qualified lead", "mql", "mqls", "sql", "sqls", "marketing qualified leads",
        "sales qualified leads", "qualified", "qualified_leads")),
    FieldSpec("opportunities", "Opportunities", "funnel", "number", (
        "opportunities", "opportunity", "opps", "pipeline opportunities", "deals")),
    FieldSpec("conversions", "Conversions", "funnel", "number", (
        "conversions", "conversion", "conv", "total conversions", "enrollments", "enrolments",
        "enrolled", "purchases", "admissions", "signups converted", "website purchases",
        "all conv", "deals won", "won deals", "closed won", "closed won deals"),
        critical=True),
    FieldSpec("customers", "Customers", "funnel", "number", (
        "customers", "new customers", "customer count", "customers acquired", "buyers", "paying customers")),
    FieldSpec("orders", "Orders", "funnel", "number", (
        "orders", "order count", "number of orders", "transactions", "total orders")),

    # --- Revenue -------------------------------------------------------------------------------
    FieldSpec("revenue", "Revenue", "revenue", "money", (
        # Net revenue (after discounts) is preferred over gross when both exist: earlier = preferred.
        "net revenue", "net sales", "revenue net", "net revenue inr",
        "revenue", "total revenue", "conversion value", "conv value", "purchase conversion value",
        "purchases conversion value", "website purchases conversion value", "sales amount",
        "sales revenue", "gross revenue", "income", "turnover", "gmv", "revenue inr",
        "total conversion value", "all conv value", "deal value", "won deal value",
        "closed won value", "won revenue"),
        critical=True),
    FieldSpec("average_order_value", "Average order value", "revenue", "money", (
        "average order value", "aov", "avg order value", "order value", "average basket value")),
    FieldSpec("gross_profit", "Gross profit", "revenue", "money", (
        # A plain "Profit" / "Net profit" column may already subtract marketing cost, so it is
        # never mapped here automatically (see PROFIT_HEADERS).
        "gross profit", "gross margin amount", "gross_profit")),
    FieldSpec("margin", "Margin %", "revenue", "number", (
        "margin", "gross margin", "margin percent", "margin pct", "gross margin percent", "profit margin")),

    # --- Product -------------------------------------------------------------------------------
    FieldSpec("product", "Product / course", "product", "category", (
        "product", "product name", "course", "course name", "item", "item name", "programme", "program")),
    FieldSpec("product_category", "Product / course category", "product", "category", (
        "product category", "course category", "category", "product type", "course type",
        "product line", "programme category", "program category")),
    FieldSpec("sku", "SKU", "product", "id", ("sku", "product id", "item id", "product code", "course id")),
)

FIELD_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}
CRITICAL_FIELDS = tuple(f.name for f in FIELDS if f.critical)

# Header words that are genuinely ambiguous: they are NEVER mapped automatically, only offered
# as candidates for the user to choose from. (Meta's "Results" can mean leads or purchases.)
AMBIGUOUS_HEADERS: dict[str, tuple[str, ...]] = {
    "results": ("leads", "conversions"),
    "result": ("leads", "conversions"),
    "sales": ("revenue", "conversions", "orders"),
    "amount": ("spend", "revenue"),
    "value": ("revenue", "average_order_value"),
    "total": ("spend", "revenue"),
    "actions": ("leads", "conversions", "engagements"),
    "goal completions": ("leads", "conversions"),
    "key events": ("leads", "conversions"),        # GA4's new name for conversions: any event
    "signups": ("leads", "conversions"),
    "sign ups": ("leads", "conversions"),
    "registrations": ("leads", "conversions"),
}

# Ratio / derived columns found in platform exports. They are recognised but not mapped:
# the platform always recomputes ratios from summed totals (never averages them).
DERIVED_METRIC_HEADERS = {
    "ctr", "click through rate", "cpc", "avg cpc", "average cpc", "cost per click", "cpm",
    "cost per 1000 impressions", "cpl", "cost per lead", "cost per result", "cost per results",
    "cpa", "cost per acquisition", "cost per conversion", "cost conv", "roas",
    "purchase roas", "return on ad spend", "conversion rate", "conv rate", "cvr",
    "frequency", "roi", "cac", "result rate", "ctr all", "ctr link click through rate",
}

# Known alternative spellings of category LABELS (not headers), used when checking for
# inconsistent values such as "FB" / "facebook" / "Meta Ads". Keys and values are normalized
# (lower-case, single spaces).
LABEL_ALIASES: dict[str, str] = {
    "fb": "meta", "facebook": "meta", "facebook ads": "meta", "meta ads": "meta",
    "meta platforms": "meta", "fb ads": "meta",
    "google": "google ads", "adwords": "google ads", "google adwords": "google ads",
    "yt": "youtube", "youtube ads": "youtube",
    "linkedin ads": "linkedin", "li": "linkedin",
    "ppc": "paid search", "sem": "paid search", "search": "paid search",
    "social": "paid social", "paid socials": "paid social",
    "e-mail": "email", "e mail": "email", "emailer": "email",
}

# Preferred display spelling for well-known channel/platform labels, keyed by the normalized
# label AFTER aliases (so "FB", "facebook" and "Meta Ads" all become "Meta").
CANONICAL_LABELS: dict[str, str] = {
    "meta": "Meta", "google ads": "Google Ads", "youtube": "YouTube", "linkedin": "LinkedIn",
    "paid search": "Paid Search", "paid social": "Paid Social", "email": "Email",
    "video": "Video", "affiliate": "Affiliate", "professional network": "Professional Network",
}

# Fields holding whole-number counts (stored as integers after cleaning).
COUNT_FIELDS = ("impressions", "reach", "clicks", "engagements", "views", "leads",
                "qualified_leads", "opportunities", "conversions", "customers", "orders")

# Channel type: "paid" media is bought per impression/click/result; "owned" channels (email,
# SMS, organic ...) run on the company's own audience with mostly fixed costs. Owned channels
# are NOT ranked against paid media (their ROAS is not comparable: no auction, existing
# audience) and are left out of efficiency indices; they are reported on a separate line.
# Keys are normalized channel labels (lower-case). Anything not listed is treated as paid.
CHANNEL_TYPES: dict[str, str] = {
    "email": "owned", "sms": "owned", "whatsapp": "owned", "push notifications": "owned",
    "organic social": "owned", "organic search": "owned", "seo": "owned", "direct": "owned",
    "referral": "owned", "website": "owned",
}


def channel_type(channel) -> str:
    """'owned' or 'paid' for a channel label (case and spacing ignored)."""
    key = " ".join(str(channel).strip().lower().split())
    if key.startswith("email") or key.startswith("e-mail"):      # e.g. platform "Email (in-house)"
        return "owned"
    return CHANNEL_TYPES.get(key, "paid")


# --- Rules that keep mapping safe on any marketing file ----------------------------------------

# Words that mark a DERIVED column (a ratio, rate or average that the app recalculates itself).
# Such columns are never suggested for count or money fields.
DERIVED_WORDS = {"rate", "ratio", "percent", "percentage", "pct", "ctr", "cpc", "cpm", "cpl", "cpa",
                 "cac", "roas", "roi", "aov", "average", "avg", "per", "cvr", "share"}
# Fields that legitimately hold a ratio (exempt from the derived-word rule when the name matches).
RATIO_FIELDS = {"margin"}

# Costs that are NOT advertising spend.
NOT_SPEND_COSTS = {"total cost", "total costs", "product cost", "product costs", "cogs",
                   "cost of goods", "cost of goods sold", "operating cost", "operating costs",
                   "operating expense", "operating expenses", "opex", "shipping cost",
                   "fulfilment cost", "fulfillment cost", "unit cost", "cost price",
                   "production cost", "landed cost", "total expense", "total expenses"}

# Broader marketing costs: may include agency fees, salaries or tools. Used for Spend only when
# the file has no clear spend column, and then only as an uncertain suggestion.
MARKETING_COSTS = {"marketing cost", "marketing costs", "marketing spend", "marketing expense",
                   "marketing expenses", "promotion cost", "promotional cost", "investment"}

# Profit columns that may already have marketing cost subtracted: never mapped to gross profit
# automatically (ROI would subtract spend twice).
PROFIT_HEADERS = {"profit", "net profit", "operating profit", "net income", "profit amount",
                  "contribution", "margin amount", "net margin amount"}

