"""Mapping knowledge base: what Python knows about marketing column names and values.

The knowledge is kept as plain CSV files in this folder (open them in Excel to read or extend
them); this module only loads them. The mapper (ingestion/mapper.py) uses them after the
built-in synonyms in config/fields.py, which always keep priority.

    field_synonyms.csv       extra header names per canonical field, by source (Google Ads, Meta,
                             LinkedIn, YouTube, TikTok, Amazon Ads, Microsoft Ads, GA4, Shopify,
                             WooCommerce, HubSpot, Salesforce, Zoho, Mailchimp, Klaviyo, generic)
    recognised_columns.csv   columns the app knows but does not analyse (sessions, saves,
                             bounce counts, ...): shown as "Recognised: <name> (not used in this
                             analysis)" and never sent to Gemini
    derived_metrics.csv      extra ratio/derived headers (ACOS, CPV, Conv. value / cost, ...),
                             recalculated by the app and never mapped
    short_forms.csv          short forms used in hand-made sheets (amt, spnd, rev, qty, txn, ...)
    value_vocabulary.csv     known cell values: channels, platforms, regions, cities, devices,
                             customer types and campaign objectives

Alternative considered: keep everything as Python dictionaries in config/fields.py. CSV was
chosen so the knowledge can be reviewed and extended without editing code.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent


def _rows(file_name: str) -> list[dict[str, str]]:
    with open(KNOWLEDGE_DIR / file_name, encoding="utf-8", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


@lru_cache(maxsize=None)
def field_synonyms() -> tuple[tuple[str, str, str], ...]:
    """(field, header, source) in file order (earlier = preferred within a field)."""
    return tuple((r["field"], r["synonym"], r["source"]) for r in _rows("field_synonyms.csv"))


@lru_cache(maxsize=None)
def recognised_columns() -> dict[str, str]:
    """Known-but-not-analysed header -> the plain name shown to the user."""
    return {r["header"]: r["name"] for r in _rows("recognised_columns.csv")}


@lru_cache(maxsize=None)
def derived_metrics() -> tuple[str, ...]:
    return tuple(r["header"] for r in _rows("derived_metrics.csv"))


@lru_cache(maxsize=None)
def short_forms() -> dict[str, str]:
    return {r["short"]: r["full"] for r in _rows("short_forms.csv")}


@lru_cache(maxsize=None)
def value_vocabulary() -> dict[str, dict[str, str]]:
    """kind -> {known value (lower case): canonical value}."""
    out: dict[str, dict[str, str]] = {}
    for r in _rows("value_vocabulary.csv"):
        out.setdefault(r["kind"], {})[r["value"]] = r["canonical"]
    return out
