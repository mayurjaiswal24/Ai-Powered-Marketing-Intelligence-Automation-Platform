"""The structure Gemini must answer in (validated with Pydantic before anything is shown).

Every item carries `evidence_ids` pointing at the evidence pack (E01, E02, ...), so each
statement can be traced back to verified numbers. Hypotheses must say how to validate them;
recommendations must name the metric to watch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InsightItem(BaseModel):
    text: str = Field(description="One or two plain-English sentences. Quote numbers exactly as "
                                  "they appear in the evidence.")
    evidence_ids: list[str] = Field(description="IDs of the evidence items that support this "
                                                "statement, e.g. ['E03', 'E07']. Never empty.")


class Hypothesis(InsightItem):
    validation_step: str = Field(description="A concrete check that would confirm or rule out "
                                             "this possible explanation.")


class Recommendation(InsightItem):
    priority: Literal["high", "medium", "low"]
    metric_to_watch: str = Field(description="The KPI to monitor to see whether the action worked.")


class AIInsights(BaseModel):
    executive_summary: InsightItem
    key_findings: list[InsightItem]
    performance_concerns: list[InsightItem]
    hypotheses: list[Hypothesis]
    investigation_areas: list[InsightItem]
    recommendations: list[Recommendation]
    limitations: list[InsightItem]


# Section order and the content label each section carries (SPEC §16).
SECTIONS = [
    ("executive_summary", "Executive summary", "AI interpretation"),
    ("key_findings", "Key findings", "AI interpretation"),
    ("performance_concerns", "Performance concerns", "AI interpretation"),
    ("hypotheses", "Hypotheses", "Hypothesis"),
    ("investigation_areas", "Investigation areas", "AI interpretation"),
    ("recommendations", "Recommendations", "Recommendation"),
    ("limitations", "Limitations", "AI interpretation"),
]
