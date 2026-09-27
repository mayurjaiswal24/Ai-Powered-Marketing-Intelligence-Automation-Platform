"""Hallucination control: check Gemini's answer against the evidence before showing it.

1. Every item must cite evidence IDs that exist in the pack. Unknown IDs are removed; an item
   left with no valid ID is DROPPED (it cannot be traced to verified numbers).
2. Every number the AI writes (₹4.2 Cr, 38%, 3.45x, 1,23,456 ...) must match a number in the
   evidence it cites, allowing for rounding. Otherwise the item is kept but marked UNVERIFIED
   with a warning, so a reader knows not to trust that figure.
3. A summary counts kept, verified, unverified and dropped items.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai.schemas import SECTIONS

# A number with optional sign, rupee sign, Indian/Western grouping, decimals and unit.
NUMBER = re.compile(
    r"(?<![A-Za-z0-9.])(?P<sign>[+-])?(?P<rupee>₹\s?)?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(?P<unit>Cr\b|crore\b|L\b|lakh\b|lakhs\b|%|x\b))?")
MULTIPLIER = {"Cr": 1e7, "crore": 1e7, "L": 1e5, "lakh": 1e5, "lakhs": 1e5}
ID_PATTERN = re.compile(r"^E\d{2,3}$")
RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class Number:
    value: float
    kind: str          # money / percent / ratio / plain
    tolerance: float   # half a unit of the last shown digit (rounding)
    text: str


def extract_numbers(text: str, checkable_only: bool = True) -> list[Number]:
    """Numbers in a piece of text. With checkable_only, small bare integers (like '3 weeks'),
    years and date days are skipped: they are not metric values."""
    found = []
    for m in NUMBER.finditer(text or ""):
        raw = m.group("num").rstrip(",")
        unit = m.group("unit") or ""
        rupee = bool(m.group("rupee"))
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        mult = MULTIPLIER.get(unit, 1.0)
        kind = "money" if rupee else "percent" if unit == "%" else "ratio" if unit == "x" else "plain"
        if checkable_only and kind == "plain":
            if "," not in raw and value < 100:
                continue                      # "3 campaigns", "4 weeks", "9 Feb"
            if "," not in raw and decimals == 0 and 1900 <= value <= 2100:
                continue                      # a year
        found.append(Number(value * mult, kind, 0.5 * 10 ** (-decimals) * mult, m.group(0).strip()))
    return found


def _matches(n: Number, evidence: list[Number]) -> bool:
    for e in evidence:
        compatible = n.kind == e.kind or {n.kind, e.kind} <= {"plain", "money"}
        if not compatible:
            continue
        a, b = abs(n.value), abs(e.value)
        if abs(a - b) <= n.tolerance + e.tolerance + 1e-9 or abs(a - b) <= RELATIVE_TOLERANCE * max(b, 1e-9):
            return True
    return False


@dataclass
class Evaluation:
    kept: int = 0
    verified: int = 0
    unverified: int = 0
    dropped: int = 0
    dropped_items: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"kept": self.kept, "verified": self.verified, "unverified": self.unverified,
                "dropped": self.dropped, "dropped_items": self.dropped_items, "warnings": self.warnings}


def evaluate(insights: dict, pack) -> tuple[dict, Evaluation]:
    """Return (checked insights, evaluation). Checked items gain `status`, `warnings` and `label`.
    `insights` is the validated AIInsights as a dict; `pack` is the EvidencePack that was sent."""
    ev = Evaluation()
    evidence_numbers = {eid: extract_numbers(" ".join([item.title, *item.facts.values()]), checkable_only=False)
                        for eid, item in pack.by_id.items()}
    checked: dict = {}
    for key, title, label in SECTIONS:
        raw = insights.get(key)
        items = raw if isinstance(raw, list) else ([raw] if raw else [])
        kept_items = []
        for item in items:
            result = _check_item(dict(item), label, pack, evidence_numbers, ev, title)
            if result is not None:
                kept_items.append(result)
        checked[key] = (kept_items[0] if kept_items else None) if key == "executive_summary" else kept_items
    return checked, ev


def _check_item(item: dict, label: str, pack, evidence_numbers, ev: Evaluation, section: str) -> dict | None:
    cited = [str(i).strip().upper() for i in item.get("evidence_ids") or []]
    valid = [i for i in dict.fromkeys(cited) if ID_PATTERN.match(i) and i in pack.by_id]
    unknown = [i for i in cited if i not in valid]
    warnings = []
    if not valid:
        ev.dropped += 1
        ev.dropped_items.append({"section": section, "text": item.get("text", ""),
                                 "reason": "no valid evidence ID" if not cited else
                                           f"cites unknown evidence {', '.join(unknown)}"})
        return None
    if unknown:
        warnings.append(f"Ignored unknown evidence reference(s): {', '.join(unknown)}.")

    pool = [n for eid in valid for n in evidence_numbers[eid]]
    texts = [item.get("text", "")] + [item.get(k, "") for k in ("validation_step", "metric_to_watch")]
    wrong = [n.text for t in texts for n in extract_numbers(t) if not _matches(n, pool)]
    if wrong:
        warnings.append("Not found in the cited evidence: " + ", ".join(dict.fromkeys(wrong)) +
                        ". Treat these figures as unverified.")
    item.update(evidence_ids=valid, label=label, warnings=warnings,
                status="unverified" if wrong else "verified")
    ev.kept += 1
    if wrong:
        ev.unverified += 1
        ev.warnings.append(f"{section}: {item.get('text', '')[:80]}...")
    else:
        ev.verified += 1
    return item
