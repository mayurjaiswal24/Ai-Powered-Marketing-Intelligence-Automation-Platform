"""Hallucination control: check Gemini's answer against the evidence before showing it.

1. Every item must cite evidence IDs that exist in the pack. Unknown IDs are removed; an item
   left with no valid ID is DROPPED (it cannot be traced to verified numbers).
2. Every number the AI writes (₹4.2 Cr, 38%, 3.45x, 1,23,456 ...) must match a number in the
   evidence it cites, allowing for rounding. Otherwise the item is kept but marked UNVERIFIED
   with a warning, so a reader knows not to trust that figure.
3. Quality checks (v3). Failing items are kept but marked WEAK with a warning:
   - a key finding that cites only one deterministic finding just restates it;
   - a recommendation that cites an incident must quote that incident's estimated rupee impact.
4. "₹ at stake" per recommendation, calculated in code (never by the AI):
   - a budget move with a test share: test_shift_pct x the spend of the campaign/channel the
     budget moves FROM (money put at risk by the test - NOT a projected gain);
   - a recommendation citing an incident: that incident's estimated rupee impact.
   - (v6) a recommendation citing the profitability evidence and naming a loss-making campaign:
     that campaign's ₹ lost.
   The larger of these is used, and ALL recommendations are ranked by it (highest first);
   those with nothing at stake keep their order after them.
5. A summary counts kept, verified, unverified, weak and dropped items.
6. Forecasts (v6): a money range with one shared unit ("₹2.7-5.9 Cr") is read as two amounts, so
   both ends are checked. An item that quotes figures from the forecast evidence must quote at
   least two of them (a range) and the past error; otherwise it is marked WEAK.
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
# "₹2.7-5.9 Cr" / "₹2.7 to ₹5.9 Cr": the unit written once applies to both ends of the range.
MONEY_RANGE = re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)\s?(–|-|to)\s?(₹\s?)?(\d[\d,]*(?:\.\d+)?)\s?(Cr|crore|L|lakhs|lakh)\b")
from config.settings import AI_NUMBER_TOLERANCE as RELATIVE_TOLERANCE  # noqa: E402


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
    text = MONEY_RANGE.sub(lambda m: f"₹{m.group(1)} {m.group(5)} to ₹{m.group(4)} {m.group(5)}", text or "")
    for m in NUMBER.finditer(text):
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
    weak: int = 0
    dropped: int = 0
    dropped_items: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"kept": self.kept, "verified": self.verified, "unverified": self.unverified,
                "weak": self.weak, "dropped": self.dropped, "dropped_items": self.dropped_items,
                "warnings": self.warnings}


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
                _quality_checks(key, result, pack, ev)
                kept_items.append(result)
        if key == "recommendations":
            for item in kept_items:
                _add_at_stake(item, pack)
            kept_items = rank_by_at_stake(kept_items)
        checked[key] = (kept_items[0] if kept_items else None) if key == "executive_summary" else kept_items
    return checked, ev


def _incident_impact(item, pack) -> tuple[list[str], list[Number]]:
    """IDs of cited incident evidence and the rupee impact numbers stated in them."""
    ids, amounts = [], []
    for eid in item.get("evidence_ids", []):
        ev_item = pack.by_id.get(eid)
        if ev_item is not None and ev_item.type == "incident":
            ids.append(eid)
            amounts += [n for n in extract_numbers(ev_item.facts.get("estimated impact", ""), checkable_only=False)
                        if n.kind == "money"]
    return ids, amounts


def _quality_checks(section: str, item: dict, pack, ev: Evaluation) -> None:
    weak = []
    if section == "key_findings":
        ids = item.get("evidence_ids", [])
        if len(ids) == 1 and pack.by_id[ids[0]].type == "finding":
            weak.append("Restates a single deterministic finding without adding insight.")
    if section == "recommendations":
        ids, amounts = _incident_impact(item, pack)
        if ids and amounts:
            stated = [n for n in extract_numbers(item.get("text", "")) if n.kind == "money"]
            if not any(_matches(n, amounts) for n in stated):
                shown = ", ".join(f"{i}: {pack.by_id[i].facts.get('estimated impact', '').split(' (')[0]}" for i in ids)
                weak.append(f"Does not mention the rupee impact of the incident it relies on ({shown}).")
    weak += _forecast_checks(item, pack)
    item["weak"] = bool(weak)
    item["weak_reasons"] = weak
    if weak:
        ev.weak += 1
        item["warnings"] = item.get("warnings", []) + weak


def _forecast_checks(item: dict, pack) -> list[str]:
    """v6: a forecast is always a range with its past accuracy. Applies only when the item quotes a
    figure from a cited forecast evidence item."""
    texts = " ".join(item.get(k, "") or "" for k in ("text", "validation_step", "metric_to_watch"))
    stated = extract_numbers(texts)
    for eid in item.get("evidence_ids", []):
        fc = pack.by_id.get(eid)
        if fc is None or fc.type != "forecast":
            continue
        lines = [v for k, v in fc.facts.items() if k not in ("range meaning", "limits")]
        amounts = [n for v in lines for n in extract_numbers(v.split(";")[0], checkable_only=False)]
        errors = [n for v in lines for n in extract_numbers(v, checkable_only=False) if n.kind == "percent"]
        quoted = [n for n in stated if n.kind != "percent" and _matches(n, amounts)]
        if not quoted:
            continue
        reasons = []
        if len(quoted) < 2:
            reasons.append("States a forecast as a single number instead of a range.")
        if not any(n.kind == "percent" and _matches(n, errors) for n in stated):
            reasons.append("States a forecast without its past accuracy.")
        return reasons
    return []


def budget_source(item: dict, pack):
    """The campaign/channel evidence item a budget move takes money FROM: the one the AI named in
    `source_evidence_id`, else (older answers) the cited campaign/channel with the lowest ROAS."""
    named = pack.by_id.get(str(item.get("source_evidence_id") or "").strip().upper())
    if named is not None and "spend" in named.values:
        return named
    cited = [pack.by_id[e] for e in item.get("evidence_ids", [])
             if e in pack.by_id and "spend" in pack.by_id[e].values]
    with_roas = [c for c in cited if "roas" in c.values]
    if with_roas:
        return min(with_roas, key=lambda c: c.values["roas"])
    return cited[0] if len(cited) == 1 else None


def _add_at_stake(item: dict, pack) -> None:
    """Set `rupees_at_stake` (a number) and `at_stake_text` (how it was calculated)."""
    from utils.formatting import format_inr
    candidates = []
    pct = item.get("test_shift_pct")
    source = budget_source(item, pack) if pct else None
    if source is not None:
        spend = source.values["spend"]
        amount = pct / 100 * spend
        candidates.append((amount, f"{pct:g}% test shift × {format_inr(spend, compact=True)} spend of "
                                   f"{source.title} ({source.id}); money at risk, not a projected gain"))
    for eid in item.get("evidence_ids", []):
        prof = pack.by_id.get(eid)
        if prof is None or prof.type != "profitability":
            continue
        for key, lost in prof.values.items():
            name = key.split(":", 1)[1]
            if key.startswith("loss:") and name.lower() in (item.get("text", "") or "").lower():
                candidates.append((lost, f"₹ lost by the loss-making campaign {name}"))
    _, impacts = _incident_impact(item, pack)
    if impacts:
        biggest = max(n.value for n in impacts)
        candidates.append((biggest, "estimated rupee impact of the incident it addresses"))
    if candidates:
        amount, basis = max(candidates, key=lambda c: c[0])
        item["rupees_at_stake"] = amount
        item["at_stake_text"] = f"₹ at stake: {format_inr(amount, compact=True)} ({basis})"
    else:
        item["rupees_at_stake"] = 0.0
        item["at_stake_text"] = ""


def rank_by_at_stake(items: list[dict]) -> list[dict]:
    """Highest ₹ at stake first (enforced in code, whatever order the AI used)."""
    ranked = sorted(enumerate(items), key=lambda p: (-(p[1].get("rupees_at_stake") or 0.0), p[0]))
    return [item for _, item in ranked]


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
