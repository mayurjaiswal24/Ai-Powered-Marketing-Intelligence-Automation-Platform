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
7. v7, also marked WEAK: causal-certainty wording (prompts.BANNED_CAUSAL_PHRASES); calling an
   incident of AI_SHORT_INCIDENT_DAYS or fewer the primary cause of a trend without saying it
   is partial; and, when the pacing evidence is not "On Pace", recommendations that never refer
   to it (the flag is put on the top-ranked recommendation, where the reader sees it).
8. v8, also WEAK: when the pacing evidence says "Underspending", a recommendation linked to pacing
   that only reallocates between channels must either deploy the unused budget into the channel(s)
   with the lowest cost of the next conversion, or say that the reallocation addresses efficiency,
   not the underspend. Also WEAK: a recommendation that puts added spend INTO a channel paid per
   result (e.g. Affiliate) beyond the room stated in the evidence ("at most about ₹X more a week").
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
from config.settings import AI_SHORT_INCIDENT_DAYS  # noqa: E402
from ai.prompts import BANNED_CAUSAL_PHRASES  # noqa: E402

BANNED = re.compile(r"\b(" + "|".join(re.escape(p) for p in BANNED_CAUSAL_PHRASES) + r")\b", re.IGNORECASE)
PRIMARY_CAUSE = re.compile(r"\b(primar(y|ily)|main(ly)?|mostly|largely|chiefly|driven by|caused by)\b", re.IGNORECASE)
PARTIAL = re.compile(r"\b(partial(ly)?|part of|contribut\w*|unexplained)\b", re.IGNORECASE)
PACING_WORDS = re.compile(r"\b(pacing|underspen\w*|overspen\w*|unused budget|of plan)\b", re.IGNORECASE)
# v8: moving money between channels vs. spending the unused budget.
REALLOCATE = re.compile(r"\b(reallocat\w*|shift\w*|mov(e|es|ed|ing)|transfer\w*|redistribut\w*)\b", re.IGNORECASE)
UNUSED_BUDGET = re.compile(r"\b(unused|unspent|underspent|remaining|leftover|left-over)\b[^.;]{0,40}(budget|spend|₹)"
                           r"|\b(increas\w*|rais\w*|lift\w*)\b[^.;]{0,15}\btotal spend", re.IGNORECASE)
NEXT_CONVERSION = re.compile(r"\bnext conversion\b|\bmarginal cost\b", re.IGNORECASE)
EFFICIENCY_NOT_UNDERSPEND = re.compile(
    r"\befficien\w*\b[^.]{0,60}\b(not|rather than|instead of)\b[^.]{0,40}\bunderspen\w*"
    r"|\b(does|do|will|would) not (address|fix|solve|close|resolve|correct)\b[^.]{0,30}\bunderspen\w*", re.IGNORECASE)


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
            _pacing_check(kept_items, pack, ev)
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
    weak += _wording_checks(item, pack)
    if section == "recommendations":
        weak += _underspend_check(item, pack)
        weak += _capacity_check(item, pack)
    item["weak"] = bool(weak)
    item["weak_reasons"] = weak
    if weak:
        ev.weak += 1
        item["warnings"] = item.get("warnings", []) + weak


def _incident_days(ev_item) -> int | None:
    """Length in days of an incident evidence item ("period": "8 Sep 2026 to 9 Sep 2026")."""
    import pandas as pd
    try:
        start, end = (pd.Timestamp(x.strip()) for x in ev_item.facts.get("period", "").split(" to "))
    except (ValueError, TypeError):
        return None
    return int((end - start).days) + 1


def _wording_checks(item: dict, pack) -> list[str]:
    """v7: banned causal-certainty wording; a short incident called the primary cause of a trend."""
    text = item.get("text", "") or ""
    reasons = []
    used = BANNED.findall(text)
    if used:
        reasons.append(f'Uses causal-certainty wording ("{used[0].lower()}"); use "consistent with", '
                       '"may indicate" or "one contributing factor".')
    if PRIMARY_CAUSE.search(text) and not PARTIAL.search(text):
        for eid in item.get("evidence_ids", []):
            ev_item = pack.by_id.get(eid)
            days = _incident_days(ev_item) if ev_item is not None and ev_item.type == "incident" else None
            if days is not None and days <= AI_SHORT_INCIDENT_DAYS:
                reasons.append(f"Treats a {days}-day incident as the main cause of a longer trend; describe it "
                               "as a partial or contributing factor.")
                break
    return reasons


def _underspend_check(item: dict, pack) -> list[str]:
    """v8: a reallocation linked to an underspent month keeps total spend the same, so it must either
    deploy the unused budget where the next conversion costs least (a) or say it only addresses
    efficiency, not the underspend (b)."""
    pacing = next((i for i in pack.by_id.values() if i.type == "pacing"), None)
    if pacing is None or pacing.facts.get("status") != "Underspending":
        return []
    text = item.get("text", "") or ""
    ids = item.get("evidence_ids", [])
    if not (pacing.id in ids or PACING_WORDS.search(text)) or not REALLOCATE.search(text):
        return []
    cites_next_cost = NEXT_CONVERSION.search(text) or any(
        pack.by_id[i].type == "marginal_cost" for i in ids if i in pack.by_id)
    if (UNUSED_BUDGET.search(text) and cites_next_cost) or EFFICIENCY_NOT_UNDERSPEND.search(text):
        return []
    return ["Reallocating between channels keeps total spend the same, so it does not fix the underspend: "
            "deploy the unused budget into the channel(s) with the lowest cost of the next conversion, or say "
            "that the move addresses efficiency, not the underspend."]


CAPACITY = re.compile(r"at most about (₹\s?[\d,]+(?:\.\d+)?(?:\s?(?:Cr|L))?) more a week", re.IGNORECASE)
WEEKLY_NOW = re.compile(r"now (₹\s?[\d,]+(?:\.\d+)?(?:\s?(?:Cr|L))?) a week", re.IGNORECASE)
UNUSED_PHRASE = re.compile(r"\b(unused|unspent|underspent|remaining|leftover)\b[^.;]{0,20}\bbudget\b", re.IGNORECASE)
PROPOSE_VERB = (r"\b(deploy\w*|mov\w*|shift\w*|add\w*|put\w*|allocat\w*|reallocat\w*|invest\w*|spend\w*|redirect\w*|"
                r"channel\w*|increas\w*|rais\w*)\b(?:\s+(?:up to|about|around|roughly|the|an|a|extra|additional|"
                r"unused|remaining|further|another))*\s+$")
PER_WEEK = re.compile(r"^\s*(?:more\s+)?(?:a|per|each|every)\s+week\b|^\s*weekly\b|^\s*/\s*week\b", re.IGNORECASE)


def _money(text: str) -> float | None:
    nums = [n for n in extract_numbers(text, checkable_only=False) if n.kind == "money"]
    return nums[0].value if nums else None


def _capacity_check(item: dict, pack) -> list[str]:
    """Added spend INTO a channel paid per result must stay within its stated room ("at most about ₹X
    more a week" in the marginal_cost evidence). The proposed amount is, in this order: a ₹ figure
    written right after a proposing verb ("deploy ₹2 L", "put up to ₹60,000 a week"); else the month's
    unused budget when the text proposes "the unused budget" (plan to date - spend to date, from the
    pacing evidence); else the test share of the source channel's current weekly spend. A weekly
    amount is compared with the weekly room; any other amount is treated as one month's spending and
    compared with the room over the latest month's days."""
    from utils.formatting import format_inr
    cost = next((i for i in pack.by_id.values() if i.type == "marginal_cost"), None)
    if cost is None:
        return []
    text = item.get("text", "") or ""
    reasons = []
    for channel, fact in cost.facts.items():
        room = CAPACITY.search(fact or "")
        if not room or not re.search(rf"\b(into|to|towards?|in|on|scal\w*|increas\w*|grow\w*|boost\w*|expand\w*)\s+"
                                     rf"(the\s+)?{re.escape(channel)}\b", text, re.IGNORECASE):
            continue
        weekly_room = _money(room.group(1))
        pacing = next((i for i in pack.by_id.values() if i.type == "pacing"), None)
        days = re.search(r"day \d+ of (\d+)", pacing.facts.get("latest month", "")) if pacing else None
        month_room = weekly_room * (int(days.group(1)) if days else 30) / 7
        proposals = []                                     # (amount, weekly?)
        for m in NUMBER.finditer(text):
            if m.group("rupee") and re.search(PROPOSE_VERB, text[:m.start()], re.IGNORECASE):
                amount = _money(m.group(0))
                if amount is not None:
                    proposals.append((amount, bool(PER_WEEK.search(text[m.end():]))))
        if not proposals and pacing is not None and UNUSED_PHRASE.search(text):
            spent_vs_plan = [n.value for n in extract_numbers(pacing.facts.get("spend to date vs plan to date", ""),
                                                             checkable_only=False) if n.kind == "money"]
            if len(spent_vs_plan) >= 2 and spent_vs_plan[1] > spent_vs_plan[0]:
                proposals.append((spent_vs_plan[1] - spent_vs_plan[0], False))
        if not proposals and item.get("test_shift_pct"):
            source = budget_source(item, pack)
            name = source.title.split(": ", 1)[-1] if source is not None else ""
            now = WEEKLY_NOW.search(cost.facts.get(name, "") or "")
            if now and name != channel:
                proposals.append((item["test_shift_pct"] / 100 * _money(now.group(1)), True))
        for amount, weekly in proposals:
            limit = weekly_room if weekly else month_room
            if amount > limit * (1 + RELATIVE_TOLERANCE):
                period = "a week" if weekly else "over the month"
                reasons.append(f"Proposes about {format_inr(amount, compact=True)} {period} into {channel}, but its "
                               f"stated capacity is only about {room.group(1)} more a week"
                               + ("." if weekly else f" (about {format_inr(month_room, compact=True)} over the month)."))
                break
    return reasons


def _pacing_check(recommendations: list[dict], pack, ev) -> None:
    """v7: when pacing is off plan, at least one recommendation must refer to it."""
    pacing = [i for i in pack.by_id.values() if i.type == "pacing" and i.facts.get("status") != "On Pace"]
    if not pacing or not recommendations:
        return
    if any(pacing[0].id in r.get("evidence_ids", []) or PACING_WORDS.search(r.get("text", "") or "")
           for r in recommendations):
        return
    top = recommendations[0]
    reason = (f"No recommendation refers to budget pacing ({pacing[0].facts.get('status')} in the latest "
              f"month, {pacing[0].id}).")
    if not top.get("weak"):
        ev.weak += 1
    top["weak"] = True
    top["weak_reasons"] = top.get("weak_reasons", []) + [reason]
    top["warnings"] = top.get("warnings", []) + [reason]


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
