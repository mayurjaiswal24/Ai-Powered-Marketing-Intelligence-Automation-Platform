"""Demo seed: saved AI insights for the Kalpa sample, so the live demo never needs a live call.

`data/demo/kalpa_demo_seed.json` holds one real Gemini answer for the Kalpa clean sample (made
with scripts/refresh_demo_seed.py), plus the sample's field mapping for reference. When the
Kalpa sample is analysed and the database has no saved insights for it (e.g. a fresh deployment,
whose storage starts empty), the answer is checked again by the evaluator against the CURRENT
evidence pack and stored in the cache. The AI page then shows it straight away with
"No API call used". Numbers are still only accepted if they match the current evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from ai.schemas import AIInsights
from config.settings import PROJECT_ROOT

SEED_PATH = PROJECT_ROOT / "data" / "demo" / "kalpa_demo_seed.json"
SEED_VERSION = 1


def to_raw_insights(sections: dict) -> dict:
    """Checked sections (with status/warnings added by the evaluator) -> Gemini's answer shape."""
    raw = {}
    for name, info in AIInsights.model_fields.items():
        value = sections.get(name)
        item_fields = _item_fields(info.annotation)
        if isinstance(value, list):
            raw[name] = [{k: v for k, v in item.items() if k in item_fields} for item in value]
        elif isinstance(value, dict):
            raw[name] = {k: v for k, v in value.items() if k in item_fields}
    return AIInsights.model_validate(raw).model_dump()


def _item_fields(annotation) -> set[str]:
    model = getattr(annotation, "__args__", (annotation,))[0]
    return set(getattr(model, "model_fields", {}))


def build_seed(*, file_name: str, file_hash: str, mapping: dict, model: str, prompt_version: str,
               sections: dict, created_at: str | None = None) -> dict:
    return {
        "seed_version": SEED_VERSION,
        "note": "Real Gemini answer for the Kalpa clean sample; regenerate with "
                "scripts/refresh_demo_seed.py (one real call).",
        "dataset": {"file_name": file_name, "file_hash": file_hash},
        "mapping": mapping,
        "ai": {"model": model, "prompt_version": prompt_version,
               "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "insights": to_raw_insights(sections)},
    }


def write_seed(seed: dict, path: Path = SEED_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def load_seed(path: Path | None = None) -> dict | None:
    try:
        seed = json.loads(Path(path or SEED_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return seed if seed.get("seed_version") == SEED_VERSION else None


def preload_demo_insights(output, report, settings, db_path=None, path: Path | None = None) -> bool:
    """Store the seeded insights for this run if it is the Kalpa sample and nothing is cached
    for its evidence yet. Returns True when insights were preloaded."""
    from ai.cache import load_cached
    from ai.context_builder import build_evidence
    from ai.evaluator import evaluate
    from database import repository
    from database.connection import DatabaseError

    seed = load_seed(path if path is not None else SEED_PATH)
    if seed is None or output is None or report is None:
        return False
    if report.file_hash != seed["dataset"]["file_hash"]:
        return False
    pack = build_evidence(output.analysis, model=settings.gemini_model or "")
    if load_cached(pack.fingerprint, db_path=db_path) is not None:
        return False
    try:
        answer = AIInsights.model_validate(seed["ai"]["insights"])
    except ValidationError:
        return False
    checked, evaluation = evaluate(answer.model_dump(), pack)   # re-checked against today's evidence
    try:
        repository.save_ai_insights(
            output.run_id, pack.fingerprint,
            {"model": seed["ai"]["model"], "prompt_version": seed["ai"]["prompt_version"],
             "sections": checked, "seeded": True}, evaluation.as_dict(), db_path=db_path)
    except DatabaseError:
        return False
    return True
