"""Regenerate the demo seed (data/demo/kalpa_demo_seed.json) with ONE real Gemini call.

Run this once, just before deployment (Phase 15), with AI configured in .env:

    .venv\\Scripts\\python scripts\\refresh_demo_seed.py

What it does:
  1. analyses the Kalpa clean sample (in a temporary database - your app database is untouched),
  2. makes ONE Gemini call for the AI insights (no automatic second call; if Gemini is busy the
     script stops and you can try again later),
  3. writes the checked answer and the sample's field mapping to the seed file.
The deployed app then shows these insights for the Kalpa sample without any API call.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.client import AIError, client_from_settings  # noqa: E402
from ai.demo_seed import SEED_PATH, build_seed, write_seed  # noqa: E402
from ai.prompts import PROMPT_VERSION  # noqa: E402
from ai.service import generate_ai_insights  # noqa: E402
from config.settings import settings  # noqa: E402
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline  # noqa: E402


def main() -> int:
    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    prep = prepare(raw, report)
    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "seed.db"
        output = run_pipeline(prep, db_path=db)
        try:
            client = client_from_settings(settings)
        except AIError as exc:
            print(f"AI is not ready: {exc.user_message}")
            return 1
        run = generate_ai_insights(output.analysis, client, run_id=output.run_id, max_calls=1, db_path=db)
    if not run.ok:
        print(f"The Gemini call did not succeed: {run.error.user_message}")
        return 1
    seed = build_seed(file_name=report.filename, file_hash=report.file_hash,
                      mapping={m.column: m.field for m in prep.mapping.columns},
                      model=run.model, prompt_version=PROMPT_VERSION, sections=run.insights)
    path = write_seed(seed, SEED_PATH)
    ev = run.evaluation
    print(f"Seed written to {path}")
    print(f"Model {run.model}: {ev['kept']} kept, {ev['verified']} verified, {ev['unverified']} "
          f"unverified, {ev['dropped']} dropped; ~{run.input_tokens} input / {run.output_tokens} "
          "output tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
