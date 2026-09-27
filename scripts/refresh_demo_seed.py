"""Regenerate the demo files for the Kalpa clean sample:

    data/demo/kalpa_demo_snapshot.pkl.gz   the finished analysis (opens instantly in the demo)
    data/demo/kalpa_demo_seed.json         ONE real Gemini answer for it (+ the field mapping)

Run just before deployment, with AI configured in .env:

    .venv\\Scripts\\python scripts\\refresh_demo_seed.py            # snapshot + ONE Gemini call
    .venv\\Scripts\\python scripts\\refresh_demo_seed.py --snapshot-only   # no AI call

Rules for the AI call: exactly one call, with the MAIN model (GEMINI_MODEL); no fallback model.
If Google answers "high demand" (or anything else goes wrong) the seed JSON is left unchanged
and the script says so - try again later. The analysis runs in a temporary database, so the
app's own database is untouched.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.client import AIError, GeminiClient  # noqa: E402
from ai.demo_seed import SEED_PATH, build_seed, write_seed  # noqa: E402
from ai.prompts import PROMPT_VERSION  # noqa: E402
from ai.service import generate_ai_insights  # noqa: E402
from config.settings import settings  # noqa: E402
from dashboard.demo import write_demo_snapshot  # noqa: E402
from dashboard.pipeline import SAMPLE_DIR, load_raw, prepare, run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--snapshot-only", action="store_true", help="write the snapshot, make no AI call")
    opts = args.parse_args(argv)

    raw, report = load_raw(SAMPLE_DIR / "marketing_clean.csv")
    prep = prepare(raw, report)
    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "seed.db"
        output = run_pipeline(prep, db_path=db)
        output.run_id = None                               # a run id is assigned when restored
        output.analysis.metadata["run_id"] = None
        snapshot = write_demo_snapshot(prep, output)
        print(f"Snapshot written: {snapshot} ({snapshot.stat().st_size / 1e6:.1f} MB)")
        if opts.snapshot_only:
            return 0

        if not (settings.ai_enabled and settings.has_gemini_key and settings.gemini_model):
            print("AI is not configured (AI_ENABLED, GEMINI_API_KEY, GEMINI_MODEL). Seed unchanged.")
            return 1
        try:
            client = GeminiClient(settings.gemini_api_key, settings.gemini_model, fallback_model="")
        except AIError as exc:
            print(f"AI is not ready: {exc.user_message} Seed unchanged.")
            return 1
        run = generate_ai_insights(output.analysis, client, max_calls=1, db_path=db)

    if not run.ok:
        print(f"The Gemini call did not succeed ({settings.gemini_model}): {run.error.user_message} "
              "Seed unchanged.")
        return 2
    if run.model != settings.gemini_model:                  # never accept another model's answer
        print(f"Answer came from {run.model}, not {settings.gemini_model}. Seed unchanged.")
        return 2
    seed = build_seed(file_name=report.filename, file_hash=report.file_hash,
                      mapping={m.column: m.field for m in prep.mapping.columns},
                      model=run.model, prompt_version=PROMPT_VERSION, sections=run.insights)
    path = write_seed(seed, SEED_PATH)
    ev = run.evaluation
    print(f"Seed written: {path}")
    print(f"Model {run.model}, prompt {PROMPT_VERSION}: {ev['kept']} kept, {ev['verified']} verified, "
          f"{ev['unverified']} unverified, {ev.get('weak', 0)} weak, {ev['dropped']} dropped; "
          f"~{run.input_tokens} input / {run.output_tokens} output tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
