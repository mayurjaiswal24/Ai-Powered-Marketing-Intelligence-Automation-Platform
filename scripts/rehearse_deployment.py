"""Rehearse a fresh public deployment on this computer (no GitHub, no AI calls).

    .venv\\Scripts\\python scripts\\rehearse_deployment.py

1. Copies exactly the files a push would contain (git-tracked + new, not ignored: no .env, no
   database) into an empty temporary folder.
2. Adds a hosting-style .streamlit/secrets.toml made from secrets.toml.example, with
   PUBLIC_MODE=true, AI switched off and no key.
3. Starts the real Streamlit server from the copy and checks its health endpoint.
4. Acts as two visitors: sample load (instant demo), dashboard, AI page, PDF + Excel, a real
   upload, an over-limit upload, and session isolation. Prints what happened.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8599


def build_copy(target: Path) -> None:
    listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    for name in filter(None, listing.stdout.splitlines()):
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    lines = []
    for line in (ROOT / ".streamlit" / "secrets.toml.example").read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY"):
            line = 'GEMINI_API_KEY = ""'
        elif line.startswith("AI_ENABLED"):
            line = 'AI_ENABLED = "false"'
        lines.append(line)
    (target / ".streamlit" / "secrets.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"copied {sum(1 for p in target.rglob('*') if p.is_file())} files | .env present: "
          f"{(target / '.env').exists()} | database present: {(target / 'data' / 'app').exists()}")


def check_server(copy: Path) -> None:
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                               "--server.headless", "true", "--server.port", str(PORT)],
                              cwd=copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status = "no answer"
    try:
        for _ in range(60):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2) as r:
                    status = r.read().decode()
                break
            except OSError:
                continue
    finally:
        server.terminate()
        server.wait(timeout=30)
    print(f"real server started from the copy, health check: {status}")


def visitors() -> None:
    """Runs inside the copy (current folder = the copy)."""
    here = Path.cwd()
    sys.path.insert(0, str(here))
    import config.settings as cs
    from streamlit.testing.v1 import AppTest

    s = cs.settings
    db = here / "data" / "app" / "marketing_intelligence.db"
    print(f"settings: PUBLIC_MODE={s.public_mode} AI_ENABLED={s.ai_enabled} key_set={s.has_gemini_key} "
          f"upload_limit={s.upload_limit_mb}MB | database before first visit: {db.exists()}")

    a = AppTest.from_file(str(here / "app.py"), default_timeout=300)
    t0 = time.perf_counter()
    a.run()
    print(f"visitor A: first page {time.perf_counter() - t0:.1f}s, error={bool(a.exception)}, "
          f"demo notice={any('only for this session' in i.value for i in a.info)}")
    t0 = time.perf_counter()
    a.button(key="load_sample").click().run()
    out = a.session_state["output"]
    print(f"  Load sample dataset: {time.perf_counter() - t0:.1f}s, analysis ready={out is not None}, "
          f"instant demo={any('opened instantly' in c.value for c in a.caption)}, database created={db.exists()}")
    a.sidebar.radio(key="page").set_value("Executive Overview").run()
    print(f"  Executive Overview: error={bool(a.exception)}, KPI cards={len(a.metric)}")
    a.sidebar.radio(key="page").set_value("AI Insights").run()
    print(f"  AI page: {[x.value for x in a.success][:1]}")
    a.sidebar.radio(key="page").set_value("Reports").run()
    a.button(key="generate_pdf").click().run()
    a.button(key="generate_excel").click().run()
    print(f"  Reports: error={bool(a.exception)}, download buttons={len(a.get('download_button'))}")

    a.sidebar.radio(key="page").set_value("Upload & Profile").run()
    meta = (here / "data" / "sample" / "meta_ads_export_style.csv").read_bytes()
    a.file_uploader(key="uploader").upload("meta_export.csv", meta, "text/csv").run()
    print(f"  upload meta_export.csv: waits for a choice={a.button(key='run_analysis').disabled}")
    a.selectbox(key="confirm_Results").set_value("leads").run()
    a.button(key="apply_mapping").click().run()
    t0 = time.perf_counter()
    a.button(key="run_analysis").click().run()
    out2 = a.session_state["output"]
    print(f"  uploaded file analysed in {time.perf_counter() - t0:.1f}s: rows={out2.analysis.metadata['rows']}, "
          f"error={bool(a.exception)}")
    big = b"Date,Spend\n" + b"2026-01-01,1000\n" * 700_000      # about 11 MB
    a.file_uploader(key="uploader").upload("too_big.csv", big, "text/csv").run()
    print(f"  11 MB upload: {[e.value for e in a.error][:1]}")
    a.run()
    print(f"  visitor A sees runs: {sorted(b.key for b in a.button if b.key and b.key.startswith('open_run_'))}")

    b = AppTest.from_file(str(here / "app.py"), default_timeout=300)
    b.run()
    print(f"visitor B (another browser) sees runs: "
          f"{sorted(x.key for x in b.button if x.key and x.key.startswith('open_run_'))}, error={bool(b.exception)}")


def main() -> int:
    if "--visitors" in sys.argv:
        visitors()
        return 0
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "app"
        copy.mkdir()
        build_copy(copy)
        check_server(copy)
        result = subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).resolve()), "--visitors"],
                                cwd=copy, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("\n".join(line for line in (result.stdout + result.stderr).splitlines()
                        if line.strip() and "ScriptRunContext" not in line))
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
