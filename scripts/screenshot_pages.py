"""Take headless-browser screenshots of app pages (for visual review; not used by the app).

Starts the app on a spare port with a temporary database, opens it in headless Chrome (or Edge)
through the Chrome DevTools Protocol, loads the sample dataset, opens each requested page and
saves a PNG at laptop width and, optionally, phone width.

    python scripts/screenshot_pages.py --out docs/review/charts --prefix after \
        --pages "Executive Overview" Channels --phone

Needs only Chrome/Edge and the `websockets` package (already installed with Streamlit).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
APP_PORT, DEBUG_PORT = 8599, 9333


class Page:
    def __init__(self, ws):
        self.ws, self.next_id = ws, 0

    async def send(self, method: str, **params):
        self.next_id += 1
        my_id = self.next_id
        await self.ws.send(json.dumps({"id": my_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == my_id:
                return msg.get("result", {})

    async def js(self, expression: str):
        result = await self.send("Runtime.evaluate", expression=expression, returnByValue=True,
                                 awaitPromise=True)
        return result.get("result", {}).get("value")

    async def wait_for(self, expression: str, timeout: float = 60) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(expression):
                return True
            await asyncio.sleep(0.5)
        return False

    async def idle(self, settle: float = 2.5) -> None:
        """Wait until Streamlit has finished running the script, then let charts draw."""
        await asyncio.sleep(1.0)
        await self.wait_for("!document.querySelector('[data-testid=\"stStatusWidget\"]')", 90)
        await asyncio.sleep(settle)

    async def width(self, px: int, height: int) -> None:
        await self.send("Emulation.setDeviceMetricsOverride", width=px, height=height,
                        deviceScaleFactor=1, mobile=px < 700)

    async def expand_sidebar(self) -> None:
        """Streamlit collapses the sidebar at phone width; open it again for laptop shots."""
        if await self.js("(() => { const side = document.querySelector('[data-testid=\"stSidebar\"]');"
                         " if (!side || side.getAttribute('aria-expanded') !== 'false') return false;"
                         " const b = document.querySelector('[data-testid=\"stExpandSidebarButton\"]');"
                         " if (b) { b.click(); return true; } return false; })()"):
            await asyncio.sleep(1.5)

    async def click_text(self, selector: str, text: str) -> bool:
        return await self.js(f"""(() => {{
            const el = [...document.querySelectorAll({json.dumps(selector)})]
                .find(e => e.innerText.trim().endsWith({json.dumps(text)}));
            if (el) {{ el.click(); return true; }} return false; }})()""")

    async def shot(self, path: Path) -> None:
        # Streamlit scrolls inside its own container; measure it so the whole page is captured.
        h = await self.js("Math.max(document.querySelector('[data-testid=\"stMain\"]')?.scrollHeight || 0,"
                          " document.body.scrollHeight)") or 1200
        metrics = await self.send("Page.getLayoutMetrics")
        w = metrics["cssLayoutViewport"]["clientWidth"]
        await self.width(w, int(h) + 40)
        await asyncio.sleep(1.5)
        data = await self.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        path.write_bytes(base64.b64decode(data["data"]))
        print("saved", path)


async def run(args) -> None:
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json").read())
    target = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=2 ** 28) as ws:
        page = Page(ws)
        await page.send("Page.enable")
        await page.width(1440, 900)
        await page.send("Page.navigate", url=f"http://localhost:{APP_PORT}")
        await page.wait_for("!!document.querySelector('[data-testid=\"stSidebar\"]')", 90)
        await page.idle()
        await page.click_text("button", "Load Sample Dataset")
        await page.wait_for("document.body.innerText.includes('Analysis complete') || "
                            "document.body.innerText.includes('Analysis #')", 120)
        await page.idle()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # Laptop pass first, phone pass second: at phone width Streamlit collapses the sidebar,
        # which would otherwise be caught half-open in the next laptop shot.
        for width, height, suffix in [(1440, 900, "laptop")] + ([(390, 844, "phone")] if args.phone else []):
            for name in args.pages:
                await page.width(width, height)
                if suffix == "laptop":
                    await page.expand_sidebar()
                await page.click_text("[data-testid='stSidebar'] label", name)
                await page.idle(4)
                slug = name.lower().replace(" & ", "_").replace(" ", "_")
                await page.shot(out / f"{args.prefix}_{slug}_{suffix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", default="page")
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--targets", help='U5: targets to pre-save for the sample, as JSON, e.g. {"cpl": 240}')
    args = parser.parse_args()
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        sys.exit("Chrome or Edge is needed for screenshots.")
    tmp = Path(tempfile.mkdtemp(prefix="mi_shots_"))
    env = dict(os.environ, DATABASE_PATH=str(tmp / "shots.db"), AI_ENABLED="false")
    if args.targets:                     # saved for the sample's column layout, as the Targets page does
        sys.path.insert(0, str(ROOT))
        from ai.mapping import layout_key
        from dashboard.pipeline import DEFAULT_SAMPLE, load_raw, sample_path
        from database import repository
        raw, _ = load_raw(sample_path(DEFAULT_SAMPLE))
        repository.save_targets(layout_key(raw.columns), json.loads(args.targets), db_path=tmp / "shots.db")
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                            "--server.port", str(APP_PORT)], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chrome = subprocess.Popen([browser, "--headless=new", f"--remote-debugging-port={DEBUG_PORT}",
                               f"--user-data-dir={tmp / 'profile'}", "--hide-scrollbars",
                               "--no-first-run", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{APP_PORT}/_stcore/health", timeout=1)
                urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=1)
                break
            except OSError:
                time.sleep(1)
        asyncio.run(run(args))
    finally:
        chrome.terminate()
        app.terminate()


if __name__ == "__main__":
    main()
