"""The product mark (assets/logo.svg): one file used by the sidebar, the browser tab and the PDF.

Streamlit shows the SVG directly (sidebar <img>, favicon). ReportLab cannot read SVG, so
`draw_logo` draws the same file's shapes on a PDF canvas. It understands only what the logo uses:
<rect> (with rx), <polyline> and <circle>, in a 64 x 64 viewBox.
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from functools import lru_cache

from config.settings import PROJECT_ROOT

LOGO_PATH = PROJECT_ROOT / "assets" / "logo.svg"
_NS = "{http://www.w3.org/2000/svg}"


def logo_data_uri() -> str:
    """The SVG as a data URI for an <img> tag ("" if the file is missing)."""
    try:
        return "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except OSError:
        return ""


@lru_cache(maxsize=1)
def _shapes() -> tuple[float, list[tuple[str, dict]]]:
    root = ET.parse(LOGO_PATH).getroot()
    view = float(root.get("viewBox", "0 0 64 64").split()[2])
    return view, [(el.tag.replace(_NS, ""), dict(el.attrib)) for el in root]


def draw_logo(canvas, x: float, y: float, size: float) -> None:
    """Draw the mark with its bottom-left corner at (x, y), `size` points wide and high."""
    from reportlab.lib import colors
    try:
        view, shapes = _shapes()
    except (OSError, ET.ParseError):
        return
    k = size / view
    px = lambda v: x + float(v) * k                  # noqa: E731 - SVG x -> PDF x
    py = lambda v: y + size - float(v) * k           # noqa: E731 - SVG y (down) -> PDF y (up)
    canvas.saveState()
    for tag, a in shapes:
        if tag == "rect":
            w, h = float(a["width"]) * k, float(a["height"]) * k
            canvas.setFillColor(colors.HexColor(a["fill"]))
            canvas.roundRect(px(a["x"]), py(a["y"]) - h, w, h, float(a.get("rx", 0)) * k, stroke=0, fill=1)
        elif tag == "polyline":
            pts = [tuple(float(v) for v in p.split(",")) for p in a["points"].split()]
            canvas.setStrokeColor(colors.HexColor(a["stroke"]))
            canvas.setLineWidth(float(a.get("stroke-width", 1)) * k)
            canvas.setLineCap(1)                     # round
            canvas.setLineJoin(1)
            path = canvas.beginPath()
            path.moveTo(px(pts[0][0]), py(pts[0][1]))
            for sx, sy in pts[1:]:
                path.lineTo(px(sx), py(sy))
            canvas.drawPath(path, stroke=1, fill=0)
        elif tag == "circle":
            canvas.setFillColor(colors.HexColor(a["fill"]))
            canvas.circle(px(a["cx"]), py(a["cy"]), float(a["r"]) * k, stroke=0, fill=1)
    canvas.restoreState()
