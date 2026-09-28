"""Draw the browser-tab icon (an "MJ" monogram) into assets/brand/mj_icon.png.

Run once with:  python scripts/make_brand_icon.py   (the PNG is committed, so this is only
needed if the colours or letters change). Uses the theme's primary colour and the bundled font.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = "#1c5cab"      # the interface brand blue (dashboard/theme.py BRAND)
SIZE = 128


def main() -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=28, fill=PRIMARY)
    font = ImageFont.truetype(str(ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf"), 60)
    draw.text((SIZE / 2, SIZE / 2 + 2), "MJ", font=font, fill="white", anchor="mm")
    out = ROOT / "assets" / "brand" / "mj_icon.png"
    img.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
