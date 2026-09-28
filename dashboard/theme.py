"""The single visual theme: colours, fonts and the Plotly template used by every chart.

Colours come from a validated, colour-blind-safe reference palette:
  - one primary (blue) and one accent (orange) for most charts,
  - neutral greys for text, gridlines and axes,
  - green / red ONLY for good / bad changes (never for ordinary series),
  - categorical colours assigned to an entity in a FIXED order, so "Paid Search" keeps its
    colour when filters change.
The PDF (matplotlib) and Excel outputs will import the same constants.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# One font across the app and its charts: Inter (loaded in .streamlit/config.toml), with system
# fallbacks if it cannot be downloaded.
FONT_FAMILY = 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'

# Surfaces and ink
PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
PANEL = "#f0efec"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Series colours (fixed order; never cycled past 8 - extra items fold into "Other")
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PRIMARY = CATEGORICAL[0]
ACCENT = CATEGORICAL[1]
# Ordered steps of one hue, for ordered categories such as funnel stages (light -> dark).
ORDINAL_BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]

# Status colours: meaning only (good / bad), always shown with a label or sign.
GOOD = "#0ca30c"
GOOD_TEXT = "#006300"
BAD = "#d03b3b"
WARNING = "#fab219"
NEUTRAL = "#898781"

# Interface accents (active menu item, logo mark, step numbers): a deep blue of the primary hue.
BRAND = "#1c5cab"
BRAND_SOFT = "#e8f0fb"
BORDER = "#e4e3dd"
BORDER_HOVER = "#cfcdc4"
CARD_SHADOW = "0 1px 2px rgba(16, 24, 40, 0.05)"
CARD_SHADOW_HOVER = "0 4px 12px rgba(16, 24, 40, 0.08)"
TRANSITION = "0.15s ease"

TEMPLATE_NAME = "marketing"
# Row highlight in the mapping table for columns mapped by Gemini, by check result.
AI_ROW_BACKGROUND = {"verified": "#e3f4e8", "check": "#fff3d1", "failed": "#fde2e1", "": "#e8f0fb"}
TOP_MARGIN = 56                 # room for the title plus a small note above the plot
TOP_MARGIN_WITH_LEGEND = 84     # title row + legend row
VALUE_LABEL_MARGIN = 64         # right margin so values printed after the longest bar are not cut off
LABEL_WRAP = 24                 # category labels longer than this wrap onto a second line
TITLE_WRAP = 38                  # insight titles wrap so they fit a phone screen (390 px)


def build_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=13, color=INK_SECONDARY),
        # Title pinned to the top edge of the whole figure (not the plot), so legends and
        # "Average = 100"-style notes above the plot can never slide under it.
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", xref="container",
                   y=1, yanchor="top", yref="container", pad=dict(t=10, l=8)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=CATEGORICAL,
        margin=dict(l=8, r=16, t=TOP_MARGIN, b=8),
        xaxis=dict(showgrid=False, linecolor=AXIS, linewidth=1, ticks="", zeroline=False,
                   tickfont=dict(color=INK_MUTED), title=dict(font=dict(color=INK_MUTED))),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS,
                   ticks="", tickfont=dict(color=INK_MUTED), title=dict(font=dict(color=INK_MUTED))),
        # Legend in its own row between the title and the plot (charts with a legend get
        # TOP_MARGIN_WITH_LEGEND so the two never touch).
        legend=dict(orientation="h", yanchor="bottom", y=1.0, yref="paper", xanchor="left", x=0,
                    font=dict(color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT_FAMILY, color=INK)),
        hovermode="closest",
        bargap=0.45,
        barcornerradius=4,
        separators=".,",
    )
    return t


pio.templates[TEMPLATE_NAME] = build_template()


# ---------------------------------------------------------------------------------------------
# Chart standard (docs/CHART_STYLE_GUIDE.md): shared by the app (Plotly) and the PDF (matplotlib)
# ---------------------------------------------------------------------------------------------

FOCUS = PRIMARY                 # the one item a chart's title talks about
CONTEXT = "#b5b4ac"             # everything else: grey, so the focus stands out
SUBTITLE_SIZE = 12
MAX_CATEGORIES = 8              # more than this -> top 7 + "Other"
OTHER_LABEL = "Other"

# Fixed colour per channel, the same on every page, in the PDF and whatever is filtered.
# Keys are lower-case; common platform names share their channel's colour. The dark green and
# red of the palette are kept for channels not listed here (they could be read as good/bad).
CHANNEL_COLORS = {
    "paid search": CATEGORICAL[0], "search": CATEGORICAL[0], "google ads": CATEGORICAL[0],
    "microsoft ads": CATEGORICAL[0], "bing ads": CATEGORICAL[0],
    "paid social": CATEGORICAL[1], "social": CATEGORICAL[1], "meta": CATEGORICAL[1],
    "facebook": CATEGORICAL[1], "instagram": CATEGORICAL[1], "meta ads": CATEGORICAL[1],
    "professional network": CATEGORICAL[2], "linkedin": CATEGORICAL[2], "linkedin ads": CATEGORICAL[2],
    "affiliate": CATEGORICAL[3], "affiliate network": CATEGORICAL[3], "affiliates": CATEGORICAL[3],
    "email": CATEGORICAL[4], "email (in-house)": CATEGORICAL[4], "e-mail": CATEGORICAL[4],
    "video": CATEGORICAL[6], "youtube": CATEGORICAL[6], "youtube ads": CATEGORICAL[6],
}
_FREE_ORDER = [CATEGORICAL[i] for i in (0, 1, 2, 3, 4, 6, 5, 7)]


def entity_colors(entities) -> dict[str, str]:
    """Colour per channel/platform: the fixed CHANNEL_COLORS first, then the unused palette
    colours in a fixed order for any other names (sorted, so the result never depends on the
    order or filtering of the data). "Other" is always the context grey."""
    names = sorted({str(e) for e in entities})
    out = {n: CHANNEL_COLORS[n.lower()] for n in names if n.lower() in CHANNEL_COLORS}
    free = [c for c in _FREE_ORDER if c not in out.values()]
    for name in (n for n in names if n not in out and n != OTHER_LABEL):
        out[name] = free.pop(0) if free else NEUTRAL
    if OTHER_LABEL in names:
        out[OTHER_LABEL] = CONTEXT
    return out


def color_map(entities) -> dict[str, str]:
    """Fixed colour per entity, based on the sorted list of ALL entities in the dataset (not the
    filtered view), so a channel keeps its colour whatever is filtered."""
    ordered = sorted({str(e) for e in entities})
    return {e: CATEGORICAL[i] if i < len(CATEGORICAL) else NEUTRAL for i, e in enumerate(ordered)}


# ---------------------------------------------------------------------------------------------
# Interface CSS: the ONE place where the app's look is defined (pages only use these classes).
# ---------------------------------------------------------------------------------------------

def nav_css(section_starts: dict[int, str]) -> str:
    """Section labels ("DATA", "ANALYSIS", ...) above the first menu item of each group.
    `section_starts` maps a 1-based menu position to its label (empty label = divider only)."""
    rules = []
    for position, label in section_starts.items():
        item = f".st-key-mi_nav [role='radiogroup'] > div:nth-child({position})"
        rules.append(f"{item} {{ margin-top: {'1.7rem' if label else '0.9rem'}; }}")
        if label:
            rules.append(f"{item}::before {{ content: '{label.upper()}'; }}")
        else:
            rules.append(f"{item}::before {{ content: ''; top: -0.45rem; right: 0.6rem; "
                         f"border-top: 1px solid {BORDER}; }}")
    return "<style>" + "\n".join(rules) + "</style>"


APP_CSS = f"""
<style>
  /* --- Streamlit chrome: no developer menu, Deploy button, footer or coloured top bar --- */
  #MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
  [data-testid="stMainMenu"] {{ display: none !important; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  /* --- Page frame and typography --- */
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.015em; }}
  h1 {{ font-size: 1.6rem; font-weight: 650; }}
  h2, [data-testid="stMarkdownContainer"] h2 {{ font-size: 1.2rem !important; font-weight: 640;
      margin-top: 1.1rem; padding: 0.6rem 0 0.3rem 0; }}
  h3, [data-testid="stMarkdownContainer"] h3 {{ font-size: 1.03rem !important; font-weight: 620;
      padding: 0.4rem 0 0.2rem 0; }}

  /* Page header: title + one-line description + a hairline, identical on every page */
  .mi-page-header {{ border-bottom: 1px solid {BORDER}; padding-bottom: 0.7rem; margin-bottom: 1.2rem; }}
  .mi-page-header .mi-eyebrow {{ color: {BRAND}; font-size: 0.72rem; font-weight: 650;
                                  letter-spacing: 0.08em; text-transform: uppercase; }}
  .mi-page-title {{ font-size: 1.6rem; font-weight: 680; color: {INK}; letter-spacing: -0.02em;
                    line-height: 1.25; margin: 0.1rem 0 0.2rem 0; }}
  .mi-scope {{ display: inline-block; margin-top: 0.5rem; padding: 2px 10px; border-radius: 12px;
               background: {PANEL}; color: {INK_SECONDARY}; font-size: 0.78rem; }}
  .mi-chips .mi-scope {{ margin-right: 6px; }}
  .mi-page-header p {{ color: {INK_SECONDARY}; font-size: 0.93rem; margin: 0; }}
  .mi-caption {{ color: {INK_SECONDARY}; font-size: 0.86rem; margin: -0.3rem 0 0.8rem 0; }}

  /* --- KPI cards and charts: subtle border + soft shadow; a slightly stronger shadow on hover --- */
  [data-testid="stMetric"] {{ background: {SURFACE}; box-shadow: {CARD_SHADOW}; border-radius: 10px;
                              height: 100%; transition: box-shadow {TRANSITION}, border-color {TRANSITION}; }}
  [data-testid="stMetric"]:hover {{ box-shadow: {CARD_SHADOW_HOVER}; border-color: {BORDER_HOVER}; }}
  [data-testid="stMetricLabel"] [data-testid="stTooltipIcon"] {{ color: {INK_MUTED}; }}

  /* Bordered cards (upload, demo, reports, about, recent analyses) */
  [data-testid="stVerticalBlock"][class*="stVerticalBlockBorder"],
  div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px;
      transition: box-shadow {TRANSITION}, border-color {TRANSITION}; }}
  div[data-testid="stVerticalBlockBorderWrapper"]:hover {{ box-shadow: {CARD_SHADOW_HOVER}; }}

  /* --- Buttons: smooth colour/shadow change on hover, no movement --- */
  .stButton button, .stDownloadButton button, .stFormSubmitButton button, .stLinkButton a {{
      border-radius: 8px; transition: background-color {TRANSITION}, border-color {TRANSITION},
      box-shadow {TRANSITION}, color {TRANSITION}; }}
  .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover,
  .stLinkButton a:hover {{ box-shadow: {CARD_SHADOW_HOVER}; }}

  /* --- Keyboard focus: always visible (accessibility) --- */
  button:focus-visible, a:focus-visible, [role="tab"]:focus-visible, summary:focus-visible,
  input:focus-visible, textarea:focus-visible {{ outline: 2px solid {BRAND} !important;
      outline-offset: 2px; }}
  .st-key-mi_nav label:has(input:focus-visible) {{ outline: 2px solid {BRAND}; outline-offset: -2px;
      border-radius: 8px; }}
  [data-testid="stMetricLabel"] p {{ color: {INK_SECONDARY}; font-size: 0.82rem; font-weight: 500; }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 650; color: {INK}; }}
  [data-testid="stMetricDelta"] {{ font-weight: 600; }}

  /* --- Small building blocks --- */
  .mi-tag {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.75rem;
             background: {PANEL}; color: {INK_SECONDARY}; margin-right: 6px; }}
  .mi-tag-weak {{ background: #fdf1d6; color: #8a5d00; }}
  .mi-finding {{ border: 1px solid {BORDER}; border-left: 3px solid {PRIMARY}; border-radius: 8px;
                 padding: 0.55rem 0.9rem; margin: 0.45rem 0; background: {SURFACE};
                 transition: box-shadow {TRANSITION}; }}
  .mi-finding:hover {{ box-shadow: {CARD_SHADOW_HOVER}; }}
  .mi-finding .mi-title {{ font-weight: 600; color: {INK}; }}
  /* Empty state: icon + one sentence (+ an action button placed right below) */
  .mi-empty {{ display: flex; align-items: center; gap: 0.75rem; border: 1px dashed {AXIS};
               border-radius: 10px; padding: 1rem 1.2rem; color: {INK_SECONDARY}; background: {SURFACE};
               margin-bottom: 0.6rem; }}
  .mi-empty-icon {{ font-family: "Material Symbols Rounded"; font-size: 1.6rem; line-height: 1;
                    color: {BRAND}; background: {BRAND_SOFT}; border-radius: 50%; padding: 0.45rem;
                    flex: none; font-weight: normal; font-style: normal; letter-spacing: normal;
                    text-transform: none; white-space: nowrap; direction: ltr;
                    -webkit-font-feature-settings: "liga"; font-feature-settings: "liga"; }}

  /* Upload page: 3-step guide, demo card, recent-analysis cards */
  .mi-steps {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem;
               margin: 0.2rem 0 1.3rem 0; }}
  .mi-step {{ border: 1px solid {BORDER}; border-radius: 10px; padding: 0.8rem 0.95rem;
              background: {SURFACE}; box-shadow: {CARD_SHADOW};
              transition: box-shadow {TRANSITION}, border-color {TRANSITION}; }}
  .mi-step:hover {{ box-shadow: {CARD_SHADOW_HOVER}; border-color: {BORDER_HOVER}; }}
  .mi-step-no {{ display: inline-flex; align-items: center; justify-content: center;
                 width: 1.55rem; height: 1.55rem; border-radius: 50%; background: {BRAND_SOFT};
                 color: {BRAND}; font-weight: 700; font-size: 0.8rem; margin-right: 0.45rem; }}
  .mi-step-title {{ font-weight: 620; color: {INK}; }}
  .mi-step p {{ color: {INK_SECONDARY}; font-size: 0.85rem; margin: 0.35rem 0 0 0; }}
  .mi-card-title {{ font-weight: 640; color: {INK}; font-size: 1rem; margin-bottom: 0.1rem; }}
  .mi-card-sub {{ color: {INK_SECONDARY}; font-size: 0.85rem; margin-bottom: 0.2rem; }}
  .mi-run-name {{ font-weight: 620; color: {INK}; overflow: hidden; text-overflow: ellipsis;
                  white-space: nowrap; }}
  .mi-run-meta {{ color: {INK_MUTED}; font-size: 0.8rem; }}

  /* About page */
  .mi-about-list {{ margin: 0.3rem 0 0 0; padding-left: 1.1rem; color: {INK_SECONDARY}; }}
  .mi-about-list li {{ margin: 0.55rem 0; }}
  .mi-about-list b {{ display: block; color: {INK}; font-weight: 600; }}
  .mi-about-list span {{ display: block; font-size: 0.92rem; }}
  .mi-creator-name {{ font-size: 1.25rem; font-weight: 680; color: {INK}; }}
  .mi-creator-name a {{ color: {BRAND}; text-decoration: none; }}
  .mi-creator-row {{ margin: 0.35rem 0; color: {INK_SECONDARY}; font-size: 0.92rem; }}
  .mi-creator-row b {{ color: {INK}; font-weight: 600; }}
  .mi-open-to {{ display: inline-block; margin: 0.5rem 0 0.4rem 0; padding: 0.3rem 0.7rem;
                 border-radius: 8px; background: {BRAND_SOFT}; color: {BRAND}; font-size: 0.86rem;
                 font-weight: 560; }}

  /* --- Sidebar: brand header, grouped menu, signature --- */
  [data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
  .mi-brand {{ display: flex; align-items: center; gap: 0.6rem; margin: -0.4rem 0 0.2rem 0; }}
  .mi-logo {{ width: 34px; height: 34px; border-radius: 9px; background: {BRAND}; flex: none;
              display: flex; align-items: flex-end; justify-content: center; gap: 3px; padding: 8px 7px; }}
  .mi-logo span {{ width: 5px; background: #ffffff; border-radius: 2px; }}
  .mi-brand-name {{ font-weight: 680; color: {INK}; font-size: 0.98rem; line-height: 1.15; }}
  .mi-brand-sub {{ color: {INK_MUTED}; font-size: 0.74rem; }}
  .st-key-mi_nav [role="radiogroup"] {{ gap: 0.1rem; }}
  /* Menu = the page radio (key "page") restyled: no circles, full-width rows, active highlight */
  .st-key-mi_nav .stRadio, .st-key-mi_nav [role="radiogroup"] {{ width: 100% !important; }}
  .st-key-mi_nav [role="radiogroup"] {{ display: flex; flex-direction: column; align-items: stretch; }}
  .st-key-mi_nav [role="radiogroup"] > div {{ position: relative; width: 100%; margin: 0; }}
  .st-key-mi_nav [role="radiogroup"] > div::before {{ position: absolute; left: 0.6rem; top: -1.25rem;
      font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; color: {INK_MUTED}; }}
  .st-key-mi_nav [data-testid="stRadioOption"] {{ display: flex; width: 100%; box-sizing: border-box;
      padding: 0.38rem 0.6rem;
      border-radius: 8px; transition: background 0.12s; cursor: pointer; }}
  .st-key-mi_nav [data-testid="stRadioOption"] > div > div:not([data-testid]) {{ display: none; }}
  .st-key-mi_nav [data-testid="stRadioOption"]:hover {{ background: {PANEL}; }}
  .st-key-mi_nav [role="radiogroup"] > div[data-selected="true"] [data-testid="stRadioOption"] {{
      background: {BRAND_SOFT}; box-shadow: inset 3px 0 0 {BRAND}; }}
  .st-key-mi_nav [role="radiogroup"] > div[data-selected="true"] p {{ color: {BRAND}; font-weight: 620; }}
  .st-key-mi_nav [role="radiogroup"] p {{ font-size: 0.92rem; color: {INK_SECONDARY}; }}
  .st-key-mi_nav [role="radiogroup"] p span[role="img"] {{ margin-right: 0.35rem; font-size: 1.1rem; }}
  .mi-sidebar-label {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; color: {INK_MUTED};
                       text-transform: uppercase; margin: 0.2rem 0 -0.2rem 0; }}
  .mi-signature {{ border-top: 1px solid {BORDER}; margin-top: 1.4rem; padding-top: 0.8rem;
                   font-size: 0.8rem; color: {INK_MUTED}; }}
  .mi-signature a {{ color: {BRAND}; font-weight: 600; text-decoration: none; }}
  .mi-signature a:hover {{ text-decoration: underline; }}

  /* --- Phone-width screens --- */
  @media (max-width: 640px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 3rem; }}  /* clear of the menu button */
    h1, .mi-page-title {{ font-size: 1.3rem; }}
    .mi-steps {{ grid-template-columns: 1fr; }}
    [data-testid="stMetricValue"] {{ font-size: 1.25rem; }}
  }}
</style>
"""
