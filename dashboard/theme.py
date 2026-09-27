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

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'

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

TEMPLATE_NAME = "marketing"
TOP_MARGIN = 56                 # room for the title plus a small note above the plot
TOP_MARGIN_WITH_LEGEND = 84     # title row + legend row
LABEL_WRAP = 24                 # category labels longer than this wrap onto a second line
TITLE_WRAP = 60


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


def color_map(entities) -> dict[str, str]:
    """Fixed colour per entity, based on the sorted list of ALL entities in the dataset (not the
    filtered view), so a channel keeps its colour whatever is filtered."""
    ordered = sorted({str(e) for e in entities})
    return {e: CATEGORICAL[i] if i < len(CATEGORICAL) else NEUTRAL for i, e in enumerate(ordered)}


# Custom CSS for a calm, corporate look (Streamlit's own theme is set in .streamlit/config.toml).
APP_CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.65rem; font-weight: 650; }}
  h2 {{ font-size: 1.25rem; font-weight: 620; margin-top: 0.6rem; }}
  h3 {{ font-size: 1.05rem; font-weight: 600; }}
  [data-testid="stMetric"] {{ background: {SURFACE}; }}
  [data-testid="stMetricLabel"] p {{ color: {INK_SECONDARY}; font-size: 0.85rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 620; color: {INK}; }}
  .mi-caption {{ color: {INK_SECONDARY}; font-size: 0.86rem; margin: -0.3rem 0 0.8rem 0; }}
  .mi-tag {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.75rem;
             background: {PANEL}; color: {INK_SECONDARY}; margin-right: 6px; }}
  .mi-finding {{ border-left: 3px solid {PRIMARY}; padding: 0.35rem 0.8rem; margin: 0.35rem 0;
                 background: {SURFACE}; }}
  .mi-finding .mi-title {{ font-weight: 600; color: {INK}; }}
  .mi-empty {{ border: 1px dashed {AXIS}; border-radius: 8px; padding: 1.2rem; color: {INK_SECONDARY};
               background: {SURFACE}; }}
</style>
"""
