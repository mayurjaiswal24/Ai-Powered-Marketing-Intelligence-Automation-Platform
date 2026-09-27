"""Small percentages never show as 0.0%, and chart titles never collide with legends."""

import pandas as pd
from openpyxl import load_workbook

from dashboard import charts, theme
from reports.pdf_charts import _wrap
from utils.formatting import format_change, format_pct, format_value


def test_small_percentages_get_more_decimals():
    assert format_pct(12.345) == "12.3%"
    assert format_pct(0.456) == "0.46%"          # below 1%: 2 decimals
    assert format_pct(0.0456) == "0.046%"        # below 0.1%: 3 decimals
    assert format_pct(0) == "0.0%"               # a real zero stays 0.0%
    assert format_pct(None) == "N/A"
    assert format_value(0.04, "percent") == "0.040%"
    assert format_change(0.3) == "+0.30%" and format_change(-0.05) == "-0.050%"


def test_excel_small_percent_formats(tmp_path):
    from reports.excel_report import _Book
    path = tmp_path / "p.xlsx"
    book = _Book(path)
    ws = book.sheet("t")
    for row, value in enumerate([12.3, 0.45, 0.045, 0.0]):
        book.write_value(ws, row, 0, value, book.fmt["percent"])
    book.writer.close()
    cells = load_workbook(path)["t"]
    formats = [cells.cell(row=r, column=1).number_format for r in range(1, 5)]
    assert formats == ['0.0"%"', '0.00"%"', '0.000"%"', '0.0"%"']
    assert cells.cell(row=3, column=1).value == 0.045           # still a real number


def test_title_sits_above_the_legend():
    df = pd.DataFrame({"channel": ["A very long channel name for testing wrap", "B"],
                       "spend_share": [60.0, 40.0], "revenue_share": [55.0, 45.0]})
    fig = charts.share_comparison_chart(df, "channel", {"Share of spend": "spend_share",
                                                        "Share of revenue": "revenue_share"},
                                        "Where the money goes and what it returns")
    layout, template = fig.layout, fig.layout.template.layout
    assert layout.showlegend
    assert template.title.yref == "container" and template.title.y == 1   # title at the very top
    assert template.legend.y >= 1.0 and template.legend.yanchor == "bottom"  # legend above plot
    assert layout.margin.t >= theme.TOP_MARGIN_WITH_LEGEND               # room for both rows
    assert "<br>" in fig.data[0].y[-1] or "<br>" in fig.data[0].y[0]     # long label wrapped


def test_long_titles_and_labels_wrap():
    fig = charts.bar_chart(pd.DataFrame({"c": ["x"], "v": [1.0]}), "c", "v", "count",
                           "A very long chart title that would otherwise run off the edge of a phone")
    assert "<br>" in fig.layout.title.text and fig.layout.margin.t > theme.TOP_MARGIN
    assert _wrap("Smartphone Accessories and Other Gadgets", 20).count("\n") == 1
