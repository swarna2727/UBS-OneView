"""Charts for the PDF artifacts (memo, meeting brief, decision record), drawn with ReportLab graphics.

Colours follow the workspace: light blue = current portfolio, dark blue = proposed / after,
status green / amber / red only for rule outcomes.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors

BEFORE = colors.HexColor("#86b6ef")
AFTER = colors.HexColor("#1c5cab")
SERIES = [colors.HexColor(c) for c in ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")]
INK, MUTED, GRID = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#e1e0d9")
WIDTH = 520


def _title(d: Drawing, text: str, height: float) -> None:
    d.add(String(0, height - 12, text, fontName="Helvetica-Bold", fontSize=9, fillColor=INK))


def _legend(d: Drawing, x: float, y: float, items: Sequence[Tuple[colors.Color, str]]) -> None:
    legend = Legend()
    legend.x, legend.y = x, y
    legend.alignment = "right"
    legend.fontName, legend.fontSize = "Helvetica", 7
    legend.columnMaximum = 8
    legend.boxAnchor = "nw"
    legend.dx = legend.dy = 7
    legend.colorNamePairs = list(items)
    d.add(legend)


def before_after_bars(title: str, categories: List[str], before: List[float], after: List[float],
                      unit: str = "%", width: float = WIDTH, height: float = 150) -> Drawing:
    """Grouped horizontal bars: current (light) vs proposed (dark) per category."""
    d = Drawing(width, height)
    _title(d, title, height)
    chart = HorizontalBarChart()
    chart.x, chart.y = 120, 18
    chart.width, chart.height = width - 220, height - 40
    chart.data = [list(reversed(before)), list(reversed(after))]
    chart.categoryAxis.categoryNames = list(reversed(categories))
    chart.categoryAxis.labels.fontName, chart.categoryAxis.labels.fontSize = "Helvetica", 7
    chart.categoryAxis.labels.boxAnchor = "e"
    chart.categoryAxis.joinAxisMode = "left"
    chart.valueAxis.valueMin = min(0, min(before + after)) * 1.15
    chart.valueAxis.valueMax = max(1, max(before + after)) * 1.15
    chart.valueAxis.labels.fontName, chart.valueAxis.labels.fontSize = "Helvetica", 7
    chart.valueAxis.labelTextFormat = "%.0f" + unit.replace("%", "%%")
    chart.valueAxis.gridStrokeColor, chart.valueAxis.visibleGrid = GRID, True
    chart.bars[0].fillColor, chart.bars[1].fillColor = BEFORE, AFTER
    chart.bars.strokeColor = None
    chart.barSpacing, chart.groupSpacing = 1, 6
    d.add(chart)
    _legend(d, width - 85, height - 25, [(BEFORE, "Current"), (AFTER, "Proposed")])
    return d


def risk_band(title: str, band: Sequence[float], points: Dict[str, float], width: float = WIDTH, height: float = 95) -> Drawing:
    """The client's risk band on a 1-5 scale with markers for willingness, capacity and portfolio scores."""
    d = Drawing(width, height)
    _title(d, title, height)
    x0, x1, y = 120, width - 100, 32
    scale = lambda v: x0 + (min(max(v, 1.0), 5.0) - 1.0) / 4.0 * (x1 - x0)
    d.add(Line(x0, y, x1, y, strokeColor=MUTED, strokeWidth=0.6))
    for tick in range(1, 6):
        d.add(Line(scale(tick), y - 3, scale(tick), y + 3, strokeColor=MUTED, strokeWidth=0.6))
        d.add(String(scale(tick), y - 13, str(tick), fontName="Helvetica", fontSize=7, fillColor=MUTED, textAnchor="middle"))
    d.add(Rect(scale(band[0]), y + 6, scale(band[1]) - scale(band[0]), 14, fillColor=colors.HexColor("#cde2fb"), strokeColor=None))
    d.add(String(scale(band[0]), y + 23, f"Applied band {band[0]:.1f}-{band[1]:.1f}", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    d.add(String(0, y + 9, "Risk score", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    items = []
    for i, (label, value) in enumerate(points.items()):
        color = SERIES[i % len(SERIES)]
        d.add(Rect(scale(value) - 3, y + 10, 6, 6, fillColor=color, strokeColor=colors.white))
        items.append((color, f"{label} {value:.2f}"))
    _legend(d, width - 95, height - 20, items)
    return d


def signed_bars(title: str, labels: List[str], values: List[float], width: float = WIDTH, height: Optional[float] = None) -> Drawing:
    """Horizontal bars for signed moves (event impact per holding, stress results)."""
    height = height or max(70, 18 * len(labels) + 40)
    d = Drawing(width, height)
    _title(d, title, height)
    chart = HorizontalBarChart()
    chart.x, chart.y = 120, 18
    chart.width, chart.height = width - 160, height - 40
    chart.data = [list(reversed(values))]
    chart.categoryAxis.categoryNames = list(reversed(labels))
    chart.categoryAxis.labels.fontName, chart.categoryAxis.labels.fontSize = "Helvetica", 7
    chart.categoryAxis.labels.boxAnchor = "e"
    chart.categoryAxis.joinAxisMode = "left"  # keep labels at the chart edge, not on the zero line
    low, high = min(values + [0]), max(values + [0])
    pad = max(abs(low), abs(high)) * 0.2 or 1
    chart.valueAxis.valueMin, chart.valueAxis.valueMax = low - pad, high + pad
    chart.valueAxis.labels.fontName, chart.valueAxis.labels.fontSize = "Helvetica", 7
    chart.valueAxis.labelTextFormat = "%.1f%%"
    chart.valueAxis.gridStrokeColor, chart.valueAxis.visibleGrid = GRID, True
    chart.bars.strokeColor = None
    for i, v in enumerate(reversed(values)):
        chart.bars[(0, i)].fillColor = SERIES[0] if v >= 0 else SERIES[7]
    d.add(chart)
    return d


def grouped_columns(title: str, groups: List[str], series: Dict[str, List[float]], width: float = WIDTH, height: float = 140) -> Drawing:
    """Vertical grouped columns, e.g. stress impact before vs after per scenario."""
    d = Drawing(width, height)
    _title(d, title, height)
    chart = VerticalBarChart()
    chart.x, chart.y = 40, 25
    chart.width, chart.height = width - 150, height - 45
    chart.data = list(series.values())
    chart.categoryAxis.categoryNames = groups
    chart.categoryAxis.labels.fontName, chart.categoryAxis.labels.fontSize = "Helvetica", 7
    chart.categoryAxis.joinAxisMode = "bottom"  # keep labels below the chart when every value is negative
    flat = [v for s in series.values() for v in s] + [0]
    pad = max(abs(min(flat)), abs(max(flat))) * 0.2 or 1
    chart.valueAxis.valueMin, chart.valueAxis.valueMax = min(flat) - pad, max(flat) + pad
    chart.valueAxis.labels.fontName, chart.valueAxis.labels.fontSize = "Helvetica", 7
    chart.valueAxis.labelTextFormat = "%.1f%%"
    chart.valueAxis.gridStrokeColor, chart.valueAxis.visibleGrid = GRID, True
    palette = [BEFORE, AFTER] if len(series) == 2 else SERIES
    for i in range(len(series)):
        chart.bars[i].fillColor = palette[i % len(palette)]
    chart.bars.strokeColor = None
    d.add(chart)
    _legend(d, width - 95, height - 20, [(palette[i % len(palette)], name) for i, name in enumerate(series)])
    return d


def allocation_pie(title: str, weights: Dict[str, float], width: float = 260, height: float = 140) -> Drawing:
    d = Drawing(width, height)
    _title(d, title, height)
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 10, 10, 100, 100
    pie.data = [v for _, v in items]
    pie.labels = None
    pie.slices.strokeColor, pie.slices.strokeWidth = colors.white, 1.5
    for i in range(len(items)):
        pie.slices[i].fillColor = SERIES[i % len(SERIES)]
    pie.innerRadiusFraction = 0.5
    d.add(pie)
    _legend(d, 125, height - 25, [(SERIES[i % len(SERIES)], f"{k} {v:.0f}%") for i, (k, v) in enumerate(items)])
    return d


# --------------------------------------------------------------------------------------
# Chart sets per artifact
# --------------------------------------------------------------------------------------

def _trades(ideas: List) -> List[Dict]:
    return [{"ticker": i.ticker, "action": i.action, "target_weight_pct": i.target_weight_pct,
             "funding_source_ticker": i.funding_source_ticker} for i in ideas]


def _by_asset_class(holdings: List) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in holdings:
        out[h.asset_class] = out.get(h.asset_class, 0.0) + h.weight_pct
    return out


def portfolio_change_charts(profile, ideas: List) -> List[Drawing]:
    """Allocation shift, risk score vs band, and stress impact for a proposed idea set."""
    from src.tools.event_tools import stress_impact
    from src.tools.exposure import portfolio_exposure_calculator
    from src.tools.risk_engine import portfolio_metrics

    delta = portfolio_exposure_calculator(profile.parsed_holdings, _trades(ideas))
    before, after = _by_asset_class(delta.before_holdings), _by_asset_class(delta.after_holdings)
    classes = sorted(set(before) | set(after), key=lambda c: -after.get(c, 0.0))
    risk_before = portfolio_metrics(delta.before_holdings, profile)["risk_score"]
    risk_after = portfolio_metrics(delta.after_holdings, profile)["risk_score"]
    stress_before, stress_after = stress_impact(delta.before_holdings), stress_impact(delta.after_holdings)
    return [
        before_after_bars("Asset allocation: current vs proposed", classes,
                          [round(before.get(c, 0.0), 1) for c in classes], [round(after.get(c, 0.0), 1) for c in classes]),
        risk_band(f"Portfolio risk vs the client's {profile.calculated_risk_tier} band", profile.risk_score_range,
                  {"Current": risk_before, "Proposed": risk_after}),
        grouped_columns("Illustrative stress impact (estimated portfolio move)", list(stress_before),
                        {"Current": list(stress_before.values()), "Proposed": list(stress_after.values())}),
    ]


def profile_charts(profile, willingness: Optional[float] = None, capacity: Optional[float] = None) -> List[Drawing]:
    """Current allocation and where the portfolio sits against the client's risk band."""
    from src.tools.risk_engine import portfolio_metrics

    points = {}
    if willingness is not None:
        points["Willingness"] = willingness
    if capacity is not None:
        points["Capacity"] = capacity
    points["Current portfolio"] = portfolio_metrics(profile.parsed_holdings, profile)["risk_score"]
    return [allocation_pie("Current allocation by asset class", profile.current_asset_class_weights, width=WIDTH),
            risk_band("Risk profile", profile.risk_score_range, points)]


def event_charts(impact) -> List[Drawing]:
    rows = sorted(impact.holdings, key=lambda r: r.contribution_pct)
    return [signed_bars(f"{impact.event_title}: estimated contribution by holding (portfolio % points)",
                        [r.ticker for r in rows], [r.contribution_pct for r in rows])]
