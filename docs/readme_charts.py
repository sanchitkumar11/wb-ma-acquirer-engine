"""Draw the README's result charts from the numbers in docs/experiment-log.md (Sections 6-7).

Reads nothing: the figures are the logged results, typed in once here. Writes PNGs to
docs/images/. Run from the repo root: python docs/readme_charts.py (needs a Chromium browser
for kaleido; see BROWSER_PATH in .env.example).
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = Path(__file__).parent / "images"
BLUE, ORANGE, GREY, NAVY = "#2a78d6", "#eb6834", "#3C4650", "#1B2A4A"   # validated adjacent pair
LAYOUT = dict(template="simple_white", font=dict(family="Arial", size=13, color=GREY),
              margin=dict(l=60, r=30, t=70, b=50), height=380, width=900, bargap=0.35, bargroupgap=0.08,
              legend=dict(orientation="h", y=1.12, x=0, title=None))


def _bar(name: str, x: list[str], y: list[float], colour: str, fmt: str) -> go.Bar:
    return go.Bar(name=name, x=x, y=y, marker=dict(color=colour, cornerradius=4), text=[fmt.format(v) for v in y],
                  textposition="outside", textfont=dict(color=GREY), cliponaxis=False)


def ranking() -> go.Figure:
    """Back-test: actual buyer within the top 10 / 20 / 25, baseline vs model (127 deals, 2022-24)."""
    x = ["Top 10", "Top 20", "Top 25"]
    fig = go.Figure([_bar("Most active buyer (baseline)", x, [44, 65, 68], ORANGE, "{:.0f}%"),
                     _bar("Ranking model", x, [47, 91, 96], BLUE, "{:.0f}%")])
    fig.update_layout(**LAYOUT, title=dict(text="Actual buyer found in the list (127 unseen deals, 2022-24)", font=dict(color=NAVY)))
    fig.update_yaxes(range=[0, 110], ticksuffix="%", title=None, showgrid=True, gridcolor="#EEF1F5")
    return fig


def writing() -> go.Figure:
    """W2 vs W3 on the live target, mean of 3 runs: pages passing, thesis overlap, cost."""
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.09, subplot_titles=(
        "Pages passing the validator (of 10)", "Highest overlap between two theses", "Cost per run ($)"))
    arms = (("W2: one call per buyer", ORANGE, [4.3, 0.35, 0.52]),
            ("W3: + peer lines + repair (chosen)", BLUE, [10, 0.23, 0.79]))
    fmts = ("{:.1f}", "{:.2f}", "${:.2f}")
    for name, colour, values in arms:
        for col, (v, fmt) in enumerate(zip(values, fmts, strict=True), 1):
            bar = _bar(name, [name.split(":")[0]], [v], colour, fmt)
            bar.showlegend = col == 1
            fig.add_trace(bar, row=1, col=col)
    fig.update_layout(**{**LAYOUT, "width": 1000, "margin": dict(l=30, r=30, t=90, b=80),
                         "legend": dict(orientation="h", y=-0.12, x=0, title=None)},
                      title=dict(text="Writing the pages: W2 vs W3 (live target, 3 runs each)", font=dict(color=NAVY)))
    fig.update_annotations(font=dict(size=13, color=GREY))   # the three panel titles
    fig.update_yaxes(showticklabels=False, showgrid=False, ticks="")
    for col, top in enumerate((11.5, 0.42, 0.95), 1):
        fig.update_yaxes(range=[0, top], row=1, col=col)
    return fig


def listbuild() -> go.Figure:
    """A1 rules vs A2 rules + LLM reviewer on 30 frozen back-test deals: a tie."""
    x = ["Actual buyer in Tier 1", "Actual buyer on the list", "Tier 1 names passing banker checks"]
    fig = go.Figure([_bar("A1: rules (chosen)", x, [46.7, 93.3, 98.7], BLUE, "{:.1f}%"),
                     _bar("A2: rules + LLM reviewer (now opt-in suggestions)", x, [46.7, 93.3, 100], ORANGE, "{:.1f}%")])
    fig.update_layout(**LAYOUT, title=dict(text="Building the list: A1 vs A2 (30 frozen deals; one deal = 3.3 points)", font=dict(color=NAVY)))
    fig.update_yaxes(range=[0, 115], ticksuffix="%", title=None, showgrid=True, gridcolor="#EEF1F5")
    return fig


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, make in (("ranking_backtest", ranking), ("writing_w2_w3", writing), ("listbuild_a1_a2", listbuild)):
        make().write_image(str(OUT / f"{name}.png"), scale=2)
        print("wrote", OUT / f"{name}.png")
