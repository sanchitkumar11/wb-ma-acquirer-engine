"""Plotly figures for the summary page: buyer map, funnel, and buyers by home sub-sector (rings).

Static by design (the app shows them with staticPlot): they are slides, not dashboards. Colours:
conviction uses the validated reference blue / orange / aqua (aqua darkened to #14996b so labels on
it read; its colour-blind separation from orange is in the floor band, so points also carry direct
labels); everything else is navy, grey and the single teal accent. Identity is never colour alone:
Tier 1 points are
labelled and the conviction legend is always shown. export() writes PNGs for the Markdown report.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from acquirer_engine.rings import GROUP, adjacent_sub_sectors, model_words, ring, ring_titles
from acquirer_engine.summary import funnel, map_points, target_price_ranges

if TYPE_CHECKING:
    from collections.abc import Callable

    from acquirer_engine.report_types import Report
    from acquirer_engine.review import Move

log = logging.getLogger(__name__)
DEAL_SIZE_TICKS = [
    25,
    50,
    100,
    200,
    400,
    800,
    1600,
]  # $M, doubling: the deal-size axis on a ratio scale
DEAL_SIZE_LABELS = [f"${t / 1000:.1f}B" if t >= 1000 else f"${t}M" for t in DEAL_SIZE_TICKS]
NAVY, GREY, LIGHT = "#1B2A4A", "#3C4650", "#D9DEE5"
# Conviction is ordinal, so it gets a sequential ramp of the theme navy, not categorical hues
CONVICTION_COLOUR = {
    "High": "#2a78d6",
    "Medium": "#eb6834",
    "Low": "#14996b",
}  # validated reference slots; aqua darkened for labels
LAYOUT = dict(
    template="simple_white",
    font=dict(family="Arial", size=12, color=GREY),
    margin=dict(l=60, r=30, t=50, b=50),
)


def buyer_map(rep: Report) -> go.Figure:
    """Scatter: sector evidence (across) vs usual deal size against the ask (up, log).

    Tier 1: filled, coloured by conviction, labelled "priority. name"; Tier 2: hollow grey, name on
    hover. Marker size = number of deals. Reference line at usual deal = ask; quadrant captions.
    """
    pts = map_points(rep)

    def size(p: dict) -> float:
        """Return the marker size in px: 8-19 for Tier 1, 5-16 for Tier 2, growing with deal
        count."""
        return (8 if p["tier"] == "Tier 1" else 5) + 0.45 * min(p["deals"], 25)

    fig = go.Figure()
    t2 = [p for p in pts if p["tier"] == "Tier 2"]
    fig.add_trace(
        go.Scatter(
            x=[p["fit"] for p in t2],
            y=[p["deal_mm"] for p in t2],
            mode="markers",
            name="Tier 2",
            hovertext=[p["buyer"] for p in t2],
            hoverinfo="text",
            marker=dict(
                symbol="circle-open",
                color="#8A94A6",
                size=[size(p) for p in t2],
                line=dict(width=1.5),
            ),
        )
    )
    t1 = sorted((p for p in pts if p["tier"] == "Tier 1"), key=lambda p: (p["deal_mm"], p["fit"]))
    flat = [(p["fit"], math.log10(p["deal_mm"]), f"{p['priority']}. {p['buyer']}") for p in t1]
    xr = (min(p["fit"] for p in pts) - 0.3, max(p["fit"] for p in pts) + 0.3)
    yr = (
        min(math.log10(p["deal_mm"]) for p in pts) - 0.1,
        max(math.log10(p["deal_mm"]) for p in pts) + 0.1,
    )
    where = dict(
        zip((p["buyer"] for p in t1), _label_sides(flat, xr, yr), strict=True)
    )  # first free side, so labels don't collide
    for level, colour in CONVICTION_COLOUR.items():
        sel = [p for p in t1 if p["conviction"] == level]
        if sel:
            fig.add_trace(
                go.Scatter(
                    x=[p["fit"] for p in sel],
                    y=[p["deal_mm"] for p in sel],
                    mode="markers+text",
                    name=f"Tier 1, {level} conviction",
                    text=[f"{p['priority']}. {p['buyer']}" for p in sel],
                    textposition=[where[p["buyer"]] for p in sel],
                    textfont=dict(size=10, color=GREY),
                    hovertext=[p["buyer"] for p in sel],
                    hoverinfo="text",
                    marker=dict(
                        color=colour,
                        symbol=[
                            "diamond" if p["type"] == "Financial sponsor" else "circle" for p in sel
                        ],
                        size=[size(p) for p in sel],
                        line=dict(color="white", width=2),
                    ),
                )
            )
    mx = sorted(p["fit"] for p in pts)[len(pts) // 2]
    fig.add_vline(x=mx, line=dict(color=LIGHT, dash="dash"))
    ask = rep.res.target.enterprise_value_mm
    xs = [p["fit"] for p in pts]
    fig.add_trace(
        go.Scatter(
            x=[min(xs) - 0.3, max(xs) + 0.3],
            y=[ask, ask],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color="#9AA4B2", dash="dash", width=1.5),
        )
    )  # a trace, so it sits in plain data units
    fig.add_annotation(
        x=max(xs) + 0.3,
        y=math.log10(ask),
        text=f"${ask:.0f}M ask",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=10, color="#8A94A6"),
    )  # annotations on a log axis take log10 values
    corners = (
        (0.99, 0.99, "Proven sector fit, bigger cheques", "right", "top"),
        (0.01, 0.99, "Bigger cheques, less sector proof", "left", "top"),
        (0.99, 0.01, "Proven sector fit, smaller cheques", "right", "bottom"),
        (0.01, 0.01, "Weakest on both", "left", "bottom"),
    )
    for x, y, text, xa, ya in corners:
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor=xa,
            yanchor=ya,
            font=dict(size=11, color="#8A94A6"),
        )
    fig.update_layout(
        **{**LAYOUT, "margin": dict(l=60, r=30, t=50, b=160)},
        height=640,
        legend=dict(orientation="h", y=-0.2, title=None),
        title=dict(text="Buyer map: track record against typical deal size", font=dict(color=NAVY)),
    )
    fig.add_annotation(
        x=0,
        y=-0.32,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        font=dict(size=10, color=GREY),
        text="Shape: diamond = sponsor, circle = strategic. Size: bigger = more deals in the file.",
    )  # the key, under the legend
    sector = rep.res.target.sector
    scope = (
        f"deals in the target's sub-sector and neighbouring {model_words(sector)[1]}"
        if adjacent_sub_sectors(sector)
        else "deals in the target's sub-sector"
    )
    fig.update_xaxes(
        title=dict(
            text="Track record in this kind of business<br><sup>"
            f"{scope}; recent deals count more<br>Vertical axis is a ratio scale: each step "
            "doubles.</sup>"
        ),
        zeroline=False,
    )
    fig.update_yaxes(
        title="Typical deal size",
        type="log",
        zeroline=False,
        tickvals=DEAL_SIZE_TICKS,
        ticktext=DEAL_SIZE_LABELS,
    )
    return fig


def football_field_figure(rep: Report) -> go.Figure:
    """Target-level football field: comps range, what Tier 1 buyers pay, sponsor LBO maximum; ask
    as a dashed line."""
    rows, ask = target_price_ranges(rep), rep.res.target.enterprise_value_mm
    fig = go.Figure()
    min_width = 0.01 * max(
        r["high"] for r in rows
    )  # a range of one value (no platform premium) still shows as a sliver
    for r in rows:
        point = round(r["low"]) == round(r["high"])
        fig.add_trace(
            go.Bar(
                y=[r["label"]],
                x=[max(r["high"] - r["low"], min_width)],
                base=[r["low"] - (min_width / 2 if point else 0)],
                orientation="h",
                showlegend=False,
                marker=dict(color="#0E7C86", cornerradius=4),
                hovertext=r["detail"],
                hoverinfo="text",
                text=f"${r['low']:,.0f}M" if point else f"${r['low']:,.0f}M-${r['high']:,.0f}M",
                textposition="outside",
                textfont=dict(color=GREY),
                cliponaxis=False,
            )
        )
    fig.add_vline(
        x=ask,
        line=dict(color=NAVY, dash="dash", width=2),
        annotation_text=f"ask ${ask:.0f}M",
        annotation_position="top",
    )
    fig.update_layout(
        **LAYOUT,
        height=260,
        bargap=0.45,
        title=dict(text="Where the ask sits", font=dict(color=NAVY)),
    )
    fig.update_yaxes(autorange="reversed", title=None)
    lo, hi = min(r["low"] for r in rows), max(r["high"] for r in rows)
    fig.update_xaxes(
        title="Enterprise value ($M); multiples converted at the EBITDA implied by the ask at the "
        "comparable median",
        range=[0.93 * min(lo, ask), 1.15 * max(hi, ask)],
    )
    return fig


def funnel_figure(rep: Report) -> go.Figure:
    """Bars, one per stage (not a funnel shape: the stages are not nested), each drop annotated
    with its reason."""
    stages = funnel(rep)
    log = [m for m in rep.res.review.log if m.model_rank <= 25]

    def names(pred: Callable[[Move], bool]) -> str:
        """Return the buyers of the logged moves that match."""
        return ", ".join(m.buyer for m in log if pred(m))

    reasons = {
        "Removed (defunct or acquired)": names(lambda m: m.action == "removed"),
        "Merged into a listed buyer": "; ".join(
            f"{m.buyer} into {m.merged_into}" for m in log if m.action == "merged"
        ),
        "Off-mandate (to Tier 2)": names(lambda m: m.code == "MANDATE_MISMATCH"),
        "Moved by an accepted rule (to Tier 2)": names(
            lambda m: m.code in ("CAPACITY_RULE", "GEOGRAPHY_RULE")
        ),
        "Flagged by the deal team (off the list)": ", ".join(
            m.buyer for m in rep.res.review.log if m.code == "USER_FLAG"
        ),
        "Moved by the deal team (accepted suggestions)": ", ".join(
            f"{m.buyer} {m.action}"
            for m in rep.res.review.log
            if m.code in ("USER_PROMOTE", "USER_DEMOTE")
        ),
    }
    if reasons["Off-mandate (to Tier 2)"]:
        reasons["Off-mandate (to Tier 2)"] += ": technology investors"
    colours = [
        {
            "Buyers in the file, scored": NAVY,
            "Top 25 reviewed": NAVY,
            "Tier 1 (one-pagers)": "#0E7C86",
            "Tier 2 (incl. adjacent buyers added)": "#5BA8AF",
        }.get(s["stage"], "#8A94A6")
        for s in stages
    ]
    text = [
        f"{s['n']}  " + (f"<i>{reasons[s['stage']]}</i>" if reasons.get(s["stage"]) else "")
        for s in stages
    ]
    fig = go.Figure(
        go.Bar(
            y=[s["stage"] for s in stages],
            x=[s["n"] for s in stages],
            orientation="h",
            marker=dict(color=colours, cornerradius=4),
            text=text,
            textposition="outside",
            textfont=dict(color=GREY),
            cliponaxis=False,
        )
    )
    fig.update_layout(
        **LAYOUT,
        height=380,
        showlegend=False,
        title=dict(text="From every buyer in the file to the call list", font=dict(color=NAVY)),
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(
        title="Buyers", showgrid=False, range=[0, 1.5 * max(s["n"] for s in stages)]
    )  # room for reasons
    return fig


def rings_figure(rep: Report) -> go.Figure:
    """Concentric rings around the target's sub-sector. Each home sub-sector is a grey bubble (size
    =
    buyers); listed buyers sit beside their sub-sector, Tier 1 numbered by priority and coloured by
    conviction, Tier 2 smaller. Ring labels carry the buyer count."""
    ranked, target = rep.res.ranked.set_index("acquirer"), rep.res.target.sector
    pts = {p["buyer"]: p for p in map_points(rep)}
    fig = go.Figure()
    radius = {0: 0.0, 1: 1.7, 2: 2.6, 3: 3.4}  # inner ring wider: room for the centre label
    for k in range(1, 4):
        rk = radius[k]
        fig.add_shape(
            type="circle",
            x0=-rk,
            y0=-rk,
            x1=rk,
            y1=rk,
            line=dict(color=LIGHT, width=1.5),
            fillcolor="rgba(0,0,0,0)",
        )
    by_ring: dict[int, list[str]] = {}
    for sector in GROUP:
        by_ring.setdefault(ring(sector, target), []).append(sector)
    start = {
        1: math.pi / 4,
        2: 0.0,
        3: math.pi / 2,
    }  # fixed starting angles keep bubbles and labels apart
    ring_count = {k: int(ranked.home_sector.isin(secs).sum()) for k, secs in by_ring.items()}
    for k, sectors in by_ring.items():
        for i, sector in enumerate(sectors):
            angle = start.get(k, 0.0) + 2 * math.pi * i / len(sectors)
            x, y = radius[k] * math.cos(angle), radius[k] * math.sin(angle)
            n = int((ranked.home_sector == sector).sum())
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    showlegend=False,
                    hoverinfo="skip",
                    marker=dict(
                        size=14 + 3 * n, color="#E6EAF0", line=dict(color="white", width=2)
                    ),
                    text=[f"<b>{sector}</b> ({n})"],
                    textfont=dict(size=10, color=GREY),
                    textposition="bottom center"
                    if k == 0
                    else ("middle right" if x >= 0 else "middle left"),
                )
            )
            listed = [b for b in ranked.index[ranked.home_sector == sector] if b in pts]
            for j, b in enumerate(listed):  # small markers around the sub-sector bubble
                if k == 0:  # around the centre bubble
                    a2, r2 = math.pi / 2 + 2 * math.pi * j / max(len(listed), 1), 0.45
                elif k == 1:  # inward of the bubble, fanned out
                    a2, r2 = angle + (j - (len(listed) - 1) / 2) * 0.22, radius[k] - 0.5
                else:  # along the ring beside the bubble, clear of its label
                    a2, r2 = (
                        angle + (1 if j % 2 else -1) * (0.24 + 0.13 * (j // 2)),
                        radius[k],
                    )  # alternate either side
                bx, by_ = r2 * math.cos(a2), r2 * math.sin(a2)
                p = pts[b]
                tier1 = p["tier"] == "Tier 1"
                fig.add_trace(
                    go.Scatter(
                        x=[bx],
                        y=[by_],
                        mode="markers+text" if tier1 else "markers",
                        showlegend=False,
                        text=[str(p["priority"])] if tier1 else None,
                        textfont=dict(size=9, color="white"),
                        hovertext=[f"{b} ({p['tier']}, {p['conviction']})"],
                        hoverinfo="text",
                        marker=dict(
                            size=18 if tier1 else 9,
                            color=CONVICTION_COLOUR[p["conviction"]] if tier1 else "white",
                            line=dict(color=CONVICTION_COLOUR[p["conviction"]], width=1.5),
                        ),
                    )
                )
    ring_text = ring_titles(rep.res.target.sector)
    for k, text in ring_text.items():
        if k and ring_count.get(
            k, 0
        ):  # an empty ring (Health IT has no adjacent sub-sector) gets no caption
            fig.add_annotation(
                x=0,
                y=-radius[k],
                text=f"{text}: {ring_count.get(k, 0)} buyers",
                showarrow=False,
                yshift=-9,
                font=dict(size=10, color="#8A94A6"),
                bgcolor="white",
            )
    key = "  ".join(f"{i}. {b}" for i, b in enumerate(rep.tier1, 1))
    fig.add_annotation(
        x=0.5,
        y=-0.04,
        xref="paper",
        yref="paper",
        text=key,
        showarrow=False,
        font=dict(size=9, color=GREY),
    )
    fig.update_layout(
        **LAYOUT,
        height=660,
        title=dict(
            text=f"Buyers by home sub-sector ({ring_text[0].lower()}: "
            f"{ring_count.get(0, 0)} buyers); numbers = Tier 1 priority, hollow = Tier 2",
            font=dict(color=NAVY),
        ),
    )
    fig.update_xaxes(visible=False, range=[-4.4, 4.4], scaleanchor="y")
    fig.update_yaxes(visible=False, range=[-4.1, 4.1])
    return fig


def _label_sides(
    points: list[tuple[float, float, str]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    width_px: float = 980,
    height_px: float = 400,
) -> list[str]:
    """Return a text position per point so labels don't overlap: greedy, first free side of above,
    below, right, left (inward first at the plot edges). Points are (x, log10 y, label); boxes are
    estimated at 6 px per character and 14 px high in a plot of width_px by height_px."""

    def px(x: float, y: float) -> tuple[float, float]:
        """Return a point's position in pixels."""
        return (
            (x - x_range[0]) / (x_range[1] - x_range[0]) * width_px,
            (y - y_range[0]) / (y_range[1] - y_range[0]) * height_px,
        )

    def box(cx: float, cy: float, w: float, side: str) -> tuple[float, float, float, float]:
        """Return a label's bounding box for one side of its point."""
        return {
            "top center": (cx - w / 2, cy + 8, cx + w / 2, cy + 22),
            "bottom center": (cx - w / 2, cy - 22, cx + w / 2, cy - 8),
            "middle right": (cx + 9, cy - 7, cx + 9 + w, cy + 7),
            "middle left": (cx - 9 - w, cy - 7, cx - 9, cy + 7),
        }[side]

    def overlap(p: tuple[float, ...], q: tuple[float, ...]) -> bool:
        """Return True if two boxes intersect."""
        return p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]

    placed, sides = (
        [(cx - 7, cy - 7, cx + 7, cy + 7) for cx, cy in (px(x, y) for x, y, _ in points)],
        [],
    )
    for x, y, label in points:
        cx, cy = px(x, y)
        w = 6 * len(label)
        order = ["top center", "bottom center", "middle right", "middle left"]
        if cx + 9 + w > width_px:
            order = ["middle left", "top center", "bottom center"]
        elif cx - w / 2 < 0:
            order = ["middle right", "top center", "bottom center"]
        fits = [
            o
            for o in order
            if not any(
                overlap(box(cx, cy, w, o), q)
                for q in placed
                if q != (cx - 7, cy - 7, cx + 7, cy + 7)
            )
        ]
        side = (fits or order)[0]
        sides.append(side)
        placed.append(box(cx, cy, w, side))
    return sides


def compare_map(a: Report, b: Report) -> go.Figure:
    """Both targets' Tier 1 on one buyer map: track record in each target's business (across)
    against the
    buyer's typical deal size (up, ratio scale), one colour per target, a dotted line joining a
    shared
    buyer's two positions, and each target's ask as a dashed line in its colour."""
    colours = {
        0: CONVICTION_COLOUR["High"],
        1: CONVICTION_COLOUR["Medium"],
    }  # validated adjacent pair (blue, orange)
    fig, pos = go.Figure(), {}
    both = [[p for p in map_points(rep) if p["tier"] == "Tier 1"] for rep in (a, b)]
    flat = [(p["fit"], math.log10(p["deal_mm"]), p["buyer"]) for pts in both for p in pts]
    xr = (min(x for x, _, _ in flat) - 0.3, max(x for x, _, _ in flat) + 0.3)
    yr = (min(y for _, y, _ in flat) - 0.1, max(y for _, y, _ in flat) + 0.1)
    all_sides = _label_sides(flat, xr, yr)
    sides = [all_sides[: len(both[0])], all_sides[len(both[0]) :]]
    for i, rep in enumerate((a, b)):
        t = rep.res.target
        name = f"{'AB'[i]}: {t.sector}, ${t.enterprise_value_mm:.0f}M"
        pts = both[i]
        for p in pts:
            pos.setdefault(p["buyer"], []).append((p["fit"], p["deal_mm"]))
        fig.add_trace(
            go.Scatter(
                x=[p["fit"] for p in pts],
                y=[p["deal_mm"] for p in pts],
                mode="markers+text",
                name=name,
                text=[p["buyer"] for p in pts],
                textfont=dict(size=10, color=GREY),
                textposition=sides[i],
                marker=dict(
                    size=13,
                    color=colours[i],
                    line=dict(color="white", width=2),
                    symbol=[
                        "diamond" if p["type"] == "Financial sponsor" else "circle" for p in pts
                    ],
                ),
                hovertext=[f"{p['buyer']} ({name}; {p['conviction']} conviction)" for p in pts],
                hoverinfo="text",
            )
        )
    xs = [x for xy in pos.values() for x, _ in xy]
    for i, rep in enumerate(
        (a, b)
    ):  # ask lines as traces in data units: a shape on a log axis misplaces them
        ask = rep.res.target.enterprise_value_mm
        fig.add_trace(
            go.Scatter(
                x=[min(xs) - 0.3, max(xs) + 0.3],
                y=[ask, ask],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(color=colours[i], dash="dash", width=1.5),
            )
        )
        fig.add_annotation(
            x=1.005,
            xref="paper",
            y=math.log10(ask),
            text=f"{'AB'[i]} ask ${ask:.0f}M",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color=GREY),
        )  # in the right margin; log axis takes log10
    for _buyer, xy in pos.items():
        if len(xy) == 2:
            fig.add_trace(
                go.Scatter(
                    x=[xy[0][0], xy[1][0]],
                    y=[xy[0][1], xy[1][1]],
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(color="#9AA4B2", dash="dot", width=1.5),
                )
            )
    fig.update_layout(
        **{**LAYOUT, "margin": dict(l=60, r=110, t=50, b=110)},
        height=560,
        legend=dict(orientation="h", y=-0.22, title=None),
        title=dict(
            text="Both targets' Tier 1 (diamond = sponsor; a dotted line joins a buyer on both "
            "lists)",
            font=dict(color=NAVY),
        ),
    )
    fig.update_xaxes(
        title=dict(
            text="Track record in each target's kind of business<br><sup>deals in that target's "
            "sub-sector "
            "and its business model; recent deals count more</sup>"
        ),
        zeroline=False,
    )
    fig.update_yaxes(
        title="Typical deal size",
        type="log",
        zeroline=False,
        tickvals=DEAL_SIZE_TICKS,
        ticktext=DEAL_SIZE_LABELS,
    )
    return fig


# common browser locations, tried only when BROWSER_PATH is not set (see .env.example)
BROWSER_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
PNG_WIDTH, PNG_SCALE = 1100, 2


def write_png(fig: go.Figure, path: Path) -> bool:
    """Write a figure as a PNG; return False (and log why) if no Chromium browser can be started.

    kaleido renders through a Chromium browser. Without one the report still renders, just
    without images, so a missing browser is a warning, not an error. Only the browser and file
    errors are caught (kaleido raises RuntimeError subclasses such as ChromeNotFoundError).
    """
    if not os.environ.get("BROWSER_PATH"):
        found = next((b for b in BROWSER_CANDIDATES if Path(b).exists()), None)
        if found:
            os.environ["BROWSER_PATH"] = found
    try:
        fig.write_image(str(path), width=PNG_WIDTH, scale=PNG_SCALE)
    except (RuntimeError, OSError) as exc:
        log.warning(
            "Chart %s not exported (%s); set BROWSER_PATH to a Chromium browser", path.name, exc
        )
        return False
    return True


def export(rep: Report, out_dir: Path) -> dict[str, str]:
    """Write the four summary figures as PNGs; return name -> path relative to the report.

    A figure that can't be written (no browser) is left out of the dict, so the report omits it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    makers = (
        ("buyer_map", buyer_map),
        ("football_field", football_field_figure),
        ("funnel", funnel_figure),
        ("rings", rings_figure),
    )
    return {
        name: f"{out_dir.name}/{name}.png"
        for name, make in makers
        if write_png(make(rep), out_dir / f"{name}.png")
    }
