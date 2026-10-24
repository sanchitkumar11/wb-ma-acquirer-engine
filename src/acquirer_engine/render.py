"""Render one-pagers as deck-style Markdown: a headline, a strip of key numbers, four boxes of an
action title with bullets, the risks, and conviction. The Streamlit view (Step 5) uses the same
pieces.

The key-number strip comes straight from the fact pack, not from the LLM, so it can never be wrong.
"""

from __future__ import annotations

from acquirer_engine.factpack import adjacent_key
from acquirer_engine.onepager import CITATION, OnePager
from acquirer_engine.rings import WIDE_FOOTPRINTS
from acquirer_engine.wording import plain_terms

NOT_RESEARCHED = (
    "Buyer profile: not researched. There are no dated, sourced facts for this buyer, so its "
    "current "
    "status, mandate and size are unchecked; everything on this page comes from the deal file."
)

CITATION_LEGEND = (
    "D: deals in the provided CSV; K, T, V: figures computed from it; S: likelihood factors; "
    "P: dated, sourced buyer facts."
)
"""What the source tags on a page mean (the Markdown report and the app)."""
SECTIONS = [
    ("acquirer_overview", "Acquirer overview"),
    ("strategic_fit_thesis", "Strategic fit"),
    ("precedent_activity", "Precedent activity"),
    ("valuation_context", "Valuation context"),
]


def ordinal(n: int) -> str:
    """Return 1st, 2nd, 3rd, 4th..."""
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def region_label(region: str) -> str:
    """Return the tile label for deals in the target's region: "In the Midwest", or "National
    deals"."""
    return f"{region} deals" if region in WIDE_FOOTPRINTS else f"In the {region}"


def key_numbers(pack: dict) -> list[tuple[str, str]]:
    """Return (label, value) tiles for the strip under the headline."""
    r, k = pack["track_record"], {d["id"]: d["value"] for d in pack["derived"]}
    tiles = [
        ("Deals in the file", f"{r['deals']} ({r['first_year']}-{r['last_year']})"),
        ("Closed", k["K:closed_share"]),
        (
            "In the target's sub-sector / adjacent",
            f"{r['same_sub_sector_deals']} / {r[adjacent_key(pack['target']['sector'])]}",
        ),
        (region_label(pack["target"]["region"]), str(r["deals_in_target_region"])),
        ("Median deal", r["median_deal"]),
        ("Median EV/EBITDA paid", r["median_ev_ebitda_paid"]),
    ]
    if "sponsor_angles" in pack:
        tiles.append(
            (
                "Already owns a platform company in the sub-sector",
                pack["sponsor_angles"]["open_platform"],
            )
        )
    return tiles


def plain(text: str) -> str:
    """Return text with [fact ID] citations turned into small superscript-style source tags."""
    return CITATION.sub(lambda m: f"<sup>{m.group(1)}</sup>", plain_terms(text))


def _bought_as(precedent: dict, buyer: str) -> str:
    """Return " (as Atrium Health)" when an entity now merged into the buyer bought the deal."""
    return f" (as {precedent['buyer_as']})" if precedent["buyer_as"] != buyer else ""


def onepager_markdown(page: OnePager, pack: dict) -> str:
    """Return one buyer's page as Markdown."""
    b, rk = pack["buyer"], pack["ranking"]
    lines = [
        f"## {page.buyer}",
        f"### {plain(page.headline)}",
        f"*{b['type']} | Tier 1, priority #{rk['position_on_list']} of 10 | ranked "
        f"{ordinal(rk['overall_rank'])} of "
        f"{rk['buyers_scored']} buyers | likelihood {rk['likelihood']} | conviction "
        f"**{page.conviction.level}** | "
        f"deal data through {pack['target']['data_through']}*",
        "",
        "| " + " | ".join(label for label, _ in key_numbers(pack)) + " |",
        "|" + "---|" * len(key_numbers(pack)),
        "| " + " | ".join(value for _, value in key_numbers(pack)) + " |",
        "",
        f"> **What sets it apart:** {plain(page.differentiator)}",
        "",
    ]
    for i, (field, label) in enumerate(SECTIONS, 1):
        box = getattr(page, field)
        lines += [
            f"#### {i}. {label}: {plain(box.title)}",
            *[f"- {plain(p)}" for p in box.points],
            "",
        ]
        if field == "precedent_activity" and pack["precedents"]:
            lines += [
                "| Deal | Year | Sub-sector | Size | Type | Region | EV/EBITDA | EV/Revenue | "
                "Outcome |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            lines += [
                f"| {p['target']}{_bought_as(p, b['name'])} | {p['year']} | "
                f"{p['sub_sector']} | {p['size']} | {p['deal_type']} | {p['region']} | "
                f"{p['ev_ebitda']} | "
                f"{p['ev_revenue']} | {p['outcome']} |"
                for p in pack["precedents"]
            ]
            lines.append("")
    lines += ["#### 5. Risk flags", "| Risk | Concern | Mitigant | Reality |", "|---|---|---|---|"]
    lines += [
        f"| **{plain(r.title)}** | {plain(r.concern)} | {plain(r.mitigant)} | {plain(r.reality)} |"
        for r in page.risk_flags
    ]
    lines += [
        "",
        f"#### 6. Conviction: {page.conviction.level}",
        plain(page.conviction.rationale),
        "",
    ]
    if pack["profile_not_researched"]:
        lines.append(f"*{NOT_RESEARCHED}*")
    if pack["profile_facts"]:
        lines.append(
            "*Sources: "
            + "; ".join(
                f"{f['id']} {f['source']} (as of {f['as_of']})" for f in pack["profile_facts"]
            )
            + "*"
        )
    lines.append(f"*{CITATION_LEGEND}*")
    return "\n".join(lines)
