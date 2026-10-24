"""Comparison mode: how the buyer list changes between two target profiles, and why.

Everything except the short note is computed: shared Tier 1 buyers, buyers unique to each list,
and for each shared buyer its conviction under both targets with the named signals that differ
(the conviction basis is a list of named points, so the change is the points that are not in
both). The note is written by the LLM from a comparison pack and checked like the executive
summary (every number and ID must be in the pack); with no cached note, a data-built one is used.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import yaml

from acquirer_engine.execsummary import written_summary
from acquirer_engine.factpack import buyer_conviction, money
from acquirer_engine.paths import ROOT
from acquirer_engine.summary import call_list

if TYPE_CHECKING:
    from acquirer_engine.report_types import Report

PROMPTS = yaml.safe_load((ROOT / "prompts/compare_v2.yaml").read_text())

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _short(rep: Report) -> str:
    """Return a short name for a target, e.g. "Health IT, $300M, National"."""
    t = rep.res.target
    return f"{t.sector}, {money(t.enterprise_value_mm)}, {t.region}"


def conviction_points(basis: list[str]) -> dict[str, str]:
    """Return the conviction basis as named points: sector record, add-on route or region, top 3,
    concerns.

    The basis (factpack.conviction) always starts with the sector-record point and the add-on-route
    (sponsor) or region (strategic) point; the top-3 point and the concerns are added only when they
    hold. A concern is keyed by its text before the first colon with every number taken out (the
    deal-size concern by name), so the same concern under two targets matches even though its
    numbers (a cheque, an EV) depend on the target.
    """
    points = {
        "Sector record": basis[0],
        "Add-on route or region": basis[1],
        "Among the 3 highest-scored after the review": next(
            (x for x in basis if "highest-scored" in x), "neutral: not among the 3 highest-scored"
        ),
    }
    for x in basis[2:]:
        if x.startswith("against:"):
            points["Concern: " + _concern_key(x)] = x
    return points


def _concern_key(point: str) -> str:
    """Return a concern's name without its target-specific numbers, e.g. "deal size"."""
    if "median deal" in point:
        return "deal size"
    head = point.split(": ", 1)[1].split(":")[0]
    return re.sub(r"\$?\d[\d,.]*\s?[%xMB]?", "N", head).strip()


def shared_rows(a: Report, b: Report) -> list[dict]:
    """Return one row per buyer in both Tier 1 lists: priority and conviction under each, and what
    changed.

    "What changed" is each named conviction point whose direction (for, neutral, against) differs
    between the two targets, e.g. TPG: "Concern: deal size" is against for a $300M target only.
    """
    rows = []
    for buyer in [x for x in a.tier1 if x in b.tier1]:
        (level_a, basis_a), (level_b, basis_b) = (
            buyer_conviction(a.res, buyer),
            buyer_conviction(b.res, buyer),
        )
        pa, pb = conviction_points(basis_a), conviction_points(basis_b)
        none = "neutral: does not apply"
        differ = [
            dict(point=k, a=pa.get(k, none), b=pb.get(k, none))
            for k in dict.fromkeys([*pa, *pb])
            if pa.get(k, none).split(":")[0] != pb.get(k, none).split(":")[0]
        ]
        rows.append(
            dict(
                buyer=buyer,
                priority_a=a.tier1.index(buyer) + 1,
                priority_b=b.tier1.index(buyer) + 1,
                conviction_a=level_a,
                conviction_b=level_b,
                changed=level_a != level_b,
                points_that_differ=differ,
            )
        )
    return rows


def unique_rows(a: Report, b: Report) -> list[dict]:
    """Return a's Tier 1 buyers that are not in b's Tier 1, with type, conviction, why them, and
    where b put them."""
    calls = {r["Buyer"]: r for r in call_list(a)}
    where_b = {x: "Tier 2" for x in b.tier2} | {
        m.buyer: m.action for m in b.res.review.log if m.buyer not in b.tier2
    }
    return [
        dict(
            buyer=x,
            type=calls[x]["Type"],
            conviction=calls[x]["Conviction"],
            why_them=calls[x]["Why them"],
            in_other_list=where_b.get(x, "not listed"),
        )
        for x in a.tier1
        if x not in b.tier1
    ]


def comparison_pack(a: Report, b: Report) -> dict:
    """Return every fact the comparison note may use, with IDs (Q:...)."""

    def target_block(rep: Report, tag: str) -> dict:
        """Return one target's facts for the pack (ID Q:target_a or Q:target_b)."""
        v, lbo = rep.res.valuation, rep.res.sponsor_lbo
        return dict(
            id=f"Q:target_{tag}",
            name=_short(rep),
            line=rep.res.target.label(),
            comparable_deals=v["n_comps"],
            ev_ebitda_median=f"{v['ev_ebitda']['p50']}x",
            defensible_percentile=f"{v['defensible_percentile']}th",
            defensible_multiple=f"{v['ev_ebitda']['p' + str(v['defensible_percentile'])]}x",
            sponsor_irr_at_ask=f"{100 * lbo['irr_at_ask']:.1f}%",
            sponsor_target_return=f"{100 * lbo['target_irr']:.0f}%",
            sponsor_clears_target_return="yes" if lbo["irr_at_ask"] >= lbo["target_irr"] else "no",
            sponsor_max_price=money(
                max(lbo["max_price_same_exit"], lbo["max_price_platform_exit"])
            ),
            tier1_sponsors=sum(r["Type"] == "Financial sponsor" for r in call_list(rep)),
        )

    return dict(
        target_a=target_block(a, "a"),
        target_b=target_block(b, "b"),
        shared=dict(
            id="Q:shared",
            buyers=[
                dict(
                    buyer=r["buyer"],
                    conviction_a=r["conviction_a"],
                    conviction_b=r["conviction_b"],
                    points_that_differ=r["points_that_differ"],
                )
                for r in shared_rows(a, b)
            ],
        ),
        only_a=dict(
            id="Q:only_a",
            buyers=[
                dict(buyer=r["buyer"], type=r["type"], conviction=r["conviction"])
                for r in unique_rows(a, b)
            ],
        ),
        only_b=dict(
            id="Q:only_b",
            buyers=[
                dict(buyer=r["buyer"], type=r["type"], conviction=r["conviction"])
                for r in unique_rows(b, a)
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Note (LLM, validated; template on replay without a cached note)
# ---------------------------------------------------------------------------


def template_note(pack: dict) -> dict:
    """Return a data-built note: overlap, what each list leans on, and the shared buyers'
    conviction."""
    a, b, shared = pack["target_a"], pack["target_b"], pack["shared"]["buyers"]
    names = ", ".join(s["buyer"] for s in shared) or "no buyer"
    moves = [
        f"{s['buyer']} is {s['conviction_a']} for the first and {s['conviction_b']} for the second"
        for s in shared
        if s["conviction_a"] != s["conviction_b"]
    ]
    return dict(
        headline="The two targets draw largely different buyers; only a few sponsors appear on "
        "both lists",
        sentences=[
            f"Both lists include {names}; the other names are specific to each target.",
            f"The {a['name']} list leans on {a['tier1_sponsors']} sponsors, the {b['name']} list "
            f"on {b['tier1_sponsors']}.",
            (
                moves[0] + "."
                if moves
                else "Shared buyers keep the same conviction under both targets."
            ),
        ],
    )


def comparison_note(
    a: Report, b: Report, model: str = "writer", allow_paid: bool = False
) -> tuple[dict, bool]:
    """Return (note with headline and sentences, True if LLM-written); the data-built note
    otherwise."""
    pack = comparison_pack(a, b)
    user = PROMPTS["user"].format(comparison_pack=json.dumps(pack, indent=1))
    written = written_summary(
        PROMPTS,
        PROMPTS["system"],
        user,
        pack,
        model=model,
        approach="compare",
        purpose="comparison note",
        allow_paid=allow_paid,
    )
    return (written, True) if written else (template_note(pack), False)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _differ(rows: list[dict]) -> str:
    """Return the conviction points that differ, e.g. "Concern: deal size (A: against, median deal
    $88M ...; B: neutral)"."""
    return (
        "; ".join(f"{d['point']} (A: {d['a']}; B: {d['b']})" for d in rows)
        or "same points under both"
    )


def compare_markdown(
    a: Report, b: Report, note: dict, from_llm: bool, figure: str | None = None
) -> str:
    """Return the comparison page: note, buyer map, shared buyers with what changed, buyers unique
    to each."""
    na, nb = _short(a), _short(b)
    lines = [
        f"# {note['headline']}",
        f"*A: {a.res.target.label()}.*",
        f"*B: {b.res.target.label()}.*",
        "",
        *[f"- {s}" for s in note["sentences"]],
        "",
    ]
    if not from_llm:
        lines += ["*Template note: the LLM-written comparison is not in the cache.*", ""]
    if figure:
        lines += [f"![Both targets' Tier 1 on one buyer map]({figure})", ""]
    lines += [
        f"## Shared Tier 1 buyers ({len(shared_rows(a, b))})",
        f"| Buyer | Priority A / B | Conviction A ({na}) | Conviction B ({nb}) | "
        "Conviction points that differ |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['buyer']} | {r['priority_a']} / {r['priority_b']} | {r['conviction_a']} | "
        f"{r['conviction_b']} | "
        f"{_differ(r['points_that_differ'])} |"
        for r in shared_rows(a, b)
    ]
    for rep, other, label in ((a, b, f"A only ({na})"), (b, a, f"B only ({nb})")):
        rows = unique_rows(rep, other)
        lines += [
            "",
            f"## {label}: {len(rows)} buyers",
            "| Buyer | Type | Conviction | Why them | On the other list |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['buyer']} | {r['type']} | {r['conviction']} | {r['why_them']} | "
            f"{r['in_other_list']} |"
            for r in rows
        ]
    lines += [
        "",
        f"*{a.res.target.data_note()}. Conviction points come from each target's own fact packs; "
        "the note is checked "
        "so that every number in it appears in the comparison data.*",
    ]
    return "\n".join(lines)
