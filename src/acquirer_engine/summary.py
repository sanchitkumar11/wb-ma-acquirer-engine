"""The summary page: what a banker reads before the one-pagers, and the Markdown version of it.

Built as plain rows so the Markdown report and the Streamlit app show the same content. Order:
executive summary, Tier 1 call list, recommended approach, buyer map, Tier 2 in four groups, sponsor
scorecard, then "how we built the list" (funnel, rings, motive framework, review log, method note).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import pandas as pd

from acquirer_engine.factpack import MOTIVES, buyer_conviction, money, pct
from acquirer_engine.feedback import learned_rules, suggestions
from acquirer_engine.profiles import capital_fact, revenue_fact
from acquirer_engine.rings import fit_labels, model_key, model_sectors, model_words, ring
from acquirer_engine.wording import CSV_ONLY_NOTE, PLATFORM_MEANING, plain_terms

if TYPE_CHECKING:
    from acquirer_engine.report_types import Report

# experiment log Section 3.7: in sponsor-won test deals, where the winner ranked among the 15
# sponsors
SPONSOR_ORDERING_FINDING = (
    "the file barely separates sponsors: in back-tests the winning sponsor was in the top 3 "
    "sponsors 28% of the time, against 20% for a random order, so the order among the sponsors "
    "on this list is close to a tie"
)

# ---------------------------------------------------------------------------
# Tier 1 call list
# ---------------------------------------------------------------------------


def _billions(mm: float) -> str:
    """Return $M as a short $B figure, e.g. 11000 -> $11B, 75600 -> $75.6B."""
    b = mm / 1000
    return f"${b:.0f}B" if b >= 100 or b == int(b) else f"${b:.1f}B"


def can_pay(rep: Report, buyer: str) -> str:
    """Return the ability-to-pay verdict with the buyer's own evidence.

    Sponsors: the quick-LBO verdict (comfortable if the same-exit maximum clears the ask, add-on
    exit
    if only the platform exit does, else stretch), plus the sponsor's usual deal and its fund size
    or AUM when researched, e.g. "add-on exit; usual deal $77M, fund $11B". Strategics: the evidence
    for the verdict, e.g. "comfortable: closed $436M" or "stretch: biggest deal $37M".
    """
    ask, res = rep.res.target.enterprise_value_mm, rep.res
    own = res.deals[res.deals.acquirer == buyer]
    if res.ranked.set_index("acquirer").loc[buyer, "sponsor"]:
        lbo = res.sponsor_lbo
        verdict = (
            "comfortable"
            if lbo["max_price_same_exit"] >= ask
            else "add-on exit"
            if lbo["max_price_platform_exit"] >= ask
            else "stretch"
        )
        capital = capital_fact(res.profiles.get(buyer, []))
        extra = (
            f", {'fund' if capital.kind == 'fund_size' else 'AUM'} "
            f"{_billions(capital.extra['amount_usd_mm'])}"
            if capital
            else ""
        )
        return f"{verdict}; usual deal {money(own.deal_size_mm.median())}{extra}"
    closed = own[own.outcome == "Closed"]
    revenue = revenue_fact(res.profiles.get(buyer, []))
    if len(closed) and closed.deal_size_mm.max() >= ask:
        return f"comfortable: closed {money(closed.deal_size_mm.max())}"
    if any(code == "CAPACITY_CLEARED" for code, _ in res.review.flags.get(buyer, [])) and revenue:
        return f"comfortable: revenue {_billions(revenue.extra['amount_usd_mm'])}"
    return f"stretch: biggest deal {money(own.deal_size_mm.max())}"


def why_them(rep: Report, buyer: str, from_pages: bool = True) -> str:
    """Return the page's why_them line (if from_pages and cached), else a line built from the data.

    The executive summary uses from_pages=False, so it depends on engine outputs only and can be
    written at the same time as the pages.
    """
    page = (
        ({r["buyer"]: r["page"] for r in (rep.results or [])}.get(buyer) or {})
        if from_pages
        else {}
    )
    if page.get("why_them"):
        return plain_terms(page["why_them"])
    pack = rep.packs[buyer]
    if "sponsor_angles" in pack and pack["sponsor_angles"]["angles"]:
        return plain_terms(_sentence_case("; ".join(pack["sponsor_angles"]["angles"])))
    n = pack["track_record"]["same_sub_sector_deals"]
    return f"{n} deal{'s' if n != 1 else ''} in the target's own sub-sector"


def call_list(rep: Report, from_pages: bool = True) -> list[dict]:
    """Return Tier 1 in priority order (conviction, then model rank), numbered 1-10."""
    return [
        {
            "Priority": i,
            "Buyer": b,
            "Type": rep.packs[b]["buyer"]["type"],
            "Conviction": rep.packs[b]["conviction"]["level"],
            "Why them": why_them(rep, b, from_pages),
            "Can pay": can_pay(rep, b),
        }
        for i, b in enumerate(rep.tier1, 1)
    ]


def recommended_approach(rep: Report) -> list[str]:
    """Return the process recommendation (Jazz deck pattern): targeted Tier 1, Tier 2 in
    parallel."""
    conv = Counter(p["conviction"]["level"] for p in rep.packs.values())
    high = [b for b in rep.tier1 if rep.packs[b]["conviction"]["level"] == "High"]
    v = rep.res.valuation
    lead = (
        f"Lead with the {conv['High']} High-conviction buyers: {', '.join(high)}."
        if high
        else "No buyer reaches High conviction: lead with the top of the call list and test "
        "appetite early."
    )
    anchor = (
        f"Anchor on the 75th-percentile multiple ({v['ev_ebitda']['p75']}x), justified by the "
        "target's margin"
        if v["defensible_percentile"] == 75
        else f"Anchor on the median multiple ({v['ev_ebitda']['p50']}x): "
        f"{v['defensible_reason'].lower()}"
    )
    return [
        lead,
        f"Approach the rest of Tier 1 in parallel ({conv['Medium']} Medium, {conv['Low']} Low), "
        "and reach out to Tier 2 "
        "at the same time to keep tension in the process.",
        f"{anchor}; {_sponsor_pay_line(rep)}.",
    ]


def _sponsor_pay_line(rep: Report) -> str:
    """Return what the quick LBO says sponsors can pay against the ask, in one clause."""
    lbo, ask = rep.res.sponsor_lbo, rep.res.target.enterprise_value_mm
    irr, same, platform = (
        f"{100 * lbo['target_irr']:.0f}%",
        lbo["max_price_same_exit"],
        lbo["max_price_platform_exit"],
    )
    if same >= ask:
        return (
            f"sponsors clear a {irr} return at the ask on the same exit multiple "
            f"(up to {money(same)})"
        )
    if platform >= ask:
        return (
            f"sponsors clear a {irr} return at {money(platform)} only with a platform-style exit, "
            "so add-on buyers can pay most"
        )
    return (
        f"sponsors fall short of a {irr} return at the ask (maximum {money(max(same, platform))}), "
        "so strategics and add-on buyers set the price"
    )


# ---------------------------------------------------------------------------
# Tier 2 in four groups
# ---------------------------------------------------------------------------


def tier2_specs(target_sector: str) -> list[tuple[str, str]]:
    """Return the four Tier 2 groups as (name, what would promote a buyer), worded for the target.

    For a provider-services target: technology investors, sponsors without a distinct angle, health
    systems and provider strategics, adjacent-sector buyers.
    """
    adj, name = model_words(target_sector)
    strategics = (
        "Health systems and provider strategics" if adj == "provider" else f"{adj} strategics"
    )
    return [
        (
            f"Technology investors the file shows in {name}",
            f"Confirmation, on a first call, that the firm wants tech-enabled {name}, or a dated "
            f"statement of a {name.replace(' ', '-')} strategy",
        ),
        (
            "Sponsors without a distinct angle",
            f"Evidence that the sponsor {PLATFORM_MEANING}, or a recent deal in the target's "
            "region",
        ),
        (
            f"{_sentence_case(strategics)} with thin or large-only records",
            "A closed deal near the target's size, or a stated plan to expand into the target's "
            "region",
        ),
        ("Adjacent-sector buyers considered", f"A cited, dated strategy to own {name}"),
    ]


def _sentence_case(text: str) -> str:
    """Return text with its first letter capitalised and the rest untouched (str.capitalize
    lowercases)."""
    return text[:1].upper() + text[1:]


def _record_line(rep: Report, buyer: str) -> str:
    """Return "24 deals, median $77M" for a Tier 2 row."""
    own = rep.res.deals[rep.res.deals.acquirer == buyer]
    return (
        f"{len(own)} deal{'s' if len(own) != 1 else ''}, median {money(own.deal_size_mm.median())}"
    )


def _missing_point(rep: Report, buyer: str) -> str:
    """Return the specific points the buyer lacks, from its conviction basis (never a rank)."""
    _, basis = buyer_conviction(rep.res, buyer)
    gaps = [b.split(": ", 1)[1] for b in basis if b.startswith(("neutral:", "against:"))]
    return plain_terms(
        _sentence_case(
            "; ".join(gaps[:2]) or "Credible, but no stand-out angle against the Tier 1 sponsors"
        )
    )


def _tech_investor_line(rep: Report, buyer: str, r: pd.Series, note: str) -> str:
    """Return both facts and the action for a technology investor the file shows buying in the
    target's business model."""
    platform = " and a platform company in the target's sub-sector" if r.open_platform else ""
    says = note.replace("Mandate: ", "").strip()
    return (
        f"The file credits {buyer} with {int(r.n_model)} {model_words(rep.res.target.sector)[0]} "
        f"deals{platform}; the firm describes itself as {says}. "
        "Confirm appetite for tech-enabled services before outreach."
    )


def tier2_groups(rep: Report) -> list[dict]:
    """Return Tier 2 as four groups: name, what would promote a buyer, rows (Buyer, Type, Record,
    Why not Tier 1)."""
    ranked = rep.res.ranked.set_index("acquirer")
    moves = {m.buyer: m for m in rep.res.review.log}
    ask, sector = rep.res.target.enterprise_value_mm, rep.res.target.sector
    specs = tier2_specs(sector)
    groups = {name: [] for name, _ in specs}
    for b in rep.tier2:
        r, m = ranked.loc[b], moves.get(b)
        sponsor = bool(r.sponsor)
        own = rep.res.deals[rep.res.deals.acquirer == b]
        if m and m.code == "MANDATE_MISMATCH":
            name, why = specs[0][0], _tech_investor_line(rep, b, r, m.note)
        elif sponsor:
            name, why = specs[1][0], _missing_point(rep, b)
        elif r.closest_ring <= 1:
            name = specs[2][0]
            if (own.outcome == "Closed").sum() == 0:
                why = "No completed deal in the file"
            elif len(own) <= 2 and own.deal_size_mm.min() > 2 * ask:
                why = (
                    f"Only {len(own)} deal{'s' if len(own) > 1 else ''}, all above {money(2 * ask)}"
                )
            elif len(own) <= 2:
                why = f"Thin record: {len(own)} deal{'s' if len(own) > 1 else ''}"
            elif r.closest_ring == 1:
                why = f"Buys in {r.home_sector}, not the target's own sub-sector"
            else:
                why = _missing_point(rep, b)
        else:
            name, why = (
                specs[3][0],
                (
                    f"Home sub-sector is {r.home_sector} "
                    f"({fit_labels(sector)[int(r.closest_ring)]}); "
                    f"no cited {model_words(sector)[0]} strategy"
                ),
            )
        groups[name].append(
            {
                "Buyer": b,
                "Type": "Financial sponsor" if sponsor else "Strategic",
                "Record": _record_line(rep, b),
                "Why not Tier 1": why,
            }
        )
    return [
        dict(name=name, promote=promote, rows=groups[name])
        for name, promote in specs
        if groups[name]
    ]


# ---------------------------------------------------------------------------
# Sponsor scorecard, review log, sources
# ---------------------------------------------------------------------------


def sponsor_scorecard(rep: Report) -> list[dict]:
    """Return the sponsor matrix for listed sponsors, as a participation-style scorecard."""
    listed = set(rep.tier1) | set(rep.tier2)
    return [
        {
            "Sponsor": r.acquirer,
            "Tier": "1" if r.acquirer in rep.tier1 else "2",
            f"{_sentence_case(model_words(rep.res.target.sector)[0])} deals": r.provider_deals,
            "Already owns a platform company in the sub-sector": r.open_platform,
            "Median deal": f"${r.median_deal_mm}M",
            "Median EV/EBITDA paid": (
                "no closed deal" if pd.isna(r.median_ev_ebitda) else f"{r.median_ev_ebitda}x"
            ),
            "Minority deals": r.minority_deals,
            "Capital": r.capital,
            "Angles": plain_terms(", ".join(r.angles)) or "none",
        }
        for r in rep.res.review.matrix.itertuples()
        if r.acquirer in listed
    ]


def review_log_rows(rep: Report) -> list[dict]:
    """Return every move the review made, plus the model's top 25: model rank vs final placement."""
    final = {b: "Tier 1" for b in rep.tier1} | {b: "Tier 2" for b in rep.tier2}
    log = {m.buyer: m for m in rep.res.review.log}
    rows = []
    for r in rep.res.ranked.itertuples():
        m = log.get(r.acquirer)
        if r.model_rank > 25 and not m:
            continue
        placement = final.get(r.acquirer) or {
            "merged": "Merged",
            "removed": "Removed",
            "flagged": "Flagged",
        }.get(m.action if m else "", "Not listed")
        rows.append(
            {
                "Model rank": int(r.model_rank),
                "Buyer": r.acquirer,
                "Final tier": placement,
                "Reason code": m.code if m else "",
                "Note": m.note if m else "",
                "Source": m.source if m else "",
            }
        )
    return rows


def not_researched(rep: Report) -> list[str]:
    """Return listed buyers (Tier 1, then Tier 2) with no profile: shown as "not researched", never
    guessed."""
    return [b for b in rep.tier1 + rep.tier2 if not rep.res.profiles.get(b)]


def sources_rows(rep: Report) -> list[dict]:
    """Return every profile fact used, with its source and date."""
    return [
        dict(
            buyer=b,
            kind=f.kind,
            value=f.value,
            as_of=f.as_of.isoformat(),
            source=f.source,
            accession=f.extra.get("accession", ""),
        )
        for b, facts in sorted(rep.res.profiles.items())
        for f in facts
    ]


# ---------------------------------------------------------------------------
# How we built the list
# ---------------------------------------------------------------------------


def funnel(rep: Report) -> list[dict]:
    """Return the funnel stages: scored, reviewed, set aside (by reason), listed.

    Deal-team flags and accepted rules get their own stage only when they moved someone, counted
    wherever the buyer ranked (a flagged refill can sit below the top 25).
    """
    top = set(rep.res.ranked[rep.res.ranked.model_rank <= 25].acquirer)
    log = rep.res.review.log
    count = Counter(
        "off-mandate" if m.code == "MANDATE_MISMATCH" else m.action for m in log if m.buyer in top
    )
    flagged = sum(m.code == "USER_FLAG" for m in log)
    team = sum(m.code in ("USER_PROMOTE", "USER_DEMOTE") for m in log)
    by_rule = sum(m.code in ("CAPACITY_RULE", "GEOGRAPHY_RULE") for m in log)
    stages = [
        dict(stage="Buyers in the file, scored", n=len(rep.res.ranked)),
        dict(stage="Top 25 reviewed", n=len(top)),
        dict(stage="Removed (defunct or acquired)", n=count["removed"]),
        dict(stage="Merged into a listed buyer", n=count["merged"]),
        dict(stage="Off-mandate (to Tier 2)", n=count["off-mandate"]),
    ]
    stages += [dict(stage="Moved by an accepted rule (to Tier 2)", n=by_rule)] if by_rule else []
    stages += [dict(stage="Flagged by the deal team (off the list)", n=flagged)] if flagged else []
    stages += [dict(stage="Moved by the deal team (accepted suggestions)", n=team)] if team else []
    return [
        *stages,
        dict(stage="Tier 1 (one-pagers)", n=len(rep.tier1)),
        dict(stage="Tier 2 (incl. adjacent buyers added)", n=len(rep.tier2)),
    ]


def _has_motive(tags: list[str], motive: dict) -> bool:
    """Return True if a deal's rationale tags include any of the motive's tags."""
    return bool(set(tags) & set(motive["tags"]))


def motive_rows(rep: Report) -> list[dict]:
    """Return the motive framework with tag evidence: share of deals in the target's business model
    (provider deals for a provider target), and the Tier 1 buyers who show each motive most."""
    deals, adj = rep.res.deals, model_words(rep.res.target.sector)[0]
    prov_tags = deals[
        deals.sector.isin(model_sectors(rep.res.target.sector))
    ].strategic_rationale_tags.str.split("|")
    rows = []
    for m in MOTIVES:
        share = prov_tags.apply(_has_motive, motive=m).mean()
        own = {
            b: deals[deals.acquirer == b].strategic_rationale_tags.str.split("|") for b in rep.tier1
        }
        leaders = sorted(
            ((t.apply(_has_motive, motive=m).mean(), b) for b, t in own.items() if len(t) >= 3),
            reverse=True,
        )[:2]  # 3+ deals: one deal at 100% is not a pattern
        rows.append(
            {
                "Motive": m["motive"],
                "What does that mean?": m["meaning"],
                "Source": m["source"],
                "Evidence in the file (CSV tags)": (
                    f"{', '.join(m['tags'])}: {pct(share)} of {len(prov_tags)} {adj} deals"
                ),
                "Tier 1 buyers who show it most": ", ".join(f"{b} ({pct(s)})" for s, b in leaders),
                "_share": share,
            }
        )
    return rows


def motives_key(target_sector: str) -> str:
    """Return the summary-pack key for the common motives, e.g. "top_motives_in_provider_deals"."""
    return f"top_motives_in_{model_key(target_sector)}_deals"


def top_model_motives(rep: Report, k: int = 4) -> list[dict]:
    """Return the most common motives across deals in the target's business model, with evidence
    lines."""
    rows = sorted(motive_rows(rep), key=lambda r: -r["_share"])[:k]
    return [{"motive": r["Motive"], "evidence": r["Evidence in the file (CSV tags)"]} for r in rows]


def map_points(rep: Report) -> list[dict]:
    """Return buyer-map points for Tier 1 and Tier 2: fit evidence, deal size, conviction, type.

    x: precedents in the target's sub-sector or adjacent sub-sectors of its business model, each
    weighted
    by recency (a deal 3 years old counts half). y: the buyer's median deal / the ask (its usual
    cheque; about 1 means "writes cheques this size"). Not the quick LBO, which is the same for
    every
    sponsor, and not the biggest deal, which is several times the ask for most buyers.
    """
    year, ask = rep.res.target.scoring_quarter()[0], rep.res.target.enterprise_value_mm
    ranked = rep.res.ranked.set_index("acquirer")
    points = []
    for b in rep.tier1 + rep.tier2:
        own = rep.res.deals[rep.res.deals.acquirer == b]
        near = own[own.sector.map(lambda s: ring(s, rep.res.target.sector)) <= 1]
        points.append(
            dict(
                buyer=b,
                tier="Tier 1" if b in rep.tier1 else "Tier 2",
                deals=len(own),
                deal_mm=float(own.deal_size_mm.median()),
                priority=rep.tier1.index(b) + 1 if b in rep.tier1 else None,
                type="Financial sponsor" if ranked.loc[b, "sponsor"] else "Strategic",
                conviction=buyer_conviction(rep.res, b)[0],
                fit=round(float((0.5 ** ((year - near.deal_year) / 3)).sum()), 2),
                size=round(float(own.deal_size_mm.median() / ask), 2),
            )
        )
    return points


def target_price_ranges(rep: Report) -> list[dict]:
    """Return target-level price ranges ($M EV) for the summary's football field.

    Multiples become EV at the EBITDA implied by the ask at the comparable median multiple (the
    target's own EBITDA is not given), as on the buyer pages.
    """
    v, lbo, ask = rep.res.valuation, rep.res.sponsor_lbo, rep.res.target.enterprise_value_mm
    ebitda = ask / v["ev_ebitda"]["p50"]
    paid = []
    for b in rep.tier1:
        closed = rep.res.deals[(rep.res.deals.acquirer == b) & (rep.res.deals.outcome == "Closed")]
        if len(closed):
            paid.append(float(closed.ev_ebitda_multiple.median()))
    rows = [
        dict(
            label="Comparable closed deals (25th-75th percentile)",
            low=v["ev_ebitda"]["p25"] * ebitda,
            high=v["ev_ebitda"]["p75"] * ebitda,
            detail=f"{v['ev_ebitda']['p25']}x-{v['ev_ebitda']['p75']}x EV/EBITDA",
        )
    ]
    if paid:
        rows.append(
            dict(
                label="What Tier 1 buyers usually pay (range of their medians)",
                low=min(paid) * ebitda,
                high=max(paid) * ebitda,
                detail=f"{min(paid):.1f}x-{max(paid):.1f}x EV/EBITDA",
            )
        )
    return [
        *rows,
        dict(
            label=f"Sponsor quick LBO: maximum price at a {100 * lbo['target_irr']:.0f}% IRR",
            low=lbo["max_price_same_exit"],
            high=lbo["max_price_platform_exit"],
            detail=f"{money(lbo['max_price_same_exit'])} same exit; "
            f"{money(lbo['max_price_platform_exit'])} with an add-on exit"
            if lbo["max_price_platform_exit"] != lbo["max_price_same_exit"]
            else f"{money(lbo['max_price_same_exit'])}; the file shows no premium for a platform "
            "exit",
        ),
    ]


# ---------------------------------------------------------------------------
# Executive summary pack (the LLM writes from this; the template uses it on replay)
# ---------------------------------------------------------------------------


def summary_pack(rep: Report) -> dict:
    """Return every fact the executive summary may use, with IDs."""
    v, lbo, syn = rep.res.valuation, rep.res.sponsor_lbo, rep.res.strategic_synergies
    log, sector = rep.res.review.log, rep.res.target.sector
    tech = [m.buyer for m in log if m.code == "MANDATE_MISMATCH"]
    sponsors = [
        r["Buyer"] for r in call_list(rep, from_pages=False) if r["Type"] == "Financial sponsor"
    ]
    ordering = (
        {
            "sponsor_ordering": dict(
                id="E:sponsor_ordering",
                tier1_sponsors=len(sponsors),
                deals_in_target_sub_sector=int((rep.res.deals.sector == sector).sum()),
                finding=SPONSOR_ORDERING_FINDING,
            )
        }
        if len(sponsors) >= 2
        else {}
    )
    return dict(
        target=dict(id="E:target", line=rep.res.target.label()),
        tier1=dict(
            id="E:tier1",
            buyers=[
                {
                    "priority": r["Priority"],
                    "buyer": r["Buyer"],
                    "type": r["Type"],
                    "conviction": r["Conviction"],
                    "why_them": r["Why them"],
                    "can_pay": r["Can pay"],
                }
                for r in call_list(rep, from_pages=False)
            ],
        ),  # engine outputs only
        conviction=dict(
            id="E:conviction",
            counts=dict(Counter(p["conviction"]["level"] for p in rep.packs.values())),
        ),
        tier2=dict(
            id="E:tier2",
            groups={g["name"]: [r["Buyer"] for r in g["rows"]] for g in tier2_groups(rep)},
        ),
        valuation=dict(
            id="E:valuation",
            comparable_deals=v["n_comps"],
            ev_ebitda_p25=f"{v['ev_ebitda']['p25']}x",
            ev_ebitda_median=f"{v['ev_ebitda']['p50']}x",
            ev_ebitda_p75=f"{v['ev_ebitda']['p75']}x",
            defensible_percentile=f"{v['defensible_percentile']}th",
            defensible_multiple=f"{v['ev_ebitda']['p' + str(v['defensible_percentile'])]}x",
            why=v["defensible_reason"],
        ),
        ability_to_pay=dict(
            id="E:lbo",
            sponsor_irr_at_ask=f"{100 * lbo['irr_at_ask']:.1f}%",
            target_irr=f"{100 * lbo['target_irr']:.0f}%",
            sponsor_max_price_same_exit=money(lbo["max_price_same_exit"]),
            sponsor_max_price_platform_exit=money(lbo["max_price_platform_exit"]),
            strategic_synergy_headroom=f"{money(syn['synergy_value_low'])}-{money(syn['synergy_value_high'])}",
        ),
        review=dict(
            id="E:review",
            removed=[
                f"{m.buyer}: {m.note}" for m in log if m.action == "removed" and m.model_rank <= 25
            ],
            merged=[
                f"{m.buyer}: {m.note}" for m in log if m.action == "merged" and m.model_rank <= 25
            ],
            technology_investors_kept_in_tier_2=dict(
                buyers=tech,
                why=f"the file shows them buying {model_words(sector)[1]}, but they describe "
                "themselves as technology "
                "investors; confirm appetite for tech-enabled services before outreach"
                if tech
                else "",
            ),
        ),
        **{motives_key(sector): dict(id="E:motives", motives=top_model_motives(rep))},
        **ordering,
    )


def _and(names: list[str]) -> str:
    """Return 'A, B and C'."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _review_sentence(n_set_aside: int, tech: list[str]) -> str:
    """Return the template's sentence on what the review set aside and kept in Tier 2."""
    parts = [f"set aside {n_set_aside} defunct or merged names"] if n_set_aside else []
    parts += (
        [f"kept {_and(tech)} in Tier 2: technology investors to sound out first"] if tech else []
    )
    return (
        f"The review {' and '.join(parts)}."
        if parts
        else "The review removed no buyer from the top of the list."
    )


def template_summary(rep: Report) -> dict:
    """Return a data-built executive summary (headline, sentences) for replay without a cached LLM
    one."""
    pack = summary_pack(rep)
    high = [b["buyer"] for b in pack["tier1"]["buyers"] if b["conviction"] == "High"]
    v, rv = pack["valuation"], pack["review"]
    sector = rep.res.target.sector
    adj = model_words(sector)[0]
    top = pack[motives_key(sector)]["motives"][0]
    kinds = sorted({b["type"] for b in pack["tier1"]["buyers"]})
    strategics = "health systems" if adj == "provider" else "strategics"
    who = (
        f"{_sentence_case(adj)}-focused sponsors and {strategics}"
        if len(kinds) == 2
        else (
            f"{_sentence_case(adj)}-focused sponsors"
            if kinds == ["Financial sponsor"]
            else _sentence_case(strategics)
        )
    )
    first = high or [pack["tier1"]["buyers"][0]["buyer"]]
    multiple = v["ev_ebitda_p75"] if v["defensible_percentile"] == "75th" else v["ev_ebitda_median"]
    tech = rv["technology_investors_kept_in_tier_2"]["buyers"]
    price = (
        f"upper-quartile pricing at {v['defensible_multiple']} EV/EBITDA is defensible"
        if v["defensible_percentile"] == "75th"
        else f"the comparable median of {v['defensible_multiple']} EV/EBITDA is the defensible "
        "price"
    )
    return dict(
        headline=f"{who} form the buyer universe for this target; {price}",
        sentences=[
            f"Call {_and(first)} first: each pairs {adj} experience with a clear route to the deal"
            + (
                "; the order among the sponsors is close to a tie."
                if "sponsor_ordering" in pack
                else "."
            ),
            f"The {v['defensible_percentile']}-percentile multiple of {multiple} is defensible on "
            "margin; "
            f"{_sponsor_pay_line(rep)}.",
            _review_sentence(len(rv["removed"]) + len(rv["merged"]), tech),
            f"Across {adj} deals the most common motive is {top['motive'].lower()} "
            f"({top['evidence'].split(': ')[1]}).",
        ],
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(rows: list[dict], columns: list[str] | None = None) -> list[str]:
    """Return a Markdown table for rows (dicts), in the given column order (columns starting _
    hidden)."""
    if not rows:
        return ["*None.*"]
    cols = columns or [c for c in rows[0] if not c.startswith("_")]
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [
        "| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |" for r in rows
    ]


def headline_text(rep: Report) -> str:
    """Return the summary headline as shown: the cached LLM headline without the "own-sub-sector"
    label."""
    text = re.sub(r"\b[Oo]wn-sub-sector\s+", "", rep.exec_summary["headline"]).strip()
    return text[:1].upper() + text[1:]


def target_line(rep: Report) -> str:
    """Return 'Healthcare Services, $200M, ...; Deal data through 2024 Q4.' (the label already
    carries the date)."""
    return rep.res.target.label().replace("; deal data through", ". Deal data through") + "."


METHOD_NOTE = (
    "Method: every buyer in the file is scored by a model per buyer type (sponsor or strategic) on "
    "banker signals computed "
    "only from earlier deals, tested on 127 deals from 2022-24 it never saw (actual buyer in its "
    "top 20 for 91% of them). A "
    "banker review then removes defunct and merged buyers and moves off-mandate investors, using "
    "dated, sourced profiles (see "
    "Sources). Conviction comes from named data signals; the LLM writes the pages from fact packs, "
    "and every number is checked "
    "against them. The quick LBO is set by the target (its EBITDA, growth and today's debt terms), "
    "so it is the same for every "
    "sponsor; differences in sponsor capacity come from fund size and usual deal size. Glossary: "
    "an existing platform company "
    f"means the sponsor {PLATFORM_MEANING}."
)


def summary_markdown(rep: Report, figures: dict[str, str] | None = None) -> str:
    """Return page one: headline, executive summary, buyer map, call list, approach, football
    field."""
    figures, es = figures or {}, rep.exec_summary
    lines = [f"## {headline_text(rep)}", f"*{target_line(rep)}*", ""]
    if rep.res.review.csv_only:
        lines += [
            f"> {CSV_ONLY_NOTE}",
            "",
        ]
    lines += [*[f"- {x}" for x in es["sentences"]], ""]
    if "buyer_map" in figures:
        lines += [
            f"![Buyer map]({figures['buyer_map']})",
            "*Tier 1 numbered by priority; hollow markers are Tier 2. Vertical axis is a ratio "
            "scale: each step doubles.*",
            "",
        ]
    lines += [
        "## Tier 1: who to call, in order",
        *_table(call_list(rep)),
        "",
        "## Recommended approach",
        *[f"{i}. {x}" for i, x in enumerate(recommended_approach(rep), 1)],
        "",
    ]
    if "football_field" in figures:
        lines += [f"![Where the ask sits]({figures['football_field']})", ""]
    lines.append(f"*{target_line(rep)}*")
    return "\n".join(lines)


def bench_markdown(rep: Report) -> str:
    """Return the bench: Tier 2 in four groups, then the sponsor scorecard."""
    lines = [
        "# The bench: Tier 2 buyers, by what holds them back",
        f"*{rep.res.target.data_note()}.*",
        "",
    ]
    for g in tier2_groups(rep):
        lines += [
            f"### {g['name']}",
            f"*What would promote one: {g['promote']}.*",
            *_table(g["rows"]),
            "",
        ]
    lines += [
        "## Sponsor scorecard: the file barely separates sponsors, so compare them side by side",
        *_table(sponsor_scorecard(rep)),
    ]
    return "\n".join(lines)


def learning_markdown(rep: Report) -> list[str]:
    """Return the "learning from the deal team's flags" lines: suggested rules (with their history
    check) and accepted ones."""
    res = rep.res
    found = suggestions(res.target, res.user_flags, res.deals, res.profiles, res.review.flags)
    accepted = learned_rules(res.target.sector)
    if not (res.user_flags or found or accepted):
        return []
    flagged = (
        ", ".join(
            f"{b} ({line['reason']})"
            for b, line in res.user_flags.items()
            if line["action"] == "flag"
        )
        or "none"
    )
    lines = ["### Learning from the deal team's flags", f"Flagged: {flagged}.", ""]
    lines += [
        f"- **Suggested, not applied:** {s['label']} Check against history: {s['history']}. "
        f"Accept with `python -m acquirer_engine rules --accept {i}` or on the app's method page."
        for i, s in enumerate(found, 1)
    ]
    lines += [
        f"- **Accepted {r['accepted'][:10]}:** {r['rule']} ({r['history']})." for r in accepted
    ]
    return [*lines, ""]


def appendix_markdown(rep: Report, figures: dict[str, str] | None = None) -> str:
    """Return the appendix: how we built the list (stages, rings, motives, review log, method
    note)."""
    figures = figures or {}
    lines = [
        "# Appendix: how we built the list",
        f"*{rep.res.target.data_note()}.*",
        "",
        "### From the whole file to the call list",
    ]
    lines += (
        [f"![Stages]({figures['funnel']})"]
        if "funnel" in figures
        else _table(funnel(rep), ["stage", "n"])
    )
    lines.append("")
    if "rings" in figures:
        lines += [f"![Buyers by home sub-sector]({figures['rings']})", ""]
    lines += learning_markdown(rep)
    lines += [
        "### Buyer motives and the evidence for them",
        *_table(motive_rows(rep)),
        "",
        "### Review log: model rank vs final tier, with the reason for every move",
        *_table(review_log_rows(rep)),
        "",
        f"*{rep.res.target.data_note()}. {METHOD_NOTE}*",
    ]
    return "\n".join(lines)


def sources_markdown(rep: Report) -> str:
    """Return the sources page: every profile fact with its URL (or EDGAR accession) and date."""
    rows = [
        {
            "Buyer": r["buyer"],
            "Fact": r["kind"],
            "Value": r["value"],
            "As of": r["as_of"],
            "Source": r["source"] + (f" (accession {r['accession']})" if r["accession"] else ""),
        }
        for r in sources_rows(rep)
    ]
    missing = not_researched(rep)
    gap = (
        [
            f"**Not researched** (no sourced profile facts; deal-file evidence only): "
            f"{', '.join(missing)}.",
            "",
        ]
        if missing and not rep.res.review.csv_only
        else []
    )
    return "\n".join(
        [
            "# Sources",
            f"*{rep.res.target.data_note()}. Real-world facts below come from dated, sourced buyer "
            "profiles; "
            "every deal figure comes from the provided CSV.*",
            "",
            *gap,
            *_table(rows),
        ]
    )
