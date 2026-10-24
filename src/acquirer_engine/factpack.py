"""Fact packs: the only evidence the LLM may write from, one per buyer.

Code assembles every fact; the LLM only writes. Each fact has an ID (D: a deal in the CSV,
P: a profile fact, V: valuation, S: a model signal, R: a review note) so the one-pager can cite
it, and numbers are pre-formatted the way they should appear in the text, so the validator can
check that every number in a one-pager exists here.

Conviction is decided here, from named data signals, not by the LLM: that keeps it consistent,
explainable and varied across buyers. The LLM writes the 1-2 sentence reason.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from acquirer_engine.lbo import SYNERGY_RANGE_TEXT
from acquirer_engine.profiles import revenue_fact
from acquirer_engine.rings import adjacent_sub_sectors, fit_labels, model_key, model_words, ring
from acquirer_engine.valuation import buyer_precedents

if TYPE_CHECKING:
    import pandas as pd

    from acquirer_engine.engine import EngineResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIGNAL_NAMES = dict(
    activity="number of past deals",
    recency="deals in recent years",
    sector_share="share of its deals in the target's sub-sector",
    sector_rel="deals in this or neighbouring sub-sectors",
    geo_fit="has bought in this or a neighbouring region",
    own_fit_s="has bought from this kind of seller",
    type_fit_s="does control deals of private companies",
    margin_fit="target margin close to what it usually buys",
    growth_fit="target growth close to what it usually buys",
    close_rate_s="share of its deals that closed",
    bidder_exp="used to competitive auctions",
    precedent_sim="close precedents (sub-sector, size, region, margin)",
    not_bigger="deal is not bigger than any it has done",
    not_smaller="deal is not smaller than its usual range",
    open_platform="owns a recent platform in the sub-sector",
    open_platform_prov="owns a recent provider platform",
)
# Plain-English labels for the adjacency rings come from rings.fit_labels: the report is read by an
# MD,
# not the model's builder, and the labels follow the target's business model
SEVERE_FLAGS = {
    "NO_CLOSED_DEAL",
    "CAPACITY_STRETCH",
    "VERTICAL_MOVE",
    "CAPACITY_FUND",
    "STATED_NO_ACQUISITIONS",
}

# ---------------------------------------------------------------------------
# Formatting: one place, so text and pack always agree
# ---------------------------------------------------------------------------


def money(x: float) -> str:
    """Return $M with no decimals and thousands separators, e.g. $1,304M."""
    return f"${x:,.0f}M"


def multiple(x: float) -> str:
    """Return an EV multiple to one decimal, e.g. 12.8x."""
    return f"{x:.1f}x"


def pct(x: float) -> str:
    """Return a share (0-1) as a whole percentage, e.g. 0.6 -> 60%."""
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _precedents(res: EngineResult, buyer: str, merged_from: list[str]) -> list[dict]:
    """Return the buyer's closest earlier deals (and those of entities merged into it), cited by
    deal ID."""
    fit, rows = fit_labels(res.target.sector), []
    for name in [buyer, *merged_from]:
        for d in buyer_precedents(res.deals, name, res.target).itertuples():
            rows.append(
                dict(
                    id=f"D:{d.transaction_id}",
                    buyer_as=name,
                    target=d.target_company,
                    year=int(d.deal_year),
                    sub_sector=d.sector,
                    fit=fit[d.ring],
                    size=money(d.deal_size_mm),
                    deal_type=d.deal_type,
                    region=d.geography,
                    ev_ebitda=multiple(d.ev_ebitda_multiple),
                    ev_revenue=f"{d.ev_revenue_multiple:.2f}x",
                    outcome=d.outcome,
                )
            )
    order = {label: k for k, label in fit.items()}
    rows.sort(key=lambda r: (order[r["fit"]], -r["year"]))
    return rows[:6]


def adjacent_key(target_sector: str) -> str:
    """Return the track-record key for Ring 1 deals, e.g. "adjacent_provider_deals" for a provider
    target."""
    return f"adjacent_{model_key(target_sector)}_deals"


def _record(res: EngineResult, buyer: str, merged_from: list[str]) -> dict:
    """Return the buyer's track record in the file: counts, sizes, prices paid, regions.

    Key names follow the target's business model (adjacent_provider_deals for a provider target,
    adjacent_health_it_deals for a Health IT one), so the LLM never reads a provider label for a
    target that isn't one.
    """
    outside = model_words(res.target.sector)[1].lower().replace(" ", "_")
    own = res.deals[res.deals.acquirer.isin([buyer, *merged_from])]
    closed = own[own.outcome == "Closed"]
    in_ring = own.sector.map(lambda s: ring(s, res.target.sector))
    return dict(
        deals=len(own),
        closed_deals=len(closed),
        first_year=int(own.deal_year.min()),
        last_year=int(own.deal_year.max()),
        same_sub_sector_deals=int((in_ring == 0).sum()),
        **{
            adjacent_key(res.target.sector): int((in_ring == 1).sum()),
            f"deals_outside_{outside}": int((in_ring >= 2).sum()),
        },
        median_deal=money(own.deal_size_mm.median()),
        smallest_deal=money(own.deal_size_mm.min()),
        biggest_deal=money(own.deal_size_mm.max()),
        median_ev_ebitda_paid=multiple(closed.ev_ebitda_multiple.median())
        if len(closed)
        else "no closed deal",
        deals_in_target_region=int((own.geography == res.target.region).sum()),
        top_regions={k: int(v) for k, v in own.geography.value_counts().head(3).items()},
        deal_types={k: int(v) for k, v in own.deal_type.value_counts().head(4).items()},
        minority_deals=int((own.deal_type == "Minority Investment").sum()),
    )


def _derived(res: EngineResult, buyer: str, merged_from: list[str]) -> list[dict]:
    """Return counts, shares and medians a banker would quote, each with an ID.

    Pre-computed so the one-pager never needs arithmetic: the prompt forbids computing numbers, and
    the validator rejects any number not in the pack.
    """
    own = res.deals[res.deals.acquirer.isin([buyer, *merged_from])]
    t, year = res.target, res.target.scoring_quarter()[0]
    band = own.deal_size_mm.between(0.5 * t.enterprise_value_mm, 2 * t.enterprise_value_mm)
    control = own.deal_type != "Minority Investment"
    items = [
        (
            "K:closed_share",
            "deals that closed",
            f"{int((own.outcome == 'Closed').sum())} of {len(own)} "
            f"({pct((own.outcome == 'Closed').mean())})",
        ),
        (
            "K:by_region",
            "deals by region",
            {k: int(v) for k, v in own.geography.value_counts().items()},
        ),
        (
            "K:by_sub_sector",
            "deals by sub-sector",
            {k: int(v) for k, v in own.sector.value_counts().items()},
        ),
        (
            "K:size_band",
            f"deals in the comparable size band "
            f"({money(0.5 * t.enterprise_value_mm)}-{money(2 * t.enterprise_value_mm)})",
            f"{int(band.sum())} of {len(own)}",
        ),
        ("K:recent", f"deals since {year - 3}", int((own.deal_year >= year - 3).sum())),
        (
            "K:control_share",
            "control deals (not minority stakes)",
            f"{int(control.sum())} of {len(own)} ({pct(control.mean())})",
        ),
        (
            "K:median_ev_revenue",
            "median EV/Revenue of its closed deals",
            f"{own[own.outcome == 'Closed'].ev_revenue_multiple.median():.2f}x"
            if (own.outcome == "Closed").any()
            else "no closed deal",
        ),
        ("K:years_active", "years with at least one deal", int(own.deal_year.nunique())),
    ]
    return [dict(id=i, fact=label, value=value) for i, label, value in items]


def _signals(res: EngineResult, row: pd.DataFrame) -> list[dict]:
    """Return the three signals that lift this buyer most and the one that holds it back most.

    A signal with the same value for every buyer (e.g. the provider-platform signal for a Health IT
    target) moves no buyer against another, so it is never named.
    """
    contrib = res.model.contributions(row).iloc[0]
    varies = [
        name for name in contrib.index if name in res.ranked and res.ranked[name].nunique() > 1
    ]
    contrib = contrib[(contrib != 0) & contrib.index.isin(varies)]
    picks = list(contrib.sort_values(ascending=False).head(3).items())
    if (contrib < 0).any():
        picks.append((contrib.idxmin(), contrib.min()))
    return [
        dict(
            id=f"S:{name}",
            signal=SIGNAL_NAMES[name],
            strength=_strength(v),
            direction="for" if v > 0 else "against",
        )
        for name, v in picks
    ]


def _strength(log_odds: float) -> str:
    """Return how much a signal moves the buyer's likelihood, in words (an MD can't read odds
    ratios)."""
    size = abs(log_odds)
    if size >= 1.0:
        return "the strongest factor"
    return (
        "a strong factor"
        if size >= 0.3
        else ("a modest factor" if size >= 0.1 else "a slight factor")
    )


def conviction(
    survivor_rank: int | None,
    record: dict,
    flags: list[tuple[str, str]],
    is_sponsor: bool,
    open_platform: bool,
    ev: float,
    median_deal_mm: float,
    target_sector: str,
) -> tuple[str, list[str]]:
    """Return (High | Medium | Low, the named signals behind it).

    One point each for: a sector record (2+ closed deals, with 2+ in the target's sub-sector or 5+
    across it and the adjacent sub-sectors of the same business model); a sponsor's open platform
    in the sub-sector or a
    strategic's deal in the target's region; a top-3 model score among the buyers left after the
    review (removals, merges and mandate moves), which doesn't depend on the list's own order, so
    the list can be sorted by conviction without circularity. Minus one for each severe
    flag and for a sponsor whose median deal is under half the EV (a stretch). High at 2+,
    Low at 0 or below, else Medium. A buyer that states it makes no acquisitions is capped at Low.
    """
    points, basis = 0, []
    same, adjacent = record["same_sub_sector_deals"], record[adjacent_key(target_sector)]
    adj = model_words(target_sector)[0]
    nearby = (
        f" and {adjacent} in adjacent {adj} sub-sectors"
        if adjacent_sub_sectors(target_sector)
        else ""
    )
    if record["closed_deals"] >= 2 and (same >= 2 or same + adjacent >= 5):
        points += 1
        basis.append(f"for: {same} deal(s) in the target's own sub-sector{nearby}")
    else:
        basis.append(f"neutral: only {same} deal(s) in the target's own sub-sector{nearby}")
    if is_sponsor:
        if open_platform:
            points += 1
            basis.append("for: open platform in the sub-sector (add-on route)")
        else:
            basis.append("neutral: no open platform in the sub-sector")
    elif record["deals_in_target_region"] > 0:
        points += 1
        basis.append(f"for: {record['deals_in_target_region']} deal(s) in the target's region")
    else:
        basis.append("neutral: no deal in the target's region")
    if survivor_rank is not None and survivor_rank <= 3:
        points += 1
        basis.append(
            f"for: among the 3 highest-scored buyers left after the review (#{survivor_rank})"
        )
    for code, note in flags:
        if code in SEVERE_FLAGS:
            points -= 1
            basis.append(f"against: {note}")
    if is_sponsor and median_deal_mm < 0.5 * ev:
        points -= 1
        basis.append(
            f"against: median deal {money(median_deal_mm)} is under half the {money(ev)} EV"
        )
    level = "High" if points >= 2 else ("Low" if points <= 0 else "Medium")
    if any(
        code == "STATED_NO_ACQUISITIONS" for code, _ in flags
    ):  # its own policy outweighs the file's deal record
        level = "Low"
        basis.append(
            "capped at Low: the buyer states it does not make acquisitions, so its record in the "
            "file is not evidence it would buy"
        )
    return level, basis


def survivor_ranks(res: EngineResult) -> dict[str, int]:
    """Return buyer -> rank by model score among buyers the review neither removed, merged nor
    moved."""
    # deal-team moves (USER_*) never change other buyers' conviction
    moved = {
        m.buyer
        for m in res.review.log
        if m.action in ("removed", "merged", "to Tier 2") and not m.code.startswith("USER_")
    }
    survivors = [b for b in res.ranked.acquirer if b not in moved]
    return {b: i for i, b in enumerate(survivors, 1)}


def buyer_conviction(res: EngineResult, buyer: str) -> tuple[str, list[str]]:
    """Return (level, basis) for any buyer on the list, Tier 1 or Tier 2."""
    r = res.ranked.set_index("acquirer").loc[buyer]
    merged_from = _merged_from(res, buyer)
    own = res.deals[res.deals.acquirer.isin([buyer, *merged_from])]
    return conviction(
        survivor_ranks(res).get(buyer),
        _record(res, buyer, merged_from),
        res.review.flags.get(buyer, []),
        bool(r.sponsor),
        bool(r.open_platform),
        res.target.enterprise_value_mm,
        float(own.deal_size_mm.median()),
        res.target.sector,
    )


def _merged_from(res: EngineResult, buyer: str) -> list[str]:
    """Return the buyers the review merged into this one (e.g. Atrium Health into Advocate
    Health)."""
    return [
        m.buyer for m in res.review.log if m.code == "ENTITY_DUPLICATE" and m.merged_into == buyer
    ]


# ---------------------------------------------------------------------------
# Motives: why this buyer has bought before, from its own deals' rationale tags
# ---------------------------------------------------------------------------

# Buyer motives from the M&A guide (economies of scale and cost synergies, revenue synergies through
# customers and market share, geographic expansion, vertical integration, diversification) and the
# LBO guide (platform build and add-ons; growth in fragmented markets), each evidenced by CSV tags.
# "Carve-out Complexity" describes the deal, not a motive, so it is left out.
MOTIVES = [
    dict(
        motive="Cost synergies and scale",
        source="M&A guide: consolidation / economies of scale",
        meaning="Combine with similar businesses to cut costs and buy better",
        tags=["Cost Synergies", "Scale", "Margin Improvement"],
    ),
    dict(
        motive="Revenue synergies",
        source="M&A guide: acquire customers or distribution; gain market share",
        meaning="Sell more to the combined customer base, or take share in a fragmented market",
        tags=["Cross-sell", "Market Share"],
    ),
    dict(
        motive="Geographic expansion",
        source="M&A guide: geographic expansion",
        meaning="Enter or deepen a region",
        tags=["Geographic Expansion"],
    ),
    dict(
        motive="Vertical integration",
        source="M&A guide: synergies along the value chain",
        meaning="Own the next step in the chain (e.g. a payer buying care delivery)",
        tags=["Vertical Integration"],
    ),
    dict(
        motive="Diversification",
        source="M&A guide: product expansion or diversification; key technologies",
        meaning="Add a new service line, capability or growth market",
        tags=["Diversification", "Technology Capability"],
    ),
    dict(
        motive="Platform build and add-ons",
        source="LBO guide: add-on acquisitions",
        meaning="A sponsor buys a platform, then bolts on smaller deals",
        tags=["Platform Build"],
    ),
    dict(
        motive="Growth",
        source="LBO guide: fast-growing, fragmented markets",
        meaning="Buy into above-market growth",
        tags=["High Growth"],
    ),
]
TAG_TO_MOTIVE = {tag: m["motive"] for m in MOTIVES for tag in m["tags"]}


def _motives(res: EngineResult, buyer: str, merged_from: list[str]) -> list[dict]:
    """Return the buyer's own past deals' rationale tags, most frequent first, each with an ID.

    As-of: only deals before the target's scoring quarter. The target's own tags are never used
    (the live target has none, and in a back-test they would leak the answer).
    """
    _, q = res.target.scoring_quarter()
    own = res.deals[res.deals.acquirer.isin([buyer, *merged_from]) & (res.deals.q < q)]
    counts = own.strategic_rationale_tags.str.split("|").explode().value_counts()
    # how common each tag is across every deal in the file: a tag on 90% of deals says little
    base = (
        res.deals[res.deals.q < q].strategic_rationale_tags.str.split("|").explode().value_counts()
        / (res.deals.q < q).sum()
    )
    return [
        dict(
            id=f"M:{tag.lower().replace(' ', '_').replace('-', '_')}",
            tag=tag,
            motive=TAG_TO_MOTIVE[tag],
            deals=f"{int(n)} of {len(own)}",
            share_of_all_deals_in_file=pct(base[tag]),
        )
        for tag, n in counts.items()
        if tag in TAG_TO_MOTIVE
    ]


# ---------------------------------------------------------------------------
# Fact pack
# ---------------------------------------------------------------------------


def build_fact_pack(res: EngineResult, buyer: str, tier1: list[str] | None = None) -> dict:
    """Return the fact pack for one Tier 1 buyer (a JSON-ready dict).

    `tier1` is the final list when an LLM approach (A2, A3) changed it; default: the rules' Tier 1.
    """
    tier1 = tier1 or res.review.tier1
    rv, t, v = res.review, res.target, res.valuation
    row = res.ranked[res.ranked.acquirer == buyer]
    r = row.iloc[0]
    is_sponsor = bool(r.sponsor)
    merged_from = _merged_from(res, buyer)
    record = _record(res, buyer, merged_from)
    flags = rv.flags.get(buyer, [])
    position = (
        tier1.index(buyer) + 1
    )  # priority order: conviction, then model rank (report.priority_order)
    level, basis = buyer_conviction(res, buyer)
    pack = dict(
        buyer=dict(
            id="B:buyer",
            name=buyer,
            type="Financial sponsor" if is_sponsor else "Strategic",
            merged_into_this_buyer=merged_from,
            closest_fit=fit_labels(t.sector)[int(r.closest_ring)],
            home_sub_sector=r.home_sector,
        ),
        target=dict(
            id="X:target",
            sector=t.sector,
            ev=money(t.enterprise_value_mm),
            region=t.region,
            ownership=t.ownership,
            ebitda_margin=f"{t.ebitda_margin_pct}%",
            revenue_growth=f"{t.revenue_growth_pct}%",
            data_through=f"{t.as_of[:4]} {t.as_of[4:]}",
        ),
        ranking=dict(
            id="R:rank",
            list="priority list of 10",
            position_on_list=position,
            overall_rank=int(r.model_rank),
            likelihood=f"{100 * r.share:.1f}%",
            buyers_scored=len(res.ranked),
        ),
        signals=_signals(res, row),
        track_record=dict(id="T:record", **record),
        derived=_derived(res, buyer, merged_from),
        motives=_motives(res, buyer, merged_from),
        precedents=_precedents(res, buyer, merged_from),
        valuation=dict(
            id="V:comps",
            comparable_deals=v["n_comps"],
            same_sub_sector_comparable_deals=v["n_core"],
            size_band=f"{money(v['size_band_mm'][0])}-{money(v['size_band_mm'][1])}",
            ev_ebitda_p25=f"{v['ev_ebitda']['p25']}x",
            ev_ebitda_median=f"{v['ev_ebitda']['p50']}x",
            ev_ebitda_p75=f"{v['ev_ebitda']['p75']}x",
            ev_revenue_median=f"{v['ev_revenue']['p50']:.2f}x",
            ebitda_needed_at_median=f"${v['ebitda_at']['p50']}M",
            ebitda_needed_at_p75=f"${v['ebitda_at']['p75']}M",
            target_margin_above_share_of_comps=f"{v['margin_percentile']}%",
            comps_margin_median=f"{v['comps_margin_median']}%",
            defensible_percentile=f"{v['defensible_percentile']}th",
            why=v["defensible_reason"],
            buyer_median_ev_ebitda_paid=record["median_ev_ebitda_paid"],
        ),
        ability_to_pay=_ability(res, buyer, is_sponsor),
        review=dict(
            flags=[
                dict(id=f"R:{code}", note=note) for code, note in flags
            ],  # no bare codes: plain notes only
            moves=[dict(id=f"R:{m.code}", note=m.note) for m in rv.log if m.buyer == buyer],
        ),
        profile_facts=[
            dict(
                id=f"P:{i + 1}",
                kind=f.kind,
                value=f.value,
                as_of=f.as_of.isoformat(),
                source=f.source,
            )
            for i, f in enumerate(res.profiles.get(buyer, []))
        ],
        profile_not_researched=not res.profiles.get(buyer),
        conviction=dict(id="C:conviction", level=level, basis=basis),
    )
    peers = closest_peers(res, buyer, tier1)
    if peers:
        pack["closest_peers"] = dict(
            id="N:closest_peers",
            note="strategics on the priority list from the same home sub-sector, "
            "with their stated strategy: say how this buyer differs from them",
            peers=peers,
        )
    if is_sponsor:
        m = rv.matrix.set_index("acquirer").loc[buyer]
        pack["sponsor_angles"] = dict(
            id="A:angles",
            angles=m.angles,
            open_platform=m.open_platform,
            minority_deals=int(m.minority_deals),
            **{f"{model_key(t.sector)}_deals": m.provider_deals},
        )  # provider_deals for a provider target
    return pack


def closest_peers(res: EngineResult, buyer: str, tier1: list[str]) -> list[dict]:
    """Return the other Tier 1 strategics with the same home sub-sector and a sourced stated
    strategy.

    Gives a writer the one thing the shared peer lines can't: how a look-alike buyer says it grows
    (e.g. Privia buys provider groups; Aledade partners with practices rather than owning them), so
    two similar strategics get different theses. Empty for sponsors (they have sponsor angles) and
    for strategics whose look-alikes have no stated strategy in their profiles.
    """
    ranked = res.ranked.set_index("acquirer")
    if ranked.loc[buyer, "sponsor"]:
        return []
    out = []
    for other in tier1:
        if (
            other == buyer
            or ranked.loc[other, "sponsor"]
            or ranked.loc[other, "home_sector"] != ranked.loc[buyer, "home_sector"]
        ):
            continue
        stated = [
            f
            for f in res.profiles.get(other, [])
            if f.kind in ("mandate", "acquisition_8k", "divestiture")
        ]
        if stated:
            own = res.deals[res.deals.acquirer == other]
            out.append(
                dict(
                    buyer=other,
                    stated=[f"{f.value} (as of {f.as_of.isoformat()})" for f in stated],
                    same_sub_sector_deals=int((own.sector == res.target.sector).sum()),
                    median_deal=money(own.deal_size_mm.median()),
                )
            )
    return out


def _ability(res: EngineResult, buyer: str, is_sponsor: bool) -> dict:
    """Return the ability-to-pay facts: the quick LBO for sponsors, synergy headroom for
    strategics."""
    if is_sponsor:
        lbo = res.sponsor_lbo
        return dict(
            id="V:lbo",
            leverage=f"{lbo['leverage']}x",
            interest_rate=f"{100 * lbo['interest_rate']:.1f}%",
            equity_cheque=money(lbo["equity_cheque"]),
            irr_at_ask=f"{100 * lbo['irr_at_ask']:.1f}%",
            target_irr=f"{100 * lbo['target_irr']:.0f}%",
            max_price_same_exit=money(lbo["max_price_same_exit"]),
            max_price_platform_exit=money(lbo["max_price_platform_exit"]),
            platform_exit_multiple=multiple(lbo["platform_exit_multiple"]),
        )
    syn = res.strategic_synergies
    revenue = revenue_fact(res.profiles.get(buyer, []))
    return dict(
        id="V:synergy",
        synergy_headroom=f"{money(syn['synergy_value_low'])}-{money(syn['synergy_value_high'])}",
        basis=f"cost synergies of {SYNERGY_RANGE_TEXT}, capitalised at the median multiple, half "
        "realised",
        buyer_revenue=revenue.value if revenue else "not researched",
    )


def pack_hash(pack: dict) -> str:
    """Return a stable SHA-256 of the pack, the cache key's evidence half."""
    return hashlib.sha256(json.dumps(pack, sort_keys=True, default=str).encode()).hexdigest()
