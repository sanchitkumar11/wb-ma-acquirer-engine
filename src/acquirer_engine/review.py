"""Banker review: turn the model's ranking into a list a banker would sign, logging every move.

The model sees only the CSV. The review adds what a banker knows: real-world status, mergers and
mandates (from the dated buyer profiles), capacity, how far the buyer sits from the target's
sub-sector (rings), and list-level checks. Rules never re-rank by judgment; they remove, merge or
demote with a reason code, and fill the tiers in model order. Rings never exclude a buyer: an
outer-ring strategic moves to Tier 2, where the banker still sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from acquirer_engine.feedback import flag_note, rule_applies
from acquirer_engine.profiles import Fact, capital_fact, profile_notes, revenue_fact
from acquirer_engine.rings import fit_labels, model_words, ring
from acquirer_engine.thresholds import (
    FUND_CONCENTRATION,
    OBVIOUS_YEARS,
    REVENUE_CLEARS,
    STRETCH_SHARE,
    TIER1_SIZE,
    TIER2_SIZE,
)

if TYPE_CHECKING:
    from acquirer_engine.config import Target

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REMOVE_CODES = {"STATUS_DEFUNCT", "STATUS_ACQUIRED", "ENTITY_DUPLICATE"}
LEARNED_CODE = {
    "mandate_focus": "MANDATE_MISMATCH",
    "capacity_flag": "CAPACITY_RULE",
    "needs_regional_deal": "GEOGRAPHY_RULE",
}

# ---------------------------------------------------------------------------
# Sponsor differentiation
# ---------------------------------------------------------------------------


def sponsor_angles(
    history: pd.DataFrame, buyer: str, target: Target, signals: pd.Series
) -> list[str]:
    """Return what sets this sponsor apart for this target, from its earlier deals only.

    Angles: add-on to an open platform in the sub-sector; specialist in the target's business model
    (60%+ of its deals in the target's sub-sector or Ring 1, e.g. "provider specialist"); pays up
    (median EV/EBITDA of its closed deals at or
    above the market's 75th percentile); minority option (3+ minority deals); size fit (median
    deal 0.5-2x the EV) or capacity (median deal above 2x).
    """
    own = history[history.acquirer == buyer]
    closed = history[history.outcome == "Closed"]
    angles = []
    if signals.open_platform:
        angles.append("add-on to an open platform in the sub-sector")
    if (own.sector.map(lambda s: ring(s, target.sector)) <= 1).mean() >= 0.6:
        angles.append(f"{model_words(target.sector)[0]} specialist")
    own_closed = own[own.outcome == "Closed"]
    if len(
        own_closed
    ) and own_closed.ev_ebitda_multiple.median() >= closed.ev_ebitda_multiple.quantile(0.75):
        angles.append("pays up")
    if (own.deal_type == "Minority Investment").sum() >= 3:
        angles.append("minority / partial-liquidity option")
    ratio = own.deal_size_mm.median() / target.enterprise_value_mm
    if 0.5 <= ratio <= 2:
        angles.append("size fit")
    elif ratio > 2:
        angles.append("capacity for larger deals")
    return angles


def sponsor_matrix(
    ranked: pd.DataFrame, history: pd.DataFrame, target: Target, profiles: dict[str, list[Fact]]
) -> pd.DataFrame:
    """Return one row per sponsor: the differentiators a banker compares side by side.

    Columns: acquirer, model_rank, provider_deals (deals in the target's business model, n of total;
    the key keeps its name so provider-target fact packs are unchanged), open_platform,
    median_deal_mm,
    median_ev_ebitda (closed deals), minority_deals, capital (latest fund size or AUM from the
    profile, else "not researched"), angles.
    """
    rows = []
    for r in ranked[ranked.sponsor == 1].itertuples():
        own = history[history.acquirer == r.acquirer]
        closed = own[own.outcome == "Closed"]
        capital = capital_fact(profiles.get(r.acquirer, []))
        rows.append(
            dict(
                acquirer=r.acquirer,
                model_rank=r.model_rank,
                provider_deals=f"{int(r.n_model)} of {int(r.n)}",
                open_platform="yes" if r.open_platform else "no",
                median_deal_mm=round(own.deal_size_mm.median()),
                median_ev_ebitda=round(closed.ev_ebitda_multiple.median(), 1)
                if len(closed)
                else None,
                minority_deals=int((own.deal_type == "Minority Investment").sum()),
                capital=capital.value if capital else "not researched",
                angles=sponsor_angles(history, r.acquirer, target, pd.Series(r._asdict())),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


@dataclass
class Move:
    """One change the review made to the model's list, as shown in the review log."""

    buyer: str
    model_rank: int
    action: str
    """removed | merged | flagged | to Tier 2 | to Tier 1 | added to Tier 2"""
    code: str
    """Reason code, e.g. STATUS_DEFUNCT, MANDATE_MISMATCH, USER_FLAG."""
    note: str
    """Plain-English reason shown to the banker (display text: never parse it)."""
    source: str = ""
    """Citation for profile-based moves; "deal file" for in-file rules."""
    merged_into: str = ""
    """For a merge: the listed buyer this one is now part of (so callers need not parse the
    note)."""


@dataclass
class Review:
    """The review's result: the two lists, every move it made, and the flags behind conviction."""

    tier1: list[str]
    """Priority list, in model order (report.priority_order sorts it by conviction)."""
    tier2: list[str]
    """The bench: next buyers in model order, plus coverage additions."""
    log: list[Move]
    """Every move, in the order made."""
    flags: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    """Buyer -> (code, note) risk flags; they feed risks and conviction, never the tiers."""
    matrix: pd.DataFrame | None = None
    """Sponsor differentiation matrix (one row per sponsor)."""
    csv_only: bool = False
    """True when profiles were ignored (status, merger and mandate rules off)."""


def _flags(
    r: Any, target: Target, profiles: dict[str, list[Fact]], sponsor_cheque_mm: float | None
) -> list[tuple[str, str]]:
    """Return risk flags for one buyer: they feed Risk Flags and Conviction, not the tiers."""
    out = []
    if r.sponsor == 1 and sponsor_cheque_mm:
        capital = capital_fact(profiles.get(r.acquirer, []))
        if capital is None:
            out.append(
                ("FUND_SIZE_UNKNOWN", "Fund size not researched: equity capacity unconfirmed")
            )
        elif sponsor_cheque_mm > FUND_CONCENTRATION * capital.extra["amount_usd_mm"]:
            out.append(
                (
                    "CAPACITY_FUND",
                    f"Equity cheque of ${sponsor_cheque_mm:.0f}M is over {FUND_CONCENTRATION:.0%} "
                    f"of {capital.value}",
                )
            )
    stance = next(
        (
            f
            for f in profiles.get(r.acquirer, [])
            if f.kind == "mandate" and f.extra.get("stance") == "no_acquisitions"
        ),
        None,
    )
    if (
        stance
    ):  # a flag, not a gate: the file still credits the buyer with deals, so it stays visible
        out.append(
            ("STATED_NO_ACQUISITIONS", f"States it does not make acquisitions: {stance.value}")
        )
    if r.n_closed == 0:
        out.append(("NO_CLOSED_DEAL", "No completed deal in the file"))
    if r.sponsor == 0 and r.max_size < STRETCH_SHARE * target.enterprise_value_mm:
        revenue = revenue_fact(profiles.get(r.acquirer, []))
        if (
            revenue
            and revenue.extra["amount_usd_mm"] >= REVENUE_CLEARS * target.enterprise_value_mm
        ):
            out.append(
                (
                    "CAPACITY_CLEARED",
                    f"Biggest deal in the file ${r.max_size:.0f}M, but {revenue.value}",
                )
            )
        else:
            out.append(
                (
                    "CAPACITY_STRETCH",
                    f"Biggest deal in the file ${r.max_size:.0f}M: "
                    f"${target.enterprise_value_mm:.0f}M would be a stretch",
                )
            )
    if r.sponsor == 0 and r.closest_ring == 2:
        out.append(
            (
                "VERTICAL_MOVE",
                f"Vertical move from {r.home_sector}: needs a stated strategy; integration risk",
            )
        )
    return out


def review(
    ranked: pd.DataFrame,
    history: pd.DataFrame,
    target: Target,
    profiles: dict[str, list[Fact]],
    csv_only: bool = False,
    angle_rule: bool = False,
    sponsor_cheque_mm: float | None = None,
    user_flags: dict[str, dict] | None = None,
    learned: list[dict] | None = None,
) -> Review:
    """Return Tier 1, Tier 2, the review log, risk flags and the sponsor matrix.

    `ranked` is cli.rank_buyers output (best first); `history` is every deal known at the as-of
    date; `profiles` should already be filtered with profiles.facts_as_of in a back-test. With
    csv_only, profiles are ignored, so status, merger and mandate rules are off. sponsor_cheque_mm
    (the quick LBO's equity cheque) turns on the fund-size capacity flag.

    user_flags (feedback.active_flags) takes each flagged buyer off the list with reason code
    USER_FLAG, so the tiers refill from the next eligible buyer; learned (feedback.learned_rules)
    are rules the deal team accepted from its flags, each moving matching buyers to Tier 2.

    angle_rule (off by default) makes "each Tier 1 sponsor needs a distinct angle" a gate. Tested
    against history (experiments/rules_backtest.py) it pushed the real buyer out of Tier 1 in 35%
    of test deals, so it stays a writing requirement: the matrix shows each sponsor's angles and
    the one-pager must state one.
    """
    profiles = {} if csv_only else profiles
    user_flags, learned = user_flags or {}, learned or []
    buyers = set(ranked.acquirer)
    log: list[Move] = []
    flags: dict[str, list[tuple[str, str]]] = {}
    eligible, demoted = [], []

    # 1-4. buyer-level rules: status and entity, mandate, capacity, rings
    for r in ranked.itertuples():
        notes = profile_notes(r.acquirer, profiles, target.sector, buyers)
        removal = next(((c, n, f) for c, n, f in notes if c in REMOVE_CODES), None)
        if removal:
            code, note, fact = removal
            log.append(
                Move(
                    r.acquirer,
                    r.model_rank,
                    "merged" if code == "ENTITY_DUPLICATE" else "removed",
                    code,
                    note,
                    fact.cite(),
                    merged_into=fact.extra["merged_into"] if code == "ENTITY_DUPLICATE" else "",
                )
            )
            continue
        # computed before the deal-team flag, so a flagged buyer still carries its capacity flags
        # (the "can't pay" rule suggestion looks for a flag the flagged buyers share)
        flags[r.acquirer] = _flags(r, target, profiles, sponsor_cheque_mm)
        if user_flags.get(r.acquirer, {}).get("action") == "flag":
            line = user_flags[r.acquirer]
            log.append(
                Move(
                    r.acquirer,
                    r.model_rank,
                    "flagged",
                    "USER_FLAG",
                    flag_note(line),
                    f"deal-team flag, {line['timestamp']}",
                )
            )
            continue
        mandate = next(((c, n, f) for c, n, f in notes if c == "MANDATE_MISMATCH"), None)
        own = history[history.acquirer == r.acquirer]
        rule = next(
            (
                x
                for x in learned
                if rule_applies(x, r.acquirer, profiles, flags[r.acquirer], own, target)
            ),
            None,
        )
        if user_flags.get(r.acquirer, {}).get("action") == "demote":  # accepted reviewer suggestion
            line = user_flags[r.acquirer]
            log.append(
                Move(
                    r.acquirer,
                    r.model_rank,
                    "to Tier 2",
                    "USER_DEMOTE",
                    flag_note(line),
                    f"deal-team move, {line['timestamp']}",
                )
            )
            demoted.append(r.acquirer)
        elif mandate:
            log.append(Move(r.acquirer, r.model_rank, "to Tier 2", *mandate[:2], mandate[2].cite()))
            demoted.append(r.acquirer)
        elif rule:
            log.append(
                Move(
                    r.acquirer,
                    r.model_rank,
                    "to Tier 2",
                    LEARNED_CODE[rule["kind"]],
                    f"Rule accepted from deal-team flags on {rule['accepted'][:10]}: "
                    f"{rule['rule']}",
                    "configs/learned_rules.yaml",
                )
            )
            demoted.append(r.acquirer)
        elif r.sponsor == 0 and r.closest_ring == 3:
            if r.model_rank <= TIER1_SIZE + TIER2_SIZE:  # log only moves a banker would notice
                log.append(
                    Move(
                        r.acquirer,
                        r.model_rank,
                        "to Tier 2",
                        "DISTANT_SECTOR",
                        f"Buyer from a distant sector ({r.home_sector}) with no cited "
                        f"{model_words(target.sector)[1]} strategy",
                        "deal file",
                    )
                )
            demoted.append(r.acquirer)
        else:
            eligible.append(r.acquirer)

    # 5. sponsor differentiation (a gate only if angle_rule; see the docstring)
    matrix = sponsor_matrix(ranked, history, target, profiles)
    angles = dict(zip(matrix.acquirer, matrix.angles, strict=True))
    rank_of = dict(zip(ranked.acquirer, ranked.model_rank, strict=True))
    type_of = dict(zip(ranked.acquirer, ranked.sponsor, strict=True))
    tier1, taken = [], []
    for buyer in eligible:
        if len(tier1) == TIER1_SIZE:
            break
        if type_of[buyer] == 1 and angle_rule:
            angle = sorted(angles[buyer])
            if not angle or angle in taken:
                log.append(
                    Move(
                        buyer,
                        rank_of[buyer],
                        "to Tier 2",
                        "SPONSOR_NO_ANGLE",
                        "No angle that a higher-ranked Tier 1 sponsor doesn't already offer: "
                        + (", ".join(angle) or "none"),
                        "deal file",
                    )
                )
                demoted.append(buyer)
                continue
            taken.append(angle)
        tier1.append(buyer)

    # 5b. accepted reviewer promotions: in, displacing the lowest-ranked buyer not promoted
    promoted = [
        b
        for b, line in user_flags.items()
        if line["action"] == "promote" and b in rank_of and (b in eligible or b in demoted)
    ]
    for buyer in sorted(promoted, key=rank_of.get):
        line = user_flags[buyer]
        if buyer in tier1:  # already there after other moves: still log the deal team's decision
            log.append(
                Move(
                    buyer,
                    rank_of[buyer],
                    "to Tier 1",
                    "USER_PROMOTE",
                    flag_note(line) + "; already in Tier 1 after other moves",
                    f"deal-team move, {line['timestamp']}",
                )
            )
            continue
        if len(tier1) == TIER1_SIZE:
            out = max((b for b in tier1 if b not in promoted), key=rank_of.get)
            tier1.remove(out)
            demoted.append(out)
        tier1.append(buyer)
        if buyer in demoted:
            demoted.remove(buyer)
        log.append(
            Move(
                buyer,
                rank_of[buyer],
                "to Tier 1",
                "USER_PROMOTE",
                flag_note(line),
                f"deal-team move, {line['timestamp']}",
            )
        )
    tier1.sort(key=rank_of.get)

    # 6. both buyer types in Tier 1
    for missing in (0, 1):
        if tier1 and not any(type_of[b] == missing for b in tier1):
            best = next(
                (
                    b
                    for b in eligible
                    if type_of[b] == missing and b not in tier1 and b not in demoted
                ),
                None,
            )
            if best:
                out = max((b for b in tier1 if type_of[b] != missing), key=rank_of.get)
                tier1[tier1.index(out)] = best
                log.append(
                    Move(
                        best,
                        rank_of[best],
                        "to Tier 1",
                        "BUYER_TYPE_MIX",
                        f"No {'sponsor' if missing else 'strategic'} in Tier 1; replaces {out}",
                        "deal file",
                    )
                )
                demoted.append(out)

    # 7. Tier 2: the next buyers in model order, plus demoted ones, plus coverage checks
    rest = [b for b in ranked.acquirer if b not in tier1 and (b in eligible or b in demoted)]
    tier2 = rest[:TIER2_SIZE]
    fit = fit_labels(target.sector)
    for k in (1, 2, 3):
        best = next(
            (
                r.acquirer
                for r in ranked.itertuples()
                if r.sponsor == 0 and r.closest_ring == k and r.acquirer in rest
            ),
            None,
        )
        if best and best not in tier2:
            tier2.append(best)
            log.append(
                Move(
                    best,
                    rank_of[best],
                    "added to Tier 2",
                    "ADJACENCY_COVERAGE",
                    f"Best-ranked strategic from {_article(fit[k])}, shown so adjacent buyers are "
                    "visibly considered",
                    "deal file",
                )
            )
    for buyer in _obvious_buyers(history, target):
        if buyer in rest and buyer not in tier2:
            tier2.append(buyer)
            log.append(
                Move(
                    buyer,
                    rank_of[buyer],
                    "added to Tier 2",
                    "OBVIOUS_BUYER",
                    f"Recent similar-size {model_words(target.sector)[0]} deal in the target's "
                    "region",
                    "deal file",
                )
            )
    return Review(tier1, tier2, log, flags, matrix, csv_only)


def _article(label: str) -> str:
    """Return the fit label with "a" or "an" ("an adjacent provider sub-sector"); "outside ..."
    takes none."""
    return (
        label if label.startswith("outside") else ("an " if label[0] in "aeiou" else "a ") + label
    )


def _obvious_buyers(history: pd.DataFrame, target: Target) -> list[str]:
    """Return buyers with a closed, similar-size, core or Ring 1 deal in the target's region
    recently."""
    year, _ = target.scoring_quarter()
    recent = history[
        (history.deal_year >= year - OBVIOUS_YEARS)
        & (history.outcome == "Closed")
        & (history.geography == target.region)
        & history.deal_size_mm.between(
            0.5 * target.enterprise_value_mm, 2 * target.enterprise_value_mm
        )
        & (history.sector.map(lambda s: ring(s, target.sector)) <= 1)
    ]
    return list(dict.fromkeys(recent.acquirer))
