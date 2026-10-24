"""Arbitrary targets: the two contrasting samples (Health IT, Dental) re-centre every
target-specific part.

Each test is one item of the definition of done: rings, comparable deals and the defensible
percentile recompute; the technology-investor rule stays off for a technology target; buyers with
no profile are shown as "not researched"; every committed page passes the validator; and the
default target's fact packs are unchanged, so its cached pages still replay.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acquirer_engine.config import load_target
from acquirer_engine.factpack import pack_hash
from acquirer_engine.onepager import OnePager, _numbers, validate
from acquirer_engine.render import NOT_RESEARCHED, onepager_markdown
from acquirer_engine.report import build
from acquirer_engine.rings import fit_labels, ring
from acquirer_engine.summary import not_researched, sources_markdown, summary_pack

ROOT = Path(__file__).resolve().parents[1]
TECH_INVESTORS = {"Vista Equity", "Veritas Capital", "Francisco Partners"}


@pytest.fixture(scope="module")
def reports():
    with pytest.MonkeyPatch.context() as mp:  # paths in the engine are relative to the repo
        mp.chdir(ROOT)
        return {
            name: build(load_target(ROOT / f"configs/target_{name}.yaml"), replay=True)
            for name in ("default", "healthit", "dental")
        }


def _pack_text(rep) -> str:
    """Return every fact pack and the summary pack as text, without quoted profile facts (a buyer's
    own words)."""
    packs = [
        {k: v for k, v in p.items() if k not in ("profile_facts", "closest_peers")}
        for p in rep.packs.values()
    ]  # quoted profile text
    return json.dumps(packs) + json.dumps(summary_pack(rep))


def test_rings_recentre_on_the_target(reports):
    """Fit labels follow the target: a Health IT pack never calls anything 'provider'."""
    assert (
        ring("Health IT", "Health IT") == 0
        and ring("Dental", "Health IT") == 3
        and ring("Dental", "Dental") == 0
    )
    assert ring("Healthcare Services", "Dental") == 1
    assert fit_labels("Health IT")[2] == "neighbouring sector (revenue cycle, devices or pharma)"
    assert fit_labels("Healthcare Services")[2] == "neighbouring sector (payer or revenue cycle)"
    tech = reports["healthit"]
    assert "provider" not in _pack_text(tech).lower()
    for pack in tech.packs.values():
        assert {"adjacent_health_it_deals", "deals_outside_health_it"} <= set(pack["track_record"])
        for p in pack["precedents"]:
            assert (p["fit"] == "same sub-sector as the target") == (p["sub_sector"] == "Health IT")


def test_comparable_pool_and_defensible_percentile_recompute(reports):
    """Comps are the target's sub-sector and its business model, at 0.5-2x its EV; the anchor
    follows the margin."""
    tech, dental = reports["healthit"].res.valuation, reports["dental"].res.valuation
    assert tech["size_band_mm"] == (150, 600) and tech["n_comps"] == tech["n_core"] == 13
    assert (
        dental["size_band_mm"] == (40, 160) and dental["n_comps"] == 63 and dental["n_core"] == 12
    )
    # Health IT: margin above the comps' median but only at their 54th percentile, so the median
    # anchors
    assert tech["margin_percentile"] == 54 and tech["defensible_percentile"] == 50
    assert "not in their top quarter" in tech["defensible_reason"]
    assert dental["margin_percentile"] == 92 and dental["defensible_percentile"] == 75
    assert reports["default"].res.valuation["defensible_percentile"] == 75


def test_platform_exit_needs_evidence_of_a_premium(reports):
    """Default: 15.0x from larger provider deals. Health IT has one such deal, Dental's trade
    lower: no premium."""
    lbo = {k: r.res.sponsor_lbo for k, r in reports.items()}
    assert lbo["default"]["platform_exit_multiple"] == 15.0
    assert (
        lbo["healthit"]["platform_exit_multiple"]
        == reports["healthit"].res.valuation["ev_ebitda"]["p50"]
    )
    assert (
        lbo["dental"]["platform_exit_multiple"]
        == reports["dental"].res.valuation["ev_ebitda"]["p50"]
    )


def test_technology_investor_rule_is_off_for_a_technology_target(reports):
    """Health IT: no mandate move, and Veritas and Francisco Partners reach Tier 1. Dental: all
    three move."""
    tech, dental = reports["healthit"], reports["dental"]
    assert not any(m.code == "MANDATE_MISMATCH" for m in tech.res.review.log)
    assert {"Veritas Capital", "Francisco Partners"} <= set(tech.tier1)
    assert (
        "Vista Equity" in tech.res.review.tier1 + tech.res.review.tier2
    )  # eligible; ranked 21st by the model
    moved = {m.buyer for m in dental.res.review.log if m.code == "MANDATE_MISMATCH"}
    assert moved == TECH_INVESTORS and not moved & set(dental.tier1)


def test_buyers_without_a_profile_are_shown_as_not_researched(reports):
    """No guessed facts: an unresearched Tier 1 buyer has no profile facts, and its page and
    Sources say so."""
    rep = reports["dental"]
    missing = [b for b in rep.tier1 if rep.packs[b]["profile_not_researched"]]
    assert {"Heartland Dental", "Aspen Dental (ADMI)", "Dental365"} <= set(missing)
    for b in missing:
        assert rep.packs[b]["profile_facts"] == []
        assert (
            rep.packs[b]["ability_to_pay"].get("buyer_revenue", "not researched")
            == "not researched"
        )
    page = next(r for r in rep.results if r["buyer"] == "Heartland Dental")
    assert NOT_RESEARCHED in onepager_markdown(
        OnePager.model_validate(page["page"]), rep.packs["Heartland Dental"]
    )
    assert set(missing) <= set(not_researched(rep)) and "Not researched" in sources_markdown(rep)


def test_health_it_tier1_is_fully_profiled(reports):
    """The six Health IT buyers profiled on Oct 23 (EDGAR first) carry sourced facts into their
    packs."""
    rep = reports["healthit"]
    assert not [b for b in rep.tier1 if rep.packs[b]["profile_not_researched"]]
    for b in (
        "Evolent Health",
        "Epic Systems",
        "McKesson",
        "Aledade",
        "Privia Health",
        "Warburg Pincus",
    ):
        assert 4 <= len(rep.packs[b]["profile_facts"]) <= 6


def test_stated_no_acquisitions_caps_conviction_at_low(reports):
    """Epic says it makes no acquisitions: flagged (not removed), Low conviction, and the page says
    so."""
    rep = reports["healthit"]
    assert "Epic Systems" in rep.tier1 and rep.packs["Epic Systems"]["conviction"]["level"] == "Low"
    assert any(
        f["id"] == "R:STATED_NO_ACQUISITIONS" for f in rep.packs["Epic Systems"]["review"]["flags"]
    )
    page = next(r for r in rep.results if r["buyer"] == "Epic Systems")["page"]
    assert "no acquisitions" in json.dumps(page).lower()


def test_look_alike_strategics_see_each_others_stated_strategy(reports):
    """Aledade and Privia: each pack names the other with its stated strategy; sponsors and the
    default target get none."""
    packs = reports["healthit"].packs
    peers = {
        b: {p["buyer"] for p in packs[b].get("closest_peers", {}).get("peers", [])}
        for b in ("Aledade", "Privia Health")
    }
    assert "Privia Health" in peers["Aledade"] and "Aledade" in peers["Privia Health"]
    assert "closest_peers" not in packs["Blackstone"]
    assert not any("closest_peers" in p for p in reports["default"].packs.values())


def test_summary_says_sponsor_order_is_weak(reports):
    """Every summary pack with two or more Tier 1 sponsors carries the back-test finding, and the
    summary uses it."""
    for name in ("default", "healthit", "dental"):
        rep = reports[name]
        assert "28%" in summary_pack(rep)["sponsor_ordering"]["finding"]
        assert "tie" in " ".join(rep.exec_summary["sentences"]).lower() and rep.exec_from_llm


@pytest.mark.parametrize("name", ["default", "healthit", "dental"])
def test_every_committed_page_passes_the_validator(reports, name):
    """The committed pages replay from the cache and pass every check against their own fact
    packs."""
    rep = reports[name]
    assert rep.results is not None and len(rep.results) == 10
    for r in rep.results:
        assert validate(OnePager.model_validate(r["page"]), rep.packs[r["buyer"]]) == [], r["buyer"]
    assert rep.exec_from_llm


def test_default_target_packs_are_unchanged(reports):
    """Re-centring the wording must not change a provider target's fact packs (their hashes key the
    cache)."""
    expected = json.loads((ROOT / "tests/fixtures/default_pack_hashes.json").read_text())
    rep = reports["default"]
    assert {b: pack_hash(p) for b, p in rep.packs.items()} == expected["packs"]
    assert pack_hash(summary_pack(rep)) == expected["summary"]


def test_digits_inside_a_name_are_not_numbers():
    """Dental365 is a buyer name, not the number 65 (a false failure found on the Dental run)."""
    assert _numbers("Dental365 paid $1,304M at 15.5x in 2023") == {"1304", "15.5", "2023"}
