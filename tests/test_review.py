"""Step 3: valuation uses stated multiples only, the quick LBO matches the paper LBO, and the
banker review gives the agreed Tier 1 with a reason code and a source for every move."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from acquirer_engine.config import load_target
from acquirer_engine.data import load_deals
from acquirer_engine.engine import run
from acquirer_engine.lbo import sponsor_ability_to_pay
from acquirer_engine.valuation import comparable_deals

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/ma_transactions_500.csv"
MODEL = ROOT / "models/ranking_model.json"
AGREED_TIER1 = [
    "Advocate Health",
    "Nordic Capital",
    "UPMC",
    "General Atlantic",
    "Blackstone",
    "Welsh Carson",
    "GTCR",
    "TPG Capital",
    "Advent International",
    "Intermountain Health",
]


@pytest.fixture(scope="module")
def result(request):
    import os

    os.chdir(ROOT)  # profiles are read from data/profiles relative to the repo root
    return run(load_target(ROOT / "configs/target_default.yaml"), CSV, model_path=MODEL)


def test_comparable_deals_are_closed_control_deals_in_band():
    target = load_target(ROOT / "configs/target_default.yaml")
    comps = comparable_deals(load_deals(CSV), target)
    assert len(comps) == 70
    assert (comps.outcome == "Closed").all() and (comps.deal_type != "Minority Investment").all()
    assert comps.deal_size_mm.between(100, 400).all()


def test_valuation_never_uses_the_ebitda_field():
    source = (ROOT / "src/acquirer_engine/valuation.py").read_text() + (
        ROOT / "src/acquirer_engine/lbo.py"
    ).read_text()
    assert "target_ebitda_mm" not in source


def test_quick_lbo_matches_the_paper_lbo():
    # research/paper_lbo.py at a 20% margin: ~19% IRR at $200M, max $194M at 20% IRR, $217M exiting
    # at 15x
    lbo = sponsor_ability_to_pay(200, 12.8, 0.11, 0.20, 15.0)
    assert lbo["irr_at_ask"] == pytest.approx(0.189, abs=0.001)
    assert (lbo["max_price_same_exit"], lbo["max_price_platform_exit"]) == (194, 217)


def test_review_gives_the_agreed_tier_1(result):
    assert result.review.tier1 == AGREED_TIER1
    assert len(result.review.tier2) >= 15


def test_every_move_has_a_code_and_a_source(result):
    for move in result.review.log:
        assert move.code and move.source
        if move.code in (
            "STATUS_DEFUNCT",
            "STATUS_ACQUIRED",
            "ENTITY_DUPLICATE",
            "MANDATE_MISMATCH",
        ):
            assert "as of" in move.source  # profile facts are cited with their date


def test_adjacency_coverage_shows_every_outer_ring(result):
    rings = set(
        result.ranked.set_index("acquirer")
        .loc[result.review.tier2]
        .query("sponsor == 0")
        .closest_ring
    )
    assert {1, 2, 3} <= rings


def test_csv_only_switches_profile_rules_off():
    res = run(
        load_target(ROOT / "configs/target_default.yaml"), CSV, csv_only=True, model_path=MODEL
    )
    assert res.review.csv_only and not res.profiles
    assert not any(
        m.code in ("STATUS_DEFUNCT", "ENTITY_DUPLICATE", "MANDATE_MISMATCH") for m in res.review.log
    )
    assert (
        "Steward Health Care" in res.review.tier1
    )  # without profiles the bankrupt buyer stays: that is the point


def test_other_regions_run(result):
    for region in ("Southeast", "National"):
        res = run(replace(result.target, region=region), CSV, model_path=MODEL)
        assert len(res.review.tier1) == 10
