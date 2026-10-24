"""No look-ahead: a deal's signals depend only on deals dated before it, never on the deal itself
or later ones."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from acquirer_engine.data import load_deals
from acquirer_engine.signals import buyer_signals

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def deals():
    return load_deals(ROOT / "data/ma_transactions_500.csv")


def test_signals_ignore_later_deals_and_the_deal_itself(deals):
    """Scrambling every later deal, and the scored deal's own outcome, price and tags, changes
    nothing."""
    i = 300
    deal = deals.loc[i]
    before = buyer_signals(deal, deals[deals.t < deal.t])
    tampered = deals.copy()
    later = tampered.t >= deal.t
    tampered.loc[later, "deal_size_mm"] *= 7  # later deals (including this one) rewritten
    tampered.loc[later, "acquirer"] = "Someone Else"
    tampered.loc[i, ["outcome", "strategic_rationale_tags", "ev_ebitda_multiple"]] = [
        "Withdrawn",
        "Scale",
        99.0,
    ]
    after = buyer_signals(tampered.loc[i], tampered[tampered.t < deal.t])
    pd.testing.assert_frame_equal(before, after)


def test_scored_deal_fields_are_not_read(deals):
    """Only the target's profile fields reach buyer_signals; the deal's own outcome, tags and price
    can't."""
    deal = deals.loc[250].copy()
    history = deals[deals.t < deal.t]
    a = buyer_signals(deal, history)
    (
        deal["outcome"],
        deal["strategic_rationale_tags"],
        deal["ev_ebitda_multiple"],
        deal["acquirer"],
    ) = "Rumored", "Scale", 1.0, "X"
    pd.testing.assert_frame_equal(a, buyer_signals(deal, history))
