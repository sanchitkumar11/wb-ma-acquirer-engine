"""Banker thresholds the review, the feedback loop and the list builders share: one place to change
them.

A separate, dependency-free module so every layer can read them without importing the review
(which sits above the feedback loop). Each was chosen by banker logic and, where it gates a
buyer, tested against history first (docs/experiment-log.md, Section 3).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# List shape
# ---------------------------------------------------------------------------

TIER1_SIZE = 10
"""Buyers on the priority list, each with a one-pager."""
TIER2_SIZE = 15
"""Buyers on the bench in model order, before coverage checks add a few."""

# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

STRETCH_SHARE = 0.5
"""A strategic whose biggest deal is under this share of the EV is a stretch (a flag, not a
gate)."""
REVENUE_CLEARS = 20
"""...unless its sourced revenue is at least this multiple of the EV."""
FUND_CONCENTRATION = 0.10
"""A sponsor equity cheque above this share of its fund or AUM is a capacity flag."""

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

OBVIOUS_YEARS = 3
"""A recent similar deal in the target's business model and region (within this many years)
puts the buyer on the list (Tier 1 or 2)."""
