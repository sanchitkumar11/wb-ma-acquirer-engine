"""Buyer signals: what each buyer's earlier deals say about one target.

Every signal is a criterion a banker uses to screen buyers ("has done this sub-sector", "has written
a cheque this big"). They are computed from history only: when a past deal is the target, history
is every deal dated before it, so the model never sees the answer (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from acquirer_engine.rings import (
    PROVIDER_SECTORS,
    model_sectors,
    region_similarity,
    ring,
    sector_similarity,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Control deals of private companies: the kind of deal a $200M private seller is offering
CONTROL_TYPES = {
    "Platform Investment",
    "Bolt-on Acquisition",
    "Leveraged Buyout",
    "Recapitalization",
    "Management Buyout",
    "Strategic Acquisition",
}
# Deal types that start a sponsor platform (later deals can be add-ons to it)
PLATFORM_TYPES = {
    "Platform Investment",
    "Leveraged Buyout",
    "Recapitalization",
    "Management Buyout",
    "Carve-out",
}
RECENCY_HALF_LIFE_YEARS = 3  # a deal 3 years ago counts half
OPEN_PLATFORM_QUARTERS = 20  # a platform bought in the last 5 years is still "open"
SHRINK_PSEUDO_DEALS = 2  # thin histories are pulled toward the buyer-type average

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def buyer_signals(target: dict, history: pd.DataFrame) -> pd.DataFrame:
    """Return one row per buyer in `history` with its signals against `target`.

    `target` is a deal row or Target.signal_inputs(): it needs sector, lsize, geography,
    ebitda_margin_pct, revenue_growth_pct, target_ownership_pre, deal_year and q. Only those fields
    are read, so the target's own price, tags, deal type and outcome can't leak in.

    Besides the model signals, each row carries plain facts used later by the review and the fact
    pack (n, n_closed, max_size, med_size, n_model, closest_ring, home_sector).
    """
    rows = []
    for buyer, g in history.groupby("acquirer"):
        n = len(g)
        ssim = np.array([sector_similarity(s, target["sector"]) for s in g.sector])
        gsim = np.array([region_similarity(r, target["geography"]) for r in g.geography])
        rec = 0.5 ** ((target["deal_year"] - g.deal_year.values) / RECENCY_HALF_LIFE_YEARS)
        size_close = np.exp(-np.abs(g.lsize.values - target["lsize"]))
        margin_close = np.exp(-np.abs(g.ebitda_margin_pct.values - target["ebitda_margin_pct"]) / 5)
        recent_platforms = g[
            (g.q >= target["q"] - OPEN_PLATFORM_QUARTERS) & g.deal_type.isin(PLATFORM_TYPES)
        ]
        rows.append(
            dict(
                acquirer=buyer,
                sponsor=int(g.acquirer_type.iloc[0] == "Financial Sponsor"),
                n=n,
                # track record
                activity=np.log1p(n),
                recency=rec.sum(),
                close_rate=(g.outcome == "Closed").mean(),
                bidder_exp=g.num_bidders.mean(),
                # sector
                sector_share=(g.sector.values == target["sector"]).mean(),
                sector_rel=(ssim * rec).sum(),
                precedent_sim=(
                    ssim
                    * (0.4 + 0.6 * size_close)
                    * (0.6 + 0.4 * gsim)
                    * (0.7 + 0.3 * margin_close)
                    * rec
                ).sum(),
                # size
                size_fit=-abs(target["lsize"] - g.lsize.mean()),
                size_in_range=float(
                    g.lsize.min() - 0.25 <= target["lsize"] <= g.lsize.max() + 0.25
                ),
                not_bigger=-max(0.0, target["lsize"] - g.lsize.max()),  # above its biggest deal
                not_smaller=-max(0.0, g.lsize.min() - target["lsize"]),  # below its smallest
                # deal profile
                geo_fit=(gsim * rec).sum() / rec.sum(),
                own_fit=(g.target_ownership_pre.values == target["target_ownership_pre"]).mean(),
                type_fit=g.deal_type.isin(CONTROL_TYPES).mean(),
                margin_fit=-abs(target["ebitda_margin_pct"] - g.ebitda_margin_pct.median()),
                growth_fit=-abs(target["revenue_growth_pct"] - g.revenue_growth_pct.median()),
                # sponsor add-on route
                open_platform=float((recent_platforms.sector == target["sector"]).any()),
                open_platform_prov=float(
                    recent_platforms.sector.isin(PROVIDER_SECTORS).any()
                    and target["sector"] in PROVIDER_SECTORS
                ),
                # plain facts for the review and the fact pack
                n_closed=(g.outcome == "Closed").sum(),
                n_control=g.deal_type.isin(CONTROL_TYPES).sum(),
                n_own=(g.target_ownership_pre == target["target_ownership_pre"]).sum(),
                max_size=g.deal_size_mm.max(),
                med_size=g.deal_size_mm.median(),
                n_model=g.sector.isin(
                    model_sectors(target["sector"])
                ).sum(),  # deals in the target's business model
                closest_ring=min(ring(s, target["sector"]) for s in g.sector),
                home_sector=g.sector.mode()[0],
            )
        )
    signals = pd.DataFrame(rows)
    # One deal isn't a pattern: blend each ratio with the buyer-type average, as if the buyer
    # had 2 extra deals at that average
    for ratio, count in (
        ("close_rate", "n_closed"),
        ("type_fit", "n_control"),
        ("own_fit", "n_own"),
    ):
        type_mean = signals.groupby("sponsor")[ratio].transform("mean")
        signals[ratio + "_s"] = (signals[count] + SHRINK_PSEUDO_DEALS * type_mean) / (
            signals.n + SHRINK_PSEUDO_DEALS
        )
    return signals


def training_pairs(deals: pd.DataFrame) -> pd.DataFrame:
    """Return buyer x deal rows for every deal with earlier history: label 1 for the actual buyer.

    Each past deal becomes a multiple-choice question: "of the buyers active before this deal,
    which one did it?". Columns: the buyer_signals columns plus event (deal row), year and label.
    Takes about a minute on the 500-deal file, so the fitted model is saved (see model.py).
    """
    parts = []
    for i, deal in deals.iterrows():
        history = deals[deals.t < deal.t]
        if history.empty:
            continue
        signals = buyer_signals(deal, history)
        signals["event"] = i
        signals["year"] = deal.deal_year
        signals["label"] = (signals.acquirer == deal.acquirer).astype(int)
        parts.append(signals)
    return pd.concat(parts, ignore_index=True)
