"""Valuation context: what comparable closed deals in the file say the target is worth.

Uses the stated EV/EBITDA and EV/Revenue multiples only. The file's EBITDA field doesn't tie to
deal size in 447 of 500 rows, so nothing here divides by it. The target's EBITDA is unknown too,
so the output is framed as "what $X EV means at each multiple" rather than an implied price.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from acquirer_engine.rings import ring

if TYPE_CHECKING:
    import pandas as pd

    from acquirer_engine.config import Target

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZE_BAND = (0.5, 2.0)  # comparable deals are 0.5x to 2x the target's EV ($100-400M for $200M)
MAX_RING = 1  # core sub-sector plus Ring 1 (same business model)
PERCENTILES = (0.25, 0.50, 0.75)
PLATFORM_BAND = (2.5, 5.0)  # a larger platform: 2.5x to 5x the target's EV ($500M-1B for $200M)
MIN_PLATFORM_DEALS = 5
MIN_COMPS = 5  # quartiles of fewer comparable deals say nothing about price

# ---------------------------------------------------------------------------
# Comparable deals
# ---------------------------------------------------------------------------


def comparable_deals(deals: pd.DataFrame, target: Target) -> pd.DataFrame:
    """Return closed control deals in the target's sub-sector or Ring 1, within the size band.

    Minority stakes are left out because they carry no control premium. Deals dated after the
    target's as-of quarter are left out, so a back-test can reuse this function.
    """
    _year, q = target.scoring_quarter()
    low, high = SIZE_BAND[0] * target.enterprise_value_mm, SIZE_BAND[1] * target.enterprise_value_mm
    in_ring = deals.sector.map(lambda s: ring(s, target.sector) <= MAX_RING)
    keep = (
        (deals.outcome == "Closed")
        & (deals.deal_type != "Minority Investment")
        & in_ring
        & deals.deal_size_mm.between(low, high)
        & (deals.q < q)
    )
    return deals[keep].copy()


def valuation_summary(comps: pd.DataFrame, target: Target) -> dict:
    """Return the valuation facts for the fact pack and the report.

    Keys: n_comps, n_core, size_band_mm, ev_ebitda and ev_revenue (p25/p50/p75), ebitda_at
    (the EBITDA a buyer must believe in for the target's EV to equal each EV/EBITDA percentile),
    margin_percentile (where the target's margin sits among the comps), margin and growth medians
    of the comps, the high-margin comps' medians, and the defensible percentile with its reason:
    the 75th only when the target's margin beats at least 75% of the comps, else the median.

    Raises ValueError when there are fewer than MIN_COMPS comparable deals (e.g. a $5B dental
    target): a price range from two deals would look precise and mean nothing.
    """
    if len(comps) < MIN_COMPS:
        raise ValueError(
            f"Only {len(comps)} comparable closed control deals for {target.label()}; at least "
            f"{MIN_COMPS} are needed for a price range. Try a size closer to the file's deals."
        )
    ev = target.enterprise_value_mm
    ev_ebitda = comps.ev_ebitda_multiple.quantile(PERCENTILES)
    ev_revenue = comps.ev_revenue_multiple.quantile(PERCENTILES)
    margin_pct = (comps.ebitda_margin_pct < target.ebitda_margin_pct).mean()
    high_margin = comps[comps.ebitda_margin_pct >= comps.ebitda_margin_pct.quantile(0.75)]
    margin_median = round(float(comps.ebitda_margin_pct.median()), 1)
    above_peers = target.ebitda_margin_pct > margin_median
    top_quarter = margin_pct >= 0.75  # upper-quartile pricing needs an upper-quartile margin
    return dict(
        n_comps=len(comps),
        n_core=int((comps.sector == target.sector).sum()),
        size_band_mm=(SIZE_BAND[0] * ev, SIZE_BAND[1] * ev),
        ev_ebitda={f"p{int(p * 100)}": round(float(v), 1) for p, v in ev_ebitda.items()},
        ev_revenue={f"p{int(p * 100)}": round(float(v), 2) for p, v in ev_revenue.items()},
        ebitda_at={f"p{int(p * 100)}": round(float(ev / v), 1) for p, v in ev_ebitda.items()},
        margin_percentile=round(100 * margin_pct),
        comps_margin_median=margin_median,
        comps_growth_median=round(float(comps.revenue_growth_pct.median()), 1),
        high_margin_n=len(high_margin),
        high_margin_ev_revenue_median=round(float(high_margin.ev_revenue_multiple.median()), 2),
        high_margin_ev_ebitda_median=round(float(high_margin.ev_ebitda_multiple.median()), 1),
        defensible_percentile=75 if top_quarter else 50,
        defensible_reason=(
            f"Margin of {target.ebitda_margin_pct}% sits above the comparable deals' median "
            f"of {margin_median}%, which supports upper-quartile pricing"
            if top_quarter
            else f"Margin of {target.ebitda_margin_pct}% is above the comparable deals' median of "
            f"{margin_median}% but not in their top quarter, so the median is the defensible anchor"
            if above_peers
            else f"Margin of {target.ebitda_margin_pct}% is at or below the comparable deals' "
            "median "
            f"of {margin_median}%, so the median is the defensible anchor"
        ),
        deal_ids=comps.transaction_id.tolist(),
    )


def platform_exit_multiple(deals: pd.DataFrame, target: Target) -> float:
    """Return the exit multiple for selling the target inside a larger platform, from the file.

    The median stated EV/EBITDA of closed control deals in the target's business model at 2.5-5x its
    EV (for $200M: $500M-1B, 15.4x), rounded down to a whole multiple to stay conservative. With
    fewer than MIN_PLATFORM_DEALS such deals, or if they trade below the comparable median, there is
    no evidence of a platform premium, so the exit is the comparable median (no multiple expansion).
    """
    _, q = target.scoring_quarter()
    low, high = (
        PLATFORM_BAND[0] * target.enterprise_value_mm,
        PLATFORM_BAND[1] * target.enterprise_value_mm,
    )
    in_ring = deals.sector.map(lambda s: ring(s, target.sector) <= MAX_RING)
    big = deals[
        (deals.outcome == "Closed")
        & (deals.deal_type != "Minority Investment")
        & in_ring
        & deals.deal_size_mm.between(low, high)
        & (deals.q < q)
    ]
    same_exit = round(float(comparable_deals(deals, target).ev_ebitda_multiple.median()), 1)
    if len(big) < MIN_PLATFORM_DEALS:
        return same_exit
    return max(float(math.floor(big.ev_ebitda_multiple.median())), same_exit)


def buyer_precedents(
    deals: pd.DataFrame, buyer: str, target: Target, limit: int = 6
) -> pd.DataFrame:
    """Return the buyer's own deals before the as-of quarter, closest ring first, then most recent.

    Columns: transaction_id, target_company, sector, deal_year, deal_size_mm, deal_type, geography,
    ev_ebitda_multiple, ev_revenue_multiple, outcome, ring.
    """
    _, q = target.scoring_quarter()
    own = deals[(deals.acquirer == buyer) & (deals.q < q)].copy()
    own["ring"] = own.sector.map(lambda s: ring(s, target.sector))
    own = own.sort_values(["ring", "t"], ascending=[True, False])
    cols = [
        "transaction_id",
        "target_company",
        "sector",
        "deal_year",
        "deal_size_mm",
        "deal_type",
        "geography",
        "ev_ebitda_multiple",
        "ev_revenue_multiple",
        "outcome",
        "ring",
    ]
    return own[cols].head(limit)
