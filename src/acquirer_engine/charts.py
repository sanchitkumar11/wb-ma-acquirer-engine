"""Football-field data for one buyer: the price ranges a banker would line up against the ask.

All bars are in $M of enterprise value. The target's EBITDA isn't given, so multiples become EV at
the EBITDA implied by the ask at the comparable deals' median multiple ($200M / 12.8x = $15.6M);
that assumption is printed under the chart. Plotting is left to the app.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from acquirer_engine.lbo import SYNERGY_RANGE_TEXT

if TYPE_CHECKING:
    from acquirer_engine.report_types import Report


def buyer_price_ranges(rep: Report, buyer: str) -> tuple[pd.DataFrame, float, str]:
    """Return (rows: label, low, high, detail; the ask; the footnote) for the buyer's football
    field."""
    res, ask = rep.res, rep.res.target.enterprise_value_mm
    v = res.valuation
    ebitda = ask / v["ev_ebitda"]["p50"]
    rows = [
        dict(
            label="Comparable closed deals (25th-75th percentile)",
            low=v["ev_ebitda"]["p25"] * ebitda,
            high=v["ev_ebitda"]["p75"] * ebitda,
            detail=f"{v['ev_ebitda']['p25']}x-{v['ev_ebitda']['p75']}x EV/EBITDA, {v['n_comps']} "
            "deals",
        )
    ]
    own = res.deals[
        (res.deals.acquirer.isin([buyer, *rep.packs[buyer]["buyer"]["merged_into_this_buyer"]]))
        & (res.deals.outcome == "Closed")
    ].ev_ebitda_multiple
    if len(own):
        low, high = (
            (own.quantile(0.25), own.quantile(0.75)) if len(own) >= 4 else (own.min(), own.max())
        )
        span = (
            "25th-75th percentile"
            if len(own) >= 4
            else ("its one deal" if len(own) == 1 else "range")
        )
        rows.append(
            dict(
                label=f"{buyer}'s own closed deals ({span})",
                low=low * ebitda,
                high=high * ebitda,
                detail=f"{low:.1f}x-{high:.1f}x EV/EBITDA, {len(own)} closed "
                f"deal{'s' if len(own) != 1 else ''}",
            )
        )
    if rep.packs[buyer]["buyer"]["type"] == "Financial sponsor":
        lbo = res.sponsor_lbo
        rows.append(
            dict(
                label=f"Quick LBO: maximum price at a {100 * lbo['target_irr']:.0f}% IRR",
                low=lbo["max_price_same_exit"],
                high=lbo["max_price_platform_exit"],
                detail=f"same exit to {lbo['platform_exit_multiple']}x platform exit, "
                f"{lbo['leverage']}x debt",
            )
        )
    else:
        syn = res.strategic_synergies
        rows.append(
            dict(
                label="Ask plus cost-synergy headroom",
                low=ask + syn["synergy_value_low"],
                high=ask + syn["synergy_value_high"],
                detail=f"{SYNERGY_RANGE_TEXT}, half realised",
            )
        )
    note = (
        f"EV in $M. Multiples are converted at the EBITDA implied by the ${ask:.0f}M ask at the "
        "comparable median "
        f"({v['ev_ebitda']['p50']}x), i.e. ${ebitda:.1f}M; the target's own EBITDA is not given. "
        f"{res.target.data_note()}."
    )
    return pd.DataFrame(rows).round(0), ask, note
