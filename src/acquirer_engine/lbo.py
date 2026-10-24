"""Ability to pay: a quick LBO for sponsors and synergy headroom for strategics.

Back-of-envelope on purpose (the LBO guide's "paper LBO"): it answers "can this buyer pay ~$200M
and still hit its return?", not "what exactly will it bid". A three-statement model would not
change who belongs on the list.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration: today's financing terms are inputs, not model signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LboAssumptions:
    """Financing and operating assumptions for the quick LBO (today's terms, one set for all)."""

    leverage: float = 5.0  # debt / EBITDA at entry
    interest_rate: float = 0.085  # cost of that debt
    years: int = 5
    tax_rate: float = 0.25
    capex_pct_revenue: float = 0.03
    nwc_pct_revenue_change: float = 0.05
    da_pct_revenue: float = 0.025
    fees_pct: float = 0.03
    target_irr: float = 0.20  # typical sponsor hurdle


DEFAULT_ASSUMPTIONS = LboAssumptions()  # frozen, so one shared default is safe
SYNERGY_PCT_OPEX = (0.03, 0.05)  # cost synergies as a share of the target's operating costs
SYNERGY_REALISED = 0.5  # haircut: only half is usually delivered
SYNERGY_RANGE_TEXT = (
    f"{SYNERGY_PCT_OPEX[0]:.0%}-{SYNERGY_PCT_OPEX[1]:.0%}".replace("%-", "-") + " of implied opex"
)  # "3-5% of implied opex": the one wording every page and table uses

# ---------------------------------------------------------------------------
# Sponsor: quick LBO
# ---------------------------------------------------------------------------


def lbo_returns(
    price: float,
    ebitda: float,
    growth: float,
    margin: float,
    exit_multiple: float,
    a: LboAssumptions = DEFAULT_ASSUMPTIONS,
) -> dict:
    """Return entry equity, entry debt, exit EV, debt left, money multiple and IRR for one price.

    Revenue grows at `growth` with a constant margin; free cash flow after interest, tax, capex
    and working capital pays down debt; exit at `exit_multiple` x final-year EBITDA.
    """
    revenue = ebitda / margin
    debt = entry_debt = a.leverage * ebitda
    equity = price * (1 + a.fees_pct) - entry_debt
    for _ in range(a.years):
        new_revenue = revenue * (1 + growth)
        year_ebitda = new_revenue * margin
        interest = debt * a.interest_rate
        taxes = max(0.0, year_ebitda - new_revenue * a.da_pct_revenue - interest) * a.tax_rate
        fcf = (
            year_ebitda
            - interest
            - taxes
            - new_revenue * a.capex_pct_revenue
            - (new_revenue - revenue) * a.nwc_pct_revenue_change
        )
        debt = max(0.0, debt - fcf)
        revenue = new_revenue
    exit_ev = year_ebitda * exit_multiple
    money_multiple = (exit_ev - debt) / equity
    return dict(
        equity=equity,
        debt=entry_debt,
        exit_ev=exit_ev,
        debt_left=debt,
        money_multiple=money_multiple,
        irr=money_multiple ** (1 / a.years) - 1,
    )


def max_price(
    ebitda: float,
    growth: float,
    margin: float,
    exit_multiple: float,
    a: LboAssumptions = DEFAULT_ASSUMPTIONS,
) -> float:
    """Return the highest EV at which the LBO still reaches `a.target_irr` (bisection, $M)."""
    low, high = 0.5 * ebitda, 40 * ebitda
    for _ in range(60):
        mid = (low + high) / 2
        if lbo_returns(mid, ebitda, growth, margin, exit_multiple, a)["irr"] > a.target_irr:
            low = mid
        else:
            high = mid
    return low


def sponsor_ability_to_pay(
    ev: float,
    entry_multiple: float,
    growth: float,
    margin: float,
    platform_exit_multiple: float,
    a: LboAssumptions = DEFAULT_ASSUMPTIONS,
) -> dict:
    """Return the quick-LBO facts for a sponsor paying `ev`.

    EBITDA is implied as ev / entry_multiple (the comparable deals' median stated multiple), never
    taken from the file's EBITDA field. Two exits: at the entry multiple (no multiple expansion),
    and at `platform_exit_multiple` for a sponsor that sells the target inside a bigger platform.
    """
    ebitda = ev / entry_multiple
    base = lbo_returns(ev, ebitda, growth, margin, entry_multiple, a)
    return dict(
        implied_ebitda=round(ebitda, 1),
        leverage=a.leverage,
        interest_rate=a.interest_rate,
        equity_cheque=round(base["equity"]),
        irr_at_ask=round(base["irr"], 3),
        money_multiple_at_ask=round(base["money_multiple"], 2),
        target_irr=a.target_irr,
        max_price_same_exit=round(max_price(ebitda, growth, margin, entry_multiple, a)),
        max_price_platform_exit=round(max_price(ebitda, growth, margin, platform_exit_multiple, a)),
        platform_exit_multiple=platform_exit_multiple,
        entry_multiple=entry_multiple,
    )


# ---------------------------------------------------------------------------
# Strategic: synergy headroom
# ---------------------------------------------------------------------------


def strategic_synergy_headroom(ev: float, entry_multiple: float, margin: float) -> dict:
    """Return the value of cost synergies a strategic could pay away, at 3% and 5% of target opex.

    Opex is implied revenue minus implied EBITDA. Synergies are capitalised at the entry multiple
    and haircut to half, the usual realised share. Headroom = how far above `ev` it could go.
    """
    ebitda = ev / entry_multiple
    opex = ebitda / margin - ebitda
    low, high = (opex * p * entry_multiple * SYNERGY_REALISED for p in SYNERGY_PCT_OPEX)
    return dict(
        implied_ebitda=round(ebitda, 1),
        implied_opex=round(opex, 1),
        synergy_value_low=round(low),
        synergy_value_high=round(high),
        synergy_pct_opex=SYNERGY_PCT_OPEX,
        realised_share=SYNERGY_REALISED,
    )
