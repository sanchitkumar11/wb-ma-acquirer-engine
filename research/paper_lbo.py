"""Illustrative paper LBO for the target and data checks behind the PDF review. Usage: python3 -I paper_lbo.py <csv>"""
import sys
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv(sys.argv[1])
PROV = ["Healthcare Services", "Physician Groups", "Behavioral Health", "Home Health/Hospice", "Dental"]
cl = df[(df.outcome == "Closed") & (df.deal_type != "Minority Investment")]
cp = cl[cl.sector.isin(PROV)]
print("== provider closed control: median stated EV/EBITDA by size bucket (multiple-arbitrage evidence)")
b = pd.cut(cp.deal_size_mm, [0, 50, 100, 250, 500, 1000, 1e6], labels=["<50", "50-100", "100-250", "250-500", "500-1000", ">1000"])
print(cp.groupby(b, observed=True).ev_ebitda_multiple.agg(["median", "count"]).round(1).T.to_string())
print("spearman size~mult provider:", stats.spearmanr(cp.deal_size_mm, cp.ev_ebitda_multiple))
h = df[df.sector == "Healthcare Services"]
print("HCS growth median", h.revenue_growth_pct.median(), " provider growth median", df[df.sector.isin(PROV)].revenue_growth_pct.median())

# revealed capacity for strategics
st = df[df.acquirer_type == "Strategic"].groupby("acquirer").deal_size_mm.agg(["max", "median", "count"])
print("\nstrategics whose largest deal < $200M:", (st["max"] < 200).sum(), "of", len(st))
prov_st = df[(df.acquirer_type == "Strategic") & df.sector.isin(PROV)].acquirer.unique()
s2 = st.loc[prov_st].sort_values("max")
print(s2.round(0).to_string())

def paper_lbo(price_ev, ebitda0, growth, margin, lev, rate, exit_mult, years=5, tax=0.25, capex_pct=0.03, nwc_pct=0.05, fees_pct=0.03, da_pct=0.025):
    rev0 = ebitda0 / margin
    debt0 = lev * ebitda0
    equity0 = price_ev * (1 + fees_pct) - debt0
    debt = debt0; rev = rev0
    for y in range(years):
        rev_new = rev * (1 + growth)
        ebitda = rev_new * margin
        interest = debt * rate
        ebt = ebitda - rev_new * da_pct - interest
        taxes = max(0, ebt) * tax
        fcf = ebitda - interest - taxes - rev_new * capex_pct - (rev_new - rev) * nwc_pct
        debt = max(0, debt - fcf)
        rev = rev_new
    exit_ev = ebitda * exit_mult
    eq = exit_ev - debt
    mom = eq / equity0
    irr = mom ** (1 / years) - 1
    return dict(equity0=equity0, debt0=debt0, exit_ebitda=ebitda, exit_ev=exit_ev, debt_left=debt, mom=mom, irr=irr)

E0 = 200 / 12.8  # EBITDA implied by $200M at the comp-set median stated multiple
base = dict(ebitda0=E0, growth=0.11, margin=0.20, lev=5.0, rate=0.085, exit_mult=12.8)
r = paper_lbo(200, **base)
print("\n== paper LBO at $200M (EBITDA %.1f, 11%% growth, 5.0x debt at 8.5%%, exit 12.8x, 5 yrs)" % E0)
print({k: round(v, 3) for k, v in r.items()})

def max_price(target_irr, **kw):
    lo, hi = 50, 400
    for _ in range(60):
        mid = (lo + hi) / 2
        if paper_lbo(mid, **kw)["irr"] > target_irr: lo = mid
        else: hi = mid
    return lo

for lev in (4.0, 5.0, 6.0):
    kw = dict(base, lev=lev)
    print(f"lev {lev}x: IRR at $200M {paper_lbo(200, **kw)['irr']:.1%}; max EV for 20% IRR ${max_price(0.20, **kw):.0f}M; for 25% ${max_price(0.25, **kw):.0f}M")
for g in (0.07, 0.11, 0.15):
    kw = dict(base, growth=g)
    print(f"growth {g:.0%}: IRR at $200M {paper_lbo(200, **kw)['irr']:.1%}; max EV @20% ${max_price(0.20, **kw):.0f}M")
for em in (11.0, 12.8, 15.0):
    kw = dict(base, exit_mult=em)
    print(f"exit {em}x: IRR at $200M {paper_lbo(200, **kw)['irr']:.1%}; max EV @20% ${max_price(0.20, **kw):.0f}M")
# returns attribution at base
r = paper_lbo(200, **base)
ebitda_growth = (r["exit_ebitda"] - E0) * 12.8
mult_exp = 0
debt_pay = r["debt0"] - r["debt_left"]
print("attribution: EBITDA growth $%.0fM, multiple expansion $%.0fM, debt paydown $%.0fM" % (ebitda_growth, mult_exp, debt_pay))

# seller yield vs strategic cost of acquisition (rule of thumb)
nopat = (E0 - (E0 / 0.20) * 0.025) * (1 - 0.25)
print("\nseller unlevered yield at $200M: %.1f%%" % (nopat / 200 * 100))
for name, pre in [("cash @4%", 0.04), ("debt @7%", 0.07)]:
    print(f"  after-tax cost {name}: {pre * 0.75:.2%}")
print("  stock cost = 1/PE; breakeven P/E = %.1fx" % (200 / nopat))
# cost-synergy headroom: 3-5% of target opex capitalised at the comp multiple
opex = E0 / 0.20 - E0
for pct in (0.03, 0.05):
    syn = opex * pct
    print(f"  cost synergies {pct:.0%} of opex = ${syn:.1f}M/yr -> worth ${syn * 12.8:.0f}M at 12.8x; with 50% realised ${syn * 12.8 * 0.5:.0f}M")
