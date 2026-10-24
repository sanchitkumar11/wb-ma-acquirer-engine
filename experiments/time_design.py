"""How should the model be trained over time? Fixed split vs walk-forward refits, recency-weighted training, sliding window,
and market-state signals. Leaky signals (tag_fit, price_fit) are left out here. Usage: python3 -I time_design.py pairs.pkl <csv>"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
P = pd.read_pickle(sys.argv[1]); D = pd.read_csv(sys.argv[2])
D["t"] = D.deal_year * 10 + D.deal_quarter.str[1].astype(int); D = D.sort_values(["t", "transaction_id"]).reset_index(drop=True)
D["q"] = D.deal_year * 4 + D.deal_quarter.str[1].astype(int) - 1
P["q"] = P.event.map(D.q)

# market-state signals, all as-of (only deals in earlier quarters)
spon = (D.acquirer_type == "Financial Sponsor").astype(int)
heat = {}
for q in D.q.unique():
    w = D[(D.q < q) & (D.q >= q - 8)]
    heat[q] = spon[w.index].mean() if len(w) else 0.58
P["lane_heat"] = np.where(P.sponsor == 1, P.q.map(heat), 1 - P.q.map(heat))          # share of the last 8 quarters' deals won by this buyer's lane
cnt = D.groupby(["acquirer", "q"]).size()
def momentum(a, q): return sum(cnt.get((a, k), 0) for k in range(q - 4, q))
P["momentum_4q"] = [momentum(a, q) for a, q in zip(P.acquirer, P.q)]                  # buyer's deals in the last 4 quarters

BASE = ["activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
        "margin_fit", "growth_fit", "close_rate", "bidder_exp", "precedent_sim"]
PRIOR = {0: dict(activity=.3, recency=.3, sector_share=1.0, sector_rel=.5, size_fit=.5, size_in_range=.3, geo_fit=.5, own_fit=.2, type_fit=.2,
                 margin_fit=.2, growth_fit=.2, close_rate=.3, bidder_exp=0, precedent_sim=.5, lane_heat=.3, momentum_4q=.3),
         1: dict(activity=.6, recency=.6, sector_share=.4, sector_rel=.6, size_fit=.6, size_in_range=.4, geo_fit=.2, own_fit=.4, type_fit=.5,
                 margin_fit=.3, growth_fit=.2, close_rate=.3, bidder_exp=.2, precedent_sim=.4, lane_heat=.3, momentum_4q=.3)}

def fit_map(X, y, sw, b0, lam, w_int0):
    Xa = np.column_stack([np.ones(len(X)), X]); prior = np.concatenate([[w_int0], b0]); pen = np.r_[0.0, np.full(len(b0), lam)]
    def f(w):
        z = Xa @ w; p = 1 / (1 + np.exp(-z))
        return -(sw * (y * z - np.logaddexp(0, z))).sum() + 0.5 * (pen * (w - prior) ** 2).sum(), Xa.T @ (sw * (p - y)) + pen * (w - prior)
    return minimize(f, prior.copy(), jac=True, method="L-BFGS-B").x

def score(tr, te, feats, half_life=None, lam=30.0):
    """Lane-specific logistic with banker priors, fit on tr, scored on te. half_life: recency weighting of training deals, in years."""
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    s = np.zeros(len(te))
    for flag in (0, 1):
        a, b = tr[tr.sponsor == flag], te[te.sponsor == flag]
        if half_life:
            sw = 0.5 ** ((te.year.min() - a.year.values) / half_life); sw = sw / sw.mean()
        else: sw = np.ones(len(a))
        br = a.label.mean()
        w = fit_map(((a[feats] - mu) / sd).values, a.label.values, sw, np.array([PRIOR[flag][f] for f in feats]), lam, np.log(br / (1 - br)))
        s[(te.sponsor == flag).values] = w[0] + ((b[feats] - mu) / sd).values @ w[1:]
    return s

def ranks(df, s):
    out = {}
    for e, g in df.assign(s=s).groupby("event"):
        if g.label.sum(): out[e] = int(g.s.rank(ascending=False, method="first")[g.label == 1].iloc[0])
    return pd.Series(out)
def summ(r): return f"R@10 {(r <= 10).mean():.0%}  R@20 {(r <= 20).mean():.0%}  MRR {(1 / r).mean():.3f}  (n={len(r)})"

def walk_forward(feats, step="year", half_life=None, window=None, years=range(2018, 2025)):
    out = []
    keys = sorted(P[P.year.isin(years)].q.unique()) if step == "quarter" else list(years)
    for k in keys:
        te = P[P.q == k] if step == "quarter" else P[P.year == k]
        cut = (lambda X: X.q < k) if step == "quarter" else (lambda X: X.year < k)
        tr = P[cut(P)]
        if window: tr = tr[tr.year >= te.year.min() - window]
        out.append(ranks(te, score(tr, te, feats, half_life)))
    return pd.concat(out)

T = P.year >= 2022
res = {}
res["A. Current: fit once on 2015-21, frozen for 2022-24"] = ranks(P[T], score(P[P.year <= 2021], P[T], BASE))
res["B. Refit every year on all earlier deals"] = walk_forward(BASE)
res["C. Refit every quarter on all earlier deals"] = walk_forward(BASE, step="quarter", years=range(2022, 2025))
res["D. Yearly refit, recent deals weighted (2-yr half-life)"] = walk_forward(BASE, half_life=2)
res["E. Yearly refit, recent deals weighted (4-yr half-life)"] = walk_forward(BASE, half_life=4)
res["F. Yearly refit, last 4 years of deals only"] = walk_forward(BASE, window=4)
res["G. B + market-state signals (lane heat, buyer 4-quarter momentum)"] = walk_forward(BASE + ["lane_heat", "momentum_4q"])
act = []
for y in range(2018, 2025):
    te = P[P.year == y]; act.append(ranks(te, te.activity + 1e-6 * te.recency))
res["Baseline: most active buyer so far"] = pd.concat(act)

ev_year = P.groupby("event").year.first()
print("== 2022-24 test deals (same deals in every row)")
for k, r in res.items():
    r = r[ev_year[r.index] >= 2022]; print(f"  {k:68s} {summ(r)}")
print("\n== R@10 by test year (walk-forward rows only; each year is trained on everything before it)")
hdr = "".join(f"{y:>7}" for y in range(2018, 2025)); print(f"  {'':52s}{hdr}{'  2018-24':>10}")
for k in ["B. Refit every year on all earlier deals", "D. Yearly refit, recent deals weighted (2-yr half-life)", "F. Yearly refit, last 4 years of deals only",
          "G. B + market-state signals (lane heat, buyer 4-quarter momentum)", "Baseline: most active buyer so far"]:
    r = res[k]; yy = ev_year[r.index]
    print(f"  {k[:52]:52s}" + "".join(f"{(r[yy == y] <= 10).mean():7.0%}" for y in range(2018, 2025)) + f"{(r <= 10).mean():10.0%}")
print("  events per year:".ljust(54) + "".join(f"{(ev_year[res['B. Refit every year on all earlier deals'].index] == y).sum():7d}" for y in range(2018, 2025)))

a, b = res["A. Current: fit once on 2015-21, frozen for 2022-24"], res["B. Refit every year on all earlier deals"]
b = b[a.index]; print(f"\n== A vs B on the same 2022-24 deals: B finds the buyer in the top 10 where A misses on {((b <= 10) & (a > 10)).sum()} deals, "
                       f"the reverse on {((a <= 10) & (b > 10)).sum()}")
print("\n== Market state in the data: share of deals won by sponsors over the trailing 8 quarters")
print("  " + ", ".join(f"{y}: {np.mean([heat[q] for q in range(y * 4, y * 4 + 4) if q in heat]):.0%}" for y in range(2017, 2025)))
