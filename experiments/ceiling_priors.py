"""Accuracy ceiling, recall@k curve, and banker-prior (MAP) logistic models. Usage: python3 -I ceiling_priors.py pairs.pkl <csv>"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
P = pd.read_pickle(sys.argv[1]); D = pd.read_csv(sys.argv[2])
D["t"] = D.deal_year * 10 + D.deal_quarter.str[1].astype(int); D = D.sort_values(["t", "transaction_id"]).reset_index(drop=True)
P["deal_sector"] = P.event.map(D.sector); P["win_lane"] = P.event.map((D.acquirer_type == "Financial Sponsor").astype(int))
FEATS = ["activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
         "margin_fit", "growth_fit", "close_rate", "bidder_exp", "precedent_sim"]
tr = P[P.year <= 2021].copy(); te = P[P.year >= 2022].copy()
mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
for X in (tr, te): X[FEATS] = (X[FEATS] - mu) / sd

def ranks(df, score):
    out = []
    for e, g in df.assign(s=score).groupby("event"):
        if g.label.sum() == 0: continue
        out.append(int((g.s.rank(ascending=False, method="first")[g.label == 1]).iloc[0]))
    return np.array(out)
def summary(r, ks=(5, 10, 15, 20, 25, 30)):
    return {f"R@{k}": round((r <= k).mean(), 2) for k in ks} | {"MRR": round((1 / r).mean(), 3), "median rank": int(np.median(r))}

print("== Ceiling checks on the 127 test deals with a known buyer")
# Oracle 1: told the winner's lane and the deal's sub-sector; ranks that lane's buyers with history in that sub-sector first, by activity
o1 = te.sponsor.eq(te.win_lane) * 100 + (te.sector_share > te.sector_share.min()) * 10 * te.sponsor.eq(te.win_lane) + te.activity
print("  Oracle told winner's lane + sub-sector:", summary(ranks(te, o1)))
# Oracle 2: told the lane only
o2 = te.sponsor.eq(te.win_lane) * 100 + te.activity
print("  Oracle told winner's lane only:       ", summary(ranks(te, o2)))
spon = te[te.win_lane == 1].event.nunique(); print(f"  sponsor-won test deals: {spon} of {te.event.nunique()}; active sponsors per deal ~{te[te.sponsor == 1].groupby('event').size().median():.0f}")
print("  Baseline most active:                 ", summary(ranks(te, te.activity)))

def fit_map(X, y, b0, lam, w_int0):
    Xa = np.column_stack([np.ones(len(X)), X]); prior = np.concatenate([[w_int0], b0])
    pen = np.r_[0.0, np.full(len(b0), lam)]
    def f(w):
        z = Xa @ w; p = 1 / (1 + np.exp(-z))
        nll = -(y * z - np.logaddexp(0, z)).sum()
        return nll + 0.5 * (pen * (w - prior) ** 2).sum(), Xa.T @ (p - y) + pen * (w - prior)
    return minimize(f, prior.copy(), jac=True, method="L-BFGS-B").x
PRIOR = {  # banker starting weights, log-odds per 1 standard deviation of each signal
    "Strategic": dict(activity=.3, recency=.3, sector_share=1.0, sector_rel=.5, size_fit=.5, size_in_range=.3, geo_fit=.5, own_fit=.2, type_fit=.2,
                      margin_fit=.2, growth_fit=.2, price_fit=.3, close_rate=.3, tag_fit=.4, bidder_exp=0, precedent_sim=.5),
    "Sponsor": dict(activity=.6, recency=.6, sector_share=.4, sector_rel=.6, size_fit=.6, size_in_range=.4, geo_fit=.2, own_fit=.4, type_fit=.5,
                    margin_fit=.3, growth_fit=.2, price_fit=.3, close_rate=.3, tag_fit=.3, bidder_exp=.2, precedent_sim=.4)}
def lane_model(lam, data_only=False):
    s = np.zeros(len(te)); W = {}
    for lane, flag in (("Strategic", 0), ("Sponsor", 1)):
        a, b = tr[tr.sponsor == flag], te[te.sponsor == flag]
        b0 = np.array([PRIOR[lane][f] for f in FEATS]) * (0 if data_only else 1)
        br = a.label.mean(); w = fit_map(a[FEATS].values, a.label.values, b0, lam, np.log(br / (1 - br)))
        s[(te.sponsor == flag).values] = w[0] + b[FEATS].values @ w[1:]; W[lane] = dict(zip(["intercept"] + FEATS, np.round(w, 2)))
    return s, W
print("\n== Banker priors: how much the data is allowed to move the weights (lam = prior strength)")
for name, lam, do in [("Data only (weak ridge toward 0)", 1.0, True), ("Priors, lam=1 (data dominates)", 1.0, False), ("Priors, lam=30", 30.0, False),
                      ("Priors, lam=300", 300.0, False), ("Priors, lam=3000 (~banker weights only)", 3000.0, False)]:
    s, W = lane_model(lam, do); print(f"  {name:42s}", summary(ranks(te, s)))
    if lam == 30.0 and not do: W30 = W
print("\n== Weights at lam=30 vs banker starting weights (selected signals)")
for lane in ("Strategic", "Sponsor"):
    print(f"  {lane}: " + ", ".join(f"{f} {PRIOR[lane][f]:+.1f}->{W30[lane][f]:+.2f}" for f in ["sector_share", "sector_rel", "precedent_sim", "activity", "size_fit", "geo_fit", "type_fit", "own_fit", "bidder_exp"]))
