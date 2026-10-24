"""Leak-free benchmark table (tag match and price match removed). Train 2015-21, test on 127 warm 2022-24 deals.
Usage: python3 -I bench.py pairs.pkl"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
warnings.filterwarnings("ignore")
P = pd.read_pickle(sys.argv[1])
F = ["activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
     "margin_fit", "growth_fit", "close_rate", "bidder_exp", "precedent_sim"]
PRIOR = {0: dict(activity=.3, recency=.3, sector_share=1.0, sector_rel=.5, size_fit=.5, size_in_range=.3, geo_fit=.5, own_fit=.2, type_fit=.2,
                 margin_fit=.2, growth_fit=.2, close_rate=.3, bidder_exp=0, precedent_sim=.5),
         1: dict(activity=.6, recency=.6, sector_share=.4, sector_rel=.6, size_fit=.6, size_in_range=.4, geo_fit=.2, own_fit=.4, type_fit=.5,
                 margin_fit=.3, growth_fit=.2, close_rate=.3, bidder_exp=.2, precedent_sim=.4)}
tr = P[P.year <= 2021].copy(); te = P[P.year >= 2022].copy()
mu, sd = tr[F].mean(), tr[F].std().replace(0, 1)
Ztr, Zte = (tr[F] - mu) / sd, (te[F] - mu) / sd

def ranks(df, s):
    out = []
    for e, g in df.assign(s=np.asarray(s)).groupby("event"):
        if g.label.sum(): out.append(int(g.s.rank(ascending=False, method="first")[g.label == 1].iloc[0]))
    return np.array(out)
def row(name, s):
    r = ranks(te, s); print(f"  {name:62s} top10 {(r <= 10).mean():4.0%}  top20 {(r <= 20).mean():4.0%}  top25 {(r <= 25).mean():4.0%}  MRR {(1 / r).mean():.3f}")
    return r

def fit_map(X, y, b0, lam, w0, lock=False):
    Xa = np.column_stack([np.ones(len(X)), X]); pr = np.concatenate([[w0], b0]); pen = np.r_[0.0, np.full(len(b0), lam)]
    def f(w):
        z = Xa @ w; p = 1 / (1 + np.exp(-z)); return -(y * z - np.logaddexp(0, z)).sum() + .5 * (pen * (w - pr) ** 2).sum(), Xa.T @ (p - y) + pen * (w - pr)
    bounds = [(None, None)] + [(0, None)] * len(b0) if lock else None
    return minimize(f, np.clip(pr, 0, None) if lock else pr.copy(), jac=True, method="L-BFGS-B", bounds=bounds).x
def lane(lam, use_prior=True, lock=False, show_w=False):
    s = np.zeros(len(te)); W = {}
    for flag in (0, 1):
        m_tr, m_te = (tr.sponsor == flag).values, (te.sponsor == flag).values
        b0 = np.array([PRIOR[flag][f] for f in F]) * (1 if use_prior else 0); br = tr.label[m_tr].mean()
        w = fit_map(Ztr[m_tr].values, tr.label[m_tr].values, b0, lam, np.log(br / (1 - br)), lock)
        s[m_te] = w[0] + Zte[m_te].values @ w[1:]; W[flag] = w
    if show_w:
        for flag, nm in ((0, "Strategic"), (1, "Sponsor")):
            print(f"     {nm} weights: " + ", ".join(f"{f} {v:+.2f}" for f, v in zip(F, W[flag][1:])))
    return s

print("== 127 warm test deals, 2022-24")
row("Most active buyer so far", te.activity + 1e-6 * te.recency)
base = tr.label.mean(); blo = np.log(base / (1 - base)); woe = np.zeros(len(te))
for f in F:
    m = LogisticRegression(C=1e6).fit(Ztr[[f]], tr.label); woe += m.decision_function(Zte[[f]]) - blo
row("Per-signal additive (one model per signal, log-odds summed)", woe)
X2tr, X2te = Ztr.assign(sponsor=tr.sponsor.values), Zte.assign(sponsor=te.sponsor.values)
row("Joint logistic regression (all signals, one model)", LogisticRegression(C=1.0, max_iter=3000).fit(X2tr, tr.label).decision_function(X2te))
row("Lane-specific logistic, data only", lane(1.0, use_prior=False))
row("Lane-specific + banker priors, lam=30", lane(30.0))
row("Lane-specific + banker priors, lam=30, signs locked", lane(30.0, lock=True, show_w=True))
row("Lane-specific + banker priors, lam=3000 (~banker weights)", lane(3000.0))
g_tr = tr.groupby("event", sort=False).size().values
tr_sorted = tr.sort_values("event"); X3tr = ((tr_sorted[F] - mu) / sd).assign(sponsor=tr_sorted.sponsor.values)
g_tr = tr_sorted.groupby("event").size().values
rs = []
for seed in range(5):
    m = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=seed, verbose=-1).fit(X3tr, tr_sorted.label, group=g_tr)
    r = ranks(te, m.predict(X2te)); rs.append(((r <= 10).mean(), (r <= 20).mean(), (r <= 25).mean(), (1 / r).mean()))
rs = np.array(rs); print(f"  {'LightGBM ranker, 5 seeds (min-max)':62s} top10 {rs[:,0].min():.0%}-{rs[:,0].max():.0%}  top20 {rs[:,1].min():.0%}-{rs[:,1].max():.0%}  top25 {rs[:,2].min():.0%}-{rs[:,2].max():.0%}  MRR {rs[:,3].min():.3f}-{rs[:,3].max():.3f}")
D = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
