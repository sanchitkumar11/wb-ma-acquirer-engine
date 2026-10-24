"""Compare explainable buyer-ranking models on a time split. Usage: python3 -I models.py pairs.pkl"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")

P = pd.read_pickle(sys.argv[1])
FEATS = ["sponsor", "activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
         "margin_fit", "growth_fit", "price_fit", "close_rate", "tag_fit", "bidder_exp", "precedent_sim"]
tr = P[P.year <= 2021].copy(); te = P[P.year >= 2022].copy()
mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
for X in (tr, te): X[FEATS] = (X[FEATS] - mu) / sd
n_test_events = 142  # all 2022-24 deals, including first-time acquirers who can't be ranked

def evaluate(df, score, k=10):
    hits, rr = [], []
    for e, g in df.assign(s=score).groupby("event"):
        if g.label.sum() == 0: continue
        r = (g.s.rank(ascending=False, method="first")[g.label == 1]).iloc[0]
        hits.append(r <= k); rr.append(1 / r)
    return np.sum(hits) / n_test_events, np.sum(hits) / len(hits), np.mean(rr)

base = tr.label.mean(); base_lo = np.log(base / (1 - base))
print(f"train pairs {len(tr)}, test pairs {len(te)}, base rate {base:.4f}\n")
print("== Per-signal logistic regressions: coef, test AUC, KS")
uni = {}
for f in FEATS:
    m = LogisticRegression(C=1e6).fit(tr[[f]], tr.label)
    s = m.decision_function(te[[f]])
    auc = roc_auc_score(te.label, s)
    ks = ks_2samp(s[te.label == 1], s[te.label == 0]).statistic
    uni[f] = (m.coef_[0][0], m.intercept_[0], auc, ks)
    print(f"  {f:14s} coef {m.coef_[0][0]:+.2f}  AUC {auc:.2f}  KS {ks:.2f}")

def woe_score(X, feats):
    return sum(uni[f][0] * X[f] + uni[f][1] - base_lo for f in feats)

print("\n== Ranking results on 2022-24 (recall@10 over all 142 deals | among deals with a known buyer | MRR)")
def show(name, s):
    a, b, c = evaluate(te, s); print(f"  {name:48s} {a:.2f} | {b:.2f} | {c:.3f}")
show("Baseline: most active (deal count)", te.activity)
show("Baseline: recency-weighted activity", te.recency)
show("Naive WoE, all 17 signals", woe_score(te, FEATS))
# greedy forward selection on a validation split inside training (<=2019 fit, 2020-21 validate) (incremental lift)
trA = tr[tr.year <= 2019]; va = tr[tr.year.between(2020, 2021)]
uniA = {f: LogisticRegression(C=1e6).fit(trA[[f]], trA.label) for f in FEATS}
def evA(feats):
    s = sum(uniA[f].decision_function(va[[f]]) for f in feats)
    hits = []
    for e, g in va.assign(s=s).groupby("event"):
        if g.label.sum() == 0: continue
        hits.append((g.s.rank(ascending=False, method="first")[g.label == 1]).iloc[0] <= 10)
    return np.mean(hits)
sel, best = [], 0
while True:
    cand = [(evA(sel + [f]), f) for f in FEATS if f not in sel]
    if not cand: break
    sc, f = max(cand)
    if sc <= best + 0.005: break
    sel.append(f); best = sc
print("  greedy-selected signals:", sel)
show("Naive WoE, greedy-selected signals", woe_score(te, sel))
jl = LogisticRegression(C=1.0, max_iter=2000).fit(tr[FEATS], tr.label)
show("Joint logistic regression (all signals)", jl.decision_function(te[FEATS]))
print("     joint coefs:", {f: round(c, 2) for f, c in zip(FEATS, jl.coef_[0])})
try:
    from statsmodels.discrete.conditional_models import ConditionalLogit
    cl = ConditionalLogit(tr.label.values, tr[FEATS].drop(columns=["sponsor"]).values, groups=tr.event.values).fit(disp=0)
    show("Conditional logit (choice model)", te[FEATS].drop(columns=["sponsor"]).values @ cl.params)
except Exception as ex:
    print("  conditional logit failed:", ex)
import lightgbm as lgb
g_tr = tr.groupby("event").size().values
rk = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1).fit(tr[FEATS], tr.label, group=g_tr)
show("LightGBM LambdaRank (benchmark)", rk.predict(te[FEATS]))

# example explanation for one test deal using naive WoE contributions
e = te[te.label == 1].event.iloc[5]
g = te[te.event == e].copy(); g["s"] = woe_score(g, sel)
top = g.sort_values("s", ascending=False).head(3)
print(f"\n== Example explanation, test event {e}: top 3 candidates and their top-3 signal contributions")
for _, r in top.iterrows():
    contrib = {f: uni[f][0] * r[f] + uni[f][1] - base_lo for f in sel}
    best3 = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
    print(f"  {r.acquirer:22s} label={r.label} total={r.s:+.2f}  ", ", ".join(f"{k} {v:+.2f}" for k, v in best3))
