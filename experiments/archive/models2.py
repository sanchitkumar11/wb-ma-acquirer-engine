"""Glass-box variants: additive boosted model with banker-direction constraints, lane interactions, provider subset. Usage: python3 -I models2.py pairs.pkl <csv>"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
warnings.filterwarnings("ignore")
P = pd.read_pickle(sys.argv[1]); D = pd.read_csv(sys.argv[2])
D["t"] = D.deal_year * 10 + D.deal_quarter.str[1].astype(int)
D = D.sort_values(["t", "transaction_id"]).reset_index(drop=True)
P["deal_sector"] = P.event.map(D.sector)
PROV = {"Healthcare Services", "Physician Groups", "Behavioral Health", "Home Health/Hospice", "Dental"}
FEATS = ["sponsor", "activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
         "margin_fit", "growth_fit", "price_fit", "close_rate", "tag_fit", "bidder_exp", "precedent_sim"]
MONO = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
tr = P[P.year <= 2021].copy(); te = P[P.year >= 2022].copy()
n_all = te.event.nunique() + (142 - te.event.nunique())

def evaluate(df, score, k=10, subset=None):
    df = df.assign(s=score)
    if subset is not None: df = df[df.deal_sector.isin(subset)]
    hits, rr, ev = [], [], df.event.nunique()
    for e, g in df.groupby("event"):
        if g.label.sum() == 0: continue
        r = (g.s.rank(ascending=False, method="first")[g.label == 1]).iloc[0]
        hits.append(r <= k); rr.append(1 / r)
    return len(hits), np.mean(hits), np.mean(rr)

def show(name, s):
    n1, a, m = evaluate(te, s); n2, b, m2 = evaluate(te, s, subset=PROV)
    print(f"  {name:52s} all-warm n={n1} R@10 {a:.2f} MRR {m:.3f} | provider n={n2} R@10 {b:.2f} MRR {m2:.3f}")

show("Baseline: most active", te.activity)
show("Baseline: sector share x activity", te.sector_share * 3 + te.activity)
mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
Xtr = (tr[FEATS] - mu) / sd; Xte = (te[FEATS] - mu) / sd
jl = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, tr.label); show("Joint logistic regression", jl.decision_function(Xte))
# lane interactions: separate logistic per lane
s = np.zeros(len(te))
for lane in (0, 1):
    m = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr[tr.sponsor == lane], tr.label[tr.sponsor == lane])
    s[te.sponsor.values == lane] = m.decision_function(Xte[te.sponsor == lane]) + (np.log(tr.label[tr.sponsor == lane].mean()) if False else 0)
# put lanes on a common scale: add each lane's log base rate
for lane in (0, 1):
    br = tr.label[tr.sponsor == lane].mean(); s[te.sponsor.values == lane] += 0
show("Lane-specific logistic regressions", s)
g_tr = tr.groupby("event").size().values
add = lgb.LGBMRanker(n_estimators=400, learning_rate=0.03, max_depth=1, num_leaves=2, min_child_samples=30, monotone_constraints=MONO, verbose=-1)
add.fit(tr[FEATS], tr.label, group=g_tr); show("Additive boosted model (stumps) + banker directions", add.predict(te[FEATS]))
full = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, monotone_constraints=MONO, verbose=-1)
full.fit(tr[FEATS], tr.label, group=g_tr); show("LightGBM ranker + banker directions", full.predict(te[FEATS]))
free = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1)
free.fit(tr[FEATS], tr.label, group=g_tr); show("LightGBM ranker, unconstrained", free.predict(te[FEATS]))
# seed stability of the unconstrained ranker
rs = []
for seed in range(5):
    m = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1).fit(tr[FEATS], tr.label, group=g_tr)
    rs.append(evaluate(te, m.predict(te[FEATS]))[1])
print("  unconstrained ranker R@10 across 5 seeds:", np.round(rs, 2))
# per-feature importance for the additive model (sum of |contrib|)
contrib = add.predict(te[FEATS], pred_contrib=True)[:, :-1]
imp = pd.Series(np.abs(contrib).mean(0), index=FEATS).sort_values(ascending=False).round(3)
print("\n== Additive model: average |contribution| per signal\n", imp.to_string())
# example: provider test deal, top 3 + contributions
e = te[(te.label == 1) & te.deal_sector.isin(PROV)].event.iloc[3]
g = te[te.event == e].copy(); g["s"] = add.predict(g[FEATS]); c = add.predict(g[FEATS], pred_contrib=True)[:, :-1]
g = g.reset_index(drop=True); order = g.s.sort_values(ascending=False).index[:4]
true = g[g.label == 1].index[0]
print(f"\n== Example: deal {D.loc[e,'transaction_id']} ({D.loc[e,'sector']}, ${D.loc[e,'deal_size_mm']}M, {D.loc[e,'geography']}), actual buyer {D.loc[e,'acquirer']} ranked #{int((g.s > g.s[true]).sum()) + 1} of {len(g)}")
for i in list(order) + ([true] if true not in order else []):
    top = sorted(zip(FEATS, c[i]), key=lambda kv: -abs(kv[1]))[:3]
    print(f"  {g.acquirer[i]:24s} score {g.s[i]:+.2f}  " + ", ".join(f"{k} {v:+.2f}" for k, v in top))
