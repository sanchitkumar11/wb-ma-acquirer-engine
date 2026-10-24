"""Score all 107 acquirers for the take-home target with a pooled logistic model (lane interactions). Usage: python3 -I target_try.py pairs.pkl <csv>"""
import sys, importlib.util, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
P = pd.read_pickle(sys.argv[1]); df = pd.read_csv(sys.argv[2])
spec = importlib.util.spec_from_file_location("pairs_helpers", __file__.replace("archive/target_try.py", "build_pairs.py"))
src = open(spec.origin).read().split("rows = []")[0]  # reuse helper defs only
ns = {}; exec(src.replace("sys.argv[1]", repr(sys.argv[2])), ns)
D = ns["df"]; sec_sim, geo_sim, TYPES = ns["sec_sim"], ns["geo_sim"], ns["CONTROL_PRIVATE_TYPES"]
FEATS = ["activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
         "margin_fit", "growth_fit", "price_fit", "close_rate", "tag_fit", "bidder_exp", "precedent_sim"]
def target_pairs(t):
    rows = []
    for a, g in D.groupby("acquirer"):
        ssim = np.array([sec_sim(s, t["sector"]) for s in g.sector]); gsim = np.array([geo_sim(x, t["geo"]) for x in g.geography])
        size_close = np.exp(-np.abs(g.lsize.values - np.log(t["ev"]))); marg_close = np.exp(-np.abs(g.ebitda_margin_pct.values - t["margin"]) / 5)
        rec = 0.5 ** ((2025 - g.deal_year.values) / 3)
        rows.append(dict(acquirer=a, sponsor=int(g.acquirer_type.iloc[0] == "Financial Sponsor"), activity=np.log1p(len(g)), recency=rec.sum(),
            sector_share=(g.sector.values == t["sector"]).mean(), sector_rel=(ssim * rec).sum(), size_fit=-abs(np.log(t["ev"]) - g.lsize.mean()),
            size_in_range=float(g.lsize.min() - .25 <= np.log(t["ev"]) <= g.lsize.max() + .25), geo_fit=(gsim * rec).sum() / rec.sum(),
            own_fit=(g.target_ownership_pre == t["own"]).mean(), type_fit=g.deal_type.isin(TYPES).mean(),
            margin_fit=-abs(t["margin"] - g.ebitda_margin_pct.median()), growth_fit=-abs(t["growth"] - g.revenue_growth_pct.median()),
            price_fit=-abs(t["mult"] - g.ev_ebitda_multiple.median()), close_rate=(g.outcome == "Closed").mean(),
            tag_fit=np.mean([len(x & t["tags"]) / len(x | t["tags"]) for x in g.tags]), bidder_exp=g.num_bidders.mean(),
            precedent_sim=(ssim * (0.4 + 0.6 * size_close) * (0.6 + 0.4 * gsim) * (0.7 + 0.3 * marg_close) * rec).sum()))
    return pd.DataFrame(rows)
def design(X, mu, sd):
    Z = (X[FEATS] - mu) / sd
    for f in FEATS: Z[f + "_x_sp"] = Z[f] * X["sponsor"]
    Z["sponsor"] = X["sponsor"]; return Z
mu, sd = P[FEATS].mean(), P[FEATS].std().replace(0, 1)
m = LogisticRegression(C=1.0, max_iter=4000).fit(design(P, mu, sd), P.label)
for geo in ["Southeast", "Midwest"]:
    t = dict(sector="Healthcare Services", ev=200, margin=20, growth=11, mult=12.5, own="Private", geo=geo,
             tags={"Geographic Expansion", "Platform Build", "Market Share", "Cost Synergies"})
    T = target_pairs(t); T["logit"] = m.decision_function(design(T, mu, sd)); T["p"] = 1 / (1 + np.exp(-T.logit))
    T["share"] = T.p / T.p.sum()
    top = T.sort_values("p", ascending=False).head(14)
    print(f"\n== Target: HC Services, $200M, {geo} -> top 14 (share of probability among 107)")
    for i, r in enumerate(top.itertuples(), 1):
        print(f"  {i:2d}. {r.acquirer:26s} {'Sponsor' if r.sponsor else 'Strategic':9s} {r.share:.1%}")
    print("  strategics in top 10:", int((T.sort_values('p', ascending=False).head(10).sponsor == 0).sum()))
