"""Worked examples for the log: one test deal scored end to end (signs locked), double counting, banker priors. Usage: python3 -I example.py pairs.pkl <csv>"""
import sys, warnings, numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
P = pd.read_pickle(sys.argv[1]); D = pd.read_csv(sys.argv[2])
D["t"] = D.deal_year * 10 + D.deal_quarter.str[1].astype(int); D = D.sort_values(["t", "transaction_id"]).reset_index(drop=True)
F = ["activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
     "margin_fit", "growth_fit", "close_rate", "bidder_exp", "precedent_sim"]
PRIOR = {0: dict(activity=.3, recency=.3, sector_share=1.0, sector_rel=.5, size_fit=.5, size_in_range=.3, geo_fit=.5, own_fit=.2, type_fit=.2,
                 margin_fit=.2, growth_fit=.2, close_rate=.3, bidder_exp=0, precedent_sim=.5),
         1: dict(activity=.6, recency=.6, sector_share=.4, sector_rel=.6, size_fit=.6, size_in_range=.4, geo_fit=.2, own_fit=.4, type_fit=.5,
                 margin_fit=.3, growth_fit=.2, close_rate=.3, bidder_exp=.2, precedent_sim=.4)}
def fit_map(X, y, b0, lam, w0):
    Xa = np.column_stack([np.ones(len(X)), X]); pr = np.concatenate([[w0], b0]); pen = np.r_[0.0, np.full(len(b0), lam)]
    def f(w):
        z = Xa @ w; p = 1 / (1 + np.exp(-z)); return -(y * z - np.logaddexp(0, z)).sum() + .5 * (pen * (w - pr) ** 2).sum(), Xa.T @ (p - y) + pen * (w - pr)
    lock = lam > 1e-3 and lam < 1e3
    return minimize(f, np.clip(pr, 0, None) if lock else pr.copy(), jac=True, method="L-BFGS-B", bounds=([(None, None)] + [(0, None)] * len(b0)) if lock else None).x
tr = P[P.year <= 2021]; te = P[P.year >= 2022].copy()
mu, sd = tr[F].mean(), tr[F].std().replace(0, 1)
W = {}
for lane in (0, 1):
    a = tr[tr.sponsor == lane]; br = a.label.mean()
    W[lane] = fit_map(((a[F] - mu) / sd).values, a.label.values, np.array([PRIOR[lane][f] for f in F]), 30.0, np.log(br / (1 - br)))
Z = (te[F] - mu) / sd
te["logit"] = [W[l][0] + z @ W[l][1:] for l, z in zip(te.sponsor, Z.values)]
te["p"] = 1 / (1 + np.exp(-te.logit))
te["share"] = te.p / te.groupby("event").p.transform("sum")
te["rank"] = te.groupby("event").logit.rank(ascending=False, method="first")
prov = {"Healthcare Services", "Physician Groups", "Behavioral Health", "Home Health/Hospice", "Dental"}
win = te[te.label == 1].merge(D[["target_company", "sector", "deal_size_mm", "geography", "deal_type", "acquirer_type", "ebitda_margin_pct", "revenue_growth_pct", "target_ownership_pre", "deal_year", "deal_quarter"]], left_on="event", right_index=True)
cand = win[win.sector.isin(prov)].sort_values("rank")
print(cand[["event", "target_company", "sector", "deal_size_mm", "geography", "deal_type", "acquirer", "acquirer_type", "rank", "share", "deal_year"]].head(25).to_string())
print("\nprovider deals: n", len(cand), "top10", (cand["rank"] <= 10).mean().round(2))

NAMES = dict(activity="number of past deals", recency="deals in recent years", sector_share="share of its deals in this sub-sector",
             sector_rel="deals in this or adjacent sub-sectors", size_fit="deal size vs its usual size", size_in_range="size inside its past range",
             geo_fit="has bought in this region", own_fit="has bought this ownership type", type_fit="does control deals of private companies",
             margin_fit="margin vs what it usually buys", growth_fit="growth vs what it usually buys", close_rate="share of its deals that closed",
             bidder_exp="used to competitive auctions", precedent_sim="close precedents (sector, size, region, margin)")
def show(ev):
    d = D.loc[ev]; g = te[te.event == ev].sort_values("rank")
    print(f"\n=== {d.target_company}: {d.sector}, ${d.deal_size_mm}M, {d.geography}, {d.deal_type}, margin {d.ebitda_margin_pct}%, growth {d.revenue_growth_pct}%, "
          f"{d.target_ownership_pre}, {d.deal_year} {d.deal_quarter}. Actual buyer: {d.acquirer} ({d.acquirer_type}). Field: {len(g)} buyers")
    for r in g.head(6).itertuples(): print(f"   #{int(r.rank)} {r.acquirer:28s} {'Sponsor' if r.sponsor else 'Strategic':9s} share {r.share:.1%}  {'<- actual' if r.label else ''}")
    w = g[g.label == 1].iloc[0]; lane = int(w.sponsor); z = ((w[F] - mu) / sd).astype(float); c = pd.Series(W[lane][1:] * z.values, index=F)
    hist = D[(D.t < d.t) & (D.acquirer == d.acquirer)]
    print(f"   {d.acquirer} history before this deal: {len(hist)} deals; sub-sectors {hist.sector.value_counts().to_dict()}; sizes {sorted(hist.deal_size_mm.round(0).tolist())}; regions {hist.geography.value_counts().to_dict()}")
    print(f"   raw signals: " + ", ".join(f"{f}={w[f]:.2f}" for f in F))
    print(f"   intercept {W[lane][0]:+.2f}; total logit {w.logit:+.2f}; contributions vs average candidate (log-odds, x odds):")
    for f, v in c.sort_values(ascending=False).items(): print(f"      {v:+.2f} (x{np.exp(v):.2f})  {NAMES[f]}  [z={z[f]:+.2f}, weight {W[lane][1 + F.index(f)]:+.2f}]")
show(379)
miss = cand[cand["rank"] > 20].sort_values("deal_size_mm", ascending=False)
print("\nmisses:", miss[["event", "target_company", "sector", "deal_size_mm", "acquirer", "acquirer_type", "rank"]].head(6).to_string())

# double counting: three sector signals fitted one at a time vs together (strategic lane)
a = tr[tr.sponsor == 0]; X = (a[F] - mu) / sd; y = a.label.values; br = a.label.mean()
print("\n=== double counting, strategic lane (log-odds per 1 SD)")
for f in ["sector_share", "sector_rel", "precedent_sim"]:
    w1 = fit_map(X[[f]].values, y, np.zeros(1), 1e-6, np.log(br / (1 - br))); print(f"   {f}: alone {w1[1]:+.2f}", end="")
    print()
wj = fit_map(X[["sector_share", "sector_rel", "precedent_sim"]].values, y, np.zeros(3), 1e-6, np.log(br / (1 - br))); print("   together:", np.round(wj[1:], 2), "sum", round(wj[1:].sum(), 2))
print("   corr:", X[["sector_share", "sector_rel", "precedent_sim"]].corr().round(2).values.tolist())

# priors: banker weight -> data-only -> lam 30
print("\n=== priors (strategic / sponsor)")
for lane in (0, 1):
    a = tr[tr.sponsor == lane]; X = ((a[F] - mu) / sd).values; br = a.label.mean()
    w0 = fit_map(X, a.label.values, np.zeros(len(F)), 1.0, np.log(br / (1 - br)))
    for f in ["sector_share", "activity", "recency", "size_fit", "geo_fit", "type_fit", "precedent_sim", "bidder_exp"]:
        i = F.index(f); print(f"   lane {lane} {f:14s} banker {PRIOR[lane][f]:+.2f}  data-only {w0[1+i]:+.2f}  blended(lam30) {W[lane][1+i]:+.2f}")
