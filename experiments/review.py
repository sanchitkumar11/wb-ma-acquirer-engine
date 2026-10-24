"""Banker-logic fixes and checks.
1) Model-level fixes suggested by banker logic: thin-history shrinkage, one-sided size test, sponsor 'open platform' signal.
2) Sponsor separation measured directly: rank of the winning sponsor among sponsors only.
3) In-file banker checks: how often would each check have thrown out the real buyer (back-test)?
4) Live target: model top 25, then checks (in-file + dated real-world notes) and the rule-based review.
Usage: python3 -I review.py <csv>"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
D = pd.read_csv(sys.argv[1])
D["t"] = D.deal_year * 10 + D.deal_quarter.str[1].astype(int); D = D.sort_values(["t", "transaction_id"]).reset_index(drop=True)
D["q"] = D.deal_year * 4 + D.deal_quarter.str[1].astype(int) - 1; D["lsize"] = np.log(D.deal_size_mm)
GROUP = {"Healthcare Services": "prov", "Physician Groups": "prov", "Behavioral Health": "prov", "Home Health/Hospice": "prov", "Dental": "prov",
         "Revenue Cycle": "rcm", "Health Insurance": "payer", "Health IT": "tech", "Medical Devices": "prod", "Pharma/Biotech": "prod"}
ADJ = {frozenset(p) for p in [("prov", "rcm"), ("prov", "payer"), ("rcm", "tech"), ("prod", "tech")]}
def sec_sim(a, b):
    if a == b: return 1.0
    if GROUP[a] == GROUP[b]: return 0.6
    return 0.3 if frozenset((GROUP[a], GROUP[b])) in ADJ else 0.1
def ring(a, b):
    """Adjacency ring of sub-sector a relative to target sub-sector b: 0 core, 1 same business model, 2 adjacent, 3 distant."""
    if a == b: return 0
    if GROUP[a] == GROUP[b]: return 1
    return 2 if frozenset((GROUP[a], GROUP[b])) in ADJ else 3
RING_NAME = {0: "Core", 1: "Ring 1", 2: "Ring 2", 3: "Ring 3"}
RADJ = {frozenset(p) for p in [("Northeast", "Mid-Atlantic"), ("Mid-Atlantic", "Southeast"), ("Southeast", "Southwest"), ("Midwest", "Great Plains"),
        ("Mountain West", "West Coast"), ("Mountain West", "Southwest"), ("Mountain West", "Great Plains"), ("Midwest", "Northeast")]}
def geo_sim(a, b):
    if a == b: return 1.0
    return 0.5 if ("National" in (a, b) or "Multi-Regional" in (a, b) or frozenset((a, b)) in RADJ) else 0.0
CONTROL = {"Platform Investment", "Bolt-on Acquisition", "Leveraged Buyout", "Recapitalization", "Management Buyout", "Strategic Acquisition"}
PLATFORM = {"Platform Investment", "Leveraged Buyout", "Recapitalization", "Management Buyout", "Carve-out"}
PROV = {k for k, v in GROUP.items() if v == "prov"}

def features(d, hist):
    """All signals for one deal (or the live target) against every buyer active in hist. d needs sector, lsize, geography,
    ebitda_margin_pct, revenue_growth_pct, target_ownership_pre, deal_year, q."""
    rows = []
    for a, g in hist.groupby("acquirer"):
        n = len(g); ssim = np.array([sec_sim(s, d["sector"]) for s in g.sector]); gsim = np.array([geo_sim(x, d["geography"]) for x in g.geography])
        rec = 0.5 ** ((d["deal_year"] - g.deal_year.values) / 3)
        size_close = np.exp(-np.abs(g.lsize.values - d["lsize"])); marg_close = np.exp(-np.abs(g.ebitda_margin_pct.values - d["ebitda_margin_pct"]) / 5)
        recent_plat = g[(g.q >= d["q"] - 20) & g.deal_type.isin(PLATFORM)]
        rows.append(dict(acquirer=a, sponsor=int(g.acquirer_type.iloc[0] == "Financial Sponsor"), n=n,
            activity=np.log1p(n), recency=rec.sum(), sector_share=(g.sector.values == d["sector"]).mean(), sector_rel=(ssim * rec).sum(),
            size_fit=-abs(d["lsize"] - g.lsize.mean()), size_in_range=float(g.lsize.min() - .25 <= d["lsize"] <= g.lsize.max() + .25),
            geo_fit=(gsim * rec).sum() / rec.sum(), own_fit=(g.target_ownership_pre.values == d["target_ownership_pre"]).mean(),
            type_fit=g.deal_type.isin(CONTROL).mean(), margin_fit=-abs(d["ebitda_margin_pct"] - g.ebitda_margin_pct.median()),
            growth_fit=-abs(d["revenue_growth_pct"] - g.revenue_growth_pct.median()), close_rate=(g.outcome == "Closed").mean(),
            bidder_exp=g.num_bidders.mean(), precedent_sim=(ssim * (0.4 + 0.6 * size_close) * (0.6 + 0.4 * gsim) * (0.7 + 0.3 * marg_close) * rec).sum(),
            # banker-logic signals
            not_bigger=-max(0.0, d["lsize"] - g.lsize.max()),        # how far the deal is above the biggest deal it has done (capacity)
            not_smaller=-max(0.0, g.lsize.min() - d["lsize"]),       # how far below its smallest deal (tuck-ins are usually fine)
            open_platform=float((recent_plat.sector == d["sector"]).any()),           # sponsor bought a platform in this sub-sector in the last 5 years
            open_platform_prov=float(recent_plat.sector.isin(PROV).any() and d["sector"] in PROV),
            n_closed=(g.outcome == "Closed").sum(), n_control=g.deal_type.isin(CONTROL).sum(), n_own=(g.target_ownership_pre == d["target_ownership_pre"]).sum(),
            max_size=g.deal_size_mm.max(), med_size=g.deal_size_mm.median(), n_prov=g.sector.isin(PROV).sum(),
            closest_ring=min(ring(s, d["sector"]) for s in g.sector), home_sector=g.sector.mode()[0]))
    X = pd.DataFrame(rows)
    # thin-history shrinkage: a buyer with one deal is pulled toward its lane's average (2 pseudo-deals)
    for raw, cnt in (("close_rate", "n_closed"), ("type_fit", "n_control"), ("own_fit", "n_own")):
        lane_mean = X.groupby("sponsor")[raw].transform("mean")
        X[raw + "_s"] = (X[cnt] + 2 * lane_mean) / (X.n + 2)
    return X

rows = []
for i, d in D.iterrows():
    hist = D[D.t < d.t]
    if hist.empty: continue
    X = features(d, hist); X["event"] = i; X["year"] = d.deal_year; X["label"] = (X.acquirer == d.acquirer).astype(int); rows.append(X)
P = pd.concat(rows, ignore_index=True)
import os; os.makedirs("out", exist_ok=True); P.to_pickle("out/pairs_v2.pkl")

BASE = ["activity", "recency", "sector_share", "sector_rel", "size_fit", "size_in_range", "geo_fit", "own_fit", "type_fit",
        "margin_fit", "growth_fit", "close_rate", "bidder_exp", "precedent_sim"]
PRI = {0: dict(activity=.3, recency=.3, sector_share=1.0, sector_rel=.5, size_fit=.5, size_in_range=.3, geo_fit=.5, own_fit=.2, type_fit=.2, margin_fit=.2,
               growth_fit=.2, close_rate=.3, bidder_exp=0, precedent_sim=.5, not_bigger=.6, not_smaller=.1, open_platform=0, open_platform_prov=0),
       1: dict(activity=.6, recency=.6, sector_share=.4, sector_rel=.6, size_fit=.6, size_in_range=.4, geo_fit=.2, own_fit=.4, type_fit=.5, margin_fit=.3,
               growth_fit=.2, close_rate=.3, bidder_exp=.2, precedent_sim=.4, not_bigger=.6, not_smaller=.3, open_platform=.5, open_platform_prov=.3)}
for k in (0, 1):
    for f in ("close_rate", "type_fit", "own_fit"): PRI[k][f + "_s"] = PRI[k][f]
def fit(X, y, b0, lam=30.0, w0=0.0):
    Xa = np.column_stack([np.ones(len(X)), X]); pr = np.concatenate([[w0], b0]); pen = np.r_[0.0, np.full(len(b0), lam)]
    def f(w):
        z = Xa @ w; p = 1 / (1 + np.exp(-z)); return -(y * z - np.logaddexp(0, z)).sum() + .5 * (pen * (w - pr) ** 2).sum(), Xa.T @ (p - y) + pen * (w - pr)
    return minimize(f, np.clip(pr, 0, None), jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * len(b0)).x   # signs locked
class Model:
    def __init__(self, tr, feats):
        self.F = feats; self.mu, self.sd = tr[feats].mean(), tr[feats].std().replace(0, 1); self.W = {}
        for k in (0, 1):
            a = tr[tr.sponsor == k]; br = a.label.mean()
            self.W[k] = fit(((a[feats] - self.mu) / self.sd).values, a.label.values, np.array([PRI[k][f] for f in feats]), w0=np.log(br / (1 - br)))
    def score(self, X):
        Z = ((X[self.F] - self.mu) / self.sd).values; s = np.zeros(len(X))
        for k in (0, 1):
            m = (X.sponsor == k).values; s[m] = self.W[k][0] + Z[m] @ self.W[k][1:]
        return s
def ranks(df, s, only_sponsors=False):
    out = {}
    for e, g in df.assign(s=s).groupby("event"):
        if g.label.sum() == 0: continue
        if only_sponsors:
            if g[g.label == 1].sponsor.iloc[0] != 1: continue
            g = g[g.sponsor == 1]
        out[e] = int(g.s.rank(ascending=False, method="first")[g.label == 1].iloc[0])
    return pd.Series(out)

tr, te = P[P.year <= 2021], P[P.year >= 2022]
asc = D.index[D.target_company == "Premier Medical Management"][0]; hea = D.index[D.target_company == "Madison Dental Solutions"][0]
swap = lambda feats, old, new: [new.get(f, f) for f in feats if f not in old or f in new] if False else feats
SHR = {"close_rate": "close_rate_s", "type_fit": "type_fit_s", "own_fit": "own_fit_s"}
variants = {
    "Current model (signs locked)": BASE,
    "+ thin-history shrinkage": [SHR.get(f, f) for f in BASE],
    "+ one-sided size test (capacity)": [f for f in BASE if f not in ("size_fit", "size_in_range")] + ["not_bigger", "not_smaller"],
    "+ sponsor open-platform signal": BASE + ["open_platform", "open_platform_prov"],
}
variants["All three"] = [SHR.get(f, f) for f in variants["+ one-sided size test (capacity)"]] + ["open_platform", "open_platform_prov"]
print("== Back-test, 127 warm 2022-24 deals. Sponsor columns: winning sponsor's rank among the 15 sponsors only (sponsor-won deals)")
print(f"  {'':36s} top10 top20 top25   MRR | sponsor top3  sponsor mean rank | Ascension  Heartland")
M = {}
for name, feats in variants.items():
    m = Model(tr, feats); s = m.score(te); r = ranks(te, s); rs = ranks(te, s, only_sponsors=True); M[name] = (m, feats)
    print(f"  {name:36s} {(r<=10).mean():5.0%} {(r<=20).mean():5.0%} {(r<=25).mean():5.0%} {(1/r).mean():5.3f} | {(rs<=3).mean():11.0%} {rs.mean():17.1f} | {r.get(asc):9d} {r.get(hea):10d}")
print(f"  {'Random order among sponsors':36s} {'':23s} | {3/15:11.0%} {8.0:17.1f}")

# in-file banker checks: how often would each have thrown out the actual buyer?
best_name = "All three"; m, feats = M[best_name]; te = te.assign(s=m.score(te))
win = te[te.label == 1].merge(D[["deal_size_mm", "sector"]], left_on="event", right_index=True)
checks = {
    "No closed deal on record": win.n_closed == 0,
    "Strategic whose biggest deal is under half this deal's size": (win.sponsor == 0) & (win.max_size < 0.5 * win.deal_size_mm),
    "Sponsor whose median deal is >3x or <1/3 this size": (win.sponsor == 1) & ((win.med_size > 3 * win.deal_size_mm) | (win.med_size < win.deal_size_mm / 3)),
    "Buyer with no deal in the target's own sub-sector (core-only rule)": win.closest_ring > 0,
}
print(f"\n== In-file banker checks on the actual buyers of the 127 test deals ('{best_name}' model): share of real buyers each check would have thrown out")
for k, v in checks.items(): print(f"  {k:66s} {v.mean():5.0%}  ({v.sum()} deals)")
# where do real buyers come from, by adjacency ring? (closest ring among the buyer's earlier deals)
A = P[P.label == 1]
print("\n== Where real buyers came from: closest earlier deal to the target's sub-sector (all deals by repeat buyers)")
for nm, sub in (("All", A), ("Sponsors", A[A.sponsor == 1]), ("Strategics", A[A.sponsor == 0])):
    c = sub.closest_ring.value_counts().reindex(range(4), fill_value=0)
    print(f"  {nm:10s} n={len(sub):3d}  " + "  ".join(f"{RING_NAME[k]} {v:3d} ({v / len(sub):3.0%})" for k, v in c.items()))

# live target, scored as of the end of the data with the model refit on every deal
TGT = dict(sector="Healthcare Services", lsize=np.log(200), geography="Midwest", ebitda_margin_pct=16.8, revenue_growth_pct=11.0,
           target_ownership_pre="Private", deal_year=2025, q=2025 * 4)
mall = Model(P, feats); T = features(TGT, D); T["s"] = mall.score(T); T["p"] = 1 / (1 + np.exp(-T.s)); T["share"] = T.p / T.p.sum()
T = T.sort_values("s", ascending=False).reset_index(drop=True); T["model_rank"] = T.index + 1
# Real-world status, mergers and mandates come from the dated, sourced buyer profiles (data/profiles/),
# not from lists in this script. The live target is reviewed with every fact known today.
from acquirer_engine.profiles import load_profiles, profile_notes
PROFILES = load_profiles(Path(__file__).resolve().parents[1] / "data/profiles")
SEVERITY = {"STATUS_DEFUNCT": "FAIL", "STATUS_ACQUIRED": "FAIL", "ENTITY_DUPLICATE": "FAIL", "MANDATE_MISMATCH": "FLAG"}
def check(r):
    out = [(SEVERITY[code], note) for code, note, _ in profile_notes(r.acquirer, PROFILES, TGT["sector"], set(T.acquirer))]
    if r.n_closed == 0: out.append(("FLAG", "No closed deal on record"))
    # sector rationale by ring: no ring excludes a buyer; outer rings need more evidence and carry a risk flag
    # (strategics only: sponsors cross rings routinely, so for them the ring shapes the rationale, not eligibility)
    if r.sponsor == 0 and r.closest_ring == 2: out.append(("NOTE", f"Ring 2 (vertical move from {r.home_sector}): needs a stated strategy; integration risk flag"))
    if r.sponsor == 0 and r.closest_ring == 3: out.append(("FLAG", f"Ring 3 (from {r.home_sector}): Tier 2 unless a cited, dated services strategy"))
    if r.sponsor == 0 and r.max_size < 100: out.append(("FLAG", f"Biggest deal ${r.max_size:.0f}M: $200M would be a stretch"))
    if r.sponsor == 1 and not (200 / 3 <= r.med_size <= 600): out.append(("FLAG", f"Median deal ${r.med_size:.0f}M: off its usual size"))
    return out
print("\n== Live target: Healthcare Services, $200M, Midwest, 16.8% margin (75th percentile), 11% growth, private; deal data through Q4 2024")
def line(r):
    c = check(r); tag = "FAIL" if any(x[0] == "FAIL" for x in c) else ("FLAG" if any(x[0] == "FLAG" for x in c) else "PASS")
    extra = f"{RING_NAME[r.closest_ring]}; " + (f"open platform in sub-sector: {'yes' if r.open_platform else 'no'}; provider deals {int(r.n_prov)}/{int(r.n)}; median deal ${r.med_size:.0f}M" if r.sponsor else f"provider deals {int(r.n_prov)}/{int(r.n)}; biggest deal ${r.max_size:.0f}M")
    return f"  #{r.model_rank:2d} {r.acquirer:26s} {'Sponsor' if r.sponsor else 'Strategic':9s} {r.share:5.1%}  {tag}  {'; '.join(x[1] for x in c)}  [{extra}]"
for r in T.head(25).itertuples(): print(line(r))
print("\n== Adjacency coverage: best-ranked buyers from each outer ring (shown in Tier 2 so the banker sees they were considered)")
for k in (1, 2, 3):
    for r in T[(T.closest_ring == k) & (T.sponsor == 0)].head(3).itertuples(): print(line(r))
# rules-only Tier 1: drop FAILs (defunct, acquired, duplicate of a listed buyer), send mandate FLAGs to Tier 2, take the next 10
def codes(r): return {code for code, _, _ in profile_notes(r.acquirer, PROFILES, TGT["sector"], set(T.acquirer))}
tier1 = [r.acquirer for r in T.itertuples() if not codes(r) & {"STATUS_DEFUNCT", "STATUS_ACQUIRED", "ENTITY_DUPLICATE", "MANDATE_MISMATCH"}][:10]
print("\n== Rules-only Tier 1 (from profile facts):", ", ".join(tier1))
T.to_pickle("out/live_target_scores.pkl")
