"""Compute every figure the EDA page shows and write it to one JSON file.

Usage: python3 -I build_data.py <csv> <out.json>
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

CSV, OUT = sys.argv[1], sys.argv[2]
df = pd.read_csv(CSV)
FS, ST = "Financial Sponsor", "Strategic"

RINGS = {
    "Healthcare Services": 0,
    "Physician Groups": 1, "Behavioral Health": 1, "Home Health/Hospice": 1, "Dental": 1,
    "Revenue Cycle": 2, "Health Insurance": 2,
    "Health IT": 3, "Medical Devices": 3, "Pharma/Biotech": 3,
}
RING_W = {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.1}  # illustrative weights, for discussion only
PROV = [s for s, r in RINGS.items() if r <= 1]
df["ring"] = df.sector.map(RINGS)
df["ev_ebitda_recalc"] = df.deal_size_mm / df.target_ebitda_mm
df["margin_recalc"] = df.target_ebitda_mm / df.target_revenue_mm * 100
df["closed"] = df.outcome == "Closed"
df["control"] = df.deal_type != "Minority Investment"
r1 = lambda x: None if pd.isna(x) else round(float(x), 1)
r2 = lambda x: None if pd.isna(x) else round(float(x), 2)

D = {}

# ---------- overview ----------
D["overview"] = dict(
    deals=len(df), acquirers=int(df.acquirer.nunique()),
    sponsors=int(df[df.acquirer_type == FS].acquirer.nunique()),
    strategics=int(df[df.acquirer_type == ST].acquirer.nunique()),
    sponsor_deals=int((df.acquirer_type == FS).sum()), strategic_deals=int((df.acquirer_type == ST).sum()),
    sectors=int(df.sector.nunique()), year_min=int(df.deal_year.min()), year_max=int(df.deal_year.max()),
    closed=int(df.closed.sum()),
)
D["by_year"] = [dict(year=int(y), sponsor=int(((df.deal_year == y) & (df.acquirer_type == FS)).sum()),
                     strategic=int(((df.deal_year == y) & (df.acquirer_type == ST)).sum()))
                for y in sorted(df.deal_year.unique())]
D["outcomes"] = [dict(outcome=k, n=int(v)) for k, v in df.outcome.value_counts().items()]

# ---------- acquirer roster ----------
sector_med_mult = df[df.closed].groupby("sector").ev_ebitda_multiple.median()
df["mult_resid"] = df.ev_ebitda_multiple - df.sector.map(sector_med_mult)

NOTES = {
    "Steward Health Care": "Filed for Chapter 11 in May 2024. The dataset shows it closing deals through 2023.",
    "Kindred at Home": "Acquired by Humana in 2021 and renamed CenterWell Home Health, so it is the same buyer as CenterWell Home Health.",
    "CenterWell Home Health": "Humana's home health business (formerly Kindred at Home). The dataset lists both names separately, including CenterWell deals dated before the brand existed.",
    "Change Healthcare": "Part of UnitedHealth Group's Optum since 2022.",
    "Cerner (Oracle)": "Oracle Health since 2022, so any deal is an Oracle decision.",
    "Ciox Health": "Merged with Datavant in 2021.",
    "Amedisys": "Agreed in 2023 to be acquired by UnitedHealth Group's Optum.",
    "Vista Equity": "A technology-focused investor in reality. The dataset gives it 19 provider-services deals.",
    "Veritas Capital": "Known for government and healthcare technology investing, not provider roll-ups.",
    "Francisco Partners": "A technology-focused investor in reality.",
}

roster = []
for (name, typ), g in df.groupby(["acquirer", "acquirer_type"]):
    gc = g[g.closed]
    sizes = g.deal_size_mm
    roster.append(dict(
        name=name, type="Sponsor" if typ == FS else "Strategic",
        n=len(g), closed=int(g.closed.sum()), close_rate=r2(g.closed.mean()),
        failed=int(g.outcome.isin(["Withdrawn", "Terminated"]).sum()),
        hcs=int((g.sector == "Healthcare Services").sum()), prov=int(g.sector.isin(PROV).sum()),
        relevance=r1(g.ring.map(RING_W).sum()),
        top_sector=g.sector.value_counts().index[0], n_sectors=int(g.sector.nunique()),
        sectors=", ".join(f"{k} {v}" for k, v in g.sector.value_counts().items()),
        med_size=r1(sizes.median()), min_size=r1(sizes.min()), max_size=r1(sizes.max()),
        in_band=int(sizes.between(100, 400).sum()),
        last_year=int(g.deal_year.max()), since_2022=int((g.deal_year >= 2022).sum()),
        med_mult=r1(gc.ev_ebitda_multiple.median()) if len(gc) else None,
        resid=r1(gc.mult_resid.median()) if len(gc) else None,
        private_share=r2((g.target_ownership_pre == "Private").mean()),
        regional_share=r2((~g.geography.isin(["National", "Multi-Regional"])).mean()),
        platform_bolton=int(any(
            ((g.deal_type == "Platform Investment") & (g.sector == s)).any()
            and (g[(g.deal_type == "Bolt-on Acquisition") & (g.sector == s)].deal_year
                 >= g[(g.deal_type == "Platform Investment") & (g.sector == s)].deal_year.min()).any()
            for s in g.sector.unique())),
        tag_platform=r2(g.strategic_rationale_tags.str.contains("Platform Build").mean()),
        tag_geo=r2(g.strategic_rationale_tags.str.contains("Geographic Expansion").mean()),
        tag_cost=r2(g.strategic_rationale_tags.str.contains("Cost Synergies").mean()),
        note=NOTES.get(name),
    ))
roster.sort(key=lambda r: (-r["n"], r["name"]))
D["roster"] = roster
D["strategic_count_dist"] = [dict(deals=int(k), acquirers=int(v)) for k, v in
                             df[df.acquirer_type == ST].acquirer.value_counts().value_counts().sort_index().items()]

# pure relevance ranking, to show what happens with no buyer-mix rule
rk = sorted(roster, key=lambda r: -r["relevance"])
first_st = next(i for i, r in enumerate(rk, 1) if r["type"] == "Strategic")
D["pure_rank"] = dict(top=[dict(rank=i, name=r["name"], type=r["type"], relevance=r["relevance"]) for i, r in enumerate(rk[:20], 1)],
                      first_strategic_rank=first_st, first_strategic=rk[first_st - 1]["name"])

# ---------- deals (compact, for the acquirer detail panel) ----------
D["deals"] = [dict(id=r.transaction_id, a=r.acquirer, s=r.sector, y=int(r.deal_year), t=r.deal_type, g=r.geography,
                   ev=r1(r.deal_size_mm), m=r1(r.ev_ebitda_multiple), mr=r1(r.ev_ebitda_recalc), rv=r2(r.ev_revenue_multiple),
                   mg=r1(r.ebitda_margin_pct), o=r.outcome, own=r.target_ownership_pre, tags=r.strategic_rationale_tags)
              for r in df.itertuples()]

# ---------- sectors ----------
sec = []
for s, g in df.groupby("sector"):
    gc = g[g.closed]
    sec.append(dict(
        sector=s, ring=RINGS[s], n=len(g), sponsor=int((g.acquirer_type == FS).sum()), strategic=int((g.acquirer_type == ST).sum()),
        strategic_buyers=int(g[g.acquirer_type == ST].acquirer.nunique()),
        med_size=r1(g.deal_size_mm.median()), in_band=int(g.deal_size_mm.between(100, 400).sum()),
        m_med=r1(gc.ev_ebitda_multiple.median()), m_q1=r1(gc.ev_ebitda_multiple.quantile(.25)), m_q3=r1(gc.ev_ebitda_multiple.quantile(.75)),
        mr_med=r1(gc.ev_ebitda_recalc.median()), mr_q1=r1(gc.ev_ebitda_recalc.quantile(.25)), mr_q3=r1(gc.ev_ebitda_recalc.quantile(.75)),
        rv_med=r2(gc.ev_revenue_multiple.median()), rv_q1=r2(gc.ev_revenue_multiple.quantile(.25)), rv_q3=r2(gc.ev_revenue_multiple.quantile(.75)),
        margin=r1(g.ebitda_margin_pct.median()), growth=r1(g.revenue_growth_pct.median()),
    ))
# distance from Healthcare Services on standardized sector medians
feat = pd.DataFrame(sec).set_index("sector")[["med_size", "m_med", "rv_med", "margin", "growth"]].astype(float)
feat["med_size"] = np.log(feat["med_size"])
z = (feat - feat.mean()) / feat.std()
dist = np.sqrt(((z - z.loc["Healthcare Services"]) ** 2).sum(axis=1))
for row in sec:
    row["distance"] = r2(dist[row["sector"]])
sec.sort(key=lambda r: (r["ring"], r["distance"]))
D["sectors"] = sec

sp_names = [r["name"] for r in roster if r["type"] == "Sponsor"]
sector_order = [r["sector"] for r in sec]
ct = pd.crosstab(df[df.acquirer_type == FS].acquirer, df[df.acquirer_type == FS].sector).reindex(index=sp_names, columns=sector_order, fill_value=0)
D["heat"] = dict(rows=sp_names, cols=sector_order, values=ct.values.tolist())

# comp pool by ring, $100-400M, closed control deals
pool = []
for k in range(4):
    m = df.ring.le(k) & df.deal_size_mm.between(100, 400)
    pool.append(dict(ring=k, all=int(m.sum()), closed_control=int((m & df.closed & df.control).sum())))
D["pool"] = pool

# ---------- reconciliation ----------
c = df
D["recon"] = dict(
    rev_ok=int(((c.deal_size_mm / c.target_revenue_mm) / c.ev_revenue_multiple - 1).abs().le(0.05).sum()),
    mult_ok=int((c.ev_ebitda_recalc / c.ev_ebitda_multiple - 1).abs().le(0.05).sum()),
    margin_ok=int((c.margin_recalc - c.ebitda_margin_pct).abs().le(1).sum()),
    mult_vs_margin_ok=int(((c.deal_size_mm / c.ev_ebitda_multiple) / c.target_revenue_mm * 100 - c.ebitda_margin_pct).abs().le(1).sum()),
    points=[[r.transaction_id, r1(r.ev_ebitda_multiple), r1(r.ev_ebitda_recalc), r.sector] for r in c.itertuples()],
    max_recalc=r1(c.ev_ebitda_recalc.max()),
)
ratio = (df.deal_size_mm / df.ev_ebitda_multiple) / df.target_ebitda_mm
rho, p = stats.spearmanr(ratio - 1, df.revenue_growth_pct)
D["forward_test"] = dict(q05=r2(ratio.quantile(.05)), q50=r2(ratio.quantile(.5)), q95=r2(ratio.quantile(.95)),
                         below_one=r2((ratio < 1).mean()), rho=r2(rho), p=r2(p))
comp = df[df.sector.isin(PROV) & df.deal_size_mm.between(100, 400) & df.closed & df.control]
D["comp"] = dict(n=len(comp),
                 stated=[r1(comp.ev_ebitda_multiple.quantile(q)) for q in (.25, .5, .75)],
                 recalc=[r1(comp.ev_ebitda_recalc.quantile(q)) for q in (.25, .5, .75)],
                 rev=[r2(comp.ev_revenue_multiple.quantile(q)) for q in (.25, .5, .75)],
                 by_year={int(k): int(v) for k, v in comp.deal_year.value_counts().sort_index().items()})

# ---------- field signal tests ----------
def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2, pv, _, _ = stats.chi2_contingency(t)
    return round(float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))), 2), float(pv)

def fmt_p(pv):
    return "<0.001" if pv < 0.001 else f"{pv:.2f}"

tests = {}
for col in ["deal_type", "financing_type", "geography", "target_ownership_pre", "outcome", "deal_quarter"]:
    v1, p1 = cramers_v(df[col], df.acquirer_type); v2, p2 = cramers_v(df[col], df.sector)
    tests[col] = dict(v_type=v1, p_type=fmt_p(p1), v_sector=v2, p_sector=fmt_p(p2))
v, pv = cramers_v(df.acquirer_type, df.sector); tests["acquirer_type"] = dict(v_sector=v, p_sector=fmt_p(pv))
for col in ["synergy_pct_of_deal", "num_bidders", "days_to_close"]:
    s = df[[col, "ev_ebitda_multiple", "deal_size_mm"]].dropna()
    tests[col] = dict(rho_mult=r2(stats.spearmanr(s[col], s.ev_ebitda_multiple)[0]),
                      rho_size=r2(stats.spearmanr(s[col], s.deal_size_mm)[0]))
sp = df[df.acquirer_type == FS]
cl = df[df.closed]
tests["sponsor_sector_p"] = fmt_p(stats.chi2_contingency(pd.crosstab(sp.acquirer, sp.sector))[1])
tests["sponsor_size_p"] = fmt_p(stats.kruskal(*[np.log(g.deal_size_mm) for _, g in sp.groupby("acquirer")]).pvalue)
tests["sponsor_close_p"] = fmt_p(stats.chi2_contingency(pd.crosstab(sp.acquirer, sp.closed))[1])
tests["sponsor_year_p"] = fmt_p(stats.kruskal(*[g.deal_year for _, g in sp.groupby("acquirer")]).pvalue)
big = cl.groupby("acquirer").filter(lambda x: len(x) >= 4)
tests["acq_resid_p"] = fmt_p(stats.kruskal(*[g.mult_resid for _, g in big.groupby("acquirer")]).pvalue)
tests["sector_mult_p"] = fmt_p(stats.kruskal(*[g.ev_ebitda_multiple for _, g in cl.groupby("sector")]).pvalue)
tests["size_mult_rho"] = r2(stats.spearmanr(cl.deal_size_mm, cl.ev_ebitda_multiple)[0])
tests["year_mult_rho"] = r2(stats.spearmanr(cl.deal_year, cl.ev_ebitda_multiple)[0])
tags = df.strategic_rationale_tags.str.get_dummies("|")
tests["tags"] = [dict(tag=t, n=int(tags[t].sum()),
                      p_sector=fmt_p(stats.chi2_contingency(pd.crosstab(tags[t], df.sector))[1]),
                      p_type=fmt_p(stats.chi2_contingency(pd.crosstab(tags[t], df.acquirer_type))[1]))
                 for t in tags.columns]
tests["scale_min"] = r1(df.loc[tags["Scale"] == 1, "deal_size_mm"].min())
tests["noscale_max"] = r1(df.loc[tags["Scale"] == 0, "deal_size_mm"].max())
tests["carveout_tag_match"] = int(((df.deal_type == "Carve-out") & (tags["Carve-out Complexity"] == 1)).sum())
tests["carveouts"] = int((df.deal_type == "Carve-out").sum())
np_sys = ["Atrium Health", "UPMC", "Ascension Health", "CHRISTUS Health", "Advocate Health", "Bon Secours Mercy Health",
          "Mass General Brigham", "Intermountain Health", "Trinity Health"]
npd = df[df.acquirer.isin(np_sys)]
tests["nonprofit_stock"] = int(npd.financing_type.isin(["Stock", "Cash + Stock"]).sum()); tests["nonprofit_deals"] = len(npd)
tests["sponsor_moe"] = int(((df.acquirer_type == FS) & (df.deal_type == "Merger of Equals")).sum())
tests["sponsor_stock"] = int(((df.acquirer_type == FS) & df.financing_type.isin(["Stock", "Cash + Stock"])).sum())
tests["strategic_lbo"] = int(((df.acquirer_type == ST) & df.deal_type.isin(["Leveraged Buyout", "Management Buyout"])).sum())
D["tests"] = tests

# ---------- textbook claims ----------
cp = cl[cl.sector.isin(PROV)]
hc = cl[cl.sector == "Healthcare Services"]
mwu = lambda a, b: fmt_p(stats.mannwhitneyu(a, b).pvalue)
claims = dict(
    st_vs_fs_all=[r1(cl[cl.acquirer_type == ST].ev_ebitda_multiple.median()), r1(cl[cl.acquirer_type == FS].ev_ebitda_multiple.median())],
    st_vs_fs_prov=[r1(cp[cp.acquirer_type == ST].ev_ebitda_multiple.median()), r1(cp[cp.acquirer_type == FS].ev_ebitda_multiple.median()),
                   mwu(cp[cp.acquirer_type == ST].ev_ebitda_multiple, cp[cp.acquirer_type == FS].ev_ebitda_multiple),
                   int((cp.acquirer_type == ST).sum()), int((cp.acquirer_type == FS).sum())],
    st_vs_fs_hcs=[r1(hc[hc.acquirer_type == ST].ev_ebitda_multiple.median()), r1(hc[hc.acquirer_type == FS].ev_ebitda_multiple.median()),
                  mwu(hc[hc.acquirer_type == ST].ev_ebitda_multiple, hc[hc.acquirer_type == FS].ev_ebitda_multiple),
                  int((hc.acquirer_type == ST).sum()), int((hc.acquirer_type == FS).sum())],
    bidders=[dict(b=int(k), m=r1(v.median()), n=len(v)) for k, v in cp.groupby("num_bidders").ev_ebitda_multiple],
    bidders_rho=r2(stats.spearmanr(cp.num_bidders, cp.ev_ebitda_multiple)[0]),
    margin_mult_rho_prov=r2(stats.spearmanr(cp.ebitda_margin_pct, cp.ev_ebitda_multiple)[0]),
    margin_rev_rho_prov=r2(stats.spearmanr(cp.ebitda_margin_pct, cp.ev_revenue_multiple)[0]),
    margin_rev_p_prov=fmt_p(stats.spearmanr(cp.ebitda_margin_pct, cp.ev_revenue_multiple)[1]),
    margin_mult_rho_all=r2(stats.spearmanr(cl.ebitda_margin_pct, cl.ev_ebitda_multiple)[0]),
    minority_vs_control_prov=[r1(cp[cp.deal_type == "Minority Investment"].ev_ebitda_multiple.median()),
                              r1(cp[cp.deal_type != "Minority Investment"].ev_ebitda_multiple.median())],
    minority_all=r1(cl[cl.deal_type == "Minority Investment"].ev_ebitda_multiple.median()),
    size_buckets=[dict(b=lab, m=r1(g.ev_ebitda_multiple.median()), n=len(g)) for lab, g in
                  cl.groupby(pd.cut(cl.deal_size_mm, [0, 50, 100, 250, 500, 1000, 1e6],
                                    labels=["<$50M", "$50–100M", "$100–250M", "$250–500M", "$0.5–1B", ">$1B"]), observed=True)],
    by_year=[dict(y=int(k), m=r1(v.median()), n=len(v)) for k, v in cl.groupby("deal_year").ev_ebitda_multiple],
    by_year_prov=[dict(y=int(k), m=r1(v.median()), n=len(v)) for k, v in cp.groupby("deal_year").ev_ebitda_multiple],
    synergy=[r1(df[df.acquirer_type == ST].synergy_pct_of_deal.median()), r1(df[df.acquirer_type == FS].synergy_pct_of_deal.median())],
    synergy_mult_rho=r2(stats.spearmanr(df.synergy_pct_of_deal, df.ev_ebitda_multiple)[0]),
    leveraged=[r2((df[df.acquirer_type == FS].financing_type == "Leveraged").mean()), r2((df[df.acquirer_type == ST].financing_type == "Leveraged").mean())],
    allcash=[r2((df[df.acquirer_type == FS].financing_type == "All Cash").mean()), r2((df[df.acquirer_type == ST].financing_type == "All Cash").mean())],
    earnout=[r2((df[df.target_ownership_pre.isin(["Private", "PE-Backed"])].financing_type == "Earnout").mean()),
             r2((df[df.target_ownership_pre == "Public"].financing_type == "Earnout").mean())],
    platform_bolt=[r1(cp[cp.deal_type == "Platform Investment"].ev_ebitda_multiple.median()),
                   r1(cp[cp.deal_type == "Bolt-on Acquisition"].ev_ebitda_multiple.median())],
    target_margin=[r1(df[df.sector.isin(PROV) & (df.acquirer_type == FS)].ebitda_margin_pct.median()),
                   r1(df[df.sector.isin(PROV) & (df.acquirer_type == ST)].ebitda_margin_pct.median())],
    hcs_margin_q=[r1(df[df.sector == "Healthcare Services"].ebitda_margin_pct.quantile(q)) for q in (.5, .75, .9)],
)
D["claims"] = claims

# ---------- buyer mix ----------
band = df[df.sector.isin(PROV) & df.deal_size_mm.between(100, 400)]
bp = band[band.target_ownership_pre == "Private"]
hb = df[(df.sector == "Healthcare Services") & df.deal_size_mm.between(100, 400)]
D["mix"] = dict(band_n=len(band), band_st=int((band.acquirer_type == ST).sum()), band_fs=int((band.acquirer_type == FS).sum()),
                private_n=len(bp), private_st=int((bp.acquirer_type == ST).sum()),
                hcs_n=len(hb), hcs_st=int((hb.acquirer_type == ST).sum()),
                band_st_buyers=sorted(band[band.acquirer_type == ST].acquirer.unique().tolist()))

# ---------- quick backtest probe (naive score vs most-active baseline) ----------
def naive_top(deal, hist, k=10):
    s = {}
    for a, g in hist.groupby("acquirer"):
        s[a] = (g.sector == deal.sector).sum() + 0.3 * len(g) / 10 + np.exp(-abs(np.log(g.deal_size_mm.median() / deal.deal_size_mm)))
    return sorted(s, key=s.get, reverse=True)[:k]

test = df[df.deal_year >= 2022]
hit_naive, hit_active, pool_sizes = [], [], []
for d in test.itertuples():
    hist = df[df.deal_year < d.deal_year]
    hit_naive.append(d.acquirer in naive_top(d, hist))
    hit_active.append(d.acquirer in hist.acquirer.value_counts().head(10).index)
    pool_sizes.append(hist.acquirer.nunique())
D["backtest"] = dict(n=len(test), naive=r2(np.mean(hit_naive)), active=r2(np.mean(hit_active)),
                     random=r2(np.mean([10 / p for p in pool_sizes])))

with open(OUT, "w") as f:
    json.dump(D, f, separators=(",", ":"))
print("wrote", OUT, f"{len(json.dumps(D)) / 1024:.0f} KB")
print("sector order/distance:", [(r["sector"], r["ring"], r["distance"]) for r in sec])
print("pool:", pool, " first strategic in pure rank:", first_st, rk[first_st - 1]["name"])
print("mix:", {k: v for k, v in D["mix"].items() if k != "band_st_buyers"})
print("comp:", D["comp"])
print("forward:", D["forward_test"], " recon:", {k: v for k, v in D["recon"].items() if k != "points"})
print("platform_bolton sponsors:", [r["name"] for r in roster if r["platform_bolton"] and r["type"] == "Sponsor"])
print("tests:", {k: v for k, v in tests.items() if k != "tags"})
