"""Build as-of (deal, candidate acquirer) pairs with banker-logic signals. Usage: python3 -I pairs.py <csv> <out.parquet-or-pkl>"""
import sys
import numpy as np, pandas as pd

df = pd.read_csv(sys.argv[1])
df["t"] = df.deal_year * 10 + df.deal_quarter.str[1].astype(int)
df = df.sort_values(["t", "transaction_id"]).reset_index(drop=True)
df["tags"] = df.strategic_rationale_tags.str.split("|").apply(set)
df["lsize"] = np.log(df.deal_size_mm)

GROUP = {"Healthcare Services": "prov", "Physician Groups": "prov", "Behavioral Health": "prov", "Home Health/Hospice": "prov",
         "Dental": "prov", "Revenue Cycle": "rcm", "Health Insurance": "payer", "Health IT": "tech", "Medical Devices": "prod", "Pharma/Biotech": "prod"}
ADJ = {frozenset(p) for p in [("prov", "rcm"), ("prov", "payer"), ("rcm", "tech"), ("prod", "tech")]}
def sec_sim(a, b):
    if a == b: return 1.0
    ga, gb = GROUP[a], GROUP[b]
    if ga == gb: return 0.6
    if frozenset((ga, gb)) in ADJ: return 0.3
    return 0.1
REGION_ADJ = {frozenset(p) for p in [("Northeast", "Mid-Atlantic"), ("Mid-Atlantic", "Southeast"), ("Southeast", "Southwest"), ("Midwest", "Great Plains"),
              ("Mountain West", "West Coast"), ("Mountain West", "Southwest"), ("Mountain West", "Great Plains"), ("Midwest", "Northeast")]}
def geo_sim(a, b):
    if a == b: return 1.0
    if "National" in (a, b) or "Multi-Regional" in (a, b): return 0.5
    if frozenset((a, b)) in REGION_ADJ: return 0.5
    return 0.0
CONTROL_PRIVATE_TYPES = {"Platform Investment", "Bolt-on Acquisition", "Leveraged Buyout", "Recapitalization", "Management Buyout", "Strategic Acquisition"}

rows = []
for i, d in df.iterrows():
    hist = df[df.t < d.t]
    if hist.empty: continue
    for a, g in hist.groupby("acquirer"):
        n = len(g)
        ssim = np.array([sec_sim(s, d.sector) for s in g.sector])
        gsim = np.array([geo_sim(x, d.geography) for x in g.geography])
        size_close = np.exp(-np.abs(g.lsize.values - d.lsize))
        marg_close = np.exp(-np.abs(g.ebitda_margin_pct.values - d.ebitda_margin_pct) / 5)
        grow_close = np.exp(-np.abs(g.revenue_growth_pct.values - d.revenue_growth_pct) / 5)
        own = (g.target_ownership_pre.values == d.target_ownership_pre).astype(float)
        tagj = np.array([len(t & d.tags) / len(t | d.tags) for t in g.tags])
        rec = 0.5 ** ((d.deal_year - g.deal_year.values) / 3)
        # precedent similarity: how close each past deal is to this one, recency weighted
        deal_sim = ssim * (0.4 + 0.6 * size_close) * (0.6 + 0.4 * gsim) * (0.7 + 0.3 * marg_close) * rec
        rows.append(dict(
            event=i, year=d.deal_year, acquirer=a, label=int(a == d.acquirer), sponsor=int(g.acquirer_type.iloc[0] == "Financial Sponsor"),
            activity=np.log1p(n), recency=rec.sum(),
            sector_share=(g.sector.values == d.sector).mean(), sector_rel=(ssim * rec).sum(),
            size_fit=-abs(d.lsize - g.lsize.mean()), size_in_range=float(g.lsize.min() - 0.25 <= d.lsize <= g.lsize.max() + 0.25),
            geo_fit=(gsim * rec).sum() / rec.sum(), own_fit=own.mean(),
            type_fit=g.deal_type.isin(CONTROL_PRIVATE_TYPES).mean(),
            margin_fit=-abs(d.ebitda_margin_pct - g.ebitda_margin_pct.median()), growth_fit=-abs(d.revenue_growth_pct - g.revenue_growth_pct.median()),
            price_fit=-abs(d.ev_ebitda_multiple - g.ev_ebitda_multiple.median()),
            close_rate=(g.outcome == "Closed").mean(), tag_fit=tagj.mean(),
            bidder_exp=g.num_bidders.mean(), precedent_sim=deal_sim.sum(),
        ))
P = pd.DataFrame(rows)
P.to_pickle(sys.argv[2])
print(P.shape, "positives", P.label.sum(), "events", P.event.nunique())
