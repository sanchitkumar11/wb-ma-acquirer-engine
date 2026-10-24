"""Exploratory signal tests for the M&A CSV. Prints results; no side effects."""
import sys
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(sys.argv[1])
FS, ST = "Financial Sponsor", "Strategic"
PROV = ["Healthcare Services", "Physician Groups", "Behavioral Health", "Home Health/Hospice", "Dental"]


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(t)
    n = t.values.sum()
    v = np.sqrt(chi2 / (n * (min(t.shape) - 1)))
    return round(v, 3), p


print("== 1. Forward / adjusted multiple hypotheses")
df["ebitda_implied"] = df.deal_size_mm / df.ev_ebitda_multiple
df["ratio"] = df.ebitda_implied / df.target_ebitda_mm
print("implied/reported EBITDA ratio quantiles", df.ratio.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())
print("share ratio in [1.0,1.4] (plausible 1-yr forward):", ((df.ratio >= 1.0) & (df.ratio <= 1.4)).mean().round(3))
print("share ratio < 1 (forward EBITDA BELOW LTM):", (df.ratio < 1).mean().round(3))
print("spearman(ratio-1, revenue_growth):", stats.spearmanr(df.ratio - 1, df.revenue_growth_pct))
df["gap_pct_ev"] = (df.ebitda_implied - df.target_ebitda_mm) / df.deal_size_mm * 100
print("spearman(EBITDA gap as % EV, synergy_pct):", stats.spearmanr(df.gap_pct_ev, df.synergy_pct_of_deal))
df["implied_margin"] = df.ebitda_implied / df.target_revenue_mm * 100
print("spearman(implied margin from stated multiple, stated margin):", stats.spearmanr(df.implied_margin, df.ebitda_margin_pct))
print("stated margin within 1pt of implied-from-multiple:", ((df.implied_margin - df.ebitda_margin_pct).abs() <= 1).sum())
print("computed margin within 1pt of stated:", (((df.target_ebitda_mm / df.target_revenue_mm * 100) - df.ebitda_margin_pct).abs() <= 1).sum())
print("EV/Rev exact tie (2dp):", ((df.deal_size_mm / df.target_revenue_mm).round(2).sub(df.ev_revenue_multiple).abs() <= 0.011).sum())

print("\n== 2. Categorical fields vs acquirer type / sector (Cramer's V, p)")
for c in ["deal_type", "financing_type", "geography", "target_ownership_pre", "deal_quarter", "outcome"]:
    print(f"{c:22s} vs acquirer_type", cramers_v(df[c], df.acquirer_type), " vs sector", cramers_v(df[c], df.sector))
print("acquirer_type vs sector", cramers_v(df.acquirer_type, df.sector))

print("\n== 3. Implausible combinations")
systems = ["Atrium Health", "UPMC", "Steward Health Care", "Ascension Health", "Ardent Health Services", "CHRISTUS Health",
           "Advocate Health", "Bon Secours Mercy Health", "HCA Healthcare", "Mass General Brigham", "Intermountain Health", "Trinity Health"]
nonprofit = [s for s in systems if s not in ("Steward Health Care", "Ardent Health Services", "HCA Healthcare")]
hs = df[df.acquirer.isin(systems)]
print("health-system deals:", len(hs), "deal types:", hs.deal_type.value_counts().to_dict())
npf = df[df.acquirer.isin(nonprofit)]
print("non-profit system deals paid with Stock / Cash+Stock:", npf.financing_type.isin(["Stock", "Cash + Stock"]).sum(), "of", len(npf))
print("sponsor deals paid with Stock / Cash+Stock:", df[(df.acquirer_type == FS) & df.financing_type.isin(["Stock", "Cash + Stock"])].shape[0], "of", (df.acquirer_type == FS).sum())
print("sponsor 'Merger of Equals':", ((df.acquirer_type == FS) & (df.deal_type == "Merger of Equals")).sum())
print("strategic 'Leveraged Buyout'/'Management Buyout':", ((df.acquirer_type == ST) & df.deal_type.isin(["Leveraged Buyout", "Management Buyout"])).sum())
print("Public targets <$50M:", ((df.target_ownership_pre == "Public") & (df.deal_size_mm < 50)).sum())
print("Carve-out deal_type with Carve-out Complexity tag:", ((df.deal_type == "Carve-out") & df.strategic_rationale_tags.str.contains("Carve-out Complexity")).sum(),
      "/ carve-outs", (df.deal_type == "Carve-out").sum(), "/ tag total", df.strategic_rationale_tags.str.contains("Carve-out Complexity").sum())

print("\n== 4. Numeric fields: do they relate to anything?")
num = ["synergy_pct_of_deal", "num_bidders", "days_to_close"]
for c in num:
    for o in ["ev_ebitda_multiple", "deal_size_mm", "ebitda_margin_pct", "revenue_growth_pct", "deal_year"]:
        s = df[[c, o]].dropna()
        r, p = stats.spearmanr(s[c], s[o])
        if abs(r) > 0.1 or p < 0.05:
            print(f"  {c} ~ {o}: rho={r:.2f} p={p:.3f}")
    print(c, "by acquirer type medians:", df.groupby("acquirer_type")[c].median().to_dict(),
          " MWU p=", round(stats.mannwhitneyu(df[df.acquirer_type == FS][c].dropna(), df[df.acquirer_type == ST][c].dropna()).pvalue, 3))
print("days_to_close by deal_type medians:", df.groupby("deal_type").days_to_close.median().round(0).to_dict())

print("\n== 5. Tags")
tags = df.strategic_rationale_tags.str.get_dummies("|")
for t in tags.columns:
    a = df.loc[tags[t] == 1, "deal_size_mm"].median(); b = df.loc[tags[t] == 0, "deal_size_mm"].median()
    fs = (df.loc[tags[t] == 1, "acquirer_type"] == FS).mean()
    print(f"  {t:22s} n={tags[t].sum():3d} median size with/without {a:7.1f}/{b:7.1f}  sponsor share {fs:.2f}")
print("Scale tag min size:", df.loc[tags["Scale"] == 1, "deal_size_mm"].min(), " max size without Scale:", df.loc[tags["Scale"] == 0, "deal_size_mm"].max())
print("tag count vs size spearman:", stats.spearmanr(tags.sum(axis=1), df.deal_size_mm))

print("\n== 6. Sponsor sector preferences (chi-square vs pooled sponsor mix)")
sp = df[df.acquirer_type == FS]
t = pd.crosstab(sp.acquirer, sp.sector)
chi2, p, dof, _ = stats.chi2_contingency(t)
print("sponsor x sector chi2 p=", p, " Cramer V", cramers_v(sp.acquirer, sp.sector))
# permutation: max share in one sector per sponsor vs shuffled
obs = (t.max(axis=1) / t.sum(axis=1)).mean()
rng = np.random.default_rng(0); sims = []
for _ in range(2000):
    s2 = sp.assign(sector=rng.permutation(sp.sector.values))
    t2 = pd.crosstab(s2.acquirer, s2.sector); sims.append((t2.max(axis=1) / t2.sum(axis=1)).mean())
print("mean top-sector share obs", round(obs, 3), "perm mean", round(np.mean(sims), 3), "perm p", np.mean(np.array(sims) >= obs))

print("\n== 7. Strategic geography concentration (strategics with >=3 deals)")
st = df[df.acquirer_type == ST]
g = st.groupby("acquirer").filter(lambda x: len(x) >= 3)
obsg = g.groupby("acquirer").geography.agg(lambda s: s.value_counts().iloc[0] / len(s)).mean()
sims = []
for _ in range(2000):
    g2 = g.assign(geography=rng.permutation(g.geography.values))
    sims.append(g2.groupby("acquirer").geography.agg(lambda s: s.value_counts().iloc[0] / len(s)).mean())
print("mean top-geo share obs", round(obsg, 3), "perm mean", round(np.mean(sims), 3), "perm p", np.mean(np.array(sims) >= obsg))
for a in ["Atrium Health", "UPMC", "Steward Health Care", "CenterWell Home Health", "VNS Health"]:
    print("  ", a, df[df.acquirer == a].geography.tolist())

print("\n== 8. Sector drives multiples? Size? Year?")
c = df[df.outcome == "Closed"]
print("Kruskal sector->EV/EBITDA", stats.kruskal(*[g.ev_ebitda_multiple for _, g in c.groupby("sector")]))
print("spearman size~EV/EBITDA", stats.spearmanr(c.deal_size_mm, c.ev_ebitda_multiple))
print("spearman year~EV/EBITDA", stats.spearmanr(c.deal_year, c.ev_ebitda_multiple))
cp = c[c.sector.isin(PROV)]
print("provider: spearman margin~EV/EBITDA", stats.spearmanr(cp.ebitda_margin_pct, cp.ev_ebitda_multiple),
      " margin~EV/Rev", stats.spearmanr(cp.ebitda_margin_pct, cp.ev_revenue_multiple))
print("provider: spearman growth~EV/EBITDA", stats.spearmanr(cp.revenue_growth_pct, cp.ev_ebitda_multiple))
print("provider: spearman bidders~EV/EBITDA", stats.spearmanr(cp.num_bidders, cp.ev_ebitda_multiple))
print("all closed: spearman margin~EV/EBITDA", stats.spearmanr(c.ebitda_margin_pct, c.ev_ebitda_multiple))
print("provider by type", cp.groupby("acquirer_type").ev_ebitda_multiple.agg(["median", "count"]).round(1).to_dict(),
      " MWU p", round(stats.mannwhitneyu(cp[cp.acquirer_type == FS].ev_ebitda_multiple, cp[cp.acquirer_type == ST].ev_ebitda_multiple).pvalue, 3))
h = c[c.sector == "Healthcare Services"]
print("HCS by type MWU p", round(stats.mannwhitneyu(h[h.acquirer_type == FS].ev_ebitda_multiple, h[h.acquirer_type == ST].ev_ebitda_multiple).pvalue, 3))
print("minority vs control (provider closed):", cp.groupby(cp.deal_type == "Minority Investment").ev_ebitda_multiple.median().to_dict())
print("platform vs bolt-on (provider closed):", cp[cp.deal_type.isin(["Platform Investment", "Bolt-on Acquisition"])].groupby("deal_type").ev_ebitda_multiple.agg(["median", "count"]).to_dict())
print("earnout by ownership share:", df.groupby("target_ownership_pre").financing_type.apply(lambda s: (s == "Earnout").mean()).round(3).to_dict())
print("leveraged financing share by acquirer type:", df.groupby("acquirer_type").financing_type.apply(lambda s: (s == "Leveraged").mean()).round(3).to_dict())
print("sponsor vs strategic target margin (provider):", df[df.sector.isin(PROV)].groupby("acquirer_type").ebitda_margin_pct.median().to_dict())
print("withdrawn/terminated multiples vs closed (provider):", df[df.sector.isin(PROV)].groupby("outcome").ev_ebitda_multiple.median().round(1).to_dict())
print("closed comp counts provider 100-400M control by year:", c[c.sector.isin(PROV) & c.deal_size_mm.between(100, 400) & (c.deal_type != "Minority Investment")].deal_year.value_counts().sort_index().to_dict())
