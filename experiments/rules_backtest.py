"""Back-test the banker review rules before they become gates: how often would each have pushed the
real buyer out of Tier 1 (or off the list)? Uses the package's review() on the 127 test deals
(2022-24), with the model fitted on 2015-21 and only profile facts dated before each deal.
Usage: python3 -I experiments/rules_backtest.py <csv>   (run experiments/review.py first: it writes out/pairs_v2.pkl)"""
import sys, warnings
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
from acquirer_engine.config import Target
from acquirer_engine.data import load_deals
from acquirer_engine.model import RankingModel
from acquirer_engine.profiles import facts_as_of, load_profiles
from acquirer_engine.review import review
from acquirer_engine.rings import model_sectors
warnings.filterwarnings("ignore")

D = load_deals(Path(sys.argv[1])); P = pd.read_pickle("out/pairs_v2.pkl"); PROFILES = load_profiles()
model = RankingModel.fit(P[P.year <= 2021])
te = P[P.year >= 2022].copy(); te["s"] = model.score(te)

def deal_target(d):
    prev = d.q - 1                                        # as_of = the quarter before the deal, so scoring_quarter() == deal quarter
    return Target(d.sector, d.deal_size_mm, d.geography, d.target_ownership_pre, d.ebitda_margin_pct,
                  d.revenue_growth_pct, f"{prev // 4}Q{prev % 4 + 1}")

rows = []
for e, g in te.groupby("event"):
    if g.label.sum() == 0: continue
    d = D.loc[e]; ranked = g.sort_values("s", ascending=False).reset_index(drop=True); ranked["model_rank"] = ranked.index + 1
    # the package's review reads n_model (earlier deals in the deal's business model); the pairs file predates it
    earlier = D[D.t < d.t]; same_model = earlier[earlier.sector.isin(model_sectors(d.sector))].acquirer.value_counts()
    ranked["n_model"] = ranked.acquirer.map(same_model).fillna(0)
    winner = d.acquirer; hist = D[D.t < d.t]
    known = facts_as_of(PROFILES, date(d.deal_year, 3 * int(d.deal_quarter[1]) - 2, 1))
    out = {"event": e, "sponsor_won": int(d.acquirer_type == "Financial Sponsor"), "model_rank": int(ranked.loc[ranked.acquirer == winner, "model_rank"].iloc[0])}
    for arm, kw in (("model top 10", None), ("rules, no angle rule", dict(angle_rule=False)), ("rules + angle rule", dict(angle_rule=True))):
        if kw is None:
            t1 = list(ranked.acquirer[:10]); t2 = list(ranked.acquirer[10:25]); moved = None
        else:
            rv = review(ranked, hist, deal_target(d), known, **kw); t1, t2 = rv.tier1, rv.tier2
            moved = next((m.code for m in rv.log if m.buyer == winner), None)
        out[arm] = "Tier 1" if winner in t1 else ("Tier 2" if winner in t2 else "off list")
        out[arm + " move"] = moved
    rows.append(out)
R = pd.DataFrame(rows)
print(f"== {len(R)} test deals (2022-24) with a known buyer; sponsor-won: {R.sponsor_won.sum()}")
for arm in ("model top 10", "rules, no angle rule", "rules + angle rule"):
    t1 = (R[arm] == "Tier 1").mean(); t12 = (R[arm] != "off list").mean()
    print(f"  {arm:22s} real buyer in Tier 1 {t1:4.0%}   in Tier 1 or 2 {t12:4.0%}")
for code in ("SPONSOR_NO_ANGLE", "DISTANT_SECTOR", "MANDATE_MISMATCH", "STATUS_DEFUNCT", "ENTITY_DUPLICATE", "STATUS_ACQUIRED"):
    n = (R["rules + angle rule move"] == code).sum(); print(f"  real buyers moved by {code:18s} {n:3d} ({n / len(R):.0%})")
w = R[R.sponsor_won == 1]
print(f"  sponsor-won deals: winner in Tier 1 without angle rule {(w['rules, no angle rule'] == 'Tier 1').mean():.0%}, with it {(w['rules + angle rule'] == 'Tier 1').mean():.0%}")
