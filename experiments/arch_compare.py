"""Architecture comparison, trimmed (experiment log Section 7).

Part A, how the list is built: A1 (model + rules) against A2 (A1 + an LLM reviewer).
  - Live target, 3 runs each: Tier 1 stability (overlap between runs), moves, cost, time.
  - Back-test on 30 frozen 2022-24 deals: the model is fitted on 2015-21, each deal sees only
    earlier deals and profile facts dated before it. Real buyer in Tier 1 and in Tier 1 + 2, real
    buyers moved out of Tier 1 by the stage, and the share of Tier 1 names passing the banker
    checks (no severe flag: no closed deal, capacity stretch, fund concentration, vertical move).
Part B, how the one-pagers are written: W2 (one call per buyer) against W3 (W2 + peer lines + one
  repair), on the live target, 3 runs each: validator pass rate, specificity (cited facts per
  bullet), thesis overlap, generic phrases, cost, time. W3 run 1 is the committed live run.

Usage:
  python3 -I experiments/arch_compare.py --dry           # sizes and cost estimate, no calls
  python3 -I experiments/arch_compare.py --part A --allow-paid
  python3 -I experiments/arch_compare.py --part B --allow-paid
Results go to out/arch_compare_<part>.json; replays from the cache once run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from acquirer_engine.config import Target, load_target
from acquirer_engine.data import load_deals
from acquirer_engine.engine import EngineResult, run
from acquirer_engine.factpack import SEVERE_FLAGS, build_fact_pack
from acquirer_engine.graph import write_onepagers
from acquirer_engine.listbuild import PROMPTS, _card, a2_reviewer
from acquirer_engine.llm import LEDGER_PATH, cost_usd
from acquirer_engine.metrics import summarise
from acquirer_engine.model import RankingModel
from acquirer_engine.onepager import CITATION, OnePager
from acquirer_engine.profiles import facts_as_of, load_profiles
from acquirer_engine.report import priority_order
from acquirer_engine.review import review
from acquirer_engine.signals import buyer_signals

FROZEN = ROOT / "experiments/frozen_30.json"
# Part A ran on Oct 23 with the profiles of that commit; new profiles change the reviewer's evidence
# cards (and so its cache keys), so Part A always replays with this frozen copy
PROFILES_STEP4 = ROOT / "experiments/results/profiles_step4"
RUNS = 3
OUTPUT_TOKENS_A2 = 3000   # estimate per reviewer reply, hidden reasoning included


def ledger_lines() -> int:
    return len(LEDGER_PATH.read_text().splitlines()) if LEDGER_PATH.exists() else 0


def spent_since(n: int) -> float:
    return sum(json.loads(x)["usd"] for x in LEDGER_PATH.read_text().splitlines()[n:]) if LEDGER_PATH.exists() else 0.0


def overlap(lists: list[list[str]]) -> float:
    """Return the mean share of Tier 1 names two runs have in common (1.0 = identical lists)."""
    pairs = list(itertools.combinations(lists, 2))
    return round(float(np.mean([len(set(a) & set(b)) / 10 for a, b in pairs])), 2) if pairs else 1.0


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------


def deal_target(d) -> Target:
    """Return the deal as a target profile, scored in its own quarter (as experiments/rules_backtest.py)."""
    prev = d.q - 1
    return Target(d.sector, d.deal_size_mm, d.geography, d.target_ownership_pre, d.ebitda_margin_pct,
                  d.revenue_growth_pct, f"{prev // 4}Q{prev % 4 + 1}")


def frozen_deals(deals: pd.DataFrame) -> list[str]:
    """Return 30 test deals (2022-24, buyer seen before), spread evenly by date; frozen in a file on first use."""
    if FROZEN.exists():
        return json.loads(FROZEN.read_text())
    seen, ids = set(), []
    for d in deals.sort_values("t").itertuples():
        if d.deal_year >= 2022 and d.acquirer in seen:
            ids.append(d.transaction_id)
        seen.add(d.acquirer)
    picks = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, 30)]
    FROZEN.write_text(json.dumps(picks, indent=1))
    return picks


def backtest_result(d, deals: pd.DataFrame, model: RankingModel, profiles) -> EngineResult:
    """Return an EngineResult for one historical deal: earlier deals only, facts dated before it."""
    target, hist = deal_target(d), deals[deals.t < d.t]
    ranked = buyer_signals(target.signal_inputs(), hist)
    ranked["s"] = model.score(ranked)
    ranked["p"] = 1 / (1 + np.exp(-ranked.s))
    ranked["share"] = ranked.p / ranked.p.sum()
    ranked = ranked.sort_values("s", ascending=False).reset_index(drop=True)
    ranked["model_rank"] = ranked.index + 1
    known = facts_as_of(profiles, date(d.deal_year, 3 * int(d.deal_quarter[1]) - 2, 1))
    rv = review(ranked, hist, target, known)
    return EngineResult(target, hist, ranked, model, known, {}, {}, {}, rv, {})


def checks_pass(res: EngineResult, tier1: list[str]) -> float:
    """Return the share of Tier 1 names with no severe review flag."""
    return float(np.mean([not any(c in SEVERE_FLAGS for c, _ in res.review.flags.get(b, [])) for b in tier1]))


def reviewer_prompt_tokens(res: EngineResult) -> int:
    cards = [_card(res, b, "Tier 1" if b in res.review.tier1 else "Tier 2")[0] for b in res.review.tier1 + res.review.tier2]
    user = PROMPTS["reviewer_user"].format(target_line=res.target.label(), cards="\n".join(cards))
    return (len(PROMPTS["reviewer_system"]) + len(user)) // 3


def part_a(allow_paid: bool, dry: bool) -> dict:
    deals = load_deals(ROOT / "data/ma_transactions_500.csv")
    profiles = load_profiles(PROFILES_STEP4)
    model = RankingModel.fit(pd.read_pickle("out/pairs_v2.pkl").query("year <= 2021"))
    live = run(load_target(), profiles_dir=PROFILES_STEP4)
    test = [deals[deals.transaction_id == i].iloc[0] for i in frozen_deals(deals)]
    results = [backtest_result(d, deals, model, profiles) for d in test]
    if dry:
        tokens = [reviewer_prompt_tokens(live)] * RUNS + [reviewer_prompt_tokens(r) for r in results]
        usd = sum(cost_usd("writer", t, OUTPUT_TOKENS_A2) for t in tokens)
        return dict(calls=len(tokens), input_tokens=sum(tokens), est_output_tokens=OUTPUT_TOKENS_A2 * len(tokens), est_usd=round(usd, 2))
    out = dict(live={}, backtest={})
    # live target: A1 is deterministic, so 3 runs are identical by construction
    n0, t0 = ledger_lines(), time.perf_counter()
    a2 = [a2_reviewer(live, allow_paid=allow_paid, run=k) for k in range(1, RUNS + 1)]
    out["live"] = dict(
        A1=dict(tier1=live.review.tier1, stability=1.0, usd=0.0),
        A2=dict(tier1s=[x["tier1"] for x in a2], stability=overlap([x["tier1"] for x in a2]),
                accepted=[x["accepted"] for x in a2], rejected=[len(x["rejected"]) for x in a2],
                changed_vs_A1=[sorted(set(x["tier1"]) ^ set(live.review.tier1)) for x in a2],
                usd=round(spent_since(n0), 3), seconds_per_run=round((time.perf_counter() - t0) / RUNS, 1)))
    rows = []
    n0, t0 = ledger_lines(), time.perf_counter()
    for d, res in zip(test, results):
        a2 = a2_reviewer(res, allow_paid=allow_paid)
        winner = d.acquirer
        for arm, t1, t2 in (("A1", res.review.tier1, res.review.tier2), ("A2", a2["tier1"], a2["tier2"])):
            rows.append(dict(deal=d.transaction_id, arm=arm, in_tier1=winner in t1, in_list=winner in t1 + t2,
                             moved_out=winner in res.review.tier1 and winner not in t1 if arm == "A2" else
                             any(m.buyer == winner and m.action in ("removed", "merged", "to Tier 2") for m in res.review.log),
                             checks=checks_pass(res, t1), moves=len(a2["accepted"]) if arm == "A2" else 0))
    R = pd.DataFrame(rows)
    out["backtest"] = {arm: dict(deals=int(len(g)), winner_in_tier1=round(g.in_tier1.mean(), 3), winner_in_list=round(g.in_list.mean(), 3),
                                 winner_moved_out=int(g.moved_out.sum()), tier1_checks_pass=round(g.checks.mean(), 3),
                                 moves_per_deal=round(g.moves.mean(), 2)) for arm, g in R.groupby("arm")}
    out["backtest"]["A2"]["usd"] = round(spent_since(n0), 3)
    out["backtest"]["A2"]["seconds_total"] = round(time.perf_counter() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------


def specificity(results: list[dict]) -> float:
    """Return cited fact IDs per bullet, averaged over every box bullet of every page."""
    counts = []
    for r in results:
        page = OnePager.model_validate(r["page"])
        for box in (page.acquirer_overview, page.strategic_fit_thesis, page.precedent_activity, page.valuation_context):
            counts += [sum(len(g.split(",")) for g in CITATION.findall(p)) for p in box.points]
    return round(float(np.mean(counts)), 2)


def part_b(allow_paid: bool, dry: bool) -> dict:
    res = run(load_target())
    tier1 = priority_order(res, res.review.tier1)
    packs = {b: build_fact_pack(res, b, tier1) for b in tier1}
    if dry:
        w2 = 10 * RUNS * cost_usd("writer", 7500, 4500)
        w3 = 2 * (10 + 5) * cost_usd("writer", 8000, 4500)   # runs 2-3; about half the pages need a repair
        return dict(W2_est_usd=round(w2, 2), W3_runs_2_3_est_usd=round(w3, 2), est_usd=round(w2 + w3, 2))
    out = {}
    for approach in ("W2", "W3"):
        runs = []
        for k in range(1, RUNS + 1):
            n0, t0 = ledger_lines(), time.perf_counter()
            results = write_onepagers(packs, res.target.label(), approach, "writer", allow_paid, run=k)
            m = summarise(results, time.perf_counter() - t0)
            live_usd = spent_since(n0)
            if approach == "W3" and k == 1:   # the committed live run: cost and time as measured then
                saved = json.loads(Path("samples/default_target/metrics.json").read_text())
                m["usd"], m["wall_clock_seconds"] = saved["usd"], saved["wall_clock_seconds"]
            else:
                m["usd"] = round(live_usd, 3) if live_usd else m["usd"]
            runs.append(dict(run=k, pass_before_repair=m["pass_before_repair"], pass_after_repair=m["pass_after_repair"],
                             specificity=specificity(results), overlap_max=m["thesis_overlap_max"],
                             overlap_mean=m["thesis_overlap_mean"], generic_final=m["generic_hits_final"],
                             usd=m["usd"], seconds=m["wall_clock_seconds"]))
        out[approach] = runs
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"])
    ap.add_argument("--allow-paid", action="store_true")
    ap.add_argument("--dry", action="store_true", help="estimate sizes and cost; no calls")
    args = ap.parse_args()
    parts = ["A", "B"] if args.dry or not args.part else [args.part]
    for part in parts:
        result = (part_a if part == "A" else part_b)(args.allow_paid, args.dry)
        print(f"== Part {part}" + (" (estimate)" if args.dry else ""))
        print(json.dumps(result, indent=1, default=str))
        if not args.dry:
            Path("out").mkdir(exist_ok=True)
            Path(f"out/arch_compare_{part}.json").write_text(json.dumps(result, indent=1, default=str))
