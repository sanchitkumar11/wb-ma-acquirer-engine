"""Command line.

python -m acquirer_engine rank      the model's ranked buyers, with profile notes
python -m acquirer_engine review    Tier 1 / Tier 2, valuation, ability to pay, sponsor matrix,
review log
python -m acquirer_engine run       the full engine: review, fact packs, LLM one-pagers, report
python -m acquirer_engine flag      flag a buyer as irrelevant for a target (applied on the next
run)
python -m acquirer_engine unflag    remove a flag
python -m acquirer_engine rules     rules suggested by the flags; --accept N writes one
python -m acquirer_engine compare   two targets' buyer lists side by side, with a short note
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from acquirer_engine.config import DEFAULT_TARGET_PATH, load_target
from acquirer_engine.data import DEFAULT_CSV_PATH
from acquirer_engine.engine import EngineResult, run
from acquirer_engine.errors import LLMError
from acquirer_engine.feedback import (
    REASONS,
    accept,
    feedback_path,
    learned_path,
    record,
    suggestions,
)
from acquirer_engine.lbo import SYNERGY_RANGE_TEXT
from acquirer_engine.paths import ROOT
from acquirer_engine.profiles import PROFILES_DIR, load_profiles, profile_notes
from acquirer_engine.ranking import rank_buyers
from acquirer_engine.rings import RING_NAME
from acquirer_engine.wording import CSV_ONLY_NOTE

# rank, review, flag and rules never load the LLM, LangGraph or Plotly stacks: those modules are
# imported inside the commands that need them (run, compare, review --suggest), so the light
# commands start in under a second.

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def format_row(r: Any) -> str:
    """Return one printed line per buyer: rank, name, type, likelihood, ring and size facts."""
    buyer_type = "Sponsor" if r.sponsor else "Strategic"
    facts = f"{RING_NAME[r.closest_ring]}; same-model deals {int(r.n_model)}/{int(r.n)}; "
    if r.sponsor:
        facts += (
            f"open platform in sub-sector: {'yes' if r.open_platform else 'no'}; "
            f"median deal ${r.med_size:.0f}M"
        )
    else:
        facts += f"biggest deal ${r.max_size:.0f}M"
    return f"  #{r.model_rank:2d} {r.acquirer:26s} {buyer_type:9s} {r.share:5.1%}  [{facts}]"


def print_review(res: EngineResult) -> None:
    """Print the reviewed list, the numbers behind it and every move the review made."""
    rv, v, lbo, syn = res.review, res.valuation, res.sponsor_lbo, res.strategic_synergies
    by_name = res.ranked.set_index("acquirer")
    print(f"== Banker review: {res.target.label()}")
    if rv.csv_only:
        print(f"   {CSV_ONLY_NOTE}")
    for tier, names in (("Tier 1", rv.tier1), ("Tier 2", rv.tier2)):
        print(f"\n{tier}")
        for b in names:
            r = by_name.loc[b]
            flags = "; ".join(code for code, _ in rv.flags.get(b, []))
            print(
                f"  #{r.model_rank:<3d} {b:26s} {'Sponsor' if r.sponsor else 'Strategic':9s} "
                f"{r.share:5.1%}  {RING_NAME[r.closest_ring]:6s} {flags}"
            )
    print(
        f"\nValuation context: {v['n_comps']} comparable closed control deals "
        f"(core + Ring 1, ${v['size_band_mm'][0]:.0f}-{v['size_band_mm'][1]:.0f}M; {v['n_core']} "
        "core)"
    )
    print(
        f"  EV/EBITDA p25 / median / p75: {v['ev_ebitda']['p25']}x / {v['ev_ebitda']['p50']}x / "
        f"{v['ev_ebitda']['p75']}x; "
        f"EV/Revenue {v['ev_revenue']['p25']}x / {v['ev_revenue']['p50']}x / "
        f"{v['ev_revenue']['p75']}x"
    )
    print(f"  Target margin is above {v['margin_percentile']}% of comps. {v['defensible_reason']}.")
    print(
        f"Quick LBO at ${res.target.enterprise_value_mm:.0f}M ({lbo['leverage']}x debt at "
        f"{lbo['interest_rate']:.1%}): "
        f"IRR {lbo['irr_at_ask']:.1%}, equity cheque ${lbo['equity_cheque']}M; max price at a "
        f"{lbo['target_irr']:.0%} IRR ${lbo['max_price_same_exit']}M, or "
        f"${lbo['max_price_platform_exit']}M "
        f"exiting at {lbo['platform_exit_multiple']}x inside a platform"
    )
    print(
        f"Strategic synergy headroom: ${syn['synergy_value_low']}-{syn['synergy_value_high']}M "
        "above the ask "
        f"({SYNERGY_RANGE_TEXT}, half realised)"
    )
    print("\nSponsor differentiation (Tier 1 and Tier 2 sponsors)")
    m = rv.matrix[rv.matrix.acquirer.isin(rv.tier1 + rv.tier2)]
    for r in m.itertuples():
        print(
            f"  #{r.model_rank:<3d} {r.acquirer:22s} same-model {r.provider_deals:9s} platform "
            f"{r.open_platform:3s} "
            f"median ${r.median_deal_mm}M at "
            f"{'no closed deal' if pd.isna(r.median_ev_ebitda) else f'{r.median_ev_ebitda}x'}; "
            f"minority {r.minority_deals}; angles: {', '.join(r.angles) or 'none'}"
        )
    print("\nReview log")
    for mv in rv.log:
        print(f"  #{mv.model_rank:<3d} {mv.buyer:26s} {mv.action:16s} {mv.code:18s} {mv.note[:80]}")


def default_report_path(target_path: Path, only: str | None = None) -> Path:
    """Return samples/default_target/report.md for the default target, else out/<target
    name>/report.md.

    A one-buyer run (--only) always writes under out/, so it can never overwrite a committed
    sample with a one-page report.
    """
    if only:
        return (
            ROOT / "out" / target_path.stem / f"only_{only.lower().replace(' ', '_')}" / "report.md"
        )
    if target_path.resolve() == DEFAULT_TARGET_PATH.resolve():
        return ROOT / "samples/default_target/report.md"
    return ROOT / "out" / target_path.stem / "report.md"


def run_full(args: argparse.Namespace) -> None:
    """Review, fact packs, LLM one-pagers through the LangGraph pipeline, then the Markdown
    report."""
    from acquirer_engine.graph import total_usage
    from acquirer_engine.llm import provider
    from acquirer_engine.metrics import summarise
    from acquirer_engine.onepager import OnePager
    from acquirer_engine.render import onepager_markdown
    from acquirer_engine.report import build
    from acquirer_engine.summary import (
        appendix_markdown,
        bench_markdown,
        sources_markdown,
        summary_markdown,
    )

    start = time.perf_counter()
    from acquirer_engine.llm import LEDGER_PATH

    ledger_before = len(LEDGER_PATH.read_text().splitlines()) if LEDGER_PATH.exists() else 0
    rep = build(
        load_target(args.target),
        args.csv,
        args.csv_only,
        args.list_approach,
        args.approach,
        args.model,
        args.allow_paid,
        args.only,
    )
    if rep.results is None:
        raise SystemExit(f"Stopped before any spend: {rep.missing}")
    metrics = summarise(rep.results, time.perf_counter() - start)
    out = args.out or default_report_path(args.target, args.only)
    from acquirer_engine.figures import export

    figures = export(rep, out.parent / "figures")
    parts = [summary_markdown(rep, figures), "", "---", ""]
    for r in rep.results:
        parts += [
            onepager_markdown(OnePager.model_validate(r["page"]), rep.packs[r["buyer"]]),
            "",
            "---",
            "",
        ]
        status = "passes all checks" if not r["problems"] else "FAILED: " + "; ".join(r["problems"])
        print(
            f"  {r['buyer']:24s} {status} (repairs: {r['repairs']}; "
            f"{metrics['words_per_page'].get(r['buyer'])} words)"
        )
    parts += [
        bench_markdown(rep),
        "",
        "---",
        "",
        appendix_markdown(rep, figures),
        "",
        sources_markdown(rep),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    if metrics[
        "llm_calls"
    ]:  # a replay has no tokens, cost or first drafts: keep the live run's metrics
        out.with_name("metrics.json").write_text(json.dumps(metrics, indent=1, default=dict))
    usage = total_usage(rep.results)
    live = (
        len(LEDGER_PATH.read_text().splitlines()) if LEDGER_PATH.exists() else 0
    ) > ledger_before  # any paid call this run
    source = (
        f"provider {provider()}" if live else "replay (cache)"
    )  # no live call: say so, whatever .env sets
    lines = LEDGER_PATH.read_text().splitlines()[ledger_before:] if LEDGER_PATH.exists() else []
    spent = sum(json.loads(x)["usd"] for x in lines)  # pages, repairs and the executive summary
    print(
        f"\nWrote {out}. {source}, list {args.list_approach}, approach {args.approach}, model "
        f"{args.model}: "
        f"{usage.input_tokens:,} input / {usage.output_tokens:,} output tokens on the pages, "
        f"${spent:.3f} this run in all; "
        f"{time.perf_counter() - start:.1f}s."
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def run_feedback(args: argparse.Namespace) -> None:
    """flag / unflag a buyer, or list (and accept) the rules the flags suggest."""

    target = load_target(args.target)
    if args.command in ("flag", "unflag"):
        res = run(target, args.csv)
        if args.buyer not in set(res.ranked.acquirer):
            raise SystemExit(
                f"Unknown buyer {args.buyer!r}: use the name as written in the deal file"
            )
        if args.command == "unflag" and args.buyer not in res.user_flags:
            raise SystemExit(f"{args.buyer} is not flagged for this target")
        listed = args.buyer in res.review.tier1 + res.review.tier2 or args.buyer in res.user_flags
        line = record(
            target,
            args.buyer,
            getattr(args, "reason", ""),
            getattr(args, "note", "") or "",
            args.command,
        )
        print(
            f"{'Flagged' if args.command == 'flag' else 'Unflagged'} {args.buyer} for "
            f"{target.label()} "
            f"({line['reason'] or 'flag removed'}); written to {feedback_path()}."
            + ("" if listed else " Note: this buyer is not on the current list.")
            + " Applied on the next run."
        )
        return
    res = run(target, args.csv)
    found = suggestions(target, res.user_flags, res.deals, res.profiles, res.review.flags)
    print(
        f"== Flags for {target.label()}: "
        + (", ".join(f"{b} ({f['reason']})" for b, f in res.user_flags.items()) or "none")
    )
    if not found:
        print(
            "No rule suggested: a rule needs two or more flags with the same reason and a shared, "
            "rule-shaped property."
        )
    for i, s in enumerate(found, 1):
        print(f"  [{i}] {s['label']}\n      from: {', '.join(s['from_flags'])}; {s['history']}")
    if args.accept:
        if not 1 <= args.accept <= len(found):
            raise SystemExit(f"No suggestion {args.accept}")
        entry = accept(found[args.accept - 1])
        print(
            f"Accepted [{args.accept}] into {learned_path()}: applies to {entry['sector']} targets "
            "from the next run."
        )


def print_suggestions(res: EngineResult, args: argparse.Namespace) -> None:
    """Print the LLM reviewer's suggested moves; --accept / --reject N records one through the
    feedback loop."""
    from acquirer_engine.listbuild import suggested_moves

    out = suggested_moves(res, allow_paid=args.allow_paid)
    print("\nSuggested moves (LLM reviewer; not applied: the list above is the rules' list)")
    if not out["moves"]:
        print(
            "  none"
            + (
                f" ({out['rejected']} refused by code for missing evidence)"
                if out["rejected"]
                else ""
            )
        )
    for i, m in enumerate(out["moves"], 1):
        print(f"  [{i}] {m['action']} {m['buyer']} ({m['code']}): {m['reason']}")
        for fid, text in m["facts"]:
            print(f"        [{fid}] {text}")
    for n, action in ((args.accept, "accept"), (args.reject, "reject")):
        if n:
            if not 1 <= n <= len(out["moves"]):
                raise SystemExit(f"No suggestion {n}")
            m = out["moves"][n - 1]
            cited = "; ".join(f"[{fid}] {text}" for fid, text in m["facts"])
            if action == "accept":
                record(
                    res.target,
                    m["buyer"],
                    m["code"],
                    f"{m['reason']} Evidence: {cited}",
                    m["action"],
                )
            else:
                record(res.target, m["buyer"], f"{m['action']}:{m['code']}", m["reason"], "dismiss")
            print(
                f"{'Accepted' if action == 'accept' else 'Rejected'} [{n}]: {m['action']} "
                f"{m['buyer']}; "
                f"recorded in the feedback file, applied on the next run."
                if action == "accept"
                else f"Rejected [{n}]: {m['action']} {m['buyer']}; it will not be suggested again."
            )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def run_compare(args: argparse.Namespace) -> None:
    """Build both targets (pages from the cache), write the comparison note and page, and print a
    summary."""
    from acquirer_engine.compare import compare_markdown, comparison_note, shared_rows, unique_rows
    from acquirer_engine.figures import compare_map, write_png
    from acquirer_engine.llm import LEDGER_PATH
    from acquirer_engine.report import build

    start = time.perf_counter()
    lines_before = len(LEDGER_PATH.read_text().splitlines()) if LEDGER_PATH.exists() else 0
    a, b = (
        build(load_target(path), args.csv, replay=True) for path in (args.a, args.b)
    )  # pages: cache only
    note, from_llm = comparison_note(a, b, args.model, args.allow_paid)
    out = (
        args.out
        or ROOT
        / "out"
        / f"compare_{args.a.stem.removeprefix('target_')}_{args.b.stem.removeprefix('target_')}"
        / "report.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = (
        "compare_map.png" if write_png(compare_map(a, b), out.parent / "compare_map.png") else None
    )
    out.write_text(compare_markdown(a, b, note, from_llm, figure))
    shared = shared_rows(a, b)
    pairs = [f"{r['buyer']} ({r['conviction_a']} / {r['conviction_b']})" for r in shared]
    print(f"Shared Tier 1: {', '.join(pairs) or 'none'}")
    print(
        f"Only in A: {len(unique_rows(a, b))}; only in B: {len(unique_rows(b, a))}. Note: "
        f"{'LLM' if from_llm else 'template'}."
    )
    cost = (
        sum(json.loads(x)["usd"] for x in LEDGER_PATH.read_text().splitlines()[lines_before:])
        if LEDGER_PATH.exists()
        else 0.0
    )
    print(f"Wrote {out}. ${cost:.3f} this run; {time.perf_counter() - start:.1f}s.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run the command line; a ValueError (bad target, too few comparable deals) exits with its
    message."""
    logging.basicConfig(
        level=logging.INFO, format="%(message)s"
    )  # library notes and warnings, plainly
    try:
        _run(argv)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None


def _run(argv: list[str] | None) -> None:
    """Parse the command line and run the chosen command."""
    parser = argparse.ArgumentParser(prog="acquirer_engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("rank", help="print the model's top buyers for the target")
    rank.add_argument(
        "--target", type=Path, default=DEFAULT_TARGET_PATH, help="target profile YAML"
    )
    rank.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="deal CSV")
    rank.add_argument("--top", type=int, default=25, help="how many buyers to print")
    rank.add_argument(
        "--refit", action="store_true", help="refit the model even if a saved one matches"
    )
    rank.add_argument(
        "--csv-only",
        action="store_true",
        help="use the CSV alone: no buyer profiles, so no status, merger or mandate checks",
    )
    rev = sub.add_parser(
        "review", help="Tier 1 / Tier 2 after the banker review, with the numbers behind it"
    )
    rev.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH, help="target profile YAML")
    rev.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="deal CSV")
    rev.add_argument(
        "--csv-only", action="store_true", help="no buyer profiles: profile-based rules off"
    )
    rev.add_argument(
        "--suggest",
        action="store_true",
        help="also show the LLM reviewer's suggested moves (cited, not applied); accept or reject "
        "below",
    )
    rev.add_argument(
        "--allow-paid", action="store_true", help="allow the reviewer's paid call on a cache miss"
    )
    rev.add_argument(
        "--accept",
        type=int,
        default=0,
        help="with --suggest: record suggestion N as a deal-team move",
    )
    rev.add_argument(
        "--reject",
        type=int,
        default=0,
        help="with --suggest: dismiss suggestion N (not shown again)",
    )
    go = sub.add_parser("run", help="full engine: review, fact packs and LLM one-pagers")
    go.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH, help="target profile YAML")
    go.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="deal CSV")
    go.add_argument(
        "--csv-only", action="store_true", help="no buyer profiles: profile-based rules off"
    )
    go.add_argument(
        "--list",
        dest="list_approach",
        default="A1",
        choices=["A1", "A2", "A3"],
        help="how Tier 1 is built",
    )
    go.add_argument(
        "--approach",
        default="W3",
        choices=["W1", "W2", "W3", "W4"],
        help="how the one-pagers are written",
    )
    go.add_argument(
        "--model",
        default="writer",
        help="model role in configs/llm_prices.yaml (writer, writer_dev, ...)",
    )
    go.add_argument("--only", help="write the one-pager for this Tier 1 buyer only")
    go.add_argument(
        "--allow-paid",
        action="store_true",
        help="allow paid LLM calls on a cache miss (asks nothing else)",
    )
    go.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown report path (default: samples/default_target/report.md)",
    )

    for name, text in (
        ("flag", "flag a buyer as irrelevant for this target"),
        ("unflag", "remove a buyer's flag"),
        ("rules", "rules suggested by two or more flags with the same reason"),
    ):
        fb = sub.add_parser(name, help=text)
        fb.add_argument(
            "--target", type=Path, default=DEFAULT_TARGET_PATH, help="target profile YAML"
        )
        fb.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="deal CSV")
        if name in ("flag", "unflag"):
            fb.add_argument("buyer", help="buyer name as written in the deal file")
        if name == "flag":
            fb.add_argument("--reason", required=True, choices=REASONS)
            fb.add_argument("--note", default="", help="free text (required for reason 'other')")
        if name == "rules":
            fb.add_argument(
                "--accept",
                type=int,
                default=0,
                help="write suggestion N to configs/learned_rules.yaml",
            )
    cmp = sub.add_parser(
        "compare", help="two targets' buyer lists side by side, with a short LLM note"
    )
    cmp.add_argument("a", type=Path, help="first target profile YAML")
    cmp.add_argument("b", type=Path, help="second target profile YAML")
    cmp.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="deal CSV")
    cmp.add_argument("--model", default="writer", help="model role for the note")
    cmp.add_argument(
        "--allow-paid", action="store_true", help="allow the paid call for the note on a cache miss"
    )
    cmp.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown path (default: out/compare_<a>_<b>/report.md)",
    )
    args = parser.parse_args(argv)
    if args.command == "compare":
        run_compare(args)
        return
    if args.command in ("flag", "unflag", "rules"):
        try:
            run_feedback(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        return
    if args.command == "run":
        try:
            run_full(args)
        except LLMError as exc:  # a guard stopped the call: nothing was spent
            raise SystemExit(f"Stopped before any spend: {exc}") from None
        return
    if args.command == "review":
        start = time.perf_counter()
        res = run(load_target(args.target), args.csv, csv_only=args.csv_only)
        print_review(res)
        if args.suggest:
            try:
                print_suggestions(res, args)
            except LLMError as exc:
                raise SystemExit(f"Suggested moves not available: {exc}") from None
        print(f"\nDone in {time.perf_counter() - start:.1f}s.")
        return

    start = time.perf_counter()
    target = load_target(args.target)
    ranked = rank_buyers(target, args.csv, refit=args.refit)
    profiles = {} if args.csv_only else load_profiles(PROFILES_DIR)
    print(f"== Model ranking: {target.label()}")
    if args.csv_only:
        print(f"   {CSV_ONLY_NOTE}")
    buyers = set(ranked.acquirer)
    for r in ranked.head(args.top).itertuples():
        notes = profile_notes(r.acquirer, profiles, target.sector, buyers)
        print(format_row(r) + "".join(f"\n        {code}: {note}" for code, note, _ in notes))
    print(
        f"\n{len(ranked)} buyers scored in {time.perf_counter() - start:.1f}s"
        + ("" if args.csv_only else f"; profile facts for {len(profiles)} buyers")
        + "."
    )
