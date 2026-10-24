"""Assemble a full report (review, fact packs, one-pagers, executive summary) for the CLI and the
app.

The top of the pipeline: it calls every layer below it once. The app must work without a key,
so it calls build() with replay=True: one-pagers come only from the cache, and a target that was
never generated still gets every deterministic part (lists, matrix, valuation, review log), with
the pages marked as not generated. The Report record itself lives in report_types.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import TYPE_CHECKING

from acquirer_engine.data import DEFAULT_CSV_PATH
from acquirer_engine.engine import EngineResult, run
from acquirer_engine.errors import NotCachedError, PaidCallNotAllowedError
from acquirer_engine.execsummary import add_exec_summary
from acquirer_engine.factpack import build_fact_pack, buyer_conviction
from acquirer_engine.graph import write_onepagers
from acquirer_engine.listbuild import build_list
from acquirer_engine.llm import replay_only
from acquirer_engine.report_types import Report

if TYPE_CHECKING:
    from pathlib import Path

    from acquirer_engine.config import Target


LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def priority_order(res: EngineResult, tier1: list[str]) -> list[str]:
    """Return Tier 1 sorted by conviction (High first), then by model rank.

    Conviction never depends on this order (it uses the rank among buyers left after the review),
    so sorting by it is not circular.
    """
    rank = dict(zip(res.ranked.acquirer, res.ranked.model_rank, strict=True))
    return sorted(tier1, key=lambda b: (LEVEL_ORDER[buyer_conviction(res, b)[0]], rank[b]))


def build(
    target: Target,
    csv_path: Path = DEFAULT_CSV_PATH,
    csv_only: bool = False,
    list_approach: str = "A1",
    approach: str = "W3",
    model: str = "writer",
    allow_paid: bool = False,
    only: str | None = None,
    replay: bool = False,
) -> Report:
    """Return the report for one target. With replay, no live LLM call is made (for this build
    only).

    Raises ValueError if `only` names a buyer that is not in Tier 1.
    """
    with replay_only() if replay else nullcontext():
        return _build(target, csv_path, csv_only, list_approach, approach, model, allow_paid, only)


def _build(
    target: Target,
    csv_path: Path,
    csv_only: bool,
    list_approach: str,
    approach: str,
    model: str,
    allow_paid: bool,
    only: str | None,
) -> Report:
    """Run build()'s steps: engine, list, fact packs, then the pages and the summary in parallel."""
    res = run(target, csv_path, csv_only=csv_only)
    listed = build_list(res, list_approach, model, allow_paid)
    tier1 = priority_order(res, listed["tier1"])
    tier2 = listed.get("tier2", res.review.tier2)
    packs = {b: build_fact_pack(res, b, tier1) for b in tier1}
    if only and only not in packs:
        raise ValueError(f"Not in Tier 1: {only}. Tier 1 is {', '.join(tier1)}")
    wanted = {only: packs[only]} if only else packs
    rep = Report(res, tier1, tier2, packs, None, approach)

    # The executive summary reads engine outputs only, so it is written at the same time as the
    # pages
    with ThreadPoolExecutor(max_workers=2) as pool:
        pages = pool.submit(write_onepagers, wanted, target.label(), approach, model, allow_paid)
        summary = pool.submit(add_exec_summary, rep, model, allow_paid)
        try:
            rep.results = pages.result()
        except (NotCachedError, PaidCallNotAllowedError) as exc:
            rep.missing = str(exc)
        summary.result()
    return rep
