"""Summary page: priority order by conviction, motives as-of, call-list columns, the summary
checks."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rep():
    os.chdir(ROOT)
    from acquirer_engine.config import load_target
    from acquirer_engine.report import build

    return build(load_target(ROOT / "configs/target_default.yaml"), replay=True)


def test_conviction_split_is_unchanged(rep):
    assert Counter(p["conviction"]["level"] for p in rep.packs.values()) == {
        "High": 5,
        "Medium": 3,
        "Low": 2,
    }


def test_priority_follows_conviction_then_model_rank(rep):
    from acquirer_engine.report import LEVEL_ORDER

    keys = [
        (LEVEL_ORDER[rep.packs[b]["conviction"]["level"]], rep.packs[b]["ranking"]["overall_rank"])
        for b in rep.tier1
    ]
    assert keys == sorted(keys)
    assert [rep.packs[b]["ranking"]["position_on_list"] for b in rep.tier1] == list(range(1, 11))


def test_call_list_columns(rep):
    from acquirer_engine.summary import call_list

    rows = call_list(rep)
    assert list(rows[0]) == ["Priority", "Buyer", "Type", "Conviction", "Why them", "Can pay"]
    pay = {r["Buyer"]: r["Can pay"] for r in rows}
    assert all(
        v.split(";")[0].split(":")[0] in {"comfortable", "stretch", "add-on exit"}
        for v in pay.values()
    )
    assert len(set(pay.values())) == len(pay)  # every buyer's line carries its own evidence
    assert pay["Nordic Capital"] == "add-on exit; usual deal $77M, fund $11B"
    assert pay["Advocate Health"].startswith("comfortable: closed $")


def test_motives_use_only_earlier_deals(rep):
    """As of 2019 Q4, the motive counts must match deals before 2020 only (the as-of filter).

    The live engine refuses an as_of before the data end, so the earlier date is set on the
    result directly: _motives must filter by it, whatever deals the result holds.
    """
    from acquirer_engine.factpack import _motives

    early = replace(rep.res, target=replace(rep.res.target, as_of="2019Q4"))
    motives = _motives(early, "Nordic Capital", [])
    n_before = int(
        ((early.deals.acquirer == "Nordic Capital") & (early.deals.deal_year < 2020)).sum()
    )
    assert motives and all(m["deals"].endswith(f"of {n_before}") for m in motives)


def test_tier2_reasons_never_cite_rank(rep):
    from acquirer_engine.summary import tier2_groups

    reasons = [r["Why not Tier 1"] for g in tier2_groups(rep) for r in g["rows"]]
    assert reasons and not any("ranked" in r.lower() for r in reasons)


def test_exec_summary_validator_rejects_stats_headline(rep):
    from acquirer_engine.execsummary import ExecSummary, validate_summary
    from acquirer_engine.summary import summary_pack

    pack = summary_pack(rep)
    bad = ExecSummary(
        headline="7 sponsors and 3 strategics lead the list",
        sentences=["One [E:tier1].", "Two [E:valuation].", "Three [E:review]."],
    )
    assert any("Headline" in p for p in validate_summary(bad, pack))
    assert rep.exec_summary and 3 <= len(rep.exec_summary["sentences"]) <= 4


def test_live_engine_refuses_an_as_of_before_the_data_end(rep):
    """An earlier as_of would rank on later deals and mislabel the pages: it must stop with a
    reason."""
    from acquirer_engine.engine import run

    with pytest.raises(ValueError, match="data end"):
        run(replace(rep.res.target, as_of="2019Q4"))
