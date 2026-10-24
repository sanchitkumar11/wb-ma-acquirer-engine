"""The one-pager pipeline end to end with a stub writer (no LLM, no cost): fan-out, validation,
the single repair, caching keys, and the paid-call guard."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

from acquirer_engine import graph, llm  # noqa: E402
from acquirer_engine.config import load_target  # noqa: E402
from acquirer_engine.engine import run  # noqa: E402
from acquirer_engine.factpack import build_fact_pack  # noqa: E402
from acquirer_engine.llm import Usage  # noqa: E402
from acquirer_engine.onepager import OnePager, validate  # noqa: E402


@pytest.fixture(scope="module")
def packs():
    res = run(load_target(ROOT / "configs/target_default.yaml"))
    return {b: build_fact_pack(res, b) for b in res.review.tier1}


def stub_page(pack: dict, invent: bool) -> OnePager:
    name, p = pack["buyer"]["name"], pack["precedents"][0]
    deal = f"{p['target']} ({p['size']}, {p['year']}) [{p['id']}]"
    section = dict(
        title=f"{name} has done this before",
        points=[f"{name} bought {deal}.", f"It closed {deal}."],
    )
    risk = dict(
        title="Competitive process",
        concern="Other buyers bid.",
        mitigant="Early access.",
        reality=f"See {deal}.",
    )
    return OnePager(
        buyer=name,
        headline=f"{name} fits",
        why_them=f"{name} has done this before",
        differentiator=("Pays 99.9x" if invent else f"{name} angle"),
        acquirer_overview=section,
        strategic_fit_thesis=section,
        precedent_activity=section,
        valuation_context=dict(
            title="In range",
            points=[f"Median {pack['valuation']['ev_ebitda_median']} [V:comps].", "See [V:comps]."],
        ),
        risk_flags=[risk, risk],
        conviction=dict(level=pack["conviction"]["level"], rationale=f"Basis [{p['id']}]."),
    )


@pytest.fixture
def stub(monkeypatch):
    calls = []

    def fake(schema, system, user, *, alias, key, purpose, allow_paid=False):
        calls.append(purpose)
        buyer = purpose.split(" ", 2)[2]
        pack = next(p for p in fake.packs.values() if p["buyer"]["name"] == buyer)
        return stub_page(pack, invent="repair" not in purpose), Usage()

    monkeypatch.setattr(graph, "structured_call", fake)
    fake.calls = calls
    return fake


def test_stub_page_passes_the_validator(packs):
    pack = packs["Nordic Capital"]
    assert validate(stub_page(pack, invent=False), pack) == []
    assert any(
        "Numbers not in the fact pack" in p for p in validate(stub_page(pack, invent=True), pack)
    )


def test_jargon_and_risk_cap_are_caught(packs):
    pack = packs["Nordic Capital"]
    page = stub_page(pack, invent=False)
    page.strategic_fit_thesis.points.append(
        "It has 18 Ring 1 deals and sits in Tier 1, a NO_CLOSED_DEAL-free x1.30 story."
    )
    page.risk_flags = page.risk_flags * 2
    problems = " ".join(validate(page, pack))
    assert "Internal terms" in problems and "2 to 3 risk flags" in problems
    assert "Ring N" in problems and "Tier N" in problems and "odds multiplier" in problems


def test_w3_repairs_once_and_then_passes(packs, stub):
    stub.packs = packs
    results = graph.write_onepagers(packs, "target", approach="W3")
    assert [r["buyer"] for r in results] == list(packs)
    assert all(r["repairs"] == 1 and r["problems"] == [] for r in results)


def test_w2_does_not_repair(packs, stub):
    stub.packs = packs
    results = graph.write_onepagers(dict(list(packs.items())[:2]), "target", approach="W2")
    assert all(r["repairs"] == 0 and r["problems"] for r in results)


def test_paid_call_needs_permission(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "databricks")
    with pytest.raises(llm.PaidCallNotAllowedError):
        llm.structured_call(
            OnePager, "s", "u", alias="writer", key="nokey", purpose="test", allow_paid=False
        )


def test_replay_never_calls_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "replay")
    with pytest.raises(llm.NotCachedError):
        llm.structured_call(
            OnePager, "s", "u", alias="writer", key="nokey", purpose="test", allow_paid=True
        )


def test_budget_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "spent_usd", lambda: 49.99)
    monkeypatch.setenv("LLM_PROVIDER", "databricks")
    with pytest.raises(llm.BudgetExceededError):
        llm.structured_call(
            OnePager,
            "s" * 3000,
            "u" * 30000,
            alias="writer",
            key="nokey",
            purpose="test",
            allow_paid=True,
        )


def test_cache_key_changes_with_prompt_version_and_pack():
    a = llm.cache_key("onepager_v1", "writer", "W3", "hash1", "write")
    assert a != llm.cache_key("onepager_v2", "writer", "W3", "hash1", "write")
    assert a != llm.cache_key("onepager_v1", "writer", "W3", "hash2", "write")


def test_a2_rejects_moves_without_evidence(monkeypatch):
    from acquirer_engine import listbuild

    res = run(load_target(ROOT / "configs/target_default.yaml"))
    moves = listbuild.ReviewerMoves(
        moves=[
            listbuild.ReviewerMove(
                buyer="Bon Secours Mercy Health",
                action="promote",
                code="FOOTPRINT_OVERLAP",
                fact_ids=["P:Bon Secours Mercy Health:2"],
                reason="Ohio footprint",
            ),
            listbuild.ReviewerMove(
                buyer="TPG Capital",
                action="demote",
                code="WEAK_EVIDENCE",
                fact_ids=[],
                reason="no cite",
            ),
            listbuild.ReviewerMove(
                buyer="Nordic Capital",
                action="demote",
                code="WEAK_EVIDENCE",
                fact_ids=["X:made-up"],
                reason="bad cite",
            ),
        ]
    )
    monkeypatch.setattr(listbuild, "structured_call", lambda *a, **k: (moves, Usage()))
    out = listbuild.a2_reviewer(res)
    assert [m["buyer"] for m in out["accepted"]] == ["Bon Secours Mercy Health"]
    assert len(out["rejected"]) == 2 and len(out["tier1"]) == 10
    assert "Bon Secours Mercy Health" in out["tier1"] and "Nordic Capital" in out["tier1"]
