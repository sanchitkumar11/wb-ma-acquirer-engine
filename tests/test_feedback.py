"""Feedback loop: flags apply as USER_FLAG moves, Tier 1 refills, flags are idempotent, rules are
never silent."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from acquirer_engine.config import load_target
from acquirer_engine.engine import run
from acquirer_engine.feedback import (
    accept,
    active_flags,
    learned_rules,
    record,
    suggestions,
    target_hash,
)

ROOT = Path(__file__).resolve().parents[1]
TECH = ROOT / "configs/target_healthit.yaml"


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Point flags and learned rules at temporary files, so tests never touch the real ones."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("LEARNED_RULES_PATH", str(tmp_path / "learned_rules.yaml"))
    return tmp_path


def test_flag_applies_as_a_review_move_and_tier1_refills(files):
    target = load_target(TECH)
    before = run(target).review
    flagged = before.tier1[-1]
    record(target, flagged, "conflict / already in talks", "in talks on another asset")
    after = run(target).review
    move = next(m for m in after.log if m.buyer == flagged)
    assert (
        move.code == "USER_FLAG"
        and move.action == "flagged"
        and "in talks on another asset" in move.note
    )
    assert flagged not in after.tier1 + after.tier2 and len(after.tier1) == 10
    newcomer = (set(after.tier1) - set(before.tier1)).pop()
    assert newcomer in before.tier2  # the next eligible buyer moves up from Tier 2


def test_flags_are_idempotent_and_unflag_restores_the_list(files):
    target = load_target(TECH)
    original = run(target).review.tier1
    record(target, "Veritas Capital", "off-mandate")
    once = run(target).review
    record(target, "Veritas Capital", "off-mandate")  # same flag again
    twice = run(target).review
    assert once.tier1 == twice.tier1 and sum(m.code == "USER_FLAG" for m in twice.log) == 1
    record(target, "Veritas Capital", action="unflag")
    assert active_flags(target) == {} and run(target).review.tier1 == original


def test_flags_belong_to_one_target(files):
    tech, dental = load_target(TECH), load_target(ROOT / "configs/target_dental.yaml")
    record(tech, "Blackstone", "can't pay")
    assert target_hash(tech) != target_hash(dental) and active_flags(dental) == {}
    assert "Blackstone" in run(dental).review.tier1


def test_two_shared_flags_suggest_a_rule_but_change_nothing_until_accepted(files):
    target = load_target(TECH)
    record(target, "Veritas Capital", "off-mandate")
    res = run(target)
    assert (
        suggestions(target, res.user_flags, res.deals, res.profiles, res.review.flags) == []
    )  # one flag: no rule
    record(target, "Vista Equity", "off-mandate")
    res = run(target)
    found = suggestions(target, res.user_flags, res.deals, res.profiles, res.review.flags)
    assert (
        len(found) == 1
        and found[0]["kind"] == "mandate_focus"
        and found[0]["value"] == "technology"
    )
    assert (
        "2 of 50 Health IT deals" in found[0]["history"]
    )  # checked against history before it can be accepted
    assert "Francisco Partners" in res.review.tier1  # not applied yet: never silent
    accept(found[0])
    res = run(target)
    move = next(m for m in res.review.log if m.buyer == "Francisco Partners")
    assert move.code == "MANDATE_MISMATCH" and "accepted from deal-team flags" in move.note
    assert (
        learned_rules("Health IT") and learned_rules("Dental") == []
    )  # only for the flagged sub-sector
    assert (
        suggestions(target, res.user_flags, res.deals, res.profiles, res.review.flags) == []
    )  # not suggested twice


def test_one_off_reasons_never_become_rules(files):
    target = load_target(TECH)
    for buyer in ("Veritas Capital", "Vista Equity"):
        record(target, buyer, "conflict / already in talks")
    res = run(target)
    assert suggestions(target, res.user_flags, res.deals, res.profiles, res.review.flags) == []


def test_bad_flags_are_refused(files):
    target = load_target(TECH)
    with pytest.raises(ValueError):
        record(target, "Vista Equity", "not a reason")
    with pytest.raises(ValueError):
        record(target, "Vista Equity", "other")  # "other" needs a note


def test_cli_flag_and_unflag(files):
    env = {**os.environ, "LLM_PROVIDER": "replay"}

    def cli(*a: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "acquirer_engine", *a, "--target", str(TECH)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    assert "Flagged Vista Equity" in cli("flag", "Vista Equity", "--reason", "off-mandate").stdout
    assert (
        "Flagged Veritas Capital"
        in cli("flag", "Veritas Capital", "--reason", "off-mandate").stdout
    )
    out = cli("rules").stdout
    assert "apply the mandate rule to all technology investors" in out
    assert "Unflagged Vista Equity" in cli("unflag", "Vista Equity").stdout
    assert cli("unflag", "Vista Equity").returncode != 0  # not flagged any more
    assert cli("flag", "Nobody Capital", "--reason", "off-mandate").returncode != 0


def test_app_shows_flags_and_the_suggested_rule(files, monkeypatch):
    """The method page shows the suggestion with its accept button; the stage chart counts the
    flags."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    target = load_target(TECH)
    for buyer in ("Veritas Capital", "Vista Equity"):
        record(target, buyer, "off-mandate")
    monkeypatch.setenv("APP_TARGET", "target_healthit.yaml")
    monkeypatch.setenv("APP_START_PAGE", "how-we-built-the-list")
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert any("apply the mandate rule to all technology investors" in m.value for m in at.markdown)
    assert any(b.label == "Accept this rule" for b in at.button)


def test_llm_reviewer_refill_never_brings_back_a_rule_moved_buyer(monkeypatch):
    """A2 (cached replies): demotions refill Tier 1 in model order, skipping buyers the rules moved
    (e.g. off-mandate)."""
    from acquirer_engine.listbuild import a2_reviewer

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("LLM_PROVIDER", "replay")
    res = run(
        load_target(ROOT / "configs/target_default.yaml"),
        profiles_dir=ROOT / "experiments/results/profiles_step4",
    )
    ruled_out = {m.buyer for m in res.review.log if m.action == "to Tier 2"}
    for k in (1, 2, 3):
        out = a2_reviewer(res, run=k)
        assert len(out["tier1"]) == 10 and not ruled_out & set(out["tier1"])


def test_reviewer_suggestions_are_cited_and_applied_only_when_accepted(files, monkeypatch):
    """review --suggest: cited moves, nothing applied; accept writes a deal-team move, reject hides
    it."""
    from acquirer_engine.feedback import dismissed
    from acquirer_engine.listbuild import suggested_moves

    target = load_target(ROOT / "configs/target_default.yaml")
    res = run(target)
    monkeypatch.setenv("LLM_PROVIDER", "replay")  # cached review only
    moves = suggested_moves(res)["moves"]
    assert {(m["action"], m["buyer"]) for m in moves} >= {
        ("demote", "UPMC"),
        ("promote", "Bain Capital"),
    }
    assert all(
        m["facts"] and all(text for _, text in m["facts"]) for m in moves
    )  # every move shows its evidence
    assert "UPMC" in res.review.tier1  # suggested, not applied
    upmc = next(m for m in moves if m["buyer"] == "UPMC")
    bain = next(m for m in moves if m["buyer"] == "Bain Capital")
    record(target, "UPMC", upmc["code"], upmc["reason"], "demote")
    record(target, "Bain Capital", bain["code"], bain["reason"], "promote")
    after = run(target)
    assert (
        "UPMC" in after.review.tier2
        and "Bain Capital" in after.review.tier1
        and len(after.review.tier1) == 10
    )
    codes = {m.buyer: m.code for m in after.review.log}
    assert codes["UPMC"] == "USER_DEMOTE" and codes["Bain Capital"] == "USER_PROMOTE"
    from acquirer_engine.factpack import buyer_conviction

    assert buyer_conviction(after, "Advocate Health") == buyer_conviction(
        res, "Advocate Health"
    )  # no ripple
    for buyer in ("UPMC", "Bain Capital"):  # back to the rules' list (its review is cached)
        record(target, buyer, action="unflag")
    record(target, "Intermountain Health", "demote:FOOTPRINT_OVERLAP", "not now", "dismiss")
    assert ("Intermountain Health", "demote:FOOTPRINT_OVERLAP") in dismissed(target)
    assert "Intermountain Health" not in {m["buyer"] for m in suggested_moves(run(target))["moves"]}
