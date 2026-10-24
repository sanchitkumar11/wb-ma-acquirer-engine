"""Buyer profiles: every fact is sourced and dated, back-tests can't see later facts,
and the profile-based review gives the agreed Tier 1."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from acquirer_engine.cli import main
from acquirer_engine.config import load_target
from acquirer_engine.profiles import facts_as_of, load_profiles, profile_notes
from acquirer_engine.ranking import rank_buyers

ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = ROOT / "data/profiles"
GOOD_FACT = (
    "- kind: status\n  value: Filed for Chapter 11\n  as_of: '2024-05-06'\n"
    "  source: https://example.org/steward\n  status: defunct\n"
)


def _write(tmp_path, fact_yaml):
    (tmp_path / "x.yaml").write_text(
        "name: Test Buyer\nbuyer_type: Strategic\nfacts:\n" + fact_yaml
    )
    return tmp_path


def test_all_profiles_load_and_every_fact_is_sourced_and_dated():
    profiles = load_profiles(PROFILES_DIR)
    assert len(profiles) >= 30
    for facts in profiles.values():
        for fact in facts:
            assert fact.source.startswith("https://") and fact.as_of <= date.today()


@pytest.mark.parametrize(
    "broken, message",
    [
        (GOOD_FACT.replace("  source: https://example.org/steward\n", ""), "source"),
        (GOOD_FACT.replace("  as_of: '2024-05-06'\n", ""), "as_of"),
        (GOOD_FACT.replace("Filed for Chapter 11", "''"), "value"),
        (
            GOOD_FACT.replace("https://example.org/steward", "https://www.sec.gov/x.htm"),
            "accession",
        ),
    ],
)
def test_fact_without_source_date_or_accession_is_refused(tmp_path, broken, message):
    with pytest.raises(ValueError, match=message):
        load_profiles(_write(tmp_path, broken))


def test_back_tests_only_see_earlier_facts():
    profiles = load_profiles(PROFILES_DIR)
    assert profiles["Steward Health Care"]
    assert (
        facts_as_of(profiles, date(2024, 5, 6))["Steward Health Care"] == []
    )  # filed that day: not yet known
    assert facts_as_of(profiles, date(2024, 5, 7))["Steward Health Care"]


def test_profile_review_gives_the_agreed_tier_1():
    ranked = rank_buyers(
        load_target(ROOT / "configs/target_default.yaml"),
        ROOT / "data/ma_transactions_500.csv",
        model_path=ROOT / "models/ranking_model.json",
    )
    profiles, buyers = load_profiles(PROFILES_DIR), set(ranked.acquirer)
    moved = {"STATUS_DEFUNCT", "STATUS_ACQUIRED", "ENTITY_DUPLICATE", "MANDATE_MISMATCH"}
    tier1 = [
        b
        for b in ranked.acquirer
        if not {c for c, _, _ in profile_notes(b, profiles, "Healthcare Services", buyers)} & moved
    ][:10]
    assert tier1 == [
        "Advocate Health",
        "Nordic Capital",
        "UPMC",
        "General Atlantic",
        "Blackstone",
        "Welsh Carson",
        "GTCR",
        "TPG Capital",
        "Advent International",
        "Intermountain Health",
    ]


def test_csv_only_runs_and_says_so(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    main(["rank", "--csv-only", "--top", "5"])
    out = capsys.readouterr().out
    assert "CSV-only mode" in out and "STATUS_DEFUNCT" not in out
