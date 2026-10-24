"""Comparison mode: shared and unique buyers, the conviction point behind each change, and a
checked note."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from acquirer_engine.compare import (
    comparison_note,
    comparison_pack,
    shared_rows,
    template_note,
    unique_rows,
)
from acquirer_engine.config import load_target
from acquirer_engine.execsummary import ExecSummary, validate_summary
from acquirer_engine.report import build

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pair():
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        mp.setenv(
            "FEEDBACK_PATH", str(ROOT / "tests/fixtures/no_feedback.jsonl")
        )  # no flags: the committed lists
        return tuple(
            build(load_target(ROOT / f"configs/target_{n}.yaml"), replay=True)
            for n in ("healthit", "dental")
        )


def test_shared_and_unique_buyers(pair):
    a, b = pair
    shared = {r["buyer"] for r in shared_rows(a, b)}
    assert shared == set(a.tier1) & set(b.tier1) == {"Blackstone", "TPG Capital"}
    assert {r["buyer"] for r in unique_rows(a, b)} == set(a.tier1) - shared and len(
        unique_rows(b, a)
    ) == 8
    where = {r["buyer"]: r["in_other_list"] for r in unique_rows(a, b)}
    assert where["Francisco Partners"] == "Tier 2" and where["Evolent Health"] == "not listed"


def test_conviction_change_names_the_point_behind_it(pair):
    """TPG: Medium for the $300M target, High for the $80M one, because of the deal-size concern
    only."""
    tpg = next(r for r in shared_rows(*pair) if r["buyer"] == "TPG Capital")
    assert (tpg["conviction_a"], tpg["conviction_b"], tpg["changed"]) == ("Medium", "High", True)
    assert [d["point"] for d in tpg["points_that_differ"]] == ["Concern: deal size"]
    assert "$88M is under half the $300M EV" in tpg["points_that_differ"][0]["a"]


def test_note_is_cached_and_passes_the_checks(pair):
    note, from_llm = comparison_note(*pair)
    assert from_llm
    pack = comparison_pack(*pair)
    assert (
        validate_summary(ExecSummary(headline=note["headline"], sentences=note["sentences"]), pack)
        == []
    )
    assert "15.4x" in " ".join(
        note["sentences"]
    )  # Dental's defensible multiple, not its 13.7x median


def test_template_note_also_passes_the_checks(pair):
    pack = comparison_pack(*pair)
    note = template_note(pack)
    assert validate_summary(ExecSummary(**note), pack) == []


def test_cli_compare_replays(tmp_path):
    env = {**os.environ, "LLM_PROVIDER": "replay", "FEEDBACK_PATH": str(tmp_path / "none.jsonl")}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "acquirer_engine",
            "compare",
            "configs/target_healthit.yaml",
            "configs/target_dental.yaml",
            "--out",
            str(tmp_path / "c.md"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert (
        "Shared Tier 1: Blackstone (High / High), TPG Capital (Medium / High)" in r.stdout
        and "$0.000" in r.stdout
    )
    assert (
        "Note: LLM" in r.stdout and "## Shared Tier 1 buyers (2)" in (tmp_path / "c.md").read_text()
    )


def test_app_compare_page_renders(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("APP_START_PAGE", "compare")
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Shared Tier 1 buyers" in m.value for m in at.markdown)
    assert not any(
        "Template note" in c.value for c in at.caption
    )  # the committed pair's note is cached
