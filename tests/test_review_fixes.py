"""Regression tests for the bugs found in the Oct 24 code review, one class per area.

Each test reproduces the failure the review found (a crash, a suggestion that could never fire, a
budget race, a leaked setting) and checks it stays fixed. All offline: cached LLM outputs only.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from acquirer_engine import llm
from acquirer_engine.cli import default_report_path
from acquirer_engine.compare import conviction_points
from acquirer_engine.config import load_target
from acquirer_engine.engine import run
from acquirer_engine.errors import BudgetExceededError, EngineError, LLMError, NotCachedError
from acquirer_engine.feedback import history_check, record, suggestions
from acquirer_engine.report import build

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/target_default.yaml"
TECH = ROOT / "configs/target_healthit.yaml"


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Temporary feedback and learned-rules files, so tests never touch the real ones."""
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("LEARNED_RULES_PATH", str(tmp_path / "learned_rules.yaml"))
    return tmp_path


# --- targets the app's form allows ---


class TestOutOfRangeTargets:
    def test_zero_margin_is_refused_with_a_reason(self):
        """A 0% margin used to divide by zero in the quick LBO."""
        with pytest.raises(ValueError, match="ebitda_margin_pct"):
            replace(load_target(DEFAULT), ebitda_margin_pct=0.0)

    def test_too_few_comparable_deals_is_refused_with_a_reason(self):
        """A $5B dental target has no comparables: it used to crash on round(NaN)."""
        target = replace(load_target(DEFAULT), sector="Dental", enterprise_value_mm=5000)
        with pytest.raises(ValueError, match="comparable"):
            run(target)

    def test_unknown_sub_sector_is_refused_on_construction(self):
        """Validation lives on Target itself, so the app's direct construction is checked too."""
        with pytest.raises(ValueError, match="Unknown sector"):
            replace(load_target(DEFAULT), sector="Veterinary")


# --- feedback loop ---


class TestCantPaySuggestion:
    def test_two_cant_pay_flags_with_a_shared_capacity_flag_suggest_a_rule(self, files):
        """Flagged buyers used to lose their capacity flags, so this rule could never fire."""
        target = load_target(DEFAULT)
        for buyer in ("Bain Capital", "New Mountain Capital"):  # both: fund size not researched
            record(target, buyer, "can't pay")
        res = run(target)
        found = suggestions(target, res.user_flags, res.deals, res.profiles, res.review.flags)
        assert [s["kind"] for s in found] == ["capacity_flag"]
        assert found[0]["value"] == "FUND_SIZE_UNKNOWN"
        assert "can't be checked against the file" in found[0]["history"]  # no reassuring zero

    def test_a_rule_the_file_can_replay_reports_its_count(self):
        target = load_target(DEFAULT)
        res = run(target)
        check = history_check("capacity_flag", "CAPACITY_STRETCH", target, res.deals, res.profiles)
        assert check is not None and check[1] > 0


# --- LLM layer ---


class TestLLMGuards:
    def test_errors_share_one_hierarchy(self):
        assert issubclass(NotCachedError, LLMError) and issubclass(LLMError, EngineError)
        assert issubclass(BudgetExceededError, LLMError)

    def test_replay_only_is_scoped_and_restores_the_provider(self, monkeypatch):
        """build(replay=True) used to set LLM_PROVIDER for the rest of the process."""
        monkeypatch.setenv("LLM_PROVIDER", "databricks")
        with llm.replay_only():
            assert llm.provider() == "replay"
        assert llm.provider() == "databricks"
        build(load_target(DEFAULT), replay=True)
        assert llm.provider() == "databricks"

    def test_parallel_calls_cannot_all_pass_the_budget_check(self, monkeypatch):
        """Each in-flight call reserves its worst case under a lock: the second is refused."""
        worst = 2 * llm.cost_usd("writer", 1000, llm.MAX_OUTPUT_TOKENS)
        monkeypatch.setenv("LLM_BUDGET_USD", str(llm.spent_usd() + 1.5 * worst))
        inside, release, refused = threading.Event(), threading.Event(), []

        def first_call() -> None:
            with llm._budget_reservation("writer", 3000, attempts=2):
                inside.set()
                release.wait(5)

        thread = threading.Thread(target=first_call)
        thread.start()
        inside.wait(5)
        try:
            with llm._budget_reservation("writer", 3000, attempts=2):
                pass
        except BudgetExceededError:
            refused.append(True)
        release.set()
        thread.join()
        assert refused and llm._reserved_usd == 0

    def test_a_non_default_provider_gets_its_own_cache_keys(self, monkeypatch):
        """Databricks keys are unchanged (the committed cache replays); Anthropic keys differ."""
        monkeypatch.setenv("LLM_PROVIDER", "databricks")
        default = llm.cache_key("onepager_v4", "writer", "W3", "abc", "write")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert llm.cache_key("onepager_v4", "writer", "W3", "abc", "write") != default
        monkeypatch.setenv("LLM_PROVIDER", "replay")
        monkeypatch.setenv("LLM_CACHE_TAG", "anthropic")
        assert llm.cache_key("onepager_v4", "writer", "W3", "abc", "write") != default


# --- report and CLI ---


class TestReportAndCLI:
    def test_only_a_buyer_outside_tier1_is_refused_with_a_reason(self):
        """--only used to raise KeyError deep in build()."""
        with pytest.raises(ValueError, match="Not in Tier 1"):
            build(load_target(DEFAULT), replay=True, only="Nobody Capital")

    def test_a_one_buyer_run_never_writes_over_the_sample(self):
        path = default_report_path(DEFAULT, only="Nordic Capital")
        assert "samples" not in path.parts and ROOT / "out" in path.parents

    def test_an_as_of_before_the_data_end_is_refused(self):
        with pytest.raises(ValueError, match="data end"):
            run(replace(load_target(DEFAULT), as_of="2022Q4"))


# --- comparison ---


class TestComparison:
    def test_the_same_concern_matches_across_targets_despite_its_numbers(self):
        """A concern used to be keyed with its cheque amount, which depends on the target's EV."""
        a = conviction_points(
            ["for: x", "for: y", "against: Equity cheque of $239M is over 10% of $5B"]
        )
        b = conviction_points(
            ["for: x", "for: y", "against: Equity cheque of $53M is over 10% of $5B"]
        )
        assert a.keys() == b.keys()

    def test_compare_page_handles_the_same_target_twice(self, monkeypatch):
        """Nothing is unique when both sides are the same target: used to raise KeyError."""
        st_testing = pytest.importorskip("streamlit.testing.v1")
        monkeypatch.chdir(ROOT)
        monkeypatch.setenv("APP_START_PAGE", "compare")
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        target_b = next(box for box in at.selectbox if box.label == "Target B")
        target_b.set_value("healthit").run()
        assert not at.exception
        assert any("on the other list too" in i.value for i in at.info)
