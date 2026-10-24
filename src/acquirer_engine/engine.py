"""Run the deterministic half of the engine: rank, value, test ability to pay, review.

Kept apart from the CLI so the fact packs and the LLM pipeline (Step 4) reuse exactly the same
numbers the banker sees in the `review` command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from acquirer_engine.data import DEFAULT_CSV_PATH, data_end, load_deals
from acquirer_engine.feedback import active_flags, learned_rules
from acquirer_engine.lbo import sponsor_ability_to_pay, strategic_synergy_headroom
from acquirer_engine.model import MODEL_PATH, RankingModel, load_or_fit
from acquirer_engine.profiles import PROFILES_DIR, Fact, load_profiles
from acquirer_engine.ranking import rank_buyers
from acquirer_engine.review import Review, review
from acquirer_engine.valuation import comparable_deals, platform_exit_multiple, valuation_summary

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from acquirer_engine.config import Target


@dataclass
class EngineResult:
    """The deterministic half of a run: ranking, profiles, valuation, ability to pay, review."""

    target: Target
    deals: pd.DataFrame
    ranked: pd.DataFrame
    model: RankingModel
    profiles: dict[str, list[Fact]]
    valuation: dict
    sponsor_lbo: dict
    strategic_synergies: dict
    review: Review
    user_flags: dict  # buyer -> active deal-team flag for this target (feedback.active_flags)


def run(
    target: Target,
    csv_path: Path = DEFAULT_CSV_PATH,
    csv_only: bool = False,
    model_path: Path = MODEL_PATH,
    profiles_dir: Path = PROFILES_DIR,
) -> EngineResult:
    """Return everything the report and the fact packs need, for one target.

    profiles_dir lets an experiment replay with the profiles it was run on (a frozen copy).
    """
    deals = load_deals(csv_path)
    if target.as_of != data_end(deals):
        # The live engine scores as of the data end: the model is fitted on every deal in the file
        # and the signals read every deal, so an earlier as_of would see the future. Back-tests
        # build their inputs from earlier deals only (experiments/).
        raise ValueError(
            f"as_of is {target.as_of}, but the deal file ends in {data_end(deals)}: score the live "
            "target as of the data end (back-tests live in experiments/)"
        )
    ranked = rank_buyers(target, csv_path, model_path)
    model = load_or_fit(deals, csv_path, model_path)  # already saved by rank_buyers: a quick load
    profiles = {} if csv_only else load_profiles(profiles_dir)
    valuation = valuation_summary(comparable_deals(deals, target), target)
    entry = valuation["ev_ebitda"]["p50"]
    margin, growth = target.ebitda_margin_pct / 100, target.revenue_growth_pct / 100
    lbo = sponsor_ability_to_pay(
        target.enterprise_value_mm, entry, growth, margin, platform_exit_multiple(deals, target)
    )
    synergies = strategic_synergy_headroom(target.enterprise_value_mm, entry, margin)
    flags = active_flags(target)
    result = review(
        ranked,
        deals,
        target,
        profiles,
        csv_only=csv_only,
        sponsor_cheque_mm=lbo["equity_cheque"],
        user_flags=flags,
        learned=learned_rules(target.sector),
    )
    return EngineResult(
        target, deals, ranked, model, profiles, valuation, lbo, synergies, result, flags
    )
