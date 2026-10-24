"""The Report record: everything one run produced, passed to the summary, figures, CLI and app.

Its own module so the modules that read a report (summary, executive summary, comparison,
charts) can import it without importing report.build, which in turn calls them. That keeps the
package's layers strict: see docs/hierarchy.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acquirer_engine.engine import EngineResult

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """One target's full result: the engine's numbers, the final lists, packs and pages."""

    res: EngineResult
    """The deterministic half: ranking, valuation, ability to pay, review."""
    tier1: list[str]
    """Priority list, in priority order (conviction, then model rank)."""
    tier2: list[str]
    """The bench, in model order plus coverage additions."""
    packs: dict[str, dict]
    """Buyer -> fact pack, for every Tier 1 buyer."""
    results: list[dict] | None
    """One result per Tier 1 page (buyer, page, problems, repairs, usage); None when the pages
    are not in the cache (replay of a target never generated)."""
    approach: str
    """Writing approach used for the pages (W1-W4)."""
    missing: str = ""
    """Why results is None, when it is."""
    exec_summary: dict | None = None
    """Headline plus 3-4 sentences: the LLM version if cached and valid, else the template."""
    exec_from_llm: bool = False
    """True when exec_summary is the LLM-written version."""
