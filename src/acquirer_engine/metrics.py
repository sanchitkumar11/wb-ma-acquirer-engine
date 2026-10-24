"""Quality and cost metrics for a set of one-pagers, used by `run` and the architecture
comparison."""

from __future__ import annotations

import itertools
import re
from collections import Counter

from acquirer_engine.onepager import BANNED_PHRASES, CITATION, OnePager, _texts, thesis_overlap

SECTION_FIELDS = (
    "acquirer_overview",
    "strategic_fit_thesis",
    "precedent_activity",
    "valuation_context",
)


# length targets from the one-pager prompt (prompts/onepager_v4.yaml): headline, each box, whole
# page
HEADLINE_WORDS = 25
SECTION_WORDS = (30, 75)
PAGE_WORDS = (350, 500)


def _words(text: str) -> int:
    """Return the word count of text, ignoring [fact ID] citations."""
    return len(re.findall(r"[A-Za-z0-9$%.,'-]+", CITATION.sub(" ", text)))


def page_lengths(page: OnePager) -> dict:
    """Return word counts: headline, each section (title + body), whole page."""
    return dict(
        headline=_words(page.headline),
        sections={f: _words(" ".join(getattr(page, f).points)) for f in SECTION_FIELDS},
        page=sum(_words(t) for t in _texts(page)),
    )


def summarise(results: list[dict], seconds: float) -> dict:
    """Return the run's metrics: pass rates, overlap, conviction spread, generic hits, length,
    cost."""
    pages = {r["buyer"]: OnePager.model_validate(r["page"]) for r in results if r["page"]}
    n = len(results)
    overlaps = {
        f"{a} / {b}": round(thesis_overlap(pages[a], pages[b]), 3)
        for a, b in itertools.combinations(pages, 2)
    }
    first_generic = sum(any("Generic" in p for p in r["first_problems"]) for r in results)
    final_generic = sum(
        p in " ".join(_texts(page)).lower() for page in pages.values() for p in BANNED_PHRASES
    )
    lengths = {b: page_lengths(p) for b, p in pages.items()}

    def in_range(lengths: dict) -> bool:
        """Return True if a page meets every length target the prompt sets."""
        return (
            lengths["headline"] <= HEADLINE_WORDS
            and all(SECTION_WORDS[0] <= w <= SECTION_WORDS[1] for w in lengths["sections"].values())
            and PAGE_WORDS[0] <= lengths["page"] <= PAGE_WORDS[1]
        )

    calls = [u for r in results for u in r["usage"]]
    jargon_first = sum(any("Internal terms" in p for p in r["first_problems"]) for r in results)
    return dict(
        pages=n,
        jargon_hits_first_draft=jargon_first,
        pass_before_repair=sum(not r["first_problems"] for r in results),
        pass_after_repair=sum(not r["problems"] for r in results),
        repairs=sum(r["repairs"] for r in results),
        first_draft_problems=Counter(p.split(":")[0] for r in results for p in r["first_problems"]),
        thesis_overlap_max=max(overlaps.values()) if overlaps else 0.0,
        thesis_overlap_mean=round(sum(overlaps.values()) / len(overlaps), 3) if overlaps else 0.0,
        most_similar_pair=max(overlaps, key=overlaps.get) if overlaps else "",
        conviction=Counter(p.conviction.level for p in pages.values()),
        generic_hits_first_draft=first_generic,
        generic_hits_final=final_generic,
        words_per_page={b: L["page"] for b, L in lengths.items()},
        pages_within_length_targets=sum(in_range(L) for L in lengths.values()),
        lengths=lengths,
        input_tokens=sum(u["input_tokens"] for u in calls),
        output_tokens=sum(u["output_tokens"] for u in calls),
        usd=round(sum(u["usd"] for u in calls), 4),
        usd_per_page=round(sum(u["usd"] for u in calls) / max(n, 1), 4),
        llm_calls=sum(not u["cached"] for u in calls),
        wall_clock_seconds=round(seconds, 1),
    )
