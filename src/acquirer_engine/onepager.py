"""One-pager schema and the validator that every LLM output must pass.

The schema is the brief's six sections laid out like a banker's deck: four boxes, each an
action-title
header with 2-3 short bullets, then 2-3 risks as concern / mitigant / reality, then conviction. The
validator is deterministic code, not another LLM: every number in the text must appear in the fact
pack, every cited fact ID must exist, the risk and bullet counts hold, the conviction level matches
the one the fact pack fixed, and banned generic phrases and internal jargon are absent.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Schema (what the LLM must return)
# ---------------------------------------------------------------------------


# The models below are sent to the LLM as a JSON Schema, and pydantic copies a class docstring into
# the schema's description: a docstring here would change the prompt without a version bump. So
# they are documented by their Field descriptions and this comment.
class Section(BaseModel):
    """One box on the page: an action-title header and 2-3 bullets underneath."""

    title: str = Field(description="Action title stating the conclusion, at most 12 words")
    points: list[str] = Field(
        description="2-3 bullets, each at most 25 words and ending with [fact IDs]"
    )


class Risk(BaseModel):
    title: str = Field(description="Short name of the risk, e.g. 'Size stretch'")
    concern: str
    mitigant: str
    reality: str = Field(description="What the evidence says, with [fact IDs]")


class Conviction(BaseModel):
    level: str = Field(description="High, Medium or Low: exactly the level in the fact pack")
    rationale: str = Field(description="1-2 sentences tied to the named basis, with [fact IDs]")


class OnePager(BaseModel):
    buyer: str
    headline: str = Field(description="One sentence: why this buyer, for this target")
    differentiator: str = Field(
        description="What this buyer offers that the other Tier 1 buyers don't"
    )
    why_them: str = Field(
        default="", description="At most 12 words: the angle, for the summary's call list"
    )
    acquirer_overview: Section
    strategic_fit_thesis: Section
    precedent_activity: Section
    valuation_context: Section
    risk_flags: list[Risk] = Field(description="2-3 risks")
    conviction: Conviction


class OnePagers(BaseModel):
    """Wrapper for the one-call-writes-all approach (W1)."""

    items: list[OnePager]


class Plan(BaseModel):
    """Plan step of the plan-write-critique approach (W4)."""

    thesis: str
    differentiator: str
    fact_ids: list[str]
    top_risks: list[str]


class Critique(BaseModel):
    send_unedited: bool
    fixes: list[str]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

BANNED_PHRASES = [
    "leading healthcare company",
    "track record of strategic acquisitions",
    "well positioned",
    "well-positioned",
    "best-in-class",
    "world-class",
    "synergistic",
    "robust platform",
    "proven track record",
    "probability",
    "model score",
]
# Internal terms an MD would not understand: ring and tier labels, reason codes, odds multipliers
JARGON = {
    r"\bRing\s?\d\b": "Ring N",
    r"\bTier\s?\d\b": "Tier N",
    r"\b[Oo]uter[- ]ring\b": "outer ring",
    (
        r"\b(?:in|the|a) Core\b(?! [A-Z])"
        r"|\bCore (?:deal|ring|record|precedent|buyer|strategic|sponsor)s?\b"
    ): "Core (as a label)",
    r"\bfact pack\b": "fact pack",
    r"\b[A-Z]{3,}(?:_[A-Z0-9]+)+\b": "a reason code",
    r"(?<![\w$])x\d+(?:\.\d+)?\b": "an odds multiplier",
}
MAX_RISKS, MIN_POINTS, MAX_POINTS = 3, 2, 3
CITATION = re.compile(r"\[([^\]]+)\]")
NUMBER = re.compile(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?")  # not digits inside a name (Dental365)


def _numbers(text: str) -> set[str]:
    """Return the numbers in `text`, commas removed and trailing zeros normalised (12.80 ->
    12.8)."""
    out = set()
    for raw in NUMBER.findall(text):
        n = raw.replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def _texts(page: OnePager) -> list[str]:
    """Return every free-text field of the one-pager."""
    parts = [page.headline, page.differentiator, page.why_them, page.conviction.rationale]
    for s in (
        page.acquirer_overview,
        page.strategic_fit_thesis,
        page.precedent_activity,
        page.valuation_context,
    ):
        parts += [s.title, *s.points]
    for r in page.risk_flags:
        parts += [r.title, r.concern, r.mitigant, r.reality]
    return parts


def fact_ids(pack: dict) -> set[str]:
    """Return every fact ID in the pack (values of keys named 'id')."""
    found: set[str] = set()

    def walk(x: Any) -> None:
        """Collect every 'id' value, depth first."""
        if isinstance(x, dict):
            if isinstance(x.get("id"), str):
                found.add(x["id"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(pack)
    return found


def validate(page: OnePager, pack: dict) -> list[str]:
    """Return a list of problems (empty = the one-pager passes)."""
    problems = []
    allowed_numbers = _numbers(json.dumps(pack))
    ids = fact_ids(pack)
    text = "\n".join(_texts(page))
    cited = {i.strip() for group in CITATION.findall(text) for i in group.split(",")}
    unknown = sorted(c for c in cited if c not in ids)
    if unknown:
        problems.append(f"Cited fact IDs that are not in the fact pack: {unknown}")
    if not cited:
        problems.append("No fact IDs cited")
    bare = CITATION.sub(
        " ", text
    )  # citations carry deal IDs with digits; check numbers outside them
    invented = sorted(_numbers(bare) - allowed_numbers, key=lambda n: float(n))
    if invented:
        problems.append(
            f"Numbers not in the fact pack (copy numbers exactly, compute nothing): {invented}"
        )
    if not 2 <= len(page.risk_flags) <= MAX_RISKS:
        problems.append(f"Needs 2 to {MAX_RISKS} risk flags, not {len(page.risk_flags)}")
    for name in (
        "acquirer_overview",
        "strategic_fit_thesis",
        "precedent_activity",
        "valuation_context",
    ):
        n = len(getattr(page, name).points)
        if not MIN_POINTS <= n <= MAX_POINTS:
            problems.append(f"{name} needs {MIN_POINTS}-{MAX_POINTS} bullets, not {n}")
    if len(page.why_them.split()) > 12 or not page.why_them.strip():
        problems.append(f"why_them must be 1-12 words, not {len(page.why_them.split())}")
    bare_text = CITATION.sub(" ", text)
    jargon = sorted({label for pattern, label in JARGON.items() if re.search(pattern, bare_text)})
    if jargon:
        problems.append(f"Internal terms an MD wouldn't use (say it in plain English): {jargon}")
    for r in page.risk_flags:
        if not (r.concern.strip() and r.mitigant.strip() and r.reality.strip()):
            problems.append(f"Risk '{r.title}' is missing concern, mitigant or reality")
    if page.conviction.level != pack["conviction"]["level"]:
        problems.append(
            f"Conviction must be {pack['conviction']['level']}, not {page.conviction.level}"
        )
    lowered = text.lower()
    generic = [p for p in BANNED_PHRASES if p in lowered]
    if generic:
        problems.append(f"Generic or banned phrases: {generic}")
    if pack["buyer"]["name"].split()[0].lower() not in lowered:
        problems.append("The one-pager never names the buyer")
    return problems


def thesis_overlap(a: OnePager, b: OnePager) -> float:
    """Return the share of word 3-grams two strategic-fit theses have in common (Jaccard, 0-1)."""

    def grams(page: OnePager) -> set[tuple[str, ...]]:
        """Return the word 3-grams of a page's thesis text."""
        words = re.findall(
            r"[a-z]+", CITATION.sub(" ", " ".join(page.strategic_fit_thesis.points)).lower()
        )
        return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}

    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb) if ga | gb else 0.0
