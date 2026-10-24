"""Executive summary: 3-4 plain sentences at the top of the summary page, written by the LLM from a
summary fact pack and checked like the one-pagers (numbers and IDs must be in the pack, no internal
jargon, no numbers in the headline). With no cached LLM version, a data-built template is used, so
replay always works.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

from acquirer_engine.errors import LLMError
from acquirer_engine.factpack import pack_hash
from acquirer_engine.llm import cache_key, structured_call
from acquirer_engine.onepager import BANNED_PHRASES, CITATION, JARGON, _numbers, fact_ids
from acquirer_engine.paths import ROOT
from acquirer_engine.summary import motives_key, summary_pack, template_summary

if TYPE_CHECKING:
    from acquirer_engine.report_types import Report

log = logging.getLogger(__name__)
PROMPTS = yaml.safe_load((ROOT / "prompts/summary_v6.yaml").read_text())


HEADLINE_MAX_WORDS = 25
IMPERATIVES = (
    "call",
    "lead",
    "approach",
    "prioritise",
    "prioritize",
    "target",
    "contact",
    "start",
    "focus",
    "engage",
    "pursue",
)


class ExecSummary(BaseModel):
    headline: str = Field(
        description="One descriptive sentence: who the buyers are, then the pricing conclusion "
        "with the defensible multiple; no imperative, no other digits"
    )
    sentences: list[str] = Field(
        description="3-4 sentences, each at most 35 words, ending with [IDs]"
    )


def validate_summary(es: ExecSummary, pack: dict) -> list[str]:
    """Return problems with an executive summary (empty = passes)."""
    problems = []
    text = " ".join([es.headline, *es.sentences])
    bare = CITATION.sub(" ", text)
    invented = sorted(_numbers(bare) - _numbers(json.dumps(pack)))
    if invented:
        problems.append(f"Numbers not in the summary pack: {invented}")
    cited = {i.strip() for g in CITATION.findall(text) for i in g.split(",")}
    if cited - fact_ids(pack):
        problems.append(f"Unknown IDs: {sorted(cited - fact_ids(pack))}")
    first = es.headline.split()[0].lower().strip(",;:") if es.headline.split() else ""
    # the executive summary's headline must carry the defensible multiple (e.g. "15.3x") and no
    # other number;
    # the comparison note, which shares this check, has no single multiple, so its headline has no
    # digits
    multiple = pack.get("valuation", {}).get("defensible_multiple", "")
    missing = bool(multiple) and multiple not in es.headline
    other_digits = re.search(r"\d", es.headline.replace(multiple, "") if multiple else es.headline)
    if (
        missing
        or other_digits
        or first in IMPERATIVES
        or len(es.headline.split()) > HEADLINE_MAX_WORDS
    ):
        rule = (
            f"with the defensible multiple written exactly as {multiple} and no other number"
            if multiple
            else "with no digits"
        )
        problems.append(
            f"Headline must describe the buyers and the pricing conclusion {rule}; no imperative; "
            f"at most {HEADLINE_MAX_WORDS} words"
        )
    if not 3 <= len(es.sentences) <= 4:
        problems.append(f"Needs 3-4 sentences, not {len(es.sentences)}")
    jargon = sorted(
        {
            label
            for pat, label in JARGON.items()
            if label not in ("Tier N",) and re.search(pat, bare)
        }
    )
    if jargon or any(p in bare.lower() for p in [*BANNED_PHRASES, "model"]):
        problems.append(f"Internal or generic terms: {jargon or 'banned phrase'}")
    return problems


def written_summary(
    prompts: dict,
    system: str,
    user: str,
    pack: dict,
    *,
    model: str,
    approach: str,
    purpose: str,
    allow_paid: bool,
) -> dict | None:
    """Return {headline, sentences} written by the LLM that pass validate_summary, else None.

    Write, check, and repair once with the problems listed. Shared by the executive summary and
    the comparison note (approach "summary" or "compare" keeps their cache keys apart). None means
    "use the template": no cached output in replay, no permission to pay, the budget stop, a
    reply that failed to parse twice, or a version that still fails the checks; the reason is
    logged, never raised, because a missing summary must not stop the report.
    """
    try:
        key = cache_key(prompts["version"], model, approach, pack_hash(pack), "write")
        es, _ = structured_call(
            ExecSummary, system, user, alias=model, key=key, purpose=purpose, allow_paid=allow_paid
        )
        problems = validate_summary(es, pack)
        if problems:
            listed = "\n".join(f"- {p}" for p in problems)
            repair = (
                user
                + "\n"
                + prompts["repair"].format(problems=listed, previous=es.model_dump_json())
            )
            key = cache_key(prompts["version"], model, approach, pack_hash(pack), "repair1")
            es, _ = structured_call(
                ExecSummary,
                system,
                repair,
                alias=model,
                key=key,
                purpose=f"{purpose} repair",
                allow_paid=allow_paid,
            )
            problems = validate_summary(es, pack)
    except (LLMError, ValueError) as exc:
        log.info("%s: using the template (%s)", purpose, exc)
        return None
    if problems:
        log.warning("%s: still failing after one repair, using the template: %s", purpose, problems)
        return None
    return dict(headline=es.headline, sentences=[CITATION.sub("", s).strip() for s in es.sentences])


def add_exec_summary(rep: Report, model: str = "writer", allow_paid: bool = False) -> Report:
    """Return the report with exec_summary set: the cached or freshly written LLM version, else the
    template."""
    pack = summary_pack(rep)
    system = PROMPTS["system"].format(motives_key=motives_key(rep.res.target.sector))
    user = PROMPTS["user"].format(summary_pack=json.dumps(pack, indent=1))
    written = written_summary(
        PROMPTS,
        system,
        user,
        pack,
        model=model,
        approach="summary",
        purpose="executive summary",
        allow_paid=allow_paid,
    )
    rep.exec_summary, rep.exec_from_llm = (
        (written, True) if written else (template_summary(rep), False)
    )
    return rep
