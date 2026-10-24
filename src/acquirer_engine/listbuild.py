"""Three ways to build the Tier 1 list, swappable so the architecture comparison reuses them.

- A1  model + rules: the deterministic banker review (review.py). Default; no LLM.
- A2  A1, then an LLM reviewer makes judgment calls. It may only promote or demote, each move needs
a
      reason code and fact IDs from the cards, and code rejects any move without valid evidence.
- A3  a tool-using agent: an LLM gets the engine's tools (ranked candidates, buyer history, buyer
      profile, comparable deals, quick LBO) and builds the list itself. Code rejects any name that
      is not a buyer in the file.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from acquirer_engine.feedback import dismissed
from acquirer_engine.llm import Usage, cache_dir, cache_key, structured_call, tool_call
from acquirer_engine.paths import ROOT
from acquirer_engine.rings import RING_NAME, model_words
from acquirer_engine.thresholds import TIER1_SIZE

if TYPE_CHECKING:
    from acquirer_engine.engine import EngineResult

PROMPTS = yaml.safe_load((ROOT / "prompts/listbuild_v1.yaml").read_text())
APPROACHES = ("A1", "A2", "A3")
AGENT_MAX_STEPS = 8

# ---------------------------------------------------------------------------
# A2: LLM reviewer on top of the rules
# ---------------------------------------------------------------------------


class ReviewerMove(BaseModel):
    buyer: str
    action: str = Field(description="promote or demote")
    code: str
    fact_ids: list[str]
    reason: str


class ReviewerMoves(BaseModel):
    moves: list[ReviewerMove]


def _card(res: EngineResult, buyer: str, tier: str) -> tuple[str, dict[str, str]]:
    """Return (a compact evidence card for one buyer, fact ID -> fact text for every fact on it).

    The card text is what the reviewer reads, and its hash keys the reviewer's cache: keep its
    wording stable. The dict lets a caller show a cited fact without re-parsing the card.
    """
    r = res.ranked.set_index("acquirer").loc[buyer]
    own = res.deals[res.deals.acquirer == buyer]
    cited, lines = {}, []
    facts = [
        (
            f"F:{buyer}:deals",
            f"{len(own)} deals, {int(r.n_model)} in {model_words(res.target.sector)[1]}; median "
            f"${own.deal_size_mm.median():.0f}M; "
            f"{int((own.geography == res.target.region).sum())} in {res.target.region}",
        )
    ]
    facts += [(f"F:{buyer}:{code}", note) for code, note in res.review.flags.get(buyer, [])]
    facts += [
        (f"P:{buyer}:{i + 1}", f"{f.kind}: {f.value} (as of {f.as_of})")
        for i, f in enumerate(res.profiles.get(buyer, []))
    ]
    for fid, text in facts:
        cited[fid] = text
        lines.append(f"    [{fid}] {text}")
    head = (
        f"- {buyer} | {tier} | {'Sponsor' if r.sponsor else 'Strategic'} | model rank "
        f"{int(r.model_rank)} | "
        f"likelihood {100 * r.share:.1f}% | {RING_NAME[int(r.closest_ring)]}"
    )
    return "\n".join([head, *lines]), cited


def a2_reviewer(
    res: EngineResult, alias: str = "writer", allow_paid: bool = False, run: int = 1
) -> dict:
    """Return dict(tier1, tier2, accepted, rejected, usage) after the LLM reviewer's moves.

    run > 1 asks again under its own cache key (the stability test across repeat runs).
    """
    tier1, tier2 = list(res.review.tier1), list(res.review.tier2)
    cards, valid_ids = [], set()
    for buyer in tier1 + tier2:
        text, cited = _card(res, buyer, "Tier 1" if buyer in tier1 else "Tier 2")
        cards.append(text)
        valid_ids |= set(cited)
    user = PROMPTS["reviewer_user"].format(target_line=res.target.label(), cards="\n".join(cards))
    key = cache_key(
        PROMPTS["version"],
        alias,
        "A2",
        hashlib.sha256(user.encode()).hexdigest(),
        "review" if run == 1 else f"review-run{run}",
    )
    out, usage = structured_call(
        ReviewerMoves,
        PROMPTS["reviewer_system"],
        user,
        alias=alias,
        key=key,
        purpose="A2 reviewer",
        allow_paid=allow_paid,
    )
    accepted, rejected, seen = [], [], set()
    for m in out.moves:
        if (m.buyer, m.action) in seen:  # the same move twice would remove the buyer twice
            continue
        seen.add((m.buyer, m.action))
        ok = (
            m.action in ("promote", "demote")
            and m.fact_ids
            and set(m.fact_ids) <= valid_ids
            and (
                (m.action == "promote" and m.buyer in tier2)
                or (m.action == "demote" and m.buyer in tier1)
            )
        )
        (accepted if ok else rejected).append(m.model_dump())
    for m in accepted:
        if m["action"] == "demote":
            tier1.remove(m["buyer"])
            tier2.insert(0, m["buyer"])
    for m in accepted:
        if m["action"] == "promote" and m["buyer"] in tier2:
            tier2.remove(m["buyer"])
            tier1.append(m["buyer"])
    rank = dict(zip(res.ranked.acquirer, res.ranked.model_rank, strict=True))
    tier1.sort(key=rank.get)
    promoted = {m["buyer"] for m in accepted if m["action"] == "promote"}
    while (
        len(tier1) > TIER1_SIZE
    ):  # a promotion displaces the lowest-ranked buyer that was not promoted
        out = max((b for b in tier1 if b not in promoted), key=rank.get)
        tier1.remove(out)
        tier2.insert(0, out)
    demoted = {m["buyer"] for m in accepted if m["action"] == "demote"}
    ruled_out = {
        m.buyer for m in res.review.log if m.action == "to Tier 2"
    }  # moved by a rule (e.g. mandate): never refill
    fill = sorted(
        (b for b in tier2 if b not in demoted | ruled_out), key=rank.get
    )  # more demotions: fill in model order
    while len(tier1) < TIER1_SIZE and fill:
        b = fill.pop(0)
        tier2.remove(b)
        tier1.append(b)
    return dict(
        tier1=tier1, tier2=tier2, accepted=accepted, rejected=rejected, usage=usage.__dict__
    )


def suggested_moves(res: EngineResult, alias: str = "writer", allow_paid: bool = False) -> dict:
    """Return the LLM reviewer's moves as suggestions on top of the rules' list (A1 stays the list).

    Keys: moves (buyer, action, code, reason, facts: the cited fact IDs with their text), rejected
    (moves code refused for missing evidence), usage. Moves the deal team already accepted or
    dismissed (feedback.active_flags / dismissed) are left out. Accept or reject goes through the
    feedback loop, so a suggestion is never applied by itself. Raises NotCachedError /
    PaidCallNotAllowedError
    like any LLM call.
    """
    out = a2_reviewer(res, alias, allow_paid)
    texts = {}
    for buyer in res.review.tier1 + res.review.tier2:
        _, cited = _card(res, buyer, "Tier 1" if buyer in res.review.tier1 else "Tier 2")
        texts |= cited
    done = dismissed(res.target)
    moves = [
        dict(m, facts=[(i, texts.get(i, "")) for i in m["fact_ids"]])
        for m in out["accepted"]
        if m["buyer"] not in res.user_flags
        and (m["buyer"], f"{m['action']}:{m['code']}") not in done
    ]
    return dict(moves=moves, rejected=len(out["rejected"]), usage=out["usage"])


# ---------------------------------------------------------------------------
# A3: tool-using agent
# ---------------------------------------------------------------------------


class AgentPick(BaseModel):
    buyer: str
    reason: str


class AgentAnswer(BaseModel):
    tier1: list[AgentPick]


def _tools(res: EngineResult) -> list:
    """Return the engine functions the agent may call, each returning plain text."""
    ranked = res.ranked

    @tool
    def ranked_candidates(top_n: int = 30) -> str:
        """Model-ranked buyers for this target: rank, name, type, likelihood, ring, review
        status."""
        removed = {m.buyer: m.code for m in res.review.log if m.action in ("removed", "merged")}
        rows = [
            f"#{int(r.model_rank)} {r.acquirer} | {'Sponsor' if r.sponsor else 'Strategic'} | "
            f"{100 * r.share:.1f}% | "
            f"{RING_NAME[int(r.closest_ring)]}"
            + (f" | {removed[r.acquirer]}" if r.acquirer in removed else "")
            for r in ranked.head(min(top_n, 60)).itertuples()
        ]
        return "\n".join(rows)

    @tool
    def buyer_history(buyer: str) -> str:
        """Every deal this buyer did in the file: year, sub-sector, size, type, region, EV/EBITDA,
        outcome."""
        own = res.deals[res.deals.acquirer == buyer]
        if own.empty:
            return f"No buyer named {buyer!r} in the file."
        return "\n".join(
            f"{d.transaction_id} {d.deal_year} {d.sector} ${d.deal_size_mm:.0f}M {d.deal_type} "
            f"{d.geography} "
            f"{d.ev_ebitda_multiple}x {d.outcome}"
            for d in own.itertuples()
        )

    @tool
    def buyer_profile(buyer: str) -> str:
        """Dated, sourced real-world facts about this buyer (status, mergers, mandate, size), if
        researched."""
        facts = res.profiles.get(buyer, [])
        return (
            "\n".join(f"{f.kind}: {f.value} (as of {f.as_of})" for f in facts) or "Not researched."
        )

    @tool
    def valuation_and_ability_to_pay() -> str:
        """Comparable-deal multiples for the target, the sponsor quick LBO and strategic synergy
        headroom."""
        v, lbo, syn = res.valuation, res.sponsor_lbo, res.strategic_synergies
        return (
            f"Comps {v['n_comps']}: EV/EBITDA p25 {v['ev_ebitda']['p25']}x, median "
            f"{v['ev_ebitda']['p50']}x, p75 {v['ev_ebitda']['p75']}x. "
            f"Sponsor LBO at the ask: IRR {100 * lbo['irr_at_ask']:.1f}%, max price "
            f"${lbo['max_price_same_exit']}M "
            f"(${lbo['max_price_platform_exit']}M as a platform add-on). Strategic synergy "
            "headroom "
            f"${syn['synergy_value_low']}-{syn['synergy_value_high']}M."
        )

    return [ranked_candidates, buyer_history, buyer_profile, valuation_and_ability_to_pay]


def _engine_fingerprint(res: EngineResult) -> str:
    """Return a hash of what the agent's tools can see: the target, the ranking, the review, the
    flags.

    The agent's cache key: a changed CSV, profile, model or deal-team flag must not replay an
    old list.
    """
    seen = dict(
        target=res.target.label(),
        ranked=[(r.acquirer, round(float(r.s), 6)) for r in res.ranked.head(60).itertuples()],
        review=[(m.buyer, m.code) for m in res.review.log],
        flags=sorted((b, line["action"]) for b, line in res.user_flags.items()),
        profiles=sorted((b, len(f)) for b, f in res.profiles.items()),
    )
    return hashlib.sha256(json.dumps(seen, sort_keys=True).encode()).hexdigest()


def a3_agent(res: EngineResult, alias: str = "writer", allow_paid: bool = False) -> dict:
    """Return dict(tier1, reasons, rejected, refilled, steps, usage) from the tool-using agent
    (cached).

    Code checks the answer: only buyers still on the review's list may be named (not removed,
    merged or flagged), duplicates are dropped, and a list shorter than TIER1_SIZE is refilled
    from the review's list in model order (logged in `refilled`).
    """
    off_list = {m.buyer for m in res.review.log if m.action in ("removed", "merged", "flagged")}
    names = set(res.ranked.acquirer) - off_list
    key = cache_key(PROMPTS["version"], alias, "A3", _engine_fingerprint(res), "agent")
    path = cache_dir() / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["output"]
    tools = _tools(res)
    by_name = {t.name: t for t in tools}
    messages = [
        SystemMessage(PROMPTS["agent_system"].format(max_steps=AGENT_MAX_STEPS)),
        HumanMessage(PROMPTS["agent_user"].format(target_line=res.target.label())),
    ]
    total, steps = Usage(cached=False), 0
    while steps < AGENT_MAX_STEPS:
        reply, usage = tool_call(
            messages,
            tools,
            alias=alias,
            purpose=f"A3 agent step {steps + 1}",
            allow_paid=allow_paid,
        )
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens
        total.usd += usage.usd
        messages.append(reply)
        if not getattr(reply, "tool_calls", None):
            break
        for call in reply.tool_calls:
            steps += 1
            result = (
                by_name[call["name"]].invoke(call["args"])
                if call["name"] in by_name
                else "Unknown tool."
            )
            messages.append(ToolMessage(str(result), tool_call_id=call["id"]))
    transcript = "\n".join(
        str(m.content) for m in messages if isinstance(m, (AIMessage, ToolMessage))
    )
    final_key = cache_key(
        PROMPTS["version"], alias, "A3", hashlib.sha256(transcript.encode()).hexdigest(), "final"
    )
    answer, usage = structured_call(
        AgentAnswer,
        PROMPTS["agent_system"].format(max_steps=0),
        PROMPTS["agent_user"].format(target_line=res.target.label())
        + "\n\nYour research so far:\n"
        + transcript
        + "\n\nGive the final Tier 1 now.",
        alias=alias,
        key=final_key,
        purpose="A3 final answer",
        allow_paid=allow_paid,
    )
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    total.usd += usage.usd
    picks = list({p.buyer: p for p in answer.tier1 if p.buyer in names}.values())[:TIER1_SIZE]
    chosen = [p.buyer for p in picks]
    refill = [b for b in res.review.tier1 + res.review.tier2 if b not in chosen]
    refilled = refill[: TIER1_SIZE - len(chosen)]
    out = dict(
        tier1=chosen + refilled,
        reasons={p.buyer: p.reason for p in picks},
        rejected=[p.buyer for p in answer.tier1 if p.buyer not in names],
        refilled=refilled,
        steps=steps,
        usage=total.__dict__,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(purpose="A3 agent", output=out), indent=1))
    return out


def build_list(
    res: EngineResult, approach: str = "A1", alias: str = "writer", allow_paid: bool = False
) -> dict:
    """Return dict(tier1, tier2, ...) for the chosen list-building approach."""
    if approach == "A1":
        return dict(tier1=res.review.tier1, tier2=res.review.tier2)
    if approach == "A2":
        return a2_reviewer(res, alias, allow_paid)
    if approach == "A3":
        out = a3_agent(res, alias, allow_paid)
        out["tier2"] = [b for b in res.review.tier1 + res.review.tier2 if b not in out["tier1"]]
        return out
    raise ValueError(f"approach must be one of {APPROACHES}")
