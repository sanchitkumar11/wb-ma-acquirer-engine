"""LangGraph pipeline for the one-pagers: fan out per buyer, write, validate, repair (W3: up to
twice).

Four writing approaches share the same nodes, so the architecture comparison (experiment log
Section 7, Part B) changes only the wiring:

- W1  one call writes all ten one-pagers; each is then validated (no repair).
- W2  one call per buyer, in parallel; each sees only its own fact pack (no repair).
- W3  W2 + every call sees the other Tier 1 buyers' angles + a repair if validation fails, and a
second
      only if the first still fails (pages that pass after one repair never reach it). Default.
- W4  plan, write, critique, revise, then validate and repair once if still failing.
"""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, Annotated, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from pydantic import BaseModel

from acquirer_engine.factpack import adjacent_key, pack_hash
from acquirer_engine.llm import Usage, cache_key, structured_call
from acquirer_engine.onepager import Critique, OnePager, OnePagers, Plan, validate
from acquirer_engine.paths import ROOT
from acquirer_engine.rings import adjacent_sub_sectors, model_words, ring_names

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROMPT_PATH = ROOT / "prompts/onepager_v4.yaml"
PROMPTS = yaml.safe_load(PROMPT_PATH.read_text())
EXAMPLE_BUYER = (
    "Nordic Capital"  # the style example is from its approved page: not shown when writing it
)
APPROACHES = ("W1", "W2", "W3", "W4")
REPAIRS = {
    "W1": 0,
    "W2": 0,
    "W3": 2,
    "W4": 1,
}  # W3 2nd repair added Oct 23: one Health IT page still failed after one


def system_prompt(target_sector: str) -> str:
    """Return the system prompt with the target's own terms, e.g. "adjacent provider sub-sectors
    (...)".

    For a target with no adjacent sub-sector of the same business model (Health IT), the phrase
    names
    the neighbouring sectors instead.
    """
    adj, nearby = model_words(target_sector)[0], adjacent_sub_sectors(target_sector)
    phrase = (
        f"adjacent {adj} sub-sectors ({', '.join(nearby)})"
        if nearby
        else f"neighbouring sectors ({', '.join(ring_names(target_sector, 2))})"
    )
    return PROMPTS["system"].format(adjacent_phrase=phrase, model_adjective=adj)


def peer_line(pack: dict) -> str:
    """Return one line of a buyer's distinguishing numbers, shown to the other buyers' writers (W3,
    W4).

    The same numbers for every buyer, so each writer can see where its buyer genuinely leads and
    frame it favourably without claiming an edge the numbers contradict.
    """
    b, r = pack["buyer"], pack["track_record"]
    k = {d["id"]: d["value"] for d in pack["derived"]}
    region, t = pack["target"]["region"], pack["target"]["sector"]
    line = (
        f"- {b['name']} ({b['type'].lower()}): {r['same_sub_sector_deals']} deals in the target's "
        "sub-sector, "
        f"{r[adjacent_key(t)]} in adjacent {model_words(t)[0]} sub-sectors; "
        f"{r['deals_in_target_region']} in the {region}; "
        f"median deal {r['median_deal']}; median paid {r['median_ev_ebitda_paid']}; closed "
        f"{k['K:closed_share']}"
    )
    if "sponsor_angles" in pack:
        line += f"; open platform {pack['sponsor_angles']['open_platform']}"
    return line + f"; conviction {pack['conviction']['level']}"


# ---------------------------------------------------------------------------
# Per-buyer graph
# ---------------------------------------------------------------------------


class BuyerState(TypedDict, total=False):
    """State of one buyer's branch: its pack, the peer lines, the current draft and its problems."""

    buyer: str
    pack: dict
    peers: str
    peer_names: str
    target_line: str
    approach: str
    alias: str
    allow_paid: bool
    run: int  # repeat number for stability tests; run 1 is the normal cache entry
    plan: dict
    draft: dict
    problems: list[str]
    first_problems: list[str]
    repairs: int
    usage: list[dict]


def _call(
    state: BuyerState, schema: type[BaseModel], user: str, step: str
) -> tuple[BaseModel, dict]:
    """Make one cached, ledgered structured call for this buyer and step."""
    run = state.get("run", 1)  # a repeat run (stability test) gets its own cache entries
    key = cache_key(
        PROMPTS["version"],
        state["alias"],
        state["approach"],
        pack_hash(state["pack"]),
        step if run == 1 else f"{step}-run{run}",
    )
    out, usage = structured_call(
        schema,
        system_prompt(state["pack"]["target"]["sector"]),
        user,
        alias=state["alias"],
        key=key,
        purpose=f"{state['approach']} {step} {state['buyer']}",
        allow_paid=state["allow_paid"],
    )
    return out, usage.__dict__


def _user_prompt(state: BuyerState) -> str:
    """Return the tagged user message: target, peer angles, fact pack, style example, task."""
    peers = state["peers"] if state["approach"] in ("W3", "W4") else "(not shown in this approach)"
    example = PROMPTS.get("example", "") if state["buyer"] != EXAMPLE_BUYER else ""
    return PROMPTS["user"].format(
        target_line=state["target_line"],
        peers=peers,
        buyer=state["buyer"],
        fact_pack=json.dumps(state["pack"], indent=1),
        example=example,
    )


def plan_node(state: BuyerState) -> dict:
    """W4: plan the thesis and the facts to use before writing."""
    user = PROMPTS["plan"].format(
        target_line=state["target_line"],
        peers=state["peers"],
        buyer=state["buyer"],
        fact_pack=json.dumps(state["pack"], indent=1),
    )
    plan, usage = _call(state, Plan, user, "plan")
    return dict(plan=plan.model_dump(), usage=[*state.get("usage", []), usage])


def write_node(state: BuyerState) -> dict:
    """Write the first draft of the page."""
    user = _user_prompt(state)
    if state.get("plan"):
        user += "\n" + PROMPTS["write_from_plan"].format(plan=json.dumps(state["plan"], indent=1))
    page, usage = _call(state, OnePager, user, "write")
    return dict(draft=page.model_dump(), usage=[*state.get("usage", []), usage])


def critique_node(state: BuyerState) -> dict:
    """W4: apply the banker rubric to the draft and revise once."""
    user = PROMPTS["critique"].format(
        buyer=state["buyer"],
        peer_names=state["peer_names"],
        fact_pack=json.dumps(state["pack"], indent=1),
        draft=json.dumps(state["draft"], indent=1),
    )
    critique, usage = _call(state, Critique, user, "critique")
    revise = PROMPTS["revise"].format(
        critique="\n".join(f"- {f}" for f in critique.fixes),
        draft=json.dumps(state["draft"], indent=1),
    )
    page, usage2 = _call(state, OnePager, _user_prompt(state) + "\n" + revise, "revise")
    return dict(draft=page.model_dump(), usage=[*state.get("usage", []), usage, usage2])


def validate_node(state: BuyerState) -> dict:
    """Run the validator; keep the first draft's problems for the pass-rate metric."""
    problems = validate(OnePager.model_validate(state["draft"]), state["pack"])
    out = dict(problems=problems)
    if "first_problems" not in state:
        out["first_problems"] = problems  # before any repair: for the pass-rate metric
    return out


def repair_node(state: BuyerState) -> dict:
    """Rewrite the page with the validator's problems listed."""
    user = (
        _user_prompt(state)
        + "\n"
        + PROMPTS["repair"].format(
            buyer=state["buyer"],
            problems="\n".join(f"- {p}" for p in state["problems"]),
            level=state["pack"]["conviction"]["level"],
            previous=json.dumps(state["draft"], indent=1),
        )
    )
    page, usage = _call(state, OnePager, user, f"repair{state.get('repairs', 0) + 1}")
    return dict(
        draft=page.model_dump(),
        repairs=state.get("repairs", 0) + 1,
        usage=[*state.get("usage", []), usage],
    )


def _after_validate(state: BuyerState) -> str:
    """Return "repair" while problems remain and repairs are left, else END."""
    if state["problems"] and state.get("repairs", 0) < REPAIRS[state["approach"]]:
        return "repair"
    return END


def build_buyer_graph(approach: str) -> CompiledStateGraph:
    """Return the compiled per-buyer graph for W2, W3 or W4."""
    g = StateGraph(BuyerState)
    g.add_node("write", write_node)
    g.add_node("validate", validate_node)
    g.add_node("repair", repair_node)
    if approach == "W4":
        g.add_node("plan", plan_node)
        g.add_node("critique", critique_node)
        g.add_edge(START, "plan")
        g.add_edge("plan", "write")
        g.add_edge("write", "critique")
        g.add_edge("critique", "validate")
    else:
        g.add_edge(START, "write")
        g.add_edge("write", "validate")
    g.add_conditional_edges("validate", _after_validate, ["repair", END])
    g.add_edge("repair", "validate")
    return g.compile()


# ---------------------------------------------------------------------------
# Fan-out graph (all buyers)
# ---------------------------------------------------------------------------


class RunState(TypedDict, total=False):
    """State of the fan-out graph: every Tier 1 pack in, every buyer's result out."""

    packs: dict  # buyer -> fact pack, in Tier 1 order
    target_line: str
    approach: str
    alias: str
    allow_paid: bool
    run: int
    results: Annotated[list, operator.add]


def build_run_graph(approach: str) -> CompiledStateGraph:
    """Return the compiled graph that writes every Tier 1 one-pager with one approach."""
    if approach not in APPROACHES:
        raise ValueError(f"approach must be one of {APPROACHES}")
    buyer_graph = build_buyer_graph("W3" if approach == "W1" else approach)

    def fan_out(state: RunState) -> list[Send]:
        """Send one branch per buyer, each with the other buyers' peer lines."""
        lines = "\n".join(peer_line(p) for p in state["packs"].values())
        sends = []
        for buyer, pack in state["packs"].items():
            peers = "\n".join(x for x in lines.splitlines() if not x.startswith(f"- {buyer} ("))
            sends.append(
                Send(
                    "one_buyer",
                    BuyerState(
                        buyer=buyer,
                        pack=pack,
                        peers=peers,
                        peer_names=", ".join(b for b in state["packs"] if b != buyer),
                        target_line=state["target_line"],
                        approach=approach,
                        alias=state["alias"],
                        allow_paid=state["allow_paid"],
                        run=state.get("run", 1),
                        repairs=0,
                        usage=[],
                    ),
                )
            )
        return sends

    def one_buyer(state: BuyerState) -> dict:
        """Run one buyer's branch and return its result."""
        final = buyer_graph.invoke(state)
        return dict(
            results=[
                dict(
                    buyer=state["buyer"],
                    page=final["draft"],
                    problems=final["problems"],
                    first_problems=final.get("first_problems", final["problems"]),
                    repairs=final.get("repairs", 0),
                    usage=final.get("usage", []),
                )
            ]
        )

    def write_all(state: RunState) -> dict:
        """W1: write all ten pages in one call, then validate each (no repair)."""
        packs = list(state["packs"].values())
        joined = "\n\n".join(
            f"=== {p['buyer']['name']} ===\n{json.dumps(p, indent=1)}" for p in packs
        )
        user = PROMPTS["write_all"].format(
            target_line=state["target_line"], n=len(packs), fact_packs=joined
        )
        combined = pack_hash({"all": [pack_hash(p) for p in packs]})
        key = cache_key(PROMPTS["version"], state["alias"], "W1", combined, "write_all")
        pages, usage = structured_call(
            OnePagers,
            system_prompt(packs[0]["target"]["sector"]),
            user,
            alias=state["alias"],
            key=key,
            purpose="W1 write_all",
            allow_paid=state["allow_paid"],
        )
        by_name = {p.buyer: p for p in pages.items}
        results = []
        for i, pack in enumerate(packs):
            name = pack["buyer"]["name"]
            page = by_name.get(name)  # by name only: a misspelt name must not borrow another page
            problems = (
                validate(page, pack) if page else [f"No page for {name} in the combined output"]
            )
            results.append(
                dict(
                    buyer=name,
                    page=page.model_dump() if page else None,
                    problems=problems,
                    first_problems=problems,
                    repairs=0,
                    usage=[usage.__dict__] if i == 0 else [],
                )
            )
        return dict(results=results)

    g = StateGraph(RunState)
    if approach == "W1":
        g.add_node("write_all", write_all)
        g.add_edge(START, "write_all")
        g.add_edge("write_all", END)
    else:
        g.add_node("one_buyer", one_buyer)
        g.add_conditional_edges(START, fan_out, ["one_buyer"])
        g.add_edge("one_buyer", END)
    return g.compile()


def write_onepagers(
    packs: dict,
    target_line: str,
    approach: str = "W3",
    alias: str = "writer",
    allow_paid: bool = False,
    run: int = 1,
) -> list[dict]:
    """Return one result per buyer, in Tier 1 order: buyer, page (dict), problems, repairs, usage.

    run > 1 writes fresh pages under their own cache keys (the architecture comparison's repeat
    runs).
    """
    out = build_run_graph(approach).invoke(
        dict(
            packs=packs,
            target_line=target_line,
            approach=approach,
            alias=alias,
            allow_paid=allow_paid,
            run=run,
        )
    )
    order = list(packs)
    return sorted(out["results"], key=lambda r: order.index(r["buyer"]))


def total_usage(results: list[dict]) -> Usage:
    """Return summed tokens and cost across results (cached calls count zero)."""
    u = Usage()
    for r in results:
        for x in r["usage"]:
            u.input_tokens += x["input_tokens"]
            u.output_tokens += x["output_tokens"]
            u.usd += x["usd"]
            u.cached = u.cached and x["cached"]
    return u
