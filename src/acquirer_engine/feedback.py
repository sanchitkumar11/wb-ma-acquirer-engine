"""Deal-team feedback: "flag as irrelevant", applied on the next run, and rules suggested from it.

A flag is one line in feedback/feedback.jsonl (target hash, buyer, reason, note, timestamp, and
whether it flags or unflags). The latest line per buyer wins, so flagging twice is the same as
once and unflag undoes it. The review applies each active flag as a move with reason code
USER_FLAG; Tier 1 then refills from the next eligible buyer.

Learning is never silent. When two or more flags share a reason and the flagged buyers share a
property an existing rule acts on, suggestions() proposes that rule with a check against history
(how many real buyers in the target's sub-sector it would have caught). Only an explicit accept
writes it to configs/learned_rules.yaml, and only for that sub-sector. "Conflict / already in
talks" and "other" are about one deal, so they never become rules.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from acquirer_engine.paths import ROOT
from acquirer_engine.rings import region_similarity
from acquirer_engine.thresholds import FUND_CONCENTRATION, STRETCH_SHARE

if TYPE_CHECKING:
    import pandas as pd

    from acquirer_engine.config import Target
    from acquirer_engine.profiles import Fact

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEEDBACK_PATH = ROOT / "feedback/feedback.jsonl"
LEARNED_PATH = ROOT / "configs/learned_rules.yaml"
REASONS = ("off-mandate", "can't pay", "conflict / already in talks", "wrong geography", "other")
MIN_SHARED_FLAGS = 2
CAPACITY_CODES = ("CAPACITY_STRETCH", "CAPACITY_FUND", "FUND_SIZE_UNKNOWN")
CAPACITY_WORDS = {
    "CAPACITY_STRETCH": "whose biggest deal is under half the ask",
    "CAPACITY_FUND": f"whose equity cheque is over {FUND_CONCENTRATION:.0%} of the fund",
    "FUND_SIZE_UNKNOWN": "with no researched fund size",
}


def feedback_path() -> Path:
    """Return the feedback file (FEEDBACK_PATH in the environment overrides, e.g. for a demo)."""
    return Path(os.environ.get("FEEDBACK_PATH", str(FEEDBACK_PATH)))


def learned_path() -> Path:
    """Return the learned-rules file (LEARNED_RULES_PATH in the environment overrides)."""
    return Path(os.environ.get("LEARNED_RULES_PATH", str(LEARNED_PATH)))


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def target_hash(target: Target) -> str:
    """Return a short, stable hash of the target profile: flags apply to this exact target only."""
    return hashlib.sha256(json.dumps(asdict(target), sort_keys=True).encode()).hexdigest()[:12]


ACTIONS = ("flag", "unflag", "promote", "demote", "dismiss")
ACTION_WORDS = {"flag": "flagged", "promote": "moved to Tier 1", "demote": "moved to Tier 2"}
"""How an active deal-team action reads in a list (the app's flags panel, the review log)."""


def record(
    target: Target, buyer: str, reason: str = "", note: str = "", action: str = "flag"
) -> dict:
    """Append one deal-team action to the feedback file and return it.

    flag takes the buyer off the list; promote and demote accept an LLM reviewer suggestion (reason
    is its reason code); dismiss rejects one (reason "promote:CODE" or "demote:CODE", so it is not
    shown again); unflag clears the buyer's latest flag, promote or demote. Raises ValueError for an
    unknown action or reason, a flag with no reason, or "other" with no note.
    """
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, not {action!r}")
    if action == "flag" and reason not in REASONS:
        raise ValueError(f"Unknown reason {reason!r}; choose from {list(REASONS)}")
    if action == "flag" and reason == "other" and not note.strip():
        raise ValueError('Reason "other" needs a note saying why')
    line = dict(
        target_hash=target_hash(target),
        target=target.label(),
        buyer=buyer,
        action=action,
        reason="" if action == "unflag" else reason,
        note=note.strip(),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(line) + "\n")
    return line


def _lines(target: Target) -> list[dict]:
    """Return every feedback line for this target, oldest first."""
    path, key = feedback_path(), target_hash(target)
    if not path.exists():
        return []
    return [
        line
        for line in (json.loads(t) for t in path.read_text().splitlines() if t.strip())
        if line["target_hash"] == key
    ]


def active_flags(target: Target) -> dict[str, dict]:
    """Return buyer -> its latest flag, promote or demote for this target (unflag clears; dismiss
    is separate)."""
    latest = {}
    for line in _lines(target):
        if line["action"] != "dismiss":
            latest[line["buyer"]] = line
    return {
        b: line for b, line in latest.items() if line["action"] in ("flag", "promote", "demote")
    }


def dismissed(target: Target) -> set[tuple[str, str]]:
    """Return the (buyer, "action:CODE") suggestions the deal team rejected for this target."""
    return {
        (line["buyer"], line["reason"]) for line in _lines(target) if line["action"] == "dismiss"
    }


def flag_note(line: dict) -> str:
    """Return the review-log note for a deal-team action, e.g. "Flagged by the deal team:
    off-mandate (note)"."""
    accepted = "by the deal team (accepted reviewer suggestion)"
    verb = {
        "flag": "Flagged by the deal team",
        "promote": f"{ACTION_WORDS['promote'].capitalize()} {accepted}",
        "demote": f"{ACTION_WORDS['demote'].capitalize()} {accepted}",
    }[line["action"]]
    return f"{verb}: {line['reason']}" + (f" ({line['note']})" if line["note"] else "")


# ---------------------------------------------------------------------------
# Learned rules
# ---------------------------------------------------------------------------


def learned_rules(target_sector: str) -> list[dict]:
    """Return the accepted rules that apply to targets in this sub-sector."""
    path = learned_path()
    rules = (yaml.safe_load(path.read_text()) or {}).get("rules", []) if path.exists() else []
    return [r for r in rules if r["sector"] == target_sector]


def accept(suggestion: dict) -> dict:
    """Write a suggested rule to the learned-rules file (the method page's accept button) and
    return it."""
    path = learned_path()
    data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    rules = data.setdefault("rules", [])
    entry = {k: suggestion[k] for k in ("kind", "value", "sector", "rule", "from_flags", "history")}
    entry["accepted"] = datetime.now().isoformat(timespec="seconds")
    if not any(
        (r["kind"], r["value"], r["sector"]) == (entry["kind"], entry["value"], entry["sector"])
        for r in rules
    ):
        rules.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Rules accepted from deal-team flags (feedback.suggestions). Each applies to its "
        "sub-sector only.\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    )
    return entry


def mandate_focus(facts: list[Fact]) -> str | None:
    """Return the focus of the buyer's stated mandate (e.g. "technology"), if a profile gives
    one."""
    return next(
        (f.extra.get("focus") for f in facts if f.kind == "mandate" and f.extra.get("focus")), None
    )


def regional_deals(own: pd.DataFrame, region: str) -> int:
    """Return how many of a buyer's deals were in the region or a neighbouring one."""
    return int(sum(region_similarity(g, region) > 0 for g in own.geography))


def rule_applies(
    rule: dict,
    buyer: str,
    profiles: dict[str, list[Fact]],
    review_flags: list[tuple[str, str]],
    own: pd.DataFrame,
    target: Target,
) -> bool:
    """Return True if an accepted rule moves this buyer to Tier 2 for this target."""
    if rule["kind"] == "mandate_focus":
        return mandate_focus(profiles.get(buyer, [])) == rule["value"]
    if rule["kind"] == "capacity_flag":
        return any(code == rule["value"] for code, _ in review_flags)
    if rule["kind"] == "needs_regional_deal":
        return regional_deals(own, target.region) == 0
    raise ValueError(f"Unknown learned rule kind {rule['kind']!r}")


# ---------------------------------------------------------------------------
# Suggestions: two or more flags with the same reason and a shared, rule-shaped property
# ---------------------------------------------------------------------------


def suggestions(
    target: Target,
    flags: dict[str, dict],
    deals: pd.DataFrame,
    profiles: dict[str, list[Fact]],
    review_flags: dict[str, list[tuple[str, str]]],
) -> list[dict]:
    """Return rules the flags point to, each with the flags behind it and a check against history.

    off-mandate: the flagged buyers share a stated mandate focus -> treat every buyer with that
    focus
    as off-mandate. can't pay: they share a capacity flag -> move every buyer with it to Tier 2.
    wrong geography: none has a deal in the target's region or a neighbour -> require one.
    Rules already accepted for this sub-sector are not suggested again.
    """
    accepted = {(r["kind"], r["value"]) for r in learned_rules(target.sector)}
    by_reason: dict[str, list[str]] = {}
    for buyer, line in flags.items():
        if line["action"] == "flag":  # accepted reviewer moves are not "irrelevant" flags
            by_reason.setdefault(line["reason"], []).append(buyer)
    out = []
    for reason, buyers in by_reason.items():
        if len(buyers) < MIN_SHARED_FLAGS:
            continue
        if reason == "off-mandate":
            focus = {mandate_focus(profiles.get(b, [])) for b in buyers}
            if len(focus) == 1 and None not in focus:
                value = focus.pop()
                out.append(
                    _suggestion(
                        "mandate_focus",
                        value,
                        target,
                        buyers,
                        f"{len(buyers)} {value} investors flagged off-mandate: apply the mandate "
                        "rule to all "
                        f"{value} investors for {target.sector} targets?",
                        deals,
                        profiles,
                    )
                )
        elif reason == "can't pay":
            shared = set(CAPACITY_CODES)
            for b in buyers:
                shared &= {code for code, _ in review_flags.get(b, [])}
            if shared:
                code = sorted(shared)[0]
                out.append(
                    _suggestion(
                        "capacity_flag",
                        code,
                        target,
                        buyers,
                        f"{len(buyers)} buyers {CAPACITY_WORDS[code]} flagged as unable to pay: "
                        "move every "
                        f"such buyer to Tier 2 for {target.sector} targets?",
                        deals,
                        profiles,
                    )
                )
        elif reason == "wrong geography":
            if all(regional_deals(deals[deals.acquirer == b], target.region) == 0 for b in buyers):
                out.append(
                    _suggestion(
                        "needs_regional_deal",
                        target.region,
                        target,
                        buyers,
                        f"{len(buyers)} buyers with no deal in or next to the {target.region} "
                        "flagged for geography: "
                        f"require one for {target.sector} targets in the {target.region}?",
                        deals,
                        profiles,
                    )
                )
    return [s for s in out if (s["kind"], s["value"]) not in accepted]


def rule_text(kind: str, value: str, target: Target) -> str:
    """Return the accepted rule as a plain statement, e.g. "technology investors are off-mandate
    for Health IT targets"."""
    return {
        "mandate_focus": f"{value} investors are off-mandate for {target.sector} targets",
        "capacity_flag": (
            f"buyers {CAPACITY_WORDS.get(value, '')} go to Tier 2 for {target.sector} targets"
        ),
        "needs_regional_deal": (
            f"{target.sector} targets in the {value} need a buyer with a deal in or next to "
            "the region"
        ),
    }[kind]


def _suggestion(
    kind: str,
    value: str,
    target: Target,
    buyers: list[str],
    label: str,
    deals: pd.DataFrame,
    profiles: dict[str, list[Fact]],
) -> dict:
    """Return a rule suggestion: the rule, the flags behind it, and its check against history."""
    check = history_check(kind, value, target, deals, profiles)
    return dict(
        kind=kind,
        value=value,
        sector=target.sector,
        label=label,
        rule=rule_text(kind, value, target),
        from_flags=sorted(buyers),
        history=(
            f"in the file, the rule would have caught the real buyer in {check[0]} of {check[1]} "
            f"{target.sector} deals"
            if check
            else "it can't be checked against the file (it depends on fund sizes the file doesn't "
            "hold)"
        ),
    )


def history_check(
    kind: str, value: str, target: Target, deals: pd.DataFrame, profiles: dict[str, list[Fact]]
) -> tuple[int, int] | None:
    """Return (deals whose real buyer the rule would have caught, deals checked) in the target's
    sub-sector, or None when the rule can't be replayed on the file.

    The banker-rule test from the experiment log: before a rule becomes a gate, count the real
    buyers it would have rejected. Each deal's buyer is judged on its earlier deals only; mandate
    focus comes from today's profiles (stated mandates carry the date they were read). Of the
    capacity rules only the in-file stretch test can be replayed; fund-size rules depend on
    profile facts the file doesn't hold, so they return None rather than a reassuring zero.
    """
    if kind == "capacity_flag" and value != "CAPACITY_STRETCH":
        return None
    sector = deals[deals.sector == target.sector].sort_values("t")
    caught = 0
    for d in sector.itertuples():
        earlier = deals[(deals.acquirer == d.acquirer) & (deals.t < d.t)]
        if kind == "mandate_focus":
            caught += mandate_focus(profiles.get(d.acquirer, [])) == value
        elif (
            kind == "capacity_flag"
        ):  # a strategic whose biggest earlier deal is under half this one
            caught += (
                d.acquirer_type != "Financial Sponsor"
                and len(earlier) > 0
                and earlier.deal_size_mm.max() < STRETCH_SHARE * d.deal_size_mm
            )
        elif kind == "needs_regional_deal":
            caught += len(earlier) > 0 and regional_deals(earlier, d.geography) == 0
    return int(caught), len(sector)
