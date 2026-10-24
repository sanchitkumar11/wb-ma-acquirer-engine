"""LLM layer: one place for provider choice, the output cache and the cost ledger.

- Provider is chosen by LLM_PROVIDER: databricks (default, Databricks Model Serving through
  databricks-langchain), anthropic, or replay (cached outputs only: no key, no cost).
- Non-determinism is handled by caching: an output is keyed on the prompt version, the model, the
  writing approach and the fact-pack hash, and reused until one of those changes. The cache is
  committed, so a fresh clone replays the sample run with no key.
- Every paid call is written to the cost ledger. A call is refused if it could take spending past
  the budget, and no paid call happens unless the run was started with allow_paid (the CLI's
  --allow-paid), so nothing is spent without the user asking for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from acquirer_engine.env import load_env
from acquirer_engine.errors import BudgetExceededError, NotCachedError, PaidCallNotAllowedError
from acquirer_engine.paths import ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRICES_PATH = ROOT / "configs/llm_prices.yaml"
CACHE_DIR = ROOT / "cache/llm"  # LLM_CACHE_DIR overrides (e.g. to time a cold run)


def cache_dir() -> Path:
    """Return the output cache directory (LLM_CACHE_DIR, else cache/llm)."""
    load_env()
    return Path(os.environ.get("LLM_CACHE_DIR", str(CACHE_DIR)))


LEDGER_PATH = ROOT / "ledger/cost_ledger.jsonl"
MAX_OUTPUT_TOKENS = (
    12000  # Sonnet 5.5's hidden reasoning counts against this; 4,000 left UPMC with no answer text
)
ANTHROPIC_MODELS = {"writer": "claude-sonnet-5-5", "writer_dev": "claude-haiku-5-5"}


@dataclass
class Usage:
    """Tokens, cost and time of one LLM call (all zero and cached=True when replayed)."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    cached: bool = True
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# Prices and ledger
# ---------------------------------------------------------------------------


def prices() -> dict:
    """Return the price table (configs/llm_prices.yaml)."""
    return yaml.safe_load(PRICES_PATH.read_text())


def cost_usd(alias: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of one call: tokens x DBU per million x $/DBU."""
    table = prices()
    rate = table["models"][alias]["dbu_per_m"]
    dbus = (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1e6
    return dbus * table["usd_per_dbu"]


_BUDGET_LOCK = threading.Lock()
_reserved_usd = 0.0
"""Worst-case cost of calls in flight and not yet in the ledger (parallel page writers share it)."""


@contextmanager
def _budget_reservation(alias: str, prompt_chars: int, attempts: int) -> Iterator[None]:
    """Reserve a call's worst-case cost against the budget for as long as the call runs.

    Check-then-record alone let ten parallel page writers all pass the check before any of them
    reached the ledger; reserving under a lock closes that gap. Worst case: every attempt uses
    the full output cap. Raises BudgetExceededError before the call if spend so far, plus calls
    in flight, plus this one could pass LLM_BUDGET_USD.
    """
    global _reserved_usd
    worst = attempts * cost_usd(alias, prompt_chars // 3, MAX_OUTPUT_TOKENS)
    budget = float(os.environ.get("LLM_BUDGET_USD", prices()["budget_usd"]))
    with _BUDGET_LOCK:
        spent = spent_usd()
        if spent + _reserved_usd + worst > budget:
            raise BudgetExceededError(
                f"spent ${spent:.2f}, ${_reserved_usd:.2f} in flight, and this call up to "
                f"${worst:.2f} "
                f"would pass the ${budget:.0f} budget"
            )
        _reserved_usd += worst
    try:
        yield
    finally:
        with _BUDGET_LOCK:
            _reserved_usd -= worst


def spent_usd() -> float:
    """Return the total spend recorded in the ledger."""
    if not LEDGER_PATH.exists():
        return 0.0
    return sum(
        json.loads(line)["usd"] for line in LEDGER_PATH.read_text().splitlines() if line.strip()
    )


def _record(alias: str, endpoint: str, usage: Usage, purpose: str) -> None:
    """Append one paid call to the ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a") as ledger:
        ledger.write(
            json.dumps(
                dict(
                    time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    alias=alias,
                    endpoint=endpoint,
                    purpose=purpose,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    usd=round(usage.usd, 5),
                    seconds=round(usage.seconds, 1),
                )
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


_replay_depth = 0
"""How many replay_only() blocks are open; above zero, provider() is "replay" whatever .env says."""


@contextmanager
def replay_only() -> Iterator[None]:
    """Make every LLM call in the block replay from the cache, then restore the provider.

    Scoped instead of setting LLM_PROVIDER in os.environ, which leaked to the rest of the
    process (a later paid call in the same run was silently refused). A module counter, not a
    context variable, so the threads the page writers run in see it too.
    """
    global _replay_depth
    _replay_depth += 1
    try:
        yield
    finally:
        _replay_depth -= 1


def provider() -> str:
    """Return the active provider: databricks, anthropic or replay (inside replay_only():
    replay)."""
    if _replay_depth:
        return "replay"
    load_env()
    return os.environ.get("LLM_PROVIDER", "databricks").strip().lower()


_MODELS: dict[str, tuple] = {}
_MODELS_LOCK = threading.Lock()


def _chat_model(alias: str) -> tuple[BaseChatModel, str]:
    """Return (shared chat model, endpoint or model name) for the role alias, creating it once.

    Created under a lock: ten parallel threads each building a client would race the workspace
    login refresh (seen Oct 23: "cache update: exit status 45").
    """
    with _MODELS_LOCK:
        if alias not in _MODELS:
            _MODELS[alias] = _new_chat_model(alias)
        return _MODELS[alias]


def _new_chat_model(alias: str) -> tuple[BaseChatModel, str]:
    """Return a new LangChain chat model for the role alias (writer, writer_dev, judge,
    open_weight)."""
    name = provider()
    if name == "databricks":
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import ChatDatabricks

        endpoint = (
            os.environ.get(f"LLM_{alias.upper()}_ENDPOINT") or prices()["models"][alias]["endpoint"]
        )
        model = ChatDatabricks(
            endpoint=endpoint, max_tokens=MAX_OUTPUT_TOKENS, workspace_client=WorkspaceClient()
        )
        _ = (
            model.client
        )  # build the HTTP client now, inside the lock: its lazy creation isn't thread-safe
        return model, endpoint
    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if alias not in ANTHROPIC_MODELS:
            raise ValueError(
                f"No Anthropic model for role {alias!r}; roles available: "
                f"{sorted(ANTHROPIC_MODELS)}"
            )
        model = ANTHROPIC_MODELS[alias]
        return ChatAnthropic(model=model, max_tokens=MAX_OUTPUT_TOKENS), model
    raise NotCachedError(f"LLM_PROVIDER={name}: no live model available")


def cache_key(prompt_version: str, alias: str, approach: str, pack_hash: str, step: str) -> str:
    """Return the cache key: prompt version + model role + writing approach + fact-pack hash + step,
    plus a provider tag for any provider other than the default.

    The committed cache was written by the Databricks endpoint, so its keys carry no tag and
    replay unchanged. A live Anthropic run tags its keys "anthropic" (it must not replay, or
    overwrite, the Databricks outputs); set LLM_CACHE_TAG=anthropic to replay that run later.
    """
    tag = os.environ.get("LLM_CACHE_TAG") or ("anthropic" if provider() == "anthropic" else "")
    parts = [prompt_version, alias, approach, pack_hash, step, *([tag] if tag else [])]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def _text(content: Any) -> str:
    """Return the answer text of a reply, skipping reasoning blocks.

    Content arrives as a string, a list of blocks, or (from some endpoints) a string holding a
    JSON list of blocks such as [{"type": "reasoning", ...}, {"type": "text", "text": "..."}].
    """
    if isinstance(content, str) and content.lstrip().startswith("[{"):
        try:
            content = json.loads(content)
        except ValueError:
            return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _json_object(text: str) -> dict:
    """Return the JSON object in a model reply (tolerates code fences or a sentence around it)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    obj, _ = json.JSONDecoder().raw_decode(
        text[start:]
    )  # the first complete object; ignore anything after
    return obj


def structured_call(
    schema: type[BaseModel],
    system: str,
    user: str,
    *,
    alias: str,
    key: str,
    purpose: str,
    allow_paid: bool = False,
) -> tuple[BaseModel, Usage]:
    """Return (parsed output, usage), from the cache when possible.

    Raises NotCachedError in replay mode on a miss, PaidCallNotAllowedError on a miss without
    allow_paid,
    and BudgetExceededError if the worst-case cost of the call would pass the budget.
    """
    path = cache_dir() / f"{key}.json"
    if path.exists():
        return schema.model_validate(json.loads(path.read_text())["output"]), Usage()
    if provider() == "replay":
        raise NotCachedError(
            f"{purpose}: not in the cache (key {key}); run once with a provider to fill it"
        )
    if not allow_paid:
        raise PaidCallNotAllowedError(f"{purpose}: would make a paid call; rerun with --allow-paid")
    with _budget_reservation(
        alias, len(system) + len(user), attempts=2
    ):  # the parse retry is a 2nd call
        return _structured_call_live(schema, system, user, alias=alias, key=key, purpose=purpose)


def _structured_call_live(
    schema: type[BaseModel], system: str, user: str, *, alias: str, key: str, purpose: str
) -> tuple[BaseModel, Usage]:
    """Make the call for structured_call (budget already reserved), record it and cache the
    output."""
    path = cache_dir() / f"{key}.json"
    model, endpoint = _chat_model(alias)

    # Structured output is JSON in the prompt, checked with pydantic. Tried and rejected on these
    # endpoints (Oct 23): temperature (any value) and forced tool choice return 400; an unforced
    # single tool works for a flat schema but mangles nested fields (sections arrive as
    # '<parameter name="title">...' fragments), so the six-section page can't go through a tool.
    messages = [
        (
            "system",
            f"{system}\n\nReply with one JSON object only, no prose and no code fences, "
            f"matching this JSON Schema:\n{json.dumps(schema.model_json_schema())}",
        ),
        ("user", user),
    ]
    usage = Usage(cached=False)
    start = time.perf_counter()
    for attempt in range(2):
        reply = model.invoke(messages)
        meta = getattr(reply, "usage_metadata", None) or {}
        usage.input_tokens += meta.get("input_tokens", 0)
        usage.output_tokens += meta.get("output_tokens", 0)
        try:
            parsed = schema.model_validate(_json_object(_text(reply.content)))
            break
        except ValueError as exc:
            (ROOT / "out").mkdir(exist_ok=True)
            (ROOT / "out" / f"failed_{key}_{attempt}.txt").write_text(
                _text(reply.content)
            )  # kept for debugging
            if attempt == 1:
                usage.usd = cost_usd(alias, usage.input_tokens, usage.output_tokens)
                _record(
                    alias, endpoint, usage, purpose + " (failed to parse)"
                )  # failed calls still cost money
                raise ValueError(f"{purpose}: reply did not match the schema twice: {exc}") from exc
            messages += [
                ("assistant", _text(reply.content)),
                (
                    "user",
                    f"That reply did not fit the schema: {exc}. Reply with the JSON object only.",
                ),
            ]
    usage.seconds = time.perf_counter() - start
    usage.usd = cost_usd(alias, usage.input_tokens, usage.output_tokens)
    _record(alias, endpoint, usage, purpose)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(
                purpose=purpose,
                alias=alias,
                endpoint=endpoint,
                usage=usage.__dict__,
                output=parsed.model_dump(),
            ),
            indent=1,
        )
    )
    return parsed, usage


def tool_call(
    messages: list, tools: list, *, alias: str, purpose: str, allow_paid: bool
) -> tuple[AIMessage, Usage]:
    """Return the model's next message (it may request tools), recording its cost in the ledger.

    Used by the tool-using agent (A3), which caches its final answer itself. Same guards as
    structured_call: no call in replay mode or without allow_paid, and a budget stop.
    """
    if provider() == "replay":
        raise NotCachedError(f"{purpose}: replay mode makes no live calls")
    if not allow_paid:
        raise PaidCallNotAllowedError(f"{purpose}: would make a paid call; rerun with --allow-paid")
    model, endpoint = _chat_model(alias)
    start = time.perf_counter()
    prompt_chars = sum(len(str(getattr(m, "content", m))) for m in messages)
    with _budget_reservation(alias, prompt_chars, attempts=1):
        reply = model.bind_tools(tools).invoke(messages)
    meta = getattr(reply, "usage_metadata", None) or {}
    usage = Usage(
        meta.get("input_tokens", 0),
        meta.get("output_tokens", 0),
        cached=False,
        seconds=time.perf_counter() - start,
    )
    usage.usd = cost_usd(alias, usage.input_tokens, usage.output_tokens)
    _record(alias, endpoint, usage, purpose)
    return reply, usage
