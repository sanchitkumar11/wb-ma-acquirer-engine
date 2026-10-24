"""The engine's exceptions, in one hierarchy, so a caller can catch exactly what it means to.

    EngineError
    +-- LLMError                  an LLM call was not made, on purpose
        +-- NotCachedError        replay mode, and the output is not in the cache
        +-- PaidCallNotAllowedError  a cache miss in a run started without allow_paid
        +-- BudgetExceededError   the call could take total spend past the budget

The three LLM errors are guards, not failures: each stops before any money is spent, and the CLI
reports them as "Stopped before any spend". Bad input (an unknown sub-sector, a malformed profile)
raises ValueError with the problem named, as the standard library does.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EngineError(Exception):
    """Base class for every exception the engine raises on purpose."""


class LLMError(EngineError):
    """Base class for the guards that stop an LLM call before it is made."""


class NotCachedError(LLMError):
    """Raised in replay mode when the output is not in the cache."""


class PaidCallNotAllowedError(LLMError):
    """Raised on a cache miss when the run was not started with allow_paid."""


class BudgetExceededError(LLMError):
    """Raised before a call that could take total spend past the budget."""
