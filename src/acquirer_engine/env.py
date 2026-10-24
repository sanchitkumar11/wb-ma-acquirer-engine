"""Read local settings from .env: the one place keys and connection details come from.

A separate module because the LLM layer and the research scripts both need it, and the lowest
layer of the package must not import the LLM layer. `.env` is git-ignored; see `.env.example`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from acquirer_engine.paths import ROOT

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_PATH = ROOT / ".env"

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def load_env(env_path: Path = ENV_PATH) -> None:
    """Read KEY=VALUE lines from .env into os.environ, never overriding what is already set.

    Values set in the shell win, so a test or a one-off command can override .env. A trailing
    " # comment" is dropped and surrounding quotes stripped. A missing file is not an error:
    replay needs no settings at all.
    """
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.split(" #")[0].strip().strip('"'))
