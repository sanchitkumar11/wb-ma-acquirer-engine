"""Where the project's files live: one root, so the engine runs from any working directory.

Every data, prompt, cache, ledger and output path is built from ROOT. With the editable install
(`pip install -r requirements.txt`) ROOT is the repository; set ACQUIRER_ENGINE_ROOT to point a
copy of the package at another checkout of the data. Paths a user types on the command line
(--target, --out) stay relative to where they ran the command.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(os.environ.get("ACQUIRER_ENGINE_ROOT") or Path(__file__).resolve().parents[2])
"""The repository root: src/acquirer_engine/paths.py is two levels below it."""
