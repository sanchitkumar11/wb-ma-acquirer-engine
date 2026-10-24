"""Plain-English wording applied when text is displayed (report, app, tables).

Applied at display time, not in the fact packs: changing fact-pack text would change their hashes
and invalidate the cached one-pagers. Longest phrases first, so the specific forms win.
"""

from __future__ import annotations

import re

PLATFORM_MEANING = (
    "already owns a platform company in the target's sub-sector (bought in the last five years), "
    "so this business could be bolted on"
)

# A noun swap keeps every sentence grammatical; the full meaning (PLATFORM_MEANING) is given in the
# column headers, the Tier 2 promote line and the method glossary.
_PHRASES = [
    (r"\bopen platforms\b", "existing platform companies"),
    (r"\bopen platform\b", "existing platform company"),
    (r"\bOpen platforms\b", "Existing platform companies"),
    (r"\bOpen platform\b", "Existing platform company"),
]


def plain_terms(text: str) -> str:
    """Return text with internal shorthand ("open platform") replaced by plain banker wording."""
    for pattern, replacement in _PHRASES:
        text = re.sub(pattern, replacement, text)
    return text[:1].upper() + text[1:] if text else text


CSV_ONLY_NOTE = (
    "CSV-only mode: buyer profiles are not loaded, so status, merger and mandate checks are off."
)
"""Shown wherever a list was built without the dated buyer profiles."""
