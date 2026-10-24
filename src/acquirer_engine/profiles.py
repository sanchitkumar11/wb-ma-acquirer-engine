"""Buyer profiles: dated, sourced real-world facts that the CSV can't contain.

The model only knows the deal file. A banker also knows that Steward is bankrupt, that Atrium is
now part of Advocate Health, and that Vista invests in software. Those facts live in one YAML per
buyer in data/profiles/, and every fact carries the value, the source that was opened, and the
date it became true or public. The review rules act on these facts; nothing here ranks buyers.

Back-tests must only see facts dated before the deal being scored: use facts_as_of().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from acquirer_engine.paths import ROOT
from acquirer_engine.rings import GROUP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFILES_DIR = ROOT / "data/profiles"
FACT_KINDS = {
    "status",
    "merger",
    "parent",
    "divestiture",
    "mandate",
    "fund_size",
    "aum",
    "capital_raised",
    "revenue",
    "size",
    "footprint",
    "acquisition_8k",
}
AMOUNT_KINDS = {
    "fund_size",
    "aum",
    "revenue",
}  # capacity facts: the review compares their amount to the EV
ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
# A mandate whose focus is technology fits targets in these groups only
TECH_GROUPS = {"tech", "rcm"}

# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """One dated, sourced fact about a buyer (status, merger, mandate, size, footprint)."""

    buyer: str
    kind: str
    value: str
    as_of: date
    source: str
    extra: (
        dict  # kind-specific fields: status, merged_into, focus, amount_usd_mm, states, accession
    )

    def cite(self) -> str:
        """Return a short citation for logs and fact packs: source (and accession), dated."""
        accession = f", accession {self.extra['accession']}" if "accession" in self.extra else ""
        return f"{self.source}{accession}, as of {self.as_of.isoformat()}"


def load_profiles(profiles_dir: Path = PROFILES_DIR) -> dict[str, list[Fact]]:
    """Return buyer name (as written in the CSV) -> its facts.

    Raises ValueError on the first fact without a value, a source URL or a valid as-of date, on
    an unknown kind, on an EDGAR fact without an accession number, or on a duplicate buyer.
    """
    profiles: dict[str, list[Fact]] = {}
    for path in sorted(Path(profiles_dir).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        name = raw.get("name")
        if not name:
            raise ValueError(f"{path.name}: missing name")
        if name in profiles:
            raise ValueError(f"{path.name}: duplicate profile for {name}")
        profiles[name] = [
            _parse_fact(name, path.name, i, f) for i, f in enumerate(raw.get("facts") or [])
        ]
    return profiles


def _parse_fact(buyer: str, file_name: str, i: int, raw: dict) -> Fact:
    """Return one validated Fact, or raise ValueError naming the file and fact number."""
    where = f"{file_name} fact {i + 1}"
    if raw.get("kind") not in FACT_KINDS:
        raise ValueError(f"{where}: unknown kind {raw.get('kind')!r}")
    if not str(raw.get("value") or "").strip():
        raise ValueError(f"{where}: missing value")
    source = str(raw.get("source") or "")
    if not source.startswith("https://"):
        raise ValueError(f"{where}: source must be an https URL that was opened")
    try:
        as_of = date.fromisoformat(str(raw.get("as_of")))
    except ValueError as exc:
        raise ValueError(f"{where}: as_of must be a date like 2024-05-06") from exc
    if "sec.gov" in source and not ACCESSION.fullmatch(str(raw.get("accession", ""))):
        raise ValueError(f"{where}: EDGAR facts need an accession number like 0000049071-23-000011")
    if raw["kind"] in AMOUNT_KINDS and not isinstance(raw.get("amount_usd_mm"), (int, float)):
        raise ValueError(f"{where}: a {raw['kind']} fact needs amount_usd_mm (a number, in $M)")
    core = {"kind", "value", "as_of", "source", "accessed"}
    return Fact(
        buyer,
        raw["kind"],
        str(raw["value"]),
        as_of,
        source,
        {k: v for k, v in raw.items() if k not in core},
    )


def facts_as_of(profiles: dict[str, list[Fact]], cutoff: date) -> dict[str, list[Fact]]:
    """Return only facts dated strictly before `cutoff`: what a banker could have known then."""
    return {buyer: [f for f in facts if f.as_of < cutoff] for buyer, facts in profiles.items()}


# ---------------------------------------------------------------------------
# Notes for the review: what each fact means for this buyer and this target
# ---------------------------------------------------------------------------


def profile_notes(
    buyer: str, profiles: dict[str, list[Fact]], target_sector: str, buyers_in_file: set[str]
) -> list[tuple[str, str, Fact]]:
    """Return (reason code, note, fact) for each profile fact that should move this buyer.

    Reason codes:
    - STATUS_DEFUNCT: the buyer can no longer transact (e.g. Chapter 11 and liquidation).
    - ENTITY_DUPLICATE: the buyer is now part of another buyer in the file (merge them).
    - STATUS_ACQUIRED: the buyer is now part of a company outside the file; its parent decides.
    - MANDATE_MISMATCH: a technology investor, and the target is not a technology business.
    """
    notes = []
    for fact in profiles.get(buyer, []):
        if fact.kind == "status" and fact.extra.get("status") == "defunct":
            notes.append(("STATUS_DEFUNCT", f"Status: {fact.value}", fact))
        elif fact.kind == "merger" and fact.extra.get("merged_into"):
            parent = fact.extra["merged_into"]
            code = "ENTITY_DUPLICATE" if parent in buyers_in_file else "STATUS_ACQUIRED"
            notes.append((code, f"Now part of {parent}: {fact.value}", fact))
        elif (
            fact.kind == "mandate"
            and fact.extra.get("focus") == "technology"
            and GROUP[target_sector] not in TECH_GROUPS
        ):
            notes.append(("MANDATE_MISMATCH", f"Mandate: {fact.value}", fact))
    return notes


# ---------------------------------------------------------------------------
# Lookups: one place, so every stage reads capacity facts the same way
# ---------------------------------------------------------------------------


def capital_fact(facts: list[Fact]) -> Fact | None:
    """Return the buyer's fund size or AUM fact (the first one in its profile), or None."""
    return next((f for f in facts if f.kind in ("fund_size", "aum")), None)


def revenue_fact(facts: list[Fact]) -> Fact | None:
    """Return the buyer's annual revenue fact, or None."""
    return next((f for f in facts if f.kind == "revenue"), None)
