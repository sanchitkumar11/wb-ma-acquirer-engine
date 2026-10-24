"""Sub-sector groups, adjacency rings and region closeness.

Kept in one place because three stages use them: the buyer signals, the comparable-deals pool and
the banker review. Rings shape the rationale and the evidence a buyer needs; they never exclude one.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sub-sector groups (business models)
# ---------------------------------------------------------------------------

# prov = provider services, rcm = revenue cycle, payer = health insurance,
# tech = Health IT, prod = products (devices, pharma)
GROUP = {
    "Healthcare Services": "prov",
    "Physician Groups": "prov",
    "Behavioral Health": "prov",
    "Home Health/Hospice": "prov",
    "Dental": "prov",
    "Revenue Cycle": "rcm",
    "Health Insurance": "payer",
    "Health IT": "tech",
    "Medical Devices": "prod",
    "Pharma/Biotech": "prod",
}
# Pairs of groups one step apart (e.g. a payer moving into care delivery)
ADJACENT_GROUPS = {
    frozenset(p) for p in [("prov", "rcm"), ("prov", "payer"), ("rcm", "tech"), ("prod", "tech")]
}
PROVIDER_SECTORS = {s for s, g in GROUP.items() if g == "prov"}
RING_NAME = {0: "Core", 1: "Ring 1", 2: "Ring 2", 3: "Ring 3"}

# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

ADJACENT_REGIONS = {
    frozenset(p)
    for p in [
        ("Northeast", "Mid-Atlantic"),
        ("Mid-Atlantic", "Southeast"),
        ("Southeast", "Southwest"),
        ("Midwest", "Great Plains"),
        ("Mountain West", "West Coast"),
        ("Mountain West", "Southwest"),
        ("Mountain West", "Great Plains"),
        ("Midwest", "Northeast"),
    ]
}
WIDE_FOOTPRINTS = {"National", "Multi-Regional"}

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def ring(sector: str, target_sector: str) -> int:
    """Return the ring of `sector` relative to the target: 0 core, 1 same model, 2 adjacent, 3
    distant."""
    if sector == target_sector:
        return 0
    if GROUP[sector] == GROUP[target_sector]:
        return 1
    return 2 if frozenset((GROUP[sector], GROUP[target_sector])) in ADJACENT_GROUPS else 3


def sector_similarity(sector: str, target_sector: str) -> float:
    """Return how close two sub-sectors are, from 1.0 (same) down to 0.1 (distant), by ring."""
    return {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.1}[ring(sector, target_sector)]


# ---------------------------------------------------------------------------
# Plain words for the target's business model, so every label re-centres on the target
# ---------------------------------------------------------------------------

# group -> (adjective, name); for a provider target these give the original wording exactly
GROUP_WORDS = {
    "prov": ("provider", "provider services"),
    "rcm": ("revenue cycle", "revenue cycle"),
    "payer": ("payer", "health insurance"),
    "tech": ("Health IT", "Health IT"),
    "prod": ("product", "devices and pharma"),
}
# short names when a group is listed as a neighbour or a distant sector, in a fixed order
LIST_ORDER = [
    ("payer", ["payer"]),
    ("rcm", ["revenue cycle"]),
    ("tech", ["Health IT"]),
    ("prod", ["devices", "pharma"]),
    ("prov", ["provider services"]),
]
SHORT_SECTOR = {"Home Health/Hospice": "home health"}


def model_words(target_sector: str) -> tuple[str, str]:
    """Return (adjective, name) of the target's business model, e.g. ("provider", "provider
    services")."""
    return GROUP_WORDS[GROUP[target_sector]]


def model_key(target_sector: str) -> str:
    """Return the adjective as a fact-pack key fragment, e.g. "provider" or "health_it"."""
    return model_words(target_sector)[0].lower().replace(" ", "_")


def model_sectors(target_sector: str) -> set[str]:
    """Return the sub-sectors that share the target's business model (the target's own and Ring
    1)."""
    return {s for s, g in GROUP.items() if g == GROUP[target_sector]}


def ring_names(target_sector: str, k: int) -> list[str]:
    """Return short names of the sectors in ring k (2 or 3) around the target, e.g. ["payer",
    "revenue cycle"]."""
    groups = {GROUP[s] for s in GROUP if ring(s, target_sector) == k}
    return [name for g, names in LIST_ORDER if g in groups for name in names]


def adjacent_sub_sectors(target_sector: str) -> list[str]:
    """Return the Ring 1 sub-sectors in plain lower case, e.g. ["physician groups", ...,
    "dental"]."""
    return [SHORT_SECTOR.get(s, s.lower()) for s in GROUP if ring(s, target_sector) == 1]


def _or(names: list[str]) -> str:
    """Return "a, b or c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


def fit_labels(target_sector: str) -> dict[int, str]:
    """Return plain labels for where a deal or buyer sits relative to the target, by ring.

    For a provider-services target: "same sub-sector as the target", "adjacent provider sub-sector",
    "neighbouring sector (payer or revenue cycle)", "outside provider services".
    """
    adj, name = model_words(target_sector)
    return {
        0: "same sub-sector as the target",
        1: f"adjacent {adj} sub-sector",
        2: f"neighbouring sector ({_or(ring_names(target_sector, 2))})",
        3: f"outside {name}",
    }


def ring_titles(target_sector: str) -> dict[int, str]:
    """Return chart titles for the rings, e.g. "Neighbouring sectors (payer, revenue cycle)"."""
    adj, _ = model_words(target_sector)
    return {
        0: "Target's sub-sector",
        1: f"Adjacent {adj} sub-sectors",
        2: f"Neighbouring sectors ({', '.join(ring_names(target_sector, 2))})",
        3: f"Distant sectors ({', '.join(ring_names(target_sector, 3))})",
    }


def region_similarity(region: str, target_region: str) -> float:
    """Return 1.0 for the same region, 0.5 for a neighbour or a national footprint, else 0.0."""
    if region == target_region:
        return 1.0
    if region in WIDE_FOOTPRINTS or target_region in WIDE_FOOTPRINTS:
        return 0.5
    return 0.5 if frozenset((region, target_region)) in ADJACENT_REGIONS else 0.0
