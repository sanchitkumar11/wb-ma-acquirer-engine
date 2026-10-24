"""Target profile: the banker's words turned into numbers, read from a YAML file."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from acquirer_engine.paths import ROOT
from acquirer_engine.rings import ADJACENT_REGIONS, GROUP, WIDE_FOOTPRINTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TARGET_PATH = ROOT / "configs/target_default.yaml"
OWNERSHIP_TYPES = {"Private", "PE-Backed", "Public", "Non-Profit"}
REGIONS = {r for pair in ADJACENT_REGIONS for r in pair} | WIDE_FOOTPRINTS

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """The company being sold, as the engine sees it: the banker's words turned into numbers.

    Validated on construction (not only when read from YAML), so the app's sidebar form and the
    back-tests, which build targets directly, get the same checks.
    """

    sector: str
    """Sub-sector, one of rings.GROUP (e.g. "Healthcare Services")."""
    enterprise_value_mm: float
    """The ask: enterprise value in $M."""
    region: str
    """Home region, or "National" / "Multi-Regional" for a wide footprint."""
    ownership: str
    """Seller type before the deal: Private, PE-Backed, Public or Non-Profit."""
    ebitda_margin_pct: float
    """EBITDA margin in percent (16.8 means 16.8%)."""
    revenue_growth_pct: float
    """Annual revenue growth in percent."""
    as_of: str
    """Last quarter of deal data, e.g. "2024Q4"; the target is scored the quarter after."""

    def __post_init__(self) -> None:
        """Raise ValueError naming the first field that is out of range.

        A zero margin would divide by zero in the quick LBO; a margin of 100% or more, or growth
        at or below -100%, is not a business.
        """
        if self.sector not in GROUP:
            raise ValueError(f"Unknown sector {self.sector!r}; choose from {sorted(GROUP)}")
        if self.region not in REGIONS:
            raise ValueError(f"Unknown region {self.region!r}; choose from {sorted(REGIONS)}")
        if self.ownership not in OWNERSHIP_TYPES:
            raise ValueError(
                f"Unknown ownership {self.ownership!r}; choose from {sorted(OWNERSHIP_TYPES)}"
            )
        if self.enterprise_value_mm <= 0:
            raise ValueError("enterprise_value_mm must be positive")
        if not 0 < self.ebitda_margin_pct < 100:
            raise ValueError("ebitda_margin_pct must be above 0 and below 100")
        if self.revenue_growth_pct <= -100:
            raise ValueError("revenue_growth_pct must be above -100")
        if not re.fullmatch(r"\d{4}Q[1-4]", self.as_of):
            raise ValueError("as_of must look like 2024Q4")

    def scoring_quarter(self) -> tuple[int, int]:
        """Return (year, quarter index) of the quarter after `as_of`, when the target is scored.

        The quarter index is year * 4 + quarter - 1, as in data.load_deals. Scoring one quarter
        after the data end means every deal in the file counts as history.
        """
        year, quarter = int(self.as_of[:4]), int(self.as_of[5])
        q = year * 4 + quarter  # next quarter
        return q // 4, q

    def signal_inputs(self) -> dict:
        """Return the target in the shape signals.buyer_signals expects (same keys as a deal
        row)."""
        year, q = self.scoring_quarter()
        return dict(
            sector=self.sector,
            lsize=np.log(self.enterprise_value_mm),
            geography=self.region,
            ebitda_margin_pct=self.ebitda_margin_pct,
            revenue_growth_pct=self.revenue_growth_pct,
            target_ownership_pre=self.ownership,
            deal_year=year,
            q=q,
        )

    def data_note(self) -> str:
        """Return the as-of line every page carries, e.g. "Deal data through 2024 Q4"."""
        return f"Deal data through {self.as_of[:4]} {self.as_of[4:]}"

    def label(self) -> str:
        """Return a one-line description for report headers."""
        return (
            f"{self.sector}, ${self.enterprise_value_mm:.0f}M, {self.region}, "
            f"{self.ebitda_margin_pct}% margin, {self.revenue_growth_pct}% growth, "
            f"{self.ownership.lower()}; deal data through {self.as_of[:4]} {self.as_of[4:]}"
        )


def load_target(path: Path = DEFAULT_TARGET_PATH) -> Target:
    """Return the Target in the YAML file.

    Raises ValueError for any field Target rejects (see Target.__post_init__), or KeyError for a
    missing field.
    """
    raw = yaml.safe_load(Path(path).read_text())
    return Target(
        sector=raw["sector"],
        enterprise_value_mm=float(raw["enterprise_value_mm"]),
        region=raw["region"],
        ownership=raw["ownership"],
        ebitda_margin_pct=float(raw["ebitda_margin_pct"]),
        revenue_growth_pct=float(raw["revenue_growth_pct"]),
        as_of=str(raw["as_of"]),
    )
