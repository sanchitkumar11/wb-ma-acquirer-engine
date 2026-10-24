"""Load and validate the deal CSV.

Validation is strict on purpose: a bad row would quietly bend every buyer's signals, so the engine
refuses to run rather than produce a list nobody can trust.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from acquirer_engine.paths import ROOT
from acquirer_engine.rings import GROUP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH = ROOT / "data/ma_transactions_500.csv"
REQUIRED_COLUMNS = [
    "transaction_id",
    "target_company",
    "acquirer",
    "acquirer_type",
    "sector",
    "deal_year",
    "deal_quarter",
    "deal_type",
    "geography",
    "deal_size_mm",
    "ebitda_margin_pct",
    "revenue_growth_pct",
    "ev_ebitda_multiple",
    "ev_revenue_multiple",
    "outcome",
    "num_bidders",
    "target_ownership_pre",
]
BUYER_TYPES = {"Financial Sponsor", "Strategic"}

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def load_deals(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Return the deals sorted by date, with helper columns `t`, `q` and `lsize` added.

    `t` (year * 10 + quarter) orders deals; `q` (year * 4 + quarter - 1) counts quarters, used for
    look-back windows; `lsize` is log deal size, because a $50M gap matters more at $100M than at
    $2B. Sorting by date then transaction ID fixes the row order, which the as-of signals rely on.

    Raises ValueError naming the first problem found (missing column, blank value, unknown
    sub-sector or buyer type, bad quarter, non-positive deal size, buyer with two types).
    """
    df = pd.read_csv(csv_path)
    _validate(df)
    df["t"] = df.deal_year * 10 + df.deal_quarter.str[1].astype(int)
    df = df.sort_values(["t", "transaction_id"]).reset_index(drop=True)
    df["q"] = df.deal_year * 4 + df.deal_quarter.str[1].astype(int) - 1
    df["lsize"] = np.log(df.deal_size_mm)
    return df


def file_hash(csv_path: Path) -> str:
    """Return the SHA-256 of the file, used to tell whether a saved model matches the data."""
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()


def _validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the CSV can't be trusted; return None if it passes."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    blank = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if blank:
        raise ValueError(f"CSV has blank values in: {blank}")
    unknown = sorted(set(df.sector) - set(GROUP))
    if unknown:
        raise ValueError(f"Unknown sub-sectors (add them to rings.GROUP): {unknown}")
    bad_type = sorted(set(df.acquirer_type) - BUYER_TYPES)
    if bad_type:
        raise ValueError(f"Unknown acquirer types: {bad_type}")
    if not df.deal_quarter.str.fullmatch(r"Q[1-4]").all():
        raise ValueError("deal_quarter must be Q1 to Q4")
    if (df.deal_size_mm <= 0).any():
        raise ValueError("deal_size_mm must be positive")
    two_types = df.groupby("acquirer").acquirer_type.nunique()
    if (two_types > 1).any():
        raise ValueError(f"Buyers listed as both types: {list(two_types[two_types > 1].index)}")


def data_end(deals: pd.DataFrame) -> str:
    """Return the last quarter with a deal in the file, e.g. "2024Q4" (the live target's as_of)."""
    q = int(deals.q.max())
    return f"{q // 4}Q{q % 4 + 1}"
