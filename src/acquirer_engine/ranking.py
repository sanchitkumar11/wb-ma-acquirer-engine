"""Score every buyer in the file for one target with the saved (or freshly fitted) model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from acquirer_engine.data import DEFAULT_CSV_PATH, load_deals
from acquirer_engine.model import MODEL_PATH, load_or_fit
from acquirer_engine.signals import buyer_signals

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from acquirer_engine.config import Target


def rank_buyers(
    target: Target,
    csv_path: Path = DEFAULT_CSV_PATH,
    model_path: Path = MODEL_PATH,
    refit: bool = False,
) -> pd.DataFrame:
    """Return every buyer in the file, best first, with its signals, score and likelihood.

    Columns added to the buyer_signals columns: s (log-odds score), p (probability), share
    (likelihood, scaled to sum to 100% across all buyers) and model_rank (1 = most likely).
    """
    deals = load_deals(csv_path)
    model = load_or_fit(deals, csv_path, model_path, refit)
    ranked = buyer_signals(target.signal_inputs(), deals)
    ranked["s"] = model.score(ranked)
    ranked["p"] = 1 / (1 + np.exp(-ranked.s))
    ranked["share"] = ranked.p / ranked.p.sum()
    ranked = ranked.sort_values("s", ascending=False).reset_index(drop=True)
    ranked["model_rank"] = ranked.index + 1
    return ranked
