"""Ranking model: one logistic regression per buyer type, banker starting weights, signs locked.

Sponsors buy for returns (leverage, add-ons, exit); strategics buy for synergies and market
position, so each type gets its own weights. Each weight starts from a banker's judgment and the
data moves it as far as the evidence supports (a penalty of strength LAMBDA pulls it back). No
weight can go below zero: more of a good signal can never lower a buyer's score.

Fitting needs the buyer x deal pairs, which take about a minute to build, so the fitted model is
saved to MODEL_PATH with the hash of the CSV it came from and refit only when the data changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from acquirer_engine import signals as signals_module
from acquirer_engine.data import file_hash
from acquirer_engine.paths import ROOT
from acquirer_engine.signals import training_pairs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = ROOT / "models/ranking_model.json"
log = logging.getLogger(__name__)
LAMBDA = 30.0

# The 16 signals of the chosen model: the 14 original banker signals, with
#  - size fit and size-in-range replaced by the one-sided size test (not_bigger, not_smaller),
#  - close rate, deal type and ownership shrunk toward the buyer-type average (the "_s" versions),
#  - plus the sponsor open-platform signals.
FEATURES = [
    "activity",
    "recency",
    "sector_share",
    "sector_rel",
    "geo_fit",
    "own_fit_s",
    "type_fit_s",
    "margin_fit",
    "growth_fit",
    "close_rate_s",
    "bidder_exp",
    "precedent_sim",
    "not_bigger",
    "not_smaller",
    "open_platform",
    "open_platform_prov",
]
# Banker starting weights: log-odds per 1 standard deviation of each signal
STRATEGIC, SPONSOR = 0, 1
PRIOR = {
    STRATEGIC: dict(
        activity=0.3,
        recency=0.3,
        sector_share=1.0,
        sector_rel=0.5,
        geo_fit=0.5,
        own_fit_s=0.2,
        type_fit_s=0.2,
        margin_fit=0.2,
        growth_fit=0.2,
        close_rate_s=0.3,
        bidder_exp=0,
        precedent_sim=0.5,
        not_bigger=0.6,
        not_smaller=0.1,
        open_platform=0,
        open_platform_prov=0,
    ),
    SPONSOR: dict(
        activity=0.6,
        recency=0.6,
        sector_share=0.4,
        sector_rel=0.6,
        geo_fit=0.2,
        own_fit_s=0.4,
        type_fit_s=0.5,
        margin_fit=0.3,
        growth_fit=0.2,
        close_rate_s=0.3,
        bidder_exp=0.2,
        precedent_sim=0.4,
        not_bigger=0.6,
        not_smaller=0.3,
        open_platform=0.5,
        open_platform_prov=0.3,
    ),
}

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_weights(
    standardised: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    intercept0: float,
    lam: float = LAMBDA,
) -> np.ndarray:
    """Return [intercept, weight per signal] for one buyer type.

    `standardised` is one row per buyer x deal pair and one column per signal (z-scores); `y`
    is 1 for the actual buyer. Maximises the logistic likelihood minus lam/2 * (weight - banker
    weight)^2, with every signal weight bounded at >= 0 (the sign lock). The intercept is free
    and unpenalised.
    """
    design = np.column_stack([np.ones(len(standardised)), standardised])  # intercept column first
    start = np.concatenate([[intercept0], prior])
    penalty = np.r_[0.0, np.full(len(prior), lam)]

    def loss_and_gradient(weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the penalised negative log-likelihood and its gradient."""
        log_odds = design @ weights
        p = 1 / (1 + np.exp(-log_odds))
        loss = (
            -(y * log_odds - np.logaddexp(0, log_odds)).sum()
            + 0.5 * (penalty * (weights - start) ** 2).sum()
        )
        return loss, design.T @ (p - y) + penalty * (weights - start)

    bounds = [(None, None)] + [(0, None)] * len(prior)
    return minimize(
        loss_and_gradient, np.clip(start, 0, None), jac=True, method="L-BFGS-B", bounds=bounds
    ).x


@dataclass
class RankingModel:
    """One sign-locked logistic regression per buyer type, on standardised signals."""

    features: list[str]
    """Signal names, in column order."""
    mean: pd.Series
    """Per-signal mean of the training pairs, for standardising."""
    std: pd.Series
    """Per-signal standard deviation of the training pairs (0 replaced by 1)."""
    weights: dict[int, np.ndarray]
    """Buyer type (signals.SPONSOR / STRATEGIC) -> [intercept, weight per signal]."""

    @classmethod
    def fit(cls, pairs: pd.DataFrame, features: list[str] = FEATURES) -> RankingModel:
        """Return a model fitted on `pairs` (from signals.training_pairs)."""
        mean, std = pairs[features].mean(), pairs[features].std().replace(0, 1)
        weights = {}
        for buyer_type in (STRATEGIC, SPONSOR):
            rows = pairs[pairs.sponsor == buyer_type]
            base_rate = rows.label.mean()
            weights[buyer_type] = fit_weights(
                ((rows[features] - mean) / std).values,
                rows.label.values,
                np.array([PRIOR[buyer_type][f] for f in features]),
                np.log(base_rate / (1 - base_rate)),
            )
        return cls(features, mean, std, weights)

    def score(self, signals: pd.DataFrame) -> np.ndarray:
        """Return each buyer's log-odds score (higher = more likely)."""
        standardised = ((signals[self.features] - self.mean) / self.std).values
        scores = np.zeros(len(signals))
        for buyer_type, w in self.weights.items():
            rows = (signals.sponsor == buyer_type).values
            scores[rows] = w[0] + standardised[rows] @ w[1:]
        return scores

    def contributions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Return weight x standardised signal per buyer and signal: the "why" behind a score.

        A contribution of +1.2 multiplies the buyer's odds by about 3.3 (e^1.2) against a buyer
        that is average on that signal.
        """
        standardised = (signals[self.features] - self.mean) / self.std
        weights_by_row = np.vstack(
            [self.weights[t][1:] for t in signals.sponsor]
        )  # each row's type
        return pd.DataFrame(
            standardised.values * weights_by_row, columns=self.features, index=signals.index
        )

    # -- saving --------------------------------------------------------------

    def save(self, path: Path, csv_sha256: str) -> None:
        """Write the model as JSON. Python floats round-trip exactly, so scores are unchanged."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                dict(
                    csv_sha256=csv_sha256,
                    lam=LAMBDA,
                    features=self.features,
                    mean=self.mean.tolist(),
                    std=self.std.tolist(),
                    weights={str(k): v.tolist() for k, v in self.weights.items()},
                    prior={str(k): PRIOR[k] for k in PRIOR},
                    signals_sha256=file_hash(Path(signals_module.__file__)),
                ),
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: Path) -> tuple[RankingModel, dict]:
        """Return (model, settings it was fitted with: csv_sha256, lam, prior, signals_sha256)."""
        raw = json.loads(path.read_text())
        features = raw["features"]
        model = cls(
            features,
            pd.Series(raw["mean"], index=features),
            pd.Series(raw["std"], index=features),
            {int(k): np.array(v) for k, v in raw["weights"].items()},
        )
        return model, dict(
            csv_sha256=raw["csv_sha256"],
            lam=raw["lam"],
            prior=raw["prior"],
            signals_sha256=raw.get("signals_sha256", ""),  # absent in older files: forces a refit
        )


def load_or_fit(
    deals: pd.DataFrame, csv_path: Path, model_path: Path = MODEL_PATH, refit: bool = False
) -> RankingModel:
    """Return the saved model if it was fitted on this CSV and these settings, else refit and save.

    Fitting on all deals through the data end is what the live target is scored with; the
    back-tests in experiments/ fit on 2015-21 only.
    """
    sha = file_hash(csv_path)
    if model_path.exists() and not refit:
        model, settings = RankingModel.load(model_path)
        if settings == _settings(sha) and model.features == FEATURES:
            return model
    log.info("Fitting the ranking model on every deal in the file (about a minute)...")
    model = RankingModel.fit(training_pairs(deals))
    model.save(model_path, sha)
    return model


def _settings(csv_sha256: str) -> dict:
    """Return what a saved model must match to be reused: the data, the penalty, the banker
    weights, and the signal definitions (a hash of signals.py, so editing a signal refits)."""
    return dict(
        csv_sha256=csv_sha256,
        lam=LAMBDA,
        prior={str(k): PRIOR[k] for k in PRIOR},
        signals_sha256=file_hash(Path(signals_module.__file__)),
    )
