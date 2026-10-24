"""Model and input checks: the sign lock holds, and bad targets are refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from acquirer_engine.config import load_target
from acquirer_engine.model import RankingModel

ROOT = Path(__file__).resolve().parents[1]


def test_signs_locked():
    model, _ = RankingModel.load(ROOT / "models/ranking_model.json")
    for weights in model.weights.values():
        assert (weights[1:] >= 0).all()  # [0] is the intercept, which is free


def test_raising_a_signal_never_lowers_a_score():
    """Monotonicity: with every weight >= 0, more of any banker signal can't hurt a buyer."""
    import pandas as pd

    model, _ = RankingModel.load(ROOT / "models/ranking_model.json")
    for sponsor in (0, 1):
        base = pd.DataFrame([dict(model.mean, sponsor=sponsor)])
        for f in model.features:
            bumped = base.copy()
            bumped[f] += model.std[f]
            assert model.score(bumped)[0] >= model.score(base)[0]


def test_unknown_region_is_refused(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text(
        (ROOT / "configs/target_default.yaml").read_text().replace("Midwest", "Atlantis")
    )
    with pytest.raises(ValueError, match="Unknown region"):
        load_target(bad)
