"""The packaged engine must reproduce experiments/review.py exactly: same buyers, same order, same
scores.

The fast test compares against a fixture written from review.py's output (see
tests/fixtures/make_review_fixture.py). The slow test reruns review.py itself and refits the
package's model from scratch, so neither the fixture nor the saved model can go stale unnoticed:
    pytest -m slow
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acquirer_engine.config import load_target
from acquirer_engine.ranking import rank_buyers

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/fixtures/review_py_live_scores.csv"
CSV_PATH = ROOT / "data/ma_transactions_500.csv"
TARGET_PATH = ROOT / "configs/target_default.yaml"


def _assert_same(ours: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Fail unless both rankings list the same buyers in the same order with the same scores."""
    assert list(ours.acquirer) == list(reference.acquirer)
    assert list(ours.sponsor) == list(reference.sponsor)
    np.testing.assert_allclose(ours.s.values, reference.s.values, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ours.share.values, reference.share.values, rtol=0, atol=1e-12)


def test_rank_matches_review_fixture():
    ours = rank_buyers(
        load_target(TARGET_PATH), CSV_PATH, model_path=ROOT / "models/ranking_model.json"
    )
    _assert_same(ours, pd.read_csv(FIXTURE_PATH))


@pytest.mark.slow
def test_refit_matches_review_script(tmp_path):
    subprocess.run(
        [sys.executable, "-I", "experiments/review.py", str(CSV_PATH)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    reference = pd.read_pickle(ROOT / "out/live_target_scores.pkl")
    ours = rank_buyers(
        load_target(TARGET_PATH), CSV_PATH, model_path=tmp_path / "model.json", refit=True
    )
    _assert_same(ours, reference)
