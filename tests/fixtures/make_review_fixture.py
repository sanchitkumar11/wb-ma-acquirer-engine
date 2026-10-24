"""Write review_py_live_scores.csv from the output of experiments/review.py.

Run from the repo root, after `python3 -I experiments/review.py data/ma_transactions_500.csv`:
    python3 -I tests/fixtures/make_review_fixture.py
"""

from __future__ import annotations

import pandas as pd

scores = pd.read_pickle("out/live_target_scores.pkl")
scores[["model_rank", "acquirer", "sponsor", "s", "share"]].to_csv(
    "tests/fixtures/review_py_live_scores.csv", index=False, float_format="%.17g"
)
print(f"wrote {len(scores)} buyers")
