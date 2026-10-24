"""CLI smoke tests: the single command runs from the committed cache with no key, and says so."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, tmp_path=None):
    env = {**os.environ, "LLM_PROVIDER": "replay", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    return subprocess.run(
        [sys.executable, "-m", "acquirer_engine", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_run_replays_from_cache_with_no_key(tmp_path):
    out = tmp_path / "report.md"
    r = _run("run", "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "replay (cache)" in r.stdout and "$0.000" in r.stdout
    text = out.read_text()
    assert text.count("passes all checks") == 0  # status lines go to stdout, not the report
    assert "Deal data through 2024 Q4" in text and text.count("\n## ") >= 10


def test_rank_and_review_commands(tmp_path):
    assert "Advocate Health" in _run("rank", "--top", "3").stdout
    review = _run("review", "--csv-only").stdout
    assert "CSV-only mode" in review and "Tier 1" in review


def test_paid_call_is_refused_without_permission(tmp_path):
    """A target that was never generated stops before any spend unless --allow-paid is given."""
    target = tmp_path / "t.yaml"
    target.write_text(
        (ROOT / "configs/target_default.yaml").read_text().replace("Midwest", "Southeast")
    )
    env = {**os.environ, "LLM_PROVIDER": "databricks", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "acquirer_engine",
            "run",
            "--target",
            str(target),
            "--out",
            str(tmp_path / "r.md"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode != 0 and "Stopped before any spend" in (r.stdout + r.stderr)
