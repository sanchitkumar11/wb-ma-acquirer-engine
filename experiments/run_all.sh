#!/usr/bin/env bash
# Reproduce every experiment in docs/experiment-log.md. Run from the repo root: bash experiments/run_all.sh  (~4 minutes)
set -euo pipefail
CSV=data/ma_transactions_500.csv
mkdir -p out
echo "== 1. Build buyer x deal pairs (signals computed only from earlier deals)"
python3 -I experiments/build_pairs.py "$CSV" out/pairs.pkl
echo; echo "== 2. Model comparison (Section 5 table)"
python3 -I experiments/bench.py out/pairs.pkl
echo; echo "== 3. Accuracy ceilings and banker starting weights"
python3 -I experiments/ceiling_priors.py out/pairs.pkl "$CSV"
echo; echo "== 4. Training over time and market conditions"
python3 -I experiments/time_design.py out/pairs.pkl "$CSV"
echo; echo "== 5. Worked example, double counting, priors"
python3 -I experiments/example.py out/pairs.pkl "$CSV"
echo; echo "== 6. Banker review: logic fixes, rule tests, live target"
python3 -I experiments/review.py "$CSV"
echo; echo "== 7. Review rules back-tested (package review on the 127 test deals)"
python3 -I experiments/rules_backtest.py "$CSV"
echo; echo "== 8. Architecture comparison, trimmed (A1 vs A2, W2 vs W3): replays the committed LLM outputs, no cost"
LLM_PROVIDER=replay python3 -I experiments/arch_compare.py --part A | tail -25
LLM_PROVIDER=replay python3 -I experiments/arch_compare.py --part B
