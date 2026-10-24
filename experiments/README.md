# Experiments

One-off scripts behind the numbers in `docs/experiment-log.md`. They sit outside the package:
they may import any part of it, they may break when the package changes, and that is accepted.
If something here turns out to be generally useful, it moves into `src/acquirer_engine/`.

`bash experiments/run_all.sh` reruns every one of them, in order (a few minutes, no LLM cost:
the architecture comparison replays committed outputs).

| Script | Question | Log section |
|---|---|---|
| `build_pairs.py` | Buyer x deal pairs, with signals from earlier deals only | 4 |
| `bench.py` | Which model ranks the real buyer highest? | 5 |
| `ceiling_priors.py` | How good could any model be, and what do the banker weights start at? | 5 |
| `time_design.py` | Training over time; do market conditions belong in the ranking? | 5 |
| `example.py` | A worked example, double counting, priors | 5 |
| `review.py` | Banker review: logic fixes, rules tested, the live target | 3 |
| `rules_backtest.py` | How often would each review rule have rejected the real buyer? | 3 |
| `arch_compare.py` | A1 vs A2 (list), W2 vs W3 (pages) | 7, see `FINDINGS_arch_compare.md` |

`archive/` holds superseded scripts; `results/` holds frozen inputs (`profiles_step4/`, the
profiles Part A of the architecture comparison ran on) and result files.

## New experiments

From Oct 24, a new experiment goes in its own dated folder, `YYYY-MM-DD_purpose/`, with numbered
steps (`_common.py`, `01_assemble.py`, `02_run.py`, `03_viz.py`, `04_analysis.py`) and a
`FINDINGS.md` of about one screen: **Status / Intent / Results / Takeaway / Caveats**. The
scripts above predate this and keep their names, because the log and `run_all.sh` cite them.
