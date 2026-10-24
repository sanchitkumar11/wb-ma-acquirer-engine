# Architecture comparison (trimmed): findings

**Status:** done Oct 23 ($3.70); replays from the committed cache (`arch_compare.py --part A|B`).

**Intent:** does an LLM reviewer improve the rules' list (A2 vs A1)? Do peer lines and a repair
round improve the pages enough to pay for themselves (W3 vs W2)?

**Results**

| List, 30 frozen 2022-24 deals | A1 rules | A2 + LLM reviewer |
|---|---|---|
| Real buyer in Tier 1 / on the list | 46.7% / 93.3% | 46.7% / 93.3% |
| Real buyer moved out of Tier 1 | 0 | 2 (2 others moved in) |
| Moves per deal | 0 | 1.8 |

| Pages, live target, 3 runs | W2 | W3 |
|---|---|---|
| Pass the validator (final) | 4-5 of 10 | 10 of 10 |
| Highest thesis overlap | 0.29-0.43 | 0.22-0.24 |
| Cost / time per run | $0.52 / 33-48 s | $0.79 / 51-55 s |

**Takeaway:** A1 builds the list; A2's moves are offered as cited suggestions, applied only
through the feedback loop. W3 writes the pages.

**Caveats:** 30 deals: one deal is 3.3 points, so small gaps are ties. The back-test can only
score whether the real buyer is listed, not A2's footprint judgments. A3, W1, W4 and a second
model were not run.
