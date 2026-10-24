# The ranking model, explained for questions

Plain-language answers to the questions a technical interviewer is likely to ask about the ranking model: how the training data is built, what "per-signal additive" means and why it double-counts, what "banker starting weights" are, why signs are locked, and what the back-test numbers do and don't show. Numbers are from `docs/experiment-log.md` and reproduce with `bash experiments/run_all.sh`.

---

## 1. The training data: every past deal becomes a multiple-choice question

For each of the 500 deals, ask: *of the buyers that had already done a deal before this one, which one did it?*

- One row per (deal, candidate buyer). The actual buyer gets label 1; every other buyer active at the time gets 0.
- 495 deals have at least one earlier buyer to choose from; that gives 36,214 rows and 390 positives (in 105 deals the actual buyer had no earlier history, so no model built on history can find them).
- Every signal on a row is computed **only from deals dated before that deal**. That is the "as-of" rule. Nothing about the deal itself (its price, tags, deal type, outcome, bidders) is used, because that would leak the answer. We found and removed two signals that broke this rule (tag match and price match); accuracy fell by about a point, so they weren't propping the result up.

**Why this framing:** it turns "predict the buyer" into a standard classification problem, it mirrors how a banker thinks (compare the candidates for *this* deal), and it can be graded against what actually happened.

---

## 2. Logistic regression in two sentences

A logistic regression gives each signal a weight. The score for a buyer is the sum of weight × signal, plus a constant; that score is a log-odds, and `1 / (1 + e^-score)` turns it into a probability.

Every signal is standardised (mean 0, standard deviation 1 on the training data) before fitting, so **a weight of +1.2 means: a buyer one standard deviation above typical on that signal has e^1.2 ≈ 3.3 times the odds** of being the acquirer, other things equal. That is the sentence to use when asked "what does a weight mean?"

Within one deal, the probabilities are rescaled to add to 100% across all candidate buyers, which is the "likelihood" shown in the output.

---

## 3. "Per-signal additive" (weight of evidence): what it is and why it double-counts

**What it is.** Instead of fitting one model with all 14 signals, fit **14 tiny models, one per signal**, each predicting the label from that signal alone. Score a buyer by adding up the 14 individual log-odds contributions (each measured against the base rate). Report the top three positive contributions and the top negative one as the explanation.

This is the weight-of-evidence pattern used in credit scorecards and in production root-cause scoring: each signal's evidence is added independently.

**Why it's attractive.** Every contribution is readable on its own ("sector share: +1.5"), it's trivial to explain, and adding a signal never changes another signal's weight.

**Why it double-counts.** Adding independent contributions assumes the signals are independent. Ours aren't. The three sector signals (sub-sector share, adjacent-sector deals, close precedents) are correlated at 0.55–0.95, because they are three views of the same fact: "this buyer operates in this sector."

- Fitted one at a time (strategics), each earns a weight of **+1.47, +1.60 and +1.32**.
- Fitted together, sub-sector share takes **+1.51** and the other two fall to about **zero** (−0.02, −0.16).
- So the additive model gives a sector specialist **about +4.4 of log-odds** where the joint model gives **+1.3**: the same evidence counted three times.

**What it cost.** On the 127 test deals: per-signal additive finds the real buyer in the top 20 **83%** of the time; the joint model **90–91%**. Top 10 is a tie (43% vs 47–49%), because the double-counting mostly affects how far down the list the real buyer sits.

**What we kept from it.** The explanation style. In the final model each buyer's score is still decomposed into per-signal contributions (weight × standardised value), shown as "good news / bad news" with odds multipliers, e.g. Heartland Dental: sub-sector share ×56, region ×1.24, ownership type ×0.90. The difference is that the weights now come from the joint fit, so they don't overlap.

**If asked "why not just drop two of the three sector signals?"** We could, and the joint model effectively does (it sends their weights to about zero). Keeping them in the fit lets the data decide which view of "sector" carries the information, and keeps them available for sponsors, where the adjacent-sector signal matters more.

---

## 4. Why two models: sponsors and strategics

Sponsors buy for returns: leverage, add-ons, exit. Strategics buy for synergies and position. The data agrees: 15 sponsors did 290 of the 500 deals across every sub-sector; 92 strategics did 210, almost all inside one sub-sector. One shared set of weights would be a compromise that fits neither, so the engine fits **one logistic regression per buyer type** and scores each buyer with its own type's weights. The two sets of weights are genuinely different (sub-sector share: +1.22 for strategics, +0.20 for sponsors).

---

## 5. "Banker starting weights": a prior, in the Bayesian sense

**The problem.** With 390 positives spread over 14 signals and two buyer types, the data cannot pin down every weight. Left to itself it produces some that no banker would accept, e.g. for sponsors, "close precedents" at **−0.32** and "recency" at **−0.16**: having done similar deals recently would *lower* a sponsor's chance. That is noise from the sponsors all looking alike, not insight.

**The method.** Before fitting, write down the weight a banker would expect for each signal and buyer type. Then fit the model with a penalty for moving away from those numbers:

```
loss = (usual logistic-regression loss)  +  (λ / 2) × Σ (w_i − banker_i)²
```

Statistically this is **maximum a posteriori (MAP) estimation with a Gaussian prior centred on the banker weights**; λ is the prior's strength (the inverse of its variance). Ridge regression is the special case where the banker weights are all zero. The intercept is not penalised.

**Reading λ.**
- λ → 0: the data dominates; you get the plain logistic regression.
- λ → ∞: the banker weights are used as-is; the data changes nothing.
- λ = 30 (what we use): the data moves a weight only when it has real evidence. Examples (strategics): sub-sector share, banker +1.0 → data alone +1.40 → blended +1.22. Activity, banker +0.3 → data alone −0.12 → blended 0.00. The data pushed one up and one down, and the prior kept both reasonable.

**The starting weights** (log-odds per standard deviation):

| Signal | Strategic | Sponsor | Banker reasoning |
|---|---|---|---|
| Sub-sector share | 1.0 | 0.4 | Strategics buy what they already are; sponsors roam |
| Adjacent-sector deals | 0.5 | 0.6 | Sponsors expand across provider sub-sectors |
| Close precedents | 0.5 | 0.4 | "Has done this deal before" |
| Size fit | 0.5 | 0.6 | Cheque-size discipline, stronger for funds |
| Size in range | 0.3 | 0.4 | Has written a cheque this big |
| Activity | 0.3 | 0.6 | Sponsors are serial acquirers |
| Recency | 0.3 | 0.6 | Still deploying |
| Close rate | 0.3 | 0.3 | Gets deals done |
| Auction experience | 0.0 | 0.2 | Sponsors are comfortable in processes |
| Region | 0.5 | 0.2 | Regional density matters more to operators |
| Ownership type | 0.2 | 0.4 | Sponsors specialise by seller type |
| Deal type | 0.2 | 0.5 | Control deals of private companies |
| Margin fit | 0.2 | 0.3 | LBO candidate checklist |
| Growth fit | 0.2 | 0.2 | Growth story fits the thesis |

**What it did to accuracy.** Almost nothing: top 10 ranges 45–49% and top 20 86–91% across λ from 1 to 3,000. That is the key finding: **the data cannot settle the weights, so judgment sets them and the data adjusts them.** The prior buys stable, explainable weights at no measurable cost.

**If asked "isn't that just putting your thumb on the scale?"** Two answers. First, we tested the thumb: λ=3,000 (pure banker weights) scores within noise of λ=1 (pure data), so the thumb doesn't move the result; it removes nonsense weights. Second, every prior is written down in the repo, so a banker can disagree with a number and rerun.

---

## 6. Sign lock

Each weight is bounded at ≥ 0 during the fit (L-BFGS-B with `bounds=(0, None)`). If the data pushes a weight negative, it stops at zero and that signal is ignored for that buyer type.

- **Why:** every signal was built as "more of this makes a buyer more likely". A negative weight contradicts the signal's own definition, and a one-pager that said "its similar precedents count against it" would be thrown out.
- **What got zeroed:** for sponsors, recency, close precedents and size-in-range; for strategics, activity and size-in-range.
- **Cost:** a tie (top 10 47% vs 49%; top 20 91% vs 90%).
- **Test:** monotonicity. Raise any one signal for a buyer and its score must not fall.

The sign lock controls how the model *reasons*; the banker review after the model controls what it *outputs*. Both are banker logic, applied at different points.

---

## 7. Three banker-logic fixes to the signals

Each came from a case a banker would object to; each was a tie on accuracy and kept for the explanations.

1. **Thin-history shrinkage.** A buyer with one deal, withdrawn, had a 0% close rate. Ratios for buyers with few deals are pulled toward their buyer type's average (two pseudo-deals), so one failure isn't treated as a pattern.
2. **One-sided size test.** A $50M tuck-in is small for a health system, not a misfit. The penalty applies only to deals bigger than anything the buyer has done.
3. **Sponsor open-platform signal.** A sponsor that bought a platform in the sub-sector in the last five years is a natural add-on buyer.

---

## 8. What the back-test numbers mean

| Model | Top 10 | Top 20 | Top 25 |
|---|---|---|---|
| Most active buyer | 44% | 65% | 68% |
| Per-signal additive | 43% | 83% | 91% |
| Joint logistic regression | 48% | 87% | 98% |
| Per buyer type, data only | 45% | 86% | 98% |
| **Per buyer type + banker weights, signs locked** | **47%** | **91%** | **96%** |
| LightGBM ranker (5 seeds) | 45–54% | 84–87% | 95–97% |
| Oracle told the buyer's type | 49% | 75% | 75% |
| Oracle told type and sub-sector | 76% | 100% | 100% |

- **Test set:** 127 deals from 2022–24 whose buyer had earlier history; the model was trained on 2015–21 and never saw them. Refitting every year gives the same numbers, so the weights are stable over time.
- **Why top 20 is the headline:** a banker sends 20–25 names. At that length the model finds the real buyer 91–96% of the time, against 65–68% for "most active".
- **Why top 10 is capped around 50%:** an oracle told only the buyer's type also scores 49%. Told the sub-sector as well it reaches 76%; the gap is the 15 sponsors, who are statistically alike in this file (in sponsor-won deals, the winner lands in the sponsors' top 3 only 26–28% of the time against 20% at random). That is why the output separates sponsors on fund-level facts rather than claiming a precise order.
- **Margin of error:** on 127 deals a 50% hit rate carries about ±9 points, so differences of a few points are ties. All models are scored on the same deals, which narrows the comparison.
- **Why not LightGBM:** same accuracy, less stable across seeds (45–54%), and no per-signal explanation a banker can read.

---

## 9. Likely questions, short answers

**Why logistic regression and not something more modern?** Because the test said the data holds about 50% top-10 of information for any model, LightGBM confirmed it, and the explanations have to be read by a banker. Simple won on evidence, not taste.

**How do you know nothing leaked?** Every signal is computed from earlier deals only; a test scrambles later deals and the scored deal's own fields and checks that scores don't change. We found two leaks ourselves and removed them.

**Why not choose features by significance?** The dataset has fields that look randomly generated. Choosing by p-value would have dropped fields a banker uses (deal type, geography) and kept noise. We chose by banker logic and tested each rule's cost against history.

**How would it retrain in production?** Refit on all deals quarterly; walk-forward results show refitting changes nothing on this file, so the risk is low. Monitor top-20 recall on each quarter's new deals and the share of sponsor wins the model places in the sponsors' top 3.

**What would move the top-10 ceiling?** Data the file doesn't have: fund vintage and dry powder, current portfolio platforms, stated sector strategies. That is why the profiles and the review layer exist.

**How does conviction work?** By rule, not by the LLM: points for a sector record, an open platform (sponsors) or a regional deal (strategics), and a top-3 model rank among surviving buyers; minus points for severe review flags and sponsors whose usual deal is under half the ask. High ≥ 2, Low ≤ 0. The LLM explains the level; it cannot change it.

**Why 127 deals and not more?** That is every 2022–24 deal in the file whose buyer could be found by a history-based model. Fifteen more went to first-time buyers, which no such model can rank; the README states that limit.

---

## 10. Running it without Databricks

The LLM is Claude Sonnet; Databricks Model Serving is only the route we called it through. `LLM_PROVIDER` selects the route:

- `replay` (default for reviewers): every page and the executive summary replay from the committed cache. No account, no key, about nine seconds. This is what the single command does on a fresh clone.
- `anthropic`: the same model through the Anthropic API with an `ANTHROPIC_API_KEY`. Regenerates live for anyone with a key.
- `databricks`: our route, needs a workspace and token.

So a reviewer can run everything without Databricks, and can regenerate live with an Anthropic key. The Loom demo covers the live path, as the brief asks when a paid key is involved.

---

## 11. Libraries, and why LangChain and LangGraph this time

**What's in the stack**

| Layer | Library | What it does here |
|---|---|---|
| Data and model | pandas, numpy, scipy, scikit-learn | The buyer × deal table, the logistic regression with banker priors (`scipy.optimize.minimize`, L-BFGS-B with the ≥ 0 bounds), the baseline logistic fits |
| Benchmarks only | lightgbm, statsmodels | The LightGBM ranker and the conditional logit in `experiments/`; not used by the engine |
| LLM calls | **langchain-core**, databricks-langchain, langchain-anthropic | One `model.invoke(messages)` interface for every provider; `usage_metadata` gives the token counts the cost ledger needs; `bind_tools` for the tool-using arm |
| Pipeline | **langgraph** | The write → validate → repair graph per buyer, with a conditional edge back to repair; `Send` fans the ten buyers out in parallel |
| Schemas | pydantic | `OnePager`, `Risk`, `Conviction` with field descriptions; the JSON schema goes into the prompt and the reply is parsed and validated against it |
| Profiles | httpx, tenacity | The EDGAR client (rate limiting, retries, disk cache) |
| Output | streamlit, plotly, kaleido | The app; the buyer map, rings, stage chart and football fields; PNG export for the Markdown report |
| Tracing (optional) | langsmith | Traces, tokens and cost per run when `LANGSMITH_TRACING=true`; MLflow tracing is the equivalent on Databricks |

**How LangGraph is actually used.** `src/acquirer_engine/graph.py` builds two graphs. The buyer graph is `StateGraph(BuyerState)` with nodes `write`, `validate`, `repair`, a conditional edge from `validate` to either `repair` or `END`, and `repair` looping back to `validate`; the plan–write–critique variant adds two nodes in front. The run graph fans out with `Send("one_buyer", state)` once per Tier 1 buyer, so the ten pages run concurrently, then collects them. Everything the graph does is visible in about sixty lines.

**Why use them, when the last demo didn't.** The honest answer is that the earlier demo was a single prompt where a framework would have been ceremony. This one has the shape frameworks exist for, and each library earns its place on a concrete need:

1. **Provider independence.** The brief says reviewers may use any LLM provider. With `langchain-core`, Databricks Model Serving, the Anthropic API and the replay cache are one `invoke` call behind a switch. Without it, that is three client libraries and three message formats.
2. **A control flow with a loop and a fan-out.** Write, validate, repair-once, and ten of those in parallel, is a graph. LangGraph makes the loop explicit (a conditional edge), keeps each node a plain function that's easy to test, and gives the run a traceable state. Hand-rolling it is possible, but the branching and parallelism then live in `asyncio` code nobody can read in an interview.
3. **Swappable strategies for the architecture comparison.** The four writing approaches (W1–W4) and the list-building approaches (A1–A3) are different graph topologies built from the same nodes. Comparing them is a one-line change, which is what made the comparison cheap enough to run.
4. **Usage and cost accounting.** `usage_metadata` is uniform across providers, so one ledger function covers them all; LangSmith or MLflow can trace the same calls without code changes.

**Where we deliberately did not use the abstractions.** Three places, each for a reason worth stating if asked:

- **Prompts live in versioned YAML files, not LangChain prompt templates or the Prompt Hub.** The cache key includes the prompt version, and a reviewer must be able to read every prompt in the repo without an account.
- **Structured output is schema-in-prompt plus Pydantic validation, not `with_structured_output`.** The Databricks endpoint rejected the structured-output mode, so the schema is appended to the system prompt and the JSON reply is parsed and validated; the repair node handles failures. Same guarantee, one less dependency on provider features.
- **The validator, cache and cost ledger are plain Python.** They are the controls that make the output trustworthy (every number in the fact pack, at least two risks, conviction fixed, a hard spend stop), and controls should be readable line by line.

**The one-sentence version.** Frameworks are worth their abstraction when the problem has their shape; this one did (multiple providers, a repair loop, a parallel fan-out, interchangeable strategies), the last demo didn't, and in both cases the controls that matter were kept in plain code.
