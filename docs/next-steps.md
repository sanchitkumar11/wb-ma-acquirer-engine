# What's next

The build plan from here, in priority order. Each step has its files and a definition of done. Update the status column as you go, and record decisions in `docs/experiment-log.md`.

## 0. Deadline check (do this first)

The brief gives **72 hours from receipt** and expects about 3–4 hours of focused work. The brief was uploaded on Oct 21, 2026, so the window may close as early as **Oct 24**. Confirm the exact time.

Everything in Priority 1 is required for a complete submission. Priority 2 strengthens it. Priority 3 is optional. If time is short, cut from the bottom.

## 1. Brief checklist

| Requirement (from the brief) | Status | Where it gets done |
|---|---|---|
| Accept the CSV as the primary data source | **Done** (Oct 23): `acquirer_engine/data.py` loads and validates it | Step 1 (package) |
| Use an LLM to generate the rationale, not just filter or sort | **Done** (Oct 23): ten one-pagers, W3 on Sonnet 5.5, all pass the validator | Step 4 |
| Output all 10 acquirers in a format a banker can act on | **Done** (Oct 23): Markdown report + Streamlit app | Steps 4–5 |
| Runnable with a single command; no manual setup beyond dependencies | **Done** (Oct 23): `python -m acquirer_engine run` from the committed cache; fresh clone: install 29 s, replay 9-13 s | Steps 1, 6 |
| Each one-pager has: Acquirer Overview, Strategic Fit Thesis, Precedent Activity, Valuation Context, Risk Flags (at least 2), Conviction (H/M/L with a 1–2 sentence reason) | Not started | Step 4 |
| End to end in under 60 seconds | Live: 58 s for ten in parallel; replay from cache: ~1 s | Step 4 (parallel calls, caching) |
| README: how to run, architecture (prompt structure, CSV use, improvements), assumptions, limitations, non-determinism | **Done** (Oct 23) | Step 6 |
| No API keys in the repo; environment variables documented | Done (`.env.example`) | Keep checking |
| Recorded demo if a paid key is needed | Script ready (`docs/demo-script.md`); recording by Sanchit | Step 6 |
| Don't hardcode a ranked list | Done by design | Profiles hold facts; rules act on facts |
| Stretch: arbitrary targets / comparison mode / feedback loop | Not started | Steps 8–10 |

## 2. Decisions

**Decided (Oct 23)**

| Decision | Choice | Why |
|---|---|---|
| "Strong EBITDA margin" | **16.8%**, the 75th percentile of the file's 46 Healthcare Services deals | Margins above peers justify upper-quartile pricing (Jazz pitch-book logic). The top 10 is identical for any margin from 15.5% to 18.8%. |
| Region ("regional") | **Midwest**, stated as an assumption | Most Healthcare Services deals in the file (7). Region changes only 0–2 of the top 10; show the region table in the README. |
| Revenue growth | **11.0%**, the Healthcare Services median | – |
| LLM-only ranking tests | **Dropped** | A list that can't show its working wouldn't be accepted |
| What we test instead | **Architecture comparison** (experiment log Section 7): three ways to build the list, four ways to write the one-pagers, two models | Shows the architecture was chosen on evidence |
| LLM provider | **Databricks Model Serving** through LangChain (`databricks-langchain`), behind a provider-agnostic layer; Anthropic key as an alternative; cached replay with no key | Sanchit has Databricks model access in his Claude Code session |

The defaults are in `configs/target_default.yaml`.

**Still open**

| Decision | Options | Recommendation |
|---|---|---|
| Databricks endpoints | Whatever the workspace serves | Record the endpoint names and per-token prices (`configs/llm_prices.yaml`). |
| Writer and judge models | Whatever the workspace serves | Writer: the strongest Claude endpoint. Judge: a different model family. Open-weight model for Part C. |
| Output format | CLI + HTML report; Streamlit; PDF | **CLI that writes one HTML report plus Markdown.** The brief says a clean CLI with excellent prompts beats a polished UI. |
| Buyer profiles scope | All 107 now, or the ~35 that can reach the top 25 first | **~35 first** (default target plus region sensitivities), then expand |
| Review rules and reason codes | As in experiment log Section 3 | Confirm or edit |

## 3. Build plan

### Priority 1: the submission

| Step | What | Files | Done when |
|---|---|---|---|
| **1. Package the engine** | Move the logic from `experiments/` into a package: load and validate the CSV, rings, as-of signals, model fit and score (signs locked), live-target scoring. Target profile from a YAML config. | `acquirer_engine/{config,data,rings,signals,model}.py`, `configs/target_default.yaml`, `acquirer_engine/cli.py` | **Done Oct 23.** `rank` prints the top 25 in 0.2 s with the saved model (`models/ranking_model.json`, refit automatically if the CSV or priors change; ~50 s). Scores match `experiments/review.py` exactly (max difference 0.0): `tests/test_matches_review.py` |
| **2. Buyer profiles (first ~35)** | One YAML per buyer: type, parent, status (e.g. Chapter 11 date), mandate, size (revenue, AUM or fund size), HQ, footprint. Every fact has a source URL and an as-of date. Sources: SEC EDGAR for public companies, MSRB EMMA for nonprofit health systems, sponsor websites and fund announcements. | `data/profiles/*.yaml`, `acquirer_engine/profiles.py` | **Done Oct 23** (time-boxed): 32 profiles, 44 facts (11 EDGAR, 33 hand look-ups); validated by `tests/test_profiles.py`; `review.py` reads them; same Tier 1; `--csv-only` mode. Gaps listed in the experiment log. |
| **3. Ability to pay, valuation, banker review** | Comparable closed deals (core + Ring 1, size band): EV/EBITDA and EV/Revenue percentiles, and where the target sits by margin and growth. Quick LBO for sponsors; synergy headroom for strategics. Review rules with reason codes, plus a review log. | `acquirer_engine/{valuation,lbo,review}.py`, `acquirer_engine/reason_codes.py` | **Done Oct 23.** `python -m acquirer_engine review`: Tier 1 (agreed ten) / Tier 2 with a reason code and source for every move, valuation, quick LBO, synergy headroom, sponsor matrix; rules back-tested (`experiments/rules_backtest.py`); the distinct-angle gate failed and is off |
| **4. Fact packs and LLM one-pagers** | A fact pack per Tier 1 buyer. LangGraph pipeline: fan out per buyer → write (structured output, six sections) → validate (every number in the fact pack, at least 2 risks, conviction consistent) → repair once → render. Build the writing step as swappable strategies (W1–W4) and the review step as A1–A3, so the architecture comparison reuses the pipeline. LLM layer: LangChain with `ChatDatabricks` by default, `ChatAnthropic` as the alternative, chosen by `LLM_PROVIDER`. Prompts versioned in `prompts/`. Cache keyed on prompt version + fact-pack hash; a cost ledger with budget stop. Follow `docs/banker-style.md`. | `acquirer_engine/{factpack,graph,llm}.py`, `prompts/onepager_v1.yaml`, `prompts/system_v1.yaml` | Ten distinct one-pagers; validator passes; end to end in under 60 s with a key; **replays from cache without a key** |
| **5. Output** | HTML report: summary page (action title, criteria table, Tier 1 and Tier 2, sponsor matrix, recommended approach), then ten one-pagers, then the review log. Markdown copy. A committed sample output. | `acquirer_engine/render/`, `samples/default_target/` | **Done Oct 23.** `samples/default_target/report.md` (CLI) and `streamlit run app.py` (cache only); every page shows "Deal data through 2024 Q4"; target form wired, compare and feedback stubbed |
| **6. README, tests, demo** | Final README covering the brief's points, the results tables and "how our thinking changed". Tests: no look-ahead, sign-lock monotonicity, validator, review rules, CLI smoke test. Record a demo (Loom). | `README.md`, `tests/`, link in README | `pytest` passes; a fresh clone runs with one command |

### Priority 2: evidence that strengthens it

| Step | What | Done when |
|---|---|---|
| **7. Architecture comparison** | Experiment log Section 7. Part A: A1 model + rules, A2 + LLM reviewer, A3 tool-using agent. Part B: W1 one call for all, W2 one call per buyer, W3 per buyer + differentiation + repair, W4 plan–write–critique. Part C: the best arms on two models. Back-test on 30 frozen 2022–24 deals (facts dated before each deal); live target run 3 times for stability; blind keep/cut read by Sanchit. If time is short: A1, A2 and Part B first. | Results tables in the experiment log and README, with cost and latency; the chosen architecture named and justified |

### Priority 3: stretch goals

| Step | What | Done when |
|---|---|---|
| **8. Arbitrary targets** | Mostly free once Step 1 takes a config: any sector, size, region, margin, growth. Rings are computed from the target's sector. | `--target configs/<any>.yaml` works for a non-provider target |
| **9. Comparison mode** | Two targets side by side: shared vs distinct buyers, how each buyer's likelihood and conviction change, and why. | `python -m acquirer_engine compare a.yaml b.yaml` |
| **10. Feedback loop** | `flag <buyer> --reason ...` appends to `feedback/feedback.jsonl`. The next run applies it as a review move with reason code USER_FLAG, logs it, and refills the tier. Repeated flags of one type (e.g. "tech investor") become a suggested rule. | Flagging a buyer removes it from Tier 1 on the next run, with a logged reason |

## 4. Target package layout

```
acquirer_engine/
  cli.py            rank | run | compare | flag
  config.py         target profile (YAML) and defaults
  data.py           load and validate the CSV
  rings.py          sub-sector groups and adjacency rings
  signals.py        as-of buyer signals
  model.py          fit and score (per buyer type, banker priors, signs locked)
  profiles.py       dated, sourced buyer facts
  valuation.py      comparable deals and percentiles
  lbo.py            quick LBO / ability to pay
  review.py         banker review rules, reason codes, review log
  factpack.py       evidence bundle per buyer
  llm.py            LangChain client, cache, cost ledger
  graph.py          LangGraph pipeline: write → validate → repair → render
  render/           HTML and Markdown templates
configs/            target_default.yaml and examples
data/               ma_transactions_500.csv, profiles/
prompts/            versioned prompts
samples/            committed sample output
tests/
experiments/        back-tests behind the experiment log (keep as-is)
```

## 5. LLM budget

| Item | Tokens (input / output) | At Sonnet 5.5 list prices |
|---|---|---|
| Part A: A2 reviewer on 30 deals + live runs | ~0.25M / 0.06M | ~$1 |
| Part A: A3 agent on 30 deals + live runs (tool loops are long) | ~1.2M / 0.1M | ~$3.50 |
| Part B: W1–W4, 3 runs × 10 buyers each (W4 ~3x the calls) | ~1.1M / 0.45M | ~$7 |
| Judge scoring for Part B | ~0.5M / 0.06M | ~$1.50 |
| Part C: best arms on two models | ~0.36M / 0.15M | ~$2 |
| **Total** | **~3.4M / 0.8M** | **~$15** (hard cap $50, raised Oct 23) |

Databricks bills differently from Anthropic, so put the workspace's per-token prices into the cost ledger before the first run. Iterate prompts on a cheaper endpoint, then freeze them.

## 6. Before you submit

1. **Export a clean copy without `references/`.** The guides are paid third-party material. For example, create the submission repo from `git archive HEAD -- . ':!references'`, or run `git filter-repo --path references --invert-paths` on a copy.
2. Scan for keys: `git grep -nE "sk-ant|api_key\s*=" $(git rev-list --all)` should match nothing except this line.
3. Fresh-clone test: new virtualenv, `pip install -r requirements.txt`, the single command, under 60 s.
4. README links the demo video and the sample output.
5. Remove `data/` only if William Blair asks. They supplied the CSV, and the engine needs it to run.
