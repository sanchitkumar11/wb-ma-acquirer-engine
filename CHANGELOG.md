# Changelog

All notable changes to the engine. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the version in `pyproject.toml` is bumped by hand. Every decision behind these changes, with its
numbers, is in `docs/experiment-log.md`.

## [2.0.0] - unreleased

### Added
- Arbitrary targets: rings, comparable deals, labels and prompt wording re-centre on the target's
  business model; Health IT and Dental samples.
- Feedback loop: `flag` / `unflag` (CLI and app), USER_FLAG moves with Tier 1 refill, rules
  suggested from shared flags and checked against history, accepted only explicitly (`rules`).
- Comparison mode: `compare a.yaml b.yaml` and the app tab; validated LLM note.
- LLM reviewer as opt-in suggested moves (`review --suggest`, method page); accept or reject
  through the feedback loop.
- Profiles for the Health IT Tier 1 (37 buyers, 69 facts); a stated no-acquisitions policy caps
  conviction at Low.
- Package tooling: `pyproject.toml` (src layout, `acquirer-engine` command), ruff lint and
  format, pre-commit, `CHANGELOG.md`, `docs/hierarchy.md` with a test enforcing the layers.
- One exception hierarchy (`errors.py`): `EngineError` > `LLMError` > `NotCachedError`,
  `PaidCallNotAllowedError`, `BudgetExceededError`.

### Changed
- The executive-summary headline states the defensible multiple (prompt summary_v6).
- Upper-quartile pricing needs an upper-quartile margin; the platform exit multiple is computed
  per target.
- Paths are anchored at the repository root (`paths.ROOT`), so the CLI runs from any directory.
- Banker thresholds live in `thresholds.py`; `Report` lives in `report_types.py` (removes an
  import cycle).
- W3 allows a second repair round when the first still fails.

### Fixed
- Crashes on targets the app allows: a 0% margin, or too few comparable deals (now a clear error).
- The "can't pay" rule suggestion could never fire; fund-size rules no longer report a
  reassuring "caught 0" against a history they can't replay.
- Parallel page writers could all pass the budget check before any reached the ledger
  (reservation under a lock).
- `build(replay=True)` changed the provider for the rest of the process (now scoped).
- `run --only` raised KeyError for a buyer outside Tier 1 and could overwrite the sample.
- An `as_of` before the data end silently used later deals; the live engine now refuses it.
- LLM reviewer: a duplicated move crashed the refill, and the refill could bring back a buyer the
  mandate rule had moved; the agent arm could name removed buyers and replay a stale list.
- A cached Databricks page could be replayed for, or overwritten by, an Anthropic run.

## [1.0.0] - 2026-10-23

The take-home as submitted: ranking model, banker review, fact packs, LLM one-pagers (W3,
prompt v4), Markdown report and Streamlit app, replay from the committed cache.
