# Package layers

Modules in `src/acquirer_engine/` sit in layers. A module imports only from layers below its own,
never from its own layer or above, so there are no import cycles and each layer can be read and
tested without the ones above it. `tests/test_repo.py` checks every runtime import against this
table. Imports under `if TYPE_CHECKING:` don't count: they never run.

| Layer | Modules | What lives here |
|---|---|---|
| 0. Foundations | `paths`, `errors`, `thresholds`, `rings`, `wording`, `report_types`, `lbo`, `onepager` | Constants, exceptions, sub-sector rings, plain wording, the Report record, the quick LBO, the one-pager schema and validator. No first-party imports. |
| 1. Inputs | `env`, `config`, `data`, `profiles`, `signals`, `valuation`, `metrics` | Reading `.env`, the target, the deal file and the profiles; the as-of signals; comparable deals; page metrics. |
| 2. Services | `llm`, `model`, `feedback` | The LLM layer (cache, ledger, budget, providers), the ranking model, deal-team flags and learned rules. |
| 3. Engine stages | `ranking`, `review`, `factpack` | Scoring every buyer, the banker review, the fact packs. |
| 4. Engine | `engine` | One deterministic run: `EngineResult`. |
| 5. Writing | `graph`, `listbuild`, `render`, `summary`, `charts` | The one-pager pipeline, list-building approaches, page and summary content. |
| 6. Summary output | `execsummary`, `figures` | The executive summary (LLM, validated) and the report figures. |
| 7. Assembly | `compare`, `report` | Comparison mode and `report.build`, which runs everything below once. |
| 8. Command line | `cli` | The commands. `app.py` (repository root) sits at this level too. |
| 9. Entry point | `__main__` | `python -m acquirer_engine`. |

`acquirer_engine/edgar/` is a self-contained SEC EDGAR client reused unchanged from an earlier
project; it imports nothing else from the package, and only the research scripts use it.

## Rules that keep the layers honest

- **Generic helpers go down, never up.** If a lower module needs something from a higher one,
  the thing is in the wrong place: move it down (as `Report` moved to `report_types` and the
  banker thresholds to `thresholds`), don't import upwards inside a function.
- **Function-level imports only for weight, not for cycles.** The CLI imports the LLM, LangGraph
  and Plotly modules inside the commands that need them, so `rank` and `review` start fast; the
  LLM layer imports a provider library only when that provider is used. Both import downwards.
- **Experiments and research scripts** (`experiments/`, `research/`) sit outside the package and
  may import any layer. They may break when the package changes; that is accepted.

## Adding a module

1. Decide the lowest layer whose modules it needs, and put it one above.
2. Add it to the table above and to `LAYERS` in `tests/test_repo.py`.
3. Run `pytest tests/test_repo.py`.
