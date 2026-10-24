# Banker style guide for the output

How the engine's output should read, taken from two sell-side pitch books (paid third-party material, not included in the repo). The one-pager prompts and the renderer follow these rules.

## Patterns from the decks

| Pattern | Where it comes from | How we use it |
|---|---|---|
| **Action titles that state the conclusion**, e.g. "'75th Percentile' Valuation is Justified" | Jazz pitch book, Valuation Conclusions | Every one-pager section opens with a one-line conclusion, then the evidence |
| **Tier 1 / Tier 2 defined by explicit criteria**: size, ability to pay (cash and debt vs needing to issue stock), strategic fit | Jazz, Potential Strategic Partners | Tier 1 = 10 one-pagers; Tier 2 = next ~15 with one line each. A criteria table at the top says what puts a buyer in each tier. |
| **Criteria with "What does that mean?"** under each one | agilon discussion, investor targeting | The criteria table explains each criterion in plain words |
| **Participation scorecard** across precedent deals | agilon discussion, investor scorecard | Sponsor differentiation matrix; precedent table per buyer |
| **Valuation justified by fundamentals**: margins and growth above peers, so value at the 75th percentile | Jazz, Valuation Summary | Valuation Context positions the target within comparable deals by margin and growth, and says which percentile is defensible and why |
| **A valuation summary as numbered findings, one per method**, each a one-line conclusion (implied valuation, comparables, DCF, precedents) | Jazz, Summary of Valuation | Valuation Context: the range first, then numbered findings (comparable deals, this buyer's own precedents, ability to pay) |
| **Risk as concern / mitigant / reality** | agilon, "Turning 'Risk' to Reward" | Each risk flag: the concern, what mitigates it, and the evidence-based reality |
| **Process recommendation**: a targeted approach to Tier 1 while reaching out to Tier 2 in parallel; targeted vs broad sell-side trade-offs | Jazz, Key Recommendations | A closing "recommended approach" box on the summary page |
| **Source and as-of footnotes** | Both decks | Every number cites its source (CSV deal IDs or a dated profile fact), and every page says "deal data through Q4 2024" |

## Writing rules for the one-pagers

1. **Specific beats generic.** Each claim names the buyer's own deals, sizes, multiples and dates. Write "Completed 6 provider deals at $150–300M, median 13.2x EV/EBITDA", never "a leading healthcare company with a track record of acquisitions".
2. **No two one-pagers share a thesis.** If two buyers would get the same rationale, the second needs a differentiator or moves to Tier 2.
3. **Numbers come only from the fact pack.** The validator rejects any number that isn't in it.
4. **Conviction varies and is explained.** High / Medium / Low, with one or two sentences tied to named signals.
5. **At least two risk flags,** drawn from the risk library: antitrust or market overlap, integration complexity (especially Ring 2 and Ring 3 buyers), financing capacity, competitive process, execution record (e.g. no completed deal in the file), and real-world status or mandate notes.
6. **Plain banker English.** Short sentences, no model jargon in the output. "Model probability" becomes "likelihood".
7. **Say what is uncertain.** If the data can't separate sponsors, say so in the matrix rather than inventing precision.
