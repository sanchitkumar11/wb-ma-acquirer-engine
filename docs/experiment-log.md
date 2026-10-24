# M&A Acquirer Engine: design and experiment log

How the engine was designed, what we checked in the data, what we tested, and how our thinking changed. This feeds the README. Every number here can be reproduced with `bash experiments/run_all.sh`.

## Summary

The banker's ask, "the 10 most likely buyers for a $200M healthcare services company, with a one-page rationale for each", is two different problems:

- **Who?** A ranking problem. History can answer it, and we can test the answer against what actually happened.
- **Why?** A writing problem. This is where an LLM adds value, but its output can't be scored against history.

So the engine works in four moves:

1. **Rank** all 107 buyers in the file with a model trained on who actually bought what.
2. **Review** the list the way a banker would. Remove buyers that no longer exist, merge duplicates, question buyers that don't fit or can't pay, and promote buyers the file under-rates, with evidence for every move.
3. **Build a fact pack** for each buyer that stays on the list.
4. **Write** the one-pagers with an LLM, using only those fact packs.

We tested the ranking on deals from 2022–24 that the model never saw. The actual buyer was in the model's top 20 for 90% of deals, against 65% for a "most active buyer" list.

---

## 1. How the problem breaks down

| # | Stage | Question it answers | How | How we check it |
|---|---|---|---|---|
| 1 | Target profile | What are we selling? | Sector, EV, revenue, margin, growth, region, ownership. Defaults are data medians; every input can be changed. | – |
| 2 | Buyer universe | Who could buy? | All 107 acquirers in the file, tagged as financial sponsor (PE firm) or strategic (operating company), plus a dated, sourced real-world profile for each. | 15 of 142 recent deals went to buyers new to the file; this is a stated limit. |
| 3 | Buyer signals | What does each buyer's track record say about this target? | 16 signals from the buyer's earlier deals: the 14 in Section 2, with the three banker-logic fixes of Finding 5. | Computed only from deals before the date being scored. |
| 4 | Ranking model | How likely is each buyer to do this deal? | One logistic regression per buyer type, starting from banker-set weights, signs locked. | Back-test on 2022–24 (Section 6). |
| 5 | Ability to pay and valuation | Can they pay ~$200M? What price range is defensible? | Sponsors: a quick, back-of-envelope LBO. Strategics: synergy headroom and size. Price range from the closest precedent deals. | Compared with multiples actually paid in precedent deals. |
| 6 | **Banker review** | **Would a banker sign off on this list?** | **Rules first (status, duplicates, mandate, capacity), then an LLM reviewer for judgment calls. Every move is logged with a reason and evidence (Section 3).** | **Each rule is tested against history; the LLM reviewer is tested against rules-only.** |
| 7 | Fact pack | What can we say about each buyer? | Precedent deals, the signals that drove the score, valuation numbers, review notes, and the real-world profile with sources and dates. | Every number has a source. |
| 8 | One-pagers | Why this buyer, and what could go wrong? | The LLM writes six sections using the fact pack only. | Automatic checks: every number must appear in the fact pack, at least 2 risk flags, conviction consistent with rank. One repair attempt. |
| 9 | Output | What does the banker get? | Tier 1: top 10 with one-pagers. Tier 2: next ~15 with one line each. Plus a sponsor differentiation matrix and the review log. | Runs in under 60 seconds. |

Why split it this way:
- **Test what can be tested.** "Who" is checked against real outcomes.
- **Separate what the CSV knows from what a banker knows.** The model only sees the file. The review adds what the file can't contain, such as a bankruptcy, a merger or an investment mandate, and every such fact carries a source and a date.
- **The LLM never supplies facts.** It writes, and it makes judgment calls in the review, only from evidence that code has assembled.
- **Each stage can be checked or replaced on its own.**

---

## 2. How the ranking model works

**Training data: every past deal becomes a multiple-choice question.** "Of the buyers active before this deal, which one did it?" Each buyer–deal pair is one row: 1 for the actual buyer, 0 for everyone else. That gives 495 deals and 36,214 rows. In 390 of those deals, the actual buyer had an earlier deal in the file.

**Signals: 14 numbers per buyer–deal pair, each a criterion a banker uses to screen buyers.**

| Group | Signal | Plain meaning | Banker reason |
|---|---|---|---|
| Sector | Sub-sector share | Share of the buyer's deals in the target's sub-sector | Already owns this kind of business |
| | Adjacent sub-sector deals | Deals in the same or neighbouring sub-sectors, recent ones counting more | Expanding into a neighbouring line |
| | Close precedents | Past deals that look like this one (sub-sector, size, region, margin), recent ones counting more | Has done this deal before |
| Size | Size fit | How close the deal is to the buyer's usual deal size | Check size and fund size |
| | Size in range | Deal falls within the buyer's past size range | Has written a cheque this big |
| Track record | Activity | Number of past deals | Active acquirer |
| | Recency | Past deals, weighted so a deal 3 years ago counts half | Still buying |
| | Close rate | Share of its deals that closed | Serious buyer who gets deals done |
| | Auction experience | Average number of bidders in its past deals | Comfortable in a competitive process |
| Deal profile | Region | Has bought in this or a neighbouring region | Regional density, roll-up logic |
| | Ownership type | Has bought from this kind of seller (founder, PE-owned, public, non-profit) | Knows this kind of seller |
| | Deal type | Share of past deals that were control deals of private companies | Does this kind of transaction |
| | Margin fit | Target margin vs what the buyer usually buys | Margins that can support debt (LBO checklist) |
| | Growth fit | Target growth vs what the buyer usually buys | Growth story fits the buyer's thesis |

**Model: one logistic regression per buyer type.** Sponsors buy for returns: leverage, add-ons and exit. Strategics buy for synergies and market position. The data agrees: sponsors spread across sub-sectors, while strategics almost always stay in one.

- Each signal gets a weight.
- The score is the sum of weight × signal, converted to a probability.
- Probabilities are then scaled so they add up to 100% across all buyers for that deal.
- How to read a weight: +1.2 means a buyer one standard deviation above typical on that signal has about 3.3 times the odds (e^1.2).

**Banker starting weights.** We set a starting weight for each signal and buyer type, for example "sub-sector share matters most for strategics". The data then moves each weight as far as the evidence supports. Weights cannot flip sign (Section 3.2).

### Worked example: a deal the model never saw

**Madison Dental Solutions** (Q2 2022): Dental, $119M, Mid-Atlantic, management buyout, 12.3% margin, 18.4% growth. 95 buyers were active before it.

| Model rank | Buyer | Type | Model probability |
|---|---|---|---|
| 1 | Vista Equity | Sponsor | 5.6% |
| **2** | **Heartland Dental (actual buyer)** | **Strategic** | **5.5%** |
| 3 | General Atlantic | Sponsor | 5.4% |
| 4 | Blackstone | Sponsor | 5.4% |
| 5 | Bain Capital | Sponsor | 5.3% |

Why Heartland Dental scores high, compared with a typical buyer:

| Signal | Heartland Dental's record | Effect on odds |
|---|---|---|
| Sub-sector share | 3 of 3 earlier deals in Dental | ×56 |
| Region | Bought in the Mid-Atlantic before | ×1.24 |
| Adjacent sub-sector deals | Recent deals, all in Dental | ×1.23 |
| Margin fit | Target margin closer to its usual than most buyers | ×1.18 |
| Close rate | Every deal closed | ×1.15 |
| *Against:* ownership type | Never bought from this kind of seller | ×0.90 |

What the example shows:
- **For strategics, one fact dominates: they buy in their own sub-sector.**
- **The model can't separate the sponsors.** Four of them sit within 0.3 points of Heartland Dental. Section 3.7 covers how we handle that.
- **A banker would cut the #1 name.** Vista Equity is a software investor in reality. The model can't know that; the banker review (Section 3) exists to catch it.
- **Signals become the rationale.** The top three positive signals and the main negative one go into the fact pack as the "why" sentences.

**A miss: Ascension Health.** Ascension was the recorded buyer of Premier Medical Management (2022, Healthcare Services, $50M), and the model ranked it 25th. Section 3.6 explains why and what brings it forward.

---

## 3. Banker review: making the list survive a banker's red pen

A banker throws out a buyer list that names a bankrupt company, lists the same company twice, puts a software investor on a provider-services list, or names a buyer that can't pay. The model can't see any of that, because none of it is in the CSV. The banker review is the step that can.

### 3.1 Where banker logic enters

| When | What | Example |
|---|---|---|
| Before the model: signal design | Each signal is a banker screening criterion | "Has done this sub-sector before" |
| Inside the model: sign lock | Each signal can only push a buyer in the banker's direction | "More similar precedents can never lower a score" |
| After the model: banker review | The list is checked against facts the model can't see | "Steward Health Care filed for Chapter 11 in May 2024, so remove it" |

The sign lock controls **how the model reasons**. The review controls **what it outputs**. Both are banker logic, applied at different points.

### 3.2 How the sign lock works

- During training, each signal's weight has a floor of zero (a bound in the optimizer: SciPy L-BFGS-B, `bounds=(0, None)`).
- If the data pushes a weight below zero, it stops at zero, and that signal is simply ignored for that buyer type.
- Why: the data alone gives sponsors a weight of −0.32 on "close precedents" and −0.16 on "recency". That would mean "having done similar deals makes a sponsor less likely", which is noise from the sponsors looking alike, not insight.
- After the lock, the signals that drop to zero are: recency, close precedents and size-in-range for sponsors; activity and size-in-range for strategics.
- Cost: a tie. 47% vs 49% in the top 10; 91% vs 90% in the top 20.
- Test in the suite: a monotonicity check. Raise any one signal for a buyer, and its score must not fall.

### 3.3 Review steps, in order

1. **Entity and status.** Merge parents, subsidiaries and renamed companies; remove buyers that can't transact.
2. **Mandate.** Does the buyer invest in this kind of business, in reality?
3. **Capacity.** Can the buyer pay? For sponsors: the quick LBO and fund size (the equity cheque is ~$128M on $200M at 5.0x debt; corrected Oct 23 from an earlier ~$80–100M). For strategics: deal size against its balance sheet. The test is one-sided: a deal much smaller than usual is a tuck-in, not a misfit.
4. **Sector rationale by ring.** No ring excludes a buyer. The ring decides what evidence the rationale needs:
   - **Core** (has done the target's sub-sector) and **Ring 1** (same business model: reimbursement, labour, site of care) need no extra evidence.
   - **Ring 2** (adjacent, e.g. a payer or revenue-cycle company moving into care delivery) is a vertical move. It needs a stated strategy and carries an integration risk flag.
   - **Ring 3** (distant: Health IT, devices, pharma) stays in Tier 2 unless there is a cited, dated services strategy. None of their 72 deals in the file were in provider services.
   - For sponsors, the ring shapes the wording ("expanding from its behavioural health platform"), not eligibility. 40% of sponsor deals were outside the sub-sector they had already done.
5. **Promotions from outside the file**, each needing a reason code and a cited fact:
   - buyer archetype (health systems buy physician and ambulatory services)
   - footprint overlap with the target's region
   - an existing platform in the sub-sector (the add-on route)
   - stated strategy in the buyer's own filings
6. **List-level checks.**
   - no duplicates
   - both buyer types considered
   - every Tier 1 sponsor has a distinct angle (a writing requirement, not a gate: as a gate it failed the back-test, see Oct 23 Step 3)
   - adjacency coverage: Tier 2 always shows the best-ranked buyers from Rings 1, 2 and 3, so the banker sees adjacent buyers were considered
   - obvious buyers present: anyone with a recent provider deal of similar size in the region must be in Tier 1 or Tier 2

Every move is logged: from rank, to rank, reason code, evidence, source and date. The log goes in the output, so a banker can see exactly why the list differs from the model.

### 3.4 Rules are tested against history before they are used

A banker rule is a hypothesis too. We applied each in-file rule to the actual buyers of the 127 test deals and asked how many real buyers it would have thrown out:

| Rule | Real buyers it would have rejected | Decision |
|---|---|---|
| No closed deal on record | 3% (4 deals) | Flag only; feeds Risk Flags and Conviction |
| Strategic whose biggest deal is under half this size | 5% (6 deals) | Flag as a "stretch"; feeds Risk Flags |
| Sponsor whose usual deal is >3x or <⅓ of this size | 34% (43 deals) | **Dropped.** Sponsors in the file buy across all sizes, so this rule would have discarded a third of the real buyers. |
| Buyer has never done the target's own sub-sector (core only) | 10% (13 deals) | **Dropped.** Replaced by the ring rationale above. |

An earlier draft gated Tier 1 on "has a provider-services deal on record". It rejected no real buyers of provider targets, but it was dropped anyway: it only works for provider targets (not the arbitrary targets in the stretch goal), and it shuts out adjacent buyers, which the brief asks us to consider.

**Where real buyers came from**, measured by the closest of the buyer's earlier deals to the target's sub-sector (all 390 deals by repeat buyers):

| | Core | Ring 1 | Ring 2 | Ring 3 |
|---|---|---|---|---|
| All buyers | 71% | 16% | 9% | 4% |
| Sponsors (275) | 60% | 23% | 12% | 5% |
| Strategics (115) | 98% | 0% | 2% | 0% |

- **Adjacency matters mainly for sponsors.** 40% of their deals came from outside the sub-sector they had already done.
- **Strategics in the file almost never leave home.** The only two moves were UnitedHealth (services into insurance) and R1 RCM (revenue cycle into Health IT). In reality, payers moving into care delivery is an established pattern, so Ring 2 payers get a review path with dated evidence rather than a rule that excludes them.

Rules based on real-world facts (bankruptcy, mergers, mandates) can't be back-tested this way: today's facts would leak hindsight into 2022 deals. These rules rely on sourced, dated facts instead, and in back-tests a reviewer only sees facts dated before the deal.

### 3.5 Live example: the model's top 10 before and after review

Target settings, decided Oct 23: Healthcare Services, $200M, Midwest, 16.8% EBITDA margin (the 75th percentile of the file's 46 Healthcare Services deals, i.e. "strong"), 11% growth (their median), private, deal data through Q4 2024.

| Model rank | Buyer | Type | Review |
|---|---|---|---|
| 1 | Advocate Health | Strategic | Pass. Merge in Atrium Health (#5). |
| 2 | Steward Health Care | Strategic | **Remove: status.** Filed Chapter 11 in May 2024. |
| 3 | Nordic Capital | Sponsor | Pass |
| 4 | UPMC | Strategic | Pass |
| 5 | Atrium Health | Strategic | **Merge: entity.** Combined with Advocate Aurora to form Advocate Health in December 2022, so it is the same buyer as #1. |
| 6 | Veritas Capital | Sponsor | **Flag: mandate.** Government and healthcare technology investor in reality; usual deal $38M. Move to Tier 2. |
| 7 | General Atlantic | Sponsor | Pass |
| 8 | Vista Equity | Sponsor | **Flag: mandate.** Software investor in reality, although the file gives it 19 provider deals. Move to Tier 2. |
| 9 | Blackstone | Sponsor | Pass (fund-size check pending profiles) |
| 10 | Welsh Carson | Sponsor | Pass |

**Four of the model's top 10 would draw a banker's red pen.**

After the rules-only review, Tier 1 is:
- Advocate Health (including Atrium), Nordic Capital, UPMC, General Atlantic, Blackstone, Welsh Carson, GTCR, TPG Capital, Advent International and Intermountain Health.
- Francisco Partners (#14) is skipped for the same mandate reason as Vista and Veritas.

That leaves 7 sponsors and 3 strategics, which is why sponsor differentiation (Section 3.7) and judgment-call promotions (Section 3.6) matter.

**How sensitive is the list to the two target assumptions?** We changed one at a time and compared the model's top 10:
- **Margin:** anywhere from 15.5% to 18.8% gives the same ten names, only reordered. Only an extreme 25% swaps three. So the margin choice matters for Valuation Context and the quick LBO (how much EBITDA, and so debt, supports $200M), not for who is on the list.
- **Region:** changing the region swaps 0–2 of the ten names. The biggest change is Southeast, which brings in Hellman & Friedman and TPG.
- **Why Midwest:** it has the most Healthcare Services deals in the file (7). The final README shows the region table so the assumption is visible.

**Adjacency coverage for this target.** These are the best-ranked strategics from the outer rings, which Tier 2 shows:
- **Ring 1:** VNS Health (#30), CenterWell Home Health (#31), Dental365 (#32).
- **Ring 2:** revenue-cycle companies (Waystar #29, AdvancedMD #34, Ensemble #43).
- **Ring 3:** device makers (Stryker #33, Boston Scientific #35, Zimmer Biomet #42), each flagged as needing a cited services strategy.

The model ranks payers much lower (Molina #49, CVS Health/Aetna #88, Elevance #91), because no payer in the file has bought provider services. A banker would reverse that order, because payers buying care delivery is the stronger real-world thesis. That is a review promotion, and it must cite dated facts.

### 3.6 Ascension Health revisited

**Correction to the earlier "miss".** Ascension's two deals in the file are a 2019 $708M deal that was withdrawn, and the 2022 $50M deal, which is only "rumored". So Ascension is a buyer that has shown appetite twice but never completed a deal in the file.

**What banker logic says, and what the data showed when we tried it in the model:**

| Banker reasoning | Model change we tested | Effect on Ascension |
|---|---|---|
| One failed deal isn't a pattern | Pull ratios for buyers with few deals toward their buyer type's average | 25th → 25th |
| A $50M tuck-in is small for a health system, not a misfit | One-sided size test: penalize only deals bigger than any it has done | 25th → 25th |
| Appetite matters more than follow-through when building a list | Move close rate out of the ranking and into risk flags | 25th → 23rd |

None of these changes lifts it far, because ten other health systems in the file look just like Ascension. What would separate it is outside the file:
- **scale:** one of the largest US nonprofit health systems
- **footprint overlap** with the target's region
- **stated strategy** in its own disclosures

Large nonprofit systems publish these disclosures on MSRB EMMA (municipal bond filings), not SEC EDGAR, so the profile build must use both sources.

**How the review treats Ascension:**
- It may promote Ascension only by citing those dated facts.
- It must add a risk flag either way: "no completed deal in the file (one withdrawn, one rumored)".
- For the live Midwest target Ascension sits 26th, just outside the top 25, so footprint evidence decides whether it makes the list.

### 3.7 Separating sponsors

We measured the sub-problem directly: in sponsor-won test deals, where did the winning sponsor rank among the 15 sponsors only?

| Model | Winner in sponsors' top 3 | Average rank among 15 |
|---|---|---|
| Random order | 20% | 8.0 |
| Current model | 26% | 7.2 |
| + sponsor open-platform signal, and the other banker-logic fixes | 28% | 7.3 |

**The file barely separates sponsors, and no signal we tried changes that.** So the engine won't pretend to rank sponsors precisely. It shows a sponsor differentiation matrix, like the J.P. Morgan participation matrix, and each Tier 1 sponsor's one-pager must state a distinct angle. (Moving a sponsor without one to Tier 2 was tested as a gate on Oct 23 and dropped: it pushed the real buyer out of Tier 1 in 35% of test deals.)

| Differentiator | Source | Example from the file |
|---|---|---|
| Provider-services focus | File + mandate in profile | Nordic: 22 of 24 deals in provider services; Blackstone: 9 of 25 |
| Existing platform in the sub-sector (add-on route; can pay more through synergies and buying at a lower multiple than the platform is valued at) | File: platform deal in the last 5 years | Nordic, Welsh Carson, Advent: yes; General Atlantic, Blackstone: no |
| Usual deal size vs $200M | File + fund size in profile | Nordic's median is $77M, so $200M is a stretch; Blackstone's is $342M, so it is small for them |
| Price behaviour: who pays up | File: median EV/EBITDA paid | Advent 16.4x vs Nordic 12.7x |
| Control vs minority | File: deal types | Bain: 5 minority deals, so a partial-liquidity option for a founder |
| Capital to deploy | Profile: fund size and vintage, dated | From fund announcements and filings |

### 3.8 Rules first, then an LLM reviewer, and we test both

- **Rules first.** They are cheap, deterministic, auditable and back-testable. They handle status, duplicates, mandate and capacity.
- **Then an LLM reviewer for judgment calls**, such as archetype fit, strategy and footprint.
  - It receives the model's top 25 with fact packs and dated profiles, plus a fixed list of reason codes.
  - For each buyer it returns: keep, promote, demote, remove or merge; a reason code; and the fact IDs it relied on.
  - Code rejects any move that cites a fact not in the pack, or a fact dated after the deal.
- **The test** is Part A of the architecture comparison (Section 7).

---

## 4. Domain knowledge we use, and where

| Banker idea | Source | Where it enters the engine |
|---|---|---|
| Buyer screening criteria: sector, size, geography, deal type, seller type, track record | M&A guide; banker buyer lists | The 14 signals (Section 2) |
| Sponsors and strategics buy for different reasons (returns vs synergies) | M&A and LBO guides | Separate model per buyer type, different starting weights, different ability-to-pay tests |
| Signals push in one direction only | Banker reasoning | Sign lock (Section 3.2) |
| A list a banker can sign: no defunct or duplicate entities, on-mandate buyers, buyers that can pay | Banker practice; Jazz and J.P. Morgan buyer lists | Banker review (Section 3) |
| Adjacent sub-sectors (rings): provider services share reimbursement, labour and site-of-care economics; payers and revenue-cycle companies are one step out; Health IT, devices and pharma two | Brief; buyer-universe logic, checked against the data | Adjacent-deal signal, comparable-deals pool, the ring rationale in the review, and the adjacency-coverage check. Ring 3 gets a low weight because none of its buyers bought provider services in the file. |
| Buyer archetypes: health systems buy physician and ambulatory services | Industry practice | Review promotions with a reason code (Section 3.6) |
| Ideal LBO candidate: stable cash flow, strong margins, growth, add-on potential | LBO guide | Margin and growth signals, quick LBO, sponsor one-pager themes |
| Platform vs add-on: a sponsor that owns a platform can pay more for an add-on | LBO guide | Sponsor matrix, review promotions |
| Quick LBO to test ability to pay | LBO guide | Sponsor maximum price. Base case: $200M at 5.0x debt and 8.5% interest gives ~19% IRR. Maximum price is ~$194M at a 20% IRR, or ~$217M if exited at 15x as part of a larger platform. |
| Synergy headroom and accretion intuition | Merger model guide | Strategic ability to pay: can synergies cover the premium? |
| Appetite vs follow-through | Banker practice | A bid counts as interest in the ranking; failure to close goes into Risk Flags and Conviction |
| Multiples are driven by growth, margins and risk | Damodaran | Valuation Context places the target within its comps by margin and growth. The data agrees: EV/Revenue rises with margin in provider services (ρ 0.29). |
| Valuation range chart, precedent participation matrix | Guides; Jazz pitch book; J.P. Morgan agilon deck | Valuation Context, Precedent Activity, sponsor matrix |
| Risk library: antitrust, integration, financing capacity, competitive process; framed as concern / mitigant / reality | Brief; J.P. Morgan agilon deck | Risk Flags (at least 2 per buyer) |
| Tier 1 / Tier 2 by size, ability to pay and fit; targeted vs broad process | Jazz and J.P. Morgan decks | Output structure and process recommendation |
| Action titles that state the conclusion; source and as-of footnotes | Jazz and J.P. Morgan decks | One-pager format |

**Skipped on purpose:** full merger models, purchase price allocation, deferred taxes, NOLs, exchange ratios and three-statement LBOs. None of them changes who belongs on the list.

---

## 5. Data thinking: what we checked and what we decided

| # | Check | What we found | Decision |
|---|---|---|---|
| 1 | Do the valuation fields agree with each other? | EV/Revenue = deal size ÷ revenue in all 500 rows. Stated EV/EBITDA = deal size ÷ EBITDA in only 53. Stated margin = EBITDA ÷ revenue in only 89. | Use the stated EV/EBITDA and EV/Revenue. Never calculate with the EBITDA field. Disclose in the README. |
| 2 | Could the multiples be forward-looking? | Implied ÷ reported EBITDA ranges 0.66–2.63x. 31% of rows would imply EBITDA shrinking. No link to revenue growth (ρ −0.07). | Not forward multiples. Treat as given. |
| 3 | Who are the buyers? | 15 sponsors did 290 of 500 deals (14–25 each). 92 strategics did 210 (1–6 each), almost all in a single sub-sector. | Two models. Strategics are identified by sector; sponsors by activity and size. |
| 4 | Do buyers from Health IT, devices or pharma move into provider services? | None of their 72 deals were in provider services. Only UnitedHealth crosses over, from insurance. | Low weight for those sectors. Evidence over intuition. |
| 5 | Are there enough comparable deals? | Healthcare Services alone has 12 deals at $100–400M. Adding the four provider sub-sectors gives 87. | Comps pool = Healthcare Services plus the four provider sub-sectors. |
| 6 | What counts as "the buyer"? | Of the 127 test deals: 104 closed, 13 withdrawn, 6 pending, 4 rumored. | Any bid counts as interest (a buyer who bid is worth calling). Failure to close goes into Risk Flags. |
| 7 | Statistical tests say some fields look randomly filled | – | Keep every field a banker would use. Note the tests as data-quality caveats; don't drop fields on them. |
| 8 | Look-ahead leakage | Two signals used the scored deal's own fields: its rationale tags and the price paid. | Removed. Accuracy fell by about 1 point, so they weren't inflating the results. |
| 9 | Do banker rules hold in this data? | One sponsor size rule would have rejected 34% of real buyers; a core-only sector rule 10% (Section 3.4). | Test every rule before it becomes a gate. |
| 9b | How far from the target's sub-sector do real buyers come? | 29% of deals by repeat buyers came from Rings 1–3, almost all of them sponsors. | Rings shape the rationale and evidence needed; they never exclude a buyer. |
| 10 | Do market conditions shift over time? | Deal count peaks in 2020–21, then falls to 23 in 2024. Sponsor share (57–61%), median EV/EBITDA (15–17x) and leveraged financing (~13%) stay flat. | No market-condition signals in the model. Today's financing terms go into the quick LBO as inputs. |
| 11 | Buyers new to the file | 15 of 142 deals in 2022–24 went to buyers with no earlier deal. | Stated limit: the list can only include buyers in the file. |
| 12 | Data end vs today | Deal data ends Q4 2024; today is October 2026. | Score as of the data end and label every page with that date. Anything after 2024 comes from dated buyer profiles. |
| 13 | The file vs the real world | It lists merged or renamed companies separately (Atrium Health and Advocate Health; Kindred at Home and CenterWell), includes a bankrupt buyer (Steward Health Care), and shows technology investors (Vista, Veritas, Francisco Partners) doing provider deals. | Banker review handles each with a sourced, dated fact (Section 3). |
| 14 | How much can 127 test deals tell us? | A 50% hit rate on 127 deals has a margin of error of about ±9 points. Scoring every model on the same deals narrows it. | Treat differences of a few points as ties. |

---

## 6. Experiments: which model, and how good is it?

**Setup.**
- Train on deals from 2015–21 and test on deals from 2022–24.
- The test set is the 127 deals whose buyer had an earlier deal in the file.
- Leaky signals are removed throughout.

**Metrics.**
- **Top 10, 20, 25:** share of test deals where the actual buyer appears in the model's top 10, 20 or 25. Bankers typically send 20–25 names.
- **MRR:** the average of 1 ÷ the actual buyer's rank. Higher is better; 1.0 means always ranked first.

**Models tried.**
- **Most active buyer:** rank by number of deals so far. This is the baseline.
- **Per-signal additive (weight of evidence):** fit a small model for each signal separately, then add up their scores. This is the most explainable form, and a pattern from production root-cause scoring. Its weakness is that it double-counts signals that overlap.
- **Joint logistic regression:** all signals in one model.
- **Separate model per buyer type:** one logistic regression for sponsors, one for strategics.
- **Banker starting weights:** λ controls how hard the data must push to move a weight away from the banker's starting value.
- **LightGBM ranker:** gradient-boosted trees. A check on whether a non-linear model finds patterns the linear one misses.

| Model | Top 10 | Top 20 | Top 25 | MRR |
|---|---|---|---|---|
| Most active buyer | 44% | 65% | 68% | 0.14 |
| Per-signal additive | 43% | 83% | 91% | 0.18 |
| Joint logistic regression | 48% | 87% | 98% | 0.16 |
| Separate model per buyer type, data only | 45% | 86% | 98% | 0.19 |
| + banker starting weights (λ=30) | 49% | 90% | 96% | 0.20 |
| **+ banker starting weights, signs locked (chosen)** | **47%** | **91%** | **96%** | **0.18** |
| + banker starting weights, λ=3000 (almost pure banker weights) | 46% | 87% | 96% | 0.18 |
| LightGBM ranker (range over 5 random seeds) | 45–54% | 84–87% | 95–97% | 0.17–0.19 |
| *Ceiling: told whether the buyer is a sponsor or a strategic* | *49%* | *75%* | *75%* | *0.15* |
| *Ceiling: told the buyer type and the sub-sector* | *76%* | *100%* | *100%* | *0.22* |

**Finding 1: adding signals one at a time double-counts.** The three sector signals overlap heavily (correlation 0.55–0.95).
- Fitted one at a time for strategics, each earns +1.3 to +1.6.
- Fitted together, sub-sector share takes +1.51 and the other two drop to about zero.
- So adding separately fitted signals gives a sector specialist about three times the credit (+4.4 vs +1.3).
- That is why the per-signal model trails at top 20 (83% vs 90%).

**Finding 2: the model reaches the limit of what this data can show.** The ceilings are "oracles" given a hint, then picking the most active matching buyers.
- An oracle told only whether the winner is a sponsor or a strategic gets 49% in the top 10. Our model matches that without the hint.
- Told the sub-sector as well, the oracle reaches 76%. The gap between 49% and 76% is the 15 sponsors, which the file can't separate (Section 3.7).

**Finding 3: the right headline is top 20–25, not top 10.**
- A banker sends 20–25 names. At those list sizes the model finds the actual buyer 90–96% of the time, against 65–68% for "most active".

**Finding 4: banker weights are a judgment call.**
- Accuracy barely moves between data-only weights and almost pure banker weights: 45–49% in the top 10, 86–91% in the top 20.
- So we set them on judgment, let the data adjust them, and lock their signs.

**Finding 5: banker-logic fixes to the model are a tie on accuracy, so we keep them for the reasoning.**

| Change | Top 10 | Top 20 | Top 25 |
|---|---|---|---|
| Chosen model (signs locked) | 47% | 91% | 96% |
| + one deal isn't a pattern (ratios for thin histories pulled toward the average) | 46% | 90% | 98% |
| + one-sided size test (penalize only deals bigger than any it has done) | 46% | 91% | 96% |
| + sponsor open-platform signal | 47% | 89% | 96% |
| All three | 49% | 88% | 98% |

They make the explanations bankable: a one-pager will never say "Ascension is too big for a $50M deal". They cost nothing measurable.

**Finding 6: training once is as good as retraining every year.** Same 127 test deals:

| Training design | Top 10 | Top 20 |
|---|---|---|
| Fit once on 2015–21 | 49% | 90% |
| Refit every year on all earlier deals | 49% | 88% |
| Refit every quarter | 50% | 89% |
| Recent deals weighted more (2- or 4-year half-life) | 46–47% | 87–88% |
| Last 4 years of deals only | 45% | 89% |
| Add market-condition signals | 46% | 88% |

- Weights are stable over time. Discarding or down-weighting older deals only adds noise.
- Recency already enters through the signals, and every signal is recomputed as of each deal.
- Market-wide conditions change how many deals happen and at what price, not which buyer wins a given deal. Only conditions that hit buyer types differently (such as interest rates on leveraged sponsors) could move the ranking, and the file shows none.
- Refitting every year, the top-10 rate by test year was: 2018 64%, 2019 60%, 2020 54%, 2021 42%, 2022 47%, 2023 50%, 2024 52%. The early years look better partly because fewer buyers were in the field (52 per deal in 2018 vs 105 in 2024).

**Decisions from the experiments:**
- **Core model:** a separate logistic regression per buyer type, with banker starting weights (λ=30), signs locked, and the three banker-logic fixes.
- **Final model:** refit on all deals through Q4 2024.
- **Evaluation:** report yearly refits as the main evaluation; keep 2022–24 frozen for the architecture comparison.
- **Benchmarks in the README:** most active buyer, per-signal additive, LightGBM.

---

## 7. Architecture comparison (planned)

**Question: which architecture produces the list and one-pagers a banker would send unedited?**

Every arm starts from the same foundation: the model's scores, fact packs and dated profiles. We vary only how the list is reviewed and how the one-pagers are written.

An LLM ranking buyers on its own was considered and dropped (Oct 23). No banker would accept a list that can't show its working, and the brief asks for lists derived from the data.

### Part A: how the list is built

| Arm | How it works | Why test it |
|---|---|---|
| **A1. Model + rules** | The model ranks; deterministic rules adjust the list (status, duplicates, mandate, capacity, rings); every move is logged. | Cheapest, fully auditable baseline |
| **A2. Model + rules + LLM reviewer** | A1, then an LLM makes judgment calls (archetype, footprint, stated strategy) with a reason code and cited, dated facts. Code rejects any move without evidence. | Do judgment calls add value over rules? |
| **A3. Tool-using agent** | A LangGraph agent gets the engine's tools (score buyers, buyer history, profile, quick LBO, comparable deals) and the banker checklist, and builds the list itself. It may name only buyers in the file and must cite tool outputs. | Does letting the LLM drive the workflow beat a fixed pipeline? |

**Metrics.**
- **Back-test** on 30 frozen 2022–24 deals, using only facts dated before each deal:
  - actual buyer in Tier 1, and in Tier 1 + 2
  - real buyers wrongly removed
  - share of Tier 1 names passing the banker checks (no defunct, duplicate, off-mandate or can't-pay names)
- **Live target:**
  - stability: Tier 1 overlap across 3 runs
  - cost and latency
  - a blind read by Sanchit: the lists are shuffled and each name marked keep or cut

### Part B: how the one-pagers are written

Same Tier 1, same fact packs.

| Arm | How it works | Why test it |
|---|---|---|
| **W1. One call writes all 10** | A single prompt writes the ten one-pagers. | Sees the contrast between buyers; risks sameness, a long output and slow runs |
| **W2. One call per buyer, in parallel** | Each call sees only its own fact pack. | Fast; risks ten similar theses |
| **W3. W2 + differentiation + repair** | Each call also gets one line on every other Tier 1 buyer's angle and must say why this buyer differs. Validator, then one repair. | Targets the brief's "not 10 variations of the same generic rationale" |
| **W4. Plan, write, critique** | Plan the thesis and the facts to use; write; a critic applies the banker rubric; one revision. | Is an explicit plan and critic worth the extra cost and time? |

**Metrics.**
- **Automatic:**
  - validator pass rate (every number traces to the fact pack)
  - specificity (data citations per section)
  - distinctness (the highest overlap between any two theses)
  - generic phrases, counted from a banned list ("leading healthcare company", "track record of strategic acquisitions")
  - spread of conviction levels
  - latency (target under 60 seconds end to end)
  - cost
- **Judged:** a banker rubric ("would a VP send this to an MD without editing?"), scored blind to the arm by a model from a different family than the writer, plus Sanchit's read of a sample.

### Part C: which model

Run the winning A and W arms on two models in the Databricks workspace (a frontier model and an open-weight model), comparing quality against cost and latency.

### Fairness and set-up

- **Same inputs:** every arm uses the same target, fact packs and frozen prompts. Prompts are developed on 2020–21 deals and versioned in the repo; outputs are cached.
- **No hindsight:** back-tests only see profile facts dated before the deal.
- **Provider-agnostic LLM layer** through LangChain:
  - default: Databricks Model Serving (`databricks-langchain`)
  - alternative: an Anthropic key
  - with no key, the run replays from cache
- **Pipeline and tracking:** LangGraph runs the review and writing pipeline; LangSmith (optional) traces calls, tokens and cost.
- **Budget:** a cost ledger with a hard $50 stop (raised from $20 on Oct 23).
  - Estimated ~3.4M input and ~0.8M output tokens, about $15 at Sonnet 5.5 list prices.
  - The ledger uses the workspace's actual per-token prices once confirmed.
  - If time is short, run A1, A2 and Part B first; A3 and Part C follow.

**Status (Oct 23):** trimmed version run: A1 vs A2 and W2 vs W3 (A3, W1, W4 and Part C not run). Results and decisions are in the dated entry 'stretch Step 4' in Section 8. Script: `experiments/arch_compare.py`; frozen deals: `experiments/frozen_30.json`.

**Part A results.** Back-test on 30 frozen deals: one deal is 3.3 points, so differences of a deal or two are ties.

| | A1 rules | A2 rules + LLM reviewer |
|---|---|---|
| Real buyer in Tier 1 | 46.7% | 46.7% |
| Real buyer in Tier 1 + 2 | 93.3% | 93.3% |
| Real buyer moved out of Tier 1 by the stage | 0 | 2 (and 2 others moved in) |
| Tier 1 names passing the banker checks | 98.7% | 100% |
| Moves per deal | 0 | 1.8 |
| Live target: Tier 1 the same across 3 runs | 100% (deterministic) | 100% after the refill fix (93% before) |
| Cost and time | $0, 0.3 s | $0.025 and 11 s per review |

**Part B results** (live target, 3 runs each; W3 run 1 is the committed live run, whose time includes the summary).

| | W2 one call per buyer | W3 + peer lines + one repair |
|---|---|---|
| Pages passing the validator, first draft | 5 / 4 / 4 | 6 / 6 / 5 |
| Pages passing, final | 5 / 4 / 4 (no repair) | 10 / 10 / 10 |
| Specificity (cited facts per bullet) | 1.70 / 1.74 / 1.72 | 1.73 / 1.81 / 1.71 |
| Thesis overlap, max | 0.43 / 0.29 / 0.34 | 0.23 / 0.22 / 0.24 |
| Generic phrases, final | 0 | 0 |
| Cost per run | $0.52 / $0.50 / $0.55 | $0.78 / $0.76 / $0.84 |
| Time, ten pages in parallel | 48 / 33 / 36 s | 72* / 51 / 55 s |

---

## 8. How our thinking changed

- **Oct 22:** Use the stated multiples; never calculate with the EBITDA field.
- **Oct 22:** Choose fields by banker logic, not by significance tests.
- **Oct 22:** Rejected a hand-set points scorecard in favour of a learned model that stays explainable.
- **Oct 22:** Real-world profiles for all 107 buyers, from sourced research, SEC EDGAR, and MSRB EMMA for nonprofit health systems. The LLM may only state facts in those profiles.
- **Oct 22:** Revenue and margin are inputs that can be changed, with data medians as defaults.
- **Oct 23:** Dropped the forced 7 sponsors / 3 strategics split. The mix comes from one probability scale.
- **Oct 23:** Moved from per-signal adding to a joint model per buyer type, because the per-signal approach double-counts.
- **Oct 23:** The LLM comparison has no "LLM with no data" test.
- **Oct 23:** Removed the two leaky signals.
- **Oct 23:** Yearly refits become the main evaluation. Market conditions go into the ability-to-pay step, not the ranking model.
- **Oct 23:** LangChain, LangGraph and LangSmith as the framework, with a $20 budget. Ollama offered only as a free fallback.
- **Oct 23:** Added the banker review stage after the model: rules first, then an LLM reviewer, every move logged. Four of the model's live top 10 needed a banker's correction.
- **Oct 23:** Signs locked on the model's weights.
- **Oct 23:** Every banker rule is tested against history before it becomes a gate. The sponsor size rule was dropped.
- **Oct 23:** Sponsors are presented in a differentiation matrix, not ranked precisely; each Tier 1 sponsor needs a distinct angle.
- **Oct 23:** Corrected the Ascension "miss": its deals were withdrawn and rumored, not completed.
- **Oct 23:** The LLM reviewer test replaces the "LLM re-ranks top 25" test.
- **Oct 23:** Target defaults decided: "strong" margin = 75th percentile of Healthcare Services deals (16.8%); region = Midwest (most Healthcare Services deals in the file). The top 10 is the same for any margin from 15.5% to 18.8%, and region changes 0–2 names.
- **Oct 23:** Dropped the LLM-only ranking tests: a list that can't show its working wouldn't be accepted. Replaced with an architecture comparison: three ways to build the list (rules; rules + LLM reviewer; tool-using agent) and four ways to write the one-pagers, plus a model comparison.
- **Oct 23:** LLM calls go through Databricks Model Serving behind a provider-agnostic LangChain layer, with cached replay so the engine runs without a key.
- **Oct 23:** Dropped the "provider-services deal on record" gate. It is replaced by a sector rationale by ring (Core, Rings 1–3) plus an adjacency-coverage check, so adjacent buyers are always considered.
- **Oct 23 (Step 1):** Packaged the ranking as `acquirer_engine/` (config, data, rings, signals, model, CLI), driven by `configs/target_default.yaml`. The live model is the chosen model with all three banker-logic fixes, i.e. 16 signals. `python -m acquirer_engine rank` reproduces `experiments/review.py` exactly: same 107 buyers in the same order, largest score difference 0.0. A test checks this against a fixture and, with `pytest -m slow`, against a fresh run of `review.py`. Building the training pairs takes ~50 s, so the fitted model is saved with the CSV's hash and refit only when the data or the banker weights change; `rank` then takes 0.2 s. The CSV is validated on load (columns, blanks, known sub-sectors and buyer types, positive sizes, one type per buyer). LLM budget raised to $50.
- **Oct 23 (Step 2):** Buyer profiles, time-boxed to about an hour. Region changes the order but not the membership: only 25 buyers reach the model's top 25 for any of the 10 regions. Profiles cover those 25 plus Ascension and the six merged or renamed buyers (Kindred at Home, CenterWell, Change Healthcare, Cerner, Ciox, Amedisys): **32 buyers, 44 facts**. **11 facts from EDGAR** (companyfacts revenue and 10-K/8-K text, each cited by accession number and filing date, through the EDGAR client reused from my earlier project) and **33 hand look-ups** (27 Wikipedia pages, 5 sponsor websites, 1 PR Newswire listing). MSRB EMMA was not used: its documents sit behind an interactive terms page, so the non-profit systems (2–3 facts each) come from Wikipedia, which is secondary; upgrading them to EMMA or system annual reports is a stated improvement. **Not researched:** mandate and fund size for Bain Capital, Hellman & Friedman, New Mountain and Warburg Pincus (none is in Tier 1); revenue for Advocate (source figures inconsistent), Intermountain, CHRISTUS and Bon Secours; stated strategy for every strategic; footprint for HCA and Ardent. EDGAR confirmed four corporate events from the filings themselves: Oracle closed Cerner on 2022-06-08 (8-K Item 2.01), Humana fully acquired Kindred at Home on 2021-08-17 (10-K), UnitedHealth closed Change Healthcare on 2022-10-03 (10-K) and Amedisys on 2025-08-14 (8-K Item 2.01; Amedisys has since been delisted). The review now reads these files instead of lists in code. A merger is a duplicate when the parent is also a buyer in the file (Atrium into Advocate, Kindred into CenterWell, Change and Amedisys into UnitedHealth) and an acquisition when it isn't (Cerner into Oracle, Ciox into Datavant). This also fixes a gap: the old code never actually merged Atrium into Advocate. A technology mandate (Vista, Veritas, Francisco Partners) counts as a mismatch only when the target isn't a technology business, so the rule still works for other targets. Result: the rules-only Tier 1 is identical to Section 3.5 (Advocate Health, Nordic, UPMC, General Atlantic, Blackstone, Welsh Carson, GTCR, TPG, Advent, Intermountain), and a test checks it. `--csv-only` turns the profiles off and says so. Back-tests see only facts dated before the deal (`facts_as_of`). Mandate statements carry the date the page was read, so a back-test treats Vista as unknown before 2026; that is conservative and stated.
- **Oct 23 (Step 3):** Valuation, ability to pay and the banker review are in the package (`valuation.py`, `lbo.py`, `review.py`, assembled by `engine.py`); `python -m acquirer_engine review` prints Tier 1, Tier 2, the numbers behind them, the sponsor matrix and the review log in 0.2 s. **Valuation:** 70 closed control deals in provider services at $100–400M (9 core); stated EV/EBITDA 10.7x / 12.8x / 15.3x (p25 / median / p75), EV/Revenue 1.70x / 2.29x / 2.61x. The target's 16.8% margin is above 87% of them (median 12.9%), so upper-quartile pricing is defensible. Because the target's EBITDA is unknown, the context is framed as the EBITDA a buyer must believe in at each multiple ($13.1–18.6M). **Quick LBO** (reproduces `research/paper_lbo.py`, tested): at $200M with 5.0x debt at 8.5%, IRR 18.7% and an equity cheque of $128M (not the ~$80–100M stated earlier); maximum price $193M at a 20% IRR, or $216M if exited at 15x inside a larger platform. **Strategic synergy headroom:** $15–25M (3–5% of implied opex, capitalised at 12.8x, half realised). **Rules tested against history before use** (`experiments/rules_backtest.py`, 127 test deals, profile facts dated before each deal): model top 10 finds the real buyer 49% of the time (98% in Tier 1 or 2); after the rules 47% (97%), a tie. Status, mandate and Ring 3 rules moved no real buyer; the merger rule moved 3 (buyers that kept appearing under their own name after being acquired, correctly folded into the parent). **Dropped as a gate:** "each Tier 1 sponsor needs a distinct angle". It pushed the real buyer out of Tier 1 in 35% of test deals and cut the sponsor winner's Tier 1 rate from 56% to 39%, the same failure as the sponsor size rule. It stays a writing requirement: the matrix shows each sponsor's angles (open platform, provider specialist, pays up, minority option, size fit or capacity) and each one-pager must state one. With it off, Tier 1 is the agreed ten. New flags (they feed Risk Flags and Conviction, never the tiers): FUND_SIZE_UNKNOWN for sponsors with no researched fund size, CAPACITY_FUND if the cheque exceeds 10% of the fund (none do), CAPACITY_CLEARED when a profile shows the in-file size worry is wrong (HCA: biggest deal in the file $37M, revenue $75.6B). List-level checks add the best Ring 1, 2 and 3 strategics to Tier 2 (VNS Health, Waystar, Stryker) and any buyer with a recent similar-size provider deal in the region (InterDent).
- **Oct 23 (Step 4, one-pagers):** First live runs on Claude Sonnet 5.5 (Databricks Model Serving). **Endpoint findings:** the temperature parameter (any value) and forced tool choice are rejected (400, no charge); an unforced single tool works for a one-field schema but mangles nested arguments (sections arrived as `<parameter name="title">` fragments), so structured output is JSON matching the schema in the prompt, checked with pydantic, one retry. Sonnet's hidden reasoning counts against the output cap: at 4,000 tokens one reply had no answer text, so the cap is 12,000 (you pay only for what is used). Ten parallel calls raced the workspace login refresh; one shared client created under a lock fixed it. Determinism comes from the cache, not temperature. **Prompt v2** (`prompts/onepager_v2.yaml`): XML-tagged inputs (`<target>`, `<peer_angles>`, `<fact_pack>`, `<task>`), a style example trimmed from the approved Nordic page (not shown when writing Nordic itself), the reason for the numbers rule, an explicit audience (the VP writes for the MD; the bar is no edits), length targets, richer peer lines, neutral placeholders in the rules, and an ID on every fact-pack block plus a `derived` block of counts, shares and medians so normal banker phrasing needs no arithmetic. Writing the example exposed a false claim in the approved v1 Nordic page ("the Tier 1 buyer with an open platform": Welsh Carson and Advent have one too); the validator can't catch cross-buyer uniqueness claims, so the richer peer lines now show each buyer's platform status. **Results, W3, all ten:** validator pass rate 5/10 before repair and 10/10 after one repair (all five failures were a number not in the fact pack); thesis overlap (word 3-gram Jaccard) max 0.174 (Welsh Carson / Advent, both open-platform sponsors), mean 0.053; conviction 5 High / 3 Medium / 2 Low; generic-phrase hits 0 in first drafts and final pages; headlines 21-27 words and every section 67-119 words (all inside target), but whole pages 675-822 words against a 450-600 target, because the model writes 3-4 risk flags of three parts each; 101,538 input and 56,047 output tokens, $0.076 per page, 58 s wall clock for the ten in parallel (14 live calls; Advocate came from cache). Hand checks of every "only" claim against the CSV found none false. **Spend:** $1.47 of $50 in the ledger, of which about $0.60 was lost to the integration issues above (reasoning blocks misread, mangled tool arguments, truncated reasoning); failed replies are kept in `out/` and still charged.
- **Oct 23 (prompt v3, after Sanchit's read of v2):** Internal labels were leaking into client-facing text ("Ring 1", "Core", "Tier 1", odds multipliers such as "x83.23"), which an MD can't use. The fact packs now carry plain labels ("same sub-sector as the target", "adjacent provider sub-sector", "priority list", signal strength in words) and no bare reason codes, and the validator rejects those terms in the text. Tier 1 / Tier 2 stays only as headings on the summary page with a criteria table, as in the Jazz deck's "Tier 1 Potential Partners". Layout changed to a deck page (Jazz numbered boxes, J.P. Morgan agilon bulleted boxes): four boxes of a 12-word action title plus 2-3 bullets of at most 25 words, 2-3 risks (capped: the v2 pages ran 675-822 words with 3-4 risks), page target 350-500 words, and a strip of key numbers rendered from the fact pack rather than written by the LLM. Peer lines now give every buyer the same distinguishing numbers (same and adjacent sub-sector deals, regional deals, median deal, median paid, closed share, open platform), so a writer can frame its buyer favourably, as an MD would in a client deck, without claiming a lead the numbers contradict. The identical "return at the asking price" risk across sponsors is kept: the quick LBO really is the same for all of them.
- **Oct 23 (v3 run and Step 5):** v3 on all ten (W3, Sonnet 5.5): 4/10 pass before repair, 10/10 after (first-draft failures: 5 numbers not in the pack, 3 never naming the buyer, 3 boxes with 4 bullets); jargon hits 0; generic phrases 0; thesis overlap max 0.286 (Welsh Carson / Advent, both open-platform sponsors), mean 0.105; conviction 5/3/2; 8 of 10 pages inside every length target (pages 424-520 words, against 675-822 in v2); $0.083 per page; 59.5 s for ten in parallel, so the live path only just meets the 60 s target and the replay path takes 0.4 s. All ten "what sets it apart" claims checked true against the peer numbers. **Output:** `python -m acquirer_engine run` writes the full Markdown report (summary with criteria table, Tier 1 and Tier 2, sponsor scorecard, recommended approach and review log with model rank vs final tier; ten one-pagers; sources), committed as `samples/default_target/`. `streamlit run app.py` reads only from the cache (provider forced to replay): summary, one page per Tier 1 buyer (key-number strip, four boxes, precedent table, football field of comps range, own precedents and LBO maximum or synergy headroom against the ask, risks as concern / mitigant / reality columns, conviction), and a sources page; "Deal data through 2024 Q4" on every page. A sidebar target form re-runs everything deterministic for any profile and marks the one-pagers as not generated; compare and flag-as-irrelevant are stubs. Rendering checks in a headless browser caught two bugs the tests could not: Streamlit reads `$` as LaTeX (amounts turned into formulas) and truncated metric tiles; both fixed. Two reason codes were renamed so the log reads in plain English: DISTANT_SECTOR (was RING3_NO_STRATEGY) and VERTICAL_MOVE (was RING2_VERTICAL).
- **Oct 23 (summary redesign, prompt v4):** The summary page was rebuilt around what an MD reads first. **Root causes fixed:** the hand-written criteria table became a buyer-motive framework from the M&A guide (cost synergies and scale, revenue synergies, geographic expansion, vertical integration, diversification) and the LBO guide (platform build and add-ons, growth), each evidenced by the CSV's rationale tags; every fact pack gained a `motives` block of the buyer's own earlier deals' tags (counts, IDs, and the file-wide rate, because Geographic Expansion is on 90% of deals and says little), never the target's tags; conviction's third point became "top 3 by model score among buyers left after the review" instead of "top 3 of the priority list", so Tier 1 can be sorted by conviction without circularity; the split is unchanged (5 High / 3 Medium / 2 Low, every buyer at the same level). **Structure:** page one is the MD's one-pager (action-title headline from an LLM executive summary, validated like the pages with a template fallback; buyer map beside the summary and approach; call list sorted by conviction then model rank, with a 12-word "why them" and an evidence-based "can pay", e.g. "add-on exit; usual deal $77M, fund $11B" or "comfortable: closed $436M"; a target-level football field). Tier 2 moved to a Bench page in four groups, each with what would promote a buyer and the specific missing point (never a rank); the stage chart, rings, motive table, review log and method note moved to an appendix. **Chart decisions:** buyer map y-axis is median deal / ask on a log scale (the quick LBO is the same for every sponsor and the biggest deal is several times the ask for most buyers, so neither separates them); the funnel is a bar chart because the stages aren't nested; conviction uses a sequential navy ramp because it is ordinal. **v3 to v4 (W3, Sonnet 5.5):** pass before repair 4/10 to 6/10, after repair 10/10 both; jargon 0 and generic phrases 0 both; pages within every length target 8/10 both (v4 pages 462-526 words); thesis overlap max 0.286 to 0.226, mean 0.105 to 0.090; conviction 5/3/2 both; cost per page $0.083 to $0.078; every strategic-fit box cites at least one motive tag; Advent's uncited "low list position reflects limited Midwest history" is gone and its Midwest points now cite T:record, S:geo_fit and K:by_region. Wall clock for the live run was 71.6 s (about 60 s for the ten pages in parallel, two of them needing a repair round, plus 12 s for the executive summary and its repair, which must wait for the pages): over the brief's 60 s on a cold run; the replay path, which the app and a fresh clone use, takes about 11 s including chart export. Executive summary $0.029. Ledger total $3.10 of $50. The visual check in a headless browser caught a log-axis reference line drawn at 10x, overlapping chart labels, and three more `$`-as-LaTeX spots (box titles, risk titles); all fixed.
- **Oct 23 (fixes before Step 6):** **Install:** a reviewer's `pip install -r requirements.txt` backtracked for 20+ minutes on databricks-langchain's dependency chain (mlflow). requirements.txt is now 83 exact pins from a clean resolve of the replay, app, test and experiment packages, each set to the version in the working environment; the two provider libraries and their 91 extra packages moved to requirements-llm.txt (llm.py imports them only for live calls). **Wording:** the three technology investors are now a Tier 2 group, "Technology investors the file shows in provider services", each row stating both facts and an action ("The file credits Vista Equity with 19 provider deals and a platform company in the target's sub-sector; the firm describes itself as ... Confirm appetite for tech-enabled services before outreach"); the claim that their healthcare holdings are software was not used because no profile fact sources it. "Open platform" becomes "existing platform company" wherever text is displayed, and the full meaning (already owns a platform company in the target's sub-sector, bought in the last five years, so this business could be bolted on) is in the column headers, the Tier 2 promote line and the method glossary. This is applied at display time: changing fact-pack text would change their hashes and invalidate the cached pages. **Buyer map:** axes in banker language ("Track record in this kind of business"; "Typical deal size" with dollar ticks $25M to $1.6B and a dashed "$200M ask" line) and a ratio-scale footnote; no "log" on the chart. Conviction colours reverted to the reference blue / orange / aqua, aqua darkened to #14996b so labels read (all palette checks pass; its colour-blind separation from orange is in the floor band, covered by direct labels). **Executive summary:** prompt summary_v3 (summary_v2 had already been used for the example change): the headline describes what the page shows, with no imperative and no digits, and the validator rejects either. The summary pack's "off_mandate" field was renamed so the summary doesn't echo the term. Two summary calls, $0.010 and $0.011. The run log now says "replay (cache)" whenever no paid call was made. Replay of the full report takes 8.8 s. Ledger total $3.12 of $50.
- **Oct 23 (Step 6):** Fresh clone from GitHub on a clean Python 3.12 venv with no `.env` and no provider library: `pip install -r requirements.txt` took 29 s (no backtracking); `python -m acquirer_engine run` replayed the full report from the committed cache in 13.2 s on the first run (36 s of wall clock while Python compiled its libraries) and 9.1 s on the second, logging "replay (cache)" and $0.000, and the regenerated report was byte-identical to the committed one; `pytest` passed in 11 s; `bash experiments/run_all.sh` reproduced every number in this log in 119 s. New tests: no look-ahead (scrambling every later deal and the scored deal's own outcome, tags and price leaves its signals unchanged) and CLI smoke tests (replay with no key; rank and review commands; a never-generated target stops before any spend without --allow-paid): 49 tests. README rewritten to cover every item the brief asks for, with the measured timings: live v3 59.5 s, live v4 71.6 s (pages about 60 s plus the executive summary, which waits for them), replay 9-13 s. `.env.example` fixed to the variable names the code reads (LLM_WRITER_ENDPOINT) and to default to replay. Demo script in docs/demo-script.md; the Loom recording is Sanchit's.
- **Oct 23 (stretch Step 0, finish and tag v1.0):** The executive summary is now built from engine outputs only (call list, can-pay, valuation, review), never from the written pages, so it runs in parallel with the ten pages instead of waiting for them. Prompt summary_v4 caps the headline at 25 words, and the validator rejects anything longer as well as digits or an imperative opening. A cold live run with nothing cached (a separate cache folder via LLM_CACHE_DIR) took **65.3 s**, down from 71.6 s. The critical path is now the slowest page plus its repair round (about 50 s) plus about 8 s of chart export; the summary no longer adds to it. Cost $0.835 (ten pages plus the summary). The live headline was 21 words. Replay takes 9.6 s. `.env.example` now lists every variable the code reads (BROWSER_PATH, LLM_CACHE_DIR, EDGAR_CACHE_DIR, EDGAR_OFFLINE, the per-role endpoint overrides). Tagged v1.0 as the fallback submission. Ledger total $3.95 of $50; stretch block $0.835 of $12.
- **Oct 23 (stretch Step 1, arbitrary targets):** Two contrasting targets run end to end and committed as samples: `configs/target_healthit.yaml` (Health IT, $300M, National, 25% margin, 20% growth, PE-backed) and `configs/target_dental.yaml` (Dental, $80M, Southeast, 18% margin, 8% growth, private). **What already worked:** rings, the comparable pool and the review rules re-centred with no change. Health IT has 13 comparable deals at $150-600M, all Health IT (EV/EBITDA 19.4 / 21.4 / 23.3x); Dental has 63 at $40-160M, 12 of them dental (11.0 / 13.7 / 15.4x). The technology-investor rule does not fire for Health IT (Veritas and Francisco Partners reach Tier 1; Vista is eligible but ranked 21st, because the file credits it with no Health IT deal at all), and for Dental it moves all three to Tier 2. **What broke, and the fixes:** (1) About 20 labels assumed a provider target: ring labels ('adjacent provider sub-sector', 'outside provider services'), the review-log notes, the scorecard's 'Provider deals' column, chart ring titles, the Tier 2 group names, the motive evidence ('of 284 provider deals'), the conviction basis, peer lines and two phrases in the prompts. They are now built from the target's business model (`rings.fit_labels`, `model_words`). For a provider target the output is identical: all ten default fact packs and the summary pack hash the same (a test pins the hashes), so the cached default pages still replay, and the filled prompt matches v4 word for word. The version stays v4 because no cached output was made from a different prompt. (2) The 'provider deals' count (`n_prov`) became deals in the target's business model; for Health IT, Veritas's '10 of 16 provider deals' was really 1 of 16 Health IT deals. (3) A signal that is the same for every buyer (the provider-platform signal for a Health IT target) was named as a factor against Blackstone; such signals are now never named. (4) **Defensible percentile:** the old rule (margin above the comps' median, so upper quartile) would have claimed upper-quartile pricing for Health IT at the 54th margin percentile. The 75th percentile now needs a margin in the comps' top quarter; otherwise the median anchors. Default (87th) and Dental (92nd) are unchanged. (5) **Platform exit:** the fixed 15.0x came from $500M-1B provider deals. It is now computed per target (closed control deals in the target's business model at 2.5-5x its EV, rounded down; 15.0x for the default), falling back to the comparable median when there are fewer than 5 such deals (Health IT: 1) or they trade below it (Dental: 12.5x against a 13.7x median). Otherwise the quick LBO would have assumed a multiple contraction for Health IT (21.4x in, 15x out). (6) The recommended-approach and template sentences assumed sponsors clear the hurdle only with a platform exit; they now follow the actual LBO result (Health IT: 25.3% IRR at the ask; Dental: 14.0%, maximum $68M against an $80M ask, so strategics set the price). They also no longer crash when no buyer is High conviction or no technology investor is moved. (7) The validator read the '65' in 'Dental365' as a number and failed Advent's page; numbers now can't start inside a word. (8) Display: 'In the National' became 'National deals'; empty rings get no caption. **Not researched:** 16 of 25 listed Health IT buyers and 16 of Dental's have no sourced profile facts (Evolent, Epic, Aledade, Privia, McKesson, Heartland, Aspen and others). Each page now says 'Buyer profile: not researched' and Sources lists them; nothing is guessed. **Runs (W3, Sonnet 5.5):** Health IT: 2/10 pass before repair, 10/10 after; conviction 3 / 5 / 2; thesis overlap max 0.364 (Evolent / Privia, both physician-enablement businesses); 5/10 pages within every length target; $0.919, 72.5 s. Dental: 0/10 before repair (8 numbers not in the pack, one of them the Dental365 false failure; 6 pages that never named the buyer), 10/10 after the validator fix; conviction 7 / 3 / 0 (seven buyers with a dental record and, for sponsors, an existing platform); overlap max 0.323; $1.041, 67.9 s. First drafts fail more often on new targets than on the default (6/10), mostly on numbers not in the pack. Build-time code caught a missing case: `report.build` now treats a refused paid call like a cache miss. 14 new tests (63 in all). Ledger total $5.99 of $50; stretch block $2.87 of $12.
- **Oct 23 (stretch Step 2, feedback loop):** "Flag as irrelevant" on every Tier 1 and Tier 2 row in the app (an editable table per list: tick, reason from a fixed list, note; plus a control on each buyer page) and as CLI commands `flag` and `unflag`. Each flag appends one line to `feedback/feedback.jsonl` (target hash, buyer, reason, note, timestamp). The latest line per buyer wins, so a repeated flag changes nothing and `unflag` restores the list exactly; a test checks both. On the next run the review takes each flagged buyer off the list with reason code USER_FLAG, logged with its timestamp and shown as its own stage in the stage chart, and Tier 1 refills from the next eligible buyer. **Design decisions:** (1) Flags don't change other buyers' conviction: the 'top 3 among buyers left after the review' point ignores flagged buyers, so one flag doesn't ripple through every page. (2) Pages below a flagged buyer move up one place, and their packs include the priority number (the text can quote it, e.g. 'ranks 8th on the priority list'), so those pages are regenerated rather than shown with a stale number. A flag near the bottom of Tier 1 costs about one new page; a flag at #2 would cost about five. (3) **Learning is never silent and is tested against history first.** When two or more flags share a reason and the flagged buyers share a property an existing rule acts on (a stated mandate focus for off-mandate, a capacity flag for can't pay, no deal in or next to the region for wrong geography), the method page and the report appendix suggest that rule with the check from Section 3: how many real buyers in the target's sub-sector it would have caught. Only an accept (app button or `rules --accept N`) writes it to `configs/learned_rules.yaml`, scoped to that sub-sector. 'Conflict / already in talks' and 'other' are about one deal and never become rules. **Demo** (`samples/feedback_demo/`, its own feedback file so the main samples stay unflagged): Veritas Capital and Vista Equity flagged off-mandate on the Health IT target. Advent International refilled Tier 1 at #10 and McKesson moved to #9; only those two pages and the summary were new, $0.188, 56 s. The suggested rule ('apply the mandate rule to all technology investors for Health IT targets') would have caught the real buyer in 2 of 50 Health IT deals; accepting it also moves Francisco Partners, which a test covers. Also fixed: the sponsor LBO bar had zero width when the platform exit adds nothing (Health IT, Dental), and the stage chart labelled an empty off-mandate stage 'technology investors'. 8 new tests (71 in all). Ledger total $6.21 of $50; stretch block $3.08 of $12.
- **Oct 23 (stretch Step 3, comparison mode):** `python -m acquirer_engine compare a.yaml b.yaml` and the app's Compare tab put two targets' buyer lists side by side: shared Tier 1 buyers, buyers unique to each (and where the other list put them), one buyer map with both Tier 1 lists (one colour per target, a dotted line joining a buyer on both, each ask as a dashed line), and a short LLM note from a comparison pack, checked like the executive summary. **What changed, and why:** the conviction basis is a list of named points (sector record; add-on route or region; top 3 after the review; concerns). The comparison matches points by name and reports those whose direction differs. A plain text diff was tried first and was misleading: it flagged Blackstone as changed because its deal counts differ, although it scores the same points for both targets. **Demo pair (Health IT $300M vs Dental $80M):** 2 shared buyers (Blackstone High / High; TPG Medium / High, and the only point that differs is the deal-size concern: its $88M median deal is under half the $300M EV) and 8 unique to each. **Note iterations (all validated):** v1 passed the number check but called the Health IT buyers 'software-minded' (not in the pack) and said the dental target was 'better able to support its price', although sponsors clear their 20% return at the Health IT ask (25.3%) and miss it at the dental ask (14.0%). v2 forbids describing buyers beyond the pack's names, types and conviction, and adds whether sponsors clear their target return and their maximum price. Its first run then called 13.7x Dental's defensible multiple (that is the median; the anchor is the 75th percentile, 15.4x, which the pack did not state directly), so the pack now carries the defensible multiple itself. The third note is correct; a test checks it cites 15.4x. **Lesson:** the number validator proves numbers exist, not that they are attached to the right label or that the comparison runs the right way; for comparative claims, give the pack the conclusion as a field instead of making the writer infer it. Cost $0.049 for three notes. Also fixed: `report.build` now really treats a refused paid call like a cache miss (the Step 1 edit had not applied); the compare command keeps the live provider for the note after building both targets from the cache. 6 new tests (77 in all). Ledger total $6.26 of $50; stretch block $3.13 of $12.
- **Oct 23 (stretch Step 4, architecture comparison, trimmed):** Ran A1 against A2 and W2 against W3 (tables in Section 7). Approved spend $6.50; actual $3.70 ($0.53 Part A, $3.16 Part B). **Part A: keep A1.** On 30 frozen 2022-24 deals (model fitted on 2015-21, only facts dated before each deal) the LLM reviewer ties the rules exactly: the real buyer is in Tier 1 46.7% of the time and on the list 93.3% for both. A2 makes 1.8 moves per deal; it moved the real buyer out of Tier 1 twice and in twice. Its only measurable gain is Tier 1 names passing the banker checks, 98.7% to 100% (one flagged name avoided in 30 deals). On the live target its moves are consistent and read as banker judgment: UPMC and Intermountain demoted for having no Midwest footprint (cited profile facts), Bon Secours Mercy (Cincinnati) and Bain Capital promoted. But nothing in the back-test shows it finds real buyers better, and it adds a non-deterministic step to an otherwise auditable list. A better use, not built: show A2's moves as cited suggestions next to the rules' list, never applied silently, as with the deal-team rules. **Bug found by the run:** after demoting two buyers and promoting one, A2 refilled Tier 1 from Tier 2 in model order and brought back Veritas Capital, which the mandate rule had moved to Tier 2. Live stability was 93% because of it. Refills now skip rule-moved buyers; re-scored from the cached replies (no cost), stability is 100% and the back-test is unchanged. A test covers it. **Part B: keep W3.** Without repair, W2 leaves 5-6 of 10 pages failing the validator, so it can't ship as is. Even on first drafts W3 passes more (5.7 against 4.3 of 10, small on 3 runs). The clear effect of the peer lines is distinctness: the highest thesis overlap between two pages falls from 0.29-0.43 to 0.22-0.24. Specificity is the same (about 1.7 cited facts per bullet), and generic phrases are 0 in both. W3 costs about 50% more ($0.79 against $0.52 a run) and takes about 15 s longer (51-55 s against 33-48 s for ten pages), which is the price of 10/10 passing pages and distinct theses. Not run: A3 (agent), W1, W4, Part C (models), and the blind banker read; they stay as stated next steps. Ledger total $9.95 of $50; stretch block $6.83 of $12.
- **Oct 23 (Health IT follow-ups after Sanchit's read of Step 1):** (1) **Profiles for the six unprofiled Health IT Tier 1 buyers**, EDGAR first (`research/profiles/healthit_facts.py` lists each public buyer's 8-Ks since 2025 and the 10-K sentences on acquisitions and footprint): Evolent (status: independent, with an activist cooperation agreement in February 2025; revenue $1,876M; stated acquisition strategy; National Imaging Associates bought for $386.7M in 2023; sold its accountable care business to Privia for $100M in December 2025), McKesson (revenue $403,430M; PRISM Vision $875M and Core Ventures $2.5B; a separation strategy for Medical-Surgical), Privia (revenue $2,123M; a stated strategy of buying provider groups; the Evolent ACO deal; 43 states). Hand look-ups for the private three: Epic, Aledade, Warburg. 4-5 facts each: 37 buyers, 69 facts. **Epic's acquisition stance:** its own site says "We are a software factory (no acquisitions)", and Faulkner's profile says it has never acquired another company. That contradicts the file's four (synthetic) Epic deals. It becomes a severe flag (STATED_NO_ACQUISITIONS) that caps conviction at Low, rather than a gate. A gate would need a test against history, and the only history is the file's own deals for Epic, which the stance says are not real behaviour. Epic stays visible as a 'long-shot call, not a lead'. (2) **Look-alike strategics:** each Tier 1 strategic's pack now carries the stated strategy of the other Tier 1 strategics from the same home sub-sector (`closest_peers`). Aledade reads as a partnership model, not an owner of practices; Privia buys provider groups (and bought Evolent's ACO business). The highest thesis overlap on the Health IT list fell from 0.364 (Evolent / Privia) to 0.204. Default-target packs are unchanged: its health systems have no stated-strategy facts. (3) **Sponsor ordering:** every summary pack with two or more Tier 1 sponsors carries the Section 3.7 finding (winning sponsor in the top 3 sponsors 28% of the time, against 20% at random), and prompt summary_v5 makes the first sentence say plainly that the sponsor order is close to a tie. It is applied to all three targets, because the finding holds for any target, not just Health IT. **Also:** one Aledade page still failed after one repair because it quoted McKesson's 25.1x from the peer lines, a real number but not in Aledade's own pack. W3 now allows a second repair, used only when the first still fails (one call, $0.04), so the rule that every number is in the buyer's own pack stands. The run line now reports the cost from the ledger, including the summary. **Knock-on:** the new profiles change the LLM reviewer's evidence cards, so Part A of the architecture comparison now replays from a frozen copy of the profiles it ran on (`experiments/results/profiles_step4/`), and the feedback demo is kept as a dated record. Health IT conviction is now 2 High / 5 Medium / 3 Low; all ten pages pass. Cost $0.84 (eight Health IT pages, the second repair, four summaries, the comparison note). 5 new tests (82 in all). Ledger total $10.79 of $50; stretch block $7.67 of $12.
- **Oct 23 (A2 decision, after Sanchit's read of Step 4):** A1 stays the list builder. The LLM reviewer (A2) is now an opt-in **suggested moves** panel on the method page (toggle off by default) and `python -m acquirer_engine review --suggest`. Each suggestion shows its reason code, its reason and every cited fact written out in full. Nothing is applied by itself: accept records a deal-team promote or demote in `feedback/feedback.jsonl`, applied on the next run with reason code USER_PROMOTE or USER_DEMOTE; reject records a dismissal, so the suggestion is not shown again. Deal-team moves, like flags, never change other buyers' conviction. **Why suggestion, not application:** the back-test scores one thing, whether the real buyer is on the list, and on that A2 ties A1 exactly. The judgments A2 is good at (UPMC and Intermountain have no Midwest footprint; Bon Secours Mercy is based in Cincinnati) are plausible banker calls, but the back-test cannot score them. A synthetic file's winners don't reflect real footprints, and a fact dated today can't be replayed into 2022. A move the data can't validate goes to the banker as a cited suggestion. On the live target the cached review suggests demoting UPMC and Intermountain (footprint) and promoting Bain Capital (the most Midwest deals of any sponsor on the list), the same moves as in Step 4. One reviewer call refreshed the cache after the new profiles changed the evidence cards ($0.023). 2 new tests (84 in all).
- **Oct 23 (final pass before the blind read):** README and this log brought up to date: the Section 7 results tables are in the README with the decisions; the comparison-mode lesson (the validator proves numbers exist, not that a comparison runs the right way, so conclusions go into packs as fields); the A2 decision (suggestions, not application, because the back-test can't score footprint judgments); a samples table; a time-spent note (measured machine time and LLM spend; people time for Sanchit to fill in). The Anthropic smoke test is not run yet: `.env` has no `ANTHROPIC_API_KEY`, and the Claude Code session's own credentials were deliberately not used for project calls. The README says the provider is wired but untested. Ledger total $10.81 of $50; stretch block $7.69 of $12. 84 tests.
- **Oct 24 (Sanchit's summary-page feedback):** Buyer map: the key (diamond = sponsor, circle = strategic, bigger = more deals) moved from the title to a line under the legend; markers about half the size (Tier 1 8-19 px, Tier 2 5-16 px); Tier 1 labels placed by the same collision check as the comparison map. Headline: it now states the defensible multiple ('upper-quartile pricing at 15.3x EV/EBITDA is defensible'). Prompt summary_v6 makes that multiple the only number allowed in the headline, and the validator checks it is written exactly as in the pack. The summary pack gained `defensible_multiple`, so the multiple comes from the target's own comparable deals (Health IT 21.4x median, Dental 15.4x upper quartile) and the template headline used on replay carries it too. Checked: the same target at $400M recomputes the comparable deals (58 at $200-800M), the anchor (15.6x) and the approach line with no LLM call. Three summaries rewritten, $0.08. 84 tests.
- **Oct 24 (code review and engineering conventions):** A full review of the package against our Python guidelines plus conventions from a mature codebase (layered packages, a src layout, an exception hierarchy, strict tooling, named thresholds, docstrings everywhere, regression tests for every fix). An independent reviewer read the code cold and found 25 issues; each was reproduced before it was fixed. **Bugs fixed:** (1) the app's form allowed targets that crashed the engine (a 0% margin divided by zero in the quick LBO; a $5B dental target had no comparable deals and crashed on round(NaN)); `Target` now validates itself and fewer than 5 comparable deals stops with a reason. (2) The "can't pay" rule suggestion could never fire: flagged buyers lost their capacity flags. (3) Ten parallel page writers could all pass the $50 budget check before any reached the ledger; calls now reserve their worst case (two attempts) under a lock. (4) `build(replay=True)` switched the provider off for the rest of the process. (5) `run --only` crashed for a buyer outside Tier 1 and could overwrite the committed sample. (6) An `as_of` before the data end silently ranked on later deals and still printed "2024 Q4"; the live engine now refuses it, and the label comes from the target. (7) The LLM reviewer crashed on a duplicated move; the agent arm could name removed buyers and replay a stale list (its cache key hashed only the target label). (8) W1 matched pages by position when a name was misspelt. (9) A Databricks page could be replayed for, or overwritten by, an Anthropic run; non-default providers now tag their cache keys, leaving the committed keys unchanged. (10) The comparison treated one concern as two when its amounts differed. (11) The compare tab crashed with the same target on both sides. **Structure:** src/ layout installed as a package (`acquirer-engine` command; no `sys.path` changes in scripts); paths anchored at one root, so the CLI runs from any directory; one exception hierarchy; banker thresholds in one module; the `Report` record in its own module, which removes the one import cycle; ten layers documented in `docs/hierarchy.md` and enforced by a test; one helper for the validated LLM summary and note; one lookup for fund size and revenue (profiles must now carry the amount); structured merge links instead of parsing display notes; the model refits when a signal definition changes. **Tooling:** `pyproject.toml` (direct dependencies with the reason for each; `requirements.txt` stays the exact-pin lock), ruff with type-annotation and import rules (980 findings at the start, 0 now), pre-commit, `CHANGELOG.md`, logging instead of print in library code. Docstrings on every function and class, except the LLM schema models: pydantic copies a class docstring into the JSON schema that goes into the prompt, so those carry a comment instead (the schema hash is unchanged). **Kept as a documented trade-off:** a page's cache key covers its fact pack, not the peer lines in its prompt. Keying on the full prompt would make every flag regenerate all ten pages; the README notes it. **Not adopted, and why:** Typer instead of argparse (no behaviour gained, and a rewrite of the CLI's tests); a uv lock instead of the pinned requirements (the pins already reproduce a clean install in 29 s); moving the existing experiments into dated folders (the log and `run_all.sh` cite them by name; new experiments use dated folders with a FINDINGS.md). **Checks:** every cached output still replays; the default fact packs and the LLM schema hash are unchanged; the refitted model has identical weights; `bash experiments/run_all.sh` reproduces the log. 105 tests (91 plus 14 regressions). No LLM spend.

## 9. Open

Closed on Oct 23: endpoints and prices (`configs/llm_prices.yaml`); the architecture arms (A1, A2, W2, W3 run, Section 7); the review rules and reason codes (README and `review.py`); the output shape (Tier 1 one-pagers, Tier 2 bench, sponsor scorecard, review log).

Still open:
- Sanchit's blind read of the ten default pages (keep / edit / cut per box), then tag v2.0 and build the clean export without `references/`.
- Smoke-test `LLM_PROVIDER=anthropic` on one page against an empty `LLM_CACHE_DIR` (needs an `ANTHROPIC_API_KEY` in `.env`), and add the provider to the cache key.
- Profiles for the remaining buyers (Dental strategics first; all 107 in the end), non-profit facts from bond disclosures.
- Not run: A3 (tool-using agent), W1, W4, Part C (a second model), and a scored test of the LLM reviewer's footprint judgments against real deals.
