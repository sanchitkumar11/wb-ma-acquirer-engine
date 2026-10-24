# Demo script (Loom, about 4 minutes)

Record from a fresh clone so the viewer sees the single command work with no key.

1. **The ask (15 s).** "A banker wants the ten most likely acquirers for a $200M healthcare services company, with a one-pager each."
2. **One command (40 s).** In a terminal: `pip install -r requirements.txt` and `pip install -e . --no-deps` (already done), then `python -m acquirer_engine run`. Point at the last line: `replay (cache) ... $0.000`. Open `samples/default_target/report.md`.
3. **The app (2 min).** `streamlit run app.py`.
   - Summary: read the headline; the buyer map (track record against typical deal size, the $200M ask line); the call list sorted by conviction with "can pay" evidence; where the ask sits.
   - One buyer page (Nordic Capital): the key-number strip, the four boxes with citations, the precedent table, the football field, risks as concern / mitigant / reality, conviction.
   - Bench: the technology investors the file shows in provider services, and what would promote each group.
   - How we built the list: the stage chart (Steward removed, Atrium merged, three technology investors kept in Tier 2), the review log with reason codes.
   - Sidebar: change the region to Southeast; lists and charts update instantly; pages say "not generated".
4. **Why it can be trusted (45 s).** Every number on a page is checked against its fact pack; the ranking was back-tested on 127 deals it never saw (actual buyer in its top 20 for 91%); every rule was tested against history before use; every LLM output is cached, so the report replays identically.
5. **Regenerating (15 s).** "With a key and `--allow-paid` the same command regenerates; a cost ledger and a $50 stop apply; a full run costs about $0.80."
