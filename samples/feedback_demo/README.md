# Feedback-loop demo (Health IT sample target)

Two deal-team flags on the Health IT target, kept in this folder so the main samples stay unflagged:

```bash
export FEEDBACK_PATH=samples/feedback_demo/feedback.jsonl LEARNED_RULES_PATH=samples/feedback_demo/learned_rules.yaml
python -m acquirer_engine flag "Veritas Capital" --reason off-mandate --note "tech-only fund" --target configs/target_healthit.yaml
python -m acquirer_engine flag "Vista Equity" --reason off-mandate --target configs/target_healthit.yaml
python -m acquirer_engine run --target configs/target_healthit.yaml --out samples/feedback_demo/report.md   # replays from the cache
python -m acquirer_engine rules --target configs/target_healthit.yaml      # the suggested rule; --accept 1 writes it
```

What happened on the next run (`report.md`):
- Veritas Capital (Tier 1, #9) and Vista Equity (Tier 2) left the list with reason code USER_FLAG; both show in the review log and the stage chart.
- Tier 1 refilled with the next eligible buyer, Advent International, which became #10. McKesson moved from #10 to #9.
- Only those two pages were new to the cache (Advent's page, and McKesson's, because its priority number changed), plus the executive summary: $0.188 for the run.
- The appendix suggests a rule, not yet applied: both flagged buyers describe themselves as technology investors, so apply the mandate rule to all technology investors for Health IT targets. The history check shows it would have caught the real buyer in 2 of 50 Health IT deals. Accepting it (CLI or app) would also move Francisco Partners to Tier 2.

*Recorded Oct 23 before the Health IT profile update (Evolent, Epic, McKesson, Aledade, Privia, Warburg). That update changed the Health IT fact packs, so this report is kept as the record of the demo run; regenerating it would write new pages (about $0.70).*
