# Five-minute live finals outline and Q&A prep

Finals: Saturday, November 21, 11:00 AM ET, in front of AI researchers. Have the app
running with the demo loaded and the Benchmark page open in a second tab.

## Outline (5:00)

| Time | Part | What to show and say |
| --- | --- | --- |
| 0:00 – 0:40 | The user and the moment | A volunteer treasurer, the evening before a payment run, a spreadsheet of bills, nobody to double-check. Show the two $468.20 invoices: "Would you pay this twice?" |
| 0:40 – 1:50 | Live demo | Load demo. Disguised duplicate: same amount, references match after normalization, same date and period. Record "Possible duplicate". Jump to the legitimate recurring bill: same vendor and amount, different reference and billing period. Jump to the ambiguous case: reference missing, record "Needs more information". Export the reviewed CSV. |
| 1:50 – 2:50 | How it works | Every incoming invoice is compared with every prior payment: 13 pair features (TF-IDF character n-grams and RapidFuzz for vendor names, reference similarity and exact agreement, amount, date gap, billing period, missing flags) into logistic regression. Evidence sentences come from field comparisons, not from the model. Open "Model details" once. |
| 2:50 – 4:00 | Did it beat a simple rule? | Benchmark page. 360 held-out invoices, 20 known duplicates, 20 reviews: exact match 8, tuned fuzzy rule 12, learned model 13. Say plainly: a clear win over exact match, a one-invoice lead over the fuzzy rule, which is within noise; validation was 49 vs 43 of 60, the hand-written challenge set 11 vs 9 of 12. Precision and recall, not accuracy. |
| 4:00 – 4:40 | Where it fails | The re-sent invoice with a new number, ranked 23rd. Freelancers flagged more often than companies (4 of 93 vs 3 of 247). |
| 4:40 – 5:00 | Limits and next step | Synthetic data, prototype, not financial advice. Next: a pilot on one nonprofit's permissioned ledger. |

**One main takeaway:** a small, readable model can put the likely duplicates at the top
of a short review list, and an honest benchmark shows exactly how much that adds over a
simple rule.

## Three technical points to be ready to explain

1. **Leakage prevention.** Splits are made before any pair exists, by vendor group
   (a vendor, its aliases and its look-alike neighbours stay together) and by time
   (train to Aug 2024, validation to Jun 2025, test after). TF-IDF, scaler and classifier
   are fitted on training pairs only. Settings were chosen on validation and frozen in
   `results/frozen_config.json` before the test split was even generated. One noise
   channel formats every record, and a formatting-only classifier scores at chance.
2. **Why the metric is "correct matches in 20 reviews".** Duplicates are rare, so
   accuracy is meaningless ("flag nothing" is 94% accurate). A treasurer has limited
   time, so the question is what the top of the list contains. A proposal only counts
   when it points at the right prior payment. All methods share the same normalization
   and the fuzzy rule was tuned as hard as the model.
3. **What the score is.** `score = intercept + Σ coefficient × standardized feature`,
   the log-odds from logistic regression, squashed to 0–1 for display. It orders the
   queue. It is not a calibrated probability: the training data has about 1 positive in
   4,300 pairs and synthetic prevalence, so "0.90" does not mean 90% likely.

## Q&A prep

**How do you know test data did not leak into training?**
Different vendor words per split, later dates per split, labels in separate files,
features describe pairs only (no IDs, no single-row features). The test split did not
exist on disk until the configuration was frozen. Tests check vendor-word disjointness
and date order, and a formatting-only classifier is asserted to score at chance.

**Your results look modest. Is the model worth it?**
Against exact matching, yes: 13 vs 8. Against a rule I tuned on the same validation data
with the same criterion, the test-set lead is one invoice. I report that as it is. The
model's advantage shows on harder cases: it caught 5 of 6 late-fee duplicates on
validation where the fuzzy rule caught none, because the rule requires equal amounts.

**How does it tell a duplicate from a legitimate recurring charge?**
Both share vendor and amount. The duplicate also shares the reference or the billing
period and often the date. The recurring bill has a new reference and the next billing
period. The model learned a negative weight for "different billing period" and for
references that are similar but not equal (next invoice number).

**Why is the reference-similarity coefficient negative?**
Consecutive invoice numbers (INV-2031, INV-2032) are very similar and belong to different
bills. Exact agreement has its own positive feature. The price is the blind spot we show:
a re-sent invoice with a revised number looks like the next bill.

**Why logistic regression and not something bigger?**
LightGBM on the same features found 51 of 60 validation duplicates against 49. The rule
I wrote down first said switch only for a gain of 3 or more. I kept the model whose
coefficients I can read and show to the treasurer.

**What does a match score of 0.99 mean?**
That the pair ranks very high, nothing more. It is a ranking score. The app caps the
displayed value at 0.99 and says so. The reviewer always decides.

**What are the limits of synthetic data?**
I wrote the generator, so the benchmark tests the duplicates I imagined. Real ledgers
have messier names, credit memos, partial payments and far fewer duplicates (ours are
about 5.6% so the test is measurable), so real precision would be lower. The hand-written
challenge set is a small check outside the generator, not a substitute for real data.

**How would a pilot measure real performance?**
With one nonprofit's written permission: export 12 months of paid bills, remove bank
details, replay each past payment run, and have the treasurer mark what was truly a
duplicate. Report the same table (reviews used, correct, false flags, missed) against
exact matching and the fuzzy rule, plus minutes spent per run. Data stays on their
computer; the app already runs locally and stores nothing.

**Could it be unfair to anyone?**
A false flag delays a payment. On the test set legitimate invoices from individuals were
flagged more often than those from organizations (4 of 93 vs 3 of 247). Small counts,
but I would track it in a pilot and never auto-hold a payment.

**What happens with bad input?**
Missing columns, non-ISO dates, bad amounts and empty files give row-specific messages.
An empty payment history gives an explanation instead of a queue. Exports escape cells
that could run as spreadsheet formulas.

**What did AI tools do?**
Claude (via Claude Code) wrote and tested the code, generated the synthetic data and
drafted the documents from my specification. Be ready to walk through `matcher.py`
(features and model), `evaluate.py` (metrics) and `synth.py` (splits and noise) yourself.

## Code tour (know where things are)

- `ledger.py` `read_ledger_csv`: validation, integer cents, row-level messages.
- `matcher.py` `_pair_features`: the 13 features. `LearnedMatcher.fit`: TF-IDF, scaler,
  logistic regression on training pairs. `_rank_in_chunks`: best payment per invoice,
  tie-breaking. `compare_fields`: the evidence sentences.
- `synth.py` `SPLITS`, `vendor_pool`: who is in which split. `render`: the one noise
  channel. `duplicate_record`: the six duplicate kinds.
- `evaluate.py` `score_budget`: the metric. `search_learned_config`, `search_fuzzy_rule`:
  validation tuning. `run_test_stage`: the only place the test split is read.
