# Two-minute demo video script

Screen recording with voice-over. No editing needed. Before recording: run
`streamlit run app.py`, click **Load demo**, set **Review limit** to 10, and make the
browser about 1440 px wide. Say every number exactly as it appears on screen. About 230
words of speech, which leaves room for the clicks. Say only what is true for you; if you
have your own experience with a nonprofit's books, add one sentence about it in your own
words at the start.

| Time | On screen | Say |
| --- | --- | --- |
| **0:00 – 0:15** The problem | Review queue, demo loaded: "OC Event Rentals LLC, INV-1042, $468.20" next to "O.C. Event Rentals, 1042, $468.20". | "Would you pay this twice? Before a payment run, a volunteer treasurer at a small nonprofit has to catch the invoice that was already paid: the same bill, typed a little differently." |
| **0:15 – 0:50** The matching evidence | Point at the evidence labels on the Prior payment card, then "Why this was flagged" and the match score. Click **Possible duplicate**. | "Invoice Match takes two CSV files: incoming invoices and prior payments. A learned model ranks every incoming invoice by its closest prior payment. This one is first: same amount, and the references match once INV-dash is removed. These sentences come from plain field comparisons, not from the model. The score is a ranking, not a probability. The decision stays with the treasurer." |
| **0:50 – 1:15** The legitimate recurring bill | Click **Legitimate recurring bill**. Point at "Different billing period" and "Ranked 13 of 32, below the review limit". Then click **Ambiguous case** and **Needs more information**. | "Here is why this is hard. Lantern Street Storage, 135 dollars: same vendor and amount as last month. But the reference and the billing period are different, so the model ranks it thirteenth, below the review limit. It is just March's bill. And when a reference is missing, the treasurer can record 'Needs more information'." |
| **1:15 – 1:45** How it works and the baselines | Open **Benchmark**. Point at the table at 20 reviews. | "Does it beat a simple rule? The ledgers are synthetic. The model trained on earlier dates and was tested on later dates and vendors it never saw: 360 invoices, 20 known duplicates. With 20 reviews, exact matching catches 8, a tuned fuzzy rule catches 12, and the logistic regression model catches 13. That is clearly better than exact matching, and one invoice better than the fuzzy rule, which is within noise." |
| **1:45 – 2:00** A failure and what is next | Scroll to "Where the learned model fails". | "It also fails: this re-sent invoice has a new number, so the model ranked it twenty-third. The data is synthetic; next is a pilot on a real, permissioned ledger. This is a prototype and not financial advice." |

Timing follows the event's suggested structure: problem (15 s), demo (60 s), how it works
and the baseline (30 s), one limitation and next step (15 s).

Upload: YouTube as **Unlisted**, or Google Drive with "anyone with the link can view".
Open the link in a private window to check it plays without signing in.
