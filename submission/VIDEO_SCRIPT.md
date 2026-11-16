# Two-minute demo video script

Screen recording with voice-over. No editing needed. Before recording: run
`streamlit run app.py`, click **Load demo**, set the browser to about 1440 px wide.
Say every number exactly as it appears on screen.

| Time | On screen | Say |
| --- | --- | --- |
| **0:00 – 0:15** The problem | Review queue, demo loaded. The two cards: "OC Event Rentals LLC, INV-1042, $468.20" and "O.C. Event Rentals, 1042, $468.20". | "Would you pay this twice? I help with the books of a small nonprofit. Before a payment run, a volunteer treasurer has to catch the invoice that was already paid. Same bill, typed a little differently." |
| **0:15 – 0:50** The matching evidence | Point at each evidence label on the Prior payment card, then "Why this was flagged", then the match score and "Model details". | "Invoice Match takes two CSV files, incoming invoices and prior payments. A learned model ranks every incoming invoice by its closest prior payment. This one is first. Same amount. The references match once INV-dash is removed. Same date, same billing period. These sentences come from plain field comparisons, not from the model. The model gives a ranking score, not a probability, and you can open the details to see what pushed it up. The decision stays with me." Click **Possible duplicate**. |
| **0:50 – 1:15** The legitimate recurring bill | Click **Legitimate recurring bill** in Demo cases. Point at "Different billing period" and the different references. | "Here is why this is hard. Lantern Street Storage, 135 dollars, same vendor, same amount as last month. A simple rule would flag it. But the reference and the billing period are different, so the model ranks it lower. It is just March's bill." Click **Legitimate bill**. Then click **Ambiguous case**: "And when a reference is missing, I can record 'Needs more information'." |
| **1:15 – 1:45** How it works and the baselines | Open **Benchmark**. Point at the table at 20 reviews, then click 10 and back to 20. | "How do I know it works? I built synthetic ledgers with fictional vendors, trained on earlier dates, and tested on later dates and vendors the model never saw: 360 invoices, 20 known duplicates. With 20 reviews, exact matching catches 8. A fuzzy rule I tuned as hard as I could catches 12. The logistic regression model catches 13. That beats exact matching clearly, and the fuzzy rule by one invoice, which is within noise." |
| **1:45 – 2:00** A failure and what is next | Scroll to "Where the learned model fails". | "It also fails. This invoice was re-sent with a new number, and the model ranked it twenty-third, because it looks like the next bill. The data is synthetic, so the next step is a pilot on a real, permissioned ledger. This is a prototype and not financial advice." |

Timing follows the event's suggested structure: problem (15 s), demo (60 s), how it works
and the baseline (30 s), one limitation and next step (15 s).

Upload: YouTube as **Unlisted**, or Google Drive with "anyone with the link can view".
Open the link in a private window to check it plays without signing in.
