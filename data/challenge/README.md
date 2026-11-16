# Challenge set (hand-authored)

A small extra ledger authored case by case (with AI assistance), without the
random generator. It probes situations the generator may not cover and is **reported
separately** from the held-out synthetic benchmark. It was written before the
test set was evaluated and was never used to fit or tune anything.

- `payments.csv`: 85 prior payments, 2026-04-01 to 2026-09-12
- `incoming.csv`: 40 incoming invoices for a payment run on 2026-09-25
- `labels.csv`: the answer key (12 known duplicates). Never read by the matcher.

All vendors are fictional and appear nowhere else in the project.

## The 12 duplicates and what each one probes

| Incoming | Why it is hard |
| --- | --- |
| Wrenfield Bus Lines, WB-7291 | Exact resubmission, but the vendor has three other $500.00 payments, so the right one must be picked |
| St Olwen Parish Hall, 331 | "Saint" vs "St", and "Invoice #331" vs "331" |
| Quillfeather Books, QF-5371 | Two digits transposed (paid as QF-5731) |
| Pruett Janitorial, PJ-0623 | Re-sent a month later by a vendor that also bills monthly |
| Kestrelmoor Telecom, no reference | Statement copy with the reference left off; the real September bill is in the same batch |
| Zephyr Lane Printing, 000731 | Past-due notice with a $25.00 late fee, so the amount differs |
| Zephyr Lane Printing, 718 | Leading zeros dropped (paid as 000718) |
| Sigrun Solheim Design, SSD-024-R1 | Revised invoice number, new date |
| Oduya Bros. Moving & Storage LLC | Legal name vs trading name, resubmitted months later |
| Okonkwo, Adaeze | No reference and no billing period on either side; the same tutor has legitimate sessions at the same fee |
| Cafe Miraflor, 88213 | Accent and "#" dropped; another $640.00 order from the same caterer is legitimate |
| Brackwater Office Supply, INV-20633 | Already paid twice in history, so two prior payments are valid matches |

## Legitimate look-alikes

Monthly rent with no reference and no billing period, sequential invoice
numbers at a fixed monthly price, a quarterly premium, per-session tutors who
bill the same fee repeatedly, two different payees called Thornbury, two
vendors that both used invoice number 2041, a new vendor whose total happens to
equal an old payment, and one invoice that simply arrived late.
