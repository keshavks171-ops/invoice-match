"""build_demo.py - build the click-through demo page, submission/demo.html.

Runs the shipped model on the built-in demo ledger and writes every number the
page shows into submission/demo_template.html: the ranking, the field
comparisons, the evidence sentences and the model's feature contributions, plus
the measured benchmark. The page replays these values and scores nothing
itself, so it cannot drift from the app.

Run from the project folder:  python submission/build_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ledger  # noqa: E402
import matcher  # noqa: E402


def record(row) -> dict:
    """One ledger row as plain values for the page."""
    return {
        "row_id": row["row_id"],
        "vendor": row["vendor"],
        "invoice_ref": row["invoice_ref"],
        "amount": ledger.format_cents(row["amount_cents"]),
        "amount_plain": f"{int(row['amount_cents']) // 100}.{int(row['amount_cents']) % 100:02d}",
        "invoice_date": row["invoice_date"].date().isoformat(),
        "billing_period": row["billing_period"],
    }


def main() -> None:
    model = matcher.fit_shipped_matcher(ROOT)
    demo = ROOT / "data" / "demo"
    incoming, _ = ledger.read_ledger_csv(demo / "incoming.csv", "incoming")
    payments, _ = ledger.read_ledger_csv(demo / "payments.csv", "payments")
    queue = model.rank(incoming, payments)
    incoming_by_id = incoming.set_index("row_id", drop=False)
    payments_by_id = payments.set_index("row_id", drop=False)

    items = []
    for entry in queue.itertuples(index=False):
        invoice = incoming_by_id.loc[entry.incoming_id]
        payment = payments_by_id.loc[entry.payment_id]
        comparisons = matcher.compare_fields(invoice, payment)
        contributions = model.contributions(invoice, payment)
        items.append({
            "rank": int(entry.rank),
            "decision_value": round(float(entry.score), 4),
            "match_score": round(min(0.99, max(0.01, matcher.match_score(entry.score))), 2),
            "incoming": record(invoice),
            "payment": record(payment),
            "comparisons": [{
                "field": c.field, "label": c.label, "status": c.status, "evidence": c.evidence,
                "incoming_display": c.incoming_display, "payment_display": c.payment_display,
                "incoming_note": c.incoming_note, "payment_note": c.payment_note,
            } for c in comparisons],
            "flag_reasons": matcher.flag_reasons(comparisons),
            "difference_notes": matcher.difference_notes(comparisons),
            "contributions": [{
                "feature": row.feature, "contribution": round(float(row.contribution), 3),
            } for row in contributions.itertuples(index=False)],
            "intercept": round(float(contributions.attrs["intercept"]), 3),
        })

    payload = {
        "cases": json.loads((demo / "cases.json").read_text(encoding="utf-8")),
        "n_payments": int(len(payments)),
        "items": items,
        "benchmark": json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8")),
    }
    # "</" inside the JSON would end the script tag early, so it is written as "<\\/".
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (ROOT / "submission" / "demo_template.html").read_text(encoding="utf-8")
    page = template.replace("__DEMO_DATA__", data)
    document = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body style="margin:0">\n'
                + page + "</body>\n</html>\n")
    out = ROOT / "submission" / "demo.html"
    out.write_text(document, encoding="utf-8")
    if len(sys.argv) > 1:  # optional: also write the bare page (no html/body wrapper) to this path
        Path(sys.argv[1]).write_text(page, encoding="utf-8")
    print(f"wrote {out.name}: {len(items)} ranked invoices, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
