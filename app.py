"""app.py - the Invoice Match review tool.

Run with:  streamlit run app.py

A volunteer treasurer loads the built-in demo or uploads two CSV files
(incoming invoices and prior payments). The learned model in matcher.py ranks
the incoming invoices by how much their closest prior payment looks like an
earlier copy of the same bill. The treasurer reads the evidence side by side
and records a decision. The model only ranks; the decision stays with the person.

Design choices worth knowing:
  * Uploaded data lives only in st.session_state. Nothing is written to disk,
    printed or logged, and the app makes no network calls.
  * Every sentence of evidence comes from matcher.compare_fields (fixed
    templates over plain field comparisons), never from the model.
  * User-supplied text is escaped before it reaches HTML or a widget label.
  * All styling is in assets/app.css, loaded once in main().
"""

from __future__ import annotations

import html
import json
import string
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

import ledger
import matcher

APP_DIR = Path(__file__).resolve().parent
CSS_PATH = APP_DIR / "assets" / "app.css"
DEMO_DIR = matcher.DATA_DIR / "demo"

NOT_REVIEWED = "Not reviewed"
DECISIONS = ("Possible duplicate", "Legitimate bill", "Needs more information")
DEFAULT_REVIEW_LIMIT = 20

# Streamlit badge colour for each recorded decision in the queue. The colours
# are set in .streamlit/config.toml and every badge carries its decision as text.
DECISION_BADGES = {
    "Possible duplicate": "orange",
    "Legitimate bill": "green",
    "Needs more information": "blue",
}

# Typographic characters, named so they are not mistaken for their keyboard look-alikes.
MIDDLE_DOT = "\u00b7"
MINUS_SIGN = "\u2212"
TIMES_SIGN = "\u00d7"

SCORE_CAPTION = "A ranking score, not a probability. The decision is yours."
NO_PAYMENTS_MESSAGE = (
    "No prior payments to compare against. The prior payments file has column names but no rows, "
    "so no incoming invoice can be checked for an earlier copy. "
    "Upload a prior payments CSV with at least one row to build a review queue."
)
BENCHMARK_EMPTY_HTML = (
    '<div class="im-empty"><p>Benchmark results are not available yet. '
    "Run <code>python evaluate.py</code> to generate them.</p></div>"
)

# Plain names for the model's inputs, shown in the "Model details" table.
FEATURE_LABELS = {
    "vendor_tfidf_cosine": "Vendor name similarity (shared letter groups)",
    "vendor_fuzzy": "Vendor name similarity (spelling)",
    "ref_similarity": "Invoice reference similarity",
    "ref_exact": "Invoice references equal",
    "ref_missing_incoming": "Reference missing on incoming invoice",
    "ref_missing_payment": "Reference missing on prior payment",
    "amount_exact": "Amounts equal",
    "amount_diff": "Amount difference",
    "date_gap": "Days between invoice dates",
    "period_same": "Same billing period",
    "period_different": "Different billing period",
    "period_missing_incoming": "Billing period missing on incoming invoice",
    "period_missing_payment": "Billing period missing on prior payment",
}

EXPORT_COLUMNS = (
    "rank", "decision", "match_score",
    "incoming_row_id", "incoming_vendor", "incoming_invoice_ref",
    "incoming_amount", "incoming_invoice_date", "incoming_billing_period",
    "payment_row_id", "payment_vendor", "payment_invoice_ref",
    "payment_amount", "payment_invoice_date", "payment_billing_period",
)
# Columns that hold text typed by whoever made the CSV. They are escaped on
# export so a spreadsheet cannot run them as a formula.
EXPORT_TEXT_COLUMNS = (
    "incoming_row_id", "incoming_vendor", "incoming_invoice_ref",
    "payment_row_id", "payment_vendor", "payment_invoice_ref",
)


# --------------------------------------------------------------------------
# The data being reviewed
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    """Two ledgers and the model's ranking of the incoming invoices.

    name      'Built-in demo' or 'Uploaded files'
    incoming  ledger frame of incoming invoices, indexed by row_id
    payments  ledger frame of prior payments, indexed by row_id
    queue     the learned ranking, indexed by incoming_id (empty without payments)
    notes     information lines from the CSV reader, such as ignored columns
    cases     the demo cases from data/demo/cases.json (empty for uploads)
    """

    name: str
    incoming: pd.DataFrame
    payments: pd.DataFrame
    queue: pd.DataFrame
    notes: list[str]
    cases: list[dict]


@st.cache_resource(show_spinner="Fitting the matching model on the synthetic training ledgers (once, a few seconds)...")
def get_matcher() -> matcher.LearnedMatcher:
    """The fitted model, shared by every session. It holds training data only, never uploads."""
    return matcher.fit_shipped_matcher()


def build_dataset(name: str, incoming: pd.DataFrame, payments: pd.DataFrame, notes: list[str],
                  cases: list[dict], model: matcher.LearnedMatcher) -> Dataset:
    """Rank the incoming invoices once and index all three frames by row ID."""
    queue = model.rank(incoming, payments)
    return Dataset(
        name=name,
        incoming=incoming.set_index("row_id", drop=False),
        payments=payments.set_index("row_id", drop=False),
        queue=queue.set_index("incoming_id", drop=False),
        notes=notes,
        cases=cases,
    )


def load_demo_dataset(model: matcher.LearnedMatcher) -> Dataset:
    """The built-in demo ledger, read through the same CSV reader as an upload."""
    incoming, incoming_notes = ledger.read_ledger_csv(DEMO_DIR / "incoming.csv", "incoming")
    payments, payment_notes = ledger.read_ledger_csv(DEMO_DIR / "payments.csv", "payments")
    cases = json.loads((DEMO_DIR / "cases.json").read_text(encoding="utf-8"))
    return build_dataset("Built-in demo", incoming, payments, incoming_notes + payment_notes, cases, model)


def queue_within_limit(dataset: Dataset, limit: int) -> pd.DataFrame:
    """The part of the ranking the treasurer is asked to review: the top `limit` rows."""
    return dataset.queue.iloc[:limit]


def case_for_invoice(dataset: Dataset, incoming_id: str | None) -> dict | None:
    """The demo case that points at this incoming invoice, if there is one."""
    for case in dataset.cases:
        if case["incoming_id"] == incoming_id:
            return case
    return None


def next_unreviewed(queue_ids: list[str], decisions: dict[str, str], current_id: str | None) -> str | None:
    """The next queue item without a decision, looking after the current one first, then from the top."""
    start = queue_ids.index(current_id) + 1 if current_id in queue_ids else 0
    for incoming_id in queue_ids[start:] + queue_ids[:start]:
        if incoming_id not in decisions and incoming_id != current_id:
            return incoming_id
    return None


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def plain_amount(cents: int) -> str:
    """46820 -> '468.20', the form a spreadsheet reads as a number."""
    cents = int(cents)
    return f"{cents // 100}.{cents % 100:02d}"


def export_fields(prefix: str, row: pd.Series) -> dict[str, str]:
    """One ledger row as export columns: 'incoming_vendor', 'incoming_amount' and so on."""
    return {
        f"{prefix}_row_id": row["row_id"],
        f"{prefix}_vendor": row["vendor"],
        f"{prefix}_invoice_ref": row["invoice_ref"],
        f"{prefix}_amount": plain_amount(row["amount_cents"]),
        f"{prefix}_invoice_date": pd.Timestamp(row["invoice_date"]).strftime("%Y-%m-%d"),
        f"{prefix}_billing_period": row["billing_period"],
    }


def build_export_frame(dataset: Dataset, decisions: dict[str, str]) -> pd.DataFrame:
    """One row per reviewed invoice, in rank order, with both records and the decision."""
    rows = []
    for entry in dataset.queue.itertuples(index=False):
        if entry.incoming_id not in decisions:
            continue
        row = {
            "rank": int(entry.rank),
            "decision": decisions[entry.incoming_id],
            "match_score": f"{matcher.match_score(entry.score):.2f}",
        }
        row.update(export_fields("incoming", dataset.incoming.loc[entry.incoming_id]))
        row.update(export_fields("payment", dataset.payments.loc[entry.payment_id]))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(EXPORT_COLUMNS))


def build_export_csv(dataset: Dataset, decisions: dict[str, str]) -> bytes:
    """The reviewed queue as CSV bytes that are safe to open in a spreadsheet."""
    return ledger.to_safe_csv(build_export_frame(dataset, decisions), EXPORT_TEXT_COLUMNS)


# --------------------------------------------------------------------------
# Escaping and HTML fragments
# --------------------------------------------------------------------------
# Streamlit widget labels are Markdown and st.html takes raw HTML, so text
# from a CSV must be made harmless before it is placed in either.

def escape_markdown(text: str) -> str:
    """Show user text literally in a widget label: every punctuation mark gets a backslash."""
    single_line = " ".join(text.split())
    return "".join("\\" + ch if ch in string.punctuation else ch for ch in single_line)


def notice_html(title: str, lines: list[str], kind: str) -> str:
    """A boxed message: a title and a list of specific lines. kind is 'error' or 'info'."""
    role = "alert" if kind == "error" else "status"
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    return (
        f'<div class="im-notice im-notice-{kind}" role="{role}">'
        f'<p class="im-notice-title">{html.escape(title)}</p>'
        f"<ul>{items}</ul></div>"
    )


def schema_html() -> str:
    """The CSV columns, one line each, from ledger.SCHEMA_DOC."""
    rows = "".join(
        f"<tr><th scope=\"row\"><code>{html.escape(column)}</code></th>"
        f"<td>{html.escape(need)}</td><td>{html.escape(meaning)}</td></tr>"
        for column, need, meaning in ledger.SCHEMA_DOC
    )
    return (
        '<table class="im-schema"><caption>Both files use the same columns. '
        "The first row holds the column names.</caption>"
        '<thead><tr><th scope="col">Column</th><th scope="col">Needed</th>'
        f'<th scope="col">What it holds</th></tr></thead><tbody>{rows}</tbody></table>'
    )


def schema_help_html() -> str:
    """The schema in one short block, for the upload dialog."""
    required = ", ".join(f"<code>{column}</code>" for column in ledger.REQUIRED_COLUMNS)
    optional = ", ".join(f"<code>{column}</code>" for column in ledger.OPTIONAL_COLUMNS)
    return (
        '<div class="im-help"><p>Both files use the same columns, named in the first row. '
        f"Required: {required}. Optional: {optional}. Dates are written YYYY-MM-DD, billing periods "
        "YYYY-MM and amounts like 468.20. Other columns are ignored.</p>"
        f"<pre>{html.escape(ledger.EXAMPLE_HEADER)}\n{html.escape(ledger.EXAMPLE_ROW)}</pre></div>"
    )


def field_html(comparison: matcher.FieldComparison, side: str) -> str:
    """One row of a card: the label (with the evidence on the payment side), the value and the note.

    Every row has the same three parts, even when the note is empty, so the
    stylesheet can line the parts up across the two cards.
    """
    if side == "incoming":
        value, note, evidence = comparison.incoming_display, comparison.incoming_note, ""
    else:
        value, note = comparison.payment_display, comparison.payment_note
        evidence = (
            f'<span class="im-evidence im-evidence-{comparison.status}">'
            f"{html.escape(comparison.evidence)}</span>"
        )
    return (
        f'<div class="im-field im-field-{comparison.field}">'
        f'<span class="im-field-head"><span class="im-label">{html.escape(comparison.label)}</span>{evidence}</span>'
        f'<span class="im-value">{html.escape(value)}</span>'
        f'<span class="im-note">{html.escape(note)}</span></div>'
    )


def cards_html(comparisons: list[matcher.FieldComparison], incoming_id: str, payment_id: str) -> str:
    """The two cards, 'Incoming invoice' and 'Prior payment', with the same rows in the same order."""
    cards = ""
    for side, title, row_id in (("incoming", "Incoming invoice", incoming_id), ("payment", "Prior payment", payment_id)):
        fields = "".join(field_html(comparison, side) for comparison in comparisons)
        cards += (
            f'<section class="im-card im-card-{side}" aria-label="{title}">'
            f'<h3 class="im-card-title">{title} <span class="im-row-id">{html.escape(row_id)}</span></h3>'
            f"{fields}</section>"
        )
    return f'<div class="im-compare-box"><div class="im-compare">{cards}</div></div>'


def panel_head_html(vendor: str, amount_cents: int, rank: int, total: int, limit: int, in_queue: bool) -> str:
    """The comparison panel's title (vendor and amount) and where the invoice sits in the ranking."""
    if in_queue:
        position = f'<p class="im-position">Ranked {rank:,} of {total:,} by match score</p>'
    else:
        position = (
            f'<p class="im-position im-position-below">Ranked {rank:,} of {total:,}, '
            f"below the review limit of {limit:,}. Shown as a demo case.</p>"
        )
    return (
        f'<div class="im-panel-head"><h2>{html.escape(vendor)} '
        f'<span class="im-panel-amount">{html.escape(ledger.format_cents(amount_cents))}</span></h2>'
        f"{position}</div>"
    )


def statements_html(title: str, statements: list[str], fallback: str) -> str:
    """A titled list of evidence statements, or one fallback sentence when the list is empty."""
    if statements:
        body = "<ul>" + "".join(f"<li>{html.escape(line)}</li>" for line in statements) + "</ul>"
    else:
        body = f"<p>{html.escape(fallback)}</p>"
    return f"<section><h3>{html.escape(title)}</h3>{body}</section>"


def reasons_html(comparisons: list[matcher.FieldComparison], in_queue: bool) -> str:
    """'Why this was flagged' (plus what to check) for a queued invoice, else 'Why this ranks lower'."""
    if in_queue:
        body = statements_html(
            "Why this was flagged", matcher.flag_reasons(comparisons),
            "No field matches this prior payment. It is the closest one on file, and the "
            "review limit reaches this far down the ranking.")
        differences = matcher.difference_notes(comparisons)
        if differences:
            body += statements_html("Check before deciding", differences, "")
    else:
        body = statements_html(
            "Why this ranks lower", matcher.difference_notes(comparisons),
            "No field differs from this prior payment. Other invoices scored higher "
            "than the review limit allows.")
    return f'<div class="im-reasons">{body}</div>'


def verdict_html(comparisons: list[matcher.FieldComparison], in_queue: bool, decision_value: float) -> str:
    """One card: the plain-word reasons on the left, the model's match score on the right."""
    score = matcher.match_score(decision_value)
    return (
        f'<div class="im-verdict-box"><div class="im-verdict">{reasons_html(comparisons, in_queue)}'
        '<div class="im-score"><h3>Match score</h3>'
        f'<p class="im-score-value">{score:.2f}</p>'
        f'<p class="im-score-caption">{html.escape(SCORE_CAPTION)}</p></div></div></div>'
    )


def feature_label(name: str) -> str:
    """A readable name for a model input; an interaction 'a*b' reads 'A x B'."""
    return f" {TIMES_SIGN} ".join(FEATURE_LABELS.get(part, part) for part in name.split("*"))


def signed(value: float) -> str:
    """+1.59 or -0.03, with a real minus sign so the columns line up."""
    return f"{value:+.2f}".replace("-", MINUS_SIGN)


def contributions_html(table: pd.DataFrame, decision_value: float) -> str:
    """The model's own arithmetic for one pair: each feature's push on the score, largest first."""
    ordered = table.reindex(table["contribution"].abs().sort_values(ascending=False, kind="stable").index)
    rows = "".join(
        f'<tr><th scope="row">{html.escape(feature_label(row.feature))}</th>'
        f"<td>{row.value:.2f}</td><td>{signed(row.standardized_value)}</td>"
        f"<td>{signed(row.coefficient)}</td><td>{signed(row.contribution)}</td></tr>"
        for row in ordered.itertuples(index=False)
    )
    footer = (
        f'<tr><th scope="row" colspan="4">Starting point (intercept)</th><td>{signed(table.attrs["intercept"])}</td></tr>'
        f'<tr><th scope="row" colspan="4">Sum (decision value, in log-odds)</th><td>{signed(decision_value)}</td></tr>'
    )
    return (
        '<div class="im-table-scroll"><table class="im-contributions">'
        '<thead><tr><th scope="col">Model input</th><th scope="col">Value</th>'
        '<th scope="col">Standardized</th><th scope="col">Coefficient</th>'
        '<th scope="col">Contribution</th></tr></thead>'
        f"<tbody>{rows}</tbody><tfoot>{footer}</tfoot></table></div>"
    )


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def init_state() -> None:
    """Create the session keys the review page relies on."""
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("decisions", {})
    st.session_state.setdefault("selected_id", None)


def activate_dataset(dataset: Dataset) -> None:
    """Make a dataset the active one. Decisions belong to a dataset, so they start empty."""
    st.session_state["dataset"] = dataset
    st.session_state["decisions"] = {}
    if dataset.cases:
        st.session_state["selected_id"] = dataset.cases[0]["incoming_id"]
    else:
        st.session_state["selected_id"] = None  # the page falls back to the top of the queue


def resolve_selection(dataset: Dataset, queue_ids: list[str]) -> str | None:
    """The invoice to show: the stored choice while it is still valid, else the top of the queue.

    A demo case stays valid below the review limit; any other invoice must be in the queue.
    """
    selected = st.session_state["selected_id"]
    if selected not in queue_ids and case_for_invoice(dataset, selected) is None:
        selected = queue_ids[0] if queue_ids else None
    st.session_state["selected_id"] = selected
    return selected


def on_queue_choice() -> None:
    """The treasurer picked a row in the queue."""
    choice = st.session_state["queue_choice"]
    if choice is not None:
        st.session_state["selected_id"] = choice


def on_demo_case() -> None:
    """The treasurer picked a demo case. This changes what is shown, never the ranking."""
    key = st.session_state["demo_case"]
    for case in st.session_state["dataset"].cases:
        if case["key"] == key:
            st.session_state["selected_id"] = case["incoming_id"]


def decision_key(decision: str) -> str:
    """The widget key of a decision button: 'Legitimate bill' -> 'decide_legitimate_bill'."""
    return "decide_" + decision.lower().replace(" ", "_")


def record_decision(incoming_id: str, decision: str) -> None:
    """Store the treasurer's decision for one incoming invoice."""
    st.session_state["decisions"][incoming_id] = decision


def undo_decision(incoming_id: str) -> None:
    """Clear the decision for one incoming invoice."""
    st.session_state["decisions"].pop(incoming_id, None)


def select_invoice(incoming_id: str) -> None:
    """Show another invoice in the comparison panel."""
    st.session_state["selected_id"] = incoming_id


# --------------------------------------------------------------------------
# Upload dialog
# --------------------------------------------------------------------------

def read_upload(uploaded_file, role: str, label: str) -> tuple[pd.DataFrame | None, list[str], str]:
    """Parse one uploaded file. Returns (frame or None, reader notes, error HTML or '')."""
    if uploaded_file is None:
        return None, [], notice_html(f"{label} is missing.", ["Choose a .csv file for it above."], "error")
    try:
        frame, notes = ledger.read_ledger_csv(uploaded_file.getvalue(), role)
    except ledger.LedgerError as error:
        return None, [], notice_html(error.title, error.problems, "error")
    return frame, notes, ""


@st.dialog("Upload CSVs", width="medium")
def upload_dialog() -> None:
    """Two uploaders, a short reminder of the schema and the 'Match invoices' button.

    Nothing replaces the active dataset until both files have been read
    without a problem.
    """
    incoming_file = st.file_uploader("Incoming invoices CSV", type="csv", key="upload_incoming")
    payments_file = st.file_uploader("Prior payments CSV", type="csv", key="upload_payments")
    st.html(schema_help_html())
    if not st.button("Match invoices", type="primary", key="match_invoices"):
        return

    incoming, incoming_notes, incoming_error = read_upload(incoming_file, "incoming", "Incoming invoices CSV")
    payments, payment_notes, payments_error = read_upload(payments_file, "payments", "Prior payments CSV")
    if incoming is None or payments is None:
        st.html(incoming_error + payments_error)
        return
    with st.spinner("Matching invoices against prior payments..."):
        dataset = build_dataset("Uploaded files", incoming, payments,
                                incoming_notes + payment_notes, [], get_matcher())
    activate_dataset(dataset)
    st.rerun()


# --------------------------------------------------------------------------
# Review queue page
# --------------------------------------------------------------------------

def review_page() -> None:
    """The 'Review queue' page: load data, pick an invoice, compare, decide, export."""
    init_state()
    model = get_matcher()

    title_area, action_area = st.columns([3, 2], vertical_alignment="center")
    # The buttons are handled before the rest of the page is drawn, so a click
    # shows the new data in the same run.
    with action_area.container(horizontal=True, horizontal_alignment="right", key="page_actions"):
        if st.button("Load demo", key="load_demo"):
            activate_dataset(load_demo_dataset(model))
        if st.button("Upload CSVs", key="open_upload", type="primary"):
            upload_dialog()

    dataset: Dataset | None = st.session_state["dataset"]
    title_area.title("Invoice review", anchor=False)
    if dataset is None:
        render_empty_state()
        return

    title_area.caption(
        f"{dataset.name}. Each incoming invoice is compared with {len(dataset.payments):,} prior payments."
    )
    if dataset.notes:
        st.html(notice_html("Notes from reading the files", dataset.notes, "info"))

    limit = render_summary(dataset)
    if len(dataset.queue) == 0:
        st.html(f'<div class="im-empty"><p>{html.escape(NO_PAYMENTS_MESSAGE)}</p></div>')
        return

    queue_ids = queue_within_limit(dataset, limit)["incoming_id"].tolist()
    selected_id = resolve_selection(dataset, queue_ids)
    if dataset.cases:
        render_demo_cases(dataset, selected_id)

    queue_area, panel_area = st.columns([4, 11], gap="medium")
    with queue_area:
        render_queue(dataset, queue_ids, selected_id)
    with panel_area:
        render_comparison(dataset, model, selected_id, queue_ids, limit)


def render_empty_state() -> None:
    """Before any data is loaded: say what to do and show the CSV columns."""
    st.html(
        '<div class="im-empty"><h2>No invoices loaded</h2>'
        "<p>Load the built-in demo, or upload two CSV files: the incoming invoices you are about "
        "to pay and the prior payments you have already made. Each incoming invoice is compared "
        "with every prior payment, and the closest matches are queued for your review.</p>"
        f"{schema_html()}</div>"
    )
    st.caption("The demo files work as templates. Download them, replace the rows with your own and upload them.")
    with st.container(horizontal=True, key="template_downloads"):
        st.download_button("Incoming invoices template (CSV)", (DEMO_DIR / "incoming.csv").read_bytes(),
                           file_name="incoming.csv", mime="text/csv", key="template_incoming", on_click="ignore")
        st.download_button("Prior payments template (CSV)", (DEMO_DIR / "payments.csv").read_bytes(),
                           file_name="payments.csv", mime="text/csv", key="template_payments", on_click="ignore")


def render_summary(dataset: Dataset) -> int:
    """The counts, the review limit and the export button. Returns the review limit."""
    decisions = st.session_state["decisions"]
    with st.container(key="summary"):
        counts_area, limit_area, export_area = st.columns([6, 2, 3], vertical_alignment="bottom")
        limit = int(limit_area.number_input(
            "Review limit", min_value=1, value=DEFAULT_REVIEW_LIMIT, step=1, key="review_limit",
            persist_state="session",
        ))
        with counts_area.container(horizontal=True, key="counts"):
            st.metric("Incoming invoices", f"{len(dataset.incoming):,}", width="content")
            st.metric("Queued for review", f"{len(queue_within_limit(dataset, limit)):,}", width="content")
            st.metric("Reviewed", f"{len(decisions):,}", width="content")
        with export_area.container(key="export_area"):
            if not decisions:
                st.caption("Record a decision to enable the export.")
            st.download_button(
                "Export reviewed CSV", build_export_csv(dataset, decisions), file_name="invoice-review.csv",
                mime="text/csv", key="export_reviewed", on_click="ignore", width="stretch",
                disabled=not decisions,
            )
    return limit


def render_demo_cases(dataset: Dataset, selected_id: str | None) -> None:
    """Three prepared invoices from the demo ledger, one click away."""
    titles = {case["key"]: case["title"] for case in dataset.cases}
    shown_case = case_for_invoice(dataset, selected_id)
    # The control mirrors what is on screen, so it clears when another invoice is picked.
    st.session_state["demo_case"] = shown_case["key"] if shown_case else None
    with st.container(key="demo_cases"):
        st.segmented_control("Demo cases", list(titles), format_func=titles.get, key="demo_case",
                             on_change=on_demo_case)
        if shown_case:
            st.caption(shown_case["summary"])


def queue_row_label(row: pd.Series) -> str:
    """Vendor and amount for one queue row (Markdown; the amount is bold so the stylesheet can place it)."""
    return f"{escape_markdown(row['vendor'])} **{escape_markdown(ledger.format_cents(row['amount_cents']))}**"


def queue_row_caption(row: pd.Series, status: str) -> str:
    """Invoice reference and review status for one queue row."""
    reference = escape_markdown(row["invoice_ref"]) or "No reference"
    if status == NOT_REVIEWED:
        return f"{reference} {MIDDLE_DOT} {NOT_REVIEWED}"
    return f"{reference} :{DECISION_BADGES[status]}-badge[{status}]"


def render_queue(dataset: Dataset, queue_ids: list[str], selected_id: str) -> None:
    """The queue as a radio group: one row per incoming invoice, in the model's order."""
    decisions = st.session_state["decisions"]
    rows = dataset.incoming.loc[queue_ids]
    captions = [
        queue_row_caption(row, decisions.get(incoming_id, NOT_REVIEWED))
        for incoming_id, row in rows.iterrows()
    ]
    # The radio shows no selection while a demo case below the review limit is on screen.
    st.session_state["queue_choice"] = selected_id if selected_id in queue_ids else None
    with st.container(key="queue_column"):
        st.html(
            '<div class="im-queue-head"><h2>Review queue</h2>'
            f"<p>Top {len(queue_ids):,} of {len(dataset.queue):,}</p></div>"
        )
        with st.container(key="queue_list"):
            st.radio(
                "Review queue", queue_ids, index=None, key="queue_choice", on_change=on_queue_choice,
                format_func=lambda incoming_id: queue_row_label(rows.loc[incoming_id]),
                captions=captions, label_visibility="collapsed", width="stretch",
            )


def render_comparison(dataset: Dataset, model: matcher.LearnedMatcher, selected_id: str,
                      queue_ids: list[str], limit: int) -> None:
    """The comparison panel for one incoming invoice and its proposed prior payment."""
    entry = dataset.queue.loc[selected_id]
    incoming_row = dataset.incoming.loc[selected_id]
    payment_row = dataset.payments.loc[entry["payment_id"]]
    comparisons = matcher.compare_fields(incoming_row, payment_row)
    in_queue = selected_id in queue_ids

    st.html(panel_head_html(incoming_row["vendor"], int(incoming_row["amount_cents"]),
                            int(entry["rank"]), len(dataset.queue), limit, in_queue))
    st.html(cards_html(comparisons, selected_id, entry["payment_id"]))

    st.html(verdict_html(comparisons, in_queue, float(entry["score"])))
    render_actions(selected_id, queue_ids)
    render_model_details(model, incoming_row, payment_row, float(entry["score"]))


def render_actions(selected_id: str, queue_ids: list[str]) -> None:
    """The three decisions, Undo and Next unreviewed for the invoice on screen."""
    decisions = st.session_state["decisions"]
    decision = decisions.get(selected_id)
    recorded = f"Recorded: {decision}" if decision else "No decision recorded yet."
    upcoming = next_unreviewed(queue_ids, decisions, selected_id)
    with st.container(key="decision_card"):
        st.html(
            '<div class="im-decision-head"><h3>Your decision</h3>'
            f'<p role="status">{html.escape(recorded)}</p></div>'
        )
        with st.container(horizontal=True, key="decision_actions"):
            for label in DECISIONS:
                chosen = label == decision
                st.button(label, key=decision_key(label), type="primary" if chosen else "secondary",
                          icon=":material/check:" if chosen else None,
                          on_click=record_decision, args=(selected_id, label))
            st.button("Undo", key="undo", type="tertiary", disabled=decision is None,
                      on_click=undo_decision, args=(selected_id,))
            st.button("Next unreviewed", key="next_unreviewed", disabled=upcoming is None,
                      on_click=select_invoice, args=(upcoming,))


def render_model_details(model: matcher.LearnedMatcher, incoming_row: pd.Series, payment_row: pd.Series,
                         decision_value: float) -> None:
    """An expander with the model's own arithmetic, kept apart from the field comparisons."""
    with st.expander("Model details"):
        st.caption(
            "Model contributions in log-odds: coefficient times standardized feature value. "
            "A positive number pushed this pair up the ranking, a negative number pushed it down. "
            "This is the model's arithmetic; the field comparisons above do not depend on it."
        )
        st.html(contributions_html(model.contributions(incoming_row, payment_row), decision_value))
        settings = "frozen after validation" if model.frozen else "defaults, not yet frozen on validation"
        st.caption(
            f"Logistic regression fitted on {model.n_train_pairs:,} synthetic invoice pairs "
            f"({model.n_train_positives:,} of them true duplicates). Settings: {settings}."
        )


# --------------------------------------------------------------------------
# Benchmark page and entry point
# --------------------------------------------------------------------------

def benchmark_page() -> None:
    """The 'Benchmark' page. evaluate.py has not written results yet, so this is an empty state."""
    st.title("Benchmark", anchor=False)
    st.html(BENCHMARK_EMPTY_HTML)


def main() -> None:
    """Page setup, the stylesheet (loaded once) and the top navigation."""
    st.set_page_config(page_title="Invoice Match", page_icon=":material/receipt_long:", layout="wide")
    st.html(CSS_PATH)
    pages = [
        st.Page(review_page, title="Review queue", default=True),
        st.Page(benchmark_page, title="Benchmark", url_path="benchmark"),
    ]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()
