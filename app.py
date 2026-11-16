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
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import ledger
import matcher

APP_DIR = Path(__file__).resolve().parent
CSS_PATH = APP_DIR / "assets" / "app.css"
DEMO_DIR = matcher.DATA_DIR / "demo"
BENCHMARK_PATH = matcher.RESULTS_DIR / "benchmark.json"

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
# The score is shown and exported inside this range, so two decimals never read as 0.00 or 1.00.
SCORE_DISPLAY_MIN = 0.01
SCORE_DISPLAY_MAX = 0.99
WEAK_MATCH_MESSAGE = (
    "No similar prior payment was found for this invoice. The closest one the model could find is shown."
)
NO_PAYMENTS_MESSAGE = (
    "No prior payments to compare against. Without payment history no incoming invoice can be "
    "checked for an earlier copy. Upload a prior payments CSV with at least one row to build a review queue."
)
BENCHMARK_EMPTY_HTML = (
    '<div class="im-empty"><p>Benchmark results are not available yet. '
    "Run <code>python evaluate.py --stage test</code> to generate them.</p></div>"
)

# The columns of a benchmark results table: the key in results/benchmark.json and the heading shown.
RESULT_COLUMNS = (
    ("reviews_used", "Reviews used"),
    ("correct", "Correct matches"),
    ("false_flags", "False flags"),
    ("missed", "Duplicates missed"),
    ("precision", "Precision"),
    ("recall", "Recall"),
)
# Headings for the vendor groups in the "Who gets held for review?" table.
VENDOR_GROUP_LABELS = {"organization": "Organizations", "individual": "Individuals"}
HELD_FOR_REVIEW_NOTE = (
    "Individuals are often freelancers who bill the same fee repeatedly and leave out invoice numbers; "
    "a tool that flags them more often delays their pay."
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
    "period_same": "Same billing period",
    "period_different": "Different billing period",
    "period_missing_incoming": "Billing period missing on incoming invoice",
    "period_missing_payment": "Billing period missing on prior payment",
}

# The amount and date inputs are measured on the scale chosen in the model's config.
AMOUNT_SCALE_LABELS = {
    "log1p_cents": "Amount difference (log of cents apart)",
    "relative": "Amount difference (share of the larger amount)",
}
DATE_SCALE_LABELS = {
    "log1p_days": "Invoice date gap (log of days apart)",
    "capped_90": "Invoice date gap (days apart, up to 90)",
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


MODEL_SPINNER = "Fitting the matching model on the synthetic training ledgers (once, a few seconds)..."


@st.cache_resource(show_spinner=MODEL_SPINNER)
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

def display_score(decision_value: float) -> float:
    """The match score as it is shown and exported: kept between 0.01 and 0.99.

    A strong pair scores so close to 1 that two decimals would print 1.00,
    which reads like certainty. Only the printed number is limited; the
    ranking uses the model's decision values and is not touched.
    """
    score = matcher.match_score(decision_value)
    return min(max(score, SCORE_DISPLAY_MIN), SCORE_DISPLAY_MAX)


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
            "match_score": f"{display_score(entry.score):.2f}",
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
# from a CSV is made literal (literal_markdown, html.escape) before it is
# placed in either.

def literal_markdown(text: str) -> str:
    """Show text literally in a widget label by wrapping it in a Markdown code span.

    Inside a code span nothing is markup: no bold, no links, no colour
    directives, and a web address is not turned into a link. The span is closed
    by a run of backticks one longer than any run inside the text, so the text
    cannot end it early. The stylesheet makes the span look like ordinary text.
    """
    single_line = " ".join(text.split())
    longest_run = max((len(run) for run in re.findall("`+", single_line)), default=0)
    fence = "`" * (longest_run + 1)
    return f"{fence} {single_line} {fence}"


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


def cards_html(comparisons: list[matcher.FieldComparison], incoming_id: str, payment_id: str,
               payment_title: str = "Prior payment") -> str:
    """The two cards, 'Incoming invoice' and 'Prior payment', with the same rows in the same order.

    The benchmark page passes its own title for the payment card.
    """
    cards = ""
    for side, title, row_id in (("incoming", "Incoming invoice", incoming_id), ("payment", payment_title, payment_id)):
        fields = "".join(field_html(comparison, side) for comparison in comparisons)
        cards += (
            f'<section class="im-card im-card-{side}" aria-label="{html.escape(title)}">'
            f'<h3 class="im-card-title">{html.escape(title)} '
            f'<span class="im-row-id">{html.escape(row_id)}</span></h3>'
            f"{fields}</section>"
        )
    return f'<div class="im-compare-box"><div class="im-compare">{cards}</div></div>'


def vendor_differs(comparisons: list[matcher.FieldComparison]) -> bool:
    """True when the vendor names of the shown pair were compared as different."""
    return any(comparison.field == "vendor" and comparison.status == "differ" for comparison in comparisons)


def weak_match_html() -> str:
    """One neutral line for a pair whose vendors differ: the closest payment is not a similar one."""
    return f'<p class="im-weak-match" role="note">{html.escape(WEAK_MATCH_MESSAGE)}</p>'


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
    score = display_score(decision_value)
    return (
        f'<div class="im-verdict-box"><div class="im-verdict">{reasons_html(comparisons, in_queue)}'
        '<div class="im-score"><h3>Match score</h3>'
        f'<p class="im-score-value">{score:.2f}</p>'
        f'<p class="im-score-caption">{html.escape(SCORE_CAPTION)}</p></div></div></div>'
    )


def feature_labels(config: matcher.MatcherConfig) -> dict[str, str]:
    """Plain names for the model's inputs, with the amount and date scales this config uses."""
    labels = dict(FEATURE_LABELS)
    labels["amount_diff"] = AMOUNT_SCALE_LABELS[config.amount_transform]
    labels["date_gap"] = DATE_SCALE_LABELS[config.date_transform]
    return labels


def feature_label(name: str, labels: dict[str, str]) -> str:
    """A readable name for a model input; an interaction 'a*b' reads 'A x B'."""
    return f" {TIMES_SIGN} ".join(labels.get(part, part) for part in name.split("*"))


def signed(value: float) -> str:
    """+1.59 or -0.03, with a real minus sign so the columns line up."""
    return f"{value:+.2f}".replace("-", MINUS_SIGN)


def contributions_html(table: pd.DataFrame, decision_value: float, labels: dict[str, str]) -> str:
    """The model's own arithmetic for one pair: each feature's push on the score, largest first."""
    ordered = table.reindex(table["contribution"].abs().sort_values(ascending=False, kind="stable").index)
    rows = "".join(
        f'<tr><th scope="row">{html.escape(feature_label(row.feature, labels))}</th>'
        f"<td>{row.value:.2f}</td><td>{signed(row.standardized_value)}</td>"
        f"<td>{signed(row.coefficient)}</td><td>{signed(row.contribution)}</td></tr>"
        for row in ordered.itertuples(index=False)
    )
    empty_cells = "<td></td>" * 3  # the footer rows keep the five columns of the rows above
    footer = (
        f'<tr><th scope="row">Starting point (intercept)</th>{empty_cells}'
        f'<td>{signed(table.attrs["intercept"])}</td></tr>'
        f'<tr><th scope="row">Sum (decision value, in log-odds)</th>{empty_cells}'
        f"<td>{signed(decision_value)}</td></tr>"
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


def resolve_selection(dataset: Dataset, queue_ids: list[str]) -> str:
    """The invoice to show: the stored choice while it is still valid, else the top of the queue.

    A demo case stays valid below the review limit; any other invoice must be
    in the queue. queue_ids is never empty here.
    """
    selected = st.session_state["selected_id"]
    if selected not in queue_ids and case_for_invoice(dataset, selected) is None:
        selected = queue_ids[0]
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

def read_upload(uploaded_file: UploadedFile | None, role: str,
                label: str) -> tuple[pd.DataFrame | None, list[str], str]:
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
    try:
        model = get_matcher()
    except FileNotFoundError as error:  # the training data has not been generated
        st.html(notice_html("The matching model could not be fitted.", [str(error)], "error"))
        return

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

    if len(dataset.payments) > 0:
        title_area.caption(
            f"{dataset.name}. Each incoming invoice is compared with {len(dataset.payments):,} prior payments."
        )
    else:
        title_area.caption(f"{dataset.name}. The prior payments file has no rows.")
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


def render_demo_cases(dataset: Dataset, selected_id: str) -> None:
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
    """Vendor and amount for one queue row, as two literal spans that the stylesheet places left and right."""
    return f"{literal_markdown(row['vendor'])} {literal_markdown(ledger.format_cents(row['amount_cents']))}"


def queue_row_caption(row: pd.Series, status: str) -> str:
    """Invoice reference and review status for one queue row."""
    reference = literal_markdown(row["invoice_ref"]) if row["invoice_ref"] else "No reference"
    if status == NOT_REVIEWED:
        return f"{reference} {MIDDLE_DOT} {NOT_REVIEWED}"
    return f"{reference} :{DECISION_BADGES[status]}-badge[{status}]"


def render_queue(dataset: Dataset, queue_ids: list[str], selected_id: str) -> None:
    """The queue as a radio group: one row per incoming invoice, in the model's order."""
    decisions = st.session_state["decisions"]
    rows = dataset.incoming.loc[queue_ids]
    labels = {incoming_id: queue_row_label(row) for incoming_id, row in rows.iterrows()}
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
                format_func=labels.get, captions=captions, label_visibility="collapsed", width="stretch",
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
    if vendor_differs(comparisons):  # a note only; the ranking is left as the model made it
        st.html(weak_match_html())
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
        st.html(contributions_html(model.contributions(incoming_row, payment_row), decision_value,
                                   feature_labels(model.config)))
        settings = "frozen after validation" if model.frozen else "defaults, not yet frozen on validation"
        st.caption(
            f"Logistic regression fitted on {model.n_train_pairs:,} synthetic invoice pairs "
            f"({model.n_train_positives:,} of them true duplicates). Settings: {settings}."
        )


# --------------------------------------------------------------------------
# Benchmark page: HTML fragments
# --------------------------------------------------------------------------
# Every number on this page is read from results/benchmark.json, the file that
# evaluate.py writes. Nothing is measured or estimated here.

def load_benchmark(path: Path) -> dict | None:
    """The measured results, or None when evaluate.py has not written the file yet."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def counted(number: int, singular: str, plural: str) -> str:
    """'1 ledger' or '1,475 prior payments'."""
    return f"{number:,} {singular if number == 1 else plural}"


def percent(value: float | None) -> str:
    """0.65 -> '65%'. An em dash when there is no value (precision when nothing was flagged)."""
    if value is None:
        return matcher.BLANK_DISPLAY
    return f"{value:.0%}"


def share(part: int, whole: int) -> str:
    """'3 of 247 (1.2%)', or only the two counts when the whole is zero."""
    if whole == 0:
        return f"{part:,} of {whole:,}"
    return f"{part:,} of {whole:,} ({part / whole:.1%})"


def dataset_sentence(dataset: dict) -> str:
    """The size of a benchmark set in one sentence, from its 'dataset' entry."""
    return (
        f"{counted(dataset['incoming'], 'incoming invoice', 'incoming invoices')} with "
        f"{counted(dataset['known_duplicates'], 'known duplicate', 'known duplicates')}, checked against "
        f"{counted(dataset['prior_payments'], 'prior payment', 'prior payments')} in "
        f"{counted(dataset['ledgers'], 'ledger', 'ledgers')}, dated "
        f"{ledger.format_date(dataset['date_start'])} to {ledger.format_date(dataset['date_end'])}."
    )


def result_text(key: str, result: dict) -> str:
    """One cell of a results table: a whole percentage for precision and recall, else a count."""
    if key in ("precision", "recall"):
        return percent(result[key])
    return f"{result[key]:,}"


def results_table_html(methods: list[dict], budget: int, caption: str) -> str:
    """One row per method with its measured results at this review budget.

    Each cell repeats its column heading in data-label. On a narrow screen the
    stylesheet hides the heading row and shows that label beside the number.
    """
    headings = "".join(f'<th scope="col">{html.escape(label)}</th>' for _, label in RESULT_COLUMNS)
    rows = ""
    for method in methods:
        result = method["results"][str(budget)]
        cells = "".join(
            f'<td data-label="{html.escape(label)}">{html.escape(result_text(key, result))}</td>'
            for key, label in RESULT_COLUMNS
        )
        rows += f'<tr><th scope="row">{html.escape(method["name"])}</th>{cells}</tr>'
    return (
        f'<div class="im-bench-box"><table class="im-bench-table"><caption>{html.escape(caption)}</caption>'
        f'<thead><tr><th scope="col">Method</th>{headings}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def definitions_html(methods: list[dict]) -> str:
    """Each method's one-sentence definition, as written in the results file."""
    items = "".join(
        f"<dt>{html.escape(method['name'])}</dt><dd>{html.escape(method['definition'])}</dd>"
        for method in methods
    )
    return f'<dl class="im-definitions">{items}</dl>'


def bars_html(methods: list[dict], budget: int, known_duplicates: int) -> str:
    """Correct matches per method as horizontal bars on one scale, from zero to all known duplicates.

    Every bar is followed by its value as text ('13 of 20'), so the chart can
    be read without the bars and without their colours.
    """
    rows = ""
    for method in methods:
        correct = method["results"][str(budget)]["correct"]
        width = 100 * correct / known_duplicates if known_duplicates else 0
        kind = "learned" if method["key"] == "learned" else "baseline"
        rows += (
            f'<li><span class="im-bar-name">{html.escape(method["name"])}</span>'
            f'<span class="im-bar-track"><span class="im-bar im-bar-{kind}" style="width: {width:.1f}%" '
            f'aria-hidden="true"></span><span class="im-bar-value">{correct:,} of {known_duplicates:,}</span>'
            "</span></li>"
        )
    caption = (
        f"Correct matches in the first {budget:,} reviews. Every bar starts at zero, and the full width "
        f"stands for all {counted(known_duplicates, 'known duplicate', 'known duplicates')}."
    )
    return f'<figure class="im-bars"><figcaption>{html.escape(caption)}</figcaption><ul>{rows}</ul></figure>'


def record_display(record: dict) -> dict[str, str]:
    """The five fields of a benchmark record as shown on a card; a blank becomes an em dash."""
    return {
        "vendor": record["vendor"] or matcher.BLANK_DISPLAY,
        "invoice_ref": record["invoice_ref"] or matcher.BLANK_DISPLAY,
        "amount": ledger.format_cents(record["amount_cents"]),
        "invoice_date": ledger.format_date(record["invoice_date"]),
        "billing_period": ledger.format_period(record["billing_period"]) or matcher.BLANK_DISPLAY,
    }


def compared_payment(example: dict) -> dict:
    """The prior payment an example is compared with.

    A missed duplicate is set against the payment it really repeats; a flagged
    invoice is set against the payment the model proposed.
    """
    if example["kind"] == "missed_duplicate" and example["true_payment"] is not None:
        return example["true_payment"]
    return example["proposed_payment"]


def example_comparisons(example: dict) -> list[matcher.FieldComparison]:
    """An example's stored comparisons in the form the review page's cards take."""
    incoming_shown = record_display(example["incoming"])
    payment_shown = record_display(compared_payment(example))
    return [
        matcher.FieldComparison(item["field"], item["label"], item["status"], item["evidence"],
                                incoming_shown[item["field"]], payment_shown[item["field"]], "", "")
        for item in example["comparisons"]
    ]


def example_heading(example: dict) -> str:
    """One line that says what an example is, with its rank in the frozen ranking."""
    rank, budget = example["rank"], example["budget"]
    if example["kind"] == "missed_duplicate":
        return f"A known duplicate ranked {rank:,}, below the review budget of {budget:,}"
    if example["kind"] == "false_flag":
        return f"A legitimate invoice that was flagged for review, ranked {rank:,} of {budget:,}"
    return f"A known duplicate the learned model ranked {rank:,}, inside the review budget of {budget:,}"


def cause_html(example: dict) -> str:
    """The 'likely cause' sentence that evaluate.py wrote for an example."""
    return f'<p class="im-cause"><strong>Likely cause.</strong> {html.escape(example["likely_cause"])}</p>'


def example_html(example: dict) -> str:
    """A failure example: heading, the two records side by side with their evidence, the likely cause."""
    payment = compared_payment(example)
    if example["kind"] == "missed_duplicate":
        payment_title = "Prior payment it repeats"
    else:
        payment_title = "Proposed prior payment"
    cards = cards_html(example_comparisons(example), example["incoming"]["row_id"], payment["row_id"], payment_title)
    return (
        f'<article class="im-example"><h3>{html.escape(example_heading(example))}</h3>'
        f"{cards}{cause_html(example)}</article>"
    )


def record_line_html(title: str, record: dict) -> str:
    """One record on one line: vendor, reference, amount, invoice date and billing period."""
    values = "".join(f"<span>{html.escape(value)}</span>" for value in record_display(record).values())
    return f'<p class="im-record-line"><span class="im-label">{html.escape(title)}</span>{values}</p>'


def brief_example_html(example: dict) -> str:
    """A shorter example: each record on one line, the evidence labels, the likely cause."""
    labels = "".join(
        f'<li class="im-evidence im-evidence-{html.escape(item["status"])}">{html.escape(item["evidence"])}</li>'
        for item in example["comparisons"]
    )
    return (
        f'<article class="im-example im-example-brief"><h3>{html.escape(example_heading(example))}</h3>'
        f'<div class="im-brief">{record_line_html("Incoming invoice", example["incoming"])}'
        f'{record_line_html("Prior payment", compared_payment(example))}'
        f'<ul class="im-evidence-list">{labels}</ul></div>{cause_html(example)}</article>'
    )


def examples_of(data: dict, kind: str) -> list[dict]:
    """The examples of one kind ('missed_duplicate', 'false_flag' or 'caught_duplicate'); may be empty."""
    return [example for example in data.get("examples", []) if example["kind"] == kind]


def section_html(title: str, body: str, intro: str = "") -> str:
    """A titled section of the benchmark page. body is HTML written by the functions above."""
    lead = f'<p class="im-section-intro">{html.escape(intro)}</p>' if intro else ""
    return f'<section class="im-section"><h2>{html.escape(title)}</h2>{lead}{body}</section>'


def slices_table_html(groups: list[dict], methods: list[dict]) -> str:
    """Legitimate invoices that were flagged, per method (rows) and per kind of vendor (columns)."""
    labels = [VENDOR_GROUP_LABELS.get(group["group"], group["group"].capitalize()) for group in groups]
    headings = "".join(f'<th scope="col">{html.escape(label)}</th>' for label in labels)
    rows = ""
    for method in methods:
        cells = ""
        for group, label in zip(groups, labels):
            flagged = group["methods"][method["key"]]["legit_flagged"]
            cells += f'<td data-label="{html.escape(label)}">{html.escape(share(flagged, group["legit_invoices"]))}</td>'
        rows += f'<tr><th scope="row">{html.escape(method["name"])}</th>{cells}</tr>'
    return (
        '<div class="im-bench-box"><table class="im-bench-table">'
        "<caption>Legitimate invoices flagged for review, out of all legitimate invoices from that kind of vendor</caption>"
        f'<thead><tr><th scope="col">Method</th>{headings}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def validation_sentence(data: dict) -> str:
    """The validation results that the settings were tuned on, one count per method."""
    names = {method["key"]: method["name"] for method in data["methods"]}
    counts = []
    for key, result in data["validation_summary"].items():
        known = result["correct"] + result["missed"]
        counts.append(f"{names.get(key, key)} {result['correct']:,} of {known:,}")
    return (
        "Validation numbers used for tuning (correct matches out of known duplicates, summed over the "
        f"validation folds): {', '.join(counts)}."
    )


def how_to_read_html(data: dict) -> str:
    """What the numbers on the page can and cannot say, and how to produce them again."""
    lines = [
        "Results are measured on synthetic data. They are not a measure of real-world accuracy. "
        "This project is a prototype and is not financial advice.",
        "The test set was scored once, after the settings were frozen on validation.",
        validation_sentence(data),
    ]
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    command = f'<li>To reproduce these results, run <code>{html.escape(data["command"])}</code></li>'
    return f'<ul class="im-reading">{items}{command}</ul>'


# --------------------------------------------------------------------------
# Benchmark page and entry point
# --------------------------------------------------------------------------

def benchmark_page() -> None:
    """The 'Benchmark' page: what the three methods caught on the held-out test ledgers."""
    data = load_benchmark(BENCHMARK_PATH)
    if data is None:
        st.title("Benchmark", anchor=False)
        st.html(BENCHMARK_EMPTY_HTML)
        return

    head = st.container(key="benchmark_head")  # filled below, once the review budget is known
    budget = render_budget_selector(data)
    methods, dataset = data["methods"], data["dataset"]
    with head:
        st.html(f'<p class="im-eyebrow">{html.escape(data["label"])}</p>')
        st.title(f"How many duplicates can we catch in {budget:,} reviews?", anchor=False)
        st.html(
            f'<p class="im-lead">Held-out test set: {html.escape(dataset_sentence(dataset))} '
            "These vendors and dates were never seen in training.</p>"
        )

    st.html(
        results_table_html(methods, budget, f"Results at a review budget of {budget:,}")
        + definitions_html(methods)
    )
    st.html(bars_html(methods, budget, dataset["known_duplicates"]))
    render_examples(data)
    render_slices(data)
    render_challenge(data)
    st.html(section_html("How to read this", how_to_read_html(data)))


def render_budget_selector(data: dict) -> int:
    """The 'Review budget' control. Returns the chosen number of reviews."""
    budgets = data["budgets"]
    with st.container(key="benchmark_budget_area"):
        budget = st.segmented_control("Review budget", budgets, default=data["default_budget"],
                                      required=True, key="benchmark_budget")
        choices = ", ".join(f"{value:,}" for value in budgets[:-1]) + f" or {budgets[-1]:,}"
        st.caption(
            "Every budget is a slice of the same frozen ranking: "
            f"the first {choices} invoices of one list, reviewed from the top."
        )
    return int(budget) if budget is not None else int(data["default_budget"])


def render_examples(data: dict) -> None:
    """'Where the learned model fails' and 'What exact match misses', from the stored examples."""
    failures = examples_of(data, "missed_duplicate") + examples_of(data, "false_flag")
    if failures:
        st.html(section_html(
            "Where the learned model fails", "".join(example_html(example) for example in failures),
            "One example of each kind of mistake at the largest budget, with the same field comparisons "
            "the review page shows."))
    caught = examples_of(data, "caught_duplicate")
    if caught:
        st.html(section_html("What exact match misses",
                             "".join(brief_example_html(example) for example in caught)))


def render_slices(data: dict) -> None:
    """'Who gets held for review?': flagged legitimate invoices for organizations and for individuals."""
    groups = data.get("slices", {}).get("vendor_kind", [])
    if not groups:
        return
    note = f'<p class="im-section-note">{html.escape(HELD_FOR_REVIEW_NOTE)}</p>'
    st.html(section_html(
        "Who gets held for review?", slices_table_html(groups, data["methods"]) + note,
        f"Measured at {data['default_budget']:,} reviews."))


def render_challenge(data: dict) -> None:
    """The hand-authored challenge set: its size, its own budget and the same table as above."""
    challenge = data.get("challenge")
    if not challenge:
        return
    budget = challenge["budget"]
    table = results_table_html(challenge["methods"], budget, f"Challenge set results at a review budget of {budget:,}")
    note = f'<p class="im-section-note">Note from the results file: {html.escape(challenge["note"])}.</p>'
    st.html(section_html(
        "Challenge set (hand-authored, reported separately)", table + note,
        dataset_sentence(challenge["dataset"])))


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
