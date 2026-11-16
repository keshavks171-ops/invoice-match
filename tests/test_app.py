"""Tests for app.py: the review and benchmark pages driven by Streamlit's AppTest
(no browser), and the pure helper functions (export, escaping, HTML fragments)
called directly.

The fixtures in tests/fixtures are small hand-written ledgers with fictional
vendors. They include a vendor that starts with '=' and one that starts with
'+', a reference with leading zeros and blank optional fields.
"""

import html
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
import ledger
import matcher

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUN_TIMEOUT = 120  # seconds; the first run fits the model on the training ledgers

DEMO_DUPLICATE = "INC-0015"  # the disguised duplicate in data/demo/cases.json
DEMO_RECURRING = "INC-0028"  # the legitimate recurring bill


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def start_app() -> AppTest:
    """Run the app once, as a browser would on first load."""
    return AppTest.from_file(APP_PATH, default_timeout=RUN_TIMEOUT).run()


def start_demo() -> AppTest:
    """Run the app and click 'Load demo'."""
    at = start_app()
    at.button(key="load_demo").click().run()
    return at


def metric_value(at: AppTest, label: str) -> str:
    """The value shown by the st.metric with this label."""
    values = [metric.value for metric in at.metric if metric.label == label]
    assert len(values) == 1, f"expected one metric labelled {label!r}, found {len(values)}"
    return values[0]


def page_html(at: AppTest) -> str:
    """All HTML the app wrote itself, without the stylesheet."""
    return "\n".join(block.value for block in at.html if not block.value.startswith("<style>"))


def upload_through_dialog(at: AppTest, incoming_name: str | None, payments_name: str | None) -> AppTest:
    """Open the upload dialog, choose the fixture files and press 'Match invoices'.

    AppTest reruns the whole script (a browser would rerun only the dialog), so
    the button that opens the dialog is clicked again in the same run to keep
    the dialog on screen while 'Match invoices' is handled.
    """
    at.button(key="open_upload").click().run()
    for key, name in (("upload_incoming", incoming_name), ("upload_payments", payments_name)):
        if name is not None:
            at.file_uploader(key=key).upload(name, (FIXTURES / name).read_bytes(), "text/csv")
    at.button(key="match_invoices").click()
    at.button(key="open_upload").click()
    return at.run()


@pytest.fixture(scope="module")
def model() -> matcher.LearnedMatcher:
    """The shipped model, fitted once for the tests that call app functions directly."""
    return matcher.fit_shipped_matcher()


@pytest.fixture(scope="module")
def fixture_dataset(model: matcher.LearnedMatcher) -> app.Dataset:
    """The valid fixture ledgers, ranked by the model."""
    incoming, incoming_notes = ledger.read_ledger_csv(FIXTURES / "valid_incoming.csv", "incoming")
    payments, payment_notes = ledger.read_ledger_csv(FIXTURES / "valid_payments.csv", "payments")
    return app.build_dataset("Uploaded files", incoming, payments, incoming_notes + payment_notes, [], model)


def incoming_id_of(dataset: app.Dataset, vendor_prefix: str) -> str:
    """The row ID of the fixture's incoming invoice whose vendor starts with this text."""
    matches = dataset.incoming[dataset.incoming["vendor"].str.startswith(vendor_prefix)]
    assert len(matches) == 1
    return matches["row_id"].iloc[0]


# --------------------------------------------------------------------------
# The page, before and after loading the demo
# --------------------------------------------------------------------------

def test_app_starts_without_exception():
    at = start_app()
    assert not at.exception
    assert [title.value for title in at.title] == ["Invoice review"]
    assert {button.label for button in at.button} == {"Load demo", "Upload CSVs"}


def test_empty_state_says_what_to_do_and_shows_the_columns():
    at = start_app()
    text = page_html(at)
    assert "No invoices loaded" in text
    assert "Load the built-in demo, or upload two CSV files" in text
    for column in ledger.SCHEMA_COLUMNS:
        assert f"<code>{column}</code>" in text
    assert len(at.metric) == 0 and len(at.radio) == 0
    template_keys = {button.key for button in at.download_button}
    assert template_keys == {"template_incoming", "template_payments"}


def test_load_demo_fills_the_queue_and_selects_the_disguised_duplicate():
    at = start_demo()
    assert not at.exception
    queue = at.radio(key="queue_choice")
    assert len(queue.options) == app.DEFAULT_REVIEW_LIMIT
    assert queue.value == DEMO_DUPLICATE
    assert at.session_state["selected_id"] == DEMO_DUPLICATE
    assert at.segmented_control(key="demo_case").value == "disguised_duplicate"
    assert metric_value(at, "Incoming invoices") == "32"
    assert metric_value(at, "Queued for review") == "20"
    assert metric_value(at, "Reviewed") == "0"


def test_demo_opens_on_the_two_matching_invoices_with_plain_evidence():
    text = page_html(start_demo())
    assert text.count('<span class="im-value">$468.20</span>') == 2  # one per card
    assert "Same amount" in text
    assert "References match after normalization" in text
    assert "Normalized: 1042" in text
    assert "Why this was flagged" in text
    assert "Match score" in text and app.SCORE_CAPTION in text


def test_queue_follows_the_learned_ranking(model: matcher.LearnedMatcher):
    at = start_demo()
    incoming, _ = ledger.read_ledger_csv(app.DEMO_DIR / "incoming.csv", "incoming")
    payments, _ = ledger.read_ledger_csv(app.DEMO_DIR / "payments.csv", "payments")
    expected = model.rank(incoming, payments)["incoming_id"].tolist()[:app.DEFAULT_REVIEW_LIMIT]
    assert at.session_state["dataset"].queue["incoming_id"].tolist()[:app.DEFAULT_REVIEW_LIMIT] == expected
    assert len(at.radio(key="queue_choice").options) == len(expected)
    assert at.session_state["dataset"].queue["incoming_id"].is_unique  # one proposal per incoming invoice


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

def test_recording_a_decision_updates_the_reviewed_count_and_undo_clears_it():
    at = start_demo()
    assert at.button(key="undo").disabled
    assert at.download_button(key="export_reviewed").disabled

    at.button(key="decide_possible_duplicate").click().run()
    assert metric_value(at, "Reviewed") == "1"
    assert at.session_state["decisions"] == {DEMO_DUPLICATE: "Possible duplicate"}
    assert "Recorded: Possible duplicate" in page_html(at)
    assert not at.download_button(key="export_reviewed").disabled

    at.button(key="undo").click().run()
    assert metric_value(at, "Reviewed") == "0"
    assert at.session_state["decisions"] == {}
    assert "No decision recorded yet." in page_html(at)
    assert at.download_button(key="export_reviewed").disabled


def test_a_decision_can_be_changed_and_shows_in_the_queue_row():
    at = start_demo()
    at.button(key="decide_possible_duplicate").click().run()
    at.button(key="decide_needs_more_information").click().run()
    assert at.session_state["decisions"] == {DEMO_DUPLICATE: "Needs more information"}
    assert metric_value(at, "Reviewed") == "1"
    queue = at.radio(key="queue_choice")
    captions = list(queue.proto.captions)
    assert "Needs more information" in captions[0]
    assert all(app.NOT_REVIEWED in caption for caption in captions[1:])


def test_next_unreviewed_moves_to_the_next_item_without_a_decision():
    at = start_demo()
    queue_ids = at.session_state["dataset"].queue["incoming_id"].tolist()
    at.button(key="decide_legitimate_bill").click().run()
    at.button(key="next_unreviewed").click().run()
    assert at.session_state["selected_id"] == queue_ids[1]
    assert at.radio(key="queue_choice").value == queue_ids[1]
    assert metric_value(at, "Reviewed") == "1"


def test_decisions_reset_when_a_dataset_is_loaded():
    at = start_demo()
    at.button(key="decide_possible_duplicate").click().run()
    assert metric_value(at, "Reviewed") == "1"
    at.button(key="load_demo").click().run()
    assert metric_value(at, "Reviewed") == "0"
    assert at.session_state["decisions"] == {}


# --------------------------------------------------------------------------
# Review limit and demo cases
# --------------------------------------------------------------------------

def test_changing_the_review_limit_changes_queued_for_review():
    at = start_demo()
    at.number_input(key="review_limit").set_value(5).run()
    assert metric_value(at, "Queued for review") == "5"
    assert len(at.radio(key="queue_choice").options) == 5
    assert metric_value(at, "Incoming invoices") == "32"

    at.number_input(key="review_limit").set_value(500).run()
    assert metric_value(at, "Queued for review") == "32"  # never more than there are invoices


def test_demo_case_below_the_limit_is_shown_without_changing_the_queue():
    at = start_demo()
    ranking_before = at.session_state["dataset"].queue["incoming_id"].tolist()
    rank = ranking_before.index(DEMO_RECURRING) + 1
    assert rank > 1, "the legitimate recurring bill should not be the model's top match"
    limit = rank - 1  # the largest review limit that leaves the recurring bill out
    at.number_input(key="review_limit").set_value(limit).run()
    queue_before = list(at.radio(key="queue_choice").options)

    at.segmented_control(key="demo_case").set_value("recurring_bill").run()
    assert not at.exception
    assert at.session_state["selected_id"] == DEMO_RECURRING
    assert list(at.radio(key="queue_choice").options) == queue_before
    assert at.radio(key="queue_choice").value is None  # the shown invoice is not in the queue
    assert at.session_state["dataset"].queue["incoming_id"].tolist() == ranking_before

    text = page_html(at)
    assert f"Ranked {rank} of 32, below the review limit of {limit}. Shown as a demo case." in text
    assert "Why this ranks lower" in text and "Why this was flagged" not in text
    assert "Different billing period" in text


def test_picking_a_queue_row_clears_the_demo_case_selector():
    at = start_demo()
    dataset = at.session_state["dataset"]
    case_ids = {case["incoming_id"] for case in dataset.cases}
    other_id = next(incoming_id for incoming_id in dataset.queue["incoming_id"] if incoming_id not in case_ids)
    options_before = list(at.radio(key="queue_choice").options)
    at.radio(key="queue_choice").set_value(other_id).run()
    assert at.session_state["selected_id"] == other_id
    assert at.segmented_control(key="demo_case").value is None
    assert list(at.radio(key="queue_choice").options) == options_before


def test_ambiguous_case_shows_missing_reference_and_accepts_needs_more_information():
    at = start_demo()
    at.number_input(key="review_limit").set_value(32).run()  # every invoice is in the queue
    at.segmented_control(key="demo_case").set_value("ambiguous").run()
    text = page_html(at)
    assert "Invoice reference missing on incoming invoice" in text
    assert "im-evidence-missing" in text
    assert "Check before deciding" in text
    at.button(key="decide_needs_more_information").click().run()
    assert at.session_state["decisions"] == {"INC-0013": "Needs more information"}


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------

def test_upload_of_valid_files_builds_a_queue_and_shows_reader_notes():
    at = upload_through_dialog(start_app(), "valid_incoming.csv", "valid_payments.csv")
    assert not at.exception
    assert metric_value(at, "Incoming invoices") == "8"
    assert metric_value(at, "Queued for review") == "8"
    assert at.session_state["dataset"].name == "Uploaded files"
    assert "Ignored extra column in the prior payments file: department." in page_html(at)
    assert len(at.segmented_control) == 0  # demo cases belong to the demo only
    assert at.radio(key="queue_choice").value == at.session_state["dataset"].queue["incoming_id"].iloc[0]


@pytest.mark.parametrize("bad_file, expected", [
    ("malformed_incoming.csv", "The incoming invoices file is missing required column: amount."),
    ("bad_values_incoming.csv", "Row 2: invoice_date &#x27;14/02/2026&#x27; is not a valid date."),
    ("bad_values_incoming.csv", "Row 3: amount &#x27;about 189&#x27; is not a valid amount."),
    ("bad_values_incoming.csv", "Row 4: vendor is blank."),
    ("headerless.csv", "The incoming invoices file has no column names in its first row."),
])
def test_bad_upload_explains_the_problem_and_keeps_the_previous_dataset(bad_file: str, expected: str):
    at = start_demo()
    at.button(key="decide_possible_duplicate").click().run()
    upload_through_dialog(at, bad_file, "valid_payments.csv")
    assert not at.exception
    assert expected in page_html(at)
    assert at.session_state["dataset"].name == "Built-in demo"
    assert at.session_state["decisions"] == {DEMO_DUPLICATE: "Possible duplicate"}


def test_upload_without_files_names_the_missing_ones():
    at = upload_through_dialog(start_app(), None, None)
    text = page_html(at)
    assert "Incoming invoices CSV is missing." in text
    assert "Prior payments CSV is missing." in text
    assert at.session_state["dataset"] is None


def test_empty_payments_file_gives_a_message_instead_of_a_queue():
    at = upload_through_dialog(start_app(), "valid_incoming.csv", "empty_payments.csv")
    assert not at.exception
    assert "No prior payments to compare against" in page_html(at)
    assert metric_value(at, "Incoming invoices") == "8"
    assert metric_value(at, "Queued for review") == "0"
    assert len(at.radio) == 0


# --------------------------------------------------------------------------
# Export (the export-building functions, called directly)
# --------------------------------------------------------------------------

def test_export_escapes_text_that_could_start_a_formula(fixture_dataset: app.Dataset):
    equals_id = incoming_id_of(fixture_dataset, "=Nettlebrook")
    plus_id = incoming_id_of(fixture_dataset, "+Plus")
    decisions = {equals_id: "Possible duplicate", plus_id: "Legitimate bill"}
    text = app.build_export_csv(fixture_dataset, decisions).decode("utf-8-sig")
    assert "'=Nettlebrook Audio Hire" in text
    assert "'+Plus Harbor Couriers" in text
    for line in text.splitlines():
        for cell in line.split(","):
            assert not cell.startswith(("=", "+", "-", "@"))


def test_export_has_one_row_per_reviewed_item_in_rank_order(fixture_dataset: app.Dataset):
    queue_ids = fixture_dataset.queue["incoming_id"].tolist()
    decisions = {queue_ids[4]: "Legitimate bill", queue_ids[0]: "Possible duplicate"}
    frame = app.build_export_frame(fixture_dataset, decisions)
    assert list(frame.columns) == list(app.EXPORT_COLUMNS)
    assert frame["rank"].tolist() == [1, 5]
    assert frame["decision"].tolist() == ["Possible duplicate", "Legitimate bill"]
    assert frame["incoming_row_id"].tolist() == [queue_ids[0], queue_ids[4]]
    top = frame.iloc[0]
    assert top["incoming_vendor"] == "Tindercroft Signs LLC" and top["payment_vendor"] == "Tindercroft Signs"
    assert top["incoming_amount"] == "189.00" and top["payment_amount"] == "189.00"
    assert top["incoming_invoice_date"] == "2026-02-18"
    assert top["payment_row_id"] == "P-101"
    assert top["match_score"] == f"{app.display_score(fixture_dataset.queue['score'].iloc[0]):.2f}"
    assert all("0.01" <= score <= "0.99" for score in frame["match_score"])


def test_export_keeps_leading_zeros_and_blank_fields(fixture_dataset: app.Dataset):
    zero_id = incoming_id_of(fixture_dataset, "Bramblewick")
    plus_id = incoming_id_of(fixture_dataset, "+Plus")
    decisions = {zero_id: "Needs more information", plus_id: "Legitimate bill"}
    exported = pd.read_csv(
        pd.io.common.BytesIO(app.build_export_csv(fixture_dataset, decisions)),
        dtype=str, keep_default_na=False, encoding="utf-8-sig",
    )
    by_id = exported.set_index("incoming_row_id")
    assert by_id.loc[zero_id, "incoming_invoice_ref"] == "000418"
    assert by_id.loc[plus_id, "incoming_invoice_ref"] == ""
    assert by_id.loc[plus_id, "incoming_billing_period"] == ""


def test_export_with_no_decisions_is_only_the_header(fixture_dataset: app.Dataset):
    text = app.build_export_csv(fixture_dataset, {}).decode("utf-8-sig")
    assert text.strip() == ",".join(app.EXPORT_COLUMNS)


def test_plain_amount_never_uses_floats():
    assert app.plain_amount(46820) == "468.20"
    assert app.plain_amount(5) == "0.05"
    assert app.plain_amount(124000) == "1240.00"


# --------------------------------------------------------------------------
# Helpers that keep user text harmless
# --------------------------------------------------------------------------

def test_literal_markdown_wraps_text_in_a_code_span_it_cannot_close():
    assert app.literal_markdown("Plain Name 42") == "` Plain Name 42 `"
    assert app.literal_markdown("*bold* [x](http://e.test) :red[hi] $5$") == "` *bold* [x](http://e.test) :red[hi] $5$ `"
    assert app.literal_markdown("two\n\n# lines") == "` two # lines `"  # one line, so no heading
    assert app.literal_markdown("a ` b") == "`` a ` b ``"
    assert app.literal_markdown("a ``` b ` c") == "```` a ``` b ` c ````"


def test_queue_row_text_is_literal_markdown():
    row = pd.Series({"vendor": "**Bold** Co", "invoice_ref": "[ref](http://e.test)", "amount_cents": 46820})
    assert app.queue_row_label(row) == "` **Bold** Co ` ` $468.20 `"
    assert app.queue_row_caption(row, app.NOT_REVIEWED) == "` [ref](http://e.test) ` \u00b7 Not reviewed"
    assert app.queue_row_caption(row, "Possible duplicate") == "` [ref](http://e.test) ` :orange-badge[Possible duplicate]"
    blank = pd.Series({"vendor": "A", "invoice_ref": "", "amount_cents": 100})
    assert app.queue_row_caption(blank, app.NOT_REVIEWED) == "No reference \u00b7 Not reviewed"


def test_cards_escape_html_in_user_text():
    incoming = pd.Series({"vendor": "<script>alert(1)</script>", "invoice_ref": '"><img src=x>',
                          "amount_cents": 1000, "invoice_date": pd.Timestamp("2026-03-01"), "billing_period": ""})
    payment = pd.Series({"vendor": "A & B <b>bold</b>", "invoice_ref": "", "amount_cents": 1000,
                         "invoice_date": pd.Timestamp("2026-03-01"), "billing_period": ""})
    text = app.cards_html(matcher.compare_fields(incoming, payment), "<i>id</i>", "PAY-1")
    assert "<script>" not in text and "<img" not in text and "<b>bold" not in text and "<i>id" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "A &amp; B &lt;b&gt;bold&lt;/b&gt;" in text


def test_notice_escapes_html_in_reader_messages():
    text = app.notice_html("Title <b>", ["Ignored extra column: <img src=x onerror=alert(1)>"], "info")
    assert "<img" not in text and "<b>" not in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text


def test_blank_optional_fields_show_an_em_dash_and_missing_evidence(fixture_dataset: app.Dataset):
    plus_id = incoming_id_of(fixture_dataset, "+Plus")  # no reference and no billing period
    entry = fixture_dataset.queue.loc[plus_id]
    comparisons = matcher.compare_fields(fixture_dataset.incoming.loc[plus_id],
                                         fixture_dataset.payments.loc[entry["payment_id"]])
    text = app.cards_html(comparisons, plus_id, entry["payment_id"])
    assert f'<span class="im-value">{matcher.BLANK_DISPLAY}</span>' in text
    assert "im-evidence-missing" in text
    assert "Invoice reference missing on" in text and "Billing period missing on" in text


def test_cards_show_the_same_five_rows_in_the_same_order_on_both_sides(fixture_dataset: app.Dataset):
    top_id = fixture_dataset.queue["incoming_id"].iloc[0]
    entry = fixture_dataset.queue.loc[top_id]
    comparisons = matcher.compare_fields(fixture_dataset.incoming.loc[top_id],
                                         fixture_dataset.payments.loc[entry["payment_id"]])
    text = app.cards_html(comparisons, top_id, entry["payment_id"])
    incoming_card, payment_card = text.split('<section class="im-card im-card-payment"')
    labels = ["Vendor", "Invoice reference", "Amount", "Invoice date", "Billing period"]
    for card in (incoming_card, payment_card):
        positions = [card.index(f'<span class="im-label">{label}</span>') for label in labels]
        assert positions == sorted(positions)
    assert "im-evidence" not in incoming_card  # evidence is written once, on the payment side
    assert payment_card.count('class="im-evidence ') == 5


def test_contributions_table_adds_up_to_the_decision_value(fixture_dataset: app.Dataset,
                                                           model: matcher.LearnedMatcher):
    top_id = fixture_dataset.queue["incoming_id"].iloc[0]
    entry = fixture_dataset.queue.loc[top_id]
    table = model.contributions(fixture_dataset.incoming.loc[top_id],
                                fixture_dataset.payments.loc[entry["payment_id"]])
    total = table["contribution"].sum() + table.attrs["intercept"]
    assert total == pytest.approx(entry["score"], abs=1e-6)
    text = app.contributions_html(table, float(entry["score"]), app.feature_labels(model.config))
    assert text.count("<tr>") == len(table) + 3  # header, one row per feature, intercept, sum
    assert app.signed(float(entry["score"])) in text


def test_feature_labels_name_every_model_input_on_the_configured_scale():
    labels = app.feature_labels(matcher.DEFAULT_CONFIG)
    assert set(labels) == set(matcher.BASE_FEATURES)
    assert labels["amount_diff"] == "Amount difference (log of cents apart)"
    other = matcher.MatcherConfig(amount_transform="relative", date_transform="capped_90")
    assert app.feature_labels(other)["amount_diff"] == "Amount difference (share of the larger amount)"
    assert app.feature_labels(other)["date_gap"] == "Invoice date gap (days apart, up to 90)"
    assert app.feature_label("amount_exact", labels) == "Amounts equal"
    assert app.feature_label("amount_exact*vendor_fuzzy", labels) == (
        f"Amounts equal {app.TIMES_SIGN} Vendor name similarity (spelling)")
    assert app.feature_label("not_a_known_input", labels) == "not_a_known_input"


def test_signed_numbers_use_a_real_minus_sign():
    assert app.signed(1.5) == "+1.50"
    assert app.signed(-0.25) == f"{app.MINUS_SIGN}0.25"
    assert app.MINUS_SIGN == "\u2212" and app.TIMES_SIGN == "\u00d7"


def test_next_unreviewed_skips_decided_items_and_wraps_around():
    queue_ids = ["a", "b", "c", "d"]
    assert app.next_unreviewed(queue_ids, {}, "a") == "b"
    assert app.next_unreviewed(queue_ids, {"b": "Legitimate bill"}, "a") == "c"
    assert app.next_unreviewed(queue_ids, {"a": "Legitimate bill"}, "d") == "b"
    assert app.next_unreviewed(queue_ids, {"a": "x", "b": "x", "c": "x"}, "d") is None
    assert app.next_unreviewed(queue_ids, {}, "not in the queue") == "a"


# --------------------------------------------------------------------------
# Match score display and the weak-match note
# --------------------------------------------------------------------------

def test_display_score_stays_between_one_and_ninety_nine_hundredths():
    for decision_value in (-1000.0, -50.0, -6.0, -0.3, 0.0, 0.3, 6.0, 50.0, 1000.0):
        score = app.display_score(decision_value)
        assert 0.01 <= score <= 0.99
        assert f"{score:.2f}" not in ("0.00", "1.00")
    assert app.display_score(50.0) == 0.99 and app.display_score(-50.0) == 0.01
    assert app.display_score(0.0) == pytest.approx(0.5)  # a score inside the range is not changed
    assert app.display_score(1.0) == pytest.approx(matcher.match_score(1.0))


def test_strong_pair_is_not_shown_as_a_certain_match():
    at = start_demo()  # opens on the disguised duplicate, the strongest kind of pair
    text = page_html(at)
    assert '<p class="im-score-value">1.00</p>' not in text
    shown = text.split('<p class="im-score-value">')[1].split("</p>")[0]
    assert "0.01" <= shown <= "0.99"
    assert app.SCORE_CAPTION in text


def test_weak_match_note_depends_only_on_the_vendor_comparison():
    incoming = pd.Series({"vendor": "Harbor Lane Catering", "invoice_ref": "HL-1", "amount_cents": 5000,
                          "invoice_date": pd.Timestamp("2026-03-01"), "billing_period": "2026-03"})
    same_vendor = incoming.copy()
    same_vendor["invoice_ref"] = "HL-2"
    other_vendor = incoming.copy()
    other_vendor["vendor"] = "Quillfeather Print Works"
    assert not app.vendor_differs(matcher.compare_fields(incoming, same_vendor))
    assert app.vendor_differs(matcher.compare_fields(incoming, other_vendor))
    assert app.WEAK_MATCH_MESSAGE in app.weak_match_html()


def test_weak_match_note_is_shown_for_a_different_vendor_and_leaves_the_ranking_alone():
    at = start_demo()
    assert app.WEAK_MATCH_MESSAGE not in page_html(at)  # the disguised duplicate has a similar vendor
    dataset = at.session_state["dataset"]
    ranking_before = dataset.queue["incoming_id"].tolist()
    weak_ids = [
        entry.incoming_id for entry in dataset.queue.itertuples(index=False)
        if app.vendor_differs(matcher.compare_fields(dataset.incoming.loc[entry.incoming_id],
                                                     dataset.payments.loc[entry.payment_id]))
    ]
    if not weak_ids:
        pytest.skip("the demo ledger has no invoice whose closest payment is from a different vendor")
    at.number_input(key="review_limit").set_value(len(ranking_before)).run()  # every invoice is in the queue
    at.radio(key="queue_choice").set_value(weak_ids[0]).run()
    assert not at.exception
    assert app.WEAK_MATCH_MESSAGE in page_html(at)
    assert at.session_state["dataset"].queue["incoming_id"].tolist() == ranking_before


# --------------------------------------------------------------------------
# Benchmark page
# --------------------------------------------------------------------------

BENCHMARK_SCRIPT = "import app\napp.benchmark_page()"


@pytest.fixture(scope="module")
def benchmark_data() -> dict:
    """The measured results in results/benchmark.json, the file the Benchmark page reads."""
    if not app.BENCHMARK_PATH.exists():
        pytest.skip("results/benchmark.json has not been generated")
    return json.loads(app.BENCHMARK_PATH.read_text(encoding="utf-8"))


def start_benchmark() -> AppTest:
    """Run only the Benchmark page."""
    return AppTest.from_string(BENCHMARK_SCRIPT, default_timeout=RUN_TIMEOUT).run()


def test_benchmark_page_shows_its_empty_state_when_there_are_no_results(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "BENCHMARK_PATH", tmp_path / "benchmark.json")
    at = start_benchmark()
    assert not at.exception
    assert [title.value for title in at.title] == ["Benchmark"]
    text = page_html(at).replace("<code>", "").replace("</code>", "")
    assert ("Benchmark results are not available yet. "
            "Run python evaluate.py --stage test to generate them.") in text
    assert len(at.segmented_control) == 0


def test_benchmark_page_renders_the_measured_results(benchmark_data: dict):
    at = start_benchmark()
    assert not at.exception
    budget = benchmark_data["default_budget"]
    methods, dataset = benchmark_data["methods"], benchmark_data["dataset"]
    assert [title.value for title in at.title] == [f"How many duplicates can we catch in {budget} reviews?"]
    assert at.segmented_control(key="benchmark_budget").value == budget
    assert list(at.segmented_control(key="benchmark_budget").options) == [str(b) for b in benchmark_data["budgets"]]

    text = page_html(at)
    assert f'<p class="im-eyebrow">{benchmark_data["label"]}</p>' in text
    assert html.escape(app.dataset_sentence(dataset)) in text
    assert "These vendors and dates were never seen in training." in text
    assert app.results_table_html(methods, budget, f"Results at a review budget of {budget}") in text
    for method in methods:
        result = method["results"][str(budget)]
        assert html.escape(method["definition"]) in text
        assert f"{result['correct']} of {dataset['known_duplicates']}" in text  # the bar's text label
    for heading in ("Where the learned model fails", "What exact match misses", "Who gets held for review?",
                    "Challenge set (hand-authored, reported separately)", "How to read this"):
        assert f"<h2>{heading}</h2>" in text
    for example in benchmark_data["examples"]:
        assert html.escape(example["likely_cause"]) in text
    assert f"Measured at {budget} reviews." in text
    assert app.HELD_FOR_REVIEW_NOTE in text
    challenge = benchmark_data["challenge"]
    assert app.results_table_html(
        challenge["methods"], challenge["budget"],
        f"Challenge set results at a review budget of {challenge['budget']}") in text
    assert "They are not a measure of real-world accuracy." in text
    assert f"<code>{html.escape(benchmark_data['command'])}</code>" in text
    assert html.escape(app.validation_sentence(benchmark_data)) in text


def test_benchmark_budget_changes_the_title_and_the_table(benchmark_data: dict):
    at = start_benchmark()
    methods = benchmark_data["methods"]
    for budget in benchmark_data["budgets"]:
        at.segmented_control(key="benchmark_budget").set_value(budget).run()
        assert not at.exception
        assert [title.value for title in at.title] == [f"How many duplicates can we catch in {budget} reviews?"]
        text = page_html(at)
        assert app.results_table_html(methods, budget, f"Results at a review budget of {budget}") in text
        assert app.bars_html(methods, budget, benchmark_data["dataset"]["known_duplicates"]) in text


def test_percent_and_share_are_written_for_people():
    assert app.percent(0.65) == "65%" and app.percent(1.0) == "100%" and app.percent(1 / 3) == "33%"
    assert app.percent(None) == matcher.BLANK_DISPLAY
    assert app.share(3, 247) == "3 of 247 (1.2%)"
    assert app.share(0, 0) == "0 of 0"
    assert app.counted(1, "ledger", "ledgers") == "1 ledger"
    assert app.counted(1475, "prior payment", "prior payments") == "1,475 prior payments"


def test_results_table_shows_an_em_dash_when_nothing_was_flagged_and_escapes_names():
    nothing_flagged = {"reviews_used": 0, "correct": 0, "false_flags": 0, "missed": 4,
                       "precision": None, "recall": 0.0}
    methods = [{"key": "exact", "name": "Exact <b>match</b>", "results": {"5": nothing_flagged}}]
    text = app.results_table_html(methods, 5, "A <i>caption</i>")
    assert f'<td data-label="Precision">{matcher.BLANK_DISPLAY}</td>' in text
    assert '<td data-label="Recall">0%</td>' in text
    assert '<td data-label="Duplicates missed">4</td>' in text
    assert "<b>" not in text and "<i>" not in text


def test_bars_share_one_scale_and_carry_their_values_as_text():
    def method(key: str, correct: int) -> dict:
        return {"key": key, "name": key.capitalize(), "results": {"20": {"correct": correct}}}

    text = app.bars_html([method("exact", 0), method("fuzzy", 10), method("learned", 13)], 20, 20)
    assert 'im-bar-baseline" style="width: 0.0%"' in text
    assert 'im-bar-baseline" style="width: 50.0%"' in text
    assert 'im-bar-learned" style="width: 65.0%"' in text
    for label in ("0 of 20", "10 of 20", "13 of 20"):
        assert f'<span class="im-bar-value">{label}</span>' in text
    assert "<figcaption>" in text


def benchmark_example(kind: str) -> dict:
    """A made-up benchmark example with hostile text and blank optional fields."""
    record = {"row_id": "X-1", "vendor": "<script>alert(1)</script>", "invoice_ref": "",
              "amount_cents": 15000, "invoice_date": "2026-05-30", "billing_period": ""}
    proposed = dict(record, row_id="P-1", vendor="Proposed & Co", invoice_ref="A-1", billing_period="2026-05")
    true_payment = dict(record, row_id="P-2", vendor="True Payee", invoice_ref="B-2", billing_period="2026-04")
    comparisons = [
        {"field": "vendor", "label": "Vendor", "status": "differ", "evidence": "Different vendor names"},
        {"field": "invoice_ref", "label": "Invoice reference", "status": "missing", "evidence": "Missing <b>ref</b>"},
        {"field": "amount", "label": "Amount", "status": "match", "evidence": "Same amount"},
        {"field": "invoice_date", "label": "Invoice date", "status": "match", "evidence": "Same invoice date"},
        {"field": "billing_period", "label": "Billing period", "status": "missing", "evidence": "Missing period"},
    ]
    return {"method": "learned", "kind": kind, "budget": 20, "rank": 23, "incoming": record,
            "proposed_payment": proposed, "true_payment": true_payment if kind != "false_flag" else None,
            "comparisons": comparisons, "likely_cause": "Cause with <img src=x> in it."}


def test_benchmark_examples_compare_with_the_right_payment_and_escape_text():
    missed = app.example_html(benchmark_example("missed_duplicate"))
    assert "A known duplicate ranked 23, below the review budget of 20" in missed
    assert "True Payee" in missed and "Proposed &amp; Co" not in missed  # set against the payment it repeats
    assert "Prior payment it repeats" in missed

    flagged = app.example_html(benchmark_example("false_flag"))
    assert "A legitimate invoice that was flagged for review" in flagged
    assert "Proposed &amp; Co" in flagged and "Proposed prior payment" in flagged

    brief = app.brief_example_html(benchmark_example("caught_duplicate"))
    for text in (missed, flagged, brief):
        assert "<script>" not in text and "<img" not in text and "<b>" not in text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
        assert "Cause with &lt;img src=x&gt; in it." in text
        assert "$150.00" in text and "May 30, 2026" in text
    assert f'<span class="im-value">{matcher.BLANK_DISPLAY}</span>' in missed  # blank reference and period
    assert "im-evidence-missing" in missed and "im-evidence-match" in brief
