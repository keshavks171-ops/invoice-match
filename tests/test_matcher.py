"""Tests for matcher.py.

Every ledger here is written by hand with fictional vendors, so the tests do
not need the generated benchmark. The one test that does is skipped when
data/synthetic/train_incoming.csv is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import matcher
from ledger import read_ledger_csv
from matcher import (
    DEFAULT_CONFIG,
    DEFAULT_FUZZY_RULE,
    BenchLedger,
    FuzzyRule,
    LearnedMatcher,
    MatcherConfig,
    compare_fields,
    difference_notes,
    exact_match_queue,
    feature_names,
    fit_shipped_matcher,
    flag_reasons,
    fuzzy_rule_queue,
    load_benchmark_split,
    match_score,
    normalize_ref,
    normalize_vendor,
    vendor_core,
)

SCHEMA_HEADER = "row_id,vendor,invoice_ref,amount,invoice_date,billing_period"
BENCH_HEADER = "ledger_id," + SCHEMA_HEADER
LABELS_HEADER = "ledger_id,fold,incoming_row_id,is_duplicate,true_payment_ids,scenario,vendor_kind"
QUEUE_COLUMNS = ["incoming_id", "payment_id", "score", "rank"]

# The pair used throughout the project to explain the tool.
HERO_INCOMING = ("INC-1", "OC Event Rentals LLC", "INV-1042", "468.20", "2026-03-06", "2026-03")
HERO_PAYMENT = ("PAY-1", "O.C. Event Rentals", "1042", "468.20", "2026-03-06", "2026-03")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ledger_frame(role: str, rows: list[tuple]) -> pd.DataFrame:
    """Build a ledger frame through the real CSV reader.

    Each row is (row_id, vendor, invoice_ref, amount, invoice_date, billing_period).
    """
    lines = [SCHEMA_HEADER] + [",".join(row) for row in rows]
    frame, _notes = read_ledger_csv("\n".join(lines).encode("utf-8"), role)
    return frame


def tiny_ledger(ledger_id: str, vendors: tuple[str, str, str, str]) -> BenchLedger:
    """A small ledger with two known duplicates (I1 of P3, I3 of P4) and four legitimate bills."""
    first, second, third, fourth = vendors
    payments = ledger_frame("payments", [
        ("P1", first, "A-101", "120.00", "2024-01-05", "2024-01"),
        ("P2", first, "A-102", "120.00", "2024-02-05", "2024-02"),
        ("P3", first, "A-103", "120.00", "2024-03-05", "2024-03"),
        ("P4", second, "B-7001", "310.50", "2024-02-11", ""),
        ("P5", second, "B-7002", "95.25", "2024-03-02", ""),
        ("P6", third, "", "800.00", "2024-01-01", "2024-01"),
        ("P7", third, "", "800.00", "2024-02-01", "2024-02"),
        ("P8", third, "", "800.00", "2024-03-01", "2024-03"),
        ("P9", fourth, "D-55", "42.00", "2024-03-09", "2024-03"),
        ("P10", fourth, "D-58", "61.00", "2024-03-20", "2024-03"),
    ])
    incoming = ledger_frame("incoming", [
        ("I1", first + " LLC", "INV A-103", "120.00", "2024-03-05", "2024-03"),  # P3 sent again
        ("I2", first, "A-104", "120.00", "2024-04-05", "2024-04"),  # next month's bill
        ("I3", second, "B7001", "310.50", "2024-02-11", ""),  # P4 sent again
        ("I4", third, "", "800.00", "2024-04-01", "2024-04"),  # next month's rent
        ("I5", fourth, "D-61", "42.00", "2024-04-02", "2024-04"),  # new invoice, familiar amount
        ("I6", "Juniper Hollow Tutoring", "JH-1", "500.00", "2024-04-03", ""),  # new vendor
    ])
    truth = {"I1": frozenset({"P3"}), "I3": frozenset({"P4"})}
    return BenchLedger(ledger_id, 1, incoming, payments, truth, pd.DataFrame())


TRAINING_VENDORS = {
    "T1": ("Marrowbend Print Shop", "Calder Wick Catering", "Elmstead Hall Trust", "Pippin Lane Flowers"),
    "T2": ("Tansy Row Couriers", "Glasswater Audio Hire", "Old Mill Community Rooms", "Fennel Street Bakery"),
    "T3": ("Sorrel Bridge Signs", "Hartwell Coach Travel", "Linden Court Lettings", "Moth and Lantern Books"),
}


def tiny_training_ledgers() -> list[BenchLedger]:
    """Three small ledgers, each with its own vendors, as a stand-in for the training split."""
    return [tiny_ledger(ledger_id, vendors) for ledger_id, vendors in TRAINING_VENDORS.items()]


@pytest.fixture(scope="module")
def tiny_matcher() -> LearnedMatcher:
    return LearnedMatcher.fit(tiny_training_ledgers())


@pytest.fixture()
def hero_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The hero invoice among two ordinary ones, and its earlier payment among other payments."""
    incoming = ledger_frame("incoming", [
        ("INC-0", "Quillon Park Storage", "QP-88", "135.00", "2026-03-01", "2026-03"),
        HERO_INCOMING,
        ("INC-2", "Bramble Yard Cleaning", "", "75.00", "2026-03-04", ""),
    ])
    payments = ledger_frame("payments", [
        ("PAY-0", "Quillon Park Storage", "QP-87", "135.00", "2026-02-01", "2026-02"),
        HERO_PAYMENT,
        ("PAY-2", "Bramble Yard Cleaning", "", "60.00", "2026-02-18", ""),
        ("PAY-3", "OC Event Rentals", "0987", "212.00", "2026-01-15", "2026-01"),
    ])
    return incoming, payments


def feature_value(fitted: LearnedMatcher, incoming_row: tuple, payment_row: tuple, name: str) -> float:
    """One feature of one hand-written pair."""
    incoming = ledger_frame("incoming", [incoming_row])
    payments = ledger_frame("payments", [payment_row])
    return fitted.pair_features(incoming, payments)[0, 0, fitted.feature_names.index(name)]


def write_benchmark(folder, prefix: str, incoming: list[str], payments: list[str], labels: list[str]) -> None:
    """Write the three files of a benchmark split from ready-made CSV lines."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{prefix}incoming.csv").write_text("\n".join([BENCH_HEADER] + incoming) + "\n", encoding="utf-8")
    (folder / f"{prefix}payments.csv").write_text("\n".join([BENCH_HEADER] + payments) + "\n", encoding="utf-8")
    (folder / f"{prefix}labels.csv").write_text("\n".join([LABELS_HEADER] + labels) + "\n", encoding="utf-8")


def write_tiny_train_split(root) -> None:
    """Save the tiny training ledgers as a train split under root/data/synthetic."""
    incoming, payments, labels = [], [], []
    for ledger in tiny_training_ledgers():
        prefix = ledger.ledger_id
        for frame, lines in ((ledger.incoming, incoming), (ledger.payments, payments)):
            for row in frame.itertuples(index=False):
                amount = f"{row.amount_cents // 100}.{row.amount_cents % 100:02d}"
                lines.append(
                    f"{prefix},{prefix}-{row.row_id},{row.vendor},{row.invoice_ref},"
                    f"{amount},{row.invoice_date.date().isoformat()},{row.billing_period}"
                )
        for incoming_id in ledger.incoming["row_id"]:
            valid = "|".join(f"{prefix}-{payment_id}" for payment_id in sorted(ledger.truth.get(incoming_id, ())))
            is_duplicate = 1 if valid else 0
            labels.append(f"{prefix},1,{prefix}-{incoming_id},{is_duplicate},{valid},hand_written,organization")
    write_benchmark(root / "data" / "synthetic", "train_", incoming, payments, labels)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def test_normalize_ref_ignores_prefix_case_and_separators():
    assert normalize_ref("INV-1042") == "1042"
    assert normalize_ref("1042") == "1042"
    assert normalize_ref("inv 1042") == "1042"
    assert normalize_ref("#1042") == "1042"
    assert normalize_ref("Bill/10.42") == "1042"


def test_normalize_ref_keeps_leading_zeros():
    assert normalize_ref("0042") == "0042"
    assert normalize_ref("0042") != normalize_ref("42")


def test_normalize_ref_keeps_a_prefix_that_is_the_whole_reference():
    assert normalize_ref("INV") == "INV"
    assert normalize_ref("invoice") == "INVOICE"
    assert normalize_ref("REF-AB") == "REFAB"  # nothing with a digit would be left


def test_normalize_ref_removes_one_prefix_only():
    assert normalize_ref("INVOICE 77") == "77"  # INVOICE is tried before INV
    assert normalize_ref("INVOICE NO. 77") == "NO77"


def test_normalize_ref_blank_stays_blank():
    assert normalize_ref("") == ""
    assert normalize_ref("   ") == ""


def test_vendor_core_removes_legal_form_and_periods():
    assert vendor_core("OC Event Rentals LLC") == vendor_core("O.C. Event Rentals") == "oc event rentals"
    assert vendor_core("Harbor Print Co., Inc.") == "harbor print"


def test_vendor_normalization_strips_accents_and_handles_ampersand():
    assert normalize_vendor("Café Miró") == "cafe miro"
    assert normalize_vendor("Moth & Lantern Books") == "moth and lantern books"
    assert normalize_vendor("Rosie's  Diner") == "rosies diner"
    assert normalize_vendor("North-Gate/Supply") == "north gate supply"


def test_normalize_vendor_keeps_legal_form_but_vendor_core_drops_it():
    assert normalize_vendor("Elmstead Hall Ltd") == "elmstead hall ltd"
    assert vendor_core("Elmstead Hall Ltd") == "elmstead hall"


def test_vendor_core_is_never_empty_for_a_real_name():
    assert vendor_core("Limited") == "limited"
    assert vendor_core("Co. Inc.") == "co inc"
    assert vendor_core("***") == "***"


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

def test_matcher_config_round_trips_through_json():
    config = MatcherConfig(vendor_text="basic", vendor_scorer="WRatio", amount_transform="relative",
                           date_transform="capped_90", interactions=("ref_exact*amount_exact",),
                           C=0.3, class_weight="balanced")
    restored = MatcherConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert MatcherConfig.from_dict(DEFAULT_CONFIG.to_dict()) == DEFAULT_CONFIG


def test_fuzzy_rule_round_trips_through_json():
    rule = FuzzyRule(vendor_threshold=0.9, amount_rule="within_1_percent", rank_by="vendor", vendor_scorer="ratio")
    assert FuzzyRule.from_dict(json.loads(json.dumps(rule.to_dict()))) == rule
    assert FuzzyRule.from_dict(DEFAULT_FUZZY_RULE.to_dict()) == DEFAULT_FUZZY_RULE


@pytest.mark.parametrize("changes", [
    {"vendor_text": "raw"},
    {"vendor_scorer": "partial_ratio"},
    {"amount_transform": "squared"},
    {"date_transform": "weeks"},
    {"class_weight": "heavy"},
    {"C": 0.0},
    {"interactions": ("amount_exact",)},
    {"interactions": ("amount_exact*row_id",)},
])
def test_matcher_config_rejects_unknown_settings(changes):
    with pytest.raises(ValueError):
        MatcherConfig(**changes)


@pytest.mark.parametrize("changes", [
    {"vendor_threshold": 1.5},
    {"amount_rule": "close"},
    {"rank_by": "date"},
    {"vendor_scorer": "partial_ratio"},
])
def test_fuzzy_rule_rejects_unknown_settings(changes):
    with pytest.raises(ValueError):
        FuzzyRule(**changes)


def test_feature_names_are_base_features_then_interactions():
    assert feature_names(DEFAULT_CONFIG) == list(matcher.BASE_FEATURES)
    config = MatcherConfig(interactions=("amount_exact*vendor_fuzzy", "period_same*amount_exact"))
    assert feature_names(config) == [
        *matcher.BASE_FEATURES, "amount_exact*vendor_fuzzy", "period_same*amount_exact",
    ]


def test_no_feature_describes_a_single_record_or_an_id():
    allowed_stems = ("vendor_", "ref_", "amount_", "date_", "period_")
    assert all(name.startswith(allowed_stems) for name in matcher.BASE_FEATURES)
    assert len(matcher.BASE_FEATURES) == 13


# --------------------------------------------------------------------------
# Pair features
# --------------------------------------------------------------------------

def test_two_blank_references_are_not_a_match(tiny_matcher):
    incoming = ("I", "Elmstead Hall Trust", "", "800.00", "2024-04-01", "2024-04")
    payment = ("P", "Elmstead Hall Trust", "", "800.00", "2024-03-01", "2024-03")
    assert feature_value(tiny_matcher, incoming, payment, "ref_exact") == 0.0
    assert feature_value(tiny_matcher, incoming, payment, "ref_similarity") == 0.0
    assert feature_value(tiny_matcher, incoming, payment, "ref_missing_incoming") == 1.0
    assert feature_value(tiny_matcher, incoming, payment, "ref_missing_payment") == 1.0


def test_reference_features_for_present_and_missing_values(tiny_matcher):
    incoming = ledger_frame("incoming", [
        ("I1", "Pippin Lane Flowers", "INV-9001", "50.00", "2024-04-01", ""),
        ("I2", "Pippin Lane Flowers", "", "50.00", "2024-04-01", ""),
    ])
    payments = ledger_frame("payments", [
        ("P1", "Pippin Lane Flowers", "9001", "50.00", "2024-03-01", ""),
        ("P2", "Pippin Lane Flowers", "", "50.00", "2024-03-01", ""),
        ("P3", "Pippin Lane Flowers", "9010", "50.00", "2024-03-01", ""),
    ])
    cube = tiny_matcher.pair_features(incoming, payments)
    names = tiny_matcher.feature_names
    assert cube.shape == (2, 3, len(names))
    assert cube.dtype == np.float64

    assert cube[:, :, names.index("ref_exact")].tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert cube[:, :, names.index("ref_missing_incoming")].tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert cube[:, :, names.index("ref_missing_payment")].tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    # '9001' and '9010' differ by one swap of neighbours: 1 edit in 4 characters.
    assert cube[:, :, names.index("ref_similarity")].tolist() == [[1.0, 0.0, 0.75], [0.0, 0.0, 0.0]]


def test_amount_date_and_period_features(tiny_matcher):
    incoming = ("I", "Pippin Lane Flowers", "F-2", "50.00", "2024-04-11", "2024-04")
    same = ("P", "Pippin Lane Flowers", "F-1", "50.00", "2024-04-11", "2024-04")
    other = ("P", "Pippin Lane Flowers", "F-1", "62.50", "2024-04-01", "2024-03")
    no_period = ("P", "Pippin Lane Flowers", "F-1", "50.00", "2024-04-01", "")

    assert feature_value(tiny_matcher, incoming, same, "amount_exact") == 1.0
    assert feature_value(tiny_matcher, incoming, same, "amount_diff") == 0.0
    assert feature_value(tiny_matcher, incoming, same, "date_gap") == 0.0
    assert feature_value(tiny_matcher, incoming, same, "period_same") == 1.0
    assert feature_value(tiny_matcher, incoming, same, "period_different") == 0.0

    assert feature_value(tiny_matcher, incoming, other, "amount_exact") == 0.0
    assert feature_value(tiny_matcher, incoming, other, "amount_diff") == pytest.approx(np.log1p(1250))
    assert feature_value(tiny_matcher, incoming, other, "date_gap") == pytest.approx(np.log1p(10))
    assert feature_value(tiny_matcher, incoming, other, "period_same") == 0.0
    assert feature_value(tiny_matcher, incoming, other, "period_different") == 1.0

    assert feature_value(tiny_matcher, incoming, no_period, "period_same") == 0.0
    assert feature_value(tiny_matcher, incoming, no_period, "period_different") == 0.0
    assert feature_value(tiny_matcher, incoming, no_period, "period_missing_incoming") == 0.0
    assert feature_value(tiny_matcher, incoming, no_period, "period_missing_payment") == 1.0


def test_alternative_transforms_and_interactions():
    config = MatcherConfig(amount_transform="relative", date_transform="capped_90",
                           interactions=("amount_exact*vendor_fuzzy", "ref_exact*amount_exact"))
    fitted = LearnedMatcher.fit(tiny_training_ledgers(), config)
    incoming = ledger_frame("incoming", [("I", "Pippin Lane Flowers", "F-7", "50.00", "2024-09-01", "")])
    payments = ledger_frame("payments", [
        ("P1", "Pippin Lane Flowers Ltd", "F-7", "50.00", "2024-01-01", ""),
        ("P2", "Pippin Lane Florist", "F-7", "40.00", "2024-08-22", ""),
    ])
    cube = fitted.pair_features(incoming, payments)
    column = {name: cube[0, :, position] for position, name in enumerate(fitted.feature_names)}

    assert column["amount_diff"].tolist() == [0.0, pytest.approx(1000 / 5000)]
    assert column["date_gap"].tolist() == [90.0, 10.0]  # 244 days is counted as 90
    assert column["amount_exact*vendor_fuzzy"].tolist() == [1.0, 0.0]  # same core name, then a different amount
    assert column["ref_exact*amount_exact"].tolist() == [1.0, 0.0]
    assert 0.0 < column["vendor_fuzzy"][1] < 1.0


def test_vendor_features_use_normalized_names(tiny_matcher):
    incoming = ("I", "Pippin Lane Flowers LLC", "", "50.00", "2024-04-01", "")
    payment = ("P", "PIPPIN LANE FLOWERS", "", "50.00", "2024-04-01", "")
    assert feature_value(tiny_matcher, incoming, payment, "vendor_fuzzy") == 1.0
    assert feature_value(tiny_matcher, incoming, payment, "vendor_tfidf_cosine") == pytest.approx(1.0)


def test_unseen_vendor_names_give_finite_scores(tiny_matcher):
    # No training name contains a digit, so none of these character n-grams were ever seen.
    assert tiny_matcher.vectorizer.transform(["7777 8888", "9999"]).nnz == 0
    incoming = ledger_frame("incoming", [
        ("I1", "7777 8888", "Z-1", "10.00", "2026-01-05", ""),
        ("I2", "9999", "", "20.00", "2026-01-06", "2026-01"),
    ])
    payments = ledger_frame("payments", [
        ("P1", "7777 8888", "Z-1", "10.00", "2025-12-05", ""),
        ("P2", "9999", "", "25.00", "2025-12-06", "2025-12"),
        ("P3", "Pippin Lane Flowers", "F-1", "20.00", "2025-12-07", ""),
    ])
    cube = tiny_matcher.pair_features(incoming, payments)
    names = tiny_matcher.feature_names
    assert np.isfinite(cube).all()
    assert (cube[:, :, names.index("vendor_tfidf_cosine")] == 0.0).all()
    # The fuzzy feature still sees that the names are identical.
    assert cube[0, 0, names.index("vendor_fuzzy")] == 1.0

    assert np.isfinite(tiny_matcher.score_matrix(incoming, payments)).all()
    queue = tiny_matcher.rank(incoming, payments)
    assert len(queue) == 2 and np.isfinite(queue["score"]).all()


# --------------------------------------------------------------------------
# The learned model
# --------------------------------------------------------------------------

def test_fit_is_deterministic():
    first = LearnedMatcher.fit(tiny_training_ledgers())
    second = LearnedMatcher.fit(tiny_training_ledgers())
    assert np.array_equal(first.classifier.coef_, second.classifier.coef_)
    assert np.array_equal(first.classifier.intercept_, second.classifier.intercept_)
    assert np.array_equal(first.scaler.mean_, second.scaler.mean_)
    assert first.vectorizer.vocabulary_ == second.vectorizer.vocabulary_


def test_fit_records_what_it_was_trained_on(tiny_matcher):
    assert tiny_matcher.config == DEFAULT_CONFIG
    assert tiny_matcher.feature_names == feature_names(DEFAULT_CONFIG)
    assert tiny_matcher.n_train_pairs == 3 * 6 * 10
    assert tiny_matcher.n_train_positives == 3 * 2
    assert tiny_matcher.frozen is False
    assert tiny_matcher.classifier.coef_.shape == (1, len(tiny_matcher.feature_names))


def test_fit_labels_every_valid_payment_of_a_duplicate():
    ledger = tiny_ledger("T9", TRAINING_VENDORS["T1"])
    ledger.truth = {"I1": frozenset({"P2", "P3"}), "I3": frozenset({"P4"})}
    fitted = LearnedMatcher.fit([ledger])
    assert fitted.n_train_positives == 3


def test_fit_needs_ledgers_and_both_classes():
    with pytest.raises(ValueError):
        LearnedMatcher.fit([])
    ledger = tiny_ledger("T9", TRAINING_VENDORS["T1"])
    ledger.truth = {}
    with pytest.raises(ValueError):
        LearnedMatcher.fit([ledger])


def test_rank_returns_one_row_per_incoming_invoice(tiny_matcher, hero_frames):
    incoming, payments = hero_frames
    queue = tiny_matcher.rank(incoming, payments)

    assert list(queue.columns) == QUEUE_COLUMNS
    assert sorted(queue["incoming_id"]) == sorted(incoming["row_id"])
    assert queue["rank"].tolist() == [1, 2, 3]
    assert queue["score"].is_monotonic_decreasing
    assert set(queue["payment_id"]) <= set(payments["row_id"])
    # The hero invoice is the strongest match in this little ledger.
    assert queue.iloc[0]["incoming_id"] == "INC-1"
    assert queue.iloc[0]["payment_id"] == "PAY-1"


def test_rank_agrees_with_score_matrix(tiny_matcher, hero_frames):
    incoming, payments = hero_frames
    scores = tiny_matcher.score_matrix(incoming, payments)
    queue = tiny_matcher.rank(incoming, payments).set_index("incoming_id")
    assert scores.shape == (3, 4)
    for row, incoming_id in enumerate(incoming["row_id"]):
        assert queue.loc[incoming_id, "score"] == scores[row].max()
        assert queue.loc[incoming_id, "payment_id"] == payments["row_id"].iloc[int(scores[row].argmax())]


def test_scores_equal_the_classifier_decision_function(tiny_matcher, hero_frames):
    incoming, payments = hero_frames
    cube = tiny_matcher.pair_features(incoming, payments)
    flat = cube.reshape(-1, cube.shape[2])
    expected = tiny_matcher.classifier.decision_function(tiny_matcher.scaler.transform(flat)).reshape(3, 4)
    assert np.allclose(tiny_matcher.score_matrix(incoming, payments), expected, rtol=0, atol=1e-9)


def test_rank_handles_empty_payments(tiny_matcher, hero_frames):
    incoming, _payments = hero_frames
    no_payments = ledger_frame("payments", [])
    queue = tiny_matcher.rank(incoming, no_payments)

    assert list(queue.columns) == QUEUE_COLUMNS
    assert len(queue) == 0
    assert queue["score"].dtype == np.float64
    assert queue["rank"].dtype == np.int64
    assert queue["incoming_id"].dtype == incoming["row_id"].dtype
    assert tiny_matcher.score_matrix(incoming, no_payments).shape == (3, 0)
    assert tiny_matcher.pair_features(incoming, no_payments).shape == (3, 0, len(tiny_matcher.feature_names))


def test_rank_handles_empty_incoming(tiny_matcher, hero_frames):
    incoming, payments = hero_frames
    no_incoming = incoming.iloc[0:0]
    queue = tiny_matcher.rank(no_incoming, payments)
    assert list(queue.columns) == QUEUE_COLUMNS and len(queue) == 0
    assert tiny_matcher.score_matrix(no_incoming, payments).shape == (0, 4)


def test_queue_dtypes_match_the_ledger_ids(tiny_matcher, hero_frames):
    incoming, payments = hero_frames
    queue = tiny_matcher.rank(incoming, payments)
    assert queue["incoming_id"].dtype == incoming["row_id"].dtype
    assert queue["payment_id"].dtype == payments["row_id"].dtype
    assert queue["score"].dtype == np.float64
    assert queue["rank"].dtype == np.int64
    assert queue.index.tolist() == [0, 1, 2]


def test_learned_ties_go_to_the_most_recent_payment_then_file_order(tiny_matcher):
    incoming = ledger_frame("incoming", [("I1", "Pippin Lane Flowers", "", "50.00", "2024-04-11", "")])
    # P1 and P3 are 10 days before and after the invoice; P2 and P4 are copies of them.
    payments = ledger_frame("payments", [
        ("P1", "Pippin Lane Flowers", "", "50.00", "2024-04-01", ""),
        ("P2", "Pippin Lane Flowers", "", "50.00", "2024-04-21", ""),
        ("P3", "Pippin Lane Flowers", "", "50.00", "2024-04-21", ""),
        ("P4", "Pippin Lane Flowers", "", "50.00", "2024-04-01", ""),
    ])
    scores = tiny_matcher.score_matrix(incoming, payments)
    assert len(set(scores[0].tolist())) == 1  # identical features, identical scores
    queue = tiny_matcher.rank(incoming, payments)
    assert queue["payment_id"].tolist() == ["P2"]  # the newer date, and the earlier of its two rows


def test_queue_ties_keep_incoming_file_order(tiny_matcher):
    incoming = ledger_frame("incoming", [
        ("I1", "Pippin Lane Flowers", "", "50.00", "2024-04-11", ""),
        ("I2", "Fennel Street Bakery", "FS-1", "12.00", "2024-04-11", ""),
        ("I3", "Pippin Lane Flowers", "", "50.00", "2024-04-11", ""),
    ])
    payments = ledger_frame("payments", [("P1", "Pippin Lane Flowers", "", "50.00", "2024-04-01", "")])
    queue = tiny_matcher.rank(incoming, payments)
    assert queue["incoming_id"].tolist() == ["I1", "I3", "I2"]
    assert queue["rank"].tolist() == [1, 2, 3]


def test_chunked_scoring_matches_scoring_in_one_piece(tiny_matcher, monkeypatch):
    ledger = tiny_training_ledgers()[1]
    whole_scores = tiny_matcher.score_matrix(ledger.incoming, ledger.payments)
    whole_queue = tiny_matcher.rank(ledger.incoming, ledger.payments)

    monkeypatch.setattr(matcher, "MAX_PAIRS_PER_CHUNK", 25)  # two incoming rows per chunk
    assert matcher._chunk_bounds(6, 10) == [(0, 2), (2, 4), (4, 6)]
    assert np.array_equal(tiny_matcher.score_matrix(ledger.incoming, ledger.payments), whole_scores)
    pd.testing.assert_frame_equal(tiny_matcher.rank(ledger.incoming, ledger.payments), whole_queue)

    monkeypatch.setattr(matcher, "MAX_PAIRS_PER_CHUNK", 1)  # never fewer than one incoming row
    assert len(matcher._chunk_bounds(6, 10)) == 6
    assert np.array_equal(tiny_matcher.score_matrix(ledger.incoming, ledger.payments), whole_scores)


def test_contributions_add_up_to_the_decision_value(tiny_matcher, hero_frames):
    incoming, payments = hero_frames
    scores = tiny_matcher.score_matrix(incoming, payments)
    for row in range(len(incoming)):
        for column in range(len(payments)):
            table = tiny_matcher.contributions(incoming.iloc[row], payments.iloc[column])
            total = table["contribution"].sum() + table.attrs["intercept"]
            assert total == pytest.approx(scores[row, column], abs=1e-9)


def test_contributions_table_layout(tiny_matcher, hero_frames):
    incoming, payments = hero_frames
    table = tiny_matcher.contributions(incoming.iloc[1], payments.iloc[1])
    assert list(table.columns) == ["feature", "value", "standardized_value", "coefficient", "contribution"]
    assert table["feature"].tolist() == tiny_matcher.feature_names
    assert np.allclose(table["contribution"], table["coefficient"] * table["standardized_value"])
    assert table.set_index("feature").loc["ref_exact", "value"] == 1.0
    assert table.attrs["intercept"] == float(tiny_matcher.classifier.intercept_[0])


def test_match_score_is_a_logistic_squash_inside_zero_and_one():
    assert match_score(0.0) == 0.5
    assert match_score(2.0) == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert match_score(-3.0) < match_score(-1.0) < match_score(4.0)
    assert 0.0 < match_score(-1000.0) < match_score(1000.0) < 1.0
    assert isinstance(match_score(np.float64(1.5)), float)


# --------------------------------------------------------------------------
# Baseline 1: exact match
# --------------------------------------------------------------------------

def test_exact_match_finds_the_hero_pair(hero_frames):
    incoming, payments = hero_frames
    queue = exact_match_queue(incoming, payments)
    assert list(queue.columns) == QUEUE_COLUMNS
    assert queue["incoming_id"].tolist() == ["INC-1"]
    assert queue["payment_id"].tolist() == ["PAY-1"]
    assert queue["score"].tolist() == [0.0]  # zero days apart
    assert queue["rank"].tolist() == [1]


def test_exact_match_never_accepts_blank_references():
    incoming = ledger_frame("incoming", [("I1", "Elmstead Hall Trust", "", "800.00", "2024-03-01", "2024-03")])
    payments = ledger_frame("payments", [("P1", "Elmstead Hall Trust", "", "800.00", "2024-03-01", "2024-03")])
    queue = exact_match_queue(incoming, payments)
    assert list(queue.columns) == QUEUE_COLUMNS and len(queue) == 0


def test_exact_match_needs_vendor_reference_and_amount_together():
    incoming = ledger_frame("incoming", [
        ("I1", "Tansy Row Couriers", "TR-500", "64.00", "2024-05-01", ""),  # amount differs
        ("I2", "Tansy Row Couriers", "TR-501", "70.00", "2024-05-01", ""),  # reference differs
        ("I3", "Tansy Road Couriers", "TR-502", "81.00", "2024-05-01", ""),  # vendor differs
        ("I4", "Tansy Row Couriers Inc.", "tr 503", "92.00", "2024-05-01", ""),  # all three agree
    ])
    payments = ledger_frame("payments", [
        ("P1", "Tansy Row Couriers", "TR-500", "64.01", "2024-04-01", ""),
        ("P2", "Tansy Row Couriers", "TR-510", "70.00", "2024-04-01", ""),
        ("P3", "Tansy Row Couriers", "TR-502", "81.00", "2024-04-01", ""),
        ("P4", "Tansy Row Couriers", "TR-503", "92.00", "2024-04-01", ""),
    ])
    queue = exact_match_queue(incoming, payments)
    assert queue["incoming_id"].tolist() == ["I4"]
    assert queue["payment_id"].tolist() == ["P4"]


def test_exact_match_prefers_the_closest_date_and_orders_by_it():
    incoming = ledger_frame("incoming", [
        ("I1", "Sorrel Bridge Signs", "SB-1", "30.00", "2024-06-30", ""),
        ("I2", "Sorrel Bridge Signs", "SB-2", "45.00", "2024-06-30", ""),
    ])
    payments = ledger_frame("payments", [
        ("P1", "Sorrel Bridge Signs", "SB-1", "30.00", "2024-06-01", ""),
        ("P2", "Sorrel Bridge Signs", "SB-1", "30.00", "2024-06-20", ""),
        ("P3", "Sorrel Bridge Signs", "SB-2", "45.00", "2024-06-27", ""),
    ])
    queue = exact_match_queue(incoming, payments)
    assert queue["incoming_id"].tolist() == ["I2", "I1"]  # 3 days apart before 10 days apart
    assert queue["payment_id"].tolist() == ["P3", "P2"]
    assert queue["score"].tolist() == [-3.0, -10.0]


def test_baselines_handle_empty_frames(hero_frames):
    incoming, payments = hero_frames
    for queue in (
        exact_match_queue(incoming, ledger_frame("payments", [])),
        exact_match_queue(incoming.iloc[0:0], payments),
        fuzzy_rule_queue(incoming, ledger_frame("payments", [])),
        fuzzy_rule_queue(incoming.iloc[0:0], payments),
    ):
        assert list(queue.columns) == QUEUE_COLUMNS and len(queue) == 0


# --------------------------------------------------------------------------
# Baseline 2: fuzzy rule
# --------------------------------------------------------------------------

def test_fuzzy_rule_finds_the_hero_pair(hero_frames):
    incoming, payments = hero_frames
    queue = fuzzy_rule_queue(incoming, payments)
    assert queue.iloc[0]["incoming_id"] == "INC-1"
    assert queue.iloc[0]["payment_id"] == "PAY-1"
    assert queue.iloc[0]["score"] == pytest.approx(2.0)  # same core name plus the reference bonus
    # The rule also flags the storage bill: same vendor, same amount, a month apart.
    assert queue["incoming_id"].tolist() == ["INC-1", "INC-0"]


def test_fuzzy_rule_vendor_threshold():
    incoming = ledger_frame("incoming", [("I1", "Glasswater Audio Hire", "", "200.00", "2024-05-10", "")])
    payments = ledger_frame("payments", [("P1", "Glasswater Audio Hiring", "", "200.00", "2024-05-01", "")])
    similarity = 2 * 20 / (21 + 23)  # the names share 20 characters in order, out of 21 and 23

    strict = FuzzyRule(vendor_threshold=0.95, amount_rule="exact", rank_by="vendor", vendor_scorer="ratio")
    loose = FuzzyRule(vendor_threshold=0.90, amount_rule="exact", rank_by="vendor", vendor_scorer="ratio")
    assert len(fuzzy_rule_queue(incoming, payments, strict)) == 0
    queue = fuzzy_rule_queue(incoming, payments, loose)
    assert queue["payment_id"].tolist() == ["P1"]
    assert queue["score"].iloc[0] == pytest.approx(similarity - 9 / 100000)


def test_fuzzy_rule_threshold_is_inclusive():
    incoming = ledger_frame("incoming", [("I1", "Hartwell Coach Travel", "", "90.00", "2024-05-10", "")])
    payments = ledger_frame("payments", [("P1", "Hartwell Coach Travel Ltd", "", "90.00", "2024-05-10", "")])
    rule = FuzzyRule(vendor_threshold=1.0, amount_rule="exact", rank_by="vendor", vendor_scorer="ratio")
    assert fuzzy_rule_queue(incoming, payments, rule)["score"].tolist() == [1.0]


@pytest.mark.parametrize("amount_rule, expected", [
    ("exact", ["P1"]),
    ("within_100_cents", ["P1", "P2", "P3"]),
    ("within_1_percent", ["P1", "P2", "P3", "P4"]),
])
def test_fuzzy_rule_amount_rules(amount_rule, expected):
    payments = ledger_frame("payments", [
        ("P1", "Linden Court Lettings", "", "500.00", "2024-05-01", ""),
        ("P2", "Linden Court Lettings", "", "499.00", "2024-05-01", ""),  # 100 cents below
        ("P3", "Linden Court Lettings", "", "501.00", "2024-05-01", ""),  # 100 cents above
        ("P4", "Linden Court Lettings", "", "505.05", "2024-05-01", ""),  # 505 cents above: 1% of 505.05
        ("P5", "Linden Court Lettings", "", "494.90", "2024-05-01", ""),  # 510 cents below: more than 1%
    ])
    rule = FuzzyRule(vendor_threshold=0.9, amount_rule=amount_rule, rank_by="vendor", vendor_scorer="ratio")
    eligible = []
    for payment_id in payments["row_id"]:
        one_payment = payments[payments["row_id"] == payment_id].reset_index(drop=True)
        incoming = ledger_frame("incoming", [("I1", "Linden Court Lettings", "", "500.00", "2024-05-01", "")])
        if len(fuzzy_rule_queue(incoming, one_payment, rule)) == 1:
            eligible.append(payment_id)
    assert eligible == expected


def test_fuzzy_rule_rank_by_reference_first_or_vendor_only():
    incoming = ledger_frame("incoming", [
        ("I1", "Moth and Lantern Books", "ML-9", "40.00", "2024-05-10", ""),  # identical name, new reference
        ("I2", "Old Mill Community Room", "OM-3", "65.00", "2024-05-10", ""),  # name slightly off, same reference
    ])
    payments = ledger_frame("payments", [
        ("P1", "Moth and Lantern Books", "ML-8", "40.00", "2024-05-10", ""),
        ("P2", "Old Mill Community Rooms", "OM-3", "65.00", "2024-05-10", ""),
    ])
    by_vendor = FuzzyRule(vendor_threshold=0.85, amount_rule="exact", rank_by="vendor")
    by_ref = FuzzyRule(vendor_threshold=0.85, amount_rule="exact", rank_by="ref_then_vendor")

    assert fuzzy_rule_queue(incoming, payments, by_vendor)["incoming_id"].tolist() == ["I1", "I2"]
    queue = fuzzy_rule_queue(incoming, payments, by_ref)
    assert queue["incoming_id"].tolist() == ["I2", "I1"]
    assert queue["score"].iloc[0] > 1.85  # the reference bonus on top of a similarity of at least 0.85
    assert queue["score"].iloc[1] == 1.0  # an identical name without the bonus


def test_fuzzy_rule_breaks_ties_by_date_then_by_file_order():
    incoming = ledger_frame("incoming", [
        ("I1", "Calder Wick Catering", "", "150.00", "2024-05-11", ""),
        ("I2", "Fennel Street Bakery", "", "18.00", "2024-05-11", ""),
        ("I3", "Calder Wick Catering", "", "150.00", "2024-05-11", ""),
    ])
    payments = ledger_frame("payments", [
        ("P1", "Calder Wick Catering", "", "150.00", "2024-03-01", ""),  # same vendor, far away in time
        ("P2", "Calder Wick Catering", "", "150.00", "2024-05-01", ""),  # 10 days before
        ("P3", "Calder Wick Catering", "", "150.00", "2024-05-21", ""),  # 10 days after: the newer of the tie
        ("P4", "Calder Wick Catering", "", "150.00", "2024-05-21", ""),  # same as P3, later in the file
        ("P5", "Fennel Street Bakery", "", "18.00", "2024-05-11", ""),
    ])
    queue = fuzzy_rule_queue(incoming, payments, FuzzyRule(rank_by="vendor"))
    # I2 has no date gap; I1 and I3 tie and keep their file order.
    assert queue["incoming_id"].tolist() == ["I2", "I1", "I3"]
    assert queue["payment_id"].tolist() == ["P5", "P3", "P3"]
    assert queue["rank"].tolist() == [1, 2, 3]
    pd.testing.assert_frame_equal(queue, fuzzy_rule_queue(incoming, payments, FuzzyRule(rank_by="vendor")))


def test_baseline_queues_survive_chunking(hero_frames, monkeypatch):
    incoming, payments = hero_frames
    exact = exact_match_queue(incoming, payments)
    fuzzy = fuzzy_rule_queue(incoming, payments)
    monkeypatch.setattr(matcher, "MAX_PAIRS_PER_CHUNK", 1)
    pd.testing.assert_frame_equal(exact_match_queue(incoming, payments), exact)
    pd.testing.assert_frame_equal(fuzzy_rule_queue(incoming, payments), fuzzy)


# --------------------------------------------------------------------------
# Evidence text
# --------------------------------------------------------------------------

def hero_comparisons() -> list:
    incoming = ledger_frame("incoming", [HERO_INCOMING])
    payments = ledger_frame("payments", [HERO_PAYMENT])
    return compare_fields(incoming.iloc[0], payments.iloc[0])


def test_hero_pair_evidence():
    comparisons = hero_comparisons()
    assert [item.field for item in comparisons] == ["vendor", "invoice_ref", "amount", "invoice_date", "billing_period"]
    assert [item.label for item in comparisons] == [
        "Vendor", "Invoice reference", "Amount", "Invoice date", "Billing period",
    ]
    assert [item.evidence for item in comparisons] == [
        "Vendor names match after normalization",
        "References match after normalization",
        "Same amount",
        "Same invoice date",
        "Same billing period",
    ]
    assert [item.status for item in comparisons] == ["match"] * 5


def test_hero_pair_displays_and_notes():
    vendor, ref, amount, date, period = hero_comparisons()
    assert (vendor.incoming_display, vendor.payment_display) == ("OC Event Rentals LLC", "O.C. Event Rentals")
    assert vendor.incoming_note == "Normalized: oc event rentals"
    assert vendor.payment_note == "Normalized: oc event rentals"
    assert (ref.incoming_display, ref.payment_display) == ("INV-1042", "1042")
    assert (ref.incoming_note, ref.payment_note) == ("Normalized: 1042", "")
    assert (amount.incoming_display, amount.payment_display) == ("$468.20", "$468.20")
    assert (date.incoming_display, date.payment_display) == ("Mar 6, 2026", "Mar 6, 2026")
    assert (period.incoming_display, period.payment_display) == ("Mar 2026", "Mar 2026")
    assert (amount.incoming_note, date.incoming_note, period.incoming_note) == ("", "", "")


def test_hero_pair_flag_reasons_start_with_the_reference():
    comparisons = hero_comparisons()
    assert flag_reasons(comparisons) == [
        "References match after normalization (INV-1042 and 1042)",
        "Same amount ($468.20)",
        "Same billing period (Mar 2026)",
    ]
    assert flag_reasons(comparisons, max_items=5)[3:] == [
        "Vendor names match after normalization (OC Event Rentals LLC and O.C. Event Rentals)",
        "Same invoice date (Mar 6, 2026)",
    ]
    assert flag_reasons(comparisons, max_items=1) == ["References match after normalization (INV-1042 and 1042)"]
    assert difference_notes(comparisons) == []


def test_identical_values_are_reported_as_the_same():
    incoming = ledger_frame("incoming", [HERO_PAYMENT])
    payments = ledger_frame("payments", [HERO_PAYMENT])
    vendor, ref, _amount, _date, _period = compare_fields(incoming.iloc[0], payments.iloc[0])
    assert (vendor.status, vendor.evidence) == ("match", "Same vendor name")
    assert (ref.status, ref.evidence) == ("match", "Same invoice reference")


def test_evidence_for_a_pair_that_differs_everywhere():
    incoming = ledger_frame("incoming", [("I1", "Larch Meadow Vets", "LM-200", "480.20", "2026-04-06", "2026-04")])
    payments = ledger_frame("payments", [("P1", "Copperfield Glazing", "CG-19", "468.20", "2026-03-06", "2026-03")])
    comparisons = compare_fields(incoming.iloc[0], payments.iloc[0])
    assert [item.status for item in comparisons] == ["differ"] * 5
    assert [item.evidence for item in comparisons] == [
        "Different vendor names",
        "Different invoice references",
        "Amounts differ by $12.00",
        "Invoice dates 31 days apart",
        "Different billing period",
    ]
    assert flag_reasons(comparisons) == []
    assert difference_notes(comparisons) == [
        "Different billing period (Apr 2026 and Mar 2026)",
        "Different invoice references (LM-200 and CG-19)",
        "Amounts differ by $12.00 ($480.20 and $468.20)",
    ]
    assert difference_notes(comparisons, max_items=5)[3:] == [
        "Different vendor names (Larch Meadow Vets and Copperfield Glazing)",
        "Invoice dates 31 days apart (Apr 6, 2026 and Mar 6, 2026)",
    ]


def test_similar_vendor_and_reference_are_partial_support():
    incoming = ledger_frame("incoming", [("I1", "Glasswater Audio Hire", "GA-20418", "90.00", "2026-03-07", "")])
    payments = ledger_frame("payments", [("P1", "Glasswater Audio Hiring", "GA-20481", "90.00", "2026-03-06", "")])
    vendor, ref, _amount, date, _period = compare_fields(incoming.iloc[0], payments.iloc[0])
    assert (vendor.status, vendor.evidence) == ("partial", "Similar vendor names")
    assert (ref.status, ref.evidence) == ("partial", "Similar invoice references")
    assert (date.status, date.evidence) == ("partial", "Invoice dates 1 day apart")
    assert vendor.incoming_note == ""  # only the letter case changed
    assert ref.incoming_note == "Normalized: GA20418"


@pytest.mark.parametrize("payment_date, status, evidence", [
    ("2026-03-06", "match", "Same invoice date"),
    ("2026-03-05", "partial", "Invoice dates 1 day apart"),
    ("2026-03-13", "partial", "Invoice dates 7 days apart"),
    ("2026-02-26", "differ", "Invoice dates 8 days apart"),
])
def test_date_evidence(payment_date, status, evidence):
    incoming = ledger_frame("incoming", [("I1", "Larch Meadow Vets", "", "10.00", "2026-03-06", "")])
    payments = ledger_frame("payments", [("P1", "Larch Meadow Vets", "", "10.00", payment_date, "")])
    date = compare_fields(incoming.iloc[0], payments.iloc[0])[3]
    assert (date.status, date.evidence) == (status, evidence)


@pytest.mark.parametrize("incoming_ref, payment_ref, evidence", [
    ("", "LM-7", "Invoice reference missing on incoming invoice"),
    ("LM-7", "", "Invoice reference missing on prior payment"),
    ("", "", "Invoice reference missing on both"),
])
def test_missing_reference_evidence(incoming_ref, payment_ref, evidence):
    incoming = ledger_frame("incoming", [("I1", "Larch Meadow Vets", incoming_ref, "10.00", "2026-03-06", "")])
    payments = ledger_frame("payments", [("P1", "Larch Meadow Vets", payment_ref, "10.00", "2026-03-06", "")])
    ref = compare_fields(incoming.iloc[0], payments.iloc[0])[1]
    assert (ref.status, ref.evidence) == ("missing", evidence)
    assert ref.incoming_display == (incoming_ref or matcher.BLANK_DISPLAY)
    assert ref.payment_display == (payment_ref or matcher.BLANK_DISPLAY)
    assert ref.incoming_note == ("Normalized: LM7" if incoming_ref else "")
    assert ref.payment_note == ("Normalized: LM7" if payment_ref else "")


@pytest.mark.parametrize("incoming_period, payment_period, evidence", [
    ("", "2026-03", "Billing period missing on incoming invoice"),
    ("2026-03", "", "Billing period missing on prior payment"),
    ("", "", "Billing period missing on both"),
])
def test_missing_period_evidence(incoming_period, payment_period, evidence):
    incoming = ledger_frame("incoming", [("I1", "Larch Meadow Vets", "", "10.00", "2026-03-06", incoming_period)])
    payments = ledger_frame("payments", [("P1", "Larch Meadow Vets", "", "10.00", "2026-03-06", payment_period)])
    period = compare_fields(incoming.iloc[0], payments.iloc[0])[4]
    assert (period.status, period.evidence) == ("missing", evidence)
    assert period.incoming_display == ("Mar 2026" if incoming_period else matcher.BLANK_DISPLAY)
    assert period.payment_display == ("Mar 2026" if payment_period else matcher.BLANK_DISPLAY)


def test_difference_notes_list_differences_before_missing_fields():
    incoming = ledger_frame("incoming", [("I1", "Larch Meadow Vets", "", "25.00", "2026-04-06", "")])
    payments = ledger_frame("payments", [("P1", "Larch Meadow Vets", "LM-7", "10.00", "2026-03-06", "")])
    comparisons = compare_fields(incoming.iloc[0], payments.iloc[0])
    assert difference_notes(comparisons, max_items=10) == [
        "Amounts differ by $15.00 ($25.00 and $10.00)",
        "Invoice dates 31 days apart (Apr 6, 2026 and Mar 6, 2026)",
        "Billing period missing on both",
        "Invoice reference missing on incoming invoice",
    ]
    assert difference_notes(comparisons, max_items=2) == [
        "Amounts differ by $15.00 ($25.00 and $10.00)",
        "Invoice dates 31 days apart (Apr 6, 2026 and Mar 6, 2026)",
    ]
    assert flag_reasons(comparisons) == ["Same vendor name (Larch Meadow Vets)"]


# --------------------------------------------------------------------------
# Loading the benchmark
# --------------------------------------------------------------------------

@pytest.fixture()
def bench_dir(tmp_path):
    """A two-ledger validation split. Ledger B02 comes first in the files."""
    write_benchmark(
        tmp_path / "synthetic", "validation_",
        incoming=[
            "B02,B02-I-0001,Larch Meadow Vets,LM-7,10.00,2025-01-06,2025-01",
            "B02,B02-I-0002,Copperfield Glazing,0042,95.50,2025-01-07,",
            "B01,B01-I-0001,Larch Meadow Vets,,12.00,2024-11-02,",
        ],
        payments=[
            "B02,B02-P-0001,Larch Meadow Vets,LM-7,10.00,2024-12-06,2024-12",
            "B02,B02-P-0002,Larch Meadow Veterinary,LM7,10.00,2024-12-20,2024-12",
            "B01,B01-P-0001,Larch Meadow Vets,,12.00,2024-10-02,",
            "B02,B02-P-0003,Copperfield Glazing,0041,95.50,2024-12-07,",
        ],
        labels=[
            "B02,2,B02-I-0001,1,B02-P-0001|B02-P-0002,dup_exact,organization",
            "B02,2,B02-I-0002,0,,legit_recurring_next,organization",
            "B01,3,B01-I-0001,0,,legit_recurring_next,organization",
        ],
    )
    return tmp_path


def test_load_benchmark_split_groups_rows_by_ledger(bench_dir):
    ledgers = load_benchmark_split("validation", bench_dir)
    assert [ledger.ledger_id for ledger in ledgers] == ["B02", "B01"]  # order of the incoming file
    assert [ledger.fold for ledger in ledgers] == [2, 3]

    first, second = ledgers
    assert first.incoming["row_id"].tolist() == ["B02-I-0001", "B02-I-0002"]
    assert first.payments["row_id"].tolist() == ["B02-P-0001", "B02-P-0002", "B02-P-0003"]
    assert first.incoming.index.tolist() == [0, 1]
    assert first.incoming["invoice_ref"].tolist() == ["LM-7", "0042"]  # leading zeros survive
    assert first.incoming["amount_cents"].tolist() == [1000, 9550]
    assert list(first.incoming.columns) == [
        "row_id", "vendor", "invoice_ref", "amount_cents", "invoice_date", "billing_period",
    ]
    assert first.truth == {"B02-I-0001": frozenset({"B02-P-0001", "B02-P-0002"})}
    assert first.labels["incoming_row_id"].tolist() == ["B02-I-0001", "B02-I-0002"]
    assert first.labels["is_duplicate"].tolist() == [1, 0]
    assert first.labels["scenario"].tolist() == ["dup_exact", "legit_recurring_next"]

    assert second.payments["row_id"].tolist() == ["B01-P-0001"]
    assert second.truth == {}


def test_load_benchmark_split_reads_the_challenge_folder(tmp_path):
    write_benchmark(
        tmp_path / "challenge", "",
        incoming=["CH01,CH01-I-0001,Larch Meadow Vets,LM-7,10.00,2026-09-06,"],
        payments=["CH01,CH01-P-0001,Larch Meadow Vets,LM-7,10.00,2026-08-06,"],
        labels=["CH01,1,CH01-I-0001,1,CH01-P-0001,dup_exact,organization"],
    )
    (ledger,) = load_benchmark_split("challenge", tmp_path)
    assert (ledger.ledger_id, ledger.fold) == ("CH01", 1)
    assert ledger.truth == {"CH01-I-0001": frozenset({"CH01-P-0001"})}


def test_load_benchmark_split_names_the_missing_file_and_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        load_benchmark_split("train", tmp_path)
    assert "train_incoming.csv" in str(error.value)
    assert "python synth.py" in str(error.value)


def test_load_benchmark_split_rejects_an_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        load_benchmark_split("holdout", tmp_path)


def test_load_benchmark_split_rejects_labels_that_do_not_fit(tmp_path):
    incoming = ["B01,B01-I-0001,Larch Meadow Vets,LM-7,10.00,2025-01-06,"]
    payments = ["B01,B01-P-0001,Larch Meadow Vets,LM-7,10.00,2024-12-06,"]
    bad_label_sets = [
        ["B01,1,B01-I-0001,1,B01-P-0099,dup_exact,organization"],  # a payment that is not in the ledger
        ["B01,1,B01-I-0001,0,B01-P-0001,dup_exact,organization"],  # the two label columns disagree
        ["B01,1,B01-I-0002,0,,legit_one_off,organization"],  # an invoice that is not in the ledger
    ]
    for labels in bad_label_sets:
        write_benchmark(tmp_path / "synthetic", "train_", incoming, payments, labels)
        with pytest.raises(ValueError):
            load_benchmark_split("train", tmp_path)


def test_load_benchmark_split_applies_row_limits_per_ledger(tmp_path):
    # 2,100 incoming rows in one file is more than one upload may hold, but each ledger is small.
    incoming, payments, labels = [], [], []
    for number in range(1, 22):
        ledger_id = f"B{number:02d}"
        payments.append(f"{ledger_id},{ledger_id}-P-0001,Larch Meadow Vets,LM-1,10.00,2023-01-01,")
        for row in range(1, 101):
            incoming.append(f"{ledger_id},{ledger_id}-I-{row:04d},Larch Meadow Vets,LM-{row},10.00,2023-02-01,")
            labels.append(f"{ledger_id},1,{ledger_id}-I-{row:04d},0,,legit_one_off,organization")
    write_benchmark(tmp_path / "synthetic", "train_", incoming, payments, labels)
    ledgers = load_benchmark_split("train", tmp_path)
    assert len(ledgers) == 21
    assert all(len(ledger.incoming) == 100 and len(ledger.payments) == 1 for ledger in ledgers)


def test_fit_shipped_matcher_uses_the_frozen_config_when_present(tmp_path):
    write_tiny_train_split(tmp_path)
    default_fit = fit_shipped_matcher(tmp_path)
    assert default_fit.frozen is False
    assert default_fit.config == DEFAULT_CONFIG
    assert default_fit.n_train_pairs == 3 * 6 * 10
    assert default_fit.n_train_positives == 3 * 2

    frozen_config = MatcherConfig(vendor_text="basic", amount_transform="relative",
                                  interactions=("ref_exact*amount_exact",), C=0.5)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "frozen_config.json").write_text(
        json.dumps({"matcher": frozen_config.to_dict(), "fuzzy_rule": DEFAULT_FUZZY_RULE.to_dict()}),
        encoding="utf-8",
    )
    frozen_fit = fit_shipped_matcher(tmp_path)
    assert frozen_fit.frozen is True
    assert frozen_fit.config == frozen_config
    assert frozen_fit.feature_names[-1] == "ref_exact*amount_exact"


@pytest.mark.skipif(
    not (matcher.DATA_DIR / "synthetic" / "train_incoming.csv").exists(),
    reason="the synthetic training split has not been generated (python synth.py)",
)
def test_fit_shipped_matcher_on_the_generated_training_split():
    ledgers = load_benchmark_split("train")
    fitted = fit_shipped_matcher()
    assert fitted.n_train_pairs == sum(len(ledger.incoming) * len(ledger.payments) for ledger in ledgers)
    assert fitted.n_train_positives == sum(len(valid) for ledger in ledgers for valid in ledger.truth.values())
    assert np.isfinite(fitted.classifier.coef_).all()

    first = ledgers[0]
    queue = fitted.rank(first.incoming, first.payments)
    assert sorted(queue["incoming_id"]) == sorted(first.incoming["row_id"])
    assert queue["rank"].tolist() == list(range(1, len(first.incoming) + 1))
