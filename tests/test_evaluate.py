"""Tests for evaluate.py.

The metric tests use queues written by hand. The stage tests run on a toy
benchmark that is written to a temporary folder, so they never read or write
the project's data/ and results/ folders. The held-out test split of the
project is never touched here: the only "test split" these tests load is the
toy one inside the temporary folder.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate
from evaluate import (
    BUDGETS,
    METHOD_NAMES,
    StageError,
    TrainingTable,
    average_precision,
    challenger_decision,
    config_rank,
    correct_flags,
    evaluate_folds,
    fuzzy_grid,
    neighbours,
    outcome_table,
    pick_examples,
    pool_method,
    pooled_queue,
    pooled_truth,
    queue_from_scores,
    run_test_stage,
    run_validate_stage,
    scenario_breakdown,
    score_budget,
    selection_score,
    sum_metrics,
    vendor_kind_slices,
)
from matcher import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    FuzzyRule,
    LearnedMatcher,
    MatcherConfig,
    exact_match_queue,
    fit_shipped_matcher,
    load_benchmark_split,
)

METRIC_KEYS = {"reviews_used", "correct", "false_flags", "missed", "precision", "recall"}
RECORD_KEYS = {"row_id", "vendor", "invoice_ref", "amount_cents", "invoice_date", "billing_period"}


# --------------------------------------------------------------------------
# Hand-written queues
# --------------------------------------------------------------------------

def queue(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """A queue frame from (incoming_id, payment_id, score) rows that are already in rank order."""
    return pd.DataFrame({
        "incoming_id": pd.Series([row[0] for row in rows], dtype="str"),
        "payment_id": pd.Series([row[1] for row in rows], dtype="str"),
        "score": np.array([row[2] for row in rows], dtype=np.float64),
        "rank": np.arange(1, len(rows) + 1, dtype=np.int64),
    })


# One ledger, six proposals. I1 and I3 are duplicates with one valid payment,
# I4 is a duplicate that was already paid twice (P4a or P4b), I6 is a duplicate
# that the method pairs with the wrong payment. I2 and I5 are legitimate.
TOY_TRUTH = {
    ("L1", "I1"): frozenset({"P1"}),
    ("L1", "I3"): frozenset({"P3"}),
    ("L1", "I4"): frozenset({"P4a", "P4b"}),
    ("L1", "I6"): frozenset({"P6"}),
    ("L1", "I9"): frozenset({"P9"}),  # never proposed by the method
}
TOY_QUEUE = [
    ("I1", "P1", 9.0),  # correct
    ("I2", "P7", 8.0),  # legitimate invoice: false flag
    ("I4", "P4b", 7.0),  # correct: the second of two valid payments
    ("I6", "P8", 6.0),  # duplicate shown with the wrong payment
    ("I3", "P3", 5.0),  # correct
    ("I5", "P2", 4.0),  # legitimate invoice: false flag
]


@pytest.fixture()
def toy_pool() -> pd.DataFrame:
    return pooled_queue([("L1", queue(TOY_QUEUE))])


def test_score_budget_counts_reviews_hits_and_misses(toy_pool):
    result = score_budget(toy_pool, TOY_TRUTH, known_duplicates=5, budget=5)
    assert result == {
        "reviews_used": 5,
        "correct": 3,
        "false_flags": 2,
        "missed": 2,
        "precision": 3 / 5,
        "recall": 3 / 5,
    }


def test_a_method_with_fewer_proposals_than_the_budget_uses_fewer_reviews(toy_pool):
    result = score_budget(toy_pool, TOY_TRUTH, known_duplicates=5, budget=20)
    assert result["reviews_used"] == 6
    assert result["correct"] == 3
    assert result["false_flags"] == 3
    assert result["missed"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.6)


def test_an_empty_queue_reviews_nothing_and_misses_everything():
    empty = pooled_queue([("L1", queue([]))])
    result = score_budget(empty, TOY_TRUTH, known_duplicates=5, budget=20)
    assert result == {"reviews_used": 0, "correct": 0, "false_flags": 0, "missed": 5,
                      "precision": None, "recall": 0.0}
    assert average_precision(empty, TOY_TRUTH, known_duplicates=5) == 0.0


def test_a_duplicate_shown_with_the_wrong_payment_is_a_false_flag_and_a_miss():
    truth = {("L1", "I6"): frozenset({"P6"})}
    wrong = pooled_queue([("L1", queue([("I6", "P8", 1.0)]))])
    right = pooled_queue([("L1", queue([("I6", "P6", 1.0)]))])
    assert score_budget(wrong, truth, known_duplicates=1, budget=5) == {
        "reviews_used": 1, "correct": 0, "false_flags": 1, "missed": 1, "precision": 0.0, "recall": 0.0}
    assert score_budget(right, truth, known_duplicates=1, budget=5) == {
        "reviews_used": 1, "correct": 1, "false_flags": 0, "missed": 0, "precision": 1.0, "recall": 1.0}


@pytest.mark.parametrize("payment_id, expected", [("P4a", 1), ("P4b", 1), ("P5", 0)])
def test_either_payment_of_an_invoice_paid_twice_counts(payment_id, expected):
    truth = {("L1", "I4"): frozenset({"P4a", "P4b"})}
    pool = pooled_queue([("L1", queue([("I4", payment_id, 1.0)]))])
    assert score_budget(pool, truth, known_duplicates=1, budget=5)["correct"] == expected


def test_the_same_incoming_id_in_another_ledger_is_a_different_invoice():
    truth = {("L1", "I1"): frozenset({"P1"})}
    pool = pooled_queue([("L2", queue([("I1", "P1", 1.0)]))])
    assert correct_flags(pool, truth).tolist() == [False]


def test_smaller_budgets_review_a_prefix_of_the_same_ranking(toy_pool):
    hits = correct_flags(toy_pool, TOY_TRUTH)
    assert hits.tolist() == [True, False, True, False, True, False]
    previous_correct = 0
    for budget in (1, 2, 3, 4, 5, 6, 10, 20):
        result = score_budget(toy_pool, TOY_TRUTH, known_duplicates=5, budget=budget)
        assert result["reviews_used"] == min(budget, len(toy_pool))
        assert result["correct"] == int(hits[:budget].sum())  # the first `budget` rows of one fixed ranking
        assert result["correct"] >= previous_correct
        previous_correct = result["correct"]


def test_pooled_queue_orders_by_score_across_ledgers():
    pool = pooled_queue([
        ("L1", queue([("I1", "P1", 3.0), ("I2", "P2", 1.0)])),
        ("L2", queue([("I1", "P9", 2.0)])),
    ])
    assert list(pool.columns) == ["ledger_id", "incoming_id", "payment_id", "score", "rank", "pooled_rank"]
    assert pool["ledger_id"].tolist() == ["L1", "L2", "L1"]
    assert pool["score"].tolist() == [3.0, 2.0, 1.0]
    assert pool["rank"].tolist() == [1, 1, 2]
    assert pool["pooled_rank"].tolist() == [1, 2, 3]


def test_pooled_ties_keep_ledger_order_then_rank_inside_the_ledger():
    first = queue([("I1", "P1", 5.0), ("I2", "P2", 5.0), ("I3", "P3", 1.0)])
    second = queue([("I1", "P1", 5.0), ("I2", "P2", 5.0)])
    pool = pooled_queue([("L1", first), ("L2", second)])
    order = list(zip(pool["ledger_id"], pool["incoming_id"]))
    assert order == [("L1", "I1"), ("L1", "I2"), ("L2", "I1"), ("L2", "I2"), ("L1", "I3")]

    again = pooled_queue([("L1", first), ("L2", second)])
    pd.testing.assert_frame_equal(pool, again)  # the same input always gives the same queue

    swapped = pooled_queue([("L2", second), ("L1", first)])  # ledger order decides ties, not the ledger name
    assert swapped["ledger_id"].tolist() == ["L2", "L2", "L1", "L1", "L1"]


def test_pooled_queue_accepts_a_ledger_without_proposals():
    pool = pooled_queue([("L1", queue([])), ("L2", queue([("I1", "P1", 2.0)]))])
    assert pool["ledger_id"].tolist() == ["L2"]
    assert pool["pooled_rank"].tolist() == [1]


def test_average_precision_rewards_duplicates_found_early(toy_pool):
    # Hits at positions 1, 3 and 5: precision there is 1/1, 2/3 and 3/5. Two duplicates are never found.
    expected = (1 / 1 + 2 / 3 + 3 / 5) / 5
    assert average_precision(toy_pool, TOY_TRUTH, known_duplicates=5) == pytest.approx(expected)

    hits_first = pooled_queue([("L1", queue([("I1", "P1", 3.0), ("I3", "P3", 2.0), ("I2", "P7", 1.0)]))])
    hits_last = pooled_queue([("L1", queue([("I2", "P7", 3.0), ("I1", "P1", 2.0), ("I3", "P3", 1.0)]))])
    truth = {("L1", "I1"): frozenset({"P1"}), ("L1", "I3"): frozenset({"P3"})}
    assert average_precision(hits_first, truth, 2) == pytest.approx(1.0)
    assert average_precision(hits_last, truth, 2) == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_sum_metrics_adds_counts_and_recomputes_the_rates():
    fold_one = score_budget(pooled_queue([("L1", queue(TOY_QUEUE))]), TOY_TRUTH, 5, 5)
    fold_two = score_budget(pooled_queue([("L1", queue([]))]), TOY_TRUTH, 5, 5)
    total = sum_metrics([fold_one, fold_two])
    assert total == {"reviews_used": 5, "correct": 3, "false_flags": 2, "missed": 7,
                     "precision": 3 / 5, "recall": 3 / 10}


# --------------------------------------------------------------------------
# Selection rules
# --------------------------------------------------------------------------

def fake_evaluation(correct_20: int, correct_10: int, correct_5: int, mean_ap: float) -> dict:
    """The part of an evaluation that the selection rule reads."""
    return {
        "summed": {"20": {"correct": correct_20}, "10": {"correct": correct_10}, "5": {"correct": correct_5}},
        "mean_average_precision": mean_ap,
    }


def test_selection_score_reads_budget_20_first_then_10_then_5_then_average_precision():
    ordered = [
        fake_evaluation(50, 20, 10, 0.10),
        fake_evaluation(49, 30, 15, 0.99),
        fake_evaluation(49, 29, 15, 0.99),
        fake_evaluation(49, 29, 14, 0.99),
        fake_evaluation(49, 29, 14, 0.80),
    ]
    scores = [selection_score(evaluation) for evaluation in ordered]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_average_precision_gaps_in_the_third_decimal_are_ties():
    assert selection_score(fake_evaluation(49, 30, 15, 0.8714)) == selection_score(fake_evaluation(49, 30, 15, 0.8748))


def test_equal_learned_configs_are_separated_by_simplicity():
    evaluation = fake_evaluation(49, 30, 15, 0.87)
    plain = replace(DEFAULT_CONFIG, C=1.0)
    with_terms = replace(DEFAULT_CONFIG, C=0.03, interactions=("ref_exact*amount_exact",))
    smaller_c = replace(DEFAULT_CONFIG, C=0.1)
    assert config_rank(plain, evaluation) > config_rank(with_terms, evaluation)  # fewer interactions first
    assert config_rank(smaller_c, evaluation) > config_rank(plain, evaluation)  # then the smaller C
    better = fake_evaluation(50, 30, 15, 0.87)
    assert config_rank(with_terms, better) > config_rank(plain, evaluation)  # but results come before simplicity


def test_fuzzy_grid_covers_the_whole_family_once():
    rules = fuzzy_grid()
    assert len(rules) == 4 * 3 * 2 * 8
    assert len(set(rules)) == len(rules)
    assert sorted({rule.vendor_threshold for rule in rules}) == [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    assert {rule.amount_rule for rule in rules} == {"exact", "within_100_cents", "within_1_percent"}
    assert {rule.rank_by for rule in rules} == {"vendor", "ref_then_vendor"}
    assert {rule.vendor_scorer for rule in rules} == {"ratio", "token_sort_ratio", "token_set_ratio", "WRatio"}


def test_neighbours_change_one_coordinate_and_include_the_config_itself():
    lines = neighbours(DEFAULT_CONFIG)
    assert list(lines) == list(evaluate.COORDINATES)
    assert len(lines["C and class_weight"]) == 6 * 2
    for candidates in lines.values():
        assert DEFAULT_CONFIG in candidates
        assert len(set(candidates)) == len(candidates)
    for candidate in lines["vendor_scorer"]:
        assert replace(candidate, vendor_scorer=DEFAULT_CONFIG.vendor_scorer) == DEFAULT_CONFIG
    assert () in [candidate.interactions for candidate in lines["interactions"]]


@pytest.mark.parametrize("lightgbm_20, lightgbm_10, triggered", [
    (52, 30, True),  # three more at budget 20, equal at budget 10
    (53, 31, True),
    (52, 29, False),  # worse at budget 10
    (51, 30, False),  # only two more at budget 20
    (40, 30, False),
])
def test_challenger_rule(lightgbm_20, lightgbm_10, triggered):
    logistic = {"20": {"correct": 49}, "10": {"correct": 30}}
    lightgbm = {"20": {"correct": lightgbm_20}, "10": {"correct": lightgbm_10}}
    decision = challenger_decision(logistic, lightgbm)
    assert decision["rule_triggered"] is triggered
    assert decision["gain_at_20"] == lightgbm_20 - 49
    assert decision["gain_at_10"] == lightgbm_10 - 30
    assert ("frozen anyway" in decision["decision"]) is triggered


# --------------------------------------------------------------------------
# A toy benchmark in a temporary folder
# --------------------------------------------------------------------------

BENCH_HEADER = "ledger_id,row_id,vendor,invoice_ref,amount,invoice_date,billing_period"
LABELS_HEADER = "ledger_id,fold,incoming_row_id,is_duplicate,true_payment_ids,scenario,vendor_kind"

# split -> (file prefix, year of the invoices, [(ledger_id, fold, word that makes its vendor names unique)])
TOY_SPLITS = {
    "train": ("train_", 2023, [("T01", 1, "Marrowbend"), ("T02", 1, "Tansy Row"),
                               ("T03", 1, "Sorrel Bridge"), ("T04", 1, "Glasswater")]),
    "validation": ("validation_", 2024, [("V01", 1, "Calder Wick"), ("V02", 1, "Elmstead"),
                                         ("V03", 2, "Pippin Lane"), ("V04", 2, "Hartwell")]),
    "test": ("test_", 2025, [("E01", 1, "Linden Court"), ("E02", 1, "Fennel Street")]),
    "challenge": ("", 2026, [("CH01", 1, "Quillon Park")]),
}


def toy_ledger_lines(ledger_id: str, fold: int, word: str, year: int) -> tuple[list[str], list[str], list[str]]:
    """CSV lines of one toy ledger: 12 payments, 8 incoming invoices, 3 known duplicates.

    I0001 repeats P0003 with the vendor and reference typed differently, I0003
    copies P0004 exactly, and I0007 repeats P0012 without any reference, which
    exact match cannot catch.
    """
    printer, caterer, hall, florist, tutor = (f"{word} Print Shop", f"{word} Catering", f"{word} Hall Trust",
                                              f"{word} Flowers", f"Ada {word}")
    payments = [
        (printer, "A-101", "120.00", "01-05", "01"), (printer, "A-102", "120.00", "02-05", "02"),
        (printer, "A-103", "120.00", "03-05", "03"), (caterer, "B-7001", "310.50", "02-11", ""),
        (caterer, "B-7002", "95.25", "03-02", ""), (hall, "", "800.00", "01-01", "01"),
        (hall, "", "800.00", "02-01", "02"), (hall, "", "800.00", "03-01", "03"),
        (florist, "D-55", "42.00", "03-09", "03"), (florist, "D-58", "61.00", "03-20", "03"),
        (tutor, "", "75.00", "03-12", ""), (tutor, "", "75.00", "03-19", ""),
    ]
    incoming = [
        (printer + " LLC", "INV A-103", "120.00", "03-05", "03", "P-0003", "dup_rekeyed", "organization"),
        (printer, "A-104", "120.00", "04-05", "04", "", "legit_recurring_next", "organization"),
        (caterer, "B-7001", "310.50", "02-11", "", "P-0004", "dup_exact_resubmission", "organization"),
        (hall, "", "800.00", "04-01", "04", "", "legit_recurring_next", "organization"),
        (florist, "D-61", "42.00", "04-02", "04", "", "legit_same_amount_repeat", "organization"),
        (f"{word} Tutoring Circle", "JH-1", "500.00", "04-03", "", "", "legit_new_vendor", "organization"),
        (tutor, "", "75.00", "03-19", "", "P-0012", "dup_statement_no_ref", "individual"),
        (tutor, "", "75.00", "03-26", "", "", "legit_same_amount_repeat", "individual"),
    ]

    def period(month: str) -> str:
        return f"{year}-{month}" if month else ""

    payment_lines = [
        f"{ledger_id},{ledger_id}-P-{number:04d},{vendor},{ref},{amount},{year}-{day},{period(month)}"
        for number, (vendor, ref, amount, day, month) in enumerate(payments, start=1)
    ]
    incoming_lines, label_lines = [], []
    for number, (vendor, ref, amount, day, month, paid, scenario, kind) in enumerate(incoming, start=1):
        row_id = f"{ledger_id}-I-{number:04d}"
        true_payment = f"{ledger_id}-{paid}" if paid else ""
        incoming_lines.append(f"{ledger_id},{row_id},{vendor},{ref},{amount},{year}-{day},{period(month)}")
        label_lines.append(f"{ledger_id},{fold},{row_id},{1 if paid else 0},{true_payment},{scenario},{kind}")
    return incoming_lines, payment_lines, label_lines


def write_toy_split(root: Path, split: str) -> None:
    """Write one toy split under root/data, in the layout load_benchmark_split expects."""
    prefix, year, ledgers = TOY_SPLITS[split]
    folder = root / "data" / ("challenge" if split == "challenge" else "synthetic")
    folder.mkdir(parents=True, exist_ok=True)
    incoming, payments, labels = [], [], []
    for ledger_id, fold, word in ledgers:
        new_incoming, new_payments, new_labels = toy_ledger_lines(ledger_id, fold, word, year)
        incoming += new_incoming
        payments += new_payments
        labels += new_labels
    (folder / f"{prefix}incoming.csv").write_text("\n".join([BENCH_HEADER] + incoming) + "\n", encoding="utf-8")
    (folder / f"{prefix}payments.csv").write_text("\n".join([BENCH_HEADER] + payments) + "\n", encoding="utf-8")
    (folder / f"{prefix}labels.csv").write_text("\n".join([LABELS_HEADER] + labels) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory) -> Path:
    """A project folder with toy train and validation splits (no test split, no results)."""
    root = tmp_path_factory.mktemp("toy_project")
    write_toy_split(root, "train")
    write_toy_split(root, "validation")
    return root


@pytest.fixture(scope="module")
def toy_train(toy_root):
    return load_benchmark_split("train", toy_root / "data")


@pytest.fixture(scope="module")
def toy_validation(toy_root):
    return load_benchmark_split("validation", toy_root / "data")


@pytest.fixture(scope="module")
def toy_model(toy_train) -> LearnedMatcher:
    return LearnedMatcher.fit(toy_train)


# --------------------------------------------------------------------------
# Running methods on folds
# --------------------------------------------------------------------------

def test_evaluate_folds_pools_each_fold_separately(toy_validation):
    evaluation = evaluate_folds(toy_validation, exact_match_queue)
    assert list(evaluation["folds"]) == ["1", "2"]
    for fold in evaluation["folds"].values():
        assert fold["known_duplicates"] == 6  # two ledgers with three duplicates each
        assert fold["eligible"] == 4  # exact match proposes only the two duplicates that kept their reference
        assert set(fold["results"]) == {"5", "10", "20"}
        assert fold["results"]["20"] == {"reviews_used": 4, "correct": 4, "false_flags": 0, "missed": 2,
                                         "precision": 1.0, "recall": 4 / 6}
    assert evaluation["eligible"] == 8
    assert evaluation["summed"]["20"]["correct"] == 8
    assert evaluation["summed"]["20"]["missed"] == 4
    assert evaluation["summed"]["20"]["recall"] == pytest.approx(8 / 12)
    assert evaluation["summed"]["5"]["reviews_used"] == 8  # four per fold, fewer than the budget of five


def test_the_learned_queue_proposes_a_payment_for_every_invoice(toy_validation, toy_model):
    pool = pool_method(toy_validation, toy_model.rank)
    assert len(pool) == sum(len(ledger.incoming) for ledger in toy_validation)
    assert pool["score"].is_monotonic_decreasing
    assert pool["pooled_rank"].tolist() == list(range(1, len(pool) + 1))


def test_training_table_gives_the_same_model_as_a_direct_fit(toy_train):
    table = TrainingTable(toy_train)
    for config in (DEFAULT_CONFIG, replace(DEFAULT_CONFIG, C=0.1, class_weight="balanced"),
                   replace(DEFAULT_CONFIG, vendor_text="basic", interactions=("ref_exact*amount_exact",))):
        cached = table.fit(config)
        direct = LearnedMatcher.fit(toy_train, config)
        assert cached.config == config
        np.testing.assert_array_equal(cached.classifier.coef_, direct.classifier.coef_)
        np.testing.assert_array_equal(cached.scaler.mean_, direct.scaler.mean_)
        assert cached.feature_names == direct.feature_names


def test_queue_from_scores_reproduces_the_learned_queue(toy_validation, toy_model):
    for ledger in toy_validation:
        scores = toy_model.score_matrix(ledger.incoming, ledger.payments)
        rebuilt = queue_from_scores(ledger.incoming, ledger.payments, scores)
        pd.testing.assert_frame_equal(rebuilt, toy_model.rank(ledger.incoming, ledger.payments))


def test_queue_from_scores_breaks_ties_by_newest_payment_then_file_order(toy_validation):
    ledger = toy_validation[0]
    incoming, payments = ledger.incoming.head(2), ledger.payments.head(4)  # dated 01-05, 02-05, 03-05, 02-11
    scores = np.array([[1.0, 1.0, 1.0, 1.0],
                       [5.0, 5.0, 0.0, 0.0]])
    result = queue_from_scores(incoming, payments, scores)
    assert result["incoming_id"].tolist() == ["V01-I-0002", "V01-I-0001"]  # the higher score first
    assert result["payment_id"].tolist() == ["V01-P-0002", "V01-P-0003"]  # the most recent of the tied payments
    assert result["rank"].tolist() == [1, 2]

    same_day = payments.assign(invoice_date=payments["invoice_date"].iloc[0])
    assert queue_from_scores(incoming, same_day, scores)["payment_id"].tolist() == ["V01-P-0001", "V01-P-0001"]
    level = queue_from_scores(incoming, payments, np.ones((2, 4)))
    assert level["incoming_id"].tolist() == ["V01-I-0001", "V01-I-0002"]  # equal scores keep file order


# --------------------------------------------------------------------------
# Error analysis and examples
# --------------------------------------------------------------------------

def hand_pool(ledger_id: str, rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    """A pooled queue for one toy ledger from (incoming number, payment number, score) rows."""
    return pooled_queue([(ledger_id, queue([
        (f"{ledger_id}-I-{incoming:04d}", f"{ledger_id}-P-{payment:04d}", score) for incoming, payment, score in rows
    ]))])


def test_outcome_table_marks_what_was_reviewed_and_what_was_right(toy_validation):
    ledgers = toy_validation[:1]
    # Reviewed with a budget of 3: a caught duplicate, a legitimate invoice, a duplicate with the wrong payment.
    pool = hand_pool("V01", [(1, 3, 9.0), (2, 2, 8.0), (7, 11, 7.0), (3, 4, 6.0)])
    outcomes = outcome_table(ledgers, pool, budget=3)
    assert len(outcomes) == 8
    assert outcomes["reviewed"].tolist() == [True, True, False, False, False, False, True, False]
    assert outcomes["correct"].tolist() == [True, False, False, False, False, False, False, False]
    assert outcomes["is_duplicate"].sum() == 3

    breakdown = scenario_breakdown(outcomes)
    assert breakdown["duplicates"] == {
        "dup_exact_resubmission": {"known": 1, "caught": 0},
        "dup_rekeyed": {"known": 1, "caught": 1},
        "dup_statement_no_ref": {"known": 1, "caught": 0},
    }
    assert breakdown["false_flags"] == {"dup_statement_no_ref": 1, "legit_recurring_next": 1}

    slices = vendor_kind_slices({"learned": outcomes})
    assert slices == [
        {"group": "organization", "legit_invoices": 4, "known_duplicates": 2,
         "methods": {"learned": {"legit_flagged": 1, "duplicates_caught": 1}}},
        {"group": "individual", "legit_invoices": 1, "known_duplicates": 1,
         "methods": {"learned": {"legit_flagged": 0, "duplicates_caught": 0}}},
    ]


def test_pick_examples_takes_the_first_of_each_kind_in_pooled_order(toy_validation):
    ledgers = toy_validation[:1]
    exact_pool = pool_method(ledgers, exact_match_queue)  # catches I0001 and I0003, not the copy without reference
    learned_pool = hand_pool("V01", [
        (1, 3, 9.0),  # caught, and exact match caught it too
        (7, 12, 8.0),  # caught, exact match missed it: the caught_duplicate example
        (5, 9, 7.0),  # a legitimate invoice: the false_flag example
        (3, 4, 6.0),  # the right payment, but below a budget of 3: the missed_duplicate example
    ])
    examples = pick_examples(ledgers, learned_pool, exact_pool, budget=3)
    assert [example["kind"] for example in examples] == ["missed_duplicate", "false_flag", "caught_duplicate"]
    missed, false_flag, caught = examples

    assert (missed["rank"], missed["incoming"]["row_id"]) == (4, "V01-I-0003")
    assert missed["proposed_payment"]["row_id"] == missed["true_payment"]["row_id"] == "V01-P-0004"
    assert missed["likely_cause"].startswith("The right prior payment was found, but the pair ranked 4, "
                                             "below the review budget of 3.")

    assert (false_flag["rank"], false_flag["incoming"]["row_id"]) == (3, "V01-I-0005")
    assert false_flag["true_payment"] is None
    assert false_flag["likely_cause"].startswith("A legitimate invoice that resembles a prior payment: ")
    assert "same amount" in false_flag["likely_cause"]

    assert (caught["rank"], caught["incoming"]["row_id"]) == (2, "V01-I-0007")
    assert "invoice reference missing on both" in caught["likely_cause"]

    for example in examples:
        assert example["method"] == "learned" and example["budget"] == 3
        assert set(example["incoming"]) == RECORD_KEYS
        assert example["incoming"]["invoice_date"].startswith("2024-")
        assert isinstance(example["incoming"]["amount_cents"], int)
        assert [comparison["field"] for comparison in example["comparisons"]] == [
            "vendor", "invoice_ref", "amount", "invoice_date", "billing_period"]
        assert all(set(comparison) == {"field", "label", "status", "evidence"}
                   for comparison in example["comparisons"])


def test_pick_examples_reports_a_wrong_payment_as_both_kinds_and_skips_absent_kinds(toy_validation):
    ledgers = toy_validation[:1]
    exact_pool = pool_method(ledgers, exact_match_queue)
    wrong_payment = hand_pool("V01", [(1, 1, 9.0)])  # the duplicate of P-0003, shown with P-0001
    examples = pick_examples(ledgers, wrong_payment, exact_pool, budget=3)
    assert [example["kind"] for example in examples] == ["missed_duplicate", "false_flag"]
    assert all(example["proposed_payment"]["row_id"] == "V01-P-0001" for example in examples)
    assert all(example["true_payment"]["row_id"] == "V01-P-0003" for example in examples)
    assert examples[0]["likely_cause"].startswith("The model paired this invoice with a different prior payment")
    assert examples[1]["likely_cause"].startswith("A real duplicate, but shown with the wrong prior payment")

    flawless = hand_pool("V01", [(1, 3, 9.0), (3, 4, 8.0), (7, 12, 7.0)])
    assert [example["kind"] for example in pick_examples(ledgers, flawless, exact_pool, budget=3)] == [
        "caught_duplicate"]


# --------------------------------------------------------------------------
# The stages, end to end on the toy benchmark
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def staged_root(tmp_path_factory) -> Path:
    """A toy project on which the validate stage has run (with the LightGBM challenger)."""
    pytest.importorskip("lightgbm")
    root = tmp_path_factory.mktemp("staged_project")
    write_toy_split(root, "train")
    write_toy_split(root, "validation")
    run_validate_stage(root, challenger=True)
    return root


def read_result(root: Path, name: str) -> dict:
    return json.loads((root / "results" / name).read_text(encoding="utf-8"))


def test_test_stage_refuses_to_run_without_a_frozen_config(tmp_path):
    with pytest.raises(StageError) as error:
        run_test_stage(tmp_path)
    assert "frozen_config.json" in str(error.value)
    assert "python evaluate.py --stage validate" in str(error.value)
    assert not (tmp_path / "results").exists()  # nothing was written


def test_test_stage_refuses_to_run_without_the_test_files(staged_root):
    with pytest.raises(StageError) as error:
        run_test_stage(staged_root)
    assert "held-out test split" in str(error.value)
    assert "python synth.py --splits test" in str(error.value)
    assert isinstance(error.value.__cause__, FileNotFoundError)  # the missing file is named in the cause
    assert not (staged_root / "results" / "benchmark.json").exists()


def test_validate_stage_writes_the_frozen_config(staged_root, toy_validation):
    frozen = read_result(staged_root, "frozen_config.json")
    assert list(frozen) == ["matcher", "fuzzy_rule", "selection", "challenger"]
    MatcherConfig.from_dict(frozen["matcher"])  # both settings can be read back
    FuzzyRule.from_dict(frozen["fuzzy_rule"])
    assert frozen["selection"]["criterion"] == evaluate.SELECTION_CRITERION
    summed = frozen["selection"]["validation"]
    assert list(summed) == list(METHOD_NAMES)
    for key in METHOD_NAMES:
        assert set(summed[key]) == {"5", "10", "20"}
        assert all(set(result) == METRIC_KEYS for result in summed[key].values())
    # The exact-match numbers in the file are the ones a fresh run gives.
    assert summed["exact"] == evaluate_folds(toy_validation, exact_match_queue)["summed"]
    assert frozen["challenger"]["evaluated"] is True
    assert isinstance(frozen["challenger"]["rule_triggered"], bool)
    assert len(frozen["challenger"]["candidates"]) == len(evaluate.CHALLENGER_SETTINGS)


def test_validate_stage_writes_validation_results_and_a_model_card(staged_root):
    validation = read_result(staged_root, "validation.json")
    assert validation["budgets"] == list(BUDGETS)
    assert validation["datasets"]["train"] == {
        "ledgers": 4, "incoming": 32, "prior_payments": 48, "known_duplicates": 12,
        "date_start": "2023-01-01", "date_end": "2023-04-05"}
    assert validation["datasets"]["validation"]["known_duplicates"] == 12
    assert list(validation["methods"]) == list(METHOD_NAMES)
    assert list(validation["methods"]["learned"]["folds"]) == ["1", "2"]
    assert len(validation["fuzzy_search"]) == 192
    assert validation["learned_search"][0]["sweep"] == 1
    assert set(validation["error_analysis"]) == set(METHOD_NAMES)
    assert [part["group"] for part in validation["slices"]["vendor_kind"]] == ["organization", "individual"]

    frozen = read_result(staged_root, "frozen_config.json")
    card = read_result(staged_root, "model_card.json")
    shipped = fit_shipped_matcher(staged_root)
    assert shipped.frozen is True
    assert card["config"] == frozen["matcher"] == shipped.config.to_dict()
    assert card["feature_names"] == shipped.feature_names
    assert (card["n_train_pairs"], card["n_train_positives"]) == (4 * 8 * 12, 12)
    assert card["training"]["ledger_ids"] == ["T01", "T02", "T03", "T04"]
    np.testing.assert_allclose(card["coefficients"], shipped.classifier.coef_[0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(card["intercept"], shipped.classifier.intercept_[0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(card["scaler_means"], shipped.scaler.mean_, rtol=0, atol=1e-8)
    np.testing.assert_allclose(card["scaler_scales"], shipped.scaler.scale_, rtol=0, atol=1e-8)


def test_validate_stage_is_deterministic_and_writes_no_timestamps(staged_root, tmp_path, capsys):
    write_toy_split(tmp_path, "train")
    write_toy_split(tmp_path, "validation")
    run_validate_stage(tmp_path, challenger=True)
    printed = capsys.readouterr().out
    for name in ("frozen_config.json", "validation.json", "model_card.json"):
        first = (staged_root / "results" / name).read_bytes()
        assert first == (tmp_path / "results" / name).read_bytes()
        assert b"timestamp" not in first and b"generated_at" not in first
    assert "Validation: 4 ledgers in 2 folds, 12 known duplicates" in printed
    assert "Print Shop" not in printed and "V01-I-" not in printed  # counts only, never ledger rows


def test_test_stage_writes_the_benchmark_in_the_agreed_shape(staged_root, tmp_path, capsys):
    # A copy of the staged toy project, plus a toy test split and a toy challenge set.
    for name in ("frozen_config.json", "validation.json", "model_card.json"):
        (tmp_path / "results").mkdir(exist_ok=True)
        (tmp_path / "results" / name).write_bytes((staged_root / "results" / name).read_bytes())
    for split in ("train", "test", "challenge"):
        write_toy_split(tmp_path, split)
    frozen_before = (tmp_path / "results" / "frozen_config.json").read_bytes()

    report = run_test_stage(tmp_path)
    printed = capsys.readouterr().out
    assert report == read_result(tmp_path, "benchmark.json")
    assert (tmp_path / "results" / "frozen_config.json").read_bytes() == frozen_before  # nothing was re-tuned

    assert list(report) == ["label", "split", "command", "dataset", "budgets", "default_budget", "methods",
                            "examples", "slices", "challenge", "model", "fuzzy_rule", "validation_summary"]
    assert (report["label"], report["split"]) == ("Synthetic benchmark", "test")
    assert report["command"] == "python evaluate.py --stage test"
    assert report["dataset"] == {"ledgers": 2, "incoming": 16, "prior_payments": 24, "known_duplicates": 6,
                                 "date_start": "2025-01-01", "date_end": "2025-04-05"}
    assert (report["budgets"], report["default_budget"]) == ([5, 10, 20], 20)

    assert [method["key"] for method in report["methods"]] == ["exact", "fuzzy", "learned"]
    assert [method["name"] for method in report["methods"]] == ["Exact match", "Fuzzy rule", "Learned model"]
    for method in report["methods"]:
        assert set(method) == {"key", "name", "definition", "eligible", "results"}
        assert method["definition"].endswith(".")
        assert list(method["results"]) == ["5", "10", "20"]
        for result in method["results"].values():
            assert set(result) == METRIC_KEYS
            assert result["false_flags"] == result["reviews_used"] - result["correct"]
            assert result["missed"] == 6 - result["correct"]
    exact, _fuzzy, learned = report["methods"]
    assert exact["results"]["20"] == {"reviews_used": 4, "correct": 4, "false_flags": 0, "missed": 2,
                                      "precision": 1.0, "recall": 4 / 6}
    assert learned["eligible"] == 16

    for example in report["examples"]:
        assert set(example) == {"method", "kind", "budget", "rank", "incoming", "proposed_payment",
                                "true_payment", "comparisons", "likely_cause"}
        assert example["kind"] in ("missed_duplicate", "false_flag", "caught_duplicate")
        assert example["budget"] == 20
    kinds = [example["kind"] for example in report["examples"]]
    assert len(kinds) == len(set(kinds))

    slices = report["slices"]["vendor_kind"]
    assert [part["group"] for part in slices] == ["organization", "individual"]
    assert [part["legit_invoices"] for part in slices] == [8, 2]
    assert [part["known_duplicates"] for part in slices] == [4, 2]
    for part in slices:
        assert list(part["methods"]) == ["exact", "fuzzy", "learned"]
        assert all(set(counts) == {"legit_flagged", "duplicates_caught"} for counts in part["methods"].values())
    assert sum(part["methods"]["exact"]["duplicates_caught"] for part in slices) == 4

    challenge = report["challenge"]
    assert challenge["budget"] == challenge["dataset"]["known_duplicates"] == 3
    assert challenge["note"] == "hand-authored challenge set, reported separately"
    assert [method["key"] for method in challenge["methods"]] == ["exact", "fuzzy", "learned"]
    assert all(list(method["results"]) == ["3"] for method in challenge["methods"])

    frozen = read_result(tmp_path, "frozen_config.json")
    assert report["model"] == {"type": "LogisticRegression",
                               "features": fit_shipped_matcher(tmp_path).feature_names,
                               "config": frozen["matcher"]}
    assert report["fuzzy_rule"] == frozen["fuzzy_rule"]
    assert report["validation_summary"] == {key: frozen["selection"]["validation"][key]["20"]
                                            for key in METHOD_NAMES}
    assert "Held-out test split: 2 ledgers, 6 known duplicates" in printed
    assert "Print Shop" not in printed and "E01-I-" not in printed


def test_test_stage_reports_no_challenge_when_the_folder_is_absent(staged_root, tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "frozen_config.json").write_bytes(
        (staged_root / "results" / "frozen_config.json").read_bytes())
    write_toy_split(tmp_path, "train")
    write_toy_split(tmp_path, "test")
    assert run_test_stage(tmp_path)["challenge"] is None


def test_main_reports_a_refused_stage_without_a_traceback(tmp_path, capsys):
    assert evaluate.main(["--stage", "test"], root=tmp_path) == 2
    assert "frozen_config.json" in capsys.readouterr().err


def test_challenger_flag_is_refused_for_the_test_stage():
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--stage", "test", "--challenger"])
    assert evaluate.parse_args(["--stage", "validate", "--challenger"]).challenger is True


# --------------------------------------------------------------------------
# Guards on the source of evaluate.py
# --------------------------------------------------------------------------

def split_loads() -> list[tuple[str, object]]:
    """(function name, split argument) for every load_benchmark_split call inside a function of evaluate.py."""
    tree = ast.parse(Path(evaluate.__file__).read_text(encoding="utf-8"))
    loads = []
    for function in tree.body:
        if isinstance(function, (ast.FunctionDef, ast.ClassDef)):
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_benchmark_split":
                    split = node.args[0]
                    loads.append((function.name, split.value if isinstance(split, ast.Constant) else None))
    return loads


def test_only_the_test_stage_loads_the_test_split():
    source = Path(evaluate.__file__).read_text(encoding="utf-8")
    loads = split_loads()
    assert len(loads) == source.count("load_benchmark_split(")  # no call at module level, none overlooked
    assert all(split is not None for _function, split in loads)  # every split is written out as a literal
    assert ("run_test_stage", "test") in loads
    assert {function for function, split in loads if split in ("test", "challenge")} == {"run_test_stage"}
    assert {split for function, split in loads if function != "run_test_stage"} <= {"train", "validation"}
    for file_name_part in ("test_incoming", "test_payments", "test_labels", '"test_', "'test_"):
        assert file_name_part not in source  # no test file name is spelled out either
    assert "import synth" not in source and "from synth" not in source  # evaluate.py cannot generate data


def test_the_test_stage_calls_nothing_that_tunes():
    tree = ast.parse(Path(evaluate.__file__).read_text(encoding="utf-8"))
    tuning = {"search_fuzzy_rule", "search_learned_config", "run_challenger", "fit_challenger", "TrainingTable",
              "run_validate_stage", "fuzzy_grid", "neighbours", "selection_score"}
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in (
                "run_test_stage", "benchmark_body", "challenge_report"):
            names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
            assert not names & tuning, f"{function.name} uses {names & tuning}"


# --------------------------------------------------------------------------
# The project's own frozen results (skipped until the validate stage has run)
# --------------------------------------------------------------------------

RESULTS = PROJECT_ROOT / "results"
needs_frozen_results = pytest.mark.skipif(
    not all((RESULTS / name).exists() for name in ("frozen_config.json", "model_card.json", "validation.json")),
    reason="results/ is written by: python evaluate.py --stage validate",
)


@pytest.fixture(scope="module")
def shipped() -> LearnedMatcher:
    return fit_shipped_matcher()


@needs_frozen_results
def test_the_shipped_model_is_the_frozen_model(shipped):
    frozen = read_result(PROJECT_ROOT, "frozen_config.json")
    card = read_result(PROJECT_ROOT, "model_card.json")
    assert shipped.frozen is True
    assert shipped.config == MatcherConfig.from_dict(frozen["matcher"])
    assert card["config"] == frozen["matcher"]
    assert card["feature_names"] == shipped.feature_names
    assert (card["n_train_pairs"], card["n_train_positives"]) == (shipped.n_train_pairs, shipped.n_train_positives)
    np.testing.assert_allclose(shipped.classifier.coef_[0], card["coefficients"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(shipped.classifier.intercept_[0], card["intercept"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(shipped.scaler.mean_, card["scaler_means"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(shipped.scaler.scale_, card["scaler_scales"], rtol=0, atol=1e-8)


@needs_frozen_results
def test_the_recorded_validation_numbers_can_be_reproduced(shipped):
    frozen = read_result(PROJECT_ROOT, "frozen_config.json")
    recorded = read_result(PROJECT_ROOT, "validation.json")
    validation = load_benchmark_split("validation")
    rankers = evaluate.method_rankers(shipped, FuzzyRule.from_dict(frozen["fuzzy_rule"]))
    for key, rank in rankers.items():
        fresh = evaluate_folds(validation, rank)
        assert fresh["summed"] == frozen["selection"]["validation"][key]
        assert fresh["summed"] == recorded["methods"][key]["summed"]
        assert fresh["eligible"] == recorded["methods"][key]["eligible"]
    assert recorded["frozen"] == {"matcher": frozen["matcher"], "fuzzy_rule": frozen["fuzzy_rule"]}
    assert len(pooled_truth(validation)) == recorded["datasets"]["validation"]["known_duplicates"]
