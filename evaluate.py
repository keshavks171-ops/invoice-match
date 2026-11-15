"""evaluate.py - the benchmark of Invoice Match.

Three methods build a review queue for the same ledgers: exact match, the
fuzzy rule and the learned model (all in matcher.py). This script measures how
many known duplicate bills each one puts in front of the treasurer when only a
few invoices can be reviewed.

  python evaluate.py --stage validate [--challenger]
      Fit candidates on the train split, compare them on the three validation
      folds, pick the settings and write results/frozen_config.json,
      results/validation.json and results/model_card.json.
  python evaluate.py --stage test
      Fit the frozen model on the train split and score the three methods once
      on the held-out test split. Writes results/benchmark.json. Tunes nothing.
  python evaluate.py --stage all
      The validate stage, then the test stage.

Design choices worth knowing:
  * Train, validation and test hold different vendors and different dates
    (train is the earliest, test the latest). The test split is read by one
    function only, run_test_stage, and nothing is chosen there.
  * A method proposes one prior payment per incoming invoice. The proposals of
    a fold's six ledgers are pooled into one queue and only its top is reviewed
    (20, 10 or 5 invoices). Precision and recall are counted on the reviewed
    part. Accuracy would say little: about 1 invoice in 18 is a duplicate, so
    "flag nothing" is 94% accurate and useless.
  * The rule for picking settings (SELECTION_CRITERION) was written down before
    any candidate was scored. The learned model and the fuzzy rule are tuned
    with the same rule on the same validation folds, so the baseline gets a
    fair shot.
  * Only counts are printed. No ledger row reaches the screen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from matcher import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    VENDOR_SCORERS,
    BenchLedger,
    FieldComparison,
    FuzzyRule,
    LearnedMatcher,
    MatcherConfig,
    compare_fields,
    exact_match_queue,
    fit_shipped_matcher,
    fuzzy_rule_queue,
    load_benchmark_split,
    training_pairs,
)

BUDGETS = (5, 10, 20)
DEFAULT_BUDGET = 20
METHOD_NAMES = {"exact": "Exact match", "fuzzy": "Fuzzy rule", "learned": "Learned model"}
VENDOR_KINDS = ("organization", "individual")
POOLED_COLUMNS = ["ledger_id", "incoming_id", "payment_id", "score", "rank", "pooled_rank"]

SELECTION_CRITERION = (
    "Most correct matches at a review budget of 20, summed over the three validation folds. "
    "Ties go to the sum at budget 10, then the sum at budget 5, then the mean average precision "
    "over the folds (compared to 2 decimals), then the simpler setting: for the learned model "
    "fewer interaction terms and then a smaller C, for the fuzzy rule the one listed first in its grid."
)
# Average precision differs by about 0.02 from one validation fold to the next,
# so a gap in the third decimal is noise and must not outvote a simpler model.
AP_DECIMALS = 2

# A rank function takes (incoming frame, payments frame) and returns a queue frame.
RankFunction = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
# The answer key of a pool of ledgers: (ledger_id, incoming row_id) -> valid payment row_ids.
Truth = dict[tuple[str, str], frozenset[str]]


class StageError(RuntimeError):
    """A stage cannot run. The message says what is missing and how to get it."""


# --------------------------------------------------------------------------
# The pooled review queue and its metrics
# --------------------------------------------------------------------------

def pooled_truth(ledgers: Sequence[BenchLedger]) -> Truth:
    """The answer key of several ledgers in one dictionary. Only duplicates appear as keys."""
    return {(ledger.ledger_id, incoming_id): valid
            for ledger in ledgers for incoming_id, valid in ledger.truth.items()}


def pooled_queue(queues: Sequence[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Merge the queues of several ledgers into one review queue.

    queues holds one (ledger_id, queue frame) pair per ledger, in ledger order.
    The pooled queue is sorted by score, highest first. The sort is stable, so
    equal scores keep ledger order and then the rank inside the ledger.
    pooled_rank counts from 1.
    """
    pooled = pd.concat([queue.assign(ledger_id=ledger_id) for ledger_id, queue in queues], ignore_index=True)
    order = np.argsort(-pooled["score"].to_numpy(dtype=np.float64), kind="stable")
    pooled = pooled.iloc[order].reset_index(drop=True)
    pooled["pooled_rank"] = np.arange(1, len(pooled) + 1, dtype=np.int64)
    return pooled[POOLED_COLUMNS]


def correct_flags(pooled: pd.DataFrame, truth: Truth) -> np.ndarray:
    """True for every proposal whose payment is a valid earlier copy of its invoice.

    When an invoice was already paid twice, either payment counts.
    """
    flags = [payment_id in truth.get((ledger_id, incoming_id), frozenset())
             for ledger_id, incoming_id, payment_id
             in zip(pooled["ledger_id"], pooled["incoming_id"], pooled["payment_id"])]
    return np.array(flags, dtype=bool)


def metrics(reviews_used: int, correct: int, known_duplicates: int) -> dict:
    """The six numbers reported for one method at one budget.

    A duplicate shown with the wrong prior payment is a false flag (the
    reviewer saw misleading evidence) and also a miss (the duplicate was not
    found). Precision is None when nothing was reviewed, recall is None when
    the ledgers hold no known duplicate.
    """
    return {
        "reviews_used": reviews_used,
        "correct": correct,
        "false_flags": reviews_used - correct,
        "missed": known_duplicates - correct,
        "precision": correct / reviews_used if reviews_used else None,
        "recall": correct / known_duplicates if known_duplicates else None,
    }


def score_budget(pooled: pd.DataFrame, truth: Truth, known_duplicates: int, budget: int) -> dict:
    """Review the first `budget` proposals of a pooled queue and count the outcome.

    A method with fewer proposals than the budget uses fewer reviews. Smaller
    budgets look at a prefix of the same ranking.
    """
    reviewed = pooled.head(budget)
    return metrics(len(reviewed), int(correct_flags(reviewed, truth).sum()), known_duplicates)


def average_precision(pooled: pd.DataFrame, truth: Truth, known_duplicates: int) -> float:
    """Average precision of a pooled queue, used only to break ties between settings.

    For every duplicate that is proposed with a valid payment, take the
    precision of the queue down to that position; a duplicate that is never
    found adds 0. The result is the mean over all known duplicates.
    """
    hits = correct_flags(pooled, truth)
    precision_down_to = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision_down_to[hits].sum() / known_duplicates) if known_duplicates else 0.0


def sum_metrics(parts: Sequence[dict]) -> dict:
    """Add up the counts of several folds. Precision and recall are recomputed from the sums."""
    correct = sum(part["correct"] for part in parts)
    missed = sum(part["missed"] for part in parts)
    return metrics(sum(part["reviews_used"] for part in parts), correct, correct + missed)


# --------------------------------------------------------------------------
# Running a method on the folds of a split
# --------------------------------------------------------------------------

def split_folds(ledgers: Sequence[BenchLedger]) -> dict[int, list[BenchLedger]]:
    """fold number -> the ledgers of that fold, in file order."""
    folds: dict[int, list[BenchLedger]] = {}
    for ledger in ledgers:
        folds.setdefault(ledger.fold, []).append(ledger)
    return folds


def pool_method(ledgers: Sequence[BenchLedger], rank: RankFunction) -> pd.DataFrame:
    """Run one method inside every ledger (never across ledgers) and pool its proposals."""
    return pooled_queue([(ledger.ledger_id, rank(ledger.incoming, ledger.payments)) for ledger in ledgers])


def pool_results(ledgers: Sequence[BenchLedger], pooled: pd.DataFrame) -> dict:
    """The metrics of one pooled queue at every budget."""
    truth = pooled_truth(ledgers)
    return {
        "eligible": len(pooled),
        "known_duplicates": len(truth),
        "average_precision": average_precision(pooled, truth, len(truth)),
        "results": {str(budget): score_budget(pooled, truth, len(truth), budget) for budget in BUDGETS},
    }


def fold_pools(ledgers: Sequence[BenchLedger], rank: RankFunction) -> list[tuple[list[BenchLedger], pd.DataFrame]]:
    """One (ledgers of the fold, pooled queue of the method) pair per fold."""
    return [(fold_ledgers, pool_method(fold_ledgers, rank)) for fold_ledgers in split_folds(ledgers).values()]


def summarize_folds(pools: Sequence[tuple[list[BenchLedger], pd.DataFrame]]) -> dict:
    """The metrics of one method on every fold, and summed over the folds."""
    folds = {str(fold_ledgers[0].fold): pool_results(fold_ledgers, pooled) for fold_ledgers, pooled in pools}
    return {
        "eligible": sum(fold["eligible"] for fold in folds.values()),
        "mean_average_precision": float(np.mean([fold["average_precision"] for fold in folds.values()])),
        "summed": {str(budget): sum_metrics([fold["results"][str(budget)] for fold in folds.values()])
                   for budget in BUDGETS},
        "folds": folds,
    }


def evaluate_folds(ledgers: Sequence[BenchLedger], rank: RankFunction) -> dict:
    """Run one method on a split and return its metrics per fold and summed."""
    return summarize_folds(fold_pools(ledgers, rank))


def selection_score(evaluation: dict) -> tuple[int, int, int, float]:
    """SELECTION_CRITERION as a tuple that Python compares from left to right. Larger is better."""
    summed = evaluation["summed"]
    return (summed["20"]["correct"], summed["10"]["correct"], summed["5"]["correct"],
            round(evaluation["mean_average_precision"], AP_DECIMALS))


def search_entry(setting: dict, evaluation: dict) -> dict:
    """One compact line of a search log: the setting and the numbers it was judged by."""
    return {
        "setting": setting,
        "correct": {str(budget): evaluation["summed"][str(budget)]["correct"] for budget in BUDGETS},
        "eligible": evaluation["eligible"],
        "mean_average_precision": round(evaluation["mean_average_precision"], 4),
    }


# --------------------------------------------------------------------------
# Tuning the fuzzy rule (validation only)
# --------------------------------------------------------------------------

FUZZY_THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
FUZZY_AMOUNT_RULES = ("exact", "within_100_cents", "within_1_percent")
FUZZY_RANKINGS = ("vendor", "ref_then_vendor")


def fuzzy_grid() -> list[FuzzyRule]:
    """Every fuzzy rule that is tried: 4 scorers x 3 amount rules x 2 rankings x 8 thresholds.

    Plainer and stricter rules come first, so a tie goes to them.
    """
    return [FuzzyRule(threshold, amount_rule, rank_by, scorer)
            for scorer in VENDOR_SCORERS
            for amount_rule in FUZZY_AMOUNT_RULES
            for rank_by in FUZZY_RANKINGS
            for threshold in FUZZY_THRESHOLDS]


def search_fuzzy_rule(validation: Sequence[BenchLedger]) -> tuple[FuzzyRule, list[dict]]:
    """Try the whole grid on the validation folds. Returns (best rule, search log)."""
    rules = fuzzy_grid()
    evaluations = [evaluate_folds(validation, partial(fuzzy_rule_queue, rule=rule)) for rule in rules]
    scores = [selection_score(evaluation) for evaluation in evaluations]
    best = scores.index(max(scores))  # index() returns the first of equal scores
    log = [search_entry(rule.to_dict(), evaluation) for rule, evaluation in zip(rules, evaluations)]
    return rules[best], log


# --------------------------------------------------------------------------
# Tuning the learned model (validation only)
# --------------------------------------------------------------------------

C_OPTIONS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CLASS_WEIGHT_OPTIONS = (None, "balanced")
VENDOR_TEXT_OPTIONS = ("basic", "core")
AMOUNT_TRANSFORMS = ("log1p_cents", "relative")
DATE_TRANSFORMS = ("log1p_days", "capped_90")

# Interaction terms let the model treat a combination of clues differently from
# the sum of its parts. Each set below is one idea about how duplicates differ
# from ordinary repeat bills.
REFERENCE_TERMS = ("ref_exact*amount_exact", "ref_exact*date_gap")  # a repeated reference outweighs a new date
PERIOD_TERMS = ("amount_exact*period_same", "amount_exact*period_different")  # same month again, or next month's bill
VENDOR_TERMS = ("vendor_fuzzy*amount_exact", "vendor_fuzzy*ref_exact")  # agreement counts only for the same vendor
INTERACTION_SETS = (
    (),
    REFERENCE_TERMS,
    PERIOD_TERMS,
    VENDOR_TERMS,
    REFERENCE_TERMS + PERIOD_TERMS,
    REFERENCE_TERMS + PERIOD_TERMS + VENDOR_TERMS,
)
MAX_SWEEPS = 3


def neighbours(config: MatcherConfig) -> dict[str, list[MatcherConfig]]:
    """The configs that differ from `config` in one coordinate, per coordinate.

    C and class_weight are tried together because they share the same
    features. Every list contains `config` itself.
    """
    return {
        "C and class_weight": [replace(config, C=C, class_weight=class_weight)
                               for C in C_OPTIONS for class_weight in CLASS_WEIGHT_OPTIONS],
        "vendor_text": [replace(config, vendor_text=text) for text in VENDOR_TEXT_OPTIONS],
        "vendor_scorer": [replace(config, vendor_scorer=scorer) for scorer in VENDOR_SCORERS],
        "amount_transform": [replace(config, amount_transform=transform) for transform in AMOUNT_TRANSFORMS],
        "date_transform": [replace(config, date_transform=transform) for transform in DATE_TRANSFORMS],
        "interactions": [replace(config, interactions=terms) for terms in INTERACTION_SETS],
    }


COORDINATES = tuple(neighbours(DEFAULT_CONFIG))  # the coordinate names, in the order they are visited


def config_rank(config: MatcherConfig, evaluation: dict) -> tuple:
    """What two learned configs are compared by: the selection score, then simplicity. Larger is better."""
    return (selection_score(evaluation), -len(config.interactions), -config.C)


class TrainingTable:
    """The training pairs of the learned model, rebuilt only when the features change.

    C and class_weight do not change a feature, so candidates that differ only
    in those two share one table (about 900,000 pairs). Only the latest table
    is kept, which keeps memory small.
    """

    def __init__(self, train: Sequence[BenchLedger]) -> None:
        """Remember the training ledgers; nothing is built yet."""
        self.train = train
        self.feature_settings: MatcherConfig | None = None
        self.table: tuple[Any, np.ndarray, np.ndarray] | None = None

    def fit(self, config: MatcherConfig) -> LearnedMatcher:
        """The same model as LearnedMatcher.fit(train, config), reusing the table when possible."""
        feature_settings = replace(config, C=DEFAULT_CONFIG.C, class_weight=DEFAULT_CONFIG.class_weight)
        if feature_settings != self.feature_settings:
            self.table = training_pairs(self.train, feature_settings)
            self.feature_settings = feature_settings
        vectorizer, features, labels = self.table
        return LearnedMatcher.fit_pairs(config, vectorizer, features, labels)


def search_learned_config(train: Sequence[BenchLedger], validation: Sequence[BenchLedger],
                          ) -> tuple[MatcherConfig, list[dict]]:
    """Coordinate-wise search from DEFAULT_CONFIG. Returns (best config, search log).

    One coordinate is changed at a time while the others stay at their best
    values so far, and the best config moves only to a candidate that is
    strictly better. A sweep visits every coordinate once; the search stops
    when a whole sweep changes nothing, or after MAX_SWEEPS. Each candidate is
    fitted on the train split and judged on the validation folds only.
    """
    table = TrainingTable(train)
    tried: dict[MatcherConfig, dict] = {}
    log = []
    best = DEFAULT_CONFIG
    for sweep in range(1, MAX_SWEEPS + 1):
        best_before_sweep = best
        for coordinate in COORDINATES:
            candidates = neighbours(best)[coordinate]
            for candidate in candidates:
                if candidate not in tried:
                    tried[candidate] = evaluate_folds(validation, table.fit(candidate).rank)
                    entry = search_entry(candidate.to_dict(), tried[candidate])
                    log.append({"sweep": sweep, "coordinate": coordinate, **entry})
            # max() keeps the first of equal items, so `best` stays unless a candidate beats it.
            best = max([best] + candidates, key=lambda config: config_rank(config, tried[config]))
        if best == best_before_sweep:
            break
    return best, log


# --------------------------------------------------------------------------
# The LightGBM challenger (optional, validation only)
# --------------------------------------------------------------------------

CHALLENGER_RULE = (
    "LightGBM replaces logistic regression only if it finds at least 3 more correct matches at "
    "budget 20, summed over the three validation folds, and is not worse at budget 10."
)
CHALLENGER_MIN_GAIN = 3
# A short list of small boosted-tree models; the best one on validation is the challenger.
# min_child_weight is raised from the library default of 0.001: only about 1 training pair
# in 4,300 is a duplicate, and with the default a leaf that holds almost no weight takes an
# enormous value, so the scores blow up and the challenger would lose for the wrong reason.
CHALLENGER_SETTINGS = (
    {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 7, "min_child_weight": 1.0},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 7, "min_child_weight": 1.0},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_weight": 1.0},
)


def queue_from_scores(incoming: pd.DataFrame, payments: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Build a queue frame from a full table of pair scores, with the tie rules of matcher.py.

    scores has shape (n_incoming, n_payments) and both frames have rows. Each
    incoming invoice keeps its best-scoring payment; equal scores go to the
    most recent payment, then to the earlier row of the payments file. The
    queue is ordered by score, and equal scores keep incoming file order.
    """
    payment_days = payments["invoice_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    newest_first = np.lexsort((np.arange(len(payments)), -payment_days))
    best_payment = newest_first[scores[:, newest_first].argmax(axis=1)]  # argmax keeps the first of equal scores
    best_score = scores[np.arange(len(incoming)), best_payment]
    order = np.argsort(-best_score, kind="stable")
    return pd.DataFrame({
        "incoming_id": pd.Series(incoming["row_id"].to_numpy(dtype=object)[order], dtype="str"),
        "payment_id": pd.Series(payments["row_id"].to_numpy(dtype=object)[best_payment[order]], dtype="str"),
        "score": best_score[order].astype(np.float64),
        "rank": np.arange(1, len(order) + 1, dtype=np.int64),
    })


def fit_challenger(features: np.ndarray, labels: np.ndarray, settings: dict) -> Any:
    """Fit one LightGBM classifier on the training table of the learned model.

    LightGBM is imported here because only the --challenger option needs it.
    One thread and the deterministic switch give the same trees on every run.
    """
    from lightgbm import LGBMClassifier

    classifier = LGBMClassifier(objective="binary", random_state=0, n_jobs=1, deterministic=True,
                                force_row_wise=True, verbose=-1, **settings)
    return classifier.fit(features, labels)


def challenger_ranker(model: LearnedMatcher, classifier: Any) -> RankFunction:
    """A rank function that scores the learned model's pair features with a LightGBM classifier."""
    def rank(incoming: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
        """Score every pair of one ledger and keep each invoice's best payment."""
        cube = model.pair_features(incoming, payments)
        scores = classifier.predict(cube.reshape(-1, cube.shape[2]), raw_score=True)
        return queue_from_scores(incoming, payments, scores.reshape(cube.shape[0], cube.shape[1]))
    return rank


def challenger_decision(logistic: dict, lightgbm: dict) -> dict:
    """Apply CHALLENGER_RULE to the summed validation metrics of both models."""
    gain_at_20 = lightgbm["20"]["correct"] - logistic["20"]["correct"]
    gain_at_10 = lightgbm["10"]["correct"] - logistic["10"]["correct"]
    triggered = gain_at_20 >= CHALLENGER_MIN_GAIN and gain_at_10 >= 0
    if triggered:
        decision = ("The rule is met, so LightGBM would replace logistic regression. Logistic regression "
                    "is frozen anyway until the lead engineer decides; the shipped model was not changed.")
    else:
        decision = "The rule is not met. Logistic regression stays: it is simpler and every coefficient can be read."
    return {"gain_at_20": gain_at_20, "gain_at_10": gain_at_10, "rule_triggered": triggered, "decision": decision}


def run_challenger(train: Sequence[BenchLedger], validation: Sequence[BenchLedger],
                   model: LearnedMatcher, learned: dict) -> dict:
    """Fit the LightGBM candidates on the frozen model's training pairs and compare on validation."""
    _vectorizer, features, labels = training_pairs(train, model.config)
    candidates = []
    for settings in CHALLENGER_SETTINGS:
        classifier = fit_challenger(features, labels, settings)
        candidates.append(evaluate_folds(validation, challenger_ranker(model, classifier)))
    scores = [selection_score(evaluation) for evaluation in candidates]
    best = scores.index(max(scores))
    return {
        "evaluated": True,
        "rule": CHALLENGER_RULE,
        "features": model.feature_names,
        "candidates": [search_entry(settings, evaluation)
                       for settings, evaluation in zip(CHALLENGER_SETTINGS, candidates)],
        "lightgbm": {"setting": CHALLENGER_SETTINGS[best], "summed": candidates[best]["summed"],
                     "per_fold_correct_at_20": per_fold_correct(candidates[best])},
        "logistic_regression": {"summed": learned["summed"], "per_fold_correct_at_20": per_fold_correct(learned)},
        **challenger_decision(learned["summed"], candidates[best]["summed"]),
    }


def per_fold_correct(evaluation: dict) -> dict:
    """fold -> correct matches at the default budget."""
    return {fold: results["results"][str(DEFAULT_BUDGET)]["correct"] for fold, results in evaluation["folds"].items()}


# --------------------------------------------------------------------------
# Describing data, the model and the errors
# --------------------------------------------------------------------------

def dataset_summary(ledgers: Sequence[BenchLedger]) -> dict:
    """Sizes and the first and last invoice date of a set of ledgers."""
    dates = pd.concat([frame["invoice_date"] for ledger in ledgers for frame in (ledger.incoming, ledger.payments)])
    return {
        "ledgers": len(ledgers),
        "incoming": sum(len(ledger.incoming) for ledger in ledgers),
        "prior_payments": sum(len(ledger.payments) for ledger in ledgers),
        "known_duplicates": sum(len(ledger.truth) for ledger in ledgers),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
    }


def model_card(model: LearnedMatcher, train: Sequence[BenchLedger]) -> dict:
    """Everything needed to recompute a score by hand: the config, the scaler and the coefficients."""
    return {
        "model_type": "LogisticRegression (scikit-learn) on standardized features of an invoice-payment pair",
        "score": "decision value = intercept + sum of coefficient x (feature - mean) / scale",
        "feature_names": model.feature_names,
        "coefficients": [float(value) for value in model.classifier.coef_[0]],
        "intercept": float(model.classifier.intercept_[0]),
        "scaler_means": [float(value) for value in model.scaler.mean_],
        "scaler_scales": [float(value) for value in model.scaler.scale_],
        "n_train_pairs": model.n_train_pairs,
        "n_train_positives": model.n_train_positives,
        "training": {"split": "train", **dataset_summary(train),
                     "ledger_ids": [ledger.ledger_id for ledger in train]},
        "config": model.config.to_dict(),
    }


def method_definitions(rule: FuzzyRule, model: LearnedMatcher) -> dict[str, str]:
    """One sentence per method that states which fields have to agree."""
    amount_rules = {
        "exact": "the same amount",
        "within_100_cents": "an amount within $1.00",
        "within_1_percent": "an amount within 1 percent",
    }
    rankings = {
        "vendor": "ranked by vendor-name similarity",
        "ref_then_vendor": "ranked with equal references first, then by vendor-name similarity",
    }
    return {
        "exact": ("Flags an invoice when a prior payment has the same vendor name, the same non-blank invoice "
                  "reference and the same amount after text normalization; the closest invoice date ranks first."),
        "fuzzy": (f"Flags an invoice when a prior payment has a vendor name at least "
                  f"{rule.vendor_threshold:.0%} similar (RapidFuzz {rule.vendor_scorer}) and "
                  f"{amount_rules[rule.amount_rule]}; {rankings[rule.rank_by]}."),
        "learned": (f"No field is required to agree: logistic regression over {len(model.feature_names)} "
                    "comparisons of an invoice with a prior payment (vendor, reference, amount, date, billing "
                    "period) ranks every invoice by its best-scoring payment."),
    }


def outcome_table(ledgers: Sequence[BenchLedger], pooled: pd.DataFrame, budget: int) -> pd.DataFrame:
    """One row per incoming invoice of a pool: its answer-key columns and what the review did with it.

    reviewed  the invoice was among the first `budget` proposals
    correct   it was reviewed and shown with a valid prior payment
    scenario and vendor_kind are generator metadata. They are used here, after
    the ranking is fixed, to describe errors; no method ever sees them.
    """
    reviewed = pooled.head(budget)
    reviewed_keys = list(zip(reviewed["ledger_id"], reviewed["incoming_id"]))
    hits = correct_flags(reviewed, pooled_truth(ledgers))
    was_reviewed = set(reviewed_keys)
    was_correct = {key for key, hit in zip(reviewed_keys, hits) if hit}
    labels = pd.concat([ledger.labels for ledger in ledgers], ignore_index=True)
    label_keys = list(zip(labels["ledger_id"], labels["incoming_row_id"]))
    return pd.DataFrame({
        "scenario": labels["scenario"],
        "vendor_kind": labels["vendor_kind"],
        "is_duplicate": labels["is_duplicate"] == 1,
        "reviewed": [key in was_reviewed for key in label_keys],
        "correct": [key in was_correct for key in label_keys],
    })


def scenario_breakdown(outcomes: pd.DataFrame) -> dict:
    """Which kinds of duplicate were caught, and which kinds of invoice were flagged wrongly."""
    duplicates = outcomes[outcomes["is_duplicate"]]
    false_flags = outcomes[outcomes["reviewed"] & ~outcomes["correct"]]
    return {
        "duplicates": {scenario: {"known": len(rows), "caught": int(rows["correct"].sum())}
                       for scenario, rows in duplicates.groupby("scenario")},
        "false_flags": {scenario: len(rows) for scenario, rows in false_flags.groupby("scenario")},
    }


def vendor_kind_slices(outcomes: dict[str, pd.DataFrame]) -> list[dict]:
    """The fairness check: are legitimate invoices from individuals held for review more often?

    outcomes maps a method key to its outcome table over the same invoices.
    """
    any_table = next(iter(outcomes.values()))
    slices = []
    for group in VENDOR_KINDS:
        in_group = any_table[any_table["vendor_kind"] == group]
        methods = {}
        for key, table in outcomes.items():
            rows = table[table["vendor_kind"] == group]
            methods[key] = {
                "legit_flagged": int((rows["reviewed"] & ~rows["is_duplicate"]).sum()),
                "duplicates_caught": int(rows["correct"].sum()),
            }
        slices.append({
            "group": group,
            "legit_invoices": int((~in_group["is_duplicate"]).sum()),
            "known_duplicates": int(in_group["is_duplicate"].sum()),
            "methods": methods,
        })
    return slices


# --------------------------------------------------------------------------
# Worked examples of the learned model's errors
# --------------------------------------------------------------------------

def find_row(frame: pd.DataFrame, row_id: str) -> pd.Series:
    """The row of a ledger frame with this row_id."""
    return frame[frame["row_id"] == row_id].iloc[0]


def row_record(row: pd.Series) -> dict:
    """A ledger row as plain JSON values."""
    return {
        "row_id": row["row_id"],
        "vendor": row["vendor"],
        "invoice_ref": row["invoice_ref"],
        "amount_cents": int(row["amount_cents"]),
        "invoice_date": row["invoice_date"].date().isoformat(),
        "billing_period": row["billing_period"],
    }


def evidence_list(comparisons: Sequence[FieldComparison], statuses: Sequence[str],
                  fields: Sequence[str] = ("vendor", "invoice_ref", "amount", "invoice_date", "billing_period")) -> str:
    """The evidence sentences of the chosen comparisons, joined for use inside a sentence."""
    chosen = [comparison.evidence for comparison in comparisons
              if comparison.status in statuses and comparison.field in fields]
    return "; ".join(sentence[0].lower() + sentence[1:] for sentence in chosen)


def likely_cause(kind: str, comparisons: Sequence[FieldComparison], rank: int, budget: int,
                 is_duplicate: bool, proposed_is_valid: bool) -> str:
    """One or two fixed-template sentences that describe an example from its field comparisons.

    For a missed duplicate the comparisons are those with the payment it really
    repeats; for the other kinds they are those with the proposed payment.
    """
    support = evidence_list(comparisons, ("match", "partial")) or "no field agrees"
    against = evidence_list(comparisons, ("differ", "missing"))
    if kind == "missed_duplicate":
        if proposed_is_valid:
            opening = (f"The right prior payment was found, but the pair ranked {rank}, "
                       f"below the review budget of {budget}.")
        else:
            opening = f"The model paired this invoice with a different prior payment (rank {rank})."
        return f"{opening} Compared with the payment it repeats: {against or 'nothing argues against a match'}."
    if kind == "false_flag":
        if is_duplicate:
            return f"A real duplicate, but shown with the wrong prior payment, which agrees on: {support}."
        closing = f" What separates them: {against}." if against else ""
        return f"A legitimate invoice that resembles a prior payment: {support}.{closing}"
    blockers = evidence_list(comparisons, ("partial", "differ", "missing"), ("vendor", "invoice_ref", "amount"))
    if blockers:
        opening = f"Exact match needs the same vendor, reference and amount, but here: {blockers}."
    else:
        opening = "Exact match did not show this pair within its review budget."
    return f"{opening} The learned model ranked the pair {rank} because: {support}."


def build_example(kind: str, ledger: BenchLedger, proposal: Any, budget: int) -> dict:
    """One worked example. `proposal` is a row of the learned model's pooled queue."""
    valid = ledger.truth.get(proposal.incoming_id, frozenset())
    proposed_is_valid = proposal.payment_id in valid
    valid_in_file_order = [payment_id for payment_id in ledger.payments["row_id"] if payment_id in valid]
    true_id = proposal.payment_id if proposed_is_valid else (valid_in_file_order[0] if valid_in_file_order else None)

    incoming = find_row(ledger.incoming, proposal.incoming_id)
    proposed = find_row(ledger.payments, proposal.payment_id)
    true_payment = find_row(ledger.payments, true_id) if true_id else None
    compared_with = true_payment if kind == "missed_duplicate" else proposed
    comparisons = compare_fields(incoming, compared_with)
    return {
        "method": "learned",
        "kind": kind,
        "budget": budget,
        "rank": int(proposal.pooled_rank),
        "incoming": row_record(incoming),
        "proposed_payment": row_record(proposed),
        "true_payment": row_record(true_payment) if true_payment is not None else None,
        "comparisons": [{"field": comparison.field, "label": comparison.label, "status": comparison.status,
                         "evidence": comparison.evidence} for comparison in comparisons],
        "likely_cause": likely_cause(kind, comparisons, int(proposal.pooled_rank), budget,
                                     is_duplicate=bool(valid), proposed_is_valid=proposed_is_valid),
    }


def pick_examples(ledgers: Sequence[BenchLedger], learned_pool: pd.DataFrame, exact_pool: pd.DataFrame,
                  budget: int) -> list[dict]:
    """Up to three worked examples from the learned model's pooled queue, each the first of its kind.

    missed_duplicate  a known duplicate that was not reviewed with a valid payment
    false_flag        a reviewed proposal that is not a valid match
    caught_duplicate  a duplicate the learned model caught and exact match did not
    A kind that does not occur is left out.
    """
    truth = pooled_truth(ledgers)
    by_id = {ledger.ledger_id: ledger for ledger in ledgers}
    exact_reviewed = exact_pool.head(budget)
    exact_hits = correct_flags(exact_reviewed, truth)
    exact_caught = {(row.ledger_id, row.incoming_id)
                    for row, hit in zip(exact_reviewed.itertuples(index=False), exact_hits) if hit}

    first: dict[str, Any] = {}
    for row, hit in zip(learned_pool.itertuples(index=False), correct_flags(learned_pool, truth)):
        key = (row.ledger_id, row.incoming_id)
        caught = bool(hit) and row.pooled_rank <= budget
        if key in truth and not caught:
            first.setdefault("missed_duplicate", row)
        if row.pooled_rank <= budget and not hit:
            first.setdefault("false_flag", row)
        if caught and key not in exact_caught:
            first.setdefault("caught_duplicate", row)
    return [build_example(kind, by_id[first[kind].ledger_id], first[kind], budget)
            for kind in ("missed_duplicate", "false_flag", "caught_duplicate") if kind in first]


# --------------------------------------------------------------------------
# Writing and printing results
# --------------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    """Write a result file. No timestamps go in, so the same run gives the same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(title: str, results: dict[str, dict], budgets: Sequence[int]) -> None:
    """Print the counts of every method at every budget. Counts only, never ledger rows."""
    print(title)
    print(f"{'method':<15}{'budget':>6}{'reviewed':>10}{'correct':>9}{'false flags':>13}{'missed':>8}"
          f"{'precision':>11}{'recall':>8}")
    for key, name in METHOD_NAMES.items():
        for budget in budgets:
            row = results[key][str(budget)]
            precision = "-" if row["precision"] is None else f"{row['precision']:.2f}"
            recall = "-" if row["recall"] is None else f"{row['recall']:.2f}"
            print(f"{name:<15}{budget:>6}{row['reviews_used']:>10}{row['correct']:>9}{row['false_flags']:>13}"
                  f"{row['missed']:>8}{precision:>11}{recall:>8}")


# --------------------------------------------------------------------------
# Stage 1: validate (tune on validation, then freeze)
# --------------------------------------------------------------------------

def method_rankers(model: LearnedMatcher, rule: FuzzyRule) -> dict[str, RankFunction]:
    """The three methods as rank functions, keyed like METHOD_NAMES."""
    return {"exact": exact_match_queue, "fuzzy": partial(fuzzy_rule_queue, rule=rule), "learned": model.rank}


def validation_report(train: Sequence[BenchLedger], validation: Sequence[BenchLedger],
                      pools: dict[str, list], evaluations: dict[str, dict], frozen: dict,
                      learned_log: list[dict], fuzzy_log: list[dict], command: str) -> dict:
    """The contents of results/validation.json: every number the tuning looked at."""
    outcomes = {key: pd.concat([outcome_table(fold_ledgers, pooled, DEFAULT_BUDGET)
                                for fold_ledgers, pooled in method_pools], ignore_index=True)
                for key, method_pools in pools.items()}
    return {
        "label": "Validation results, used for tuning. The benchmark is results/benchmark.json.",
        "command": command,
        "criterion": SELECTION_CRITERION,
        "budgets": list(BUDGETS),
        "default_budget": DEFAULT_BUDGET,
        "datasets": {"train": dataset_summary(train), "validation": dataset_summary(validation)},
        "methods": {key: {"name": METHOD_NAMES[key], **evaluations[key]} for key in METHOD_NAMES},
        "frozen": {"matcher": frozen["matcher"], "fuzzy_rule": frozen["fuzzy_rule"]},
        "error_analysis": {key: scenario_breakdown(outcomes[key]) for key in METHOD_NAMES},
        "slices": {"vendor_kind": vendor_kind_slices(outcomes)},
        "learned_search": learned_log,
        "fuzzy_search": fuzzy_log,
        "challenger": frozen["challenger"],
    }


def print_validation(validation: Sequence[BenchLedger], evaluations: dict[str, dict], frozen: dict) -> None:
    """Print the validation table and what was frozen. Counts and settings only."""
    summary = dataset_summary(validation)
    print_summary(f"Validation: {summary['ledgers']} ledgers in {len(split_folds(validation))} folds, "
                  f"{summary['known_duplicates']} known duplicates (counts summed over the folds)",
                  {key: evaluation["summed"] for key, evaluation in evaluations.items()}, BUDGETS)
    for key, name in METHOD_NAMES.items():
        print(f"{name}, correct at budget {DEFAULT_BUDGET} per fold: {per_fold_correct(evaluations[key])}")
    print(f"Frozen fuzzy rule: {frozen['fuzzy_rule']}")
    print(f"Frozen learned model: {frozen['matcher']}")
    challenger = frozen["challenger"]
    if challenger["evaluated"]:
        correct = challenger["lightgbm"]["summed"][str(DEFAULT_BUDGET)]["correct"]
        print(f"LightGBM challenger: {correct} correct at budget {DEFAULT_BUDGET} "
              f"(gain {challenger['gain_at_20']:+d}). {challenger['decision']}")


def run_validate_stage(root: Path = PROJECT_ROOT, challenger: bool = False) -> dict:
    """Tune the fuzzy rule and the learned model on validation and freeze the result.

    Writes results/frozen_config.json, results/validation.json and
    results/model_card.json under `root` and returns the frozen config.
    """
    started = time.perf_counter()
    data_dir, results_dir = Path(root) / "data", Path(root) / "results"
    train = load_benchmark_split("train", data_dir)
    validation = load_benchmark_split("validation", data_dir)

    print(f"Tuning the fuzzy rule on validation ({len(fuzzy_grid())} rules) ...")
    rule, fuzzy_log = search_fuzzy_rule(validation)
    print("Tuning the learned model: each candidate is fitted on train and judged on validation ...")
    config, learned_log = search_learned_config(train, validation)

    model = LearnedMatcher.fit(train, config)  # the call fit_shipped_matcher makes once the config is frozen
    pools = {key: fold_pools(validation, rank) for key, rank in method_rankers(model, rule).items()}
    evaluations = {key: summarize_folds(method_pools) for key, method_pools in pools.items()}
    challenger_report = {"evaluated": False, "rule": CHALLENGER_RULE}
    if challenger:
        print(f"Fitting the LightGBM challenger ({len(CHALLENGER_SETTINGS)} settings) ...")
        challenger_report = run_challenger(train, validation, model, evaluations["learned"])

    frozen = {
        "matcher": config.to_dict(),
        "fuzzy_rule": rule.to_dict(),
        "selection": {"criterion": SELECTION_CRITERION,
                      "validation": {key: evaluation["summed"] for key, evaluation in evaluations.items()}},
        "challenger": challenger_report,
    }
    command = "python evaluate.py --stage validate" + (" --challenger" if challenger else "")
    write_json(results_dir / "frozen_config.json", frozen)
    write_json(results_dir / "model_card.json", model_card(model, train))
    write_json(results_dir / "validation.json",
               validation_report(train, validation, pools, evaluations, frozen, learned_log, fuzzy_log, command))

    print_validation(validation, evaluations, frozen)
    print(f"Tried {len(learned_log)} learned configs and {len(fuzzy_log)} fuzzy rules. "
          f"Validate stage finished in {time.perf_counter() - started:.0f} s.")
    return frozen


# --------------------------------------------------------------------------
# Stage 2: test (the held-out split, scored once with frozen settings)
# --------------------------------------------------------------------------

def benchmark_body(ledgers: Sequence[BenchLedger], model: LearnedMatcher, rule: FuzzyRule) -> dict:
    """Score the three methods on one pool of ledgers: the parts of benchmark.json that come from data."""
    pools = {key: pool_method(ledgers, rank) for key, rank in method_rankers(model, rule).items()}
    definitions = method_definitions(rule, model)
    methods = []
    for key, name in METHOD_NAMES.items():
        results = pool_results(ledgers, pools[key])
        methods.append({"key": key, "name": name, "definition": definitions[key],
                        "eligible": results["eligible"], "results": results["results"]})
    outcomes = {key: outcome_table(ledgers, pooled, DEFAULT_BUDGET) for key, pooled in pools.items()}
    return {
        "dataset": dataset_summary(ledgers),
        "budgets": list(BUDGETS),
        "default_budget": DEFAULT_BUDGET,
        "methods": methods,
        "examples": pick_examples(ledgers, pools["learned"], pools["exact"], DEFAULT_BUDGET),
        "slices": {"vendor_kind": vendor_kind_slices(outcomes)},
    }


def challenge_report(ledgers: Sequence[BenchLedger], model: LearnedMatcher, rule: FuzzyRule) -> dict:
    """Score the three methods on the hand-authored challenge ledgers, with a budget of one review per duplicate."""
    truth = pooled_truth(ledgers)
    budget = len(truth)
    methods = []
    for key, rank in method_rankers(model, rule).items():
        pooled = pool_method(ledgers, rank)
        methods.append({"key": key, "name": METHOD_NAMES[key],
                        "results": {str(budget): score_budget(pooled, truth, len(truth), budget)}})
    return {
        "dataset": dataset_summary(ledgers),
        "budget": budget,
        "methods": methods,
        "note": "hand-authored challenge set, reported separately",
    }


def print_benchmark(report: dict) -> None:
    """Print the test table, and the challenge table when there is one. Counts only."""
    dataset = report["dataset"]
    print_summary(f"Held-out test split: {dataset['ledgers']} ledgers, {dataset['known_duplicates']} known duplicates",
                  {method["key"]: method["results"] for method in report["methods"]}, BUDGETS)
    challenge = report["challenge"]
    if challenge is not None:
        print_summary(f"Challenge set, reported separately: {challenge['budget']} known duplicates",
                      {method["key"]: method["results"] for method in challenge["methods"]}, [challenge["budget"]])


def run_test_stage(root: Path = PROJECT_ROOT) -> dict:
    """Score the frozen settings once on the held-out test split and write results/benchmark.json.

    Refuses to run without results/frozen_config.json or without the test
    files. Nothing is tuned here: every setting is read from the frozen config.
    """
    data_dir, results_dir = Path(root) / "data", Path(root) / "results"
    frozen_path = results_dir / "frozen_config.json"
    if not frozen_path.exists():
        raise StageError(f"The test stage needs {frozen_path}, which does not exist. Settings are chosen on "
                         "validation and frozen before the test split is opened. "
                         "Run first: python evaluate.py --stage validate")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    try:
        test = load_benchmark_split("test", data_dir)
    except FileNotFoundError as error:
        raise StageError(f"The test stage needs the held-out test split under {data_dir / 'synthetic'}, and it "
                         "is not there. It is generated once, after the settings are frozen, with: "
                         "python synth.py --splits test") from error

    model = fit_shipped_matcher(Path(root))
    rule = FuzzyRule.from_dict(frozen["fuzzy_rule"])
    challenge = None
    if (data_dir / "challenge").exists():
        challenge = challenge_report(load_benchmark_split("challenge", data_dir), model, rule)

    validation = frozen["selection"]["validation"]
    report = {
        "label": "Synthetic benchmark",
        "split": "test",
        "command": "python evaluate.py --stage test",
        **benchmark_body(test, model, rule),
        "challenge": challenge,
        "model": {"type": "LogisticRegression", "features": model.feature_names, "config": model.config.to_dict()},
        "fuzzy_rule": rule.to_dict(),
        "validation_summary": {key: validation[key][str(DEFAULT_BUDGET)] for key in METHOD_NAMES},
    }
    write_json(results_dir / "benchmark.json", report)
    print_benchmark(report)
    return report


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Benchmark the three Invoice Match methods.")
    parser.add_argument("--stage", required=True, choices=("validate", "test", "all"),
                        help="validate: tune and freeze; test: score the frozen settings on the held-out split")
    parser.add_argument("--challenger", action="store_true",
                        help="with the validate stage, also try LightGBM on the same pair features")
    args = parser.parse_args(argv)
    if args.challenger and args.stage == "test":
        parser.error("--challenger belongs to the validate stage; the test stage tunes nothing")
    return args


def main(argv: Sequence[str] | None = None, root: Path = PROJECT_ROOT) -> int:
    """Command-line entry point. Returns the exit code. `root` is the project folder."""
    args = parse_args(argv)
    try:
        if args.stage in ("validate", "all"):
            run_validate_stage(root, challenger=args.challenger)
        if args.stage in ("test", "all"):
            run_test_stage(root)
    except StageError as error:
        print(f"evaluate.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
