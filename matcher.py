"""matcher.py - compares incoming invoices with prior payments.

Three ways of building a review queue live here, all on the same normalized
text so that they differ only in their matching logic:

  * exact_match_queue    same vendor, same reference and same amount
  * fuzzy_rule_queue     a hand-written rule: similar vendor name and an amount rule
  * LearnedMatcher       logistic regression over features of an invoice pair

Design choices worth knowing:
  * A feature always describes a PAIR (one incoming invoice, one prior
    payment). Nothing about a single record, no row IDs and no generator
    metadata ever reaches the model, so it cannot learn who the "usual
    suspects" are, only what a repeated bill looks like.
  * The answer key reaches the model in exactly one place: as the training
    label inside LearnedMatcher.fit.
  * The evidence sentences shown to the reviewer come from fixed templates over
    plain field comparisons. The model ranks; it never writes text.
  * Nothing here prints, logs or saves ledger contents.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import OSA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ledger import LedgerError, format_cents, format_date, format_period, read_ledger_csv

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

LEGAL_FORMS = frozenset({
    "llc", "inc", "incorporated", "co", "company", "corp", "corporation",
    "ltd", "limited", "lp", "llp", "pllc",
})
REF_SEPARATORS = "-_/.#:"
REF_PREFIXES = ("INVOICE", "INV", "BILL", "REF", "NUM", "NO")  # INVOICE is tried before INV
APOSTROPHES = "'\u2019"  # the typewriter apostrophe and the curly one


def normalize_vendor(name: str) -> str:
    """'O.C. Event Rentals & Décor' -> 'oc event rentals and decor'.

    Accents are stripped, case is folded, '&' becomes 'and', periods and
    apostrophes vanish, other punctuation becomes a space and runs of spaces
    collapse. A name made only of punctuation is kept as typed, so two such
    names never look alike just because nothing was left of them.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    text = without_accents.casefold().replace("&", " and ")
    kept = []
    for ch in text:
        if ch == "." or ch in APOSTROPHES:
            continue
        kept.append(ch if ch.isalnum() else " ")
    return " ".join("".join(kept).split()) or name.strip().casefold()


def vendor_core(name: str) -> str:
    """normalize_vendor without trailing legal forms: 'OC Event Rentals LLC' -> 'oc event rentals'.

    A name made only of legal forms (a vendor called 'Limited') keeps its
    normalized form, so the result is never empty for a non-blank name.
    """
    basic = normalize_vendor(name)
    tokens = basic.split()
    while tokens and tokens[-1] in LEGAL_FORMS:
        tokens.pop()
    return " ".join(tokens) or basic


def normalize_ref(ref: str) -> str:
    """'INV-1042', 'inv 1042' and '#1042' all become '1042'. Blank stays ''.

    Upper-case, drop whitespace and the separators - _ / . # : and then drop one
    leading word such as INV or BILL, but only when a digit is left behind
    ('INV' alone stays 'INV'). Leading zeros are kept: '0042' is not '42'.
    """
    text = "".join(ch for ch in ref.upper() if not ch.isspace() and ch not in REF_SEPARATORS)
    for prefix in REF_PREFIXES:
        if text.startswith(prefix):
            rest = text[len(prefix):]
            if any(ch.isdigit() for ch in rest):
                return rest
    return text


# --------------------------------------------------------------------------
# Settings for the learned model and for the fuzzy rule
# --------------------------------------------------------------------------

VENDOR_SCORERS = {
    "ratio": fuzz.ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "WRatio": fuzz.WRatio,
}

BASE_FEATURES = (
    "vendor_tfidf_cosine",
    "vendor_fuzzy",
    "ref_similarity",
    "ref_exact",
    "ref_missing_incoming",
    "ref_missing_payment",
    "amount_exact",
    "amount_diff",
    "date_gap",
    "period_same",
    "period_different",
    "period_missing_incoming",
    "period_missing_payment",
)


def _require_choice(value: Any, choices: Sequence[Any], field_name: str) -> None:
    """Stop early, with a readable message, when a setting is misspelled."""
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {list(choices)}, not {value!r}")


@dataclass(frozen=True)
class MatcherConfig:
    """Every choice that validation tuning may make for the learned model.

    vendor_text       'basic' (normalize_vendor) or 'core' (vendor_core)
    vendor_scorer     which RapidFuzz scorer feeds the vendor_fuzzy feature
    amount_transform  'log1p_cents' (log of 1 + cents apart) or 'relative'
                      (cents apart divided by the larger amount)
    date_transform    'log1p_days' (log of 1 + days apart) or 'capped_90'
                      (days apart, counted up to 90)
    interactions      extra features, each the product of base features and
                      written with '*', e.g. 'amount_exact*vendor_fuzzy'
    C                 inverse regularization strength of the logistic regression
    class_weight      None or 'balanced'
    """

    vendor_text: str = "core"
    vendor_scorer: str = "token_sort_ratio"
    amount_transform: str = "log1p_cents"
    date_transform: str = "log1p_days"
    interactions: tuple[str, ...] = ()
    C: float = 1.0
    class_weight: str | None = None

    def __post_init__(self) -> None:
        """Check every setting as soon as the config is created."""
        _require_choice(self.vendor_text, ("basic", "core"), "vendor_text")
        _require_choice(self.vendor_scorer, tuple(VENDOR_SCORERS), "vendor_scorer")
        _require_choice(self.amount_transform, ("log1p_cents", "relative"), "amount_transform")
        _require_choice(self.date_transform, ("log1p_days", "capped_90"), "date_transform")
        _require_choice(self.class_weight, (None, "balanced"), "class_weight")
        if self.C <= 0:
            raise ValueError(f"C must be positive, not {self.C!r}")
        for name in self.interactions:
            parts = name.split("*")
            if len(parts) < 2 or any(part not in BASE_FEATURES for part in parts):
                raise ValueError(f"interaction {name!r} must join base feature names with '*'")

    def to_dict(self) -> dict:
        """A plain dictionary that json.dump can write."""
        return {
            "vendor_text": self.vendor_text,
            "vendor_scorer": self.vendor_scorer,
            "amount_transform": self.amount_transform,
            "date_transform": self.date_transform,
            "interactions": list(self.interactions),
            "C": self.C,
            "class_weight": self.class_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MatcherConfig":
        """The reverse of to_dict."""
        return cls(
            vendor_text=data["vendor_text"],
            vendor_scorer=data["vendor_scorer"],
            amount_transform=data["amount_transform"],
            date_transform=data["date_transform"],
            interactions=tuple(data["interactions"]),
            C=float(data["C"]),
            class_weight=data["class_weight"],
        )


DEFAULT_CONFIG = MatcherConfig()


def feature_names(config: MatcherConfig) -> list[str]:
    """The model's input columns, in order: the base features, then the interactions."""
    return list(BASE_FEATURES) + list(config.interactions)


@dataclass(frozen=True)
class FuzzyRule:
    """The hand-written baseline rule.

    vendor_threshold  smallest vendor-name similarity (0..1) that counts
    amount_rule       'exact', 'within_100_cents' or 'within_1_percent' (of the larger amount)
    rank_by           'vendor' (most similar name first) or 'ref_then_vendor'
                      (pairs with the same reference first, then by name)
    vendor_scorer     which RapidFuzz scorer measures the vendor names
    """

    vendor_threshold: float = 0.85
    amount_rule: str = "exact"
    rank_by: str = "ref_then_vendor"
    vendor_scorer: str = "token_sort_ratio"

    def __post_init__(self) -> None:
        """Check every setting as soon as the rule is created."""
        if not 0.0 <= self.vendor_threshold <= 1.0:
            raise ValueError(f"vendor_threshold must be between 0 and 1, not {self.vendor_threshold!r}")
        _require_choice(self.amount_rule, ("exact", "within_100_cents", "within_1_percent"), "amount_rule")
        _require_choice(self.rank_by, ("vendor", "ref_then_vendor"), "rank_by")
        _require_choice(self.vendor_scorer, tuple(VENDOR_SCORERS), "vendor_scorer")

    def to_dict(self) -> dict:
        """A plain dictionary that json.dump can write."""
        return {
            "vendor_threshold": self.vendor_threshold,
            "amount_rule": self.amount_rule,
            "rank_by": self.rank_by,
            "vendor_scorer": self.vendor_scorer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FuzzyRule":
        """The reverse of to_dict."""
        return cls(
            vendor_threshold=float(data["vendor_threshold"]),
            amount_rule=data["amount_rule"],
            rank_by=data["rank_by"],
            vendor_scorer=data["vendor_scorer"],
        )


DEFAULT_FUZZY_RULE = FuzzyRule()


# --------------------------------------------------------------------------
# Comparing every incoming invoice with every prior payment
# --------------------------------------------------------------------------
# All pair arrays have shape (n_incoming, n_payments). They are built with
# numpy broadcasting: a column of incoming values against a row of payment
# values, never a Python loop over pairs.

@dataclass
class _Side:
    """One ledger frame reduced to the normalized columns that comparisons need."""

    vendors: np.ndarray  # normalized vendor names (text)
    refs: np.ndarray  # normalized references, '' when blank
    amounts: np.ndarray  # integer cents
    days: np.ndarray  # invoice date as a day count, so a gap is a subtraction
    periods: np.ndarray  # 'YYYY-MM', '' when blank
    vendor_vectors: Any = None  # TF-IDF rows (a sparse matrix); only the learned model fills this in


def _prepare_side(frame: pd.DataFrame, vendor_text: str) -> _Side:
    """Normalize the text of a ledger frame once and hold its columns as numpy arrays."""
    normalize = vendor_core if vendor_text == "core" else normalize_vendor
    return _Side(
        vendors=np.array([normalize(name) for name in frame["vendor"]], dtype=str),
        refs=np.array([normalize_ref(ref) for ref in frame["invoice_ref"]], dtype=str),
        amounts=frame["amount_cents"].to_numpy(dtype=np.int64),
        days=_day_numbers(frame),
        periods=np.array(frame["billing_period"].tolist(), dtype=str),
    )


def _day_numbers(frame: pd.DataFrame) -> np.ndarray:
    """Invoice dates as whole days since 1970-01-01."""
    return frame["invoice_date"].to_numpy().astype("datetime64[D]").astype(np.int64)


def _gap(incoming_values: np.ndarray, payment_values: np.ndarray) -> np.ndarray:
    """Absolute difference between every incoming value and every payment value."""
    return np.abs(incoming_values[:, None] - payment_values[None, :])


def _vendor_fuzzy(incoming_names: np.ndarray, payment_names: np.ndarray, scorer_name: str) -> np.ndarray:
    """RapidFuzz similarity of every pair of vendor names, scaled to 0..1."""
    scores = process.cdist(incoming_names, payment_names, scorer=VENDOR_SCORERS[scorer_name], dtype=np.float64)
    return scores / 100.0


def _tfidf_cosine(incoming_vectors: Any, payment_vectors: Any) -> np.ndarray:
    """Cosine similarity of every pair of TF-IDF rows.

    The vectorizer scales each row to length 1, so the dot product is the
    cosine. A name with no known character n-grams is an all-zero row and
    scores 0 against everything.
    """
    return np.clip((incoming_vectors @ payment_vectors.T).toarray(), 0.0, 1.0)


def _ref_exact(incoming_refs: np.ndarray, payment_refs: np.ndarray) -> np.ndarray:
    """True where both references are present and equal. Two blanks are not a match."""
    return (incoming_refs[:, None] == payment_refs[None, :]) & (incoming_refs != "")[:, None]


def _ref_similarity(incoming_refs: np.ndarray, payment_refs: np.ndarray) -> np.ndarray:
    """Edit similarity (0..1) of every pair of references, 0 when either is blank.

    OSA distance counts two swapped neighbours ('1042' and '1024') as one edit,
    which is the typical keying mistake.
    """
    similarity = process.cdist(incoming_refs, payment_refs, scorer=OSA.normalized_similarity, dtype=np.float64)
    both_present = (incoming_refs != "")[:, None] & (payment_refs != "")[None, :]
    return np.where(both_present, similarity, 0.0)


def _amount_diff(incoming: _Side, payments: _Side, transform: str) -> np.ndarray:
    """How far apart the amounts are, on the scale chosen in the config."""
    gap_cents = _gap(incoming.amounts, payments.amounts)
    if transform == "relative":
        return gap_cents / np.maximum(incoming.amounts[:, None], payments.amounts[None, :])
    return np.log1p(gap_cents)


def _date_gap(incoming: _Side, payments: _Side, transform: str) -> np.ndarray:
    """How far apart the invoice dates are, on the scale chosen in the config."""
    gap_days = _gap(incoming.days, payments.days)
    if transform == "capped_90":
        return np.minimum(gap_days, 90)
    return np.log1p(gap_days)


def _pair_features(incoming: _Side, payments: _Side, config: MatcherConfig) -> np.ndarray:
    """The feature cube, shape (n_incoming, n_payments, n_features), float64."""
    shape = (len(incoming.amounts), len(payments.amounts))
    ref_missing_incoming = np.broadcast_to((incoming.refs == "")[:, None], shape)
    ref_missing_payment = np.broadcast_to((payments.refs == "")[None, :], shape)
    period_missing_incoming = np.broadcast_to((incoming.periods == "")[:, None], shape)
    period_missing_payment = np.broadcast_to((payments.periods == "")[None, :], shape)
    both_periods = ~period_missing_incoming & ~period_missing_payment
    periods_equal = incoming.periods[:, None] == payments.periods[None, :]

    columns = {
        "vendor_tfidf_cosine": _tfidf_cosine(incoming.vendor_vectors, payments.vendor_vectors),
        "vendor_fuzzy": _vendor_fuzzy(incoming.vendors, payments.vendors, config.vendor_scorer),
        "ref_similarity": _ref_similarity(incoming.refs, payments.refs),
        "ref_exact": _ref_exact(incoming.refs, payments.refs),
        "ref_missing_incoming": ref_missing_incoming,
        "ref_missing_payment": ref_missing_payment,
        "amount_exact": _gap(incoming.amounts, payments.amounts) == 0,
        "amount_diff": _amount_diff(incoming, payments, config.amount_transform),
        "date_gap": _date_gap(incoming, payments, config.date_transform),
        "period_same": both_periods & periods_equal,
        "period_different": both_periods & ~periods_equal,
        "period_missing_incoming": period_missing_incoming,
        "period_missing_payment": period_missing_payment,
    }
    for name in config.interactions:
        product = np.ones(shape)
        for part in name.split("*"):
            product = product * columns[part]
        columns[name] = product
    return np.stack([columns[name] for name in feature_names(config)], axis=2, dtype=np.float64)


# --------------------------------------------------------------------------
# Turning pair scores into a review queue
# --------------------------------------------------------------------------

# Pairs scored at a time. A 2,000 x 20,000 upload is 40 million pairs; scoring
# 250,000 at a time keeps the feature cube near 30 MB instead of 4 GB.
MAX_PAIRS_PER_CHUNK = 250_000


def _chunk_bounds(n_incoming: int, n_payments: int) -> list[tuple[int, int]]:
    """Split the incoming rows into (start, stop) slices of bounded pair count."""
    rows = max(1, MAX_PAIRS_PER_CHUNK // max(1, n_payments))
    return [(start, min(start + rows, n_incoming)) for start in range(0, n_incoming, rows)]


def _build_queue(incoming_ids: np.ndarray, payment_ids: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Order the proposals by score (ties keep incoming file order) and number them from 1."""
    order = np.argsort(-scores, kind="stable")
    return pd.DataFrame({
        "incoming_id": pd.Series(incoming_ids[order], dtype="str"),
        "payment_id": pd.Series(payment_ids[order], dtype="str"),
        "score": scores[order].astype(np.float64),
        "rank": np.arange(1, len(order) + 1, dtype=np.int64),
    })


def _empty_queue() -> pd.DataFrame:
    """A queue with the usual columns and no rows."""
    no_ids = np.array([], dtype=object)
    return _build_queue(no_ids, no_ids, np.array([], dtype=np.float64))


def _rank_in_chunks(
    incoming: pd.DataFrame,
    payments: pd.DataFrame,
    prepare: Callable[[pd.DataFrame], _Side],
    score: Callable[[_Side, _Side], np.ndarray],
) -> pd.DataFrame:
    """The ranking loop shared by the learned model and both baselines.

    prepare  turns a non-empty ledger frame into a _Side
    score    returns the (n_incoming, n_payments) score array for two sides,
             with -inf wherever a pair is not eligible

    For each incoming invoice the best-scoring payment is kept. Ties go to the
    most recent payment, then to the earlier row of the payments file. An
    incoming invoice with no eligible payment is left out of the queue.
    """
    if len(incoming) == 0 or len(payments) == 0:
        return _empty_queue()
    payment_side = prepare(payments)
    # Payment columns in tie-break order: newest first, then file order. numpy's
    # argmax returns the first of equal maxima, so it lands on the preferred one.
    payment_days = _day_numbers(payments)
    preference = np.lexsort((np.arange(len(payment_days)), -payment_days))

    best_columns = []
    best_scores = []
    for start, stop in _chunk_bounds(len(incoming), len(payments)):
        scores = score(prepare(incoming.iloc[start:stop]), payment_side)
        preferred = scores[:, preference]
        best_columns.append(preference[preferred.argmax(axis=1)])
        best_scores.append(preferred.max(axis=1))
    best_column = np.concatenate(best_columns)
    best_score = np.concatenate(best_scores)

    eligible = best_score > -np.inf
    incoming_ids = incoming["row_id"].to_numpy(dtype=object)[eligible]
    payment_ids = payments["row_id"].to_numpy(dtype=object)[best_column[eligible]]
    return _build_queue(incoming_ids, payment_ids, best_score[eligible])


# --------------------------------------------------------------------------
# The benchmark: ledgers with an answer key
# --------------------------------------------------------------------------

@dataclass
class BenchLedger:
    """One fictional nonprofit's books at one payment run, with the answer key.

    truth   incoming row_id -> the payment row_ids that count as its earlier
            copy. Only known duplicates appear as keys.
    labels  this ledger's rows of the labels file (scenario and vendor_kind are
            for error analysis only and never reach a model).
    """

    ledger_id: str
    fold: int
    incoming: pd.DataFrame
    payments: pd.DataFrame
    truth: dict[str, frozenset[str]]
    labels: pd.DataFrame


def load_benchmark_split(split: str, data_dir: Path = DATA_DIR) -> list[BenchLedger]:
    """Load one benchmark split as a list of ledgers, in the order of the incoming file.

    split is 'train', 'validation' or 'test' (files in data/synthetic) or
    'challenge' (the hand-authored ledger in data/challenge).
    """
    paths = _split_paths(split, Path(data_dir))
    incoming_by_ledger = _read_ledger_groups(paths["incoming"], "incoming")
    payments_by_ledger = _read_ledger_groups(paths["payments"], "payments")
    labels = pd.read_csv(paths["labels"], dtype=str, keep_default_na=False)
    labels["fold"] = labels["fold"].astype("int64")
    labels["is_duplicate"] = labels["is_duplicate"].astype("int64")

    ledgers = []
    for ledger_id, incoming in incoming_by_ledger.items():
        if ledger_id not in payments_by_ledger:
            raise ValueError(f"{paths['payments'].name} has no rows for ledger {ledger_id}")
        payments = payments_by_ledger[ledger_id]
        rows = labels[labels["ledger_id"] == ledger_id].reset_index(drop=True)
        truth = {
            incoming_id: frozenset(payment_ids.split("|"))
            for incoming_id, payment_ids in zip(rows["incoming_row_id"], rows["true_payment_ids"])
            if payment_ids
        }
        _check_labels(ledger_id, incoming, payments, rows, truth)
        ledgers.append(BenchLedger(ledger_id, int(rows["fold"].iloc[0]), incoming, payments, truth, rows))
    return ledgers


def _split_paths(split: str, data_dir: Path) -> dict[str, Path]:
    """Where the three files of a split live. Says how to create a missing one."""
    if split == "challenge":
        folder, prefix = data_dir / "challenge", ""
        hint = "The challenge ledger is written by hand and ships with the project."
    elif split in ("train", "validation", "test"):
        folder, prefix = data_dir / "synthetic", f"{split}_"
        hint = "Create the synthetic benchmark with: python synth.py"
    else:
        raise ValueError(f"split must be 'train', 'validation', 'test' or 'challenge', not {split!r}")
    paths = {kind: folder / f"{prefix}{kind}.csv" for kind in ("incoming", "payments", "labels")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}. {hint}")
    return paths


def _read_ledger_groups(path: Path, role: str) -> dict[str, pd.DataFrame]:
    """Read a benchmark CSV that holds many ledgers: ledger_id -> ledger frame.

    The rows of each ledger go through ledger.read_ledger_csv on their own, so
    benchmark data is validated by the same code as an upload and the upload
    row limits apply to one ledger, not to the whole file. Ledgers keep the
    order in which they first appear.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    frames = {}
    for ledger_id, rows in raw.groupby("ledger_id", sort=False):
        try:
            frames[ledger_id], _notes = read_ledger_csv(rows.to_csv(index=False).encode("utf-8"), role)
        except LedgerError as error:
            raise LedgerError(f"{path.name}, ledger {ledger_id}: {error.title}", error.problems) from error
    return frames


def _check_labels(ledger_id: str, incoming: pd.DataFrame, payments: pd.DataFrame,
                  rows: pd.DataFrame, truth: dict[str, frozenset[str]]) -> None:
    """Refuse an answer key that does not line up with the ledger it describes."""
    if sorted(rows["incoming_row_id"]) != sorted(incoming["row_id"]):
        raise ValueError(f"The labels of ledger {ledger_id} do not list exactly its incoming invoices.")
    payment_ids = set(payments["row_id"])
    for incoming_id, valid_payments in truth.items():
        if not valid_payments <= payment_ids:
            raise ValueError(f"The labels of ledger {ledger_id} point {incoming_id} at a payment that is not there.")
    if set(rows.loc[rows["is_duplicate"] == 1, "incoming_row_id"]) != set(truth):
        raise ValueError(f"The labels of ledger {ledger_id} disagree: is_duplicate versus true_payment_ids.")


# --------------------------------------------------------------------------
# The learned model
# --------------------------------------------------------------------------

class LearnedMatcher:
    """Logistic regression that scores how much a prior payment looks like an incoming invoice.

    config             the MatcherConfig it was fitted with
    feature_names      the model's input columns, in order
    vectorizer         TF-IDF over character n-grams of training vendor names
    scaler             StandardScaler fitted on the training pairs
    classifier         the fitted LogisticRegression
    frozen             True when the config came from results/frozen_config.json
    n_train_pairs      how many pairs the model was fitted on
    n_train_positives  how many of those were true duplicate pairs
    """

    def __init__(self, config: MatcherConfig, vectorizer: TfidfVectorizer, scaler: StandardScaler,
                 classifier: LogisticRegression, n_train_pairs: int, n_train_positives: int) -> None:
        """Hold the fitted parts. Use LearnedMatcher.fit to make one."""
        self.config = config
        self.feature_names = feature_names(config)
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.classifier = classifier
        self.frozen = False
        self.n_train_pairs = n_train_pairs
        self.n_train_positives = n_train_positives

    @classmethod
    def fit(cls, ledgers: Sequence[BenchLedger], config: MatcherConfig = DEFAULT_CONFIG) -> "LearnedMatcher":
        """Fit on every incoming x payment pair inside each training ledger.

        A pair is labelled 1 when the payment is in the incoming invoice's
        truth set, else 0. Pairs are never formed across ledgers. The same
        ledgers and config always give the same model.
        """
        vectorizer, features, labels = training_pairs(ledgers, config)
        return cls.fit_pairs(config, vectorizer, features, labels)

    @classmethod
    def fit_pairs(cls, config: MatcherConfig, vectorizer: TfidfVectorizer,
                  features: np.ndarray, labels: np.ndarray) -> "LearnedMatcher":
        """Fit the scaler and the classifier on a training table made by training_pairs.

        This is the second half of fit. It stands alone so that evaluate.py can
        try several values of C and class_weight on one table instead of
        building the same features again for each.
        """
        if labels.min() == labels.max():
            raise ValueError("LearnedMatcher.fit needs both duplicate and non-duplicate pairs.")
        scaler = StandardScaler().fit(features)
        classifier = LogisticRegression(C=config.C, class_weight=config.class_weight, solver="lbfgs", max_iter=2000)
        classifier.fit(scaler.transform(features), labels)
        return cls(config, vectorizer, scaler, classifier,
                   n_train_pairs=len(labels), n_train_positives=int(labels.sum()))

    def _side(self, frame: pd.DataFrame) -> _Side:
        """Normalize a non-empty ledger frame and add the TF-IDF rows of its vendor names."""
        side = _prepare_side(frame, self.config.vendor_text)
        side.vendor_vectors = self.vectorizer.transform(side.vendors)
        return side

    def _score_sides(self, incoming: _Side, payments: _Side) -> np.ndarray:
        """Decision values (log-odds) for every pair of two prepared sides.

        This is the classifier's decision_function written out: each pair's
        contributions are added up row by row and the intercept is added. Done
        this way, two pairs with identical features get exactly the same score
        wherever they sit in the array, so ties can be broken by date as promised.
        """
        cube = _pair_features(incoming, payments, self.config)
        standardized = self.scaler.transform(cube.reshape(-1, cube.shape[2]))
        decision = (standardized * self.classifier.coef_[0]).sum(axis=1) + self.classifier.intercept_[0]
        return decision.reshape(cube.shape[0], cube.shape[1])

    def pair_features(self, incoming: pd.DataFrame, payments: pd.DataFrame) -> np.ndarray:
        """Raw (unscaled) features, shape (n_incoming, n_payments, n_features).

        Meant for inspection and tests: it builds the whole cube at once.
        """
        if len(incoming) == 0 or len(payments) == 0:
            return np.zeros((len(incoming), len(payments), len(self.feature_names)))
        return _pair_features(self._side(incoming), self._side(payments), self.config)

    def score_matrix(self, incoming: pd.DataFrame, payments: pd.DataFrame) -> np.ndarray:
        """Decision values (log-odds), shape (n_incoming, n_payments).

        Works through the incoming rows in chunks so memory stays bounded.
        """
        if len(incoming) == 0 or len(payments) == 0:
            return np.zeros((len(incoming), len(payments)))
        payment_side = self._side(payments)
        blocks = [
            self._score_sides(self._side(incoming.iloc[start:stop]), payment_side)
            for start, stop in _chunk_bounds(len(incoming), len(payments))
        ]
        return np.vstack(blocks)

    def rank(self, incoming: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
        """The review queue: every incoming invoice with its best prior payment, strongest first."""
        return _rank_in_chunks(incoming, payments, self._side, self._score_sides)

    def contributions(self, incoming_row: pd.Series, payment_row: pd.Series) -> pd.DataFrame:
        """How each feature moved the score of one pair, in log-odds.

        contribution = coefficient x standardized value, so a feature at its
        training average contributes 0. The contributions plus the intercept
        (in attrs['intercept']) add up to the pair's decision value.
        """
        values = self.pair_features(pd.DataFrame([incoming_row]), pd.DataFrame([payment_row]))[0, 0]
        standardized = self.scaler.transform(values.reshape(1, -1))[0]
        coefficients = self.classifier.coef_[0]
        table = pd.DataFrame({
            "feature": self.feature_names,
            "value": values,
            "standardized_value": standardized,
            "coefficient": coefficients,
            "contribution": coefficients * standardized,
        })
        table.attrs["intercept"] = float(self.classifier.intercept_[0])
        return table


def training_pairs(ledgers: Sequence[BenchLedger], config: MatcherConfig = DEFAULT_CONFIG,
                   ) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """The training table of the learned model: (vectorizer, features, labels).

    The table has one row per incoming x payment pair inside each ledger;
    pairs are never formed across ledgers. features has one column per name in
    feature_names(config); labels holds 1 where the payment is in the incoming
    invoice's truth set, else 0. The vectorizer learns its character n-grams
    from the vendor names of these ledgers only.
    """
    sides = []
    for ledger in ledgers:
        if len(ledger.incoming) > 0 and len(ledger.payments) > 0:
            incoming_side = _prepare_side(ledger.incoming, config.vendor_text)
            payment_side = _prepare_side(ledger.payments, config.vendor_text)
            sides.append((ledger, incoming_side, payment_side))
    if not sides:
        raise ValueError("LearnedMatcher.fit needs at least one ledger with invoices and payments.")

    training_names = set()
    for _ledger, incoming_side, payment_side in sides:
        training_names.update(incoming_side.vendors.tolist())
        training_names.update(payment_side.vendors.tolist())
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(sorted(training_names))

    feature_blocks = []
    label_blocks = []
    for ledger, incoming_side, payment_side in sides:
        incoming_side.vendor_vectors = vectorizer.transform(incoming_side.vendors)
        payment_side.vendor_vectors = vectorizer.transform(payment_side.vendors)
        cube = _pair_features(incoming_side, payment_side, config)
        feature_blocks.append(cube.reshape(-1, cube.shape[2]))
        label_blocks.append(_pair_labels(ledger).ravel())
    return vectorizer, np.vstack(feature_blocks), np.concatenate(label_blocks)


def _pair_labels(ledger: BenchLedger) -> np.ndarray:
    """Training labels, shape (n_incoming, n_payments): 1 where the payment is a true earlier copy."""
    labels = np.zeros((len(ledger.incoming), len(ledger.payments)), dtype=np.int64)
    column_of = {payment_id: column for column, payment_id in enumerate(ledger.payments["row_id"])}
    for row, incoming_id in enumerate(ledger.incoming["row_id"]):
        for payment_id in ledger.truth.get(incoming_id, ()):
            labels[row, column_of[payment_id]] = 1
    return labels


def fit_shipped_matcher(root: Path = PROJECT_ROOT) -> LearnedMatcher:
    """Fit the model the app uses: the train split with the frozen config when there is one.

    evaluate.py writes results/frozen_config.json after validation tuning.
    Until then the default config is used and the matcher says frozen=False.
    """
    root = Path(root)
    config_path = root / "results" / "frozen_config.json"
    config = DEFAULT_CONFIG
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        config = MatcherConfig.from_dict(saved["matcher"])
    fitted = LearnedMatcher.fit(load_benchmark_split("train", root / "data"), config)
    fitted.frozen = config_path.exists()
    return fitted


def match_score(decision_value: float) -> float:
    """Squash a decision value into (0, 1) with the logistic function.

    This is a ranking score for display, not a calibrated probability. It is
    kept strictly inside (0, 1) so that rounding can never show a certainty.
    """
    value = float(decision_value)
    if value >= 0:
        score = 1.0 / (1.0 + math.exp(-value))
    else:
        score = math.exp(value) / (1.0 + math.exp(value))
    return min(max(score, 1e-9), 1.0 - 1e-9)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def _core_side(frame: pd.DataFrame) -> _Side:
    """Both baselines compare vendor_core names, the strongest normalization on offer."""
    return _prepare_side(frame, "core")


def _exact_match_scores(incoming: _Side, payments: _Side) -> np.ndarray:
    """Eligible: same core vendor, same non-blank reference, same cents. Closest date scores highest."""
    eligible = (
        (incoming.vendors[:, None] == payments.vendors[None, :])
        & _ref_exact(incoming.refs, payments.refs)
        & (_gap(incoming.amounts, payments.amounts) == 0)
    )
    return np.where(eligible, -_gap(incoming.days, payments.days).astype(np.float64), -np.inf)


def exact_match_queue(incoming: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """Baseline 1: flag an invoice only when vendor, reference and amount all repeat exactly."""
    return _rank_in_chunks(incoming, payments, _core_side, _exact_match_scores)


def _amount_rule_holds(incoming: _Side, payments: _Side, amount_rule: str) -> np.ndarray:
    """True where the two amounts are close enough for the fuzzy rule."""
    gap_cents = _gap(incoming.amounts, payments.amounts)
    if amount_rule == "exact":
        return gap_cents == 0
    if amount_rule == "within_100_cents":
        return gap_cents <= 100
    larger = np.maximum(incoming.amounts[:, None], payments.amounts[None, :])
    return gap_cents * 100 <= larger  # within 1 percent, in whole-number arithmetic


def fuzzy_rule_queue(incoming: pd.DataFrame, payments: pd.DataFrame,
                     rule: FuzzyRule = DEFAULT_FUZZY_RULE) -> pd.DataFrame:
    """Baseline 2: flag an invoice when a payment has a similar vendor name and a matching amount.

    Score = vendor similarity, plus 1 for an equal reference when the rule
    ranks by reference first, minus a tiny penalty per day apart so that the
    closest date wins a tie.
    """
    def score(incoming_side: _Side, payment_side: _Side) -> np.ndarray:
        """Rule scores for every pair, -inf where the rule does not fire."""
        similarity = _vendor_fuzzy(incoming_side.vendors, payment_side.vendors, rule.vendor_scorer)
        amount_ok = _amount_rule_holds(incoming_side, payment_side, rule.amount_rule)
        eligible = (similarity >= rule.vendor_threshold) & amount_ok
        scores = similarity - _gap(incoming_side.days, payment_side.days) / 100000
        if rule.rank_by == "ref_then_vendor":
            scores = scores + _ref_exact(incoming_side.refs, payment_side.refs)
        return np.where(eligible, scores, -np.inf)

    return _rank_in_chunks(incoming, payments, _core_side, score)


# --------------------------------------------------------------------------
# Evidence text for the reviewer
# --------------------------------------------------------------------------
# Every sentence below is a fixed template filled in from a plain comparison of
# two field values. The learned model is not consulted.

BLANK_DISPLAY = "\u2014"  # an em dash stands for an empty field
SIMILAR_THRESHOLD = 0.80  # similarity from which two different texts are called "similar"
NEAR_DATE_DAYS = 7  # invoice dates this close still count as partial support

SUPPORT_PRIORITY = ("invoice_ref", "amount", "billing_period", "vendor", "invoice_date")
DIFFERENCE_PRIORITY = ("billing_period", "invoice_ref", "amount", "vendor", "invoice_date")


@dataclass(frozen=True)
class FieldComparison:
    """One field of an incoming invoice set against the same field of a prior payment.

    field             'vendor', 'invoice_ref', 'amount', 'invoice_date' or 'billing_period'
    label             the field's name as shown to the reviewer
    status            'match', 'partial', 'differ' or 'missing'
    evidence          the sentence that describes the comparison
    incoming_display  the incoming value as shown (an em dash when blank)
    payment_display   the payment value as shown (an em dash when blank)
    incoming_note     'Normalized: ...' when normalization changed the value, else ''
    payment_note      the same for the payment
    """

    field: str
    label: str
    status: str
    evidence: str
    incoming_display: str
    payment_display: str
    incoming_note: str
    payment_note: str


def compare_fields(incoming_row: pd.Series, payment_row: pd.Series) -> list[FieldComparison]:
    """Compare one incoming invoice with one prior payment, field by field.

    Always returns five comparisons in the order vendor, invoice_ref, amount,
    invoice_date, billing_period.
    """
    return [
        _compare_vendor(incoming_row["vendor"], payment_row["vendor"]),
        _compare_ref(incoming_row["invoice_ref"], payment_row["invoice_ref"]),
        _compare_amount(int(incoming_row["amount_cents"]), int(payment_row["amount_cents"])),
        _compare_date(incoming_row["invoice_date"], payment_row["invoice_date"]),
        _compare_period(incoming_row["billing_period"], payment_row["billing_period"]),
    ]


def _display(text: str) -> str:
    """Text as shown to the reviewer; a blank becomes an em dash."""
    return text if text.strip() else BLANK_DISPLAY


def _normalization_note(raw: str, normalized: str) -> str:
    """Say what a value was compared as, unless only its letter case changed."""
    if normalized == "" or normalized.casefold() == raw.strip().casefold():
        return ""
    return f"Normalized: {normalized}"


def _compare_vendor(incoming_vendor: str, payment_vendor: str) -> FieldComparison:
    """Same as typed, same after normalization, similar (partial) or different."""
    incoming_core = vendor_core(incoming_vendor)
    payment_core = vendor_core(payment_vendor)
    if incoming_vendor == payment_vendor:
        status, evidence = "match", "Same vendor name"
    elif incoming_core == payment_core:
        status, evidence = "match", "Vendor names match after normalization"
    elif fuzz.token_sort_ratio(incoming_core, payment_core) / 100.0 >= SIMILAR_THRESHOLD:
        status, evidence = "partial", "Similar vendor names"
    else:
        status, evidence = "differ", "Different vendor names"
    return FieldComparison(
        "vendor", "Vendor", status, evidence,
        _display(incoming_vendor), _display(payment_vendor),
        _normalization_note(incoming_vendor, incoming_core),
        _normalization_note(payment_vendor, payment_core),
    )


def _compare_ref(incoming_ref: str, payment_ref: str) -> FieldComparison:
    """Missing on a side, same as typed, same after normalization, similar (partial) or different."""
    incoming_normal = normalize_ref(incoming_ref)
    payment_normal = normalize_ref(payment_ref)
    if incoming_normal == "" and payment_normal == "":
        status, evidence = "missing", "Invoice reference missing on both"
    elif incoming_normal == "":
        status, evidence = "missing", "Invoice reference missing on incoming invoice"
    elif payment_normal == "":
        status, evidence = "missing", "Invoice reference missing on prior payment"
    elif incoming_ref == payment_ref:
        status, evidence = "match", "Same invoice reference"
    elif incoming_normal == payment_normal:
        status, evidence = "match", "References match after normalization"
    elif OSA.normalized_similarity(incoming_normal, payment_normal) >= SIMILAR_THRESHOLD:
        status, evidence = "partial", "Similar invoice references"
    else:
        status, evidence = "differ", "Different invoice references"
    return FieldComparison(
        "invoice_ref", "Invoice reference", status, evidence,
        _display(incoming_ref), _display(payment_ref),
        _normalization_note(incoming_ref, incoming_normal),
        _normalization_note(payment_ref, payment_normal),
    )


def _compare_amount(incoming_cents: int, payment_cents: int) -> FieldComparison:
    """Same amount, or the difference in dollars and cents."""
    if incoming_cents == payment_cents:
        status, evidence = "match", "Same amount"
    else:
        status, evidence = "differ", f"Amounts differ by {format_cents(abs(incoming_cents - payment_cents))}"
    return FieldComparison(
        "amount", "Amount", status, evidence,
        format_cents(incoming_cents), format_cents(payment_cents), "", "",
    )


def _compare_date(incoming_date: Any, payment_date: Any) -> FieldComparison:
    """Same date, a few days apart (partial) or further apart (differ)."""
    days_apart = abs((pd.Timestamp(incoming_date) - pd.Timestamp(payment_date)).days)
    if days_apart == 0:
        status, evidence = "match", "Same invoice date"
    else:
        status = "partial" if days_apart <= NEAR_DATE_DAYS else "differ"
        unit = "day" if days_apart == 1 else "days"
        evidence = f"Invoice dates {days_apart} {unit} apart"
    return FieldComparison(
        "invoice_date", "Invoice date", status, evidence,
        format_date(incoming_date), format_date(payment_date), "", "",
    )


def _compare_period(incoming_period: str, payment_period: str) -> FieldComparison:
    """Missing on a side, the same service month or a different one."""
    if incoming_period == "" and payment_period == "":
        status, evidence = "missing", "Billing period missing on both"
    elif incoming_period == "":
        status, evidence = "missing", "Billing period missing on incoming invoice"
    elif payment_period == "":
        status, evidence = "missing", "Billing period missing on prior payment"
    elif incoming_period == payment_period:
        status, evidence = "match", "Same billing period"
    else:
        status, evidence = "differ", "Different billing period"
    return FieldComparison(
        "billing_period", "Billing period", status, evidence,
        _display(format_period(incoming_period)), _display(format_period(payment_period)), "", "",
    )


def _statement(comparison: FieldComparison) -> str:
    """The evidence sentence with its values: 'Same amount ($468.20)' or '... (A and B)'."""
    if comparison.incoming_display == comparison.payment_display:
        return f"{comparison.evidence} ({comparison.incoming_display})"
    return f"{comparison.evidence} ({comparison.incoming_display} and {comparison.payment_display})"


def _in_priority_order(comparisons: Sequence[FieldComparison], priority: Sequence[str]) -> list[FieldComparison]:
    """The comparisons sorted by a list of field names."""
    by_field = {comparison.field: comparison for comparison in comparisons}
    return [by_field[field] for field in priority if field in by_field]


def flag_reasons(comparisons: Sequence[FieldComparison], max_items: int = 3) -> list[str]:
    """Up to max_items statements that support a match, the most telling field first."""
    supporting = [
        _statement(comparison)
        for comparison in _in_priority_order(comparisons, SUPPORT_PRIORITY)
        if comparison.status in ("match", "partial")
    ]
    return supporting[:max_items]


def difference_notes(comparisons: Sequence[FieldComparison], max_items: int = 3) -> list[str]:
    """Up to max_items statements against a match: fields that differ, then fields that cannot be compared."""
    ordered = _in_priority_order(comparisons, DIFFERENCE_PRIORITY)
    differing = [_statement(comparison) for comparison in ordered if comparison.status == "differ"]
    missing = [comparison.evidence for comparison in ordered if comparison.status == "missing"]
    return (differing + missing)[:max_items]
