"""Tests for synth.py: determinism, split hygiene, structure, schema and formatting tells.

The held-out test split is never generated or read here. Only its vendor pool
is built (names, no ledgers) to prove that the three splits share no vendor words.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import synth

OPEN_SPLITS = ("train", "validation")          # the splits tests may generate
LEGAL_SUFFIXES = {"llc", "inc", "incorporated", "co", "company", "corp", "corporation", "ltd", "limited"}
SCENARIOS = {
    "legit_recurring_next", "legit_same_amount_repeat", "legit_similar_vendor", "legit_new_vendor",
    "legit_stale", "legit_one_off", "dup_exact_resubmission", "dup_rekeyed", "dup_reissued_date",
    "dup_revised_ref", "dup_late_fee", "dup_statement_no_ref",
}


# --------------------------------------------------------------------------
# Shared data (generated once per test run, in memory)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def splits() -> dict[str, tuple]:
    """split -> (incoming, payments, labels, manifest entry) for train and validation."""
    return {split: synth.generate_split(split) for split in OPEN_SPLITS}


@pytest.fixture(scope="module")
def pools() -> dict[str, list[synth.Vendor]]:
    """The vendor pool of every split. Building the test pool creates names only, no ledgers."""
    return {split: synth.vendor_pool(split) for split in synth.SPLITS}


@pytest.fixture(scope="module")
def demo() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    return synth.build_demo()


def words(text: str) -> list[str]:
    """Lower-case words of a name, without accents, digits or punctuation."""
    return [synth.strip_accents(word).casefold() for word in re.findall(r"[^\W\d_]+", text)]


def generic_words() -> set[str]:
    """Words any vendor name may contain: trades, legal suffixes and their abbreviations."""
    phrases = [trade for pairs in synth.TRADES.values() for pair in pairs for trade in pair]
    phrases += [trade for trades in synth.PERSON_TRADES.values() for trade in trades]
    phrases += [*synth.LEGAL_FORMS, *(other for others in synth.LEGAL_FORMS.values() for other in others)]
    phrases += list(synth.ABBREVIATIONS.values())
    return set(words(" ".join(phrases))) | {"and"}


def distinctive_words(names: list[str]) -> set[str]:
    """The words that identify a vendor: everything except generic words and initials."""
    generic = generic_words()
    return {word for name in names for word in words(name) if len(word) >= 3 and word not in generic}


def split_words(split: str) -> set[str]:
    """Every word reserved for one split."""
    lists = (synth.ORG_FIRST_WORDS, synth.ORG_SECOND_WORDS, synth.LAST_NAMES, synth.FIRST_NAMES)
    return {word for word_list in lists for name in word_list[split] for word in words(name)}


def spellings(pool: list[synth.Vendor]) -> list[str]:
    """Every way the vendors of a pool can be written."""
    return [text for vendor in pool for text in (vendor.name, *vendor.aliases)]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def assert_same_ledger(one: tuple, two: tuple) -> None:
    for frame_one, frame_two in zip(one[:3], two[:3]):
        pd.testing.assert_frame_equal(frame_one, frame_two)
    assert one[3] == two[3]


def test_generating_a_ledger_twice_gives_identical_frames():
    assert_same_ledger(synth.generate_ledger("validation", 4), synth.generate_ledger("validation", 4))


def test_train_does_not_depend_on_validation_being_generated_first():
    before = synth.generate_ledger("train", 7)
    synth.generate_ledger("validation", 2)
    assert_same_ledger(before, synth.generate_ledger("train", 7))


def test_a_ledger_from_generate_split_equals_the_ledger_generated_alone(splits):
    incoming, payments, labels, entry = splits["validation"]
    alone = synth.generate_ledger("validation", 9)
    ledger_id = alone[3]["ledger_id"]
    pd.testing.assert_frame_equal(incoming[incoming["ledger_id"] == ledger_id].reset_index(drop=True), alone[0])
    pd.testing.assert_frame_equal(payments[payments["ledger_id"] == ledger_id].reset_index(drop=True), alone[1])
    assert entry["ledgers"][9] == alone[3]


def test_output_does_not_depend_on_the_python_hash_seed():
    """Set and dict ordering changes between Python processes; the generated data must not."""
    code = (
        "import hashlib, synth\n"
        "frames = synth.generate_ledger('validation', 0)[:3]\n"
        "text = ''.join(frame.to_csv(index=False) for frame in frames)\n"
        "print(hashlib.sha256(text.encode('utf-8')).hexdigest())\n"
    )
    digests = set()
    for hash_seed in ("1", "2"):
        result = subprocess.run([sys.executable, "-c", code], cwd=synth.PROJECT_ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONHASHSEED": hash_seed}, check=True)
        digests.add(result.stdout.strip())
    assert len(digests) == 1


def test_demo_is_deterministic(demo):
    again = synth.build_demo()
    pd.testing.assert_frame_equal(demo[0], again[0])
    pd.testing.assert_frame_equal(demo[1], again[1])
    assert demo[2] == again[2]


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_files_on_disk_are_what_the_generator_writes_now(split, tmp_path):
    folder = synth.DATA_DIR / "synthetic"
    names = [f"{split}_incoming.csv", f"{split}_payments.csv", f"{split}_labels.csv"]
    if not all((folder / name).exists() for name in names):
        pytest.skip("data/synthetic is not generated yet: run python synth.py")
    entry = synth.write_split(split, tmp_path)
    for name in names:
        assert (tmp_path / "synthetic" / name).read_bytes() == (folder / name).read_bytes(), name
    assert json.loads((folder / "manifest.json").read_text(encoding="utf-8"))[split] == entry


def test_demo_files_on_disk_are_what_the_generator_writes_now(tmp_path):
    folder = synth.DATA_DIR / "demo"
    names = ["incoming.csv", "payments.csv", "cases.json"]
    if not all((folder / name).exists() for name in names):
        pytest.skip("data/demo is not generated yet: run python synth.py --demo")
    synth.write_demo(tmp_path)
    for name in names:
        assert (tmp_path / "demo" / name).read_bytes() == (folder / name).read_bytes(), name


def test_manifest_keeps_other_splits_and_has_no_timestamp(tmp_path):
    folder = tmp_path / "synthetic"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"train": {"kept": True}}), encoding="utf-8")
    entry = synth.write_split("validation", tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"train": {"kept": True}, "validation": entry}
    assert set(entry) == {"seed", "date_range", "ledgers"}
    assert set(entry["ledgers"][0]) == {"ledger_id", "fold", "run_date", "incoming", "payments", "duplicates"}


def test_default_command_never_writes_the_test_split():
    default = synth.parse_args([])
    assert default.splits == ["train", "validation"] and default.demo is True
    assert synth.parse_args(["--demo"]).splits == []
    named = synth.parse_args(["--splits", "train", "validation", "test"])
    assert named.splits == ["train", "validation", "test"] and named.demo is False


# --------------------------------------------------------------------------
# Split hygiene
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word_lists", [synth.ORG_FIRST_WORDS, synth.ORG_SECOND_WORDS, synth.LAST_NAMES,
                                        synth.FIRST_NAMES], ids=["first words", "second words", "last names",
                                                                 "first names"])
def test_word_lists_are_disjoint_across_splits(word_lists):
    sets = {split: {word for name in word_lists[split] for word in words(name)} for split in synth.SPLITS}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert all(len(sets[split]) == len(word_lists[split]) for split in synth.SPLITS)   # no repeats inside a list


def test_word_list_sizes():
    assert [len(synth.ORG_FIRST_WORDS[split]) for split in synth.SPLITS] == [72, 24, 24]
    assert [len(synth.LAST_NAMES[split]) for split in synth.SPLITS] == [36, 12, 12]


def test_no_word_is_reserved_twice_or_is_a_generic_word():
    everything = [word for split in synth.SPLITS
                  for word_list in (synth.ORG_FIRST_WORDS, synth.ORG_SECOND_WORDS, synth.LAST_NAMES, synth.FIRST_NAMES)
                  for name in word_list[split] for word in words(name)]
    assert len(everything) == len(set(everything))
    assert not set(everything) & generic_words()


def test_group_keys_do_not_cross_splits(pools):
    groups = {split: {vendor.group for vendor in pool} for split, pool in pools.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    for split in synth.SPLITS:
        assert groups[split] == set(synth.ORG_FIRST_WORDS[split]) | set(synth.LAST_NAMES[split])


def test_vendor_names_share_only_generic_words_across_splits(pools):
    distinctive = {split: distinctive_words(spellings(pool)) for split, pool in pools.items()}
    for split in synth.SPLITS:
        assert distinctive[split] <= split_words(split)            # only words reserved for this split
    assert not distinctive["train"] & distinctive["validation"]
    assert not distinctive["train"] & distinctive["test"]
    assert not distinctive["validation"] & distinctive["test"]


def test_demo_vendor_words_appear_in_no_pool(pools, demo):
    incoming, payments, _cases = demo
    demo_words = distinctive_words([*incoming["vendor"], *payments["vendor"]])
    assert {"lantern", "street", "bayside"} <= demo_words
    for split, pool in pools.items():
        assert not demo_words & split_words(split), split
        assert not demo_words & distinctive_words(spellings(pool)), split


def test_split_date_ranges_are_ordered():
    train, validation, test = (synth.SPLITS[split] for split in ("train", "validation", "test"))
    assert train.first_day < train.last_day < validation.first_day < validation.last_day
    assert validation.last_day < test.first_day < test.last_day
    assert (train.first_day, train.last_day) == (date(2023, 1, 1), date(2024, 8, 31))
    assert (validation.first_day, validation.last_day) == (date(2024, 9, 1), date(2025, 6, 30))
    assert (test.first_day, test.last_day) == (date(2025, 7, 1), date(2026, 6, 30))


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_every_date_lies_inside_the_split_range(splits, split):
    incoming, payments, _labels, entry = splits[split]
    first, last = synth.SPLITS[split].first_day.isoformat(), synth.SPLITS[split].last_day.isoformat()
    assert entry["date_range"] == [first, last]
    for frame in (incoming, payments):
        assert first <= frame["invoice_date"].min() and frame["invoice_date"].max() <= last
        periods = frame.loc[frame["billing_period"] != "", "billing_period"]
        assert first[:7] <= periods.min() and periods.max() <= last[:7]
    run_dates = {ledger["ledger_id"]: ledger["run_date"] for ledger in entry["ledgers"]}
    assert all(first <= run_date <= last for run_date in run_dates.values())
    for frame in (incoming, payments):                     # nothing is dated after its ledger's payment run
        assert (frame["invoice_date"] <= frame["ledger_id"].map(run_dates)).all()
    oldest_allowed = pd.to_datetime(payments["ledger_id"].map(run_dates)) - pd.Timedelta(days=synth.HISTORY_DAYS)
    assert (pd.to_datetime(payments["invoice_date"]) >= oldest_allowed).all()
    assert (payments["invoice_date"] < payments["ledger_id"].map(run_dates)).all()


# --------------------------------------------------------------------------
# Vendor pools
# --------------------------------------------------------------------------

def test_pool_groups_and_confusable_pairs(pools):
    for split, pool in pools.items():
        by_id = {vendor.vendor_id: vendor for vendor in pool}
        assert len(by_id) == len(pool)
        organizations = [vendor for vendor in pool if vendor.kind == "organization"]
        sizes = pd.Series([vendor.group for vendor in organizations]).value_counts()
        assert sizes.min() >= 3 and sizes.max() <= 6
        pairs = synth.sibling_pairs(pool)
        n_groups = len(synth.ORG_FIRST_WORDS[split]) + len(synth.LAST_NAMES[split])
        assert 0.25 <= len(pairs) / n_groups <= 0.35           # about 30% of groups hold a confusable pair
        for one, two in pairs:
            assert by_id[one.sibling] is two and by_id[two.sibling] is one
            assert one.group == two.group and one.kind == two.kind and one.name != two.name


def test_no_spelling_can_mean_two_different_vendors(pools):
    for pool in pools.values():
        owner: dict[str, str] = {}
        for vendor in pool:
            for text in (vendor.name, *vendor.aliases):
                assert owner.setdefault(synth.name_key(text), vendor.vendor_id) == vendor.vendor_id, text


def test_initials_that_read_like_a_brand_are_not_used(pools):
    for pool in pools.values():
        for text in spellings(pool):
            assert text.split(" ")[0].replace(".", "") not in synth.BRAND_LIKE_INITIALS, text


# --------------------------------------------------------------------------
# The noise channel
# --------------------------------------------------------------------------

def sample_invoice() -> synth.Invoice:
    vendor = synth.make_organization("unit:test", "Cedar", "Hollow", "Printing", "one_off", synth.make_rng("unit"))
    vendor = replace(vendor, ref_scheme="prefixed", ref_prefix="INV-")
    return synth.Invoice(vendor, "one_off", date(2024, 3, 6), 46820, "2024-03", ref_body="1042")


def test_render_uses_the_documented_rates():
    invoice, rng = sample_invoice(), synth.make_rng("unit", "render")
    records = [synth.render(invoice, rng) for _ in range(6000)]
    assert 0.57 <= sum(record.vendor_text == invoice.vendor.name for record in records) / 6000 <= 0.63
    assert 0.67 <= sum(record.invoice_ref == "INV-1042" for record in records) / 6000 <= 0.73
    assert 0.055 <= sum(record.invoice_ref == "" for record in records) / 6000 <= 0.085
    assert 0.82 <= sum(record.billing_period == "2024-03" for record in records) / 6000 <= 0.88
    assert {record.vendor_text for record in records} == {invoice.vendor.name, *invoice.vendor.aliases}
    kept = [synth.render(invoice, rng, keep_reference=True) for _ in range(2000)]
    assert all(record.invoice_ref != "" for record in kept)


def test_reference_variants_and_typos():
    assert set(synth.reference_variants("INV-", "1042")) == {"1042", "INV1042", "INV 1042", "inv-1042", "#1042",
                                                           "INV#1042"}
    assert "418" in synth.reference_variants("", "000418")           # leading zeros are sometimes lost
    assert "INV-000418" in synth.reference_variants("", "000418")
    assert "CH-7731" not in synth.reference_variants("CH-", "7731")  # the canonical spelling is not a variant
    rng = synth.make_rng("unit", "typo")
    for _ in range(200):
        typo = synth.add_typo("INV-1042", rng)
        changed = [position for position in range(8) if typo[position] != "INV-1042"[position]]
        assert len(typo) == 8 and typo.startswith("INV-") and len(changed) in (1, 2)


def test_reference_body_keeps_leading_zeros_and_codes_the_date():
    assert synth.reference_body("zero_padded", 418, date(2025, 3, 6)) == "000418"
    assert synth.reference_body("date_coded", 3418, date(2025, 3, 6)) == "2503-418"
    assert synth.reference_body("bare", 1042, date(2025, 3, 6)) == "1042"


# --------------------------------------------------------------------------
# Structure of the benchmark files
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_ledger_sizes(splits, split):
    incoming, payments, labels, entry = splits[split]
    assert len(entry["ledgers"]) == synth.SPLITS[split].n_ledgers
    assert (incoming.groupby("ledger_id").size() == 60).all()
    assert payments.groupby("ledger_id").size().between(220, 280).all()
    for ledger in entry["ledgers"]:
        assert ledger["incoming"] == 60
        assert ledger["payments"] == (payments["ledger_id"] == ledger["ledger_id"]).sum()
        assert ledger["duplicates"] == labels.loc[labels["ledger_id"] == ledger["ledger_id"], "is_duplicate"].sum()


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_every_fold_of_six_ledgers_has_exactly_twenty_duplicates(splits, split):
    labels = splits[split][2]
    per_ledger = labels.groupby("ledger_id", sort=False)["is_duplicate"].sum().tolist()
    assert per_ledger == list(synth.DUPLICATES_PER_LEDGER) * (len(per_ledger) // 6)
    assert (labels.groupby("fold")["ledger_id"].nunique() == 6).all()
    assert (labels.groupby("fold")["is_duplicate"].sum() == 20).all()
    assert sorted(labels["fold"].unique()) == list(range(1, len(per_ledger) // 6 + 1))


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_duplicate_types_follow_the_mix_in_every_fold(splits, split):
    labels = splits[split][2]
    assert set(labels["scenario"]) == SCENARIOS
    duplicates = labels[labels["is_duplicate"] == 1]
    for _fold, rows in duplicates.groupby("fold"):
        assert rows["scenario"].value_counts().to_dict() == dict(synth.DUPLICATE_MIX)
        assert rows["true_payment_ids"].str.contains("|", regex=False).sum() == 1    # one invoice was paid twice


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_labels_match_the_ledgers(splits, split):
    incoming, payments, labels, _entry = splits[split]
    assert labels["incoming_row_id"].tolist() == incoming["row_id"].tolist()
    assert labels["ledger_id"].tolist() == incoming["ledger_id"].tolist()
    assert (labels["is_duplicate"] == labels["scenario"].str.startswith("dup_").astype(int)).all()
    assert (labels.loc[labels["is_duplicate"] == 0, "true_payment_ids"] == "").all()
    assert (labels.loc[labels["is_duplicate"] == 1, "true_payment_ids"] != "").all()
    assert set(labels["vendor_kind"]) == {"organization", "individual"}
    ledger_of_payment = dict(zip(payments["row_id"], payments["ledger_id"]))
    for ledger_id, truth in zip(labels["ledger_id"], labels["true_payment_ids"]):
        for payment_id in filter(None, truth.split("|")):
            assert ledger_of_payment[payment_id] == ledger_id


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_ids_columns_and_formats(splits, split):
    incoming, payments, labels, _entry = splits[split]
    assert list(incoming.columns) == synth.LEDGER_COLUMNS        # no label or generator columns in a ledger
    assert list(payments.columns) == synth.LEDGER_COLUMNS
    assert list(labels.columns) == synth.LABEL_COLUMNS
    for frame, letter in ((incoming, "I"), (payments, "P")):
        assert frame["row_id"].is_unique
        assert frame["row_id"].str.fullmatch(r"[A-Z]{2}\d{3}-" + letter + r"-\d{4}").all()
        assert (frame["row_id"].str[:5] == frame["ledger_id"]).all()
        assert frame["amount"].str.fullmatch(r"[1-9]\d*\.\d{2}").all()
        assert frame["invoice_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
        assert frame["billing_period"].str.fullmatch(r"(\d{4}-\d{2})?").all()
        assert (frame["vendor"].str.strip() == frame["vendor"]).all() and (frame["vendor"] != "").all()
    ordered = payments.sort_values(["ledger_id", "invoice_date", "row_id"], kind="stable")
    assert ordered.index.tolist() == payments.index.tolist()     # payments: by date, then id


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_duplicates_come_from_the_same_vendor_mix_and_ids_do_not_reveal_them(splits, split):
    labels = splits[split][2]
    people = labels["vendor_kind"] == "individual"
    assert 0.20 <= people.mean() <= 0.33                          # roughly a quarter of the vendors
    assert abs(people[labels["is_duplicate"] == 1].mean() - people[labels["is_duplicate"] == 0].mean()) <= 0.10
    position = labels["incoming_row_id"].str[-4:].astype(int)
    assert 0.40 <= roc_auc_score(labels["is_duplicate"], position) <= 0.60


def test_scenario_shares(splits):
    labels = splits["train"][2]
    share = labels["scenario"].value_counts(normalize=True)
    assert share["legit_stale"] == pytest.approx(0.10)
    assert share["legit_recurring_next"] == pytest.approx(10 / 60, abs=0.02)
    assert share["legit_same_amount_repeat"] == pytest.approx(8 / 60)


# --------------------------------------------------------------------------
# Schema: the shared reader accepts everything that is generated
# --------------------------------------------------------------------------

def test_every_generated_ledger_passes_the_shared_reader(splits, demo):
    ledger = pytest.importorskip("ledger", reason="ledger.py (owned by the lead) is not importable yet")
    tables = [(demo[0], "incoming"), (demo[1], "payments")]
    for split in OPEN_SPLITS:
        incoming, payments, _labels, _entry = splits[split]
        tables += [(rows, "incoming") for _ledger_id, rows in incoming.groupby("ledger_id", sort=False)]
        tables += [(rows, "payments") for _ledger_id, rows in payments.groupby("ledger_id", sort=False)]
    for table, role in tables:
        frame, _notes = ledger.read_ledger_csv(table.to_csv(index=False, lineterminator="\n").encode("utf-8"), role)
        assert frame["row_id"].tolist() == table["row_id"].tolist()
        assert frame["vendor"].tolist() == table["vendor"].tolist()
        assert frame["invoice_ref"].tolist() == table["invoice_ref"].tolist()      # leading zeros survive
        assert frame["billing_period"].tolist() == table["billing_period"].tolist()
        assert frame["amount_cents"].tolist() == [round(float(amount) * 100) for amount in table["amount"]]
        assert frame["invoice_date"].dt.strftime("%Y-%m-%d").tolist() == table["invoice_date"].tolist()


# --------------------------------------------------------------------------
# No formatting tell
# --------------------------------------------------------------------------

def surface_features(incoming: pd.DataFrame) -> pd.DataFrame:
    """What a single incoming row looks like, without comparing it with any payment."""
    ref, vendor = incoming["invoice_ref"], incoming["vendor"]
    last_word = vendor.str.casefold().str.replace(".", "", regex=False).str.split().str[-1]
    return pd.DataFrame({
        "ref_blank": (ref == "").astype(float),
        "ref_has_letters": ref.str.contains(r"[A-Za-z]").astype(float),
        "ref_has_prefix": ref.str.match(r"[A-Za-z#]").astype(float),
        "ref_length": ref.str.len().astype(float),
        "period_blank": (incoming["billing_period"] == "").astype(float),
        "vendor_all_caps": (vendor == vendor.str.upper()).astype(float),
        "vendor_legal_suffix": last_word.isin(LEGAL_SUFFIXES).astype(float),
        "vendor_punctuation": vendor.str.contains(r"[^\w\s]").astype(float),
        "vendor_length": vendor.str.len().astype(float),
        "amount_round_dollar": incoming["amount"].str.endswith(".00").astype(float),
    })


def invoice_age(incoming: pd.DataFrame) -> pd.Series:
    """Days between an invoice date and the latest incoming invoice date of its ledger."""
    days = pd.to_datetime(incoming["invoice_date"])
    return (days.groupby(incoming["ledger_id"]).transform("max") - days).dt.days.astype(float)


def probe_auc(train_features, train_labels, validation_features, validation_labels) -> float:
    """Fit a logistic regression on train rows and return its ROC AUC on validation rows."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    probe.fit(train_features, train_labels)
    return float(roc_auc_score(validation_labels, probe.decision_function(validation_features)))


def test_formatting_does_not_reveal_duplicates(splits):
    train, validation = splits["train"], splits["validation"]
    auc = probe_auc(surface_features(train[0]), train[2]["is_duplicate"],
                    surface_features(validation[0]), validation[2]["is_duplicate"])
    print(f"\nsurface-feature probe, ROC AUC on validation incoming rows: {auc:.3f}")
    assert 0.40 <= auc <= 0.60


def test_report_how_much_an_old_date_reveals(splits):
    """Reported, not judged: the same probe plus the age of the invoice inside its incoming batch."""
    train, validation = splits["train"], splits["validation"]
    train_features = surface_features(train[0]).assign(age_days=invoice_age(train[0]))
    validation_features = surface_features(validation[0]).assign(age_days=invoice_age(validation[0]))
    auc = probe_auc(train_features, train[2]["is_duplicate"], validation_features, validation[2]["is_duplicate"])
    age_only = float(roc_auc_score(validation[2]["is_duplicate"], validation_features["age_days"]))
    print(f"\nsurface features + invoice age, ROC AUC on validation: {auc:.3f} (age alone: {age_only:.3f})")
    assert 0.0 <= auc <= 1.0


# --------------------------------------------------------------------------
# The demo ledger
# --------------------------------------------------------------------------

def demo_row(frame: pd.DataFrame, row_id: str) -> dict[str, str]:
    return frame.set_index("row_id").loc[row_id].to_dict()


def test_demo_size_and_columns(demo):
    incoming, payments, cases = demo
    assert list(incoming.columns) == list(payments.columns) == synth.DEMO_COLUMNS
    assert len(incoming) == 32 and 130 <= len(payments) <= 150
    assert incoming["row_id"].tolist() == [f"INC-{number:04d}" for number in range(1, 33)]
    assert payments["row_id"].is_unique and payments["invoice_date"].is_monotonic_increasing
    run_date = synth.DEMO_RUN_DATE.isoformat()
    assert incoming["invoice_date"].max() <= run_date and payments["invoice_date"].max() < run_date
    assert [case["key"] for case in cases] == ["disguised_duplicate", "recurring_bill", "ambiguous"]
    assert [case["title"] for case in cases] == ["Disguised duplicate", "Legitimate recurring bill", "Ambiguous case"]
    assert all(set(case) == {"key", "title", "incoming_id", "expected_payment_id", "summary"} for case in cases)


def test_demo_disguised_duplicate(demo):
    incoming, payments, cases = demo
    case = cases[0]
    assert demo_row(incoming, case["incoming_id"]) == {
        "vendor": "OC Event Rentals LLC", "invoice_ref": "INV-1042", "amount": "468.20",
        "invoice_date": "2026-03-06", "billing_period": "2026-03"}
    assert demo_row(payments, case["expected_payment_id"]) == {
        "vendor": "O.C. Event Rentals", "invoice_ref": "1042", "amount": "468.20",
        "invoice_date": "2026-03-06", "billing_period": "2026-03"}
    assert (payments["amount"] == "468.20").sum() == 1            # no other payment shares this amount


def test_demo_recurring_bill(demo):
    incoming, payments, cases = demo
    case = cases[1]
    assert demo_row(incoming, case["incoming_id"]) == {
        "vendor": "Lantern Street Storage", "invoice_ref": "LS-20418", "amount": "135.00",
        "invoice_date": "2026-03-01", "billing_period": "2026-03"}
    assert demo_row(payments, case["expected_payment_id"]) == {
        "vendor": "Lantern Street Storage", "invoice_ref": "LS-20377", "amount": "135.00",
        "invoice_date": "2026-02-01", "billing_period": "2026-02"}
    storage = payments[payments["vendor"].str.contains("Lantern")]
    assert (storage["vendor"] == "Lantern Street Storage").all() and (storage["amount"] == "135.00").all()
    assert storage["billing_period"].tolist() == ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
    assert storage["invoice_ref"].is_unique and (storage["invoice_ref"] != "").all()
    assert (storage["invoice_date"].str[:7] == storage["billing_period"]).all()


def test_demo_ambiguous_case(demo):
    incoming, payments, cases = demo
    case = cases[2]
    assert demo_row(incoming, case["incoming_id"]) == {
        "vendor": "Bayside Print Co.", "invoice_ref": "", "amount": "125.00",
        "invoice_date": "2026-03-12", "billing_period": ""}
    assert demo_row(payments, case["expected_payment_id"]) == {
        "vendor": "Bayside Printing Co.", "invoice_ref": "BP-7731", "amount": "125.00",
        "invoice_date": "2026-02-24", "billing_period": ""}
    bayside = payments[payments["vendor"].str.contains("Bayside")]
    assert (bayside["amount"] == "125.00").sum() == 1             # no other Bayside payment at 125.00


def test_demo_has_no_field_for_field_resubmission(demo):
    incoming, payments, _cases = demo
    fields = ["vendor", "invoice_ref", "amount", "invoice_date", "billing_period"]
    assert incoming.merge(payments, on=fields).empty


def test_demo_rules_are_enforced_when_building():
    cases = synth.demo_cases()
    payments = [payment for case in cases for payment in case.payments]
    incoming = [case.incoming for case in cases]
    synth.check_demo_rules(payments, incoming)                    # the hand-written rows alone are fine
    rival = synth.hand_record("Compass Yard Hardware", "", 46820, date(2026, 1, 5), "")
    with pytest.raises(ValueError):
        synth.check_demo_rules(payments + [rival], incoming)
    with pytest.raises(ValueError):
        synth.check_demo_rules(payments, incoming + [payments[0]])
