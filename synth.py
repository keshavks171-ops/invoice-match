"""synth.py - synthetic ledgers for the Invoice Match benchmark and demo.

Every vendor, invoice and duplicate here is invented. A "ledger" is one
fictional small nonprofit's books at one payment run: about 60 incoming
invoices and 220-280 prior payments from the 180 days before the run date.

Design choices worth knowing:
  * Splits are made BEFORE any invoice exists. Train, validation and test use
    different vendor words and different date ranges, so a model is always
    tested on vendors and dates it has never seen.
  * Every record, legitimate or duplicate, paid or incoming, is written by the
    same render() function with the same error rates, so the way a row is
    formatted says nothing about whether it is a duplicate. A duplicate differs
    from its paid record only by what its type says: a later date, a late fee,
    a changed reference or a missing reference.
  * Randomness comes only from generators seeded with BASE_SEED plus the split
    name and the ledger number. Generating one split never changes another,
    and the held-out test split can be generated later, byte for byte.
  * The held-out test split is written only when it is named on the command
    line:  python synth.py --splits test

Usage:
  python synth.py                                  train + validation + demo
  python synth.py --splits train validation test   exactly the named splits
  python synth.py --demo                           only the demo ledger
"""

from __future__ import annotations

import argparse
import json
import math
import random
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

BASE_SEED = 20260
HISTORY_DAYS = 180           # prior payments cover this many days before the run date
LEDGERS_PER_FOLD = 6
DUPLICATES_PER_LEDGER = (3, 4, 3, 3, 4, 3)   # 20 in every fold of 6 ledgers

LEDGER_COLUMNS = ["ledger_id", "row_id", "vendor", "invoice_ref", "amount", "invoice_date", "billing_period"]
LABEL_COLUMNS = ["ledger_id", "fold", "incoming_row_id", "is_duplicate", "true_payment_ids", "scenario", "vendor_kind"]
DEMO_COLUMNS = LEDGER_COLUMNS[1:]


@dataclass(frozen=True)
class Split:
    """One part of the benchmark and the only dates its ledgers may contain."""

    ledger_prefix: str
    n_ledgers: int
    first_day: date
    last_day: date


SPLITS = {
    "train": Split("TR", 60, date(2023, 1, 1), date(2024, 8, 31)),
    "validation": Split("VA", 18, date(2024, 9, 1), date(2025, 6, 30)),
    "test": Split("TE", 6, date(2025, 7, 1), date(2026, 6, 30)),
}


# --------------------------------------------------------------------------
# Word lists (all fictional). Each split has its own words, so vendor names in
# different splits share only generic trade words and legal suffixes.
# --------------------------------------------------------------------------

# First word of an organisation's name. It is also the vendor GROUP key.
ORG_FIRST_WORDS = {
    "train": (
        "Cedar", "Willow", "Birch", "Alder", "Hazel", "Rowan", "Linden", "Sorrel", "Clover", "Thistle",
        "Bramble", "Fern", "Heather", "Laurel", "Poplar", "Hawthorn", "Elm", "Spruce", "Larch", "Cypress",
        "Magnolia", "Myrtle", "Briar", "Primrose", "Foxglove", "Lupine", "Yarrow", "Tansy", "Sumac", "Lavender",
        "Chicory", "Comfrey", "Larkspur", "Goldenrod", "Harebell", "Tarragon", "Woodruff", "Valerian", "Burdock",
        "Snowdrop", "Copper", "Granite", "Flint", "Russet", "Ochre", "Umber", "Pewter", "Zircon", "Jasper",
        "Saffron", "Crimson", "Agate", "Beryl", "Topaz", "Sienna", "Cerulean", "Heron", "Sparrow", "Finch",
        "Wren", "Kestrel", "Plover", "Curlew", "Egret", "Marten", "Badger", "Linnet", "Lapwing", "Siskin",
        "Thrush", "Grebe", "Skylark",
    ),
    "validation": (
        "Aster", "Bracken", "Dogwood", "Elder", "Fennel", "Gorse", "Hemlock", "Iris", "Mallow", "Nettle",
        "Quince", "Sedge", "Teasel", "Vetch", "Walnut", "Yew", "Tawny", "Teal", "Cinder", "Chalk",
        "Marble", "Brass", "Pipit", "Dunlin",
    ),
    "test": (
        "Acacia", "Catalpa", "Dahlia", "Ginkgo", "Hickory", "Jasmine", "Lilac", "Mulberry", "Nutmeg", "Olive",
        "Peony", "Rosewood", "Sycamore", "Tamarack", "Verbena", "Wisteria", "Yucca", "Bronze", "Shale", "Basalt",
        "Mica", "Scarlet", "Wagtail", "Whimbrel",
    ),
}

# Second word of an organisation's name.
ORG_SECOND_WORDS = {
    "train": (
        "Hollow", "Ridge", "Creek", "Meadow", "Grove", "Valley", "Harbor", "Bridge", "Hill", "Field",
        "Brook", "Glen", "Marsh", "Spur", "Cove", "Lake", "River", "Springs", "Falls", "Crossing",
        "Landing", "Mill", "Forge", "Gate", "Park", "Heights", "Bluff", "Canyon", "Prairie", "Orchard",
        "Garden", "Square", "Commons", "Junction", "Station", "Terrace",
    ),
    "validation": (
        "Vista", "Haven", "Dale", "Fork", "Bend", "Pass", "Mesa", "Shore", "Quarry", "Pond", "Knoll", "Trail",
    ),
    "test": ("Notch", "Moor", "Heath", "Vale", "Dell", "Fen", "Croft", "Reach", "Strand", "Peak", "Plains", "Pier"),
}

# Last name of an individual (freelancer, instructor, sole proprietor). Also the GROUP key.
LAST_NAMES = {
    "train": (
        "Raman", "Okafor", "Lindqvist", "Castellano", "Nakamura", "Abernathy", "Villanueva", "Kowalczyk", "Haddad",
        "Thibodeaux", "Petrakis", "Sandoval", "Whitlock", "Zielinski", "Fontaine", "Halvorsen", "Iwasaki",
        "Jorgensen", "Montoya", "Ostrander", "Pemberton", "Quintero", "Rasmussen", "Sokolov", "Tanaka", "Uribe",
        "Vandermeer", "Wojcik", "Yamada", "Peña", "Adeyemi", "Bhatt", "Chaudhry", "Delacroix", "Eriksen", "Müller",
    ),
    "validation": (
        "Gallardo", "Hosokawa", "Ibarra", "Jablonski", "Kimathi", "Lombardo", "Mahlangu", "Nwosu", "Olawale",
        "Pacheco", "Qureshi", "Núñez",
    ),
    "test": (
        "Szabo", "Trevino", "Ueda", "Valdez", "Winterbourne", "Xiong", "Yilmaz", "Zubiri", "Achterberg",
        "Bellweather", "Dragomir", "Côté",
    ),
}

FIRST_NAMES = {
    "train": (
        "Priya", "Mateo", "Ingrid", "Kwame", "Soren", "Leila", "Tomas", "Aiko", "Darius", "Nadia",
        "Emeka", "Freya", "Rohan", "Lucia", "Callum", "Yusuf", "Mirela", "Anders", "Chidi", "Paloma",
        "Hiroshi", "Zainab", "Oskar", "Thandi", "Rafael", "Elif", "Dmitri", "Sunita", "Joaquin", "Renée",
    ),
    "validation": (
        "Amara", "Bastien", "Carmen", "Devika", "Eamon", "Farida", "Goran", "Halima", "Ivo", "Jamila", "Kenji", "Zoë",
    ),
    "test": (
        "Marek", "Nilufar", "Ottoline", "Pavel", "Quentin", "Rosalind", "Stellan", "Tamsin", "Ulises", "Vesna",
        "Wendell", "José",
    ),
}

# What an organisation does, by how it bills. Each entry is (trade, the trade of
# a DIFFERENT business with a confusable name). Trade words are generic and
# shared by all splits.
TRADES = {
    "recurring": (
        ("Storage", "Self Storage"), ("Property Management", "Properties"), ("Internet", "Internet & Phone"),
        ("Software", "Software Systems"), ("Insurance", "Insurance Agency"), ("Bookkeeping", "Bookkeeping & Tax"),
        ("Waste Services", "Waste & Recycling"), ("Copier Leasing", "Copier Rental"),
        ("Security Monitoring", "Security Systems"), ("Water Delivery", "Water Service"),
        ("Janitorial", "Janitorial Supply"), ("Web Hosting", "Web Services"),
        ("Payroll Services", "Payroll Partners"), ("Telephone", "Telecom"),
    ),
    "standard_fee": (
        ("Cleaning Services", "Cleaners"), ("Lawn Care", "Lawn & Landscape"), ("Pest Control", "Pest Management"),
        ("Courier", "Courier Express"), ("Piano Tuning", "Piano Service"), ("Tutoring", "Tutors"),
        ("Music Lessons", "Music School"), ("Translation Services", "Translations"),
        ("Window Washing", "Window Cleaning"), ("Linen Service", "Linen Supply"),
        ("Shredding", "Document Shredding"), ("Shuttle Service", "Shuttle Lines"),
        ("Yoga Studio", "Yoga Collective"), ("Notary Services", "Notary & Mail"),
    ),
    "one_off": (
        ("Printing", "Print & Copy"), ("Catering", "Caterers"), ("Office Supply", "Office Supplies"),
        ("Hardware", "Hardware & Supply"), ("Plumbing", "Plumbing & Heating"), ("Electric", "Electrical"),
        ("Signs", "Sign & Banner"), ("Florist", "Flowers & Gifts"), ("Event Rentals", "Party Rentals"),
        ("Bakery", "Baking"), ("Art Supply", "Art Supplies"), ("Embroidery", "Embroidery & Apparel"),
        ("Photography", "Photo Studio"), ("Graphic Design", "Design Studio"), ("Auto Repair", "Auto Service"),
        ("Locksmith", "Lock & Key"), ("Roofing", "Roofing & Gutters"), ("Heating & Cooling", "Heating and Air"),
        ("Carpentry", "Woodworking"), ("Trophies & Awards", "Awards"), ("Books", "Bookshop"),
        ("Audio Visual", "Audio & Video"), ("Tent Rentals", "Tents & Staging"), ("Moving", "Movers"),
    ),
}

# What an individual does. It appears only in one alias ('Priya Raman Consulting').
PERSON_TRADES = {
    "recurring": ("Bookkeeping", "Office Cleaning", "IT Support"),
    "standard_fee": ("Tutoring", "Music Lessons", "Yoga Instruction", "Coaching", "Art Classes", "Dance Instruction"),
    "one_off": ("Consulting", "Design", "Photography", "Repairs", "Editing", "Translation", "Web Design"),
}

# Legal suffix -> other spellings of the same suffix.
LEGAL_FORMS = {
    "LLC": ("L.L.C.",),
    "Inc.": ("Inc", "Incorporated"),
    "Co.": ("Co", "Company"),
    "Ltd.": ("Ltd", "Limited"),
    "Corp.": ("Corp", "Corporation"),
}

ABBREVIATIONS = {
    "Services": "Svcs", "Service": "Svc", "Management": "Mgmt", "Insurance": "Ins.", "Printing": "Prtg",
    "Agency": "Agcy", "Hardware": "Hdwe", "Electric": "Elec.", "Electrical": "Elec.", "Photography": "Photo",
    "Systems": "Sys.",
}

# Initials that read like a well-known company are never used as a vendor alias.
BRAND_LIKE_INITIALS = ("AT", "BA", "BK", "BP", "BT", "CK", "CN", "DC", "DQ", "EA", "FB", "GE", "GM", "GQ", "GS", "HM",
                       "HP", "JP", "LG", "LV", "MG", "MS", "PG", "PS", "RC", "SK", "TD", "TM", "VW")

BILLING_KINDS = ("recurring", "standard_fee", "one_off")
BILLING_WEIGHTS = {"organization": (22, 18, 60), "individual": (12, 40, 48)}
TWIN_GROUP_SHARE = 0.30      # share of groups that hold two different vendors with confusable names

# How a vendor numbers its invoices, as (scheme, weight). 'none' never uses references.
REF_SCHEMES = {
    "organization": (("prefixed", 30), ("letter_prefix", 20), ("bare", 15), ("zero_padded", 12),
                     ("date_coded", 13), ("none", 10)),
    "individual": (("none", 45), ("bare", 25), ("prefixed", 15), ("date_coded", 10), ("zero_padded", 5)),
}
GENERIC_PREFIXES = ("INV-", "INV-", "INV", "Inv-", "BILL-", "Invoice ", "No. ")


# --------------------------------------------------------------------------
# Vendors
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Vendor:
    """One payee: a fictional business or person, and its invoicing habits."""

    vendor_id: str
    group: str                 # first word (organisations) or last name (individuals); never crosses splits
    kind: str                  # 'organization' or 'individual'
    name: str                  # the canonical way the name is written
    aliases: tuple[str, ...]   # other ways the same payee is written
    billing: str               # 'recurring', 'standard_fee' or 'one_off'
    ref_scheme: str            # see REF_SCHEMES
    ref_prefix: str            # 'INV-', 'CH-' or ''
    fills_period: bool         # writes the invoice month as billing period on non-monthly invoices
    sibling: str = ""          # vendor_id of a DIFFERENT payee with a confusable name


def make_rng(*labels: object) -> random.Random:
    """A random generator that depends only on BASE_SEED and the given labels."""
    return random.Random(":".join(str(label) for label in (BASE_SEED, *labels)))


def vendor_pool(split: str) -> list[Vendor]:
    """All vendors of one split. Their words are used by no other split."""
    rng = make_rng(split, "vendors")
    return drop_shared_aliases(organization_vendors(split, rng) + individual_vendors(split, rng))


def organization_vendors(split: str, rng: random.Random) -> list[Vendor]:
    """Three to six organisations per first word; some groups hold a confusable pair."""
    first_words = ORG_FIRST_WORDS[split]
    twin_groups = rng.sample(first_words, round(len(first_words) * TWIN_GROUP_SHARE))
    vendors: list[Vendor] = []
    for first in first_words:
        has_twin = first in twin_groups
        size = rng.randint(3, 6)
        group: list[Vendor] = []
        for second in rng.sample(ORG_SECOND_WORDS[split], size - 1 if has_twin else size):
            billing = rng.choices(BILLING_KINDS, weights=BILLING_WEIGHTS["organization"])[0]
            trade, twin_trade = rng.choice(TRADES[billing])
            group.append(make_organization(f"{split}:{first}:{len(group) + 1}", first, second, trade, billing, rng))
            if has_twin and len(group) == 1:
                group.append(make_organization(f"{split}:{first}:2", first, second, twin_trade, billing, rng))
        vendors += link_twins(group) if has_twin else group
    return vendors


def individual_vendors(split: str, rng: random.Random) -> list[Vendor]:
    """One person per last name; some last names are shared by two different people."""
    last_names = LAST_NAMES[split]
    twin_groups = rng.sample(last_names, round(len(last_names) * TWIN_GROUP_SHARE))
    billings = deal_billing_kinds(len(last_names), BILLING_WEIGHTS["individual"], rng)
    vendors: list[Vendor] = []
    for last, billing in zip(last_names, billings):
        has_twin = last in twin_groups
        group = [
            make_individual(f"{split}:{last}:{number}", first, last, billing, rng)
            for number, first in enumerate(rng.sample(FIRST_NAMES[split], 2 if has_twin else 1), start=1)
        ]
        vendors += link_twins(group) if has_twin else group
    return vendors


def deal_billing_kinds(count: int, weights: tuple[int, ...], rng: random.Random) -> list[str]:
    """`count` billing kinds in proportion to the weights (percent), in random order.

    Used for individuals: their pool is small, so the mix is dealt out instead of left to chance.
    """
    kinds = [kind for kind, weight in zip(BILLING_KINDS, weights) for _ in range(round(count * weight / 100))]
    kinds += ["one_off"] * (count - len(kinds))
    rng.shuffle(kinds)
    return kinds[:count]


def link_twins(group: list[Vendor]) -> list[Vendor]:
    """Mark the first two vendors of a group as each other's confusable sibling."""
    one, two = group[0], group[1]
    return [replace(one, sibling=two.vendor_id), replace(two, sibling=one.vendor_id), *group[2:]]


def make_organization(vendor_id: str, first: str, second: str, trade: str, billing: str,
                      rng: random.Random) -> Vendor:
    """Build an organisation such as 'Cedar Hollow Printing Co.' with its aliases and habits."""
    legal = rng.choice(tuple(LEGAL_FORMS))
    plain = f"{first} {second} {trade}"
    name = f"{plain} {legal}" if rng.random() < 0.55 else plain
    aliases = [
        plain,                                              # legal suffix dropped ...
        f"{plain} {legal}",                                 # ... or added
        f"{plain}, {legal}",                                # comma before the suffix
        *(f"{plain} {other}" for other in LEGAL_FORMS[legal]),
        swap_ampersand(name),
        abbreviate(name),
        name.upper(),
        name.replace(" ", "  ", 1),                         # doubled space
    ]
    initials = f"{first[0]}{second[0]}"
    if rng.random() < 0.30 and initials not in BRAND_LIKE_INITIALS:     # some are also known by their initials
        aliases += [f"{initials} {trade}", f"{first[0]}.{second[0]}. {trade}"]
    scheme, prefix = draw_ref_style("organization", initials, rng)
    return Vendor(vendor_id, first, "organization", name, distinct_aliases(aliases, name), billing,
                  scheme, prefix, fills_period=rng.random() < 1 / 3)


def make_individual(vendor_id: str, first: str, last: str, billing: str, rng: random.Random) -> Vendor:
    """Build a person such as 'Priya Raman' with aliases like 'P. Raman' and 'Raman, Priya'."""
    name = f"{first} {last}"
    aliases = [
        f"{first[0]}. {last}",
        f"{last}, {first}",
        f"{name} {rng.choice(PERSON_TRADES[billing])}",
        f"{name} LLC",
        strip_accents(name),
        name.upper(),
        name.replace(" ", "  ", 1),
    ]
    scheme, prefix = draw_ref_style("individual", "", rng)
    return Vendor(vendor_id, last, "individual", name, distinct_aliases(aliases, name), billing,
                  scheme, prefix, fills_period=rng.random() < 1 / 3)


def draw_ref_style(kind: str, initials: str, rng: random.Random) -> tuple[str, str]:
    """Pick how a vendor numbers its invoices: (scheme, prefix written before the number)."""
    schemes = [scheme for scheme, _weight in REF_SCHEMES[kind]]
    weights = [weight for _scheme, weight in REF_SCHEMES[kind]]
    scheme = rng.choices(schemes, weights=weights)[0]
    if scheme == "prefixed":
        return scheme, rng.choice(GENERIC_PREFIXES)
    if scheme == "letter_prefix":
        return scheme, f"{initials}-"
    return scheme, ""


def swap_ampersand(name: str) -> str:
    """'Print & Copy' -> 'Print and Copy', and the other way round."""
    if "&" in name:
        return name.replace("&", "and")
    return name.replace(" and ", " & ")


def abbreviate(name: str) -> str:
    """'Cedar Hollow Printing Co.' -> 'Cedar Hollow Prtg Co.'."""
    return " ".join(ABBREVIATIONS.get(word, word) for word in name.split(" "))


def strip_accents(text: str) -> str:
    """'Peña' -> 'Pena'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def distinct_aliases(aliases: list[str], name: str) -> tuple[str, ...]:
    """Drop repeats and anything equal to the canonical name, keeping the order."""
    kept: list[str] = []
    for alias in aliases:
        if alias != name and alias not in kept:
            kept.append(alias)
    return tuple(kept)


def name_key(text: str) -> str:
    """Letters and digits only, lower case: two spellings with the same key look like one payee."""
    return "".join(ch for ch in strip_accents(text).casefold() if ch.isalnum())


def drop_shared_aliases(vendors: list[Vendor]) -> list[Vendor]:
    """Remove every alias that could also mean a different vendor ('P. Raman' for two Ramans)."""
    owners: dict[str, list[str]] = {}
    for vendor in vendors:
        for text in (vendor.name, *vendor.aliases):
            owners.setdefault(name_key(text), []).append(vendor.vendor_id)
    return [
        replace(vendor, aliases=tuple(alias for alias in vendor.aliases
                                      if all(owner == vendor.vendor_id for owner in owners[name_key(alias)])))
        for vendor in vendors
    ]


def sibling_pairs(pool: list[Vendor]) -> list[tuple[Vendor, Vendor]]:
    """The confusable pairs of a pool, each listed once."""
    by_id = {vendor.vendor_id: vendor for vendor in pool}
    return [(vendor, by_id[vendor.sibling]) for vendor in pool if vendor.sibling and vendor.vendor_id < vendor.sibling]


# --------------------------------------------------------------------------
# Invoices and the one noise channel
# --------------------------------------------------------------------------

@dataclass
class Invoice:
    """An underlying invoice. Every row rendered from it belongs to the same invoice family."""

    vendor: Vendor
    billing: str            # 'recurring', 'standard_fee' or 'one_off'
    invoice_date: date
    amount_cents: int
    service_month: str      # 'YYYY-MM' the bill covers, '' when the vendor does not say
    scenario: str = ""      # set on incoming invoices only
    ref_body: str = ""      # reference without the vendor's prefix; '' when the vendor uses none
    family: int = -1        # number of the invoice family inside its ledger


@dataclass(frozen=True)
class Record:
    """One row as it is written to a ledger file. Amount and date come from the invoice."""

    invoice: Invoice
    vendor_text: str
    invoice_ref: str
    billing_period: str


def render(invoice: Invoice, rng: random.Random, keep_reference: bool = False) -> Record:
    """Write an invoice down the way a person or a billing system would.

    This is the only place where formatting noise is added, and it is used for
    every row of every ledger. keep_reference is for re-sent invoices that are
    defined by showing their reference again (see duplicate_record).
    """
    return Record(
        invoice=invoice,
        vendor_text=render_vendor(invoice.vendor, rng),
        invoice_ref=render_reference(invoice, rng, keep_reference),
        billing_period=invoice.service_month if rng.random() < 0.85 else "",
    )


def render_vendor(vendor: Vendor, rng: random.Random) -> str:
    """The canonical name about 60% of the time, otherwise one of the vendor's aliases."""
    if rng.random() < 0.60 or not vendor.aliases:
        return vendor.name
    return rng.choice(vendor.aliases)


def render_reference(invoice: Invoice, rng: random.Random, keep_reference: bool) -> str:
    """70% canonical, 20% another spelling, 3% a one-character typo, 7% left blank."""
    if not invoice.ref_body:
        return ""
    prefix, body = invoice.vendor.ref_prefix, invoice.ref_body
    roll = rng.random()
    if roll < 0.70:
        return prefix + body
    if roll < 0.90:
        return rng.choice(reference_variants(prefix, body))
    if roll < 0.93:
        return add_typo(prefix + body, rng)
    return prefix + body if keep_reference else ""


def reference_variants(prefix: str, body: str) -> list[str]:
    """Other ways the same reference gets typed: prefix dropped or added, separator, case, '#'."""
    letters = prefix.strip(" -.")
    if letters:
        variants = [body, f"{letters}{body}", f"{letters} {body}", f"{letters}-{body}".lower(),
                    f"#{body}", f"{letters}#{body}"]
    else:
        variants = [f"INV-{body}", f"INV {body}", f"#{body}", f"No. {body}"]
        if "-" in body:
            variants += [body.replace("-", ""), body.replace("-", "/")]
        if body.startswith("0"):
            variants.append(body.lstrip("0"))      # occasionally the leading zeros are lost
    return [variant for variant in variants if variant != prefix + body]


def add_typo(text: str, rng: random.Random) -> str:
    """Swap two neighbouring digits or change one digit."""
    digits = [position for position, ch in enumerate(text) if ch.isdigit()]
    swappable = [position for position in digits
                 if position + 1 in digits and text[position] != text[position + 1]]
    if swappable and rng.random() < 0.5:
        position = rng.choice(swappable)
        return text[:position] + text[position + 1] + text[position] + text[position + 2:]
    position = rng.choice(digits)
    wrong = rng.choice([digit for digit in "0123456789" if digit != text[position]])
    return text[:position] + wrong + text[position + 1:]


def reference_body(scheme: str, number: int, day: date) -> str:
    """The reference a vendor's system prints for invoice `number` dated `day`."""
    if scheme == "zero_padded":
        return f"{number:06d}"
    if scheme == "date_coded":
        return f"{day:%y%m}-{number % 1000:03d}"
    return str(number)


def stamp_references(invoices: list[Invoice], rng: random.Random) -> dict[str, int]:
    """Give every invoice its reference, numbering each vendor's invoices in date order.

    Returns the last number used per vendor, so a re-issued invoice can take a later one.
    """
    by_vendor: dict[str, list[Invoice]] = {}
    for invoice in invoices:
        by_vendor.setdefault(invoice.vendor.vendor_id, []).append(invoice)
    last_numbers: dict[str, int] = {}
    for vendor_id, vendor_invoices in by_vendor.items():
        scheme = vendor_invoices[0].vendor.ref_scheme
        if scheme == "none":
            continue
        number = rng.randint(100, 9000)
        biggest_step = rng.choice((1, 1, 2, 5, 20))   # 1 means a counter kept for this customer alone
        for invoice in sorted(vendor_invoices, key=lambda item: item.invoice_date):
            number += rng.randint(1, biggest_step)
            invoice.ref_body = reference_body(scheme, number, invoice.invoice_date)
        last_numbers[vendor_id] = number
    return last_numbers


# --------------------------------------------------------------------------
# Amounts and dates
# --------------------------------------------------------------------------

def round_dollars(dollars: float) -> int:
    """Round to a multiple of 5 dollars and return cents."""
    return max(5, round(dollars / 5) * 5) * 100


def one_off_amount(vendor: Vendor, rng: random.Random) -> int:
    """A purchase amount in cents: median about $180; people quote round fees more often."""
    dollars = math.exp(rng.gauss(5.2, 0.9))
    if rng.random() < (0.50 if vendor.kind == "individual" else 0.20):
        return round_dollars(dollars)
    return max(500, round(dollars * 100))


def flat_fee(vendor: Vendor, rng: random.Random) -> int:
    """The single price a standard-fee vendor charges every time (median about $120)."""
    dollars = math.exp(rng.gauss(4.8, 0.6))
    if rng.random() < (0.85 if vendor.kind == "individual" else 0.60):
        return round_dollars(dollars)
    return max(500, round(dollars * 100))


def monthly_amount(rng: random.Random) -> int:
    """The base amount of a monthly bill (median about $200); half are round."""
    dollars = math.exp(rng.gauss(5.3, 0.9))
    if rng.random() < 0.5:
        return round_dollars(dollars)
    return max(500, round(dollars * 100))


def shift_month(day: date, months: int) -> tuple[int, int]:
    """(year, month) of the month that lies `months` after the month of `day`."""
    index = day.year * 12 + day.month - 1 + months
    return index // 12, index % 12 + 1


def days_before(run_date: date, nearest: int, farthest: int, rng: random.Random) -> date:
    """A random day between `nearest` and `farthest` days before the run date."""
    return run_date - timedelta(days=rng.randint(nearest, farthest))


def purchase(vendor: Vendor, billing: str, day: date, amount_cents: int, scenario: str = "") -> Invoice:
    """An invoice that is not a monthly bill. Only vendors with that habit fill the billing period."""
    service_month = f"{day:%Y-%m}" if vendor.fills_period else ""
    return Invoice(vendor, billing, day, amount_cents, service_month, scenario)


# --------------------------------------------------------------------------
# Building one ledger
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerPlan:
    """How big a ledger is and how many invoices of each kind it holds."""

    n_incoming: int
    payments_range: tuple[int, int]   # how many prior payments, before a double payment is added
    n_recurring: int                  # vendors that bill every month
    n_standard: int                   # vendors that charge one flat fee at irregular dates
    n_one_off: int                    # vendors with ordinary purchases in the history
    n_newcomers: int                  # vendors with no history
    n_sibling_pairs: int              # confusable pairs that are both active
    n_repeats: int                    # incoming flat-fee bills from standard-fee vendors
    n_stale: int                      # incoming legitimate invoices with an old date


BENCHMARK_PLAN = LedgerPlan(n_incoming=60, payments_range=(225, 275), n_recurring=10, n_standard=8,
                            n_one_off=24, n_newcomers=5, n_sibling_pairs=5, n_repeats=8, n_stale=6)

# Share of individuals among the vendors drawn for each role.
PEOPLE_SHARE = {"recurring": 0.10, "standard_fee": 0.40, "one_off": 0.25, "newcomer": 0.25}


@dataclass
class Cast:
    """The vendors that take part in one ledger."""

    recurring: list[Vendor]
    standard: list[Vendor]
    one_off: list[Vendor]
    newcomers: list[Vendor]
    pairs: list[tuple[Vendor, Vendor]]


def draw_mixed(candidates: list[Vendor], count: int, people_share: float, rng: random.Random) -> list[Vendor]:
    """Draw up to `count` vendors, about `people_share` of them individuals."""
    people = [vendor for vendor in candidates if vendor.kind == "individual"]
    organizations = [vendor for vendor in candidates if vendor.kind == "organization"]
    n_people = min(len(people), round(max(count, 0) * people_share))
    n_organizations = min(len(organizations), max(count, 0) - n_people)
    return rng.sample(organizations, n_organizations) + rng.sample(people, n_people)


def choose_vendors(pool: list[Vendor], plan: LedgerPlan, rng: random.Random) -> Cast:
    """Pick the vendors of a ledger: some confusable pairs first, then vendors for every role."""
    all_pairs = sibling_pairs(pool)
    pairs = rng.sample(all_pairs, min(plan.n_sibling_pairs, len(all_pairs)))
    paired = [vendor for pair in pairs for vendor in pair]
    singles = [vendor for vendor in pool if not vendor.sibling]
    roles: dict[str, list[Vendor]] = {}
    for billing, count in (("recurring", plan.n_recurring), ("standard_fee", plan.n_standard),
                           ("one_off", plan.n_one_off)):
        already = [vendor for vendor in paired if vendor.billing == billing]
        candidates = [vendor for vendor in singles if vendor.billing == billing]
        roles[billing] = already + draw_mixed(candidates, count - len(already), PEOPLE_SHARE[billing], rng)
    taken = [vendor.vendor_id for group in roles.values() for vendor in group]
    spare = [vendor for vendor in singles if vendor.vendor_id not in taken]
    newcomers = draw_mixed(spare, plan.n_newcomers, PEOPLE_SHARE["newcomer"], rng)
    return Cast(roles["recurring"], roles["standard_fee"], roles["one_off"], newcomers, pairs)


def recurring_bills(vendor: Vendor, run_date: date, rng: random.Random) -> list[Invoice]:
    """One bill per month up to the run date, oldest first. The newest is the current cycle."""
    billing_day = rng.randint(3, 26)
    in_arrears = rng.random() < 0.3              # the bill covers the month before its date
    base = monthly_amount(rng)
    fixed = rng.random() < 0.6                   # otherwise the amount moves by up to 15%
    bills = []
    for months_ago in range(7, -1, -1):
        year, month = shift_month(run_date, -months_ago)
        day = date(year, month, billing_day) + timedelta(days=rng.randint(-2, 2))
        if run_date - timedelta(days=HISTORY_DAYS) <= day <= run_date:
            amount = base if fixed else round(base * rng.uniform(0.85, 1.15))
            service_year, service_month = shift_month(day, -1 if in_arrears else 0)
            bills.append(Invoice(vendor, "recurring", day, amount, f"{service_year:04d}-{service_month:02d}"))
    return bills


def standard_fee_bills(vendor: Vendor, run_date: date, rng: random.Random) -> list[Invoice]:
    """Three to eight paid bills at irregular dates, always for the same flat fee."""
    fee = flat_fee(vendor, rng)
    return [purchase(vendor, "standard_fee", days_before(run_date, 1, HISTORY_DAYS, rng), fee)
            for _ in range(rng.randint(3, 8))]


def one_off_purchases(vendors: list[Vendor], n_rows: int, run_date: date, rng: random.Random) -> list[Invoice]:
    """Spread `n_rows` paid purchases over the vendors: one each, more for the busy ones."""
    busy = [rng.choice((1, 1, 2, 3, 5, 8)) for _ in vendors]
    buyers = vendors + rng.choices(vendors, weights=busy, k=n_rows - len(vendors))
    return [purchase(vendor, "one_off", days_before(run_date, 1, HISTORY_DAYS, rng), one_off_amount(vendor, rng))
            for vendor in buyers]


def group_by_vendor(invoices: list[Invoice]) -> dict[str, list[Invoice]]:
    """vendor_id -> that vendor's invoices."""
    groups: dict[str, list[Invoice]] = {}
    for invoice in invoices:
        groups.setdefault(invoice.vendor.vendor_id, []).append(invoice)
    return groups


def arrival(vendor: Vendor, billing: str, amount_cents: int, scenario: str, paid: dict[str, list[Invoice]],
            run_date: date, rng: random.Random) -> Invoice:
    """A legitimate new invoice: dated in the last 30 days, or 30 to 150 days back when it arrives late.

    A vendor that does not number its invoices never bills the same amount twice on one day.
    Nobody could tell such an invoice from a duplicate, so the answer key would be arbitrary.
    """
    nearest, farthest = (30, 150) if scenario == "legit_stale" else (0, 30)
    day = days_before(run_date, nearest, farthest, rng)
    if vendor.ref_scheme == "none":
        taken = [old.invoice_date for old in paid.get(vendor.vendor_id, []) if old.amount_cents == amount_cents]
        while day in taken:
            day = days_before(run_date, nearest, farthest, rng)
    return purchase(vendor, billing, day, amount_cents, scenario)


def repeat_bills(cast: Cast, paid: dict[str, list[Invoice]], count: int, run_date: date,
                 rng: random.Random) -> list[Invoice]:
    """New flat-fee bills from standard-fee vendors: same vendor and amount as before, yet legitimate."""
    bills = []
    for _ in range(count):
        vendor = rng.choice(cast.standard)
        fee = paid[vendor.vendor_id][0].amount_cents
        bills.append(arrival(vendor, "standard_fee", fee, "legit_same_amount_repeat", paid, run_date, rng))
    return bills


def sibling_bills(cast: Cast, paid: dict[str, list[Invoice]], run_date: date, rng: random.Random) -> list[Invoice]:
    """One new invoice per confusable pair, sometimes for a round amount the OTHER vendor was paid.

    The amount is swapped only when it is a round-dollar amount anyway, so these invoices
    are round as often as any other purchase.
    """
    bills = []
    for pair in cast.pairs:
        sender, other = rng.sample(pair, 2)
        amount = one_off_amount(sender, rng)
        round_amounts = [old.amount_cents for old in paid[other.vendor_id] if old.amount_cents % 100 == 0]
        if amount % 100 == 0 and round_amounts and rng.random() < 0.8:
            amount = rng.choice(round_amounts)
        bills.append(arrival(sender, "one_off", amount, "legit_similar_vendor", paid, run_date, rng))
    return bills


def stale_bills(cast: Cast, paid: dict[str, list[Invoice]], count: int, run_date: date,
                rng: random.Random) -> list[Invoice]:
    """Legitimate invoices that arrive late, so an old date alone does not reveal a duplicate."""
    bills = []
    for _ in range(count):
        vendor = rng.choice(cast.standard if rng.random() < 0.3 else cast.one_off)
        if vendor.billing == "standard_fee":
            amount = paid[vendor.vendor_id][0].amount_cents
        else:
            amount = one_off_amount(vendor, rng)
        bills.append(arrival(vendor, vendor.billing, amount, "legit_stale", paid, run_date, rng))
    return bills


def draft_invoices(cast: Cast, plan: LedgerPlan, n_duplicates: int, run_date: date,
                   rng: random.Random) -> tuple[list[Invoice], list[Invoice]]:
    """Create the underlying invoices of a ledger: (already paid, newly arrived and legitimate)."""
    history: list[Invoice] = []
    arrivals: list[Invoice] = []
    for vendor in cast.recurring:
        bills = recurring_bills(vendor, run_date, rng)
        bills[-1].scenario = "legit_recurring_next"
        history += bills[:-1]
        arrivals.append(bills[-1])
    for vendor in cast.standard:
        history += standard_fee_bills(vendor, run_date, rng)
    n_purchases = rng.randint(*plan.payments_range) - len(history)
    history += one_off_purchases(cast.one_off, n_purchases, run_date, rng)

    paid = group_by_vendor(history)
    arrivals += repeat_bills(cast, paid, plan.n_repeats, run_date, rng)
    arrivals += sibling_bills(cast, paid, run_date, rng)
    arrivals += stale_bills(cast, paid, plan.n_stale, run_date, rng)
    for vendor in cast.newcomers:
        amount = one_off_amount(vendor, rng)
        arrivals.append(arrival(vendor, "one_off", amount, "legit_new_vendor", paid, run_date, rng))
    for _ in range(plan.n_incoming - n_duplicates - len(arrivals)):
        vendor = rng.choice(cast.one_off)
        amount = one_off_amount(vendor, rng)
        arrivals.append(arrival(vendor, "one_off", amount, "legit_one_off", paid, run_date, rng))
    return history, arrivals


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------

# Duplicate types per fold of 6 ledgers (20 duplicates): 25 / 25 / 20 / 10 / 10 / 10 percent.
DUPLICATE_MIX = (
    ("dup_exact_resubmission", 5),   # the paid record copied field for field
    ("dup_rekeyed", 5),              # the same invoice typed in again
    ("dup_reissued_date", 4),        # re-sent with a later invoice date
    ("dup_revised_ref", 2),          # re-sent with a later date and a changed reference
    ("dup_late_fee", 2),             # past-due notice: later date, small fee added
    ("dup_statement_no_ref", 2),     # statement copy without the reference
)
RESEND_GAP_DAYS = {"dup_reissued_date": (7, 75), "dup_revised_ref": (7, 75), "dup_late_fee": (14, 75)}
RECENT_DAYS = 45             # duplicates mostly re-bill a payment this recent ...
RECENT_SHARE = 0.70          # ... this often


def duplicate_plan(split: str, fold: int) -> list[list[str]]:
    """The duplicate types of the six ledgers of one fold, dealt 3, 4, 3, 3, 4, 3."""
    kinds = [kind for kind, count in DUPLICATE_MIX for _ in range(count)]
    make_rng(split, "fold", fold).shuffle(kinds)
    plan, start = [], 0
    for size in DUPLICATES_PER_LEDGER:
        plan.append(kinds[start:start + size])
        start += size
    return plan


def paid_twice_mode(index: int) -> str:
    """Which ledgers hold an invoice that was already paid twice.

    'rebilled'  first ledger of every fold: paid twice, and billed a third time in the incoming batch
    'quiet'     fourth ledger of every fold: paid twice, not billed again
    ''          all other ledgers
    """
    if index % LEDGERS_PER_FOLD == 0:
        return "rebilled"
    if index % LEDGERS_PER_FOLD == 3:
        return "quiet"
    return ""


def can_rebill(kind: str, paid: Record, run_date: date) -> bool:
    """Whether a paid record can be the source of a duplicate of this type."""
    age = (run_date - paid.invoice.invoice_date).days
    if kind in RESEND_GAP_DAYS and age < RESEND_GAP_DAYS[kind][0]:
        return False                              # no room for a later date before the run date
    if kind == "dup_revised_ref":
        return paid.invoice.ref_body != ""        # needs a reference to revise
    if kind == "dup_statement_no_ref":
        return paid.invoice_ref != ""             # the paid record shows a reference, the copy does not
    return True


def pick_paid_record(kind: str, payments: list[Record], rebilled: list[Record], legit: list[Record],
                     run_date: date, rng: random.Random) -> Record:
    """Choose the paid record a duplicate will bill again.

    A random legitimate arrival sets the kind of bill and of vendor, so duplicates
    come from the same mix of bills as legitimate invoices. Recent payments are preferred.
    """
    taken = [record.invoice.family for record in rebilled]
    fits = [paid for paid in payments if paid.invoice.family not in taken and can_rebill(kind, paid, run_date)]
    if not fits:
        raise ValueError(f"No prior payment can be billed again as {kind}.")
    template = rng.choice(legit).invoice
    alike = [paid for paid in fits
             if paid.invoice.billing == template.billing and paid.invoice.vendor.kind == template.vendor.kind]
    choices = alike or fits
    recent = [paid for paid in choices if (run_date - paid.invoice.invoice_date).days <= RECENT_DAYS]
    older = [paid for paid in choices if (run_date - paid.invoice.invoice_date).days > RECENT_DAYS]
    if recent and (not older or rng.random() < RECENT_SHARE):
        return rng.choice(recent)
    return rng.choice(older)


def later_day(day: date, run_date: date, kind: str, rng: random.Random) -> date:
    """A new invoice date after `day`, never after the run date."""
    shortest, longest = RESEND_GAP_DAYS[kind]
    return day + timedelta(days=rng.randint(shortest, min(longest, (run_date - day).days)))


def revised_reference(invoice: Invoice, last_numbers: dict[str, int], rng: random.Random) -> str:
    """A changed reference: a brand-new number, or the old one with a revision mark."""
    vendor = invoice.vendor
    if rng.random() < 0.5:
        last_numbers[vendor.vendor_id] += rng.randint(1, 5)
        return reference_body(vendor.ref_scheme, last_numbers[vendor.vendor_id], invoice.invoice_date)
    return invoice.ref_body + rng.choice(("-R", "R", "-A", "-2", " REV"))


def late_fee(amount_cents: int, rng: random.Random) -> int:
    """1.5% of the amount, or a flat 10 to 35 dollars."""
    if rng.random() < 0.5:
        return max(1, round(amount_cents * 0.015))
    return rng.randint(10, 35) * 100


def duplicate_record(kind: str, paid: Record, run_date: date, last_numbers: dict[str, int],
                     rng: random.Random) -> Record:
    """Bill an already paid invoice again. The new row keeps the invoice family of the paid one."""
    resent = replace(paid.invoice, scenario=kind)
    if kind == "dup_exact_resubmission":
        return replace(paid, invoice=resent)
    if kind == "dup_rekeyed":
        return render(resent, rng)
    if kind == "dup_statement_no_ref":
        if (run_date - resent.invoice_date).days >= 7 and rng.random() < 0.5:   # half carry a statement date
            resent.invoice_date = later_day(resent.invoice_date, run_date, "dup_reissued_date", rng)
        return replace(render(resent, rng), invoice_ref="")
    resent.invoice_date = later_day(resent.invoice_date, run_date, kind, rng)
    if kind == "dup_revised_ref":
        resent.ref_body = revised_reference(resent, last_numbers, rng)
    if kind == "dup_late_fee":
        resent.amount_cents += late_fee(resent.amount_cents, rng)
    # These three types are defined by the reference they show, so it is never left blank;
    # a re-sent invoice without its reference is the statement-copy type above.
    return render(resent, rng, keep_reference=True)


def build_ledger(pool: list[Vendor], plan: LedgerPlan, run_date: date, duplicate_kinds: list[str],
                 paid_twice: str, rng: random.Random) -> tuple[list[Record], list[Record]]:
    """Build one ledger. Returns (prior payments, incoming invoices), both still unordered."""
    cast = choose_vendors(pool, plan, rng)
    history, arrivals = draft_invoices(cast, plan, len(duplicate_kinds), run_date, rng)
    for family, invoice in enumerate(history + arrivals):
        invoice.family = family
    last_numbers = stamp_references(history + arrivals, rng)

    payments = [render(invoice, rng) for invoice in history]
    legit = [render(invoice, rng) for invoice in arrivals]
    rebilled: list[Record] = []
    duplicates: list[Record] = []
    for kind in duplicate_kinds:
        paid = pick_paid_record(kind, payments, rebilled, legit, run_date, rng)
        rebilled.append(paid)
        duplicates.append(duplicate_record(kind, paid, run_date, last_numbers, rng))

    if paid_twice == "rebilled":       # the invoice behind the first duplicate was already paid twice
        payments.append(render(rebilled[0].invoice, rng))
    if paid_twice == "quiet":          # an invoice was paid twice and nobody billed it again
        taken = [record.invoice.family for record in rebilled]
        untouched = [paid for paid in payments if paid.invoice.family not in taken]
        payments.append(render(rng.choice(untouched).invoice, rng))
    return payments, legit + duplicates


# --------------------------------------------------------------------------
# From records to tables
# --------------------------------------------------------------------------

def in_date_order(payments: list[Record], rng: random.Random) -> list[Record]:
    """Payments sorted by invoice date; rows with the same date are in random order."""
    shuffled = rng.sample(payments, len(payments))
    return sorted(shuffled, key=lambda record: record.invoice.invoice_date)


def record_row(row_id: str, record: Record) -> dict[str, str]:
    """The schema columns of one record, as text ready for a CSV file."""
    cents = record.invoice.amount_cents
    return {
        "row_id": row_id,
        "vendor": record.vendor_text,
        "invoice_ref": record.invoice_ref,
        "amount": f"{cents // 100}.{cents % 100:02d}",
        "invoice_date": record.invoice.invoice_date.isoformat(),
        "billing_period": record.billing_period,
    }


def label_row(ledger_id: str, fold: int, row_id: str, record: Record, payments: list[Record],
              payment_ids: list[str]) -> dict[str, object]:
    """The answer-key row of one incoming invoice.

    A duplicate's truth set is every prior payment of the same invoice family.
    A legitimate invoice has none, even when it resembles a prior payment.
    """
    is_duplicate = record.invoice.scenario.startswith("dup_")
    truth = [payment_id for payment_id, paid in zip(payment_ids, payments)
             if is_duplicate and paid.invoice.family == record.invoice.family]
    return {
        "ledger_id": ledger_id,
        "fold": fold,
        "incoming_row_id": row_id,
        "is_duplicate": int(is_duplicate),
        "true_payment_ids": "|".join(truth),
        "scenario": record.invoice.scenario,
        "vendor_kind": record.invoice.vendor.kind,
    }


def draw_run_date(split: Split, rng: random.Random) -> date:
    """A payment-run date that keeps the whole history (and its billing periods) inside the split."""
    earliest = split.first_day + timedelta(days=HISTORY_DAYS + 31)
    return earliest + timedelta(days=rng.randint(0, (split.last_day - earliest).days))


def generate_ledger(split: str, index: int, pool: list[Vendor] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate ledger number `index` (from 0) of a split: (incoming, payments, labels, manifest entry)."""
    spec = SPLITS[split]
    rng = make_rng(split, "ledger", index)
    ledger_id = f"{spec.ledger_prefix}{index + 1:03d}"
    fold = index // LEDGERS_PER_FOLD + 1
    run_date = draw_run_date(spec, rng)
    duplicate_kinds = duplicate_plan(split, fold)[index % LEDGERS_PER_FOLD]
    payments, incoming = build_ledger(pool or vendor_pool(split), BENCHMARK_PLAN, run_date, duplicate_kinds,
                                      paid_twice_mode(index), rng)
    payments = in_date_order(payments, rng)
    rng.shuffle(incoming)             # ids are assigned after shuffling, so they cannot encode the label

    payment_ids = [f"{ledger_id}-P-{number:04d}" for number in range(1, len(payments) + 1)]
    incoming_ids = [f"{ledger_id}-I-{number:04d}" for number in range(1, len(incoming) + 1)]
    payment_rows = [{"ledger_id": ledger_id, **record_row(row_id, record)}
                    for row_id, record in zip(payment_ids, payments)]
    incoming_rows = [{"ledger_id": ledger_id, **record_row(row_id, record)}
                     for row_id, record in zip(incoming_ids, incoming)]
    label_rows = [label_row(ledger_id, fold, row_id, record, payments, payment_ids)
                  for row_id, record in zip(incoming_ids, incoming)]
    entry = {
        "ledger_id": ledger_id,
        "fold": fold,
        "run_date": run_date.isoformat(),
        "incoming": len(incoming_rows),
        "payments": len(payment_rows),
        "duplicates": sum(row["is_duplicate"] for row in label_rows),
    }
    return (pd.DataFrame(incoming_rows, columns=LEDGER_COLUMNS), pd.DataFrame(payment_rows, columns=LEDGER_COLUMNS),
            pd.DataFrame(label_rows, columns=LABEL_COLUMNS), entry)


def generate_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate every ledger of a split: (incoming, payments, labels, manifest entry)."""
    spec = SPLITS[split]
    pool = vendor_pool(split)
    ledgers = [generate_ledger(split, index, pool) for index in range(spec.n_ledgers)]
    entry = {
        "seed": BASE_SEED,
        "date_range": [spec.first_day.isoformat(), spec.last_day.isoformat()],
        "ledgers": [ledger[3] for ledger in ledgers],
    }
    return (pd.concat([ledger[0] for ledger in ledgers], ignore_index=True),
            pd.concat([ledger[1] for ledger in ledgers], ignore_index=True),
            pd.concat([ledger[2] for ledger in ledgers], ignore_index=True), entry)


# --------------------------------------------------------------------------
# The demo ledger (curated, never used for metrics)
# --------------------------------------------------------------------------

DEMO_RUN_DATE = date(2026, 3, 20)
DEMO_PLAN = LedgerPlan(n_incoming=29, payments_range=(126, 134), n_recurring=4, n_standard=4,
                       n_one_off=13, n_newcomers=3, n_sibling_pairs=0, n_repeats=4, n_stale=3)
# Weaker evidence than the hand-written disguised duplicate, and never a field-for-field copy.
DEMO_DUPLICATES = ["dup_reissued_date", "dup_late_fee", "dup_statement_no_ref"]

# Hand-written, demo-only vendors: (first word, second word, trade, billing). Their words are in no split.
DEMO_ORGANIZATIONS = (
    ("Marigold", "Wharf", "Property Management", "recurring"),
    ("Tidewater", "Loft", "Insurance Agency", "recurring"),
    ("Kingfisher", "Lane", "Web Hosting", "recurring"),
    ("Sundial", "Plaza", "Waste Services", "recurring"),
    ("Driftwood", "Row", "Cleaning Services", "standard_fee"),
    ("Pelican", "Court", "Piano Tuning", "standard_fee"),
    ("Beacon", "Alley", "Catering", "one_off"),
    ("Meridian", "Pavilion", "Office Supply", "one_off"),
    ("Compass", "Yard", "Hardware", "one_off"),
    ("Anchor", "Mews", "Signs", "one_off"),
    ("Seabright", "Arcade", "Florist", "one_off"),
    ("Windmill", "Close", "Plumbing", "one_off"),
    ("Pinecone", "Steps", "Bakery", "one_off"),
    ("Foghorn", "Dock", "Audio Visual", "one_off"),
    ("Larkfield", "Green", "Tent Rentals", "one_off"),
    ("Saltmarsh", "Circle", "Electric", "one_off"),
    ("Starboard", "Parade", "Embroidery", "one_off"),
    ("Thimble", "Walk", "Art Supply", "one_off"),
)
DEMO_INDIVIDUALS = (     # (first name, last name, billing)
    ("Imogen", "Hartwell", "standard_fee"),
    ("Kofi", "Mensah", "standard_fee"),
    ("Celeste", "Marchetti", "one_off"),
    ("Tobias", "Lindgren", "one_off"),
    ("Anjali", "Deshmukh", "one_off"),
    ("Marisol", "Quiroga", "one_off"),
)


@dataclass(frozen=True)
class DemoCase:
    """A hand-written example: one incoming invoice and the payments it is shown against."""

    key: str
    title: str
    summary: str
    incoming: Record
    payments: tuple[Record, ...]      # the first one is the payment the invoice should be compared with


def hand_record(vendor: str, invoice_ref: str, amount_cents: int, day: date, billing_period: str) -> Record:
    """A demo row written by hand, field for field (it does not pass through the noise channel)."""
    payee = Vendor(f"demo:{vendor}", "demo", "organization", vendor, (), "one_off", "none", "", False)
    return Record(Invoice(payee, "one_off", day, amount_cents, billing_period), vendor, invoice_ref, billing_period)


def demo_cases() -> list[DemoCase]:
    """The three cases the app walks through."""
    disguised = DemoCase(
        "disguised_duplicate", "Disguised duplicate",
        "The same rental invoice arrives a second time. The vendor name and the reference are written "
        "differently, but the amount, the invoice date and the billing period are identical to a payment "
        "that was already made.",
        hand_record("OC Event Rentals LLC", "INV-1042", 46820, date(2026, 3, 6), "2026-03"),
        (hand_record("O.C. Event Rentals", "1042", 46820, date(2026, 3, 6), "2026-03"),
         hand_record("O.C. Event Rentals", "0987", 31200, date(2025, 11, 14), "2025-11")),
    )
    recurring = DemoCase(
        "recurring_bill", "Legitimate recurring bill",
        "The March storage bill. It has the same vendor and the same amount as the bills paid in earlier "
        "months, but a new reference and a new billing period. That is what a normal monthly bill looks like.",
        hand_record("Lantern Street Storage", "LS-20418", 13500, date(2026, 3, 1), "2026-03"),
        (hand_record("Lantern Street Storage", "LS-20377", 13500, date(2026, 2, 1), "2026-02"),
         hand_record("Lantern Street Storage", "LS-20339", 13500, date(2026, 1, 1), "2026-01"),
         hand_record("Lantern Street Storage", "LS-20298", 13500, date(2025, 12, 1), "2025-12"),
         hand_record("Lantern Street Storage", "LS-20261", 13500, date(2025, 11, 1), "2025-11"),
         hand_record("Lantern Street Storage", "LS-20224", 13500, date(2025, 10, 1), "2025-10")),
    )
    ambiguous = DemoCase(
        "ambiguous", "Ambiguous case",
        "A similarly named vendor bills the same amount 16 days after a paid invoice, with no reference and "
        "no billing period to compare. The records cannot settle whether this is a second bill for the same "
        "job, so a person has to check.",
        hand_record("Bayside Print & Copy", "", 12500, date(2026, 3, 12), ""),
        (hand_record("Bayside Printing Co.", "BP-7731", 12500, date(2026, 2, 24), ""),
         hand_record("Bayside Printing Co.", "BP-7716", 24000, date(2026, 1, 22), ""),
         hand_record("Bayside Printing Co.", "BP-7698", 8650, date(2025, 12, 9), "")),
    )
    return [disguised, recurring, ambiguous]


def demo_vendors(rng: random.Random) -> list[Vendor]:
    """The demo-only vendors, built with the same functions as the benchmark vendors."""
    vendors = [make_organization(f"demo:{first}", first, second, trade, billing, rng)
               for first, second, trade, billing in DEMO_ORGANIZATIONS]
    vendors += [make_individual(f"demo:{last}", first, last, billing, rng)
                for first, last, billing in DEMO_INDIVIDUALS]
    return drop_shared_aliases(vendors)


def check_demo_rules(payments: list[Record], incoming: list[Record]) -> None:
    """Stop if a generated row competes with a hand-written case."""
    printed = [(paid.vendor_text, paid.invoice_ref, paid.invoice.amount_cents, paid.invoice.invoice_date,
                paid.billing_period) for paid in payments]
    copies = [record for record in incoming
              if (record.vendor_text, record.invoice_ref, record.invoice.amount_cents, record.invoice.invoice_date,
                  record.billing_period) in printed]
    if sum(paid.invoice.amount_cents == 46820 for paid in payments) != 1:
        raise ValueError("Demo: exactly one prior payment may have the amount 468.20.")
    if sum(paid.invoice.amount_cents == 12500 and "Bayside" in paid.vendor_text for paid in payments) != 1:
        raise ValueError("Demo: exactly one Bayside payment may have the amount 125.00.")
    if copies:
        raise ValueError("Demo: an incoming invoice is a field-for-field copy of a prior payment.")


def build_demo() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """Build the demo ledger: (incoming, payments, cases). No answer key is produced."""
    rng = make_rng("demo")
    payments, incoming = build_ledger(demo_vendors(rng), DEMO_PLAN, DEMO_RUN_DATE, DEMO_DUPLICATES, "", rng)
    cases = demo_cases()
    for case in cases:
        payments += case.payments
        incoming.append(case.incoming)
    check_demo_rules(payments, incoming)
    payments = in_date_order(payments, rng)
    rng.shuffle(incoming)

    payment_ids = [f"PAY-{number:04d}" for number in range(1, len(payments) + 1)]
    incoming_ids = [f"INC-{number:04d}" for number in range(1, len(incoming) + 1)]
    case_rows = [
        {
            "key": case.key,
            "title": case.title,
            "incoming_id": next(row_id for row_id, record in zip(incoming_ids, incoming) if record is case.incoming),
            "expected_payment_id": next(row_id for row_id, record in zip(payment_ids, payments)
                                        if record is case.payments[0]),
            "summary": case.summary,
        }
        for case in cases
    ]
    incoming_rows = [record_row(row_id, record) for row_id, record in zip(incoming_ids, incoming)]
    payment_rows = [record_row(row_id, record) for row_id, record in zip(payment_ids, payments)]
    return (pd.DataFrame(incoming_rows, columns=DEMO_COLUMNS), pd.DataFrame(payment_rows, columns=DEMO_COLUMNS),
            case_rows)


# --------------------------------------------------------------------------
# Writing files
# --------------------------------------------------------------------------

def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a table as UTF-8 CSV with Unix line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def write_split(split: str, data_dir: Path = DATA_DIR) -> dict[str, object]:
    """Generate one split and write its three CSV files and its manifest entry. Returns the entry."""
    folder = data_dir / "synthetic"
    incoming, payments, labels, entry = generate_split(split)
    write_csv(incoming, folder / f"{split}_incoming.csv")
    write_csv(payments, folder / f"{split}_payments.csv")
    write_csv(labels, folder / f"{split}_labels.csv")

    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[split] = entry
    ordered = {name: manifest[name] for name in SPLITS if name in manifest}
    manifest_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return entry


def write_demo(data_dir: Path = DATA_DIR) -> tuple[int, int]:
    """Write the demo ledger and its cases. Returns (incoming rows, payment rows)."""
    folder = data_dir / "demo"
    incoming, payments, cases = build_demo()
    write_csv(incoming, folder / "incoming.csv")
    write_csv(payments, folder / "payments.csv")
    (folder / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    return len(incoming), len(payments)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line. With no arguments: train, validation and the demo, never test."""
    parser = argparse.ArgumentParser(description="Generate the synthetic Invoice Match ledgers.")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=None,
                        help="splits to write; the held-out test split is written only when named here")
    parser.add_argument("--demo", action="store_true", help="write the demo ledger")
    args = parser.parse_args(argv)
    if args.splits is None and not args.demo:
        args.splits, args.demo = ["train", "validation"], True
    args.splits = args.splits or []
    return args


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point. Prints counts only, never ledger contents."""
    args = parse_args(argv)
    for split in args.splits:
        entry = write_split(split)
        ledgers = entry["ledgers"]
        print(f"{split}: {len(ledgers)} ledgers, "
              f"{sum(ledger['incoming'] for ledger in ledgers)} incoming invoices, "
              f"{sum(ledger['payments'] for ledger in ledgers)} prior payments, "
              f"{sum(ledger['duplicates'] for ledger in ledgers)} known duplicates")
    if args.demo:
        n_incoming, n_payments = write_demo()
        print(f"demo: {n_incoming} incoming invoices, {n_payments} prior payments")


if __name__ == "__main__":
    main()
