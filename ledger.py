"""ledger.py - the shared CSV schema for Invoice Match.

Both input files (incoming invoices and prior payments) use the same columns.
This module turns a CSV into a clean pandas DataFrame (a "ledger frame"),
explains exactly what is wrong when a file cannot be used, and writes CSV
exports that are safe to open in a spreadsheet.

Design choices worth knowing:
  * Money is stored as integer cents (parsed with decimal.Decimal), never as a
    float, so $468.20 is exactly 46820.
  * Invoice references stay strings, so "000418" keeps its leading zeros.
  * Dates must be ISO (YYYY-MM-DD). 03/06/2026 means different days in
    different countries, so we refuse to guess.
  * Nothing here prints, logs or saves ledger contents.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, Sequence, Union

import pandas as pd

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SCHEMA_COLUMNS = ("row_id", "vendor", "invoice_ref", "amount", "invoice_date", "billing_period")
REQUIRED_COLUMNS = ("vendor", "amount", "invoice_date")
OPTIONAL_COLUMNS = ("row_id", "invoice_ref", "billing_period")

# Columns of the DataFrame returned by read_ledger_csv.
FRAME_COLUMNS = ("row_id", "vendor", "invoice_ref", "amount_cents", "invoice_date", "billing_period")

# One line per column, shown in the app's upload help and in the README.
SCHEMA_DOC = (
    ("vendor", "required", "Who sent the bill, for example Lantern Street Storage."),
    ("amount", "required", "Positive amount with at most two decimals, for example 468.20."),
    ("invoice_date", "required", "The date printed on the invoice, as YYYY-MM-DD."),
    ("invoice_ref", "optional", "The invoice number as printed. May be blank. Leading zeros are kept."),
    ("billing_period", "optional", "The month the bill covers, as YYYY-MM. May be blank."),
    ("row_id", "optional", "Your own unique ID for the row. Assigned from file order when absent."),
)

MAX_UPLOAD_BYTES = 5_000_000
MAX_ROWS = {"incoming": 2_000, "payments": 20_000}
MAX_TEXT_LENGTH = 200
MAX_AMOUNT_CENTS = 99_999_999_999  # $999,999,999.99
MAX_PROBLEMS_SHOWN = 8

ROLE_NAMES = {"incoming": "incoming invoices", "payments": "prior payments"}
ID_PREFIXES = {"incoming": "INC", "payments": "PAY"}

EXAMPLE_HEADER = "vendor,invoice_ref,amount,invoice_date,billing_period"
EXAMPLE_ROW = "Lantern Street Storage,LS-20418,135.00,2026-03-01,2026-03"

# 468.2   468.20   $1,234.50   1234      (not: 1e3, -5, 12.345, 12,50)
_AMOUNT_PATTERN = re.compile(r"^\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PERIOD_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# Characters that make a spreadsheet treat a cell as a formula.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")

CsvSource = Union[str, Path, bytes, bytearray, BinaryIO]


class LedgerError(ValueError):
    """A CSV could not be used. Carries messages written for the treasurer.

    title     one-line summary of what is wrong
    problems  specific lines that say where the problem is and how to fix it
    """

    def __init__(self, title: str, problems: Sequence[str] = ()):
        super().__init__(title)
        self.title = title
        self.problems = list(problems)


# --------------------------------------------------------------------------
# Reading and validating a CSV
# --------------------------------------------------------------------------

def read_ledger_csv(source: CsvSource, role: str) -> tuple[pd.DataFrame, list[str]]:
    """Read one ledger CSV and return (frame, notes).

    source  a file path, raw bytes, or an open binary file (an upload)
    role    "incoming" or "payments"; used for messages, limits and default IDs
    notes   short information lines, such as which extra columns were ignored

    Raises LedgerError with actionable messages when the file cannot be used.
    A prior-payments file with a header and no rows is allowed (an organisation
    may have no history yet); an incoming file with no rows is an error.
    """
    if role not in ROLE_NAMES:
        raise ValueError(f"role must be 'incoming' or 'payments', not {role!r}")
    name = ROLE_NAMES[role]

    text = _decode(_load_bytes(source, name), name)
    records = _parse_records(text, name)
    header, positions, ignored = _read_header(records, name)

    notes: list[str] = []
    if ignored:
        label = "column" if len(ignored) == 1 else "columns"
        notes.append(f"Ignored extra {label} in the {name} file: {', '.join(ignored)}.")

    rows, problems = _read_rows(records, header, positions, role)
    _check_row_ids(rows, problems, has_row_id="row_id" in positions, role=role)

    if problems:
        raise LedgerError(
            f"The {name} file has {len(problems)} "
            f"{'problem' if len(problems) == 1 else 'problems'} to fix.",
            _limit(problems),
        )
    if not rows:
        if role == "incoming":
            raise LedgerError(
                f"The {name} file has column names but no invoices.",
                ["Add at least one invoice row below the header and upload the file again."],
            )
        notes.append(f"The {name} file has no rows, so there is nothing to compare against.")
    if len(rows) > MAX_ROWS[role]:
        raise LedgerError(
            f"The {name} file has {len(rows):,} rows; this prototype handles up to {MAX_ROWS[role]:,}.",
            ["Split the file (for example by month or by quarter) and upload a smaller part."],
        )
    return _build_frame(rows), notes


def _load_bytes(source: CsvSource, name: str) -> bytes:
    """Get the raw bytes from a path, a bytes object or an open file."""
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise LedgerError(
            f"The {name} file is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB.",
            ["Split the file (for example by month or by quarter) and upload a smaller part."],
        )
    if not raw.strip():
        raise LedgerError(
            f"The {name} file is empty.",
            [f"The first row must be the column names: {EXAMPLE_HEADER}",
             f"Then one row per invoice, for example: {EXAMPLE_ROW}"],
        )
    return raw


def _decode(raw: bytes, name: str) -> str:
    """Decode UTF-8 (with or without a byte-order mark) and reject binary files."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise LedgerError(
            f"The {name} file is not UTF-8 text.",
            ["In your spreadsheet choose Save As > 'CSV UTF-8 (Comma delimited)' and upload that file."],
        ) from None
    if "\x00" in text:
        raise LedgerError(
            f"The {name} file does not look like a CSV.",
            ["Upload a .csv file, not a spreadsheet workbook (.xlsx) or a PDF. "
             "In your spreadsheet choose Save As > 'CSV UTF-8 (Comma delimited)'."],
        )
    return text


def _parse_records(text: str, name: str) -> list[list[str]]:
    """Split the text into records with the standard csv module."""
    try:
        return list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error:
        raise LedgerError(
            f"The {name} file could not be read as a CSV.",
            ["Check for an unclosed double quote. A value that contains a comma must be "
             'wrapped in double quotes, for example "Harbor Print, LLC".'],
        ) from None


def _read_header(records: list[list[str]], name: str) -> tuple[list[str], dict[str, int], list[str]]:
    """Check the header row. Returns (header, column positions, ignored column names)."""
    header = [_normalize_header(cell) for cell in records[0]] if records else []
    known = [column for column in header if column in SCHEMA_COLUMNS]

    if not known:
        raise LedgerError(
            f"The {name} file has no column names in its first row.",
            [f"The first row must be the column names: {EXAMPLE_HEADER}",
             "Add that row above your data and upload the file again."],
        )
    repeated = sorted({column for column in known if known.count(column) > 1})
    if repeated:
        raise LedgerError(
            f"The {name} file repeats a column name.",
            [f"The column '{column}' appears more than once. Keep one and delete or rename the other."
             for column in repeated],
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in known]
    if missing:
        found = ", ".join(column for column in header if column) or "(none)"
        raise LedgerError(
            f"The {name} file is missing required "
            f"{'column' if len(missing) == 1 else 'columns'}: {', '.join(missing)}.",
            [f"Columns found: {found}.",
             f"Rename or add columns so the first row contains: {', '.join(REQUIRED_COLUMNS)}.",
             "Optional columns: invoice_ref, billing_period, row_id."],
        )
    positions = {column: header.index(column) for column in known}
    ignored = [column for column in header if column and column not in SCHEMA_COLUMNS]
    return header, positions, ignored


def _normalize_header(cell: str) -> str:
    """'Invoice Date ' -> 'invoice_date', so small header differences are forgiven."""
    return re.sub(r"[\s\-]+", "_", cell.strip().lower())


def _read_rows(records: list[list[str]], header: list[str], positions: dict[str, int],
               role: str) -> tuple[list[dict], list[str]]:
    """Validate every data row. Returns (clean rows, problem messages).

    Row numbers match what a spreadsheet shows: the header is row 1.
    """
    rows: list[dict] = []
    problems: list[str] = []
    for index, record in enumerate(records[1:], start=2):
        if not any(cell.strip() for cell in record):
            continue  # a completely blank line
        if len(record) > len(header):
            problems.append(
                f"Row {index} has {len(record)} values but there are only {len(header)} columns. "
                "If a value contains a comma, wrap it in double quotes."
            )
            continue

        def cell(column: str) -> str:
            position = positions.get(column)
            if position is None or position >= len(record):
                return ""
            return record[position].strip()

        row_problems: list[str] = []
        vendor = cell("vendor")
        if not vendor:
            row_problems.append(f"Row {index}: vendor is blank. Every row needs a vendor name.")
        amount_cents = _parse_amount(cell("amount"), index, row_problems)
        invoice_date = _parse_date(cell("invoice_date"), index, row_problems)
        billing_period = cell("billing_period")
        if billing_period and not _PERIOD_PATTERN.match(billing_period):
            row_problems.append(
                f"Row {index}: billing_period {_quote(billing_period)} is not a month. "
                "Use YYYY-MM, for example 2026-03, or leave it blank."
            )
        for column in ("vendor", "invoice_ref", "row_id"):
            if len(cell(column)) > MAX_TEXT_LENGTH:
                row_problems.append(
                    f"Row {index}: {column} is longer than {MAX_TEXT_LENGTH} characters. Shorten it."
                )
        if row_problems:
            problems.extend(row_problems)
            continue
        rows.append({
            "row_number": index,
            "row_id": cell("row_id"),
            "vendor": vendor,
            "invoice_ref": cell("invoice_ref"),
            "amount_cents": amount_cents,
            "invoice_date": invoice_date,
            "billing_period": billing_period,
        })
    return rows, problems


def _parse_amount(text: str, row_number: int, problems: list[str]) -> int | None:
    """Turn an amount such as '$1,234.50' into integer cents (123450)."""
    if not text:
        problems.append(f"Row {row_number}: amount is blank. Enter the invoice total, for example 468.20.")
        return None
    match = _AMOUNT_PATTERN.match(text)
    if not match:
        problems.append(
            f"Row {row_number}: amount {_quote(text)} is not a valid amount. Use digits with a "
            "period and at most two decimals, for example 468.20 or 1,234.50."
        )
        return None
    dollars = match.group(1).replace(",", "")
    cents = int(Decimal(dollars + (match.group(2) or "")) * 100)
    if cents <= 0:
        problems.append(f"Row {row_number}: amount {_quote(text)} must be greater than zero.")
        return None
    if cents > MAX_AMOUNT_CENTS:
        problems.append(f"Row {row_number}: amount {_quote(text)} is too large for this prototype.")
        return None
    return cents


def _parse_date(text: str, row_number: int, problems: list[str]) -> date | None:
    """Accept only real ISO dates (YYYY-MM-DD)."""
    hint = "Use YYYY-MM-DD, for example 2026-03-06."
    if not text:
        problems.append(f"Row {row_number}: invoice_date is blank. {hint}")
        return None
    if _DATE_PATTERN.match(text):
        try:
            return date.fromisoformat(text)
        except ValueError:
            pass  # right shape, impossible date such as 2026-02-30
    problems.append(f"Row {row_number}: invoice_date {_quote(text)} is not a valid date. {hint}")
    return None


def _check_row_ids(rows: list[dict], problems: list[str], has_row_id: bool, role: str) -> None:
    """Assign row IDs from file order, or check the supplied ones are unique."""
    if not has_row_id:
        for position, row in enumerate(rows, start=1):
            row["row_id"] = f"{ID_PREFIXES[role]}-{position:04d}"
        return
    first_seen: dict[str, int] = {}
    for row in rows:
        row_id, row_number = row["row_id"], row["row_number"]
        if not row_id:
            problems.append(
                f"Row {row_number}: row_id is blank. Fill it in, or delete the row_id column "
                "to have IDs assigned automatically."
            )
        elif row_id in first_seen:
            problems.append(
                f"Row {row_number}: row_id {_quote(row_id)} is already used on row "
                f"{first_seen[row_id]}. Each row needs its own ID."
            )
        else:
            first_seen[row_id] = row_number


def _build_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble the ledger frame with fixed column types."""
    frame = pd.DataFrame({
        "row_id": pd.Series([row["row_id"] for row in rows], dtype="str"),
        "vendor": pd.Series([row["vendor"] for row in rows], dtype="str"),
        "invoice_ref": pd.Series([row["invoice_ref"] for row in rows], dtype="str"),
        "amount_cents": pd.Series([row["amount_cents"] for row in rows], dtype="int64"),
        "invoice_date": pd.Series(
            pd.to_datetime([row["invoice_date"].isoformat() for row in rows], format="%Y-%m-%d")
        ).astype("datetime64[ns]"),
        "billing_period": pd.Series([row["billing_period"] for row in rows], dtype="str"),
    })
    return frame[list(FRAME_COLUMNS)]


def _quote(text: str) -> str:
    """Quote a cell for an error message, shortened so messages stay readable."""
    shown = text if len(text) <= 40 else text[:37] + "..."
    return f"'{shown}'"


def _limit(problems: list[str]) -> list[str]:
    """Show the first few problems and count the rest."""
    if len(problems) <= MAX_PROBLEMS_SHOWN:
        return problems
    hidden = len(problems) - MAX_PROBLEMS_SHOWN
    return problems[:MAX_PROBLEMS_SHOWN] + [f"...and {hidden} more. Fix these and upload again to see the rest."]


# --------------------------------------------------------------------------
# Formatting for display
# --------------------------------------------------------------------------

def format_cents(cents: int) -> str:
    """46820 -> '$468.20'."""
    cents = int(cents)
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def format_date(value) -> str:
    """A date or timestamp -> 'Mar 6, 2026'."""
    stamp = pd.Timestamp(value)
    return f"{stamp.strftime('%b')} {stamp.day}, {stamp.year}"


def format_period(period: str) -> str:
    """'2026-03' -> 'Mar 2026'. A blank period stays blank."""
    if not period:
        return ""
    stamp = pd.Timestamp(f"{period}-01")
    return f"{stamp.strftime('%b')} {stamp.year}"


# --------------------------------------------------------------------------
# Safe CSV export
# --------------------------------------------------------------------------

def escape_formula(text: str) -> str:
    """Stop a spreadsheet from running user-supplied text as a formula.

    A cell that starts with = + - @ (or a tab or line break) can be executed by
    Excel or Google Sheets when the CSV is opened. Prefixing an apostrophe makes
    the spreadsheet treat it as plain text.
    """
    text = "" if text is None else str(text)
    if text.lstrip(" ").startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def to_safe_csv(frame: pd.DataFrame, text_columns: Sequence[str]) -> bytes:
    """Write a DataFrame as CSV bytes, escaping the user-supplied text columns.

    The output is UTF-8 with a byte-order mark so Excel shows accents correctly.
    """
    safe = frame.copy()
    for column in text_columns:
        safe[column] = [escape_formula(value) for value in safe[column]]
    return safe.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")
