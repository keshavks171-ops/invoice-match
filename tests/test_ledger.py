"""Tests for ledger.py: the CSV schema, validation messages, formatting and safe export."""

import csv
import io

import pandas as pd
import pytest

import ledger
from ledger import LedgerError, read_ledger_csv

HEADER = "row_id,vendor,invoice_ref,amount,invoice_date,billing_period\n"


def read(text: str, role: str = "incoming"):
    return read_ledger_csv(text.encode("utf-8"), role)


def problems_of(text: str, role: str = "incoming") -> str:
    with pytest.raises(LedgerError) as caught:
        read(text, role)
    return caught.value.title + " | " + " | ".join(caught.value.problems)


# ----------------------------------------------------------------- happy path

def test_valid_file_gives_typed_frame():
    frame, notes = read(HEADER + "A1,OC Event Rentals LLC,INV-1042,468.20,2026-03-06,2026-03\n"
                                 "A2,Bayside Print & Copy,,125,2026-03-12,\n")
    assert list(frame.columns) == list(ledger.FRAME_COLUMNS)
    assert frame["amount_cents"].tolist() == [46820, 12500]
    assert str(frame["amount_cents"].dtype) == "int64"
    assert str(frame["invoice_date"].dtype) == "datetime64[ns]"
    assert frame["invoice_date"].iloc[0] == pd.Timestamp("2026-03-06")
    assert frame["invoice_ref"].tolist() == ["INV-1042", ""]
    assert frame["billing_period"].tolist() == ["2026-03", ""]
    assert notes == []


def test_references_stay_strings_with_leading_zeros():
    frame, _ = read(HEADER + "A1,Vendor A,000418,10.00,2026-01-05,\n"
                             "A2,Vendor B,1E5,10.00,2026-01-05,\n"
                             "A3,Vendor C,0042,10.00,2026-01-05,\n")
    assert frame["invoice_ref"].tolist() == ["000418", "1E5", "0042"]


@pytest.mark.parametrize("text, cents", [
    ("468.2", 46820), ("468.20", 46820), ("$1,234.50", 123450), ("0.29", 29), ("4.35", 435),
    (" 12.00 ", 1200), ("1234", 123400), ("99999999.99", 9999999999), ("1,234", 123400),
])
def test_amounts_become_exact_cents(text, cents):
    frame, _ = read('vendor,amount,invoice_date\nVendor A,"%s",2026-01-05\n' % text)
    assert frame["amount_cents"].tolist() == [cents]


@pytest.mark.parametrize("text", ["1e3", "0", "-5", "12.345", "1,23", "NaN", "inf", "abc", "", "12,50", "(5.00)", "1005.675"])
def test_bad_amounts_are_rejected_with_the_row_number(text):
    message = problems_of('vendor,amount,invoice_date\nVendor A,"%s",2026-01-05\n' % text)
    assert "Row 2" in message and "amount" in message


@pytest.mark.parametrize("text", ["2026-02-30", "03/06/2026", "2026-3-6", "06-03-2026", "", "20260306"])
def test_bad_dates_are_rejected_with_a_hint(text):
    message = problems_of(f"vendor,amount,invoice_date\nVendor A,10.00,{text}\n")
    assert "Row 2" in message and "YYYY-MM-DD" in message


@pytest.mark.parametrize("text", ["2026-13", "Mar 2026", "2026-3", "2026"])
def test_bad_billing_periods_are_rejected(text):
    message = problems_of(f"vendor,amount,invoice_date,billing_period\nVendor A,10.00,2026-03-06,{text}\n")
    assert "billing_period" in message and "YYYY-MM" in message


# -------------------------------------------------------------- file problems

def test_empty_file():
    assert "empty" in problems_of("")
    assert "empty" in problems_of("   \n\n")


def test_header_only_incoming_is_an_error_but_empty_history_is_allowed():
    assert "no invoices" in problems_of("vendor,amount,invoice_date\n")
    frame, notes = read("vendor,amount,invoice_date\n", role="payments")
    assert len(frame) == 0
    assert list(frame.columns) == list(ledger.FRAME_COLUMNS)
    assert str(frame["invoice_date"].dtype) == "datetime64[ns]"
    assert any("no rows" in note for note in notes)


def test_missing_required_column_names_the_column():
    message = problems_of("vendor,invoice_ref,invoice_date\nVendor A,1,2026-01-05\n")
    assert "missing required column: amount" in message
    assert "Columns found: vendor, invoice_ref, invoice_date" in message


def test_headerless_file():
    message = problems_of("Vendor A,INV-1,10.00,2026-01-05,2026-01\n")
    assert "no column names" in message


def test_duplicate_header():
    assert "repeats a column name" in problems_of("vendor,amount,amount,invoice_date\nA,1.00,1.00,2026-01-05\n")


def test_header_case_spaces_and_bom_are_forgiven():
    raw = "﻿ Vendor , AMOUNT ,Invoice Date\r\nVendor A,10.00,2026-01-05\r\n".encode("utf-8")
    frame, _ = read_ledger_csv(raw, "incoming")
    assert frame["vendor"].tolist() == ["Vendor A"]


def test_extra_columns_are_ignored_and_reported():
    frame, notes = read("ledger_id,vendor,amount,invoice_date,memo\nL1,Vendor A,10.00,2026-01-05,hello\n")
    assert len(frame) == 1
    assert "ledger_id, memo" in notes[0]


def test_row_ids_are_assigned_from_file_order():
    frame, _ = read("vendor,amount,invoice_date\nA,1.00,2026-01-05\nB,2.00,2026-01-06\n", role="payments")
    assert frame["row_id"].tolist() == ["PAY-0001", "PAY-0002"]
    frame, _ = read("vendor,amount,invoice_date\nA,1.00,2026-01-05\n")
    assert frame["row_id"].tolist() == ["INC-0001"]


def test_duplicate_and_blank_row_ids():
    message = problems_of(HEADER + "A1,V,1,1.00,2026-01-05,\nA1,W,2,1.00,2026-01-05,\n,X,3,1.00,2026-01-05,\n")
    assert "already used on row 2" in message
    assert "row_id is blank" in message


def test_quoted_commas_blank_lines_and_short_rows():
    frame, _ = read('vendor,invoice_ref,amount,invoice_date,billing_period\n'
                    '"Harbor Print, LLC",7,10.00,2026-01-05\n'
                    '\n'
                    '"Said ""Hi"" Co",8,11.00,2026-01-06,2026-01\n\n')
    assert frame["vendor"].tolist() == ["Harbor Print, LLC", 'Said "Hi" Co']
    assert frame["billing_period"].tolist() == ["", "2026-01"]


def test_unquoted_comma_gives_a_quoting_hint():
    message = problems_of("vendor,amount,invoice_date\nHarbor Print, LLC,10.00,2026-01-05\n")
    assert "Row 2" in message and "double quotes" in message


def test_blank_vendor_and_many_problems_are_limited():
    body = "".join(f",abc,2026-99-99\n" for _ in range(10))
    with pytest.raises(LedgerError) as caught:
        read("vendor,amount,invoice_date\n" + body)
    assert len(caught.value.problems) == ledger.MAX_PROBLEMS_SHOWN + 1
    assert "more" in caught.value.problems[-1]
    assert "vendor is blank" in caught.value.problems[0]


def test_non_utf8_and_binary_files():
    with pytest.raises(LedgerError) as caught:
        read_ledger_csv("vendor,amount,invoice_date\nCaf\xe9,1.00,2026-01-05\n".encode("latin-1"), "incoming")
    assert "UTF-8" in caught.value.title
    with pytest.raises(LedgerError) as caught:
        read_ledger_csv(b"PK\x03\x04\x00\x00\x00\x00vendor", "incoming")
    assert "CSV" in caught.value.title or "UTF-8" in caught.value.title


def test_unicode_vendor_names_survive():
    frame, _ = read("vendor,amount,invoice_date\nCafé Azul Catering,10.00,2026-01-05\n")
    assert frame["vendor"].tolist() == ["Café Azul Catering"]


def test_row_and_size_limits():
    rows = "".join(f"V{i},1.00,2026-01-05\n" for i in range(ledger.MAX_ROWS["incoming"] + 1))
    assert "handles up to" in problems_of("vendor,amount,invoice_date\n" + rows)
    with pytest.raises(LedgerError) as caught:
        read_ledger_csv(b"vendor,amount,invoice_date\n" + b"x" * (ledger.MAX_UPLOAD_BYTES + 1), "incoming")
    assert "larger than" in caught.value.title


def test_reads_paths_and_file_objects(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("vendor,amount,invoice_date\nA,1.00,2026-01-05\n", encoding="utf-8")
    assert len(read_ledger_csv(path, "incoming")[0]) == 1
    assert len(read_ledger_csv(io.BytesIO(path.read_bytes()), "incoming")[0]) == 1


# ------------------------------------------------------------------ formatting

def test_formatting_helpers():
    assert ledger.format_cents(46820) == "$468.20"
    assert ledger.format_cents(5) == "$0.05"
    assert ledger.format_cents(123456789) == "$1,234,567.89"
    assert ledger.format_cents(-1500) == "-$15.00"
    assert ledger.format_date(pd.Timestamp("2026-03-06")) == "Mar 6, 2026"
    assert ledger.format_period("2026-03") == "Mar 2026"
    assert ledger.format_period("") == ""


# ---------------------------------------------------------------- safe export

@pytest.mark.parametrize("text", ["=SUM(A1:A9)", "+1042", "-Acme", "@cmd", "\t=1+1", "\r=1+1", "  =HYPERLINK(1)"])
def test_formula_prefixes_are_escaped(text):
    assert ledger.escape_formula(text) == "'" + text


@pytest.mark.parametrize("text", ["OC Event Rentals LLC", "INV-1042", "000418", "", "A=B", "5-Star Catering"])
def test_ordinary_text_is_unchanged(text):
    assert ledger.escape_formula(text) == text


def test_safe_csv_round_trips_and_escapes_only_text_columns():
    frame = pd.DataFrame({
        "vendor": ['=HYPERLINK("http://x")', "Harbor Print, LLC", "+Plus Co"],
        "invoice_ref": ["-77", "000418", "@ref"],
        "amount": ["468.20", "10.00", "5.00"],
    })
    raw = ledger.to_safe_csv(frame, ["vendor", "invoice_ref"])
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    assert rows[0] == ["vendor", "invoice_ref", "amount"]
    assert rows[1] == ['\'=HYPERLINK("http://x")', "'-77", "468.20"]
    assert rows[2] == ["Harbor Print, LLC", "000418", "10.00"]
    assert rows[3] == ["'+Plus Co", "'@ref", "5.00"]
    assert frame["vendor"].iloc[0].startswith("=")  # the caller's frame is not modified
