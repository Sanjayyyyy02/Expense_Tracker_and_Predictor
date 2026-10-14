from __future__ import annotations

import pandas as pd
import pytest

from src.upload_data import normalize_uploaded_transactions, suggested_mapping


def test_uploaded_csv_mapping_handles_currency_and_negative_income():
    raw = pd.DataFrame(
        {
            "Transaction Date": ["2026-01-01", "2026-01-02", "bad-date"],
            "Amount": ["Rs. 1,200.50", "-5000", "100"],
            "Merchant": ["Grocer", "Salary", "Invalid"],
            "Category": ["Groceries", "Salary", "Other"],
        }
    )
    mapping = {"date": "Transaction Date", "amount": "Amount", "description": "Merchant", "category": "Category", "transaction_type": None, "payment_method": None}
    clean, summary, rejected = normalize_uploaded_transactions(raw, mapping)
    assert len(clean) == 2
    assert summary.income_data_available is True
    assert summary.invalid_date_rows == 1
    assert clean.loc[0, "amount"] == 1200.50
    assert clean.loc[1, "transaction_type"] == "income"
    assert len(rejected) == 1


def test_uploaded_csv_requires_date_and_amount():
    with pytest.raises(ValueError, match="date column and an amount column"):
        normalize_uploaded_transactions(pd.DataFrame({"value": [10]}), {"date": None, "amount": "value"})


def test_mapping_suggestions_find_common_columns():
    raw = pd.DataFrame({"Transaction Date": [], "Amount": [], "Merchant": []})
    mapping = suggested_mapping(raw)
    assert mapping["date"] == "Transaction Date"
    assert mapping["amount"] == "Amount"
    assert mapping["description"] == "Merchant"
