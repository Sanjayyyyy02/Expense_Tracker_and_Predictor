from __future__ import annotations

import pandas as pd
import pytest

from src.data_cleaning import clean_transactions
from src.data_loader import DataValidationError, load_transactions_csv, validate_required_columns


def test_load_transactions_csv_and_required_columns(tmp_path, sample_transactions):
    path = tmp_path / "transactions.csv"
    sample_transactions.to_csv(path, index=False)
    loaded = load_transactions_csv(path)
    assert loaded.shape[0] == 5


def test_missing_required_column_raises(sample_transactions):
    with pytest.raises(DataValidationError, match="Missing required column"):
        validate_required_columns(sample_transactions.drop(columns=["amount"]))


def test_cleaning_converts_dates_amounts_and_removes_duplicates(sample_transactions):
    cleaned, summary, rejected = clean_transactions(sample_transactions)
    assert summary.input_rows == 5
    assert summary.exact_duplicate_rows_removed == 1
    assert summary.invalid_date_rows == 1
    assert summary.invalid_amount_rows == 1
    assert summary.rejected_rows == 1
    assert summary.output_rows == 3
    assert len(rejected) == 1
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert pd.api.types.is_numeric_dtype(cleaned["amount"])


def test_cleaning_rejects_non_positive_amount(sample_transactions):
    data = sample_transactions.iloc[:1].copy()
    data.loc[0, "amount"] = "0"
    cleaned, summary, _ = clean_transactions(data)
    assert cleaned.empty
    assert summary.non_positive_amount_rows == 1

