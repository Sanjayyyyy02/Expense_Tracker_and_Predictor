"""Data loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_id",
    "date",
    "description",
    "amount",
    "category",
    "transaction_type",
    "payment_method",
    "currency",
    "is_anomaly_synthetic",
}

VALID_TRANSACTION_TYPES = {"expense", "income"}
VALID_CATEGORIES = {
    "Food",
    "Transport",
    "Groceries",
    "Education",
    "Shopping",
    "Entertainment",
    "Bills",
    "Health",
    "Income",
}


class DataValidationError(ValueError):
    """Raised when input data does not satisfy the expected schema."""


def validate_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}")


def load_transactions_csv(path: str | Path) -> pd.DataFrame:
    """Load transactions from CSV and validate required columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    return df

