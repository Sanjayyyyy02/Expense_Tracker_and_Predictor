"""Cleaning pipeline for transaction data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data_loader import (
    VALID_CATEGORIES,
    VALID_TRANSACTION_TYPES,
    validate_required_columns,
)


ESSENTIAL_COLUMNS = ["transaction_id", "date", "description", "amount", "category", "transaction_type"]


@dataclass(frozen=True)
class CleaningSummary:
    input_rows: int
    exact_duplicate_rows_removed: int
    duplicate_transaction_ids_remaining: int
    rejected_rows: int
    invalid_date_rows: int
    invalid_amount_rows: int
    missing_essential_rows: int
    invalid_transaction_type_rows: int
    invalid_category_rows: int
    non_positive_amount_rows: int
    unexpected_currency_rows: int
    output_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_description(value: object) -> str:
    """Normalize whitespace/casing gently while preserving merchant detail."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def clean_transactions(
    raw_df: pd.DataFrame,
    output_path: str | Path | None = None,
    expected_currency: str = "INR",
) -> tuple[pd.DataFrame, CleaningSummary, pd.DataFrame]:
    """Validate, clean, sort, and optionally persist transactions.

    Exact duplicate rows are removed because this synthetic dataset deliberately
    includes repeated rows for cleaning practice. Reused transaction IDs with
    different details are not automatically dropped; they are reported.
    """
    validate_required_columns(raw_df)
    df = raw_df.copy()
    input_rows = len(df)

    duplicate_mask = df.duplicated(keep="first")
    exact_duplicate_rows_removed = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    df["description"] = df["description"].map(normalize_description)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["transaction_type"] = df["transaction_type"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()

    invalid_date = df["date"].isna()
    invalid_amount = df["amount"].isna()
    missing_essential = df[ESSENTIAL_COLUMNS].isna().any(axis=1) | (df["description"] == "")
    invalid_transaction_type = ~df["transaction_type"].isin(VALID_TRANSACTION_TYPES)
    invalid_category = ~df["category"].isin(VALID_CATEGORIES)
    non_positive_amount = df["amount"].fillna(0) <= 0
    unexpected_currency = df["currency"] != expected_currency.upper()

    rejected_mask = (
        invalid_date
        | invalid_amount
        | missing_essential
        | invalid_transaction_type
        | invalid_category
        | non_positive_amount
        | unexpected_currency
    )
    rejected_rows = df.loc[rejected_mask].copy()
    cleaned = df.loc[~rejected_mask].copy()
    cleaned = cleaned.sort_values(["date", "transaction_id"]).reset_index(drop=True)

    duplicate_transaction_ids_remaining = int(cleaned["transaction_id"].duplicated(keep=False).sum())

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output, index=False)

    summary = CleaningSummary(
        input_rows=input_rows,
        exact_duplicate_rows_removed=exact_duplicate_rows_removed,
        duplicate_transaction_ids_remaining=duplicate_transaction_ids_remaining,
        rejected_rows=int(rejected_mask.sum()),
        invalid_date_rows=int(invalid_date.sum()),
        invalid_amount_rows=int(invalid_amount.sum()),
        missing_essential_rows=int(missing_essential.sum()),
        invalid_transaction_type_rows=int(invalid_transaction_type.sum()),
        invalid_category_rows=int(invalid_category.sum()),
        non_positive_amount_rows=int(non_positive_amount.sum()),
        unexpected_currency_rows=int(unexpected_currency.sum()),
        output_rows=len(cleaned),
    )
    return cleaned, summary, rejected_rows

