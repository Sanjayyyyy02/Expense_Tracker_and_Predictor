"""Schema mapping and validation for user-supplied expense CSV files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import pandas as pd


STANDARD_COLUMNS = ("date", "amount", "description", "category", "transaction_type", "payment_method")


@dataclass(frozen=True)
class UploadSummary:
    input_rows: int
    accepted_rows: int
    rejected_rows: int
    duplicate_rows: int
    invalid_date_rows: int
    invalid_amount_rows: int
    income_data_available: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def suggest_column(columns: list[str], choices: list[str]) -> str | None:
    """Suggest one source column using case-insensitive exact/partial matches."""
    normalized = {column: re.sub(r"[^a-z0-9]", "", column.lower()) for column in columns}
    for choice in choices:
        target = re.sub(r"[^a-z0-9]", "", choice.lower())
        for column, value in normalized.items():
            if value == target:
                return column
    for choice in choices:
        target = re.sub(r"[^a-z0-9]", "", choice.lower())
        for column, value in normalized.items():
            if target in value or value in target:
                return column
    return None


def suggested_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)
    return {
        "date": suggest_column(columns, ["date", "transaction date", "time", "timestamp"]),
        "amount": suggest_column(columns, ["amount", "transaction amount", "value", "debit"]),
        "description": suggest_column(columns, ["description", "merchant", "narration", "details", "note"]),
        "category": suggest_column(columns, ["category", "type", "expense category"]),
        "transaction_type": suggest_column(columns, ["transaction type", "transactiontype", "credit debit", "debit credit"]),
        "payment_method": suggest_column(columns, ["payment method", "mode", "payment mode", "method"]),
    }


def parse_amount(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    negative_parentheses = text.str.match(r"^\(.*\)$", na=False)
    numeric_token = text.str.extract(r"(-?\d[\d,]*(?:\.\d+)?)", expand=False)
    parsed = pd.to_numeric(numeric_token.str.replace(",", "", regex=False), errors="coerce")
    return parsed.where(~negative_parentheses, -parsed.abs())


def normalize_uploaded_transactions(
    raw: pd.DataFrame,
    mapping: dict[str, str | None],
    positive_amount_means: str = "expense",
) -> tuple[pd.DataFrame, UploadSummary, pd.DataFrame]:
    """Map a user CSV to the dashboard schema without inventing unavailable data."""
    date_column = mapping.get("date")
    amount_column = mapping.get("amount")
    if not date_column or not amount_column:
        raise ValueError("Select a date column and an amount column before analysing the file.")

    data = pd.DataFrame(index=raw.index)
    data["date"] = pd.to_datetime(raw[date_column], errors="coerce")
    signed_amount = parse_amount(raw[amount_column])
    type_column = mapping.get("transaction_type")
    income_available = bool(type_column)
    if type_column:
        labels = raw[type_column].astype("string").str.strip().str.lower()
        income_labels = labels.str.contains(r"income|credit|salary|deposit|refund", regex=True, na=False)
        expense_labels = labels.str.contains(r"expense|debit|payment|withdraw", regex=True, na=False)
        data["transaction_type"] = "unknown"
        data.loc[income_labels, "transaction_type"] = "income"
        data.loc[expense_labels, "transaction_type"] = "expense"
        data.loc[data["transaction_type"] == "unknown", "transaction_type"] = (
            signed_amount.loc[data["transaction_type"] == "unknown"].map(lambda value: "income" if value < 0 else "expense")
        )
    else:
        data["transaction_type"] = "expense"

    if positive_amount_means == "income":
        data["transaction_type"] = data["transaction_type"].map({"income": "expense", "expense": "income"}).fillna("expense")
    elif not type_column:
        data["transaction_type"] = signed_amount.map(lambda value: "income" if value < 0 else "expense")
        income_available = bool((signed_amount < 0).any())

    data["amount"] = signed_amount.abs()
    data["description"] = raw[mapping["description"]].astype("string").fillna("Not provided") if mapping.get("description") else "Not provided"
    data["category"] = raw[mapping["category"]].astype("string").fillna("Uncategorised") if mapping.get("category") else "Uncategorised"
    data["payment_method"] = raw[mapping["payment_method"]].astype("string").fillna("Not provided") if mapping.get("payment_method") else "Not provided"
    data["transaction_id"] = range(1, len(data) + 1)
    data["currency"] = "INR"
    data["is_anomaly_synthetic"] = False

    invalid_date = data["date"].isna()
    invalid_amount = data["amount"].isna() | (data["amount"] <= 0)
    duplicate = data.duplicated(keep="first")
    rejected_mask = invalid_date | invalid_amount | duplicate
    rejected = data.loc[rejected_mask].copy()
    clean = data.loc[~rejected_mask].sort_values("date").reset_index(drop=True)
    summary = UploadSummary(
        input_rows=len(raw),
        accepted_rows=len(clean),
        rejected_rows=int(rejected_mask.sum()),
        duplicate_rows=int(duplicate.sum()),
        invalid_date_rows=int(invalid_date.sum()),
        invalid_amount_rows=int(invalid_amount.sum()),
        income_data_available=income_available,
    )
    return clean, summary, rejected
