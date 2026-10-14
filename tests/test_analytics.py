from __future__ import annotations

import pandas as pd

from src.analytics import income_vs_expenses_by_month, kpis, monthly_spending, spending_by_category
from src.data_cleaning import clean_transactions


def test_income_and_expense_separation(sample_transactions):
    cleaned, _, _ = clean_transactions(sample_transactions)
    values = kpis(cleaned)
    assert values["total_expenses"] == 300.0
    assert values["total_income"] == 5000.0
    assert values["net_cash_flow"] == 4700.0
    assert values["transaction_count"] == 3


def test_monthly_aggregation(sample_transactions):
    cleaned, _, _ = clean_transactions(sample_transactions)
    monthly = monthly_spending(cleaned)
    assert len(monthly) == 1
    assert monthly["amount"].iloc[0] == 300.0


def test_empty_dataset_handling(sample_transactions):
    cleaned, _, _ = clean_transactions(sample_transactions)
    empty = cleaned.iloc[0:0].copy()
    assert kpis(empty)["total_expenses"] == 0.0
    assert spending_by_category(empty).empty
    assert income_vs_expenses_by_month(empty).empty

