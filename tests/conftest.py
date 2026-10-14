from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transaction_id": 1,
                "date": "2026-01-01",
                "description": "Namma Metro",
                "amount": "100",
                "category": "Transport",
                "transaction_type": "expense",
                "payment_method": "UPI",
                "currency": "INR",
                "is_anomaly_synthetic": False,
            },
            {
                "transaction_id": 2,
                "date": "2026-01-02",
                "description": "Salary",
                "amount": "5000",
                "category": "Income",
                "transaction_type": "income",
                "payment_method": "Net Banking",
                "currency": "INR",
                "is_anomaly_synthetic": False,
            },
            {
                "transaction_id": 3,
                "date": "2026-01-03",
                "description": "Coffee",
                "amount": "200",
                "category": "Food",
                "transaction_type": "expense",
                "payment_method": "Card",
                "currency": "INR",
                "is_anomaly_synthetic": False,
            },
            {
                "transaction_id": 3,
                "date": "2026-01-03",
                "description": "Coffee",
                "amount": "200",
                "category": "Food",
                "transaction_type": "expense",
                "payment_method": "Card",
                "currency": "INR",
                "is_anomaly_synthetic": False,
            },
            {
                "transaction_id": 4,
                "date": "bad-date",
                "description": "Broken",
                "amount": "abc",
                "category": "Food",
                "transaction_type": "expense",
                "payment_method": "UPI",
                "currency": "INR",
                "is_anomaly_synthetic": False,
            },
        ]
    )

