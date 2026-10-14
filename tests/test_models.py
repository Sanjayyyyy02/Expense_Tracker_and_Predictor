from __future__ import annotations

import pandas as pd
import pytest

from src.anomaly_detection import detect_expense_anomalies
from src.categorization import predict_category, train_category_model
from src.spending_prediction import evaluate_monthly_forecast


def classification_fixture() -> pd.DataFrame:
    rows = []
    merchants = {
        "Food": ["Cafe Coffee Day", "Pizza Corner", "Tea Stall"],
        "Transport": ["Namma Metro", "Uber Ride", "Bus Pass"],
        "Groceries": ["Vegetable Shop", "Big Basket", "Milk Store"],
    }
    transaction_id = 1
    for category, descriptions in merchants.items():
        for idx in range(12):
            rows.append(
                {
                    "transaction_id": transaction_id,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
                    "description": descriptions[idx % len(descriptions)],
                    "amount": 100 + idx,
                    "category": category,
                    "transaction_type": "expense",
                    "payment_method": "UPI",
                    "currency": "INR",
                    "is_anomaly_synthetic": False,
                }
            )
            transaction_id += 1
    return pd.DataFrame(rows)


def monthly_fixture(months: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(months):
        rows.append(
            {
                "transaction_id": i + 1,
                "date": pd.Timestamp("2025-01-01") + pd.DateOffset(months=i),
                "description": "Monthly expense",
                "amount": 1000 + i * 50,
                "category": "Bills",
                "transaction_type": "expense",
                "payment_method": "UPI",
                "currency": "INR",
                "is_anomaly_synthetic": False,
            }
        )
    return pd.DataFrame(rows)


def test_transaction_categorization_pipeline():
    try:
        model, metrics = train_category_model(classification_fixture())
    except ImportError as exc:
        pytest.skip(str(exc))
    assert metrics["accuracy"] >= 0.5
    assert predict_category(model, "Namma Metro station") == "Transport"


def test_forecasting_insufficient_data():
    result = evaluate_monthly_forecast(monthly_fixture(5))
    assert result["supported"] is False


def test_forecasting_supported_with_history():
    try:
        result = evaluate_monthly_forecast(monthly_fixture(14), holdout_months=3)
    except ImportError as exc:
        pytest.skip(str(exc))
    assert result["supported"] is True
    assert result["next_month_forecast"] > 0


def test_anomaly_detection_on_valid_inputs():
    data = monthly_fixture(12)
    data.loc[len(data)] = data.iloc[-1].copy()
    data.loc[len(data) - 1, "transaction_id"] = 99
    data.loc[len(data) - 1, "amount"] = 10000
    try:
        result = detect_expense_anomalies(data, contamination=0.1)
    except ImportError as exc:
        pytest.skip(str(exc))
    assert "is_anomaly" in result.columns
    assert result["is_anomaly"].sum() >= 1

