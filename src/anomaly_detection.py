"""Unsupervised anomaly detection for expense transactions."""

from __future__ import annotations

import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover
    IsolationForest = None

from .analytics import expense_rows


def detect_expense_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Flag unusual expense rows using amount, calendar, and category indicators."""
    if IsolationForest is None:
        raise ImportError("scikit-learn is required for anomaly detection.")
    expenses = expense_rows(df)
    if len(expenses) < 10:
        return expenses.assign(anomaly_score=pd.Series(dtype=float), is_anomaly=False)

    contamination = min(max(float(contamination), 0.001), 0.2)
    features = expenses[["amount"]].copy()
    features["day_of_month"] = expenses["date"].dt.day
    features["month"] = expenses["date"].dt.month
    features = pd.concat([features, pd.get_dummies(expenses["category"], prefix="category")], axis=1)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(features)
    scores = model.decision_function(features)
    result = expenses.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = labels == -1
    return result.sort_values(["is_anomaly", "anomaly_score"], ascending=[False, True])

