"""Monthly spending forecasting baselines."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from .analytics import monthly_spending

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError:  # pragma: no cover
    RandomForestRegressor = None
    mean_absolute_error = mean_squared_error = None


def monthly_expense_series(df: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly_spending(df)
    if monthly.empty:
        return monthly
    full_index = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    return (
        monthly.set_index("month")
        .reindex(full_index, fill_value=0.0)
        .rename_axis("month")
        .reset_index()
    )


def build_forecast_features(monthly: pd.DataFrame) -> pd.DataFrame:
    data = monthly.copy()
    data["lag_1"] = data["amount"].shift(1)
    data["lag_2"] = data["amount"].shift(2)
    data["rolling_3"] = data["amount"].shift(1).rolling(3).mean()
    data["month_number"] = data["month"].dt.month
    data["time_index"] = np.arange(len(data))
    return data.dropna().reset_index(drop=True)


def evaluate_monthly_forecast(df: pd.DataFrame, holdout_months: int = 4) -> dict[str, Any]:
    monthly = monthly_expense_series(df)
    if len(monthly) < 8:
        return {
            "supported": False,
            "reason": "At least 8 months of expense history are needed for a defensible holdout forecast.",
            "monthly": monthly,
        }
    features = build_forecast_features(monthly)
    if len(features) <= holdout_months + 2 or RandomForestRegressor is None:
        return {
            "supported": False,
            "reason": "Not enough engineered monthly observations or scikit-learn is unavailable.",
            "monthly": monthly,
        }

    train = features.iloc[:-holdout_months]
    test = features.iloc[-holdout_months:]
    feature_cols = ["lag_1", "lag_2", "rolling_3", "month_number", "time_index"]
    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    model.fit(train[feature_cols], train["amount"])
    predictions = model.predict(test[feature_cols])
    baseline = test["lag_1"].to_numpy()

    next_month = monthly["month"].max() + pd.DateOffset(months=1)
    next_row = pd.DataFrame(
        {
            "lag_1": [monthly["amount"].iloc[-1]],
            "lag_2": [monthly["amount"].iloc[-2]],
            "rolling_3": [monthly["amount"].tail(3).mean()],
            "month_number": [next_month.month],
            "time_index": [len(monthly)],
        }
    )
    next_forecast = float(model.predict(next_row[feature_cols])[0])
    return {
        "supported": True,
        "monthly": monthly,
        "model": model,
        "holdout": test[["month", "amount"]].assign(predicted=predictions, previous_month_baseline=baseline),
        "mae": float(mean_absolute_error(test["amount"], predictions)),
        "rmse": float(sqrt(mean_squared_error(test["amount"], predictions))),
        "baseline_mae": float(mean_absolute_error(test["amount"], baseline)),
        "baseline_rmse": float(sqrt(mean_squared_error(test["amount"], baseline))),
        "next_month": next_month,
        "next_month_forecast": next_forecast,
        "limitation": "Only 18 synthetic months are available, so this forecast is a baseline demonstration.",
    }

