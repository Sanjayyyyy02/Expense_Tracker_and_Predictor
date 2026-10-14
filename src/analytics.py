"""Reusable financial analytics functions."""

from __future__ import annotations

import pandas as pd


def expense_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["transaction_type"].str.lower() == "expense"].copy()


def income_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["transaction_type"].str.lower() == "income"].copy()


def money(value: float) -> str:
    return f"\u20b9{value:,.2f}"


def kpis(df: pd.DataFrame, income_data_available: bool = True) -> dict[str, float | int | None]:
    expenses = expense_rows(df)
    income = income_rows(df)
    total_expenses = float(expenses["amount"].sum()) if not expenses.empty else 0.0
    total_income = float(income["amount"].sum()) if not income.empty else 0.0
    highest = None if expenses.empty else float(expenses["amount"].max())
    average = 0.0 if expenses.empty else float(expenses["amount"].mean())
    return {
        "total_expenses": total_expenses,
        "total_income": total_income if income_data_available else None,
        "net_cash_flow": total_income - total_expenses if income_data_available else None,
        "average_expense_transaction": average,
        "highest_expense_transaction": highest,
        "transaction_count": int(len(df)),
    }


def spending_by_category(df: pd.DataFrame) -> pd.DataFrame:
    expenses = expense_rows(df)
    if expenses.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return expenses.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)


def payment_method_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["payment_method", "count", "amount"])
    return (
        df.groupby("payment_method", as_index=False)
        .agg(count=("transaction_id", "count"), amount=("amount", "sum"))
        .sort_values("count", ascending=False)
    )


def daily_spending(df: pd.DataFrame) -> pd.DataFrame:
    return _resample_expenses(df, "D", "date")


def weekly_spending(df: pd.DataFrame) -> pd.DataFrame:
    return _resample_expenses(df, "W-MON", "week_start")


def monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    return _resample_expenses(df, "MS", "month")


def _resample_expenses(df: pd.DataFrame, freq: str, date_name: str) -> pd.DataFrame:
    expenses = expense_rows(df)
    if expenses.empty:
        return pd.DataFrame(columns=[date_name, "amount"])
    series = expenses.set_index("date")["amount"].resample(freq).sum()
    return series.rename_axis(date_name).reset_index()


def income_vs_expenses_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "expense", "income"])
    grouped = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .pivot_table(index="month", columns="transaction_type", values="amount", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for column in ["expense", "income"]:
        if column not in grouped:
            grouped[column] = 0.0
    return grouped[["month", "expense", "income"]]


def top_descriptions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    expenses = expense_rows(df)
    if expenses.empty:
        return pd.DataFrame(columns=["description", "count", "amount"])
    return (
        expenses.groupby("description", as_index=False)
        .agg(count=("transaction_id", "count"), amount=("amount", "sum"))
        .sort_values(["amount", "count"], ascending=False)
        .head(n)
    )

