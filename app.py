from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics import (
    income_vs_expenses_by_month,
    kpis,
    money,
    monthly_spending,
    payment_method_distribution,
    spending_by_category,
    top_descriptions,
    weekly_spending,
)
from src.anomaly_detection import detect_expense_anomalies
from src.categorization import predict_category, train_category_model
from src.spending_prediction import evaluate_monthly_forecast
from src.upload_data import normalize_uploaded_transactions, suggested_mapping


PROJECT_ROOT = Path(__file__).parent
st.set_page_config(page_title="Smart Expense Analyzer", page_icon="Rs", layout="wide")


def mapping_selector(raw: pd.DataFrame) -> tuple[dict[str, str | None], str]:
    suggestions = suggested_mapping(raw)
    options = ["-- Not available --", *list(raw.columns)]
    st.sidebar.subheader("Column mapping")
    st.sidebar.caption("Review these mappings before analysis. Date and amount are required.")
    labels = {
        "date": "Date column *",
        "amount": "Amount column *",
        "description": "Description / merchant",
        "category": "Category",
        "transaction_type": "Income / expense type",
        "payment_method": "Payment method",
    }
    mapping: dict[str, str | None] = {}
    for field, label in labels.items():
        suggested = suggestions[field]
        index = options.index(suggested) if suggested in options else 0
        chosen = st.sidebar.selectbox(label, options, index=index, key=f"map_{field}")
        mapping[field] = None if chosen == options[0] else chosen
    convention = st.sidebar.radio(
        "If type labels are present, positive amounts represent",
        ["expense", "income"],
        horizontal=True,
    )
    return mapping, convention


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        start, end = date_range
        df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]

    categories = sorted(df["category"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Categories", categories, default=categories)
    if selected_categories:
        df = df[df["category"].isin(selected_categories)]

    types = sorted(df["transaction_type"].dropna().unique())
    selected_types = st.sidebar.multiselect("Transaction types", types, default=types)
    if selected_types:
        df = df[df["transaction_type"].isin(selected_types)]

    methods = sorted(df["payment_method"].dropna().unique())
    selected_methods = st.sidebar.multiselect("Payment methods", methods, default=methods)
    if selected_methods:
        df = df[df["payment_method"].isin(selected_methods)]
    return df


def show_overview(df: pd.DataFrame, income_data_available: bool) -> None:
    values = kpis(df, income_data_available=income_data_available)
    cols = st.columns(4)
    cols[0].metric("Total expenses", money(values["total_expenses"]))
    cols[1].metric("Total income", money(values["total_income"]) if values["total_income"] is not None else "Not available")
    cols[2].metric("Net cash flow", money(values["net_cash_flow"]) if values["net_cash_flow"] is not None else "Not available")
    cols[3].metric("Transactions", f"{values['transaction_count']:,}")
    if not income_data_available:
        st.info("Your file does not identify income transactions. Income and net cash flow are intentionally not calculated.")


def show_charts(df: pd.DataFrame, income_data_available: bool) -> None:
    st.subheader("Spending Analytics")
    if df.empty:
        st.info("No transactions match the active filters.")
        return

    left, right = st.columns(2)
    with left:
        category = spending_by_category(df)
        st.caption("Category-wise expenses")
        if category.empty:
            st.info("No expense records in this selection.")
        else:
            st.bar_chart(category.set_index("category")["amount"])
    with right:
        monthly = monthly_spending(df)
        st.caption("Monthly expense trend")
        if monthly.empty:
            st.info("No monthly expense trend available.")
        else:
            st.line_chart(monthly.set_index("month")["amount"])

    left, right = st.columns(2)
    with left:
        weekly = weekly_spending(df)
        st.caption("Weekly spending")
        if not weekly.empty:
            st.line_chart(weekly.set_index("week_start")["amount"])
    with right:
        methods = payment_method_distribution(df)
        st.caption("Payment-method distribution")
        if not methods.empty:
            st.bar_chart(methods.set_index("payment_method")["amount"])

    if income_data_available:
        monthly_mix = income_vs_expenses_by_month(df)
        st.caption("Income versus expenses")
        if not monthly_mix.empty:
            st.bar_chart(monthly_mix.set_index("month")[["income", "expense"]])

    st.dataframe(top_descriptions(df), use_container_width=True, hide_index=True)


def show_explorer(df: pd.DataFrame) -> None:
    st.subheader("Transaction Explorer")
    query = st.text_input("Search descriptions", "")
    display = df
    if query.strip():
        display = df[df["description"].str.contains(query.strip(), case=False, na=False)]
    st.dataframe(display.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        display.to_csv(index=False).encode("utf-8"),
        "filtered_transactions.csv",
        "text/csv",
    )


def show_ml(clean_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.subheader("ML Insights")
    tabs = st.tabs(["Categorization", "Monthly forecast", "Anomalies"])

    with tabs[0]:
        description = st.text_input("Transaction description", "Namma Metro")
        if st.button("Predict category"):
            try:
                model, metrics = train_category_model(clean_df)
                st.success(f"Predicted category: {predict_category(model, description)}")
                st.write(
                    {
                        "accuracy": round(metrics["accuracy"], 3),
                        "precision_weighted": round(metrics["precision_weighted"], 3),
                        "recall_weighted": round(metrics["recall_weighted"], 3),
                        "f1_weighted": round(metrics["f1_weighted"], 3),
                    }
                )
                st.caption(metrics["limitation"])
            except Exception as exc:
                st.warning(f"Category model is unavailable: {exc}")

    with tabs[1]:
        try:
            result = evaluate_monthly_forecast(clean_df)
            if not result["supported"]:
                st.info(result["reason"])
            else:
                st.metric(
                    f"Forecast for {result['next_month'].strftime('%B %Y')}",
                    money(result["next_month_forecast"]),
                )
                st.write(
                    {
                        "model_mae": round(result["mae"], 2),
                        "model_rmse": round(result["rmse"], 2),
                        "previous_month_baseline_mae": round(result["baseline_mae"], 2),
                        "previous_month_baseline_rmse": round(result["baseline_rmse"], 2),
                    }
                )
                st.dataframe(result["holdout"], use_container_width=True, hide_index=True)
                st.caption(result["limitation"])
        except Exception as exc:
            st.warning(f"Forecast is unavailable: {exc}")

    with tabs[2]:
        sensitivity = st.slider("Anomaly sensitivity", 0.005, 0.10, 0.02, step=0.005)
        try:
            anomalies = detect_expense_anomalies(filtered_df, contamination=sensitivity)
            flagged = anomalies[anomalies["is_anomaly"]]
            st.caption("Anomalies are unusual observations, not proof of fraud or errors.")
            st.dataframe(
                flagged[["date", "description", "amount", "category", "payment_method", "anomaly_score"]],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.warning(f"Anomaly detection is unavailable: {exc}")


def show_quality(summary: dict, rejected: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    st.subheader("Data Quality")
    st.write(summary)
    st.write("Missing values in cleaned data")
    st.dataframe(clean_df.isna().sum().rename("missing_count"), use_container_width=True)
    st.write(f"Rejected rows retained for review: {len(rejected):,}")
    if not rejected.empty:
        st.dataframe(rejected, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Smart Expense & Spending Analyzer")
    st.caption("Upload a CSV, review the column mapping, and analyse only the data in that file.")

    st.sidebar.header("Dataset")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload your Kaggle expense CSV in the sidebar to begin. No sample data is shown by default.")
        return
    try:
        raw = pd.read_csv(uploaded)
        if raw.empty:
            st.warning("The uploaded CSV has no rows.")
            return
        mapping, convention = mapping_selector(raw)
        clean_df, upload_summary, rejected = normalize_uploaded_transactions(raw, mapping, convention)
        summary = upload_summary.to_dict()
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    filtered_df = filter_data(clean_df)
    income_data_available = bool(summary["income_data_available"])
    show_overview(filtered_df, income_data_available)
    show_charts(filtered_df, income_data_available)
    show_explorer(filtered_df)
    show_ml(clean_df, filtered_df)
    show_quality(summary, rejected, clean_df)


if __name__ == "__main__":
    main()

