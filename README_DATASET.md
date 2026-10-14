# Smart Expense & Spending Analyzer — Starter Dataset

This starter dataset is **synthetic** and intended for learning/prototyping.

- File: `data/raw/synthetic_transactions.csv`
- Rows: 1776
- Date coverage: 2025-01-01 through 2026-06-30
- Currency: INR
- Categories: Food, Transport, Groceries, Education, Shopping, Entertainment, Bills, Health, Income

## Important limitations
- All records are artificially generated; they are not real bank transactions or evidence of actual spending patterns.
- `category` is a ground-truth label for initial analysis/testing. Do not use it as an input feature when training a category classifier.
- `is_anomaly_synthetic` is a generator-provided marker for testing only. Do not use it as an input to an unsupervised anomaly detector.
- A small number of duplicate rows and inconsistent descriptions are deliberately included for data-cleaning practice.
- This is a starter dataset, not a benchmark of real consumer spending.

## Suggested first tasks
1. Inspect the CSV with Pandas.
2. Check row count, data types, missing values, duplicate rows and category distribution.
3. Parse `date` as datetime and validate `amount`.
4. Separate income from expenses before calculating spending totals.
5. Create monthly and category-wise summaries.
