"""Quick exploratory snapshot for the synthetic transactions dataset."""

from pathlib import Path

import pandas as pd


def main() -> None:
    file_path = Path("data/raw/synthetic_transactions.csv")
    df = pd.read_csv(file_path)

    print("First 5 transactions:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTransaction types:")
    print(df["transaction_type"].value_counts())

    print("\nCategory distribution:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()

