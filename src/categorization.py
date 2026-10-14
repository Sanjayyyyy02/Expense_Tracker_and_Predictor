"""Text classification baseline for transaction categories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
except ImportError:  # pragma: no cover - exercised when optional deps are absent in the dashboard.
    joblib = None
    TfidfVectorizer = LogisticRegression = Pipeline = None
    accuracy_score = classification_report = confusion_matrix = precision_recall_fscore_support = None
    train_test_split = None


def train_category_model(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Train a TF-IDF + Logistic Regression classifier from descriptions only."""
    if Pipeline is None:
        raise ImportError("scikit-learn and joblib are required for categorization.")

    data = df.dropna(subset=["description", "category"]).copy()
    if data["category"].nunique() < 2 or len(data) < 20:
        raise ValueError("Need at least 20 rows and two categories to train a category model.")

    x_train, x_test, y_train, y_test = train_test_split(
        data["description"].astype(str),
        data["category"].astype(str),
        test_size=0.25,
        random_state=random_state,
        stratify=data["category"],
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    labels = sorted(data["category"].unique())
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "limitation": "Synthetic repeated merchant descriptions can make test metrics optimistic for unseen merchants.",
    }
    if model_path is not None:
        if joblib is None:
            raise ImportError("joblib is required to persist the model.")
        output = Path(model_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output)
    return model, metrics


def load_category_model(model_path: str | Path) -> Any:
    if joblib is None:
        raise ImportError("joblib is required to load the model.")
    return joblib.load(model_path)


def predict_category(model: Any, description: str) -> str:
    if not description.strip():
        raise ValueError("Description cannot be empty.")
    return str(model.predict([description])[0])

