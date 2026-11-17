"""Train and evaluate the complaint classification model.

Run from the project root:
    python -m ml.train_model
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "complaints.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "smartdesk_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
REQUIRED_COLUMNS = {"text", "category"}


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the labeled complaint dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=["text", "category"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["category"] != "")]

    if len(frame) < 50:
        raise ValueError("Dataset is too small. Add at least 50 labeled complaints.")
    if frame["category"].nunique() < 2:
        raise ValueError("Dataset must contain at least two categories.")
    return frame.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.3)),
        ]
    )


def evaluate(y_true, y_pred, labels: list[str]) -> dict[str, object]:
    """Calculate real evaluation metrics on the held-out test set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def train() -> dict[str, object]:
    """Train, evaluate and save the model. Returns the metrics dictionary."""
    frame = load_dataset()
    labels = sorted(frame["category"].unique().tolist())

    # Split first, then fit the vectorizer inside the pipeline on training data
    # only. This prevents test-set information from leaking into training.
    X_train, X_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["category"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = evaluate(y_test, predictions, labels)
    metrics.update(
        {
            "dataset_size": int(len(frame)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "categories": labels,
            "random_state": RANDOM_STATE,
            "test_split": TEST_SIZE,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "TF-IDF + Multinomial Naive Bayes",
            "alpha": 0.3,
        }
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    # Save the vectorizer and classifier separately as well, so students can
    # inspect the two stages of the pipeline independently.
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
    joblib.dump(pipeline.named_steps["classifier"], CLASSIFIER_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )

    print("=" * 60)
    print("SmartDesk AI - Model Training Complete")
    print("=" * 60)
    print(f"Dataset records : {len(frame)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")
    print(f"Categories      : {', '.join(labels)}")
    print("-" * 60)
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)   : {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro) : {metrics['f1_macro']:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(
        classification_report(
            y_test, predictions, labels=labels, zero_division=0
        )
    )
    print(f"Model saved to  : {PIPELINE_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    train()
