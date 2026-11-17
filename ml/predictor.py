"""Load the trained model and predict complaint categories."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "smartdesk_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def model_exists() -> bool:
    """Check whether a trained model is available."""
    return PIPELINE_PATH.exists()


@lru_cache(maxsize=1)
def load_model():
    """Load the saved Scikit-learn pipeline (cached after the first load)."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python -m ml.train_model"
        )
    return joblib.load(PIPELINE_PATH)


def clear_model_cache() -> None:
    """Clear the cache so a newly trained model can be loaded."""
    load_model.cache_clear()


def _count_known_terms(text: str) -> int:
    """Count how many words/phrases in the text exist in the TF-IDF vocabulary."""
    pipeline = load_model()
    vectorizer = pipeline.named_steps["tfidf"]
    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    terms = set(analyzer(text))
    return sum(1 for term in terms if term in vocabulary)


def predict_complaint(text: str) -> dict[str, object]:
    """Predict the category of a complaint using the trained pipeline."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Complaint text cannot be empty.")

    pipeline = load_model()
    probabilities = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    best_index = int(probabilities.argmax())

    return {
        "category": str(classes[best_index]),
        "confidence": float(probabilities[best_index]),
        "known_terms": _count_known_terms(cleaned),
        "probabilities": {
            str(category): float(probability)
            for category, probability in zip(classes, probabilities)
        },
    }


def get_model_metrics() -> dict[str, object] | None:
    """Read the real metrics saved by the training script, if they exist."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
