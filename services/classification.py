"""Complaint classification service built on the trained ML predictor."""

from __future__ import annotations

from ml.predictor import predict_complaint

# When the model is not confident, route the ticket for manual review instead
# of presenting a weak prediction as a reliable answer.
CONFIDENCE_THRESHOLD = 0.30


def classify_complaint(subject: str, description: str) -> dict[str, object]:
    """Predict a complaint category and confidence.

    Returns a dictionary containing:
    - category: final category stored with the ticket
    - confidence: model confidence for its top prediction
    - model_category: the model's original top prediction
    - low_confidence: True when the ticket needs manual review
    - note: human-readable explanation
    """
    text = f"{subject or ''} {description or ''}".strip()
    if not text:
        return {
            "category": "General Inquiry",
            "confidence": 0.0,
            "model_category": "General Inquiry",
            "low_confidence": True,
            "note": "The complaint text was empty, so it was routed to General Inquiry.",
        }

    prediction = predict_complaint(text)
    model_category = str(prediction["category"])
    confidence = float(prediction["confidence"])
    known_terms = int(prediction["known_terms"])

    if known_terms == 0:
        return {
            "category": "General Inquiry",
            "confidence": confidence,
            "model_category": model_category,
            "low_confidence": True,
            "note": (
                "The model did not recognize any words from its training data. "
                "The ticket was routed to General Inquiry for manual review."
            ),
        }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "category": "General Inquiry",
            "confidence": confidence,
            "model_category": model_category,
            "low_confidence": True,
            "note": (
                "The model was not confident enough, so the ticket was routed "
                "to General Inquiry for manual review."
            ),
        }

    return {
        "category": model_category,
        "confidence": confidence,
        "model_category": model_category,
        "low_confidence": False,
        "note": (
            "The complaint was classified by the trained TF-IDF + "
            "Multinomial Naive Bayes model."
        ),
    }
