"""Automated tests for SmartDesk AI business logic.

Run from the project root:
    python -m pytest tests -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from database import db
from ml import predictor
from ml.train_model import train
from services.classification import classify_complaint
from services.priority import detect_priority, get_priority
from services.response import CATEGORY_RESPONSES, get_suggested_response
from services.validation import validate_complaint


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model():
    """Make tests self-contained by training the model if it is missing."""
    if not predictor.model_exists():
        train()
        predictor.clear_model_cache()


@pytest.fixture()
def test_database(tmp_path, monkeypatch):
    """Use a temporary SQLite database so project data is never modified."""
    database_path = tmp_path / "test_helpdesk.db"
    monkeypatch.setattr(db, "DB_PATH", database_path)
    db.init_db()
    return database_path


def sample_ticket(db_path=None, **overrides):
    """Create a valid ticket and return its ID."""
    values = {
        "name": "Test User",
        "email": "test.user@example.com",
        "subject": "Wifi not working",
        "description": "My wifi connection keeps disconnecting every few minutes.",
        "category": "Network Issue",
        "priority": "Medium",
        "suggested_response": get_suggested_response("Network Issue"),
        "confidence": 0.82,
    }
    values.update(overrides)
    return db.create_ticket(**values, db_path=db_path)


# ---------------------------------------------------------------- validation


def test_valid_complaint_passes_validation():
    errors = validate_complaint(
        "Pranav Sharma",
        "pranav@example.com",
        "Cannot connect to wifi",
        "My laptop cannot connect to the office wifi network since this morning.",
    )
    assert errors == {}


def test_empty_fields_are_rejected():
    errors = validate_complaint("", "", "", "")
    assert set(errors.keys()) == {"name", "email", "subject", "description"}


def test_invalid_email_is_rejected():
    errors = validate_complaint(
        "Pranav", "not-an-email", "Login issue", "I cannot login to my account today."
    )
    assert "email" in errors


def test_short_description_is_rejected():
    errors = validate_complaint(
        "Pranav", "pranav@example.com", "Help", "Slow"
    )
    assert "description" in errors


# ------------------------------------------------------------------ priority


def test_high_priority_for_system_down():
    result = detect_priority("The system is completely down for all users.")
    assert result["priority"] == "High"
    assert result["matched_keywords"]
    assert "High-priority" in result["reason"]


def test_medium_priority_for_repeated_software_failure():
    assert get_priority("The application installation fails repeatedly with an error.") == "Medium"


def test_low_priority_for_feature_request():
    assert get_priority("How do I request a new feature for the portal?") == "Low"


def test_priority_defaults_to_medium_without_keywords():
    result = detect_priority("I have a problem with my desk phone.")
    assert result["priority"] == "Medium"
    assert "defaulted to Medium" in result["reason"]


# ------------------------------------------------------------------ response


@pytest.mark.parametrize("category", list(CATEGORY_RESPONSES.keys()))
def test_every_category_has_a_response(category):
    response = get_suggested_response(category)
    assert response == CATEGORY_RESPONSES[category]
    assert len(response) > 20


def test_unknown_category_gets_general_response():
    assert get_suggested_response("Unknown Category") == CATEGORY_RESPONSES["General Inquiry"]


# -------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("subject", "description", "expected"),
    [
        (
            "Wifi disconnects",
            "My wifi keeps disconnecting and there is no internet access.",
            "Network Issue",
        ),
        (
            "Cannot login",
            "I forgot my password and cannot login to my account.",
            "Account Issue",
        ),
        (
            "Laptop screen broken",
            "My laptop screen is cracked and flickering.",
            "Hardware Issue",
        ),
        (
            "Application crash",
            "The application crashes when I open a file.",
            "Software Issue",
        ),
        (
            "Feature question",
            "How do I request a new feature for the dashboard?",
            "General Inquiry",
        ),
    ],
)
def test_clear_complaints_are_classified_correctly(subject, description, expected):
    result = classify_complaint(subject, description)
    assert result["category"] == expected
    assert 0.0 <= float(result["confidence"]) <= 1.0
    assert result["low_confidence"] is False


def test_unknown_words_are_routed_gracefully():
    result = classify_complaint("asdfgh", "qwerty zxcv plmokn")
    assert result["category"] == "General Inquiry"
    assert result["low_confidence"] is True


def test_empty_complaint_is_handled_gracefully():
    result = classify_complaint("", "")
    assert result["category"] == "General Inquiry"
    assert result["confidence"] == 0.0
    assert result["low_confidence"] is True


# ------------------------------------------------------------------ database


def test_ticket_creation_and_retrieval(test_database):
    ticket_id = sample_ticket()
    ticket = db.get_ticket(ticket_id)

    assert ticket is not None
    assert ticket["id"] == ticket_id
    assert ticket["name"] == "Test User"
    assert ticket["status"] == "Pending"
    assert ticket["category"] == "Network Issue"


def test_ticket_resulting_from_full_classification_flow(test_database):
    subject = "Printer is not printing"
    description = "The office printer is not printing any documents after the update."
    classification = classify_complaint(subject, description)
    priority = detect_priority(f"{subject} {description}")["priority"]
    response = get_suggested_response(str(classification["category"]))

    ticket_id = db.create_ticket(
        name="Integration User",
        email="integration@example.com",
        subject=subject,
        description=description,
        category=str(classification["category"]),
        priority=str(priority),
        suggested_response=response,
        confidence=float(classification["confidence"]),
    )

    ticket = db.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket["category"] == classification["category"]
    assert ticket["priority"] == priority
    assert ticket["suggested_response"] == response
    assert ticket["status"] == "Pending"


def test_ticket_status_update(test_database):
    ticket_id = sample_ticket()
    assert db.update_ticket_status(ticket_id, "In Progress") is True
    assert db.get_ticket(ticket_id)["status"] == "In Progress"

    assert db.update_ticket_status(ticket_id, "Resolved") is True
    assert db.get_ticket(ticket_id)["status"] == "Resolved"


def test_invalid_status_is_rejected(test_database):
    with pytest.raises(ValueError):
        db.update_ticket_status(1, "Closed")


def test_updating_missing_ticket_returns_false(test_database):
    assert db.update_ticket_status(9999, "Resolved") is False


def test_all_tickets_search_and_filters(test_database):
    first_id = sample_ticket(subject="Broken printer", category="Hardware Issue", priority="Low")
    second_id = sample_ticket(
        subject="Password reset",
        description="I cannot login and need to reset my password.",
        category="Account Issue",
        priority="High",
    )

    assert len(db.get_all_tickets()) == 2
    assert len(db.get_all_tickets(search="printer")) == 1
    assert db.get_all_tickets(search="printer").iloc[0]["id"] == first_id
    assert len(db.get_all_tickets(search=db.format_ticket_id(second_id))) == 1
    assert len(db.get_all_tickets(category="Account Issue")) == 1
    assert len(db.get_all_tickets(priority="Low")) == 1
    assert len(db.get_all_tickets(status="Pending")) == 2
    assert len(db.get_all_tickets(status="Resolved")) == 0


def test_dashboard_statistics_come_from_database(test_database):
    assert db.get_statistics()["total"] == 0

    first_id = sample_ticket()
    sample_ticket(subject="Password problem", category="Account Issue", priority="High")
    db.update_ticket_status(first_id, "Resolved")

    stats = db.get_statistics()
    assert stats["total"] == 2
    assert stats["status_counts"]["Resolved"] == 1
    assert stats["status_counts"]["Pending"] == 1
    assert stats["category_counts"]["Network Issue"] == 1
    assert stats["priority_counts"]["High"] == 1


def test_ticket_id_formatting_and_parsing():
    assert db.format_ticket_id(1) == "TKT-000001"
    assert db.format_ticket_id("42") == "TKT-000042"
    assert db.parse_ticket_code("TKT-000042") == 42
    assert db.parse_ticket_code("42") == 42
    assert db.parse_ticket_code("not-a-ticket") is None


# -------------------------------------------------------------------- dataset


def test_dataset_is_large_and_balanced_enough():
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "complaints.csv"
    frame = pd.read_csv(dataset_path)
    assert len(frame) >= 60
    assert frame["category"].nunique() == 5
    assert not frame["text"].isna().any()
    assert not frame["category"].isna().any()
