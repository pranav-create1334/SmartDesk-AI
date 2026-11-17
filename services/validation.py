"""Input validation for complaint submission."""

from __future__ import annotations

import re

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def validate_complaint(
    name: str, email: str, subject: str, description: str
) -> dict[str, str]:
    """Return a dictionary of field errors. An empty dictionary means valid."""
    errors: dict[str, str] = {}

    name = (name or "").strip()
    email = (email or "").strip()
    subject = (subject or "").strip()
    description = (description or "").strip()

    if not name:
        errors["name"] = "Full name is required."
    elif len(name) < 2:
        errors["name"] = "Full name must be at least 2 characters."

    if not email:
        errors["email"] = "Email address is required."
    elif not EMAIL_PATTERN.match(email):
        errors["email"] = "Please enter a valid email address (for example: user@example.com)."

    if not subject:
        errors["subject"] = "Complaint subject is required."
    elif len(subject) < 3:
        errors["subject"] = "Complaint subject must be at least 3 characters."

    if not description:
        errors["description"] = "Complaint description is required."
    elif len(description) < 10:
        errors["description"] = "Please describe the complaint in at least 10 characters."

    return errors
