"""Predefined suggested responses based on the predicted category."""

from __future__ import annotations

CATEGORY_RESPONSES = {
    "Network Issue": (
        "Please check your internet connection and restart your router. "
        "If the issue continues, contact the IT support team."
    ),
    "Account Issue": (
        "Please verify your login credentials and use the password recovery "
        "option if required."
    ),
    "Hardware Issue": (
        "Please check the hardware connection and restart your device. "
        "Contact technical support if the problem continues."
    ),
    "Software Issue": (
        "Please restart the application and verify that the required software "
        "version is installed."
    ),
    "General Inquiry": (
        "Thank you for contacting support. Your request has been received and "
        "will be reviewed by the support team."
    ),
}

DEFAULT_RESPONSE = CATEGORY_RESPONSES["General Inquiry"]


def get_suggested_response(category: str) -> str:
    """Return the standard first-line response for a category."""
    return CATEGORY_RESPONSES.get(category, DEFAULT_RESPONSE)
