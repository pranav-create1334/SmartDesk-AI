"""Rule-based complaint priority detection.

This module is deliberately simple and explainable. It does NOT use machine
learning; it looks for known phrases in the complaint text.
"""

from __future__ import annotations

HIGH_KEYWORDS = (
    "system down",
    "system is down",
    "completely down",
    "server down",
    "server is down",
    "network down",
    "network is down",
    "production down",
    "service is down",
    "not working at all",
    "no internet",
    "network outage",
    "complete outage",
    "critical failure",
    "critical",
    "data loss",
    "cannot access",
    "can not access",
    "can't access",
    "unable to access",
    "not able to access",
    "cannot login",
    "can not login",
    "can't login",
    "unable to login",
    "locked out",
    "all users",
    "entire team",
    "entire building",
    "urgent",
    "immediately",
)

MEDIUM_KEYWORDS = (
    "error",
    "errors",
    "repeatedly",
    "repeated",
    "sometimes",
    "intermittent",
    "slow",
    "performance",
    "partial",
    "installation",
    "install",
    "update",
    "not responding",
    "fails",
    "failed",
    "failure",
    "crash",
    "crashes",
    "freezing",
    "freezes",
    "jam",
    "not printing",
    "not detected",
    "expired",
    "sync",
    "incorrect",
    "wrong results",
    "glitch",
    "disconnecting",
    "drops",
)

LOW_KEYWORDS = (
    "how do i",
    "how can i",
    "question",
    "inquiry",
    "information",
    "feature request",
    "request a feature",
    "request a new feature",
    "suggestion",
    "minor",
    "guide",
    "documentation",
    "demo",
    "walkthrough",
    "feedback",
    "clarification",
    "price",
    "pricing",
    "policy",
    "maintenance schedule",
)


def _find_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    """Return all keywords found in the complaint text."""
    return [keyword for keyword in keywords if keyword in text]


def detect_priority(text: str) -> dict[str, object]:
    """Detect priority using transparent keyword rules.

    Priority order:
    1. Any high-priority phrase means High.
    2. Any medium-priority phrase means Medium (a functional problem beats a
       general question when both appear).
    3. Only low-priority phrases means Low.
    4. No matched phrase defaults to Medium for manual review.
    """
    normalized = (text or "").lower().strip()

    high_matches = _find_keywords(normalized, HIGH_KEYWORDS)
    if high_matches:
        keywords = ", ".join(high_matches[:3])
        return {
            "priority": "High",
            "matched_keywords": high_matches,
            "reason": f"High-priority phrase detected: {keywords}.",
        }

    medium_matches = _find_keywords(normalized, MEDIUM_KEYWORDS)
    if medium_matches:
        keywords = ", ".join(medium_matches[:3])
        return {
            "priority": "Medium",
            "matched_keywords": medium_matches,
            "reason": f"Medium-priority phrase detected: {keywords}.",
        }

    low_matches = _find_keywords(normalized, LOW_KEYWORDS)
    if low_matches:
        keywords = ", ".join(low_matches[:3])
        return {
            "priority": "Low",
            "matched_keywords": low_matches,
            "reason": f"Low-priority phrase detected: {keywords}.",
        }

    return {
        "priority": "Medium",
        "matched_keywords": [],
        "reason": "No specific priority keywords found; defaulted to Medium for manual review.",
    }


def get_priority(text: str) -> str:
    """Return only the priority level for the given text."""
    return str(detect_priority(text)["priority"])
