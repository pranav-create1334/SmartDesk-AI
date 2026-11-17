"""SQLite database helpers for SmartDesk AI.

The database file is created automatically the first time the application
starts, so no external database server is needed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# Project paths are resolved from this file so imports work from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "helpdesk.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"

STATUSES = ("Pending", "In Progress", "Resolved")
CATEGORIES = (
    "Network Issue",
    "Account Issue",
    "Hardware Issue",
    "Software Issue",
    "General Inquiry",
)
PRIORITIES = ("High", "Medium", "Low")


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a connection with rows accessible by column name."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path | str | None = None) -> None:
    """Create the tickets table if it does not exist yet."""
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(db_path) as connection:
        connection.executescript(schema)
        connection.commit()


def create_ticket(
    name: str,
    email: str,
    subject: str,
    description: str,
    category: str,
    priority: str,
    suggested_response: str,
    confidence: float = 0.0,
    status: str = "Pending",
    db_path: Path | str | None = None,
) -> int:
    """Insert one ticket using a parameterized query and return its ID."""
    if status not in STATUSES:
        raise ValueError(f"Invalid status: {status}")

    created_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO tickets
                (name, email, subject, description, category, confidence,
                 priority, suggested_response, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                email.strip(),
                subject.strip(),
                description.strip(),
                category,
                float(confidence),
                priority,
                suggested_response,
                status,
                created_at,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_ticket(ticket_id: int, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Fetch a single ticket by numeric ID."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_tickets(
    search: str = "",
    category: str = "All",
    priority: str = "All",
    status: str = "All",
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Return tickets filtered by search text, category, priority and status."""
    query = "SELECT * FROM tickets WHERE 1 = 1"
    parameters: list[Any] = []

    if search and search.strip():
        search_text = f"%{search.strip()}%"
        normalized = search.strip().upper()
        if normalized.startswith("TKT-"):
            normalized = normalized[4:]
        if normalized.isdigit():
            # Allow searches such as "12", "TKT-000012" or a partial subject.
            query += " AND (id = ? OR subject LIKE ? OR description LIKE ?)"
            parameters.extend([int(normalized), search_text, search_text])
        else:
            query += " AND (subject LIKE ? OR description LIKE ? OR name LIKE ?)"
            parameters.extend([search_text, search_text, search_text])
    if category and category != "All":
        query += " AND category = ?"
        parameters.append(category)
    if priority and priority != "All":
        query += " AND priority = ?"
        parameters.append(priority)
    if status and status != "All":
        query += " AND status = ?"
        parameters.append(status)

    query += " ORDER BY id DESC"
    with get_connection(db_path) as connection:
        frame = pd.read_sql_query(query, connection, params=parameters)
    return frame


def update_ticket_status(
    ticket_id: int, status: str, db_path: Path | str | None = None
) -> bool:
    """Update a ticket status. Returns True when a row was changed."""
    if status not in STATUSES:
        raise ValueError(f"Invalid status: {status}")

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id)
        )
        connection.commit()
        return cursor.rowcount > 0


def get_statistics(db_path: Path | str | None = None) -> dict[str, Any]:
    """Read live dashboard statistics directly from SQLite."""
    with get_connection(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        status_rows = connection.execute(
            "SELECT status, COUNT(*) AS total FROM tickets GROUP BY status"
        ).fetchall()
        category_rows = connection.execute(
            "SELECT category, COUNT(*) AS total FROM tickets GROUP BY category ORDER BY total DESC"
        ).fetchall()
        priority_rows = connection.execute(
            "SELECT priority, COUNT(*) AS total FROM tickets GROUP BY priority ORDER BY total DESC"
        ).fetchall()

    status_counts = {status: 0 for status in STATUSES}
    status_counts.update({row["status"]: row["total"] for row in status_rows})
    return {
        "total": total,
        "status_counts": status_counts,
        "category_counts": {row["category"]: row["total"] for row in category_rows},
        "priority_counts": {row["priority"]: row["total"] for row in priority_rows},
    }


def format_ticket_id(ticket_id: int | str) -> str:
    """Convert a numeric database ID into a readable ticket code."""
    return f"TKT-{int(ticket_id):06d}"


def parse_ticket_code(value: str) -> int | None:
    """Read either 'TKT-000012' or '12' as a numeric ticket ID."""
    text = value.strip().upper()
    if text.startswith("TKT-"):
        text = text[4:]
    return int(text) if text.isdigit() else None
