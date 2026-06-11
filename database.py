import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "routing.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

CREATE_TICKETS_TABLE = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    subject TEXT,
    message TEXT NOT NULL,
    context TEXT,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    queue TEXT NOT NULL,
    team TEXT NOT NULL,
    sla TEXT NOT NULL,
    routing_note TEXT NOT NULL,
    rag_context TEXT NOT NULL,
    escalation_required INTEGER NOT NULL,
    monitoring_note TEXT,
    assigned_at TEXT NOT NULL
)
"""

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with _connect() as conn:
        conn.execute(CREATE_TICKETS_TABLE)
        # Safe migrations: add new columns to existing databases without data loss.
        for col, definition in [
            ("context", "TEXT"),
            ("monitoring_note", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {col} {definition}")
            except Exception:
                pass  # column already exists — ignore
        conn.commit()

def save_ticket(record: dict) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO tickets (
                customer_id, channel, subject, message,
                context, category, priority, queue, team, sla,
                routing_note, rag_context, escalation_required,
                monitoring_note, assigned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["customer_id"],
                record["channel"],
                record.get("subject"),
                record["message"],
                record.get("context"),
                record["category"],
                record["priority"],
                record["queue"],
                record["team"],
                record["sla"],
                record["routing_note"],
                record["rag_context"],
                int(record["escalation_required"]),
                record.get("monitoring_note"),
                record["assigned_at"],
            ),
        )
        conn.commit()
        return cursor.lastrowid

def _row_to_dict(row: sqlite3.Row) -> dict:
    if row is None:
        return None
    return {
        "ticket_id": row["id"],
        "customer_id": row["customer_id"],
        "channel": row["channel"],
        "subject": row["subject"],
        "context": row["context"],
        "category": row["category"],
        "priority": row["priority"],
        "queue": row["queue"],
        "team": row["team"],
        "sla": row["sla"],
        "routing_note": row["routing_note"],
        "rag_context": row["rag_context"],
        "escalation_required": bool(row["escalation_required"]),
        "monitoring_note": row["monitoring_note"],
        "assigned_at": row["assigned_at"],
    }

def get_ticket_by_id(ticket_id: int) -> dict | None:
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        return _row_to_dict(row)

def get_monitoring_summary() -> dict:
    with _connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) AS total FROM tickets")
        total = cursor.fetchone()["total"]

        by_priority = {row["priority"]: row["count"] for row in conn.execute(
            "SELECT priority, COUNT(*) AS count FROM tickets GROUP BY priority"
        ).fetchall()}

        by_category = {row["category"]: row["count"] for row in conn.execute(
            "SELECT category, COUNT(*) AS count FROM tickets GROUP BY category"
        ).fetchall()}

        by_team = {row["team"]: row["count"] for row in conn.execute(
            "SELECT team, COUNT(*) AS count FROM tickets GROUP BY team"
        ).fetchall()}

        escalation_count = conn.execute(
            "SELECT COUNT(*) AS count FROM tickets WHERE escalation_required = 1"
        ).fetchone()["count"]

        return {
            "total_tickets": total,
            "by_priority": by_priority,
            "by_category": by_category,
            "by_team": by_team,
            "escalation_count": escalation_count,
        }

def get_pending_escalations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets WHERE escalation_required = 1 ORDER BY assigned_at DESC"
        ).fetchall()
        return [_row_to_dict(row) for row in rows]
