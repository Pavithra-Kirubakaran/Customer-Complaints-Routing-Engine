import json
import sqlite3
from pathlib import Path

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

CREATE_AUDIT_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
)
"""

TICKET_MIGRATIONS = [
    ("context", "TEXT"),
    ("monitoring_note", "TEXT"),
    ("escalation_reasons", "TEXT"),
    ("hitl_required", "INTEGER DEFAULT 0"),
    ("rag_sources", "TEXT"),
    ("rag_kb_version", "TEXT"),
    ("category_confidence", "REAL"),
    ("priority_confidence", "REAL"),
    ("audit_trace", "TEXT"),
    ("routing_explanation", "TEXT"),
    ("accepted_rag_count", "INTEGER DEFAULT 0"),
    ("rag_guardrail_triggered", "INTEGER DEFAULT 0"),
]


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(CREATE_TICKETS_TABLE)
        conn.execute(CREATE_AUDIT_EVENTS_TABLE)
        for col, definition in TICKET_MIGRATIONS:
            try:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {col} {definition}")
            except Exception:
                pass
        conn.commit()


def _serialize_json(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True)


def _deserialize_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def save_ticket(record: dict) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO tickets (
                customer_id, channel, subject, message,
                context, category, priority, queue, team, sla,
                routing_note, rag_context, escalation_required,
                monitoring_note, assigned_at,
                escalation_reasons, hitl_required, rag_sources, rag_kb_version,
                category_confidence, priority_confidence, audit_trace,
                routing_explanation, accepted_rag_count, rag_guardrail_triggered
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _serialize_json(record.get("escalation_reasons")),
                int(record.get("hitl_required", False)),
                _serialize_json(record.get("rag_sources")),
                record.get("rag_kb_version"),
                record.get("category_confidence"),
                record.get("priority_confidence"),
                _serialize_json(record.get("audit_trace")),
                record.get("routing_explanation"),
                int(record.get("accepted_rag_count") or 0),
                int(record.get("rag_guardrail_triggered", False)),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def save_audit_event(ticket_id: int, event_type: str, payload: dict) -> int:
    from datetime import datetime

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO audit_events (ticket_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                ticket_id,
                event_type,
                json.dumps(payload, ensure_ascii=True),
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
        return cursor.lastrowid


def _row_to_dict(row: sqlite3.Row) -> dict:
    if row is None:
        return None
    keys = row.keys()
    return {
        "ticket_id": row["id"],
        "customer_id": row["customer_id"],
        "channel": row["channel"],
        "subject": row["subject"],
        "message": row["message"],
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
        "escalation_reasons": _deserialize_json(
            row["escalation_reasons"] if "escalation_reasons" in keys else None,
            [],
        ),
        "hitl_required": bool(row["hitl_required"]) if "hitl_required" in keys else False,
        "rag_sources": _deserialize_json(
            row["rag_sources"] if "rag_sources" in keys else None,
            [],
        ),
        "rag_kb_version": row["rag_kb_version"] if "rag_kb_version" in keys else None,
        "category_confidence": row["category_confidence"] if "category_confidence" in keys else None,
        "priority_confidence": row["priority_confidence"] if "priority_confidence" in keys else None,
        "audit_trace": _deserialize_json(
            row["audit_trace"] if "audit_trace" in keys else None,
            {},
        ),
        "routing_explanation": row["routing_explanation"] if "routing_explanation" in keys else None,
        "accepted_rag_count": row["accepted_rag_count"] if "accepted_rag_count" in keys else 0,
        "rag_guardrail_triggered": bool(row["rag_guardrail_triggered"])
        if "rag_guardrail_triggered" in keys
        else False,
    }


def get_ticket_by_id(ticket_id: int) -> dict | None:
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        return _row_to_dict(row)


def get_all_tickets(limit: int = 500) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets ORDER BY assigned_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def get_monitoring_summary() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS total FROM tickets").fetchone()["total"]

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

        hitl_count = conn.execute(
            "SELECT COUNT(*) AS count FROM tickets WHERE hitl_required = 1"
        ).fetchone()["count"]

        rag_guardrail_count = conn.execute(
            "SELECT COUNT(*) AS count FROM tickets WHERE rag_guardrail_triggered = 1"
        ).fetchone()["count"]

        output_correction_count = 0
        for row in conn.execute("SELECT escalation_reasons FROM tickets").fetchall():
            reasons = _deserialize_json(row["escalation_reasons"], [])
            if "output_normalized" in reasons:
                output_correction_count += 1

        return {
            "total_tickets": total,
            "by_priority": by_priority,
            "by_category": by_category,
            "by_team": by_team,
            "escalation_count": escalation_count,
            "hitl_count": hitl_count,
            "rag_guardrail_count": rag_guardrail_count,
            "output_correction_count": output_correction_count,
        }


def get_pending_escalations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM tickets
            WHERE escalation_required = 1 OR hitl_required = 1
            ORDER BY assigned_at DESC
            """
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def get_ticket_audit(ticket_id: int) -> dict | None:
    ticket = get_ticket_by_id(ticket_id)
    if ticket is None:
        return None

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, event_type, payload_json, created_at
            FROM audit_events
            WHERE ticket_id = ?
            ORDER BY created_at ASC
            """,
            (ticket_id,),
        ).fetchall()

    events = []
    for row in rows:
        events.append(
            {
                "audit_id": row["id"],
                "event_type": row["event_type"],
                "created_at": row["created_at"],
                "payload": _deserialize_json(row["payload_json"], {}),
            }
        )

    return {
        "ticket_id": ticket_id,
        "routing_explanation": ticket.get("routing_explanation"),
        "escalation_reasons": ticket.get("escalation_reasons", []),
        "hitl_required": ticket.get("hitl_required", False),
        "audit_trace": ticket.get("audit_trace", {}),
        "events": events,
    }


def get_guardrails_summary() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM tickets").fetchone()["count"]
        if total == 0:
            return {
                "total_tickets": 0,
                "hitl_rate": 0.0,
                "escalation_rate": 0.0,
                "rag_guardrail_rate": 0.0,
                "avg_category_confidence": None,
                "avg_priority_confidence": None,
                "top_escalation_reasons": {},
            }

        hitl_count = conn.execute(
            "SELECT COUNT(*) AS count FROM tickets WHERE hitl_required = 1"
        ).fetchone()["count"]
        escalation_count = conn.execute(
            "SELECT COUNT(*) AS count FROM tickets WHERE escalation_required = 1"
        ).fetchone()["count"]
        rag_count = conn.execute(
            "SELECT COUNT(*) AS count FROM tickets WHERE rag_guardrail_triggered = 1"
        ).fetchone()["count"]
        avg_category = conn.execute(
            "SELECT AVG(category_confidence) AS avg_value FROM tickets WHERE category_confidence IS NOT NULL"
        ).fetchone()["avg_value"]
        avg_priority = conn.execute(
            "SELECT AVG(priority_confidence) AS avg_value FROM tickets WHERE priority_confidence IS NOT NULL"
        ).fetchone()["avg_value"]

    reason_counts: dict[str, int] = {}
    for ticket in get_all_tickets(limit=1000):
        for reason in ticket.get("escalation_reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "total_tickets": total,
        "hitl_rate": round(hitl_count / total, 3),
        "escalation_rate": round(escalation_count / total, 3),
        "rag_guardrail_rate": round(rag_count / total, 3),
        "avg_category_confidence": round(avg_category, 3) if avg_category is not None else None,
        "avg_priority_confidence": round(avg_priority, 3) if avg_priority is not None else None,
        "top_escalation_reasons": dict(
            sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        ),
    }
