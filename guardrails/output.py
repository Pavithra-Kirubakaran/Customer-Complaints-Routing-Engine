"""Output guardrails — validate and normalize agent routing decisions."""

from __future__ import annotations

from guardrails.constants import (
    ALLOWED_CATEGORIES,
    ALLOWED_PRIORITIES,
    ALLOWED_SLAS,
    ALLOWED_TEAMS,
    MANUAL_REVIEW_QUEUE,
    MANUAL_REVIEW_TEAM,
    TEAM_MAPPING,
)


def _normalize_priority(value: str | None) -> tuple[str, list[str]]:
    corrections: list[str] = []
    priority = (value or "low").strip().lower()
    if priority not in ALLOWED_PRIORITIES:
        corrections.append(f"priority '{value}' coerced to 'low'")
        priority = "low"
    return priority, corrections


def _normalize_category(value: str | None) -> tuple[str, list[str]]:
    corrections: list[str] = []
    category = (value or "General Inquiry").strip()
    if category not in ALLOWED_CATEGORIES:
        corrections.append(f"category '{value}' coerced to 'General Inquiry'")
        category = "General Inquiry"
    return category, corrections


def _normalize_team(value: str | None, category: str) -> tuple[str, list[str]]:
    corrections: list[str] = []
    expected = TEAM_MAPPING.get(category, "Customer Success Team")
    team = (value or expected).strip()
    if team not in ALLOWED_TEAMS:
        corrections.append(f"team '{value}' coerced to '{expected}'")
        team = expected
    return team, corrections


def _normalize_sla(value: str | None) -> tuple[str, list[str]]:
    corrections: list[str] = []
    sla = (value or "72 Hours").strip()
    if sla not in ALLOWED_SLAS:
        corrections.append(f"sla '{value}' coerced to '72 Hours'")
        sla = "72 Hours"
    return sla, corrections


def validate_and_normalize_output(record: dict) -> tuple[dict, list[str]]:
    """Clamp routing output to known enums and teams."""
    normalized = dict(record)
    corrections: list[str] = []

    category, cat_fixes = _normalize_category(normalized.get("category"))
    priority, pri_fixes = _normalize_priority(normalized.get("priority"))
    team, team_fixes = _normalize_team(normalized.get("team"), category)
    sla, sla_fixes = _normalize_sla(normalized.get("sla"))
    corrections.extend(cat_fixes + pri_fixes + team_fixes + sla_fixes)

    queue = normalized.get("queue") or category
    if queue not in ALLOWED_CATEGORIES and queue != MANUAL_REVIEW_QUEUE:
        corrections.append(f"queue '{queue}' coerced to '{category}'")
        queue = category

    normalized["category"] = category
    normalized["priority"] = priority
    normalized["team"] = team
    normalized["sla"] = sla
    normalized["queue"] = queue
    return normalized, corrections


def apply_manual_review_route(record: dict) -> dict:
    """Route uncertain or HITL-flagged tickets to the manual review queue."""
    updated = dict(record)
    updated["queue"] = MANUAL_REVIEW_QUEUE
    updated["team"] = MANUAL_REVIEW_TEAM
    updated["routing_note"] = (
        f"HITL review required. Original route hint: {record.get('team', 'unknown')}. "
        f"{record.get('routing_note') or ''}"
    ).strip()
    return updated


def build_routing_explanation(record: dict) -> str:
    """Human-readable decision provenance for audit/UI."""
    parts = [
        f"category={record.get('category')} "
        f"(confidence={record.get('category_confidence', 'n/a')})",
        f"priority={record.get('priority')} "
        f"(confidence={record.get('priority_confidence', 'n/a')})",
        f"team={record.get('team')}",
        f"sla={record.get('sla')}",
    ]
    if record.get("rag_kb_version"):
        parts.append(f"kb={record['rag_kb_version']}")
    if record.get("accepted_rag_count") is not None:
        parts.append(f"rag_matches={record['accepted_rag_count']}")
    if record.get("escalation_reasons"):
        parts.append(f"reasons={','.join(record['escalation_reasons'])}")
    return " | ".join(parts)
