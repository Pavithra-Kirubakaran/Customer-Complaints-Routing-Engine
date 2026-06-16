"""PII detection, input rejection, and output redaction for complaint routing."""

from __future__ import annotations

import re
from typing import Iterable

# Types that must never enter the system — reject at the API boundary.
BLOCKED_PII_TYPES = frozenset({"credit_card", "ssn"})

# Types redacted before agents, LLM calls, persistence, and API responses.
REDACTED_PII_TYPES = frozenset({"email", "phone", "credit_card", "ssn", "ipv4"})

REDACTION_TOKENS = {
    "email": "[EMAIL_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "ssn": "[SSN_REDACTED]",
    "ipv4": "[IP_REDACTED]",
}

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "phone",
        re.compile(
            r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "ipv4",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
]

_OUTPUT_STRING_FIELDS = (
    "customer_id",
    "subject",
    "message",
    "context",
    "routing_note",
    "rag_context",
    "monitoring_note",
)


class BlockedPIIError(ValueError):
    """Raised when input contains PII that must not be submitted."""


def _luhn_valid(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _match_credit_card(text: str) -> bool:
    for match in _PII_PATTERNS[2][1].finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return True
    return False


def scan_pii(text: str) -> list[str]:
    """Return deduplicated PII type labels found in text."""
    if not text or not text.strip():
        return []

    found: list[str] = []
    for pii_type, pattern in _PII_PATTERNS:
        if pii_type == "credit_card":
            if _match_credit_card(text):
                found.append(pii_type)
            continue
        if pattern.search(text):
            found.append(pii_type)
    return found


def validate_no_blocked_pii(text: str | None, field_name: str) -> None:
    """Reject text that contains blocked PII such as card numbers or SSNs."""
    if not text or not text.strip():
        return

    blocked = [pii_type for pii_type in scan_pii(text) if pii_type in BLOCKED_PII_TYPES]
    if blocked:
        labels = ", ".join(sorted(set(blocked)))
        raise BlockedPIIError(
            f"{field_name} contains sensitive data ({labels}). "
            "Remove credit card numbers and Social Security numbers before submitting."
        )


def redact_pii(text: str | None, types: Iterable[str] | None = None) -> str | None:
    """Replace detected PII with stable redaction tokens."""
    if text is None:
        return None
    if not text:
        return text

    allowed = set(types) if types is not None else REDACTED_PII_TYPES
    redacted = text

    for pii_type, pattern in _PII_PATTERNS:
        if pii_type not in allowed:
            continue
        token = REDACTION_TOKENS[pii_type]
        if pii_type == "credit_card":
            redacted = _redact_credit_cards(redacted, token)
        else:
            redacted = pattern.sub(token, redacted)

    return redacted


def _redact_credit_cards(text: str, token: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return token
        return match.group(0)

    return _PII_PATTERNS[2][1].sub(_replace, text)


def sanitize_ticket_record(record: dict) -> dict:
    """Redact PII from ticket fields before persistence or API response."""
    sanitized = dict(record)
    for field in _OUTPUT_STRING_FIELDS:
        if field in sanitized:
            sanitized[field] = redact_pii(sanitized.get(field))
    return sanitized
