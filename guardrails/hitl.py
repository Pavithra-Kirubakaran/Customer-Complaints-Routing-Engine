"""Human-in-the-loop escalation rules and reason codes."""

from __future__ import annotations

import os
from typing import Any

from guardrails.constants import URGENT_CATEGORIES

CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.45"))

REGULATORY_KEYWORDS = (
    "lawyer",
    "attorney",
    "lawsuit",
    "regulator",
    "regulatory",
    "gdpr",
    "hipaa",
    "pci",
    "ombudsman",
    "fraud investigation",
)

HIGH_VALUE_KEYWORDS = (
    "$10,000",
    "10000",
    "ten thousand",
    "class action",
)


def evaluate_hitl(
    *,
    message: str,
    category: str,
    priority: str,
    category_confidence: float | None,
    priority_confidence: float | None,
    escalation_required: bool,
    rag_guardrail_triggered: bool,
    rag_accepted_count: int,
    output_corrections: list[str],
) -> dict[str, Any]:
    """Return HITL flags and structured escalation reason codes."""
    reasons: list[str] = []
    text = (message or "").lower()

    if escalation_required:
        reasons.append("urgent_routing")

    if priority.lower() in {"critical", "high"}:
        if "urgent_routing" not in reasons:
            reasons.append("high_priority")

    if category in URGENT_CATEGORIES and "urgent_routing" not in reasons:
        reasons.append("urgent_category")

    if category_confidence is not None and category_confidence < CONFIDENCE_THRESHOLD:
        reasons.append("low_category_confidence")

    if priority_confidence is not None and priority_confidence < CONFIDENCE_THRESHOLD:
        reasons.append("low_priority_confidence")

    if rag_guardrail_triggered or rag_accepted_count == 0:
        reasons.append("rag_no_confident_match")

    if output_corrections:
        reasons.append("output_normalized")

    if any(keyword in text for keyword in REGULATORY_KEYWORDS):
        reasons.append("regulatory_keywords")

    if any(keyword in text for keyword in HIGH_VALUE_KEYWORDS):
        reasons.append("high_value_claim")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)

    manual_review_reasons = {
        "low_category_confidence",
        "low_priority_confidence",
        "rag_no_confident_match",
        "output_normalized",
        "regulatory_keywords",
        "high_value_claim",
    }
    hitl_required = any(reason in manual_review_reasons for reason in unique_reasons)

    return {
        "escalation_reasons": unique_reasons,
        "hitl_required": hitl_required,
        "escalation_required": bool(unique_reasons) or escalation_required,
    }
