"""Shared routing constants used by tools and output guardrails."""

TEAM_MAPPING = {
    "Technical Support": "Tech Team",
    "IT Support": "IT Team",
    "Billing and Payments": "Finance Team",
    "Customer Service": "Customer Success Team",
    "Returns and Exchanges": "Operations Team",
    "Product Support": "Product Team",
    "Sales and Pre-Sales": "Sales Team",
    "General Inquiry": "Customer Success Team",
    "Human Resources": "HR Team",
    "Service Outages and Maintenance": "Infrastructure Team",
}

MANUAL_REVIEW_TEAM = "Manual Review Team"
MANUAL_REVIEW_QUEUE = "Manual Review"

ALLOWED_TEAMS = frozenset(set(TEAM_MAPPING.values()) | {MANUAL_REVIEW_TEAM})

ALLOWED_CATEGORIES = frozenset(TEAM_MAPPING.keys()) | {MANUAL_REVIEW_QUEUE}

ALLOWED_PRIORITIES = frozenset(
    {"low", "medium", "high", "critical", "p1", "p2", "p3", "p4"}
)

ALLOWED_SLAS = frozenset({"1 Hour", "4 Hours", "24 Hours", "72 Hours"})

SLA_MAP = {
    "critical": "1 Hour",
    "high": "4 Hours",
    "medium": "24 Hours",
    "low": "72 Hours",
}

CATEGORY_SLA_OVERRIDES = {
    "Service Outages and Maintenance": "1 Hour",
    "Technical Support": "4 Hours",
    "IT Support": "4 Hours",
}

URGENT_CATEGORIES = frozenset(
    {"Service Outages and Maintenance", "Technical Support", "IT Support"}
)

CHANNEL_NOTES = {
    "email": "Track via email ticketing with SLA alerts.",
    "chat": "Route to live chat support with priority monitoring.",
    "web": "Capture in portal and escalate if critical.",
    "phone": "Route to phone support with supervisor monitoring.",
}
