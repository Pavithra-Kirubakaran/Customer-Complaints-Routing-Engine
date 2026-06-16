from typing import Any, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """Per-field state schema so parallel nodes can update different keys in one step."""

    customer_id: str
    channel: str
    subject: Optional[str]
    message: str
    category: Optional[str]
    priority: Optional[str]
    queue: Optional[str]
    team: Optional[str]
    sla: Optional[str]
    rag_context: Optional[str]
    routing_note: Optional[str]
    escalation_required: Optional[bool]
    assigned_at: str
    messages: List[Any]
    context: Optional[str]
    monitoring_note: Optional[str]
    category_confidence: Optional[float]
    priority_confidence: Optional[float]
    rag_sources: Optional[list]
    rag_kb_version: Optional[str]
    rag_guardrail_triggered: Optional[bool]
    accepted_rag_count: Optional[int]
