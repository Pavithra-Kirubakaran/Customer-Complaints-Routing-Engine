from pydantic import BaseModel, Field, field_validator

from typing import Any, Optional



from guardrails.pii import validate_no_blocked_pii



ALLOWED_CHANNELS = frozenset({"web", "email", "chat", "phone"})





class TicketRequest(BaseModel):

    customer_id: str = Field(..., min_length=1, max_length=128, example="cust-001")

    channel: str = Field(..., example="email")

    subject: Optional[str] = Field(None, max_length=256, example="Order not received")

    message: str = Field(

        ...,

        min_length=1,

        max_length=8000,

        example="I placed an order but I have not received any confirmation.",

    )



    @field_validator("customer_id", "subject", "message")

    @classmethod

    def reject_blocked_pii(cls, value: str | None, info) -> str | None:

        if value is None:

            return value

        validate_no_blocked_pii(value, info.field_name)

        return value.strip() if isinstance(value, str) else value



    @field_validator("channel")

    @classmethod

    def validate_channel(cls, value: str) -> str:

        normalized = value.strip().lower()

        if normalized not in ALLOWED_CHANNELS:

            allowed = ", ".join(sorted(ALLOWED_CHANNELS))

            raise ValueError(f"channel must be one of: {allowed}")

        return normalized





class RagSource(BaseModel):

    id: str

    title: str

    category: str = ""

    similarity: float

    accepted: bool





class TicketResponse(BaseModel):

    ticket_id: int

    customer_id: str

    channel: str

    subject: Optional[str]

    message: Optional[str] = None

    context: Optional[str]

    category: str

    priority: str

    queue: str

    team: str

    sla: str

    routing_note: str

    rag_context: str

    escalation_required: bool

    monitoring_note: Optional[str]

    assigned_at: str

    escalation_reasons: list[str] = Field(default_factory=list)

    hitl_required: bool = False

    rag_sources: list[RagSource] = Field(default_factory=list)

    rag_kb_version: Optional[str] = None

    category_confidence: Optional[float] = None

    priority_confidence: Optional[float] = None

    routing_explanation: Optional[str] = None

    accepted_rag_count: int = 0

    rag_guardrail_triggered: bool = False





class MonitoringSummary(BaseModel):

    total_tickets: int

    by_priority: dict[str, int]

    by_category: dict[str, int]

    by_team: dict[str, int]

    escalation_count: int

    hitl_count: int = 0

    rag_guardrail_count: int = 0

    output_correction_count: int = 0





class EscalationResponse(BaseModel):

    ticket_id: int

    customer_id: str

    category: str

    priority: str

    queue: str

    team: str

    sla: str

    routing_note: str

    escalation_required: bool

    assigned_at: str

    escalation_reasons: list[str] = Field(default_factory=list)

    hitl_required: bool = False

    routing_explanation: Optional[str] = None





class AuditEvent(BaseModel):

    audit_id: int

    event_type: str

    created_at: str

    payload: dict[str, Any] = Field(default_factory=dict)





class TicketAuditResponse(BaseModel):

    ticket_id: int

    routing_explanation: Optional[str] = None

    escalation_reasons: list[str] = Field(default_factory=list)

    hitl_required: bool = False

    audit_trace: dict[str, Any] = Field(default_factory=dict)

    events: list[AuditEvent] = Field(default_factory=list)





class GuardrailsSummary(BaseModel):

    total_tickets: int

    hitl_rate: float

    escalation_rate: float

    rag_guardrail_rate: float

    avg_category_confidence: Optional[float] = None

    avg_priority_confidence: Optional[float] = None

    top_escalation_reasons: dict[str, int] = Field(default_factory=dict)


