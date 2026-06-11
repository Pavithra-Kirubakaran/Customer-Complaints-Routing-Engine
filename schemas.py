from pydantic import BaseModel, Field
from typing import Optional

class TicketRequest(BaseModel):
    customer_id: str = Field(..., example="cust-001")
    channel: str = Field(..., example="email")
    subject: Optional[str] = Field(None, example="Order not received")
    message: str = Field(..., example="I placed an order but I have not received any confirmation.")

class TicketResponse(BaseModel):
    ticket_id: int
    customer_id: str
    channel: str
    subject: Optional[str]
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

class MonitoringSummary(BaseModel):
    total_tickets: int
    by_priority: dict[str, int]
    by_category: dict[str, int]
    by_team: dict[str, int]
    escalation_count: int

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
