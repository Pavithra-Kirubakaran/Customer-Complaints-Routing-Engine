from datetime import datetime

from schemas import TicketRequest
from database import save_ticket, save_audit_event
from agents.langraph_agents import routing_graph
from agents.state import AgentState
from guardrails.audit import AuditCollector
from guardrails.hitl import evaluate_hitl
from guardrails.output import (
    apply_manual_review_route,
    build_routing_explanation,
    validate_and_normalize_output,
)
from guardrails.pii import redact_pii, sanitize_ticket_record


class SupervisorAgent:
    def __init__(self):
        pass

    def process_ticket(self, ticket: TicketRequest) -> dict:
        """Process complaint ticket through LangGraph agentic workflow."""
        safe_customer_id = redact_pii(ticket.customer_id) or ticket.customer_id
        safe_subject = redact_pii(ticket.subject)
        safe_message = redact_pii(ticket.message) or ticket.message

        state: AgentState = {
            "customer_id": safe_customer_id,
            "channel": ticket.channel,
            "subject": safe_subject,
            "message": safe_message,
            "category": None,
            "priority": None,
            "queue": None,
            "team": None,
            "sla": None,
            "rag_context": None,
            "routing_note": None,
            "escalation_required": None,
            "assigned_at": datetime.utcnow().isoformat() + "Z",
            "messages": [],
        }

        audit = AuditCollector()
        result: AgentState = dict(state)
        for event in routing_graph.stream(state):
            for node_name, node_output in event.items():
                audit.record_step(node_name, node_output)
                result.update(node_output)

        ticket_record = sanitize_ticket_record({
            "customer_id": safe_customer_id,
            "channel": ticket.channel,
            "subject": safe_subject,
            "message": safe_message,
            "context": result.get("context"),
            "category": result.get("category", "General Inquiry"),
            "priority": result.get("priority", "low"),
            "queue": result.get("queue", "General Inquiry"),
            "team": result.get("team", "Customer Success Team"),
            "sla": result.get("sla", "72 Hours"),
            "routing_note": result.get("routing_note", "Ticket routed through agentic workflow"),
            "rag_context": result.get("rag_context", "No context retrieved"),
            "escalation_required": result.get("escalation_required", False),
            "monitoring_note": result.get("monitoring_note"),
            "assigned_at": result.get("assigned_at", datetime.utcnow().isoformat() + "Z"),
            "category_confidence": result.get("category_confidence"),
            "priority_confidence": result.get("priority_confidence"),
            "rag_sources": result.get("rag_sources") or [],
            "rag_kb_version": result.get("rag_kb_version"),
            "accepted_rag_count": result.get("accepted_rag_count", 0),
            "rag_guardrail_triggered": bool(result.get("rag_guardrail_triggered")),
        })

        ticket_record, output_corrections = validate_and_normalize_output(ticket_record)

        hitl = evaluate_hitl(
            message=safe_message,
            category=ticket_record["category"],
            priority=ticket_record["priority"],
            category_confidence=ticket_record.get("category_confidence"),
            priority_confidence=ticket_record.get("priority_confidence"),
            escalation_required=bool(ticket_record.get("escalation_required")),
            rag_guardrail_triggered=bool(ticket_record.get("rag_guardrail_triggered")),
            rag_accepted_count=int(ticket_record.get("accepted_rag_count") or 0),
            output_corrections=output_corrections,
        )
        ticket_record["escalation_reasons"] = hitl["escalation_reasons"]
        ticket_record["hitl_required"] = hitl["hitl_required"]
        ticket_record["escalation_required"] = hitl["escalation_required"]

        if hitl["hitl_required"]:
            ticket_record = apply_manual_review_route(ticket_record)

        audit_summary = {
            "output_corrections": output_corrections,
            "escalation_reasons": hitl["escalation_reasons"],
            "hitl_required": hitl["hitl_required"],
        }
        ticket_record["audit_trace"] = audit.finalize(audit_summary)
        ticket_record["routing_explanation"] = build_routing_explanation(ticket_record)

        ticket_id = save_ticket(ticket_record)
        ticket_record["ticket_id"] = ticket_id

        save_audit_event(
            ticket_id=ticket_id,
            event_type="ticket_routed",
            payload={
                "routing_explanation": ticket_record["routing_explanation"],
                "escalation_reasons": ticket_record["escalation_reasons"],
                "hitl_required": ticket_record["hitl_required"],
                "output_corrections": output_corrections,
                "audit_trace": ticket_record["audit_trace"],
            },
        )

        return ticket_record
