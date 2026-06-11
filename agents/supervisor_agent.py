from datetime import datetime
from pathlib import Path

from schemas import TicketRequest
from database import save_ticket
from agents.langraph_agents import routing_graph
from agents.state import AgentState


class SupervisorAgent:
    def __init__(self):
        pass

    def process_ticket(self, ticket: TicketRequest) -> dict:
        """Process complaint ticket through LangGraph agentic workflow."""
        
        # Prepare initial state
        state: AgentState = {
            "customer_id": ticket.customer_id,
            "channel": ticket.channel,
            "subject": ticket.subject,
            "message": ticket.message,
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
        
        # Run through the LangGraph workflow
        result = routing_graph.invoke(state)
        
        # Build ticket record from agents' output
        ticket_record = {
            "customer_id": ticket.customer_id,
            "channel": ticket.channel,
            "subject": ticket.subject,
            "message": ticket.message,
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
        }
        
        # Save to database
        ticket_id = save_ticket(ticket_record)
        ticket_record["ticket_id"] = ticket_id
        
        return ticket_record
