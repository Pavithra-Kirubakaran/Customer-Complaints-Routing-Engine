from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from database import (

    get_all_tickets,

    get_guardrails_summary,

    get_monitoring_summary,

    get_pending_escalations,

    get_ticket_audit,

    get_ticket_by_id,

    init_db,

)

from schemas import (

    EscalationResponse,

    GuardrailsSummary,

    MonitoringSummary,

    TicketAuditResponse,

    TicketRequest,

    TicketResponse,

)

from agents.supervisor_agent import SupervisorAgent

from guardrails.pii import sanitize_ticket_record



app = FastAPI(

    title="Complaint Routing Engine",

    description="Agentic complaint classification, prioritization, and SLA-aware routing with Chroma RAG.",

    version="1.0.0",

)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



supervisor = SupervisorAgent()

init_db()





@app.post("/tickets", response_model=TicketResponse)

async def create_ticket(ticket: TicketRequest):

    try:

        ticket_result = supervisor.process_ticket(ticket)

        return ticket_result

    except Exception as exc:

        raise HTTPException(status_code=500, detail=str(exc))





@app.get("/tickets", response_model=list[TicketResponse])

async def list_tickets(limit: int = 500):

    return [sanitize_ticket_record(ticket) for ticket in get_all_tickets(limit=limit)]





@app.get("/tickets/{ticket_id}", response_model=TicketResponse)

async def read_ticket(ticket_id: int):

    ticket = get_ticket_by_id(ticket_id)

    if ticket is None:

        raise HTTPException(status_code=404, detail="Ticket not found")

    return sanitize_ticket_record(ticket)





@app.get("/tickets/{ticket_id}/audit", response_model=TicketAuditResponse)

async def read_ticket_audit(ticket_id: int):

    audit = get_ticket_audit(ticket_id)

    if audit is None:

        raise HTTPException(status_code=404, detail="Ticket not found")

    return audit





@app.get("/monitoring/summary", response_model=MonitoringSummary)

async def monitoring_summary():

    return get_monitoring_summary()





@app.get("/monitoring/escalations", response_model=list[EscalationResponse])

async def escalations():

    return get_pending_escalations()





@app.get("/monitoring/guardrails", response_model=GuardrailsSummary)

async def guardrails_summary():

    return get_guardrails_summary()


