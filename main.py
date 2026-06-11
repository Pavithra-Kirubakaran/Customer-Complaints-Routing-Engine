from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, save_ticket, get_ticket_by_id, get_monitoring_summary, get_pending_escalations
from schemas import TicketRequest, TicketResponse, MonitoringSummary, EscalationResponse
from agents.supervisor_agent import SupervisorAgent

app = FastAPI(
    title="Complaint Routing Engine",
    description="Agentic complaint classification, prioritization, and SLA-aware routing with Chroma RAG.",
    version="1.0.0"
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

@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
async def read_ticket(ticket_id: int):
    ticket = get_ticket_by_id(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

@app.get("/monitoring/summary", response_model=MonitoringSummary)
async def monitoring_summary():
    return get_monitoring_summary()

@app.get("/monitoring/escalations", response_model=list[EscalationResponse])
async def escalations():
    return get_pending_escalations()
