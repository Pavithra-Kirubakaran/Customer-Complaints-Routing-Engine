import joblib

queue_model = joblib.load("models/queue_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")

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
    "Service Outages and Maintenance": "Infrastructure Team"
}

SLA_MAPPING = {
    "high": "4 Hours",
    "medium": "24 Hours",
    "low": "72 Hours"
}

def route_ticket(ticket_text):

    queue = queue_model.predict([ticket_text])[0]

    priority = priority_model.predict([ticket_text])[0]

    team = TEAM_MAPPING.get(queue, "Default Team")

    sla = SLA_MAPPING.get(priority, "24 Hours")

    return {
        "queue": queue,
        "priority": priority,
        "team": team,
        "sla": sla
    }