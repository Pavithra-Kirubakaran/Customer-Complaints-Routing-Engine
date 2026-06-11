from pathlib import Path
import re
import joblib
from langchain_core.tools import tool

# Load models
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
queue_model = joblib.load(MODELS_DIR / "queue_model.pkl")
priority_model = joblib.load(MODELS_DIR / "priority_model.pkl")


def _clean_text(text: str) -> str:
    """Replicate the exact preprocessing used during model training.

    Training notebook step (cells 15-16):
        text = str(text).lower()
        text = re.sub(r"http\\S+", "", text)      # strip URLs
        text = re.sub(r"[^a-zA-Z ]", " ", text)  # keep only letters + spaces
        text = re.sub(r"\\s+", " ", text)          # collapse whitespace
        return text.strip()

    The TF-IDF vectorizer was fitted on this cleaned form, so feeding
    raw text (with digits, punctuation, mixed case) will produce
    out-of-distribution TF-IDF features and degrade accuracy.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@tool
def classify_category(text: str) -> str:
    """Classify complaint into a category using the trained queue model."""
    if not text or not text.strip():
        return "General Inquiry"
    return str(queue_model.predict([_clean_text(text)])[0])


@tool
def predict_priority(text: str) -> str:
    """Predict priority level (high/medium/low) using the trained priority model."""
    if not text or not text.strip():
        return "low"
    prediction = priority_model.predict([_clean_text(text)])[0]
    return str(prediction).lower()



@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base using ChromaDB semantic vector similarity (Chroma RAG)."""
    from agents.rag_store import search as chroma_search
    return chroma_search(query, n_results=3)


@tool
def determine_sla(priority: str = None, category: str = None, payload: dict = None) -> str:
    """Determine SLA based on priority and category.

    This tool accepts either two positional args (`priority`, `category`) or a
    single `payload` dict produced by some StructuredTool runtimes. The code
    normalizes inputs to support both invocation styles.
    """
    # Normalize input from payload if provided
    if payload and isinstance(payload, dict):
        # payload may contain keys like 'priority' and 'category'
        priority = priority or payload.get("priority")
        category = category or payload.get("category")

    # Provide safe defaults
    priority = (priority or "low").lower()
    category = category or ""

    SLA_MAP = {
        "critical": "1 Hour",
        "high": "4 Hours",
        "medium": "24 Hours",
        "low": "72 Hours",
    }
    CATEGORY_OVERRIDES = {
        "Service Outages and Maintenance": "1 Hour",
        "Technical Support": "4 Hours",
        "IT Support": "4 Hours",
    }

    if category in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[category]
    return SLA_MAP.get(priority.lower(), "72 Hours")


@tool
def route_ticket(category: str = None, priority: str = None, channel: str = None, payload: dict = None) -> dict:
    """Determine the routing destination (queue, team, note) for the ticket.

    This tool accepts either positional params or a single payload dict from
    StructuredTool-style invocation.
    """
    if payload and isinstance(payload, dict):
        category = category or payload.get("category")
        priority = priority or payload.get("priority")
        channel = channel or payload.get("channel")

    category = category or "General Inquiry"
    priority = (priority or "low").lower()
    channel = (channel or "email").lower()

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
    
    CHANNEL_NOTES = {
        "email": "Track via email ticketing with SLA alerts.",
        "chat": "Route to live chat support with priority monitoring.",
        "web": "Capture in portal and escalate if critical.",
    }
    
    queue = category
    team = TEAM_MAPPING.get(queue, "Customer Success Team")
    channel_note = CHANNEL_NOTES.get(channel, "Route via standard pipeline.")
    
    routing_note = f"Route {queue} to {team}. {channel_note}"
    return {
        "queue": queue,
        "team": team,
        "routing_note": routing_note,
    }


@tool
def check_escalation(priority: str = None, category: str = None, payload: dict = None) -> bool:
    """Determine if the ticket needs escalation."""
    if payload and isinstance(payload, dict):
        priority = priority or payload.get("priority")
        category = category or payload.get("category")

    priority = (priority or "low").lower()
    category = category or "General Inquiry"

    urgent_categories = {
        "Service Outages and Maintenance",
        "Technical Support",
        "IT Support",
    }
    
    if priority.lower() in {"critical", "high"}:
        return True
    return category in urgent_categories


# Collect all tools
TOOLS = [
    classify_category,
    predict_priority,
    search_knowledge_base,
    determine_sla,
    route_ticket,
    check_escalation,
]
