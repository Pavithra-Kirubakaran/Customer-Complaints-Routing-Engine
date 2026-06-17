import joblib

import os
from pathlib import Path

# Try ONNX-based classifier first, fall back to joblib pickles
try:
    from agents.onnx_inference import ONNXSequenceClassifier
except Exception:
    ONNXSequenceClassifier = None

MODELS_DIR = Path("models")
QUEUE_ONNX_DIR = Path("models_v2") / "final_distilbert_model"
try:
    # prefer ONNX-based queue classifier; keep priority as pickle
    if ONNXSequenceClassifier and QUEUE_ONNX_DIR.exists():
        try:
            queue_model = ONNXSequenceClassifier(QUEUE_ONNX_DIR)
        except Exception:
            queue_model = joblib.load(MODELS_DIR / "queue_model.pkl")
    else:
        queue_model = joblib.load(MODELS_DIR / "queue_model.pkl")

    # priority: prefer pickle
    try:
        priority_model = joblib.load(MODELS_DIR / "priority_model.pkl")
    except Exception:
        # fallback to ONNX queue classifier (if present) — only if no pickle
        if ONNXSequenceClassifier and QUEUE_ONNX_DIR.exists():
            priority_model = ONNXSequenceClassifier(QUEUE_ONNX_DIR)
        else:
            raise
except Exception:
    # final fallback: re-raise so calling code notices failure
    raise

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