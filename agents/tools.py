from pathlib import Path
import re
import joblib
from langchain_core.tools import tool

from guardrails.constants import (
    CATEGORY_SLA_OVERRIDES,
    CHANNEL_NOTES,
    SLA_MAP,
    TEAM_MAPPING,
    URGENT_CATEGORIES,
)

# Load models
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Prefer ONNX models when available (fall back to joblib pickles)
try:
    from agents.onnx_inference import ONNXSequenceClassifier
except Exception:
    ONNXSequenceClassifier = None


def _load_model(preferred_dir: Path, pkl_path: Path):
    """Try to load ONNX model from preferred_dir, otherwise load joblib from pkl_path."""
    # look for quantized ONNX first
    if ONNXSequenceClassifier and preferred_dir:
        if (preferred_dir / "model.quant.onnx").exists() or (preferred_dir / "model.onnx").exists():
            try:
                return ONNXSequenceClassifier(preferred_dir)
            except Exception:
                pass

    # fallback to joblib
    try:
        return joblib.load(pkl_path)
    except Exception:
        return None


# queue model: prefer ONNX (transformer) then fallback to old pickle
QUEUE_ONNX_DIR = Path(__file__).resolve().parent.parent / "models_v2" / "final_distilbert_model"
queue_model = _load_model(QUEUE_ONNX_DIR, MODELS_DIR / "queue_model.pkl")

# priority model: keep using the existing sklearn pickle (smaller, faster). If missing,
# try ONNX as a fallback (only used when user didn't provide a pickle).
try:
    priority_model = joblib.load(MODELS_DIR / "priority_model.pkl")
except Exception:
    priority_model = None
    if ONNXSequenceClassifier and QUEUE_ONNX_DIR:
        try:
            priority_model = ONNXSequenceClassifier(QUEUE_ONNX_DIR)
        except Exception:
            priority_model = None


def _clean_text(text: str) -> str:
    """Replicate the exact preprocessing used during model training."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_category_with_confidence(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return "General Inquiry", 0.0
    cleaned = _clean_text(text)
    probabilities = queue_model.predict_proba([cleaned])[0]
    best_index = int(probabilities.argmax())
    category = str(queue_model.classes_[best_index])
    return category, float(probabilities[best_index])


def predict_priority_with_confidence(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return "low", 0.0
    cleaned = _clean_text(text)
    probabilities = priority_model.predict_proba([cleaned])[0]
    best_index = int(probabilities.argmax())
    priority = str(priority_model.classes_[best_index]).lower()
    return priority, float(probabilities[best_index])


@tool
def classify_category(text: str) -> str:
    """Classify complaint into a category using the trained queue model."""
    category, _confidence = classify_category_with_confidence(text)
    return category


@tool
def predict_priority(text: str) -> str:
    """Predict priority level (high/medium/low) using the trained priority model."""
    priority, _confidence = predict_priority_with_confidence(text)
    return priority


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base using ChromaDB semantic vector similarity (Chroma RAG)."""
    from agents.rag_store import search_with_guardrails

    return search_with_guardrails(query, n_results=3)["context"]


@tool
def determine_sla(priority: str = None, category: str = None, payload: dict = None) -> str:
    """Determine SLA based on priority and category."""
    if payload and isinstance(payload, dict):
        priority = priority or payload.get("priority")
        category = category or payload.get("category")

    priority = (priority or "low").lower()
    category = category or ""

    if category in CATEGORY_SLA_OVERRIDES:
        return CATEGORY_SLA_OVERRIDES[category]
    return SLA_MAP.get(priority.lower(), "72 Hours")


@tool
def route_ticket(category: str = None, priority: str = None, channel: str = None, payload: dict = None) -> dict:
    """Determine the routing destination (queue, team, note) for the ticket."""
    if payload and isinstance(payload, dict):
        category = category or payload.get("category")
        priority = priority or payload.get("priority")
        channel = channel or payload.get("channel")

    category = category or "General Inquiry"
    priority = (priority or "low").lower()
    channel = (channel or "email").lower()

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

    if priority.lower() in {"critical", "high"}:
        return True
    return category in URGENT_CATEGORIES


TOOLS = [
    classify_category,
    predict_priority,
    search_knowledge_base,
    determine_sla,
    route_ticket,
    check_escalation,
]
