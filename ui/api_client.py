"""Backend integration for the Streamlit UI — direct DB/agents or FastAPI."""

from __future__ import annotations

import os
from typing import Any

import httpx

USE_API = os.getenv("USE_API", "false").lower() in ("1", "true", "yes")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def _use_api() -> bool:
    return os.getenv("USE_API", "false").lower() in ("1", "true", "yes")


def _api_url() -> str:
    return os.getenv("API_URL", "http://localhost:8000").rstrip("/")


class BackendError(Exception):
    """Raised when a backend operation fails."""


def _api_get(path: str) -> Any:
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(f"{_api_url()}{path}")
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as exc:
        raise BackendError(
            f"Cannot reach API at {_api_url()}. Start the backend with: uvicorn main:app --reload"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise BackendError(f"API error {exc.response.status_code}: {exc.response.text}") from exc


def _api_post(path: str, payload: dict) -> Any:
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{_api_url()}{path}", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as exc:
        raise BackendError(
            f"Cannot reach API at {_api_url()}. Start the backend with: uvicorn main:app --reload"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise BackendError(f"API error {exc.response.status_code}: {exc.response.text}") from exc


def _direct_supervisor():
    from agents.supervisor_agent import SupervisorAgent

    return SupervisorAgent()


def ensure_db():
    if not _use_api():
        from database import init_db

        init_db()


def fetch_tickets(limit: int = 500) -> list[dict]:
    if _use_api():
        return _api_get(f"/tickets?limit={limit}")
    from database import get_all_tickets

    return get_all_tickets(limit=limit)


def fetch_ticket(ticket_id: int) -> dict | None:
    if _use_api():
        return _api_get(f"/tickets/{ticket_id}")
    from database import get_ticket_by_id

    return get_ticket_by_id(ticket_id)


def fetch_summary() -> dict:
    if _use_api():
        return _api_get("/monitoring/summary")
    from database import get_monitoring_summary

    return get_monitoring_summary()


def fetch_escalations() -> list[dict]:
    if _use_api():
        return _api_get("/monitoring/escalations")
    from database import get_pending_escalations

    return get_pending_escalations()


def fetch_guardrails_summary() -> dict:
    if _use_api():
        return _api_get("/monitoring/guardrails")
    from database import get_guardrails_summary

    return get_guardrails_summary()


def fetch_ticket_audit(ticket_id: int) -> dict | None:
    if _use_api():
        return _api_get(f"/tickets/{ticket_id}/audit")
    from database import get_ticket_audit

    return get_ticket_audit(ticket_id)


def submit_ticket(payload: dict, supervisor=None) -> dict:
    if _use_api():
        return _api_post("/tickets", payload)

    from schemas import TicketRequest

    agent = supervisor or _direct_supervisor()
    request = TicketRequest(**payload)
    return agent.process_ticket(request)
