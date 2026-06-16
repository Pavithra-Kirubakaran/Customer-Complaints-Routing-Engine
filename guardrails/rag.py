"""RAG retrieval guardrails — similarity floor, source attribution, KB versioning."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

_BASE_DIR = Path(__file__).resolve().parent.parent
_KB_PATH = _BASE_DIR / "data" / "knowledge_base.json"

DEFAULT_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.45"))
DEFAULT_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "3"))


def get_kb_version() -> str:
    """Stable short hash of the knowledge base file for audit provenance."""
    if not _KB_PATH.exists():
        return "missing"
    digest = hashlib.sha256(_KB_PATH.read_bytes()).hexdigest()
    return digest[:12]


def apply_rag_guardrails(
    docs: list[str],
    metas: list[dict[str, Any]],
    ids: list[str],
    distances: list[float],
    *,
    min_similarity: float | None = None,
) -> dict[str, Any]:
    """Filter weak matches and build attributed RAG payload."""
    threshold = DEFAULT_MIN_SIMILARITY if min_similarity is None else min_similarity
    sources: list[dict[str, Any]] = []
    accepted_lines: list[str] = []

    for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
        similarity = round(1 - dist, 3)
        source = {
            "id": doc_id,
            "title": meta.get("title", "Unknown"),
            "category": meta.get("category", ""),
            "similarity": similarity,
            "accepted": similarity >= threshold,
        }
        sources.append(source)
        if source["accepted"]:
            accepted_lines.append(
                f"[{source['title']}] (similarity: {similarity}, id: {doc_id})\n{doc_text}"
            )

    if accepted_lines:
        context = "\n\n---\n\n".join(accepted_lines)
    else:
        context = (
            "No knowledge-base matches met the similarity threshold "
            f"({threshold}). Policy context withheld."
        )

    return {
        "context": context,
        "sources": sources,
        "kb_version": get_kb_version(),
        "match_count": len(sources),
        "accepted_count": sum(1 for s in sources if s["accepted"]),
        "min_similarity": threshold,
        "rag_guardrail_triggered": len(accepted_lines) == 0 and len(sources) > 0,
    }
