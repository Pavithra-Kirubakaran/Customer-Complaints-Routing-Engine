"""
rag_store.py
------------
Manages the ChromaDB persistent vector store for the complaint routing knowledge base.

On first call to `get_collection()`, documents are loaded from knowledge_base.json,
embedded using a local SentenceTransformer model (all-MiniLM-L6-v2 — no API key needed),
and stored in the persistent Chroma collection at db/chroma/.

Subsequent calls reuse the persisted collection without re-embedding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
_CHROMA_DIR = _BASE_DIR / "db" / "chroma"
_KB_PATH = _BASE_DIR / "data" / "knowledge_base.json"

# ── Constants ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "complaint_knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"   # lightweight, fast, no API key required


def _load_knowledge_base() -> list[dict]:
    """Load documents from knowledge_base.json."""
    if not _KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {_KB_PATH}")
    return json.loads(_KB_PATH.read_text(encoding="utf-8"))


def get_collection() -> chromadb.Collection:
    """
    Return (and lazily initialise) the persistent Chroma collection.

    If the collection is empty, all documents from knowledge_base.json are
    embedded and inserted. Subsequent calls simply return the existing collection.
    """
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Populate on first run (idempotent — won't re-add existing IDs)
    if collection.count() == 0:
        logger.info("Chroma collection is empty — populating from knowledge base …")
        documents = _load_knowledge_base()

        ids = [doc["id"] for doc in documents]
        texts = [f"{doc['title']}. {doc['content']}" for doc in documents]
        metadatas = [
            {"title": doc["title"], "category": doc.get("category", "")}
            for doc in documents
        ]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info("Inserted %d documents into Chroma collection.", len(documents))
    else:
        logger.debug(
            "Chroma collection '%s' already has %d documents.",
            COLLECTION_NAME,
            collection.count(),
        )

    return collection


def search_with_guardrails(
    query: str,
    n_results: int = 3,
    min_similarity: float | None = None,
) -> dict:
    """
    Perform guarded semantic search with similarity floor and source attribution.
    """
    from guardrails.rag import DEFAULT_N_RESULTS, apply_rag_guardrails

    if not query or not query.strip():
        from guardrails.rag import get_kb_version

        return {
            "context": "No query provided — knowledge base search skipped.",
            "sources": [],
            "kb_version": get_kb_version(),
            "match_count": 0,
            "accepted_count": 0,
            "min_similarity": min_similarity,
            "rag_guardrail_triggered": False,
        }

    collection = get_collection()
    limit = min(n_results or DEFAULT_N_RESULTS, collection.count())
    if limit == 0:
        from guardrails.rag import get_kb_version

        return {
            "context": "No relevant knowledge context could be retrieved.",
            "sources": [],
            "kb_version": get_kb_version(),
            "match_count": 0,
            "accepted_count": 0,
            "min_similarity": min_similarity,
            "rag_guardrail_triggered": True,
        }

    results = collection.query(
        query_texts=[query],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not docs:
        from guardrails.rag import get_kb_version

        return {
            "context": "No relevant knowledge context could be retrieved.",
            "sources": [],
            "kb_version": get_kb_version(),
            "match_count": 0,
            "accepted_count": 0,
            "min_similarity": min_similarity,
            "rag_guardrail_triggered": True,
        }

    return apply_rag_guardrails(
        docs,
        metas,
        ids,
        distances,
        min_similarity=min_similarity,
    )


def search(query: str, n_results: int = 3) -> str:
    """Backward-compatible text-only RAG search."""
    return search_with_guardrails(query, n_results=n_results)["context"]
