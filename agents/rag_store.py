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


def search(query: str, n_results: int = 3) -> str:
    """
    Perform a semantic similarity search over the knowledge base.

    Parameters
    ----------
    query     : Natural-language query string (usually the complaint context).
    n_results : Number of top documents to return (default 3).

    Returns
    -------
    A formatted string with the top-k matching documents, ready for injection
    into the routing agent's prompt.
    """
    if not query or not query.strip():
        return "No query provided — knowledge base search skipped."

    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "No relevant knowledge context could be retrieved."

    lines: list[str] = []
    for doc_text, meta, dist in zip(docs, metas, distances):
        similarity = round(1 - dist, 3)   # cosine distance → cosine similarity
        title = meta.get("title", "Unknown")
        lines.append(f"[{title}] (similarity: {similarity})\n{doc_text}")

    return "\n\n---\n\n".join(lines)
