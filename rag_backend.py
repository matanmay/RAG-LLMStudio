"""
rag_backend.py
--------------
RAG (Retrieval-Augmented Generation) backend using:
- LM Studio (OpenAI-compatible API) for embeddings + chat
- ChromaDB for persistent vector storage
- pypdf for PDF text extraction
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from openai import OpenAI
from pypdf import PdfReader

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
LM_STUDIO_API_KEY: str = "lm-studio"

EMBEDDING_MODEL: str = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL: str = "google/gemma-4-e2b"
EMBEDDING_DIM: int = 768          # must match the embedding model output size

CHUNK_SIZE: int = 600
CHUNK_OVERLAP: int = 200
TOP_K: int = 3

DB_PATH: str = "./chroma_db"
COLLECTION_NAME: str = "pdf_knowledge_base"
HISTORY_COLLECTION_NAME: str = "chat_history"

PDF_FILES: list[str] = [
    "A.pdf",
    "B.pdf",
]

# ──────────────────────────────────────────────────────────────
# LM Studio client (module-level singleton)
# ──────────────────────────────────────────────────────────────

_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

# ──────────────────────────────────────────────────────────────
# PDF helpers
# ──────────────────────────────────────────────────────────────

def load_and_chunk_pdf(pdf_path: str | Path) -> list[str]:
    """Extract text from a PDF and split it into overlapping chunks."""
    reader = PdfReader(str(pdf_path))
    full_text = "\n".join(
        page.extract_text() or "" for page in reader.pages
    )

    chunks: list[str] = []
    start = 0
    while start < len(full_text):
        chunk = full_text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    log.info("Chunked '%s' → %d chunks", pdf_path, len(chunks))
    return chunks

# ──────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for a single text string."""
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text.replace("\n", " ")],
    )
    return response.data[0].embedding

# ──────────────────────────────────────────────────────────────
# ChromaDB — initialisation
# ──────────────────────────────────────────────────────────────

_chroma = chromadb.PersistentClient(path=DB_PATH)
collection = _chroma.get_or_create_collection(name=COLLECTION_NAME)
_history = _chroma.get_or_create_collection(name=HISTORY_COLLECTION_NAME)


def _populate_db() -> None:
    """Index all PDF_FILES into ChromaDB. Skipped if DB already exists."""
    all_chunks: list[str] = []
    all_meta: list[dict] = []
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []

    for pdf_file in PDF_FILES:
        path = Path(pdf_file)
        if not path.exists():
            log.warning("PDF not found, skipping: %s", pdf_file)
            continue

        log.info("Processing %s …", pdf_file)
        for i, chunk in enumerate(load_and_chunk_pdf(path)):
            all_chunks.append(chunk)
            all_meta.append({"source": path.name})
            all_ids.append(f"{path.stem}_{i}_{uuid.uuid4().hex[:8]}")
            all_embeddings.append(get_embedding(chunk))

    if all_chunks:
        collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_meta,
            ids=all_ids,
        )
        log.info("Saved %d chunks to ChromaDB.", len(all_chunks))


# Populate only on first run (empty collection = new DB)
if collection.count() == 0:
    log.info("Empty collection — indexing PDFs …")
    _populate_db()
else:
    log.info("Collection ready — %d chunks.", collection.count())

# ──────────────────────────────────────────────────────────────
# Chat history persistence
# ──────────────────────────────────────────────────────────────

def save_chat_to_db(session_id: str, question: str, answer: str) -> None:
    """Persist a single Q&A turn to ChromaDB."""
    _history.add(
        ids=[str(uuid.uuid4())],
        documents=[f"Q: {question}\nA: {answer}"],
        embeddings=[[0.0] * EMBEDDING_DIM],   # zero-vec; no similarity search needed
        metadatas=[{
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        }],
    )


def load_chat_history_from_db(session_id: str) -> list[dict[str, str]]:
    """Return all saved turns for a session as a list of {role, content} dicts, oldest first."""
    results = _history.get(
        where={"session_id": session_id},
        include=["metadatas"]
    )
    if not results["ids"]:
        return []

    turns = sorted(results["metadatas"], key=lambda m: m["timestamp"])
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    return messages


def get_all_sessions() -> list[dict[str, str]]:
    """Return a list of all unique sessions with their title and last_updated timestamp."""
    results = _history.get(include=["metadatas"])
    if not results["ids"]:
        return []
    
    sessions = {}
    for m in results["metadatas"]:
        sid = m.get("session_id", "legacy_session")
        ts = m["timestamp"]
        # Use first question as title
        if sid not in sessions or ts < sessions[sid]["created_at"]:
            title = m["question"]
            if len(title) > 40:
                title = title[:37] + "..."
            
            sessions[sid] = {
                "session_id": sid,
                "title": title,
                "created_at": ts,
                "last_updated": ts
            }
        else:
            if ts > sessions[sid]["last_updated"]:
                sessions[sid]["last_updated"] = ts
                
    # Return sorted by last_updated descending (newest first)
    return sorted(list(sessions.values()), key=lambda x: x["last_updated"], reverse=True)


def clear_chat_history_in_db(session_id: Optional[str] = None) -> None:
    """Delete every record from the chat_history collection, optionally filtered by session_id."""
    if session_id:
        ids = _history.get(where={"session_id": session_id}, include=[])["ids"]
    else:
        ids = _history.get(include=[])["ids"]
        
    if ids:
        _history.delete(ids=ids)
        log.info("Cleared %d chat history records.", len(ids))


def migrate_legacy_chats() -> None:
    """Assigns 'legacy_session' to any chat records that don't have a session_id."""
    results = _history.get(include=["metadatas"])
    if not results["ids"]: return
    
    legacy_ids = []
    legacy_metadatas = []
    
    for doc_id, m in zip(results["ids"], results["metadatas"]):
        if "session_id" not in m:
            m["session_id"] = "legacy_session"
            legacy_ids.append(doc_id)
            legacy_metadatas.append(m)
            
    if legacy_ids:
        _history.update(ids=legacy_ids, metadatas=legacy_metadatas)
        log.info("Migrated %d legacy chat records.", len(legacy_ids))

# Run migration on load
migrate_legacy_chats()

# ──────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────

def retrieve_from_db(query: str, top_k: int = TOP_K) -> list[dict[str, str]]:
    """Embed the query and return the top-k most relevant chunks."""
    results = collection.query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k,
    )
    return [
        {
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
        }
        for i in range(len(results["documents"][0]))
    ]

# ──────────────────────────────────────────────────────────────
# RAG — answer generation
# ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on "
    "the provided document context. "
    "Always cite the source file when referencing information. "
    "If the answer is not in the context, reply with: \"I don't know.\""
)


def ask_rag(question: str) -> str:
    """Retrieve relevant context and generate an answer with the LLM."""
    docs = retrieve_from_db(question)

    context = "\n---\n".join(
        f"[Source: {doc['source']}]\n{doc['text']}" for doc in docs
    )

    user_message = f"""
    Answer the question using only the provided context. If the answer cannot be found in the context, say "I don't know".
    Always mention which source file the information comes from if available in the context.

    Context:
    {context}

    Question:
    {question}
    """

    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ],
        temperature=0,
    )
    return response.choices[0].message.content