"""FastAPI service for source-grounded cybercrime research."""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.agent import SessionMemory, run_agent
from src.analytics import get_analytics, record_query
from src.ingest import ingest_directory
from src.rag import create_retriever

app = FastAPI(
    title="Verifiable Analyst Copilot",
    description=(
        "A source-grounded assistant for public cybercrime reports, "
        "with evidence citations, request tracing, and privacy-conscious analytics."
    ),
    version="1.0.0",
)

REPORTS_DIRECTORY = Path(os.getenv("REPORTS_DIRECTORY", "data/reports"))
EUROPOL_TITLE = (
    "Steal, Deal and Repeat: How Cybercriminals Trade "
    "and Exploit Your Data — IOCTA 2025"
)
EUROPOL_URL = (
    "https://www.europol.europa.eu/publication-events/"
    "main-reports/steal-deal-and-repeat-how-cybercriminals-"
    "trade-and-exploit-your-data"
)
RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "tfidf").strip().lower()
retriever = create_retriever(backend=RETRIEVER_BACKEND)
index_error: str | None = None
_sessions: OrderedDict[str, SessionMemory] = OrderedDict()
_sessions_lock = RLock()
_MAX_SESSIONS = 500


def load_documents() -> int:
    """Load available PDFs and rebuild the evidence index."""
    global retriever, index_error
    chunks = ingest_directory(REPORTS_DIRECTORY)
    updated = []
    for chunk in chunks:
        if chunk.source == "iocta_report.pdf":
            chunk = replace(chunk, title=EUROPOL_TITLE, url=EUROPOL_URL)
        updated.append(chunk)
    if not updated:
        raise FileNotFoundError(f"No readable PDF passages found in {REPORTS_DIRECTORY}")
    retriever = create_retriever(updated, RETRIEVER_BACKEND)
    index_error = None
    return len(updated)


def _get_memory(session_id: str) -> SessionMemory:
    with _sessions_lock:
        memory = _sessions.get(session_id)
        if memory is None:
            memory = SessionMemory()
            _sessions[session_id] = memory
        _sessions.move_to_end(session_id)
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)
        return memory


@app.on_event("startup")
def startup_index() -> None:
    """Attempt indexing at startup; keep API alive with an explicit degraded status."""
    global index_error
    try:
        load_documents()
    except Exception as exc:
        index_error = f"{type(exc).__name__}: {exc}"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    session_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=128,
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    evidence_count: int
    evidence: list[dict]
    tool: str
    trace: list[str]
    session_id: str


@app.get("/health")
def health():
    ready = bool(retriever.chunks)
    return {
        "status": "healthy" if ready else "degraded",
        "project": "Verifiable Analyst Copilot",
        "indexed_chunks": len(retriever.chunks),
        "index_ready": ready,
        "index_error": index_error,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    started = time.perf_counter()
    result_count = 0
    status = "ok"
    try:
        if not retriever.chunks:
            raise HTTPException(
                status_code=503,
                detail="The evidence index is unavailable. Add a PDF to data/reports and call /reload.",
            )
        memory = _get_memory(request.session_id)
        output = run_agent(
            request.question, retriever, top_k=request.top_k, memory=memory
        )
        result_count = len(output["evidence"])
        return AskResponse(
            question=request.question,
            answer=output["answer"],
            evidence_count=result_count,
            evidence=output["evidence"],
            tool=output["tool"],
            trace=output["trace"],
            session_id=request.session_id,
        )
    except HTTPException:
        status = "unavailable"
        raise
    except Exception as exc:
        status = "error"
        raise HTTPException(status_code=500, detail="Research request failed safely.") from exc
    finally:
        try:
            record_query(
                request.session_id,
                request.question,
                result_count,
                int((time.perf_counter() - started) * 1000),
                status,
            )
        except Exception:
            # Analytics must never make research unavailable.
            pass


@app.get("/analytics")
def analytics():
    """Aggregate query metadata; no raw questions or answers are returned."""
    try:
        return get_analytics()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Analytics storage is unavailable.") from exc


@app.post("/reload")
def reload_documents(x_admin_key: str | None = Header(default=None)):
    """Reload source PDFs; optionally protect the endpoint with RELOAD_ADMIN_KEY."""
    required_key = os.getenv("RELOAD_ADMIN_KEY")
    if not required_key:
        raise HTTPException(
            status_code=503,
            detail="Document reload is disabled until RELOAD_ADMIN_KEY is configured.",
        )
    if x_admin_key != required_key:
        raise HTTPException(status_code=403, detail="Invalid or missing admin key.")
    try:
        count = load_documents()
        return {"status": "success", "indexed_chunks": count}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document reload failed.") from exc
