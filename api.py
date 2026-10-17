"""FastAPI backend for the Verifiable Analyst Copilot."""

from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.answer import generate_answer
from src.ingest import ingest_directory
from src.rag import EvidenceRetriever


app = FastAPI(
    title="Verifiable Analyst Copilot",
    description=(
        "A source-grounded assistant for public cybercrime "
        "reports and statistics."
    ),
    version="0.2.0",
)

REPORTS_DIRECTORY = Path("data/reports")

EUROPOL_TITLE = (
    "Steal, Deal and Repeat: How Cybercriminals Trade "
    "and Exploit Your Data — IOCTA 2025"
)

EUROPOL_URL = (
    "https://www.europol.europa.eu/publication-events/"
    "main-reports/steal-deal-and-repeat-how-cybercriminals-"
    "trade-and-exploit-your-data"
)

retriever = EvidenceRetriever()


def load_documents() -> int:
    """Load PDFs and attach known source metadata."""
    global retriever

    chunks = ingest_directory(REPORTS_DIRECTORY)
    updated_chunks = []

    for chunk in chunks:
        if chunk.source == "iocta_report.pdf":
            chunk = replace(
                chunk,
                title=EUROPOL_TITLE,
                url=EUROPOL_URL,
            )

        updated_chunks.append(chunk)

    retriever = EvidenceRetriever(updated_chunks)
    return len(updated_chunks)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer: str
    evidence_count: int
    evidence: list[dict]


@app.get("/health")
def health():
    """Report API status and indexed passage count."""
    return {
        "status": "healthy",
        "project": "Verifiable Analyst Copilot",
        "indexed_chunks": len(retriever.chunks),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Retrieve evidence and generate an extractive answer."""
    results = retriever.search(
        request.question,
        top_k=request.top_k,
    )

    evidence = [
        {
            "title": result.chunk.title,
            "source": result.chunk.source,
            "page": result.chunk.page,
            "url": result.chunk.url,
            "score": round(result.score, 4),
            "text": result.chunk.text,
        }
        for result in results
    ]

    answer = generate_answer(request.question, results)

    return AskResponse(
        question=request.question,
        answer=answer,
        evidence_count=len(evidence),
        evidence=evidence,
    )


@app.post("/reload")
def reload_documents():
    """Re-index PDFs currently present in the reports directory."""
    try:
        count = load_documents()
        return {
            "status": "success",
            "indexed_chunks": count,
        }
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
