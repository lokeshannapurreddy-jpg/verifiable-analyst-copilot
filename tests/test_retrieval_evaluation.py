from pathlib import Path

import pytest

from src.ingest import ingest_pdf
from src.rag import (
    EvidenceRetriever,
    EmbeddingEvidenceRetriever,
    HybridEvidenceRetriever,
)

PDF_PATH = Path("data/reports/iocta_report.pdf")

QUESTIONS = [
    {
        "question": "How do criminals steal and trade personal data?",
        "expected_pages": {8, 9, 12},
    },
    {
        "question": "How do cybercriminals use stolen credentials?",
        "expected_pages": {12, 13},
    },
    {
        "question": "What role do infostealers play in cybercrime?",
        "expected_pages": {13},
    },
    {
        "question": "How do cybercriminals distribute malware?",
        "expected_pages": {13, 14, 17},
    },
    {
        "question": "How do criminals monetize stolen data?",
        "expected_pages": {8, 9, 17},
    },
]


@pytest.fixture(scope="module")
def chunks():
    if not PDF_PATH.exists():
        pytest.skip(f"Evaluation report not found: {PDF_PATH}")
    return ingest_pdf(PDF_PATH, title="Europol IOCTA 2025")


def recall_at_3(retriever, item):
    results = retriever.search(item["question"], top_k=3)
    pages = {result.chunk.page for result in results}
    expected = item["expected_pages"]
    return len(pages & expected) / len(expected)


def test_tfidf_retrieval_evaluation(chunks):
    retriever = EvidenceRetriever(chunks)
    scores = [recall_at_3(retriever, item) for item in QUESTIONS]
    mean = sum(scores) / len(scores)
    print(f"\nTF-IDF mean recall@3: {mean:.3f}")
    assert all(0 <= score <= 1 for score in scores)


def test_semantic_retrieval_evaluation(chunks):
    retriever = EmbeddingEvidenceRetriever(chunks)
    scores = [recall_at_3(retriever, item) for item in QUESTIONS]
    mean = sum(scores) / len(scores)
    print(f"\nSemantic mean recall@3: {mean:.3f}")
    assert all(0 <= score <= 1 for score in scores)


def test_hybrid_retrieval_evaluation(chunks):
    retriever = HybridEvidenceRetriever(chunks)
    scores = [recall_at_3(retriever, item) for item in QUESTIONS]
    mean = sum(scores) / len(scores)
    print(f"\nHybrid mean recall@3: {mean:.3f}")
    assert all(0 <= score <= 1 for score in scores)
