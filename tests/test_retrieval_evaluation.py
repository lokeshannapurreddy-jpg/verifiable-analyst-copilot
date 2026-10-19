from pathlib import Path

import builtins

import pytest

from src.ingest import ingest_pdf
from src.evaluation import QUESTIONS
from src.rag import (
    EvidenceRetriever,
    EmbeddingEvidenceRetriever,
    HybridEvidenceRetriever,
)

PDF_PATH = Path("data/reports/iocta_report.pdf")

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


def test_tfidf_does_not_require_optional_ml_dependency(monkeypatch):
    retriever = EvidenceRetriever()
    real_import = builtins.__import__

    def reject_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("optional dependency unavailable", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_sentence_transformers)
    assert retriever.search("cybercrime") == []
    with pytest.raises(RuntimeError, match="requirements-ml.txt"):
        EmbeddingEvidenceRetriever()


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
