from pathlib import Path

import builtins
import sys
from types import SimpleNamespace

import numpy as np

import pytest

from src.ingest import ingest_pdf
from src.evaluation import QUESTIONS
from src.rag import (
    DocumentChunk,
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


def test_semantic_retrieval_ranks_by_embedding_similarity(monkeypatch):
    vectors = {
        "alpha evidence with enough words to form a useful report passage": [0.8, 0.6],
        "beta evidence with enough words to form a useful report passage": [0.0, 1.0],
        "semantic query": [0.0, 1.0],
    }

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            pass

        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([vectors[text] for text in texts])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    chunks = [
        DocumentChunk(text=text, source="synthetic.pdf", page=page, title="Fixture")
        for page, text in enumerate(list(vectors)[:2], start=1)
    ]

    results = EmbeddingEvidenceRetriever(chunks).search("semantic query", top_k=2)

    assert [result.chunk.page for result in results] == [2, 1]
    assert results[0].score == pytest.approx(1.0)


def test_hybrid_retrieval_combines_lexical_and_semantic_rankings(monkeypatch):
    lexical_text = (
        "Stolen credentials trade through criminal marketplaces, where "
        "buyers purchase access and criminals exchange compromised account data."
    )
    semantic_text = (
        "Criminal groups acquire account passwords and resell stolen information "
        "to buyers who exploit access for additional fraud and other offences."
    )
    vectors = {
        lexical_text: [1.0, 0.0],
        semantic_text: [0.0, 1.0],
        "stolen credential marketplace": [0.0, 1.0],
    }

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            pass

        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([vectors[text] for text in texts])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    chunks = [
        DocumentChunk(text=lexical_text, source="synthetic.pdf", page=1, title="Fixture"),
        DocumentChunk(text=semantic_text, source="synthetic.pdf", page=2, title="Fixture"),
    ]

    results = HybridEvidenceRetriever(chunks).search(
        "stolen credential marketplace", top_k=2
    )

    assert {result.chunk.page for result in results} == {1, 2}
    assert results[0].chunk.page == 1


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
