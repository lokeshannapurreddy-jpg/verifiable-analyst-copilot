from pathlib import Path

import builtins
import sys
from types import SimpleNamespace

import numpy as np

import pytest

from src.ingest import ingest_pdf
from src.evaluation import QUESTIONS, ranking_metrics
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
    pages = [result.chunk.page for result in results]
    return ranking_metrics(pages, item["expected_pages"])["Recall@3"]


def test_ranking_metrics_match_hand_checkable_example():
    metrics = ranking_metrics([17, 9, 13], {9, 13, 23}, k=3)
    dcg = 1 / np.log2(3) + 1 / np.log2(4)
    ideal_dcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)

    assert metrics["Recall@3"] == pytest.approx(2 / 3)
    assert metrics["MRR@3"] == pytest.approx(1 / 2)
    assert metrics["nDCG@3"] == pytest.approx(dcg / ideal_dcg)


def test_ranking_metrics_deduplicate_pages_at_first_rank():
    metrics = ranking_metrics([17, 17, 9], {9}, k=3)

    assert metrics["Recall@3"] == pytest.approx(1.0)
    assert metrics["MRR@3"] == pytest.approx(1 / 2)
    assert metrics["nDCG@3"] == pytest.approx(1 / np.log2(3))


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_ranking_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        ranking_metrics([1], {1}, k=k)


def test_ranking_metrics_reject_empty_relevance_set():
    with pytest.raises(ValueError, match="relevant_pages must not be empty"):
        ranking_metrics([1], set())


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


def test_semantic_model_load_failure_is_actionable(monkeypatch):
    class BrokenSentenceTransformer:
        def __init__(self, model_name):
            raise OSError("C:/private/model-cache/unavailable")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=BrokenSentenceTransformer),
    )

    with pytest.raises(RuntimeError, match="Check model availability") as error:
        EmbeddingEvidenceRetriever()

    assert "C:/private" not in str(error.value)


def test_semantic_retrieval_ranks_by_embedding_similarity(monkeypatch):
    vectors = {
        "alpha evidence contains enough distinct descriptive terms for an analyst to review its report passage and verify supporting contextual details": [0.8, 0.6],
        "beta evidence contains enough distinct descriptive terms for an analyst to review its report passage and verify supporting contextual details": [0.0, 1.0],
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


def test_semantic_ties_preserve_chunk_order(monkeypatch):
    text = (
        "This synthetic passage contains enough distinct meaningful terms "
        "for a stable embedding retrieval test during analyst assessment and review."
    )
    vectors = {text: [1.0, 0.0], "same query": [1.0, 0.0]}

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            pass

        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([vectors[value] for value in texts])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    chunks = [
        DocumentChunk(text=text, source="synthetic.pdf", page=page, title="Fixture")
        for page in (1, 2)
    ]

    results = EmbeddingEvidenceRetriever(chunks).search("same query", top_k=2)

    assert [result.chunk.page for result in results] == [1, 2]


def test_semantic_filtering_and_search_options_are_consistent(monkeypatch):
    useful_text = (
        "Investigators found criminals use hidden services to distribute "
        "stolen credentials to buyers seeking account access."
    )
    short_text = "stolen credentials"
    reference_text = (
        "Smith, 2024, A study about stolen credentials. "
        "Retrieved from https://example.org/reference."
    )
    vectors = {
        useful_text: [0.0, 1.0],
        "matching query": [0.0, 1.0],
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
        for page, text in enumerate(
            [useful_text, short_text, reference_text], start=1
        )
    ]
    semantic = EmbeddingEvidenceRetriever(chunks)
    hybrid = HybridEvidenceRetriever(chunks)

    assert [result.chunk.page for result in semantic.search("matching query")] == [1]
    assert [result.chunk.page for result in hybrid.search("matching query")] == [1]
    assert semantic.search(None) == []
    assert hybrid.search("matching query", top_k=-1) == []
    assert hybrid.search("matching query", min_score=1.0) == []
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        semantic.search("matching query", top_k=1.5)
    with pytest.raises(ValueError, match="min_score must be a finite number"):
        hybrid.search("matching query", min_score=float("nan"))


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
