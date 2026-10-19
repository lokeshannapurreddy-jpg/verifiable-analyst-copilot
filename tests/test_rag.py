import pytest
import numpy as np

import src.rag as rag
from src.rag import DocumentChunk, EvidenceRetriever


def make_retriever():
    chunks = [
        DocumentChunk(
            text=(
                "Cybercriminals sell stolen credentials and breached "
                "data through underground marketplaces. Criminals "
                "also purchase initial access to compromised networks."
            ),
            source="cybercrime_report.pdf",
            page=20,
            title="Cybercrime Report",
            url="https://example.org/report",
        ),
        DocumentChunk(
            text=(
                "Cybercriminals acquire personal data through phishing, "
                "system vulnerabilities, and stolen credentials."
            ),
            source="cybercrime_report.pdf",
            page=12,
            title="Cybercrime Report",
        ),
        DocumentChunk(
            text=(
                "This publication and more information about the "
                "organisation are available on the internet."
            ),
            source="cybercrime_report.pdf",
            page=28,
            title="Cybercrime Report",
        ),
    ]
    return EvidenceRetriever(chunks)


def test_relevant_question_returns_evidence():
    retriever = make_retriever()
    results = retriever.search(
        "How do criminals sell stolen credentials?",
        top_k=3,
    )
    assert results
    assert any("stolen credentials" in r.chunk.text for r in results)


def test_unrelated_question_returns_no_evidence():
    retriever = make_retriever()
    results = retriever.search("What is the population of Mars?")
    assert results == []


def test_publication_boilerplate_is_not_returned():
    retriever = make_retriever()
    results = retriever.search(
        "Where can criminals purchase stolen data?",
        top_k=3,
    )
    assert all(
        "available on the internet" not in r.chunk.text
        for r in results
    )


def test_empty_question_returns_no_evidence():
    retriever = make_retriever()
    assert retriever.search("") == []
    assert retriever.search(None) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_top_k_returns_no_evidence(top_k):
    assert make_retriever().search("stolen credentials", top_k=top_k) == []


@pytest.mark.parametrize("top_k", [1.5, True, "2"])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        make_retriever().search("stolen credentials", top_k=top_k)


@pytest.mark.parametrize("min_score", [float("nan"), float("inf"), "0.2", True])
def test_invalid_min_score_is_rejected(min_score):
    with pytest.raises(ValueError, match="min_score must be a finite number"):
        make_retriever().search("stolen credentials", min_score=min_score)


def test_tfidf_min_score_uses_keyword_boosted_score(monkeypatch):
    retriever = make_retriever()
    monkeypatch.setattr(
        rag,
        "cosine_similarity",
        lambda query, matrix: np.full(matrix.shape[0], 0.9),
    )

    results = retriever.search("stolen credentials", min_score=1.1)

    assert results
    assert results[0].score >= 1.1


def test_bibliography_entries_are_not_returned():
    chunks = [
        DocumentChunk(
            text=(
                "TrendMicro, 2025, The Ever-Evolving Threat of the "
                "Russian-Speaking Cybercriminal Underground. "
                "Available at https://example.org/report."
            ),
            source="cybercrime_report.pdf",
            page=26,
            title="Cybercrime Report",
        ),
        DocumentChunk(
            text=(
                "Cybercriminals use phishing and stolen credentials "
                "to gain access to organisational systems. These "
                "methods can expose sensitive information and "
                "enable further criminal activity."
            ),
            source="cybercrime_report.pdf",
            page=17,
            title="Cybercrime Report",
        ),
    ]

    retriever = EvidenceRetriever(chunks)
    results = retriever.search(
        "What techniques do cybercriminals use to gain access?",
        top_k=3,
    )

    assert results
    assert all(
        "TrendMicro" not in result.chunk.text
        for result in results
    )
def test_simple_malware_question_retrieves_relevant_evidence():
    chunks = [
        DocumentChunk(
            text=(
                "Europol, 2025, Cybercrime Threat Assessment, "
                "accessible at https://example.org/malware-report."
            ),
            source="cybercrime_report.pdf",
            page=25,
            title="Cybercrime Report",
        ),
        DocumentChunk(
            text=(
                "Phishing techniques are the main vector for the "
                "distribution of infostealers. Criminals send emails, "
                "text messages or social media messages containing "
                "malicious attachments or URLs that introduce malware "
                "into victims' systems. Malicious websites can also "
                "distribute malware through search engine advertising."
            ),
            source="cybercrime_report.pdf",
            page=13,
            title="Cybercrime Report",
        ),
    ]

    retriever = EvidenceRetriever(chunks)
    results = retriever.search("What does the report say about malware?")

    assert results
    assert any(
        "introduce malware into victims' systems" in result.chunk.text
        for result in results
    )
    assert all(
        "accessible at https://example.org" not in result.chunk.text
        for result in results
    )
