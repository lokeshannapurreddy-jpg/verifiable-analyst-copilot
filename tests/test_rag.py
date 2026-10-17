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
