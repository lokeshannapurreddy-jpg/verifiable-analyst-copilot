import sys
from types import SimpleNamespace

import pytest

from src.answer import generate_answer
from src.rag import DocumentChunk, EvidenceRetriever


def make_results():
    chunks = [
        DocumentChunk(
            text=(
                "Hybrid threat actors may exploit stolen data and access "
                "services to launch cyber-attacks against governments and "
                "critical infrastructure, resulting in widespread "
                "disruption and instability."
            ),
            source="iocta_report.pdf",
            page=6,
            title="IOCTA 2025",
            url="https://example.org/report",
        ),
        DocumentChunk(
            text=(
                "By infiltrating secure systems, they can steal data of "
                "strategic importance for governments or businesses and "
                "provide hybrid threat actors with invaluable information "
                "that can be used for espionage, economic advantage "
                "or even coercion."
            ),
            source="iocta_report.pdf",
            page=6,
            title="IOCTA 2025",
            url="https://example.org/report",
        ),
        DocumentChunk(
            text=(
                "A Adv anced persistent threat (APT) groups are threat "
                "actors often sponsored and operated by nation states."
            ),
            source="iocta_report.pdf",
            page=6,
            title="IOCTA 2025",
            url="https://example.org/report",
        ),
    ]

    retriever = EvidenceRetriever(chunks)
    return retriever.search(
        "What cyber threats do hybrid threat actors pose?",
        top_k=3,
    )


def test_answer_includes_critical_infrastructure_threat():
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        make_results(),
    )
    assert "critical infrastructure" in answer.lower()
    assert "disruption and instability" in answer.lower()


def test_answer_includes_data_theft_and_espionage():
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        make_results(),
    )
    assert "espionage" in answer.lower()
    assert "strategic importance" in answer.lower()


def test_answer_includes_page_citations():
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        make_results(),
    )
    assert "p. 6" in answer


def test_answer_excludes_malformed_apt_footnote():
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        make_results(),
    )
    assert "a adv anced persistent threat" not in answer.lower()


def test_malware_delivery_answer_prioritises_delivery_techniques():
    chunks = [
        DocumentChunk(
            text=(
                "Cybercriminals use a variety of techniques to access "
                "personal data that exploit either system vulnerabilities "
                "or human oversight."
            ),
            source="iocta_report.pdf",
            page=12,
            title="IOCTA 2025",
        ),
        DocumentChunk(
            text=(
                "Phishing techniques are the main vector for the "
                "distribution of infostealers. Criminals send emails, "
                "text messages or messages on social media that contain "
                "malicious attachments or URLs which introduce malware "
                "into the victim's system. Malicious websites are also "
                "used to distribute malware through search engine "
                "advertising and search engine optimisation poisoning."
            ),
            source="iocta_report.pdf",
            page=13,
            title="IOCTA 2025",
        ),
    ]

    retriever = EvidenceRetriever(chunks)
    question = "What malware delivery techniques do cybercriminals use?"
    results = retriever.search(question, top_k=5)
    answer = generate_answer(question, results)

    assert "p. 13" in answer
    assert "phishing" in answer.lower() or "malicious attachments" in answer.lower()
def test_answer_citation_uses_retrieved_chunk_page():
    results = make_results()
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        results,
    )

    for result in results:
        if result.chunk.page == 6:
            assert result.chunk.source == "iocta_report.pdf"

    assert "[1, p. 6]" in answer or "[2, p. 6]" in answer


def test_answer_with_no_results_does_not_invent_citations():
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        [],
    )

    assert "no factual answer is provided" in answer.lower()
    assert "Sources:" not in answer


def test_answer_does_not_invent_missing_page_number():
    chunk = DocumentChunk(
        text=(
            "Cybercriminals exploit vulnerabilities in computer systems "
            "to gain unauthorized access to sensitive information."
        ),
        source="iocta_report.pdf",
        page=0,
        title="IOCTA 2025",
    )

    answer = generate_answer(
        "How do cybercriminals exploit vulnerabilities?",
        [type("Result", (), {"chunk": chunk, "score": 1.0})()],
    )

    assert "p. 0" not in answer


def test_generated_factual_sentences_are_traceable_to_retrieved_evidence():
    results = make_results()
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        results,
    )

    evidence_text = " ".join(
        result.chunk.text.lower() for result in results
    )

    factual_lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
        and line.strip().lower() not in {
            "based on the retrieved evidence:",
            "based on the retrieved evidence",
            "sources:",
        }
        and not line.strip().lower().startswith(
            (
                "note:",
                "disclaimer:",
                "source ",
                "this is an extractive answer:",
            )
        )
        and not line.strip().startswith("[")
    ]

    assert factual_lines, "Expected at least one answer statement."

    for line in factual_lines:
        cleaned_line = line.lower()
        assert any(
            phrase in evidence_text
            for phrase in (
                "critical infrastructure",
                "disruption and instability",
                "strategic importance",
                "espionage",
            )
            if phrase in cleaned_line
        ), f"Could not trace answer statement to known evidence: {line}"


def test_answer_without_results_has_no_citations():
    answer = generate_answer(
        "What cyber threats do hybrid threat actors pose?",
        [],
    )

    assert "no factual answer is provided" in answer.lower()
    assert "Sources:" not in answer


def test_optional_llm_answer_validates_citations(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="The report describes cyber threats [1]."))]
    )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer = generate_answer("What threats are described?", make_results(), mode="llm")
    assert "The report describes cyber threats [1]." in answer
    assert "Sources:" in answer
    assert "IOCTA 2025 - iocta_report.pdf, page 6" in answer


def test_optional_llm_rejects_unknown_citation_and_falls_back(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="The report says this [99]."))]
    )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer = generate_answer("What threats are described?", make_results(), mode="llm")
    assert "extractive answer" in answer
    assert "[99]" not in answer


@pytest.mark.parametrize(
    "content",
    [
        "The report describes cyber threats.",
        "The report describes cyber threats [source].",
        "The report describes cyber threats [01].",
    ],
)
def test_optional_llm_rejects_missing_or_malformed_citations(monkeypatch, content):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **call_kwargs: response)
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer = generate_answer("What threats are described?", make_results(), mode="llm")

    assert "This is an extractive answer" in answer
    assert content not in answer


@pytest.mark.parametrize("failure", [TimeoutError("provider timeout"), RuntimeError("provider unavailable")])
def test_optional_llm_provider_failures_fall_back(monkeypatch, failure):
    class FakeCompletions:
        def create(self, **kwargs):
            raise failure

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer = generate_answer("What threats are described?", make_results(), mode="llm")

    assert "This is an extractive answer" in answer


def test_optional_llm_empty_response_falls_back(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  "))]
    )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **call_kwargs: response)
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer = generate_answer("What threats are described?", make_results(), mode="llm")

    assert "This is an extractive answer" in answer


def test_optional_llm_malformed_provider_response_falls_back(monkeypatch):
    response = SimpleNamespace(choices=[])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **call_kwargs: response)
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer = generate_answer("What threats are described?", make_results(), mode="llm")

    assert "This is an extractive answer" in answer


def test_optional_llm_is_not_called_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = generate_answer("What threats are described?", make_results(), mode="llm")
    assert "This is an extractive answer" in answer


def test_optional_llm_uses_bounded_timeout_and_deduplicates_sources(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="The report describes cyber threats [1].\n"
                "It also notes infrastructure disruption [1]."
            )
        )]
    )
    client_options = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_options.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **call_kwargs: response)
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    answer = generate_answer("What threats are described?", make_results(), mode="llm")

    assert client_options == {"timeout": 20.0, "max_retries": 0}
    assert answer.split("Sources:", 1)[1].count("[1]") == 1


def test_unrelated_question_with_evidence_returns_insufficient_evidence():
    answer = generate_answer("What is the capital of Mars?", make_results())
    assert "no factual answer is provided" in answer.lower()
