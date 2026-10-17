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
