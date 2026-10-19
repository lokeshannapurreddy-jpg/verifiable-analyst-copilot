from src.agent import SessionMemory, run_agent
from src.rag import DocumentChunk, SearchResult


class FakeRetriever:
    def __init__(self):
        self.chunks = [
            DocumentChunk(
                text=("Cybercriminals use stolen credentials to access accounts. "
                      "Credential theft can enable further criminal activity."),
                source="report.pdf",
                page=4,
                title="Test report",
                url="https://example.org/report",
            )
        ]

    def search(self, query, top_k=3):
        return [SearchResult(self.chunks[0], 0.8)] if query.strip() else []


def test_agent_returns_traceable_evidence():
    output = run_agent("How are stolen credentials used?", FakeRetriever())
    assert output["tool"] == "evidence_retrieval"
    assert output["evidence"][0]["page"] == 4
    assert any("retrieval" in item.lower() for item in output["trace"])


def test_agent_index_metadata_tool():
    output = run_agent("How many passages are indexed?", FakeRetriever())
    assert output["tool"] == "index_metadata"
    assert "1 searchable passages" in output["answer"]
    assert output["evidence"] == []


def test_session_memory_is_bounded_and_follow_up_contextualized():
    memory = SessionMemory(max_turns=2)
    memory.add("How are stolen credentials used?", "They enable account access.")
    assert "How are stolen credentials used?" in memory.retrieval_query(
        "What about that?"
    )
    memory.add("Question two", "Answer two")
    memory.add("Question three", "Answer three")
    assert len(memory.turns) == 2


def test_empty_question_does_not_call_retriever():
    output = run_agent(" ", FakeRetriever())
    assert output["tool"] == "none"
    assert output["evidence"] == []
