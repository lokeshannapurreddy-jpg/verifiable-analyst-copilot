"""Small, deterministic tool-using research agent.

The agent chooses between evidence retrieval and index metadata tools. It does
not pretend to be an autonomous LLM and never invents source facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import os
from typing import Any

from src.answer import generate_answer


@dataclass
class SessionMemory:
    """Bounded in-memory conversation context for follow-up questions."""
    max_turns: int = 4
    turns: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        # Respect each instance's configured bound rather than a hard-coded default.
        self.turns = deque(self.turns, maxlen=max(1, int(self.max_turns)))

    def add(self, question: str, answer: str) -> None:
        self.turns.append({"question": question[:1000], "answer": answer[:3000]})

    def retrieval_query(self, question: str) -> str:
        """Use the latest question as context only for likely follow-ups."""
        q = question.strip()
        follow_up_starters = (
            "what about", "and ", "how about", "why", "explain that",
            "tell me more", "what does it say about", "those", "this",
        )
        if self.turns and q.lower().startswith(follow_up_starters):
            previous = self.turns[-1]["question"]
            return f"{previous} {q}"
        return q


def run_agent(question: str, retriever: Any, top_k: int = 3,
              memory: SessionMemory | None = None) -> dict:
    """Route a request to a supported tool and return traceable output."""
    q = question.strip()
    if not q:
        return {
            "tool": "none",
            "answer": "Enter a research question.",
            "evidence": [],
            "trace": ["Rejected empty question."],
        }

    lower = q.lower()
    if any(term in lower for term in (
        "how many passages", "how many chunks", "index status",
        "indexed passage", "how many pages are indexed",
    )):
        count = len(getattr(retriever, "chunks", []))
        answer = f"The current index contains {count} searchable passages."
        return {
            "tool": "index_metadata",
            "answer": answer,
            "evidence": [],
            "trace": ["Routed to index metadata tool; no report claim was inferred."],
        }

    retrieval_query = memory.retrieval_query(q) if memory else q
    results = retriever.search(retrieval_query, top_k=top_k)
    answer = generate_answer(q, results)
    evidence = [
        {
            "title": result.chunk.title,
            "source": result.chunk.source,
            "page": result.chunk.page,
            "url": result.chunk.url,
            "score": round(float(result.score), 4),
            "text": result.chunk.text,
        }
        for result in results
    ]
    trace = [
        "Tool selected: evidence_retrieval",
        f"Retrieval query: {retrieval_query}",
        f"Retrieved passages: {len(results)}",
        (
            "Answer mode: optional LLM with citation validation and extractive fallback"
            if os.getenv("ANSWER_MODE", "extractive").strip().lower() == "llm"
            else "Answer method: extractive sentence selection"
        ),
        "Source claims are not independently verified by this tool.",
    ]
    if memory:
        memory.add(q, answer)
    return {"tool": "evidence_retrieval", "answer": answer,
            "evidence": evidence, "trace": trace}
