"""Source-grounded retrieval for public cybercrime reports."""

from dataclasses import dataclass
from typing import Iterable
import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class DocumentChunk:
    """A text passage with source metadata."""
    text: str
    source: str
    page: int
    title: str
    url: str = ""


@dataclass(frozen=True)
class SearchResult:
    """A retrieved passage and its relevance score."""
    chunk: DocumentChunk
    score: float


class EvidenceRetriever:
    """Retrieve passages using TF-IDF and lexical relevance checks."""

    QUESTION_FILLER = {
        "report", "reports", "say", "says", "tell", "tells",
        "describe", "describes", "explain", "explains",
        "discuss", "discusses", "mention", "mentions",
        "according", "document", "documents",
    }

    def __init__(self, chunks: Iterable[DocumentChunk] = ()):
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.matrix = None
        self._fit()

    def _fit(self) -> None:
        """Build the searchable index, excluding empty passages."""
        self.matrix = None
        self.chunks = [
            chunk for chunk in self.chunks
            if chunk.text and chunk.text.strip()
        ]

        if not self.chunks:
            return

        self.matrix = self.vectorizer.fit_transform(
            [chunk.text for chunk in self.chunks]
        )

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        """Add passages and rebuild the index."""
        self.chunks.extend(chunks)
        self._fit()

    @classmethod
    def _keywords(cls, text: str) -> set[str]:
        """Extract meaningful topic words, excluding question filler."""
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {
            word for word in words
            if len(word) > 2
            and word not in ENGLISH_STOP_WORDS
            and word not in cls.QUESTION_FILLER
        }

    @staticmethod
    def _is_useful_passage(text: str) -> bool:
        """Reject short cover-page headers and table-of-contents fragments."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        meaningful = {
            word.lower() for word in words
            if word.lower() not in ENGLISH_STOP_WORDS
        }
        return len(meaningful) >= 12

    @staticmethod
    def _is_reference_passage(text: str) -> bool:
        """Reject bibliography entries and source-list fragments."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return True

        joined = " ".join(lines).lower()

        reference_signals = (
            "available at http://",
            "available at https://",
            "accessible at http://",
            "accessible at https://",
            "retrieved from http://",
            "retrieved from https://",
            "doi.org/",
            "the ever-evolving threat of the russian-speaking cybercriminal underground",
        )

        if any(signal in joined for signal in reference_signals):
            return True

        # PDF extraction sometimes wraps bibliography entries into one line.
        citation_start = re.compile(
            r"^(?:\[\s*\d{1,3}\s*\]\.?\s*)?"
            r"[A-Z][A-Za-z0-9&.,'’() -]{1,80},\s*(?:19|20)\d{2}\b"
        )

        citation_lines = sum(
            bool(citation_start.search(line)) for line in lines
        )

        return citation_lines >= 2 or (
            len(lines) <= 2 and citation_lines == 1
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.01,
    ) -> list[SearchResult]:
        """Return relevant, sufficiently substantive passages."""
        if not query or not query.strip() or top_k < 1:
            return []

        if self.matrix is None:
            return []

        query_keywords = self._keywords(query)
        if not query_keywords:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1]
        results = []

        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                continue

            chunk = self.chunks[int(index)]

            if not self._is_useful_passage(chunk.text):
                continue

            if self._is_reference_passage(chunk.text):
                continue

            matched_keywords = (
                query_keywords.intersection(self._keywords(chunk.text))
            )

            # A single distinctive topic word is enough for short queries.
            if not matched_keywords:
                continue

            results.append(SearchResult(chunk=chunk, score=score))

            if len(results) >= top_k:
                break

        return results


def format_evidence(results: list[SearchResult]) -> str:
    """Format retrieved passages with traceable source metadata."""
    if not results:
        return (
            "No sufficiently relevant evidence was found in the "
            "indexed documents. Do not infer an answer."
        )

    sections = []

    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        source_line = (
            f"[{number}] {chunk.title} | "
            f"Page: {chunk.page} | Source: {chunk.source}"
        )

        if chunk.url:
            source_line += f" | URL: {chunk.url}"

        sections.append(
            f"{source_line}\n"
            f"Relevance score: {result.score:.3f}\n"
            f"Evidence: {chunk.text}"
        )

    return "\n\n".join(sections)
