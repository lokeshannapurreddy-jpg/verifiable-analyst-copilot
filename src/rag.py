"""Source-grounded retrieval for public cybercrime reports."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
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


def _validate_search_options(query, top_k, min_score):
    """Normalize search options while keeping empty queries harmless."""
    if query is None:
        query = ""
    elif not isinstance(query, str):
        raise ValueError("query must be a string or None")

    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise ValueError("top_k must be a positive integer")
    if isinstance(min_score, bool) or not isinstance(min_score, Real):
        raise ValueError("min_score must be a finite number")
    if not math.isfinite(float(min_score)):
        raise ValueError("min_score must be a finite number")

    return query.strip(), int(top_k), float(min_score)


class EvidenceRetriever:
    """Retrieve passages using TF-IDF and lexical relevance checks."""

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

    @staticmethod
    def _keywords(text: str) -> set[str]:
        """Extract meaningful words."""
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {
            word for word in words
            if len(word) > 2 and word not in ENGLISH_STOP_WORDS
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
            "retrieved from http://",
            "retrieved from https://",
            "doi.org/",
            "the ever-evolving threat of the russian-speaking cybercriminal underground",
        )

        if any(signal in joined for signal in reference_signals):
            return True

        citation_start = re.compile(
            r"^(?:\[?\d{1,3}\]?\.?\s*)?"
            r"[A-Z][A-Za-z0-9&.,'?() -]{1,80},\s*(?:19|20)\d{2}\b"
        )
        citation_lines = sum(
            bool(citation_start.search(line)) for line in lines
        )

        return citation_lines >= 2 or (
            len(lines) <= 2 and citation_lines == 1
        )

    @classmethod
    def _is_searchable_chunk(cls, chunk) -> bool:
        """Apply the common nonempty, useful-passage, and reference filters."""
        text = getattr(chunk, "text", None)
        return (
            isinstance(text, str)
            and bool(text.strip())
            and cls._is_useful_passage(text)
            and not cls._is_reference_passage(text)
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.01,
    ) -> list[SearchResult]:
        """Retrieve passages; min_score uses boosted TF-IDF cosine-score units."""
        query, top_k, min_score = _validate_search_options(
            query, top_k, min_score
        )
        if not query or top_k < 1:
            return []
        if self.matrix is None:
            return []

        query_keywords = self._keywords(query)
        if not query_keywords:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        candidates = []

        for index, raw_score in enumerate(scores):
            score = float(raw_score)
            chunk = self.chunks[index]
            if not self._is_searchable_chunk(chunk):
                continue

            matched = query_keywords.intersection(self._keywords(chunk.text))
            if not matched:
                continue

            overlap = len(matched) / len(query_keywords)
            adjusted_score = score * (1.0 + 0.25 * overlap)
            if adjusted_score < min_score:
                continue
            candidates.append(
                SearchResult(chunk=chunk, score=adjusted_score)
            )

        candidates.sort(key=lambda result: result.score, reverse=True)

        # Preserve multiple relevant passages from the same page.
        return candidates[:top_k]


class EmbeddingEvidenceRetriever:
    """Retrieve passages using semantic embeddings."""

    def __init__(self, chunks=(), model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic and hybrid retrieval require the optional ML "
                "dependencies. Install them with: "
                "python -m pip install -r requirements-ml.txt"
            ) from exc
        self.chunks = [
            chunk for chunk in chunks
            if EvidenceRetriever._is_searchable_chunk(chunk)
        ]
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load semantic model {model_name!r}. Check model "
                "availability and network/cache access; semantic and hybrid "
                "retrieval cannot run until it loads."
            ) from exc
        self.embeddings = (
            self.model.encode([c.text for c in self.chunks], normalize_embeddings=True)
            if self.chunks else []
        )

    def search(self, query, top_k=3, min_score=0.01):
        query, top_k, min_score = _validate_search_options(
            query, top_k, min_score
        )
        if not self.chunks or not query or top_k < 1:
            return []
        import numpy as np
        vector = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.asarray(self.embeddings) @ vector
        # Stable sorting keeps original chunk order for equal cosine scores.
        indices = np.argsort(-scores, kind="stable")
        return [
            SearchResult(self.chunks[int(i)], float(scores[i]))
            for i in indices if scores[i] >= min_score
        ][:top_k]


class HybridEvidenceRetriever:
    """Combine lexical and semantic retrieval rankings."""

    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.lexical = EvidenceRetriever(self.chunks)
        self.semantic = EmbeddingEvidenceRetriever(self.chunks)

    def search(self, query, top_k=3, min_score=0.01):
        query, top_k, min_score = _validate_search_options(
            query, top_k, min_score
        )
        if not query or top_k < 1:
            return []
        rankings = [
            self.lexical.search(query, top_k=max(top_k * 3, 10), min_score=0),
            self.semantic.search(query, top_k=max(top_k * 3, 10), min_score=0),
        ]
        scores = {}
        results = {}
        for ranking in rankings:
            for rank, result in enumerate(ranking, start=1):
                key = (result.chunk.source, result.chunk.page, result.chunk.text)
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                results[key] = result.chunk
        # Dict insertion order follows lexical then semantic rank order, making
        # equal fused scores deterministic.
        ordered = sorted(scores, key=lambda key: -scores[key])
        return [
            SearchResult(results[key], scores[key])
            for key in ordered if scores[key] >= min_score
        ][:top_k]


def create_retriever(chunks=(), backend: str = "tfidf"):
    """Create the selected retrieval backend; embeddings remain opt-in."""
    choices = {
        "tfidf": EvidenceRetriever,
        "semantic": EmbeddingEvidenceRetriever,
        "hybrid": HybridEvidenceRetriever,
    }
    try:
        retriever_type = choices[backend.strip().lower()]
    except (AttributeError, KeyError):
        raise ValueError("RETRIEVER_BACKEND must be tfidf, semantic, or hybrid")
    return retriever_type(chunks)
