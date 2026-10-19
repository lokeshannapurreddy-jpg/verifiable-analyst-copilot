"""Small, manually page-labelled retrieval evaluation set."""

from math import log2
from numbers import Integral


def ranking_metrics(ranked_pages, relevant_pages, k: int = 3) -> dict[str, float]:
    """Compute page-level binary Recall@k, MRR@k, and nDCG@k.

    Duplicate page IDs are collapsed at their first rank. The caller may pass
    pages from the top-k passage results; metrics use the first k unique pages.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    relevant = set(relevant_pages)
    if not relevant:
        raise ValueError("relevant_pages must not be empty")

    unique_pages = list(dict.fromkeys(ranked_pages))[:int(k)]
    relevant_ranks = [
        rank for rank, page in enumerate(unique_pages, start=1)
        if page in relevant
    ]
    recall = len(set(unique_pages) & relevant) / len(relevant)
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / log2(rank + 1) for rank in relevant_ranks)
    ideal_relevant = min(len(relevant), int(k))
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        f"Recall@{k}": recall,
        f"MRR@{k}": reciprocal_rank,
        f"nDCG@{k}": ndcg,
    }

QUESTIONS = [
    {
        "question": "How do criminals steal and trade personal data?",
        "expected_pages": {9, 12},
    },
    {
        "question": "How do cybercriminals use stolen credentials?",
        "expected_pages": {12, 13},
    },
    {
        "question": "What role do infostealers play in cybercrime?",
        "expected_pages": {13},
    },
    {
        "question": (
            "How do cybercriminals distribute malware through phishing, "
            "infostealers, malicious websites and droppers?"
        ),
        "expected_pages": {13, 14, 17},
    },
    {
        "question": "How do criminals monetize stolen data?",
        "expected_pages": {9, 17},
    },
]
