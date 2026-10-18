import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from statistics import mean

from src.ingest import ingest_pdf
from src.rag import (
    EvidenceRetriever,
    EmbeddingEvidenceRetriever,
    HybridEvidenceRetriever,
)

PDF_PATH = Path("data/reports/iocta_report.pdf")

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
        "question": "How do cybercriminals distribute malware through phishing, infostealers, malicious websites and droppers?",
        "expected_pages": {13, 14, 17},
    },
    {
        "question": "How do criminals monetize stolen data?",
        "expected_pages": {9, 17},
    },
]


def evaluate(name, retriever):
    scores = []

    print(f"\n{'=' * 78}")
    print(f"{name.upper()} RETRIEVAL EVALUATION")
    print("=" * 78)

    for index, item in enumerate(QUESTIONS, start=1):
        results = retriever.search(item["question"], top_k=3)
        retrieved_pages = {result.chunk.page for result in results}
        expected_pages = item["expected_pages"]
        found_pages = retrieved_pages & expected_pages
        missed_pages = expected_pages - retrieved_pages
        recall = len(found_pages) / len(expected_pages)
        scores.append(recall)

        print(f"\nQuestion {index}: {item['question']}")
        print(f"  Expected pages:  {sorted(expected_pages)}")
        print(f"  Retrieved pages: {sorted(retrieved_pages)}")
        print(f"  Relevant found:  {sorted(found_pages)}")
        print(f"  Relevant missed: {sorted(missed_pages)}")
        print(f"  Recall@3:        {recall:.3f}")

    print(f"\n{name} mean recall@3: {mean(scores):.3f}")
    return mean(scores)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Source PDF not found: {PDF_PATH}. "
            "Run this script from the repository root."
        )

    chunks = ingest_pdf(PDF_PATH, title="Europol IOCTA 2025")
    print(f"Source: {PDF_PATH}")
    print(f"Page-level chunks loaded: {len(chunks)}")
    print("Note: expected-page labels are preliminary and need manual verification.")

    results = {}
    results["TF-IDF"] = evaluate("TF-IDF", EvidenceRetriever(chunks))
    results["Semantic"] = evaluate(
        "Semantic", EmbeddingEvidenceRetriever(chunks)
    )
    results["Hybrid"] = evaluate(
        "Hybrid", HybridEvidenceRetriever(chunks)
    )

    print(f"\n{'=' * 78}")
    print("OVERALL COMPARISON — PRELIMINARY")
    print("=" * 78)
    for name, score in results.items():
        print(f"{name:10} mean recall@3: {score:.3f}")

    best_name = max(results, key=results.get)
    print(f"\nHighest measured score on this small test set: {best_name}")
    print("This is not a validated benchmark; verify expected pages before drawing conclusions.")


if __name__ == "__main__":
    main()



