import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from statistics import mean

from src.ingest import ingest_pdf
from src.evaluation import QUESTIONS, ranking_metrics
from src.rag import (
    EvidenceRetriever,
    EmbeddingEvidenceRetriever,
    HybridEvidenceRetriever,
)

PDF_PATH = Path("data/reports/iocta_report.pdf")

def evaluate(name, retriever):
    recalls = []
    reciprocal_ranks = []
    ndcgs = []

    print(f"\n{'=' * 78}")
    print(f"{name.upper()} RETRIEVAL EVALUATION")
    print("=" * 78)

    for index, item in enumerate(QUESTIONS, start=1):
        results = retriever.search(item["question"], top_k=3)
        ranked_pages = list(dict.fromkeys(result.chunk.page for result in results))
        retrieved_pages = set(ranked_pages)
        expected_pages = item["expected_pages"]
        found_pages = retrieved_pages & expected_pages
        missed_pages = expected_pages - retrieved_pages
        metrics = ranking_metrics(ranked_pages, expected_pages, k=3)
        recall = metrics["Recall@3"]
        recalls.append(recall)
        reciprocal_ranks.append(metrics["MRR@3"])
        ndcgs.append(metrics["nDCG@3"])

        print(f"\nQuestion {index}: {item['question']}")
        print(f"  Expected pages:  {sorted(expected_pages)}")
        print(f"  Ranked pages:    {ranked_pages}")
        print(f"  Relevant found:  {sorted(found_pages)}")
        print(f"  Relevant missed: {sorted(missed_pages)}")
        print(f"  Recall@3:        {recall:.3f}")
        print(f"  MRR@3:           {reciprocal_ranks[-1]:.3f}")
        print(f"  nDCG@3:          {ndcgs[-1]:.3f}")

    summary = {
        "Recall@3": mean(recalls),
        "MRR@3": mean(reciprocal_ranks),
        "nDCG@3": mean(ndcgs),
    }
    print(
        f"\n{name} means: "
        + ", ".join(f"{metric} {score:.3f}" for metric, score in summary.items())
    )
    return summary


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Source PDF not found: {PDF_PATH}. "
            "Run this script from the repository root."
        )

    chunks = ingest_pdf(PDF_PATH, title="Europol IOCTA 2025")
    print(f"Source: {PDF_PATH}")
    print(f"Page-level chunks loaded: {len(chunks)}")
    print(
        "Page labels were checked against the local PDF text. They are "
        "single-reviewer binary judgments, not graded or independently validated."
    )
    print("Ranking de-duplicates the top-three passage results by PDF page.")

    results = {}
    results["TF-IDF"] = evaluate("TF-IDF", EvidenceRetriever(chunks))
    results["Semantic"] = evaluate(
        "Semantic", EmbeddingEvidenceRetriever(chunks)
    )
    results["Hybrid"] = evaluate(
        "Hybrid", HybridEvidenceRetriever(chunks)
    )

    print(f"\n{'=' * 78}")
    print("OVERALL COMPARISON - PRELIMINARY")
    print("=" * 78)
    for name, scores in results.items():
        summary = ", ".join(
            f"mean {metric} {score:.3f}"
            for metric, score in scores.items()
        )
        print(f"{name:10} {summary}")

    for metric in ("Recall@3", "MRR@3", "nDCG@3"):
        best_name = max(results, key=lambda name: results[name][metric])
        print(
            f"Best measured {metric}: {best_name} "
            f"({results[best_name][metric]:.3f})"
        )
    print("This five-question evaluation is diagnostic only and does not establish general retrieval quality.")


if __name__ == "__main__":
    main()



