# Retrieval Benchmark — Preliminary

`scripts/evaluate_retrieval.py` is the reproducible evaluation entry point. It uses five manually selected questions with binary page-level relevance labels checked against the local source PDF text. The labels are single-reviewer judgments, not graded or independently validated. It reports Recall@3, MRR@3, and nDCG@3 for TF-IDF, semantic, and hybrid retrieval.

The current diagnostic scores are recorded in `README.md`. Re-run the script from the repository root with the source PDF and optional ML requirements available before comparing results. Retrieval metrics do not assess answer factuality or citation entailment.

TF-IDF remains the runtime default. This benchmark is small and does not establish general retrieval quality.
