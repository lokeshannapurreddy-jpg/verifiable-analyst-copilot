# Retrieval Benchmark — Preliminary

`scripts/evaluate_retrieval.py` is the current benchmark entry point. It uses five manually labelled questions and prints Recall@3 for TF-IDF, semantic, and hybrid retrieval. Its labels still need manual verification, so the results are diagnostic only.

Earlier result figures in this note were removed because their expected-page labels did not match the current benchmark script. Rerun the script and verify its labels before quoting numeric results in a project application.

TF-IDF remains the runtime default. This benchmark is small and does not establish general retrieval quality.
