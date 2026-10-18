# Retrieval Benchmark — Preliminary

Source: Europol IOCTA 2025
Page-level chunks: 73
Evaluation questions: 5

| Retriever | Mean Recall@3 |
|---|---:|
| TF-IDF | 0.833 |
| Hybrid | 0.633 |
| Semantic | 0.433 |

Current best measured approach: TF-IDF.

Known misses:
- Malware distribution: expected page 14 was missed.
- Monetization of stolen data: expected page 9 was missed.

Caveat: The expected-page labels are preliminary and require manual verification. This small evaluation set does not establish general retrieval performance.

Tests: 14 passed.
