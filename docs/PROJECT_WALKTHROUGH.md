# Project walkthrough

## Request and evidence flow

1. `src/ingest.py` extracts text from each PDF page and splits it into overlapping passages. Every passage retains its PDF filename and one-based page number, plus its title and optional source URL.
2. `src/rag.py` provides TF-IDF retrieval by default. Semantic retrieval uses a Sentence Transformers model, and hybrid retrieval combines lexical and embedding rankings. The embedding package is an optional dependency.
3. The Streamlit app and FastAPI service use the same retrieval and answer modules. The API routes requests through `src/agent.py`, a deterministic tool router for evidence search and index metadata. Omitted API session IDs are generated uniquely; callers reuse a returned ID for their own follow-ups. Context is bounded in process memory and resets when the process restarts. Session IDs are not authentication credentials.
4. `src/answer.py` selects complete, relevant sentences from retrieved passages by default. Optional OpenAI synthesis receives the question and retrieved passages, uses a 20-second timeout and 256-token completion limit, and checks citation IDs against the retrieved result set. Missing LLM configuration returns an actionable message; provider failures or invalid output use a clearly labeled extractive fallback. Citation-ID checks do not prove factual entailment.
5. The API records aggregate request metadata through `src/analytics.py`. SQLite is the local default; Docker Compose configures PostgreSQL. Raw questions and answers are not stored, although hashes of question text are deterministic and should not be treated as anonymization.

## Design choices and trade-offs

- TF-IDF keeps the normal local path simple and does not require embedding downloads. Semantic and hybrid modes trade extra dependencies and model loading for a different retrieval signal.
- Retrieval applies the same nonempty, useful-passage, and reference filters to every backend. `min_score` uses backend-native units (boosted TF-IDF cosine, embedding cosine, or hybrid reciprocal-rank-fusion score), so scores should not be compared directly across backends. Equal semantic scores retain input order; hybrid score ties retain lexical-then-semantic insertion order.
- Extractive answers keep the default response tied to text actually present in a retrieved passage. The optional LLM mode improves synthesis but citation-ID checks only show that referenced passages were retrieved; they do not prove entailment or truth.
- The agent is intentionally deterministic. It selects from a small set of tools rather than delegating control to an autonomous LLM.
- PostgreSQL stores telemetry, not vectors. The source PDF is provided locally and excluded from Git.
- Compose binds published service ports to loopback for local use. The request and analytics endpoints have no authentication; this configuration is not suitable for public deployment.

## Verification

- `tests/test_rag.py` checks lexical retrieval behavior and bibliography/boilerplate filtering.
- `tests/test_retrieval_evaluation.py` checks semantic ranking and hybrid fusion with deterministic synthetic embeddings, then exercises the shared five-query, page-labeled evaluation set across TF-IDF and optional embedding backends when the local source PDF is available. The PDF-backed cases skip when it is absent.
- `tests/test_answer.py` checks extractive evidence/citation behavior and mocked optional-provider validation and fallback paths.
- `tests/test_agent.py`, `tests/test_api.py`, and `tests/test_analytics.py` cover routing, bounded memory, request/reload behavior, aggregate SQLite telemetry, and the opt-in PostgreSQL integration test.
- `scripts/evaluate_retrieval.py` reports page-level Recall@3, MRR@3, and nDCG@3 after de-duplicating the top-three passage results by page. Its labels are single-reviewer binary relevance judgments checked against the available PDF text; the small set is diagnostic, not a general benchmark or answer-factuality evaluation.

The OpenAI tests use a mocked client; a live provider check requires a configured key. Do not describe live integration as verified unless a real request succeeds. PostgreSQL behavior requires a separate test database through `TEST_DATABASE_URL`.
