# Verifiable Analyst Copilot

A source-grounded research prototype for exploring public cybercrime reports. It combines page-aware PDF ingestion, configurable evidence retrieval, cited answers, a Streamlit UI, a FastAPI service, bounded follow-up memory, and privacy-conscious query telemetry.

## What it does

- Extracts PDF text into page-linked passages.
- Retrieves relevant passages using TF-IDF by default, with optional semantic or hybrid embeddings.
- Answers from retrieved evidence with an extractive default and optional constrained LLM synthesis.
- Preserves source title, filename, URL, and page number.
- Exposes `/health`, `/ask`, `/reload`, and aggregate `/analytics` endpoints.
- Uses a deterministic tool router for evidence search and index metadata.
- Keeps a small in-memory context per session for likely follow-up questions.
- Offers optional LLM synthesis constrained to retrieved passages, with citation-ID validation and extractive fallback.
- Stores query telemetry without storing raw questions or answers.
- Supports SQLite by default and PostgreSQL when `DATABASE_URL` is configured.
- Includes automated tests, Docker packaging, Docker Compose, and GitHub Actions CI.

## Architecture

```text
PDF reports
    |
    v
src/ingest.py -> page-linked passages -> src/rag.py (TF-IDF by default)
                                            |
Question -> FastAPI -> src/agent.py --------+
                        |                   |
                        v                   v
                  session memory      src/answer.py
                                            |
                                            v
                              cited answer + citations
                                            |
                         +------------------+----------------+
                         v                                   v
                    Streamlit UI                       FastAPI response
                                                            |
                                                            v
                                               aggregate query telemetry
                                               SQLite or PostgreSQL
```

## Source report

The demonstration uses Europol's 2025 Internet Organised Crime Threat Assessment report, *Steal, Deal and Repeat: How Cybercriminals Trade and Exploit Your Data*.

Official publication page:
https://www.europol.europa.eu/publication-events/main-reports/steal-deal-and-repeat-how-cybercriminals-trade-and-exploit-your-data

Download the PDF and save it as `data/reports/iocta_report.pdf`. The PDF is intentionally excluded from Git. The Streamlit app can attempt to download the public report if the local file is absent; for API and Docker runs, place it in `data/reports` for reliable startup.

## Local setup (Windows PowerShell)

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Put the report at `data/reports/iocta_report.pdf`.

Start the API in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the UI in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

API documentation: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/health
Aggregate telemetry: http://127.0.0.1:8000/analytics

### Example API request

```powershell
$body = @{
  question = "How do cybercriminals use stolen credentials?"
  top_k = 3
  session_id = "demo-session"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" -Body $body
```

Reload the index after adding or replacing a PDF:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/reload
```

If `RELOAD_ADMIN_KEY` is set in the environment, include it as the `X-Admin-Key` header.

## Tests

The full test suite includes semantic and hybrid retrieval checks, so install the optional ML requirements as well as the base requirements:

```powershell
python -m pip install -r requirements.txt -r requirements-ml.txt
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements-ml.txt` remains optional for normal TF-IDF-only use.

## Docker Compose

1. Copy `.env.example` to `.env`.
2. Replace the placeholder values with long random secrets. Avoid URL-reserved characters in `POSTGRES_PASSWORD` unless URL-encoded.
3. Ensure `data/reports/iocta_report.pdf` exists.
4. Run:

```powershell
docker compose up --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

Docker Compose starts PostgreSQL for query telemetry. The application defaults to SQLite when `DATABASE_URL` is not set. The Postgres database is not a vector store: current retrieval remains TF-IDF.

## Optional semantic retrieval benchmark

Semantic and hybrid retrievers require the larger optional dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py
```

The included benchmark has only five questions and its expected-page labels are preliminary. The previous note's numbers used different labels from the current script, so no score is presented as current; rerun and verify the labels before comparing backends or quoting metrics.

To use semantic or hybrid retrieval in the API, install `requirements-ml.txt` and set `RETRIEVER_BACKEND=semantic` or `RETRIEVER_BACKEND=hybrid`. For Docker Compose, set `OPTIONAL_DEPENDENCIES=ml` in `.env` and rebuild. TF-IDF remains the default; rerun the corrected benchmark before selecting another backend.

## Optional LLM answers

The default answer path is extractive and requires no API key. To enable optional OpenAI synthesis locally, install `requirements-llm.txt`, set `ANSWER_MODE=llm`, and provide `OPENAI_API_KEY` through the environment. Set `OPENAI_MODEL` to choose a supported model. For Docker Compose, use `OPTIONAL_DEPENDENCIES=llm` and set these runtime variables in the untracked `.env` file. The LLM receives retrieved passages only; the application checks that cited evidence IDs exist and that each answer line includes a citation. Invalid or unavailable responses fall back to the extractive answer. This citation check validates references, not whether each generated claim is entailed by its cited passage, so verify the source pages.

## Privacy and security notes

- Query telemetry stores timestamp, hashed session/question identifiers, question length, result count, latency, and status. It does not store raw question text or answer text.
- Session memory is held in process memory and is bounded by a maximum number of sessions; it resets when the API process restarts.
- Set `RELOAD_ADMIN_KEY` to protect document reloads when the API is reachable by other users.
- Do not expose the demo API publicly without adding authentication, rate limiting, HTTPS, deployment-specific secret management, and a review of logging/retention requirements.
- Do not use session IDs as authentication credentials.
- The app is a research prototype, not a production intelligence platform.

## Evaluation and limitations

- Automated tests cover selected retrieval, answer selection, agent routing, session-memory, and telemetry behaviors.
- PostgreSQL telemetry has an opt-in integration check using `TEST_DATABASE_URL`; SQLite remains covered by the default unit test.
- TF-IDF is the default retriever; relevance scores are not truth or confidence scores.
- Extractive and optional LLM answers do not independently establish whether a sentence is true; citation checks validate reference IDs, not entailment.
- The current retrieval benchmark is small and not independently validated.
- PDF text extraction can produce broken sentences, missing tables, or formatting artifacts.
- The telemetry database contains aggregate operational metadata, not an analyst-grade incident database.
- PostgreSQL support is for telemetry storage. pgvector, a React frontend, and public cloud deployment are not implemented in this version.

## Responsible use

Use only authorized, publicly available reports. Verify claims in their original cited pages before relying on them. This prototype does not independently establish the truth of source statements.

## License

No license has been specified. Until one is added, do not assume the repository grants permission to reuse or redistribute the code.
