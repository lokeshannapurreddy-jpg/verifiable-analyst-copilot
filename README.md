# Verifiable Analyst Copilot

A research prototype that helps analysts find and verify evidence in long public cybercrime reports. It ingests PDF text as page-linked passages, retrieves relevant evidence, and presents answers with source citations through a Streamlit interface and a FastAPI service.

## How it works

```text
PDF -> page-aware ingestion -> retrieval (TF-IDF by default; optional semantic/hybrid)
                                      |                         |
                               Streamlit UI               FastAPI service
                                                              |
                                                 agent tools and bounded
                                                   in-memory context
                                                              |
                                              cited answers and telemetry
```

The interfaces use shared retrieval and answer modules. Extractive answers are the default. Optional OpenAI generation receives retrieved passages as evidence; the application validates citation IDs and falls back to extractive answers if generation is unavailable or citations are invalid. Citation validation checks references, not whether a claim is entailed by its source. The API exposes `/health`, `/ask`, `/reload`, and aggregate `/analytics` endpoints. Telemetry uses SQLite by default and can use PostgreSQL when configured.

## Source report

The example report is Europol's *Steal, deal and repeat: How cybercriminals trade and exploit your data*, part of the *Internet Organised Crime Threat Assessment (IOCTA) 2025*. [Official Europol publication](https://www.europol.europa.eu/publication-events/main-reports/steal-deal-and-repeat-how-cybercriminals-trade-and-exploit-your-data).

Download the PDF and save it as `data/reports/iocta_report.pdf`. The source file is intentionally excluded from Git. Streamlit attempts to download the public PDF if the local file is absent. The API and Docker Compose setup expect the local file at that path; provide it before starting them.

## Run locally (Windows PowerShell)

From the repository root, create an environment and install the base dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the report at `data/reports/iocta_report.pdf`. Start the API and UI in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The API documentation is at `http://127.0.0.1:8000/docs`; the UI is at `http://localhost:8501`.

## Configuration

| Variable | Purpose |
| --- | --- |
| `RETRIEVER_BACKEND` | `tfidf` by default; `semantic` or `hybrid` requires `requirements-ml.txt`. |
| `ANSWER_MODE` | `extractive` by default; `llm` requires `requirements-llm.txt` and `OPENAI_API_KEY`. |
| `OPENAI_API_KEY` | OpenAI credential for optional LLM answer generation. Supply it through the environment or an untracked local `.env`; never commit it. |
| `OPENAI_MODEL` | Model name for optional OpenAI generation. |
| `RELOAD_ADMIN_KEY` | Optional key to protect the API document reload endpoint. |
| `DATABASE_URL` | Optional telemetry database connection; SQLite is used by default. Compose configures PostgreSQL. |

For local TF-IDF and extractive use, the base `requirements.txt` is sufficient. Install optional features separately as needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt   # semantic/hybrid retrieval
.\.venv\Scripts\python.exe -m pip install -r requirements-llm.txt  # optional OpenAI generation
```

The Docker build supports one optional dependency mode at a time: set `OPTIONAL_DEPENDENCIES=ml` or `OPTIONAL_DEPENDENCIES=llm` in `.env`, then rebuild. These modes cannot be combined in one image. For LLM use, also set `ANSWER_MODE=llm` and supply `OPENAI_API_KEY` securely.

## Docker Compose

1. Copy `.env.example` to `.env` and replace its placeholder secrets with locally generated values.
2. Put `iocta_report.pdf` in `data/reports/`.
3. Run `docker compose up --build` from the repository root.

Compose starts the UI, API, and PostgreSQL telemetry database. Do not commit `.env` or API credentials.

## Tests

The complete suite includes semantic and hybrid retrieval tests, so install both dependency sets before running it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ml.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover TF-IDF and optional semantic/hybrid retrieval, answer selection and citation validation, agent routing and bounded memory, and SQLite telemetry. LLM tests mock the provider; they do not make live OpenAI requests. A PostgreSQL integration test is opt-in through `TEST_DATABASE_URL`; it is skipped when that variable is not configured and has not been verified against a configured test database.

## Scope and limitations

The included retrieval evaluation uses five questions with preliminary expected-page labels; it is not a validated benchmark, and no performance result is claimed. PDF extraction may lose table structure or formatting. Citation IDs establish that referenced passages exist, not that generated statements follow from them. Session context is bounded process memory and resets when the API restarts. This is a research prototype, not a production intelligence platform; verify claims against the cited source pages.

The project demonstrates practical work with document ingestion, retrieval and evaluation, evidence-linked answer generation, Python APIs and UI integration, bounded conversational state, telemetry, testing, and containerized development. These descriptions refer to implemented prototype components, not production deployment or independently validated outcomes.

## License

No license has been specified. Until one is added, do not assume the repository grants permission to reuse or redistribute the code.
