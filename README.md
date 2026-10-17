# Verifiable Analyst Copilot

A Python prototype for retrieving source-grounded evidence from public cybercrime reports.

## Project Objective

Explore how information retrieval can help analysts locate relevant passages in public cybercrime reports while preserving source information for verification.

## Current Features

- Extracts text from PDF reports.
- Splits extracted text into searchable passages.
- Uses TF-IDF and cosine similarity to retrieve relevant evidence.
- Preserves report titles, PDF page numbers, source filenames, and URLs.
- Provides a FastAPI backend and Streamlit interface.
- Includes automated tests for retrieval behavior.

## Technology Stack

Python, scikit-learn, FastAPI, Streamlit, PyPDF, and pytest.

## Architecture

- `src/ingest.py` — PDF extraction and passage creation.
- `src/rag.py` — TF-IDF indexing and evidence retrieval.
- `api.py` — REST API endpoints.
- `app.py` — Streamlit interface.
- `tests/` — automated tests.
- `docs/SOURCES.md` — data-source documentation.

## Data Source

The demonstration uses Europol's 2025 Internet Organised Crime Threat Assessment report:

**Steal, Deal and Repeat: How Cybercriminals Trade and Exploit Your Data**

Official publication:

https://www.europol.europa.eu/publication-events/main-reports/steal-deal-and-repeat-how-cybercriminals-trade-and-exploit-your-data

Download the report from the official publication page and place it at:

`data/reports/iocta_report.pdf`

The PDF is required to populate the local search index.

## Setup on Windows

### 1. Create a virtual environment

```powershell
python -m venv .venv
