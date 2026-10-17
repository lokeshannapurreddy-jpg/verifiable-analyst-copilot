# Verifiable Analyst Copilot

A source-grounded research prototype for retrieving evidence from public cybercrime reports while preserving source information for verification.

## Project Objective

This project demonstrates how information retrieval can help analysts find relevant passages in lengthy cybercrime reports without losing the context needed to verify the original source.

## Features

* Extracts text from PDF reports.
* Splits report text into searchable passages.
* Uses TF-IDF and cosine similarity for evidence retrieval.
* Preserves report titles, PDF page numbers, filenames, and source URLs.
* Provides a REST API using FastAPI.
* Provides a Streamlit user interface.
* Includes automated tests for retrieval and answer selection.

## Technology Stack

Python, scikit-learn, FastAPI, Uvicorn, Streamlit, PyPDF, pytest, and HTTPX.

## Project Structure

| File or directory | Purpose                                  |
| ----------------- | ---------------------------------------- |
| `src/ingest.py`   | PDF text extraction and passage creation |
| `src/rag.py`      | TF-IDF indexing and evidence retrieval   |
| `src/answer.py`   | Answer selection and text cleanup        |
| `api.py`          | REST API endpoints                       |
| `app.py`          | Streamlit interface                      |
| `tests/`          | Automated tests                          |
| `docs/SOURCES.md` | Data source documentation                |
| `data/reports/`   | Local PDF reports                        |

## Data Source

The demonstration uses Europol's 2025 Internet Organised Crime Threat Assessment report:

**Steal, Deal and Repeat: How Cybercriminals Trade and Exploit Your Data**

Official publication:

https://www.europol.europa.eu/publication-events/main-reports/steal-deal-and-repeat-how-cybercriminals-trade-and-exploit-your-data

Download the report from the official publication page and save it as `data/reports/iocta_report.pdf`.

The PDF is required to populate the local search index. Keep it out of Git commits.

## Setup on Windows

Run these commands from the project root directory in PowerShell.

### 1. Create the virtual environment

```powershell
python -m venv .venv
```

If `.venv` already exists, you do not need to create it again.

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Verify the report

Confirm that `data/reports/iocta_report.pdf` exists.

### 4. Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8081
```

Keep this terminal running.

API documentation: http://127.0.0.1:8081/docs

Health endpoint: http://127.0.0.1:8081/health

### 5. Start the Streamlit interface

Open a second PowerShell terminal in the project directory and run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address displayed by Streamlit in your browser.

### 6. Run automated tests

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Passing tests verify selected application behaviors but do not guarantee that every retrieved passage is relevant or every extracted statement is accurate.

## Example Questions

After the report has been indexed, try questions such as:

* What cybercrime threats are discussed in the report?
* What does the report say about trading stolen data?
* Which threat actors and attack methods are described?

Search quality depends on the question and the text extracted from the PDF.

## Limitations

* This is a research prototype, not a production-grade intelligence platform.
* Retrieval uses TF-IDF and cosine similarity.
* Answer selection uses retrieved passages and does not independently fact-check claims.
* The application does not generate synthesized answers using a large language model.
* PDF extraction may introduce broken sentences, formatting artifacts, or encoding problems.
* Retrieved claims should be checked against the original report pages.
* Results depend on successful PDF extraction and indexing.

## Responsible Use

This project is intended for research, education, and analysis of publicly available cybercrime reports. It does not independently establish the truth of claims in the source material.

## License

No license has been specified. Until a license is added, do not assume that the repository grants permission to reuse or redistribute the code.
