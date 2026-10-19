# INTERPOL AI Development Internship — requirements audit

**Audit date:** 19 October 2026
**Project reviewed:** Verifiable Analyst Copilot, published commit `0ce9aa27b892e90657b15aaefed37d5733ef030e`
**Vacancy:** [Internship - AI Development (INTERPOL Recruitment Hub)](https://209400.jobs2web.com/job/Singapore-Internship-AI-Development/1439937200/)
**General internship conditions:** [INTERPOL Internships](https://www.interpol.int/en/What-you-can-do/Careers/Internships)

This is a comparison of the public vacancy, the supplied candidate profile, and the repository. It is not an employment decision or proof of personal eligibility. The candidate's CV, degree certificate, language evidence, nationality/age evidence, and application documents were not present in the repository.

## Vacancy facts verified

The official Recruitment Hub lists **Internship - AI Development**, duty station **Singapore**, **11 months**, estimated start **1 January 2027**, and application deadline **28 October 2026**. It places the role in the AI Acceleration team within the ICT Executive Directorate.

The vacancy says the team builds end-to-end AI applications, including agents with tools and memory, LLM integrations, and RAG. Intern duties are to prototype AI/LLM features; support RAG and agent memory, including data preparation and model integration; contribute to Python/FastAPI and, *where relevant*, React/TypeScript; and help test, document, and deploy AI components.

The notice lists conscientiousness, self-discipline, teamwork, cooperation, communication, self-motivation, and ability to work internationally. It asks for a recent Bachelor's or Master's graduate in Computer Science, Computer Engineering, or a related field; AI/data-science/software-engineering coursework, projects, or experience; and says hands-on Python is **essential**. It requires English fluency; French, Spanish, or Arabic is an **asset**. The specific-skills section describes Python and AI/LLM exposure, REST/Git/software practices, interest in applied AI, and database exposure (including PostgreSQL and/or vector databases); it explicitly calls AI/LLM exposure and database exposure an **asset**, while Python is explicitly essential. React/TypeScript is conditional (“where relevant”), not a universal qualification.

INTERPOL's general internship page additionally requires that, on the application date, a candidate is enrolled in or graduated from an accredited post-secondary institution within the past year, is a national of a member country, has reached the age of majority in that country, and is fluent in English. The candidate profile says “graduated in 2025,” which is insufficient to verify the one-year rule: the exact graduation date and application submission date must be compared. The application deadline is not itself the only relevant date if the application is submitted earlier.

## Requirements and evidence matrix

“MET” for a project item means the implementation and cited tests demonstrate a prototype capability; it does not imply production readiness or independent validation. Candidate-profile statements are self-reported in the request and are not independently evidenced here.

| Exact requirement / condition | Classification | Evidence reviewed | Status | Gap and next step | Priority |
| --- | --- | --- | --- | --- | --- |
| Recent Bachelor's/Master's in Computer Science, Computer Engineering, or related field | Vacancy qualification; related-field equivalence is for INTERPOL to determine | Supplied profile: B.Tech Internet of Things, Faculty of Computer Science & Engineering, Malla Reddy University | PARTIALLY MET | The degree is plausibly related, but no certificate/transcript or formal equivalence evidence was available. State degree and relevant coursework accurately in the application. | HIGH |
| Enrolled or graduated from accredited post-secondary institution within the past year on application date | General eligibility | Supplied profile says graduated in 2025; no exact date or supporting document | NOT VERIFIABLE | Confirm exact conferral date and that the institution is accredited; compare with actual submission date. Could be disqualifying if outside the one-year window. | HIGH |
| National of one of INTERPOL's member countries; legal age of majority | General eligibility | No personal evidence in repository/profile | NOT VERIFIABLE | Verify both against official conditions and have supporting documents ready. | HIGH |
| Fluent in English | Vacancy and general eligibility | Candidate profile gives no language certification or work sample | NOT VERIFIABLE | Confirm fluency and present it truthfully; no certificate is asserted by this audit. | HIGH |
| Hands-on Python (essential) | Explicit essential qualification | Python implementation in `src/`; tests under `tests/`; profile self-reports Python | PARTIALLY MET | Project demonstrates practical code, but a repository cannot establish the candidate's authorship or interview-level proficiency. Be ready to explain individual contribution and trade-offs. | HIGH |
| Coursework, projects, or experience in AI, data science, or software engineering | Vacancy qualification | This repository implements a small RAG research prototype; profile also self-reports analytics experience | MET (project evidence) | The project is relevant evidence; personal contribution remains candidate-attested. | HIGH |
| Prototype AI/LLM-powered features | Responsibility | `src/answer.py::generate_answer` and `_generate_llm_answer`; `tests/test_answer.py` mocks provider responses and fallback | PARTIALLY MET | LLM path is optional and mocked in tests; no live provider behavior is established by this audit. Clearly describe it as an optional prototype integration. | HIGH |
| Support RAG pipelines, agent memory, data preparation, model integration | Responsibility | `src/ingest.py::ingest_pdf`; `src/rag.py`; `src/agent.py::SessionMemory` and `run_agent`; `tests/test_agent.py`, `tests/test_rag.py`, `tests/test_retrieval_evaluation.py` | MET (prototype) | Page-linked PDF chunking, retrieval, deterministic tool routing, and bounded in-memory follow-up context are implemented. It is not a long-term or persistent memory system, nor an autonomous agent. | HIGH |
| Exposure to RAG, embeddings, LLM APIs (listed as an asset) | Desirable/asset | `src/rag.py::EmbeddingEvidenceRetriever` / `HybridEvidenceRetriever`; optional `requirements-ml.txt`; `src/answer.py`; `requirements-llm.txt` | MET (prototype exposure) | Embedding and provider paths depend on optional packages. Evaluation data is only five manually labeled questions; do not present scores as general performance. | HIGH |
| Backend Python/FastAPI | Responsibility | `api.py` (`/health`, `/ask`, `/analytics`, `/reload`); `tests/test_api.py` | MET (prototype) | API tests are focused unit-level checks; they do not establish deployment readiness. | HIGH |
| React/TypeScript frontend, where relevant | Conditional responsibility | `app.py` implements the user interface with Streamlit; no React/TypeScript files are present | NOT MET (project), not a stated universal qualification | No need to add React solely for this application: the vacancy says “where relevant.” Mention Streamlit accurately and do not claim React experience based on this project. | LOW |
| Testing, documentation, deployment support | Responsibility | `tests/`, `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml`, `README.md`, `docs/PROJECT_WALKTHROUGH.md` | MET (project evidence) | Tests cover core paths, but live OpenAI and configured PostgreSQL verification are distinct from mocked/default tests. | MEDIUM |
| REST, Git, general software development practices | Specific skill | `api.py`, repository history/configuration, CI workflow, tests | MET (project evidence) | Repository demonstrates use; depth of individual Git/team workflow is not independently verifiable. | MEDIUM |
| Database exposure: SQL/PostgreSQL and/or vector DB (asset) | Desirable/asset | `src/analytics.py` SQLAlchemy schema; PostgreSQL Compose service; guarded opt-in integration in `tests/test_analytics.py` | PARTIALLY MET | PostgreSQL is used for telemetry, not vector search. The integration test only runs with `TEST_DATABASE_URL`; it was not run during this audit. Do not claim this audit verified PostgreSQL integration. | MEDIUM |
| Teamwork, cooperation, communication, self-discipline, self-motivation, international working ability | Vacancy requirements | No CV, references, teamwork artifacts, or interview evidence supplied | NOT VERIFIABLE | Address these with truthful examples in the CV/cover letter and interview; code alone does not demonstrate them. | HIGH |
| French, Spanish, or Arabic | Asset | No language information supplied | NOT VERIFIABLE | Include only if applicable and accurately describe proficiency. | LOW |
| Strong interest in applied AI and practical tools | Specific skill | Relevance of this project and supplied internship objective | PARTIALLY MET | The project is consistent with the stated interest; motivation is personal and should be explained in the cover letter. | MEDIUM |

## Repository evidence and limitations

- **Ingestion and retrieval:** `src/ingest.py` extracts per-page PDF text and creates overlapping chunks retaining source/page metadata. `src/rag.py` provides default TF-IDF and optional semantic/hybrid retrieval. The optional embedding package is separately declared in `requirements-ml.txt`. The source PDF is intentionally untracked; `tests/test_retrieval_evaluation.py` skips its PDF-backed fixture when the local report is missing. The clean CI dependency setup installs ML requirements, but does not supply that excluded PDF, so those data-backed evaluation cases are not evidence of CI execution.
- **Answers and citations:** `src/answer.py` defaults to extractive sentence selection with page references. Optional OpenAI synthesis is bounded by a 20-second timeout with retries disabled; missing credentials, provider failures, or invalid citations fall back to extraction. Citation IDs are checked against the retrieved result indices and every answer sentence must contain a citation. This checks citation form and membership, **not** semantic entailment, source truth, or factual correctness. `tests/test_answer.py` uses mocked provider behavior; it is not a live API test.
- **Agent and memory:** `src/agent.py` deterministically routes evidence and index-metadata queries. `SessionMemory` retains a bounded number of clipped question/answer turns in process memory, and `api.py` bounds the number of sessions. It resets on process restart and is not durable or cross-worker memory.
- **API and exposure:** `api.py` offers health, question, analytics, and document reload endpoints. Reload requires `RELOAD_ADMIN_KEY`; `/ask` and `/analytics` are unauthenticated. `docker-compose.yml` binds host ports to loopback for local use. Do not expose this configuration publicly without deployment authentication and controls.
- **Analytics and databases:** `src/analytics.py` stores aggregate telemetry through SQLAlchemy, using SQLite by default and PostgreSQL when configured. It omits raw question/answer columns, but stores deterministic hashes and lengths; hashes should not be represented as anonymization. `tests/test_analytics.py` contains an opt-in local PostgreSQL check restricted to the dedicated `analyst_test` database. It was not run as part of this audit. README records a prior disposable local Compose PostgreSQL run; that historical note is not a fresh verification here.
- **Evaluation:** `src/evaluation.py` contains five questions with binary expected-page sets. `scripts/evaluate_retrieval.py` reports page-deduplicated Recall@3, MRR@3, and nDCG@3. Labels are single-reviewer and small; the evaluation does not measure answer factuality, citation entailment, robustness, or production retrieval quality.
- **Project materials:** README and `docs/PROJECT_WALKTHROUGH.md` describe the prototype and its limitations. Screenshots are present. No CV, cover letter, application form, or original vacancy document was present in the repository. Candidate authorship/contribution is not independently established by repository inspection.

## Highest-value next steps

### Essential before submission

1. **Resolve eligibility first:** verify degree accreditation, exact graduation/conferral date against the one-year rule on the actual application date, nationality, age, and English fluency. The supplied “graduated in 2025” detail alone does not settle eligibility for an October 2026 application.
2. **Prepare the application evidence:** provide a concise CV and tailored cover letter that explain personal contribution to this project, relevant coursework, Python examples, and truthful teamwork/communication examples. State the B.Tech field exactly as awarded.
3. **Prepare to defend technical trade-offs:** explain why TF-IDF is default, what the five-query evaluation can and cannot show, why citation IDs do not establish entailment, and how fallback and bounded memory work.

### Valuable if time permits

- Ensure the public repository's current CI result is green, and distinguish mocked LLM tests from live provider checks and the optional PostgreSQL integration.
- Consider expanding evaluation questions and obtaining independent relevance judgments after the application deadline; do not present the current tiny set as a benchmark.

### Optional; not justified as pre-deadline work

- Do not add React, a vector database, a new agent framework, or extra model dependencies solely to mirror technologies mentioned in the vacancy. React is conditional, and vector databases are only listed as an asset.

## Overall assessment

The project provides credible, role-relevant **prototype evidence** for Python, data preparation, RAG retrieval, optional embeddings and LLM integration, citation-reference checks, bounded agent-style memory, FastAPI, SQL telemetry, testing, documentation, and local containerization. It does **not** establish production use, live OpenAI behavior, robust citation entailment, large-scale retrieval quality, durable agent memory, or the candidate's personal authorship and professional/soft skills. Eligibility and several personal qualifications remain unverified; resolve them independently before treating the application as eligible.
