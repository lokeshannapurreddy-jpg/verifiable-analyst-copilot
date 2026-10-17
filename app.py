"""Premium Streamlit interface for Verifiable Analyst Copilot."""

from dataclasses import replace
from pathlib import Path
import html
import re

import streamlit as st

from src.answer import generate_answer
from src.ingest import ingest_directory
from src.rag import EvidenceRetriever


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Verifiable Analyst Copilot",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "data" / "reports"

EUROPOL_TITLE = (
    "Steal, Deal and Repeat: How Cybercriminals Trade "
    "and Exploit Your Data — IOCTA 2025"
)
EUROPOL_URL = (
    "https://www.europol.europa.eu/publication-events/"
    "main-reports/steal-deal-and-repeat-how-cybercriminals-"
    "trade-and-exploit-your-data"
)

SAMPLE_QUESTIONS = [
    "What cyber threats do hybrid threat actors pose?",
    "How do cybercriminals trade and exploit stolen data?",
    "What techniques do cybercriminals use to gain access?",
    "What risks do cyber threats pose to organisations?",
]


# ---------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
      --bg: #07111f;
      --panel: #0d1b2d;
      --panel2: #102239;
      --line: rgba(148, 180, 219, .16);
      --text: #edf5ff;
      --muted: #9bb0ca;
      --cyan: #69e0d0;
      --blue: #72a7ff;
      --violet: #b5a0ff;
    }

    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }
    .stApp {
      background:
        radial-gradient(ellipse at 12% 0%, rgba(47, 104, 170, .19), transparent 34%),
        radial-gradient(ellipse at 88% 5%, rgba(79, 65, 153, .15), transparent 30%),
        var(--bg);
      color: var(--text);
    }
    [data-testid="stHeader"] {
      background: rgba(7, 17, 31, .75);
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0b192b 0%, #091422 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div {
      padding-top: 1.4rem;
    }
    .block-container {
      max-width: 1450px;
      padding-top: 2.1rem;
      padding-bottom: 3rem;
    }
    h1, h2, h3 {
      font-family: 'Space Grotesk', sans-serif !important;
      color: var(--text) !important;
      letter-spacing: -.035em;
    }
    p, label, .stMarkdown {
      color: var(--text);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 4px 0 20px;
    }
    .brand-icon {
      width: 44px; height: 44px;
      display: grid; place-items: center;
      border-radius: 14px;
      font-size: 22px;
      color: #07111f;
      background: linear-gradient(135deg, var(--cyan), #8ab8ff);
      box-shadow: 0 8px 30px rgba(105, 224, 208, .16);
    }
    .brand-name {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 16px; font-weight: 700; line-height: 1.2;
      color: var(--text);
    }
    .brand-sub {
      font-size: 11px; color: var(--muted);
      margin-top: 4px; letter-spacing: .08em; text-transform: uppercase;
    }
    .eyebrow {
      color: var(--cyan);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .16em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .hero {
      position: relative;
      overflow: hidden;
      border: 1px solid rgba(124, 169, 226, .22);
      border-radius: 24px;
      padding: 30px 32px;
      margin-bottom: 22px;
      background:
        radial-gradient(circle at 90% 10%, rgba(114,167,255,.20), transparent 30%),
        linear-gradient(120deg, rgba(17,42,69,.97), rgba(13,27,45,.95));
      box-shadow: 0 18px 55px rgba(0,0,0,.16);
    }
    .hero h1 {
      font-size: clamp(29px, 3vw, 42px);
      line-height: 1.12;
      margin: 0 0 12px;
    }
    .hero p {
      color: #b8cbe2;
      max-width: 740px;
      font-size: 15px;
      line-height: 1.75;
      margin: 0;
    }
    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(105,224,208,.25);
      background: rgba(105,224,208,.08);
      color: #a8f3e8;
      border-radius: 100px;
      padding: 7px 11px;
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 17px;
    }
    .metric {
      background: linear-gradient(145deg, rgba(16,34,57,.96), rgba(12,26,43,.96));
      border: 1px solid var(--line);
      border-radius: 17px;
      padding: 18px 19px;
      min-height: 116px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .10em;
    }
    .metric-value {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 29px;
      font-weight: 700;
      margin: 8px 0 4px;
      color: var(--text);
    }
    .metric-note {
      font-size: 11px;
      color: #8fa9c6;
    }
    .section-heading {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 21px;
      font-weight: 700;
      letter-spacing: -.025em;
      margin: 30px 0 5px;
      color: var(--text);
    }
    .section-sub {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 16px;
      line-height: 1.6;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 21px;
      background: rgba(13,27,45,.88);
      margin-bottom: 15px;
    }
    .answer-panel {
      border: 1px solid rgba(105,224,208,.25);
      border-radius: 19px;
      padding: 23px;
      background: linear-gradient(135deg, rgba(15,48,57,.68), rgba(13,27,45,.96));
      margin: 10px 0 20px;
    }
    .answer-label {
      color: var(--cyan);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }
    .answer-text {
      color: #e9f5ff;
      font-size: 15px;
      line-height: 1.85;
      white-space: pre-wrap;
    }
    .evidence-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }
    .evidence-title {
      font-weight: 700;
      font-size: 14px;
      color: var(--text);
    }
    .pill {
      display: inline-block;
      border-radius: 8px;
      padding: 5px 8px;
      background: rgba(114,167,255,.12);
      border: 1px solid rgba(114,167,255,.2);
      color: #b9d3ff;
      font-size: 10px;
      font-weight: 700;
      white-space: nowrap;
    }
    .evidence-text {
      color: #c2d2e5;
      font-size: 13px;
      line-height: 1.8;
      overflow-wrap: anywhere;
    }
    .small-muted {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.7;
    }
    .sidebar-card {
      border: 1px solid var(--line);
      border-radius: 13px;
      padding: 13px;
      background: rgba(16,34,57,.68);
      margin: 10px 0;
    }
    .sidebar-label {
      font-size: 10px; color: var(--muted);
      text-transform: uppercase; letter-spacing: .12em;
      font-weight: 700; margin-bottom: 7px;
    }
    .status-dot {
      display: inline-block; width: 7px; height: 7px;
      border-radius: 50%; background: var(--cyan);
      margin-right: 7px; box-shadow: 0 0 12px rgba(105,224,208,.55);
    }
    .stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"],
    .stSlider, [data-baseweb="textarea"] {
      border-radius: 12px !important;
    }
    .stTextArea textarea {
      background: #091727 !important;
      color: var(--text) !important;
      border: 1px solid #28415d !important;
      line-height: 1.6 !important;
    }
    div[data-baseweb="select"] > div {
      background: #091727 !important;
      border-color: #28415d !important;
    }
    .stButton button, .stFormSubmitButton button, .stDownloadButton button {
      border-radius: 11px !important;
      min-height: 43px;
      font-weight: 700 !important;
      border: 1px solid rgba(105,224,208,.25) !important;
      background: linear-gradient(110deg, #69e0d0, #8bbaff) !important;
      color: #07111f !important;
      transition: transform .15s ease, filter .15s ease;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
      filter: brightness(1.06);
      transform: translateY(-1px);
    }
    a { color: #8ab8ff !important; }
    hr { border-color: var(--line) !important; }
    [data-testid="stExpander"] {
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(13,27,45,.72);
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    .footer {
      color: #7188a4; text-align: center;
      font-size: 11px; margin-top: 35px; line-height: 1.8;
    }
    @media (max-width: 760px) {
      .block-container { padding: 1rem .85rem 2rem; }
      .hero { padding: 23px 19px; border-radius: 18px; }
      .metric { padding: 14px; min-height: 105px; }
      .metric-value { font-size: 24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load the report and build the local index once per Streamlit session.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Reading the report and preparing evidence search…")
def load_retriever():
    if not REPORTS_DIR.exists():
        raise FileNotFoundError(
            f"Reports folder was not found: {REPORTS_DIR}"
        )

    chunks = ingest_directory(REPORTS_DIR)
    enriched_chunks = []

    for chunk in chunks:
        if chunk.source.lower() == "iocta_report.pdf":
            chunk = replace(
                chunk,
                title=EUROPOL_TITLE,
                url=EUROPOL_URL,
            )
        enriched_chunks.append(chunk)

    return EvidenceRetriever(enriched_chunks)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-icon">⌕</div>
          <div>
            <div class="brand-name">Verifiable<br>Analyst Copilot</div>
            <div class="brand-sub">Evidence intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Research workspace", "About this project"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    try:
        retriever = load_retriever()
        indexed_count = len(retriever.chunks)
        index_ready = True
        load_error = ""
    except Exception as exc:
        retriever = None
        indexed_count = 0
        index_ready = False
        load_error = str(exc)

    status_text = "Index ready" if index_ready else "Index unavailable"
    status_color = "#69e0d0" if index_ready else "#ffb4a2"
    st.markdown(
        f"""
        <div class="sidebar-card">
          <div class="sidebar-label">System status</div>
          <div style="font-weight:700;color:{status_color};font-size:13px;">
            <span class="status-dot" style="background:{status_color};"></span>
            {status_text}
          </div>
          <div class="small-muted" style="margin-top:7px;">
            Local PDF search · TF-IDF retrieval
          </div>
        </div>
        <div class="sidebar-card">
          <div class="sidebar-label">Knowledge source</div>
          <div style="font-size:13px;font-weight:700;line-height:1.5;color:#edf5ff;">
            Europol IOCTA 2025
          </div>
          <div class="small-muted" style="margin-top:5px;">
            Public cybercrime assessment
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="small-muted">Designed for traceable research, not unsupported guesses.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Shared hero and metrics
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">✦ SOURCE-GROUNDED RESEARCH</div>
      <h1>Turn cybercrime reports<br>into traceable answers.</h1>
      <p>
        Ask a question in plain language. Explore relevant passages from
        public reports, inspect the original evidence, and follow each
        finding back to its source page.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "PASSAGES INDEXED", f"{indexed_count:,}", "Searchable report excerpts"),
    (m2, "SOURCE DOCUMENTS", "1" if indexed_count else "0", "Public report collection"),
    (m3, "ANSWER METHOD", "Extractive", "Selected from report text"),
    (m4, "TRACEABILITY", "Page-level", "Evidence keeps page metadata"),
]
for col, label, value, note in metrics:
    with col:
        st.markdown(
            f"""
            <div class="metric">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if not index_ready:
    st.error("The report index could not be loaded.")
    st.code(load_error or "Unknown indexing error")
    st.info(
        "Check that the PDF exists inside data/reports and that the "
        "project's Python environment has its requirements installed."
    )
    st.stop()


if page == "About this project":
    st.markdown('<div class="section-heading">Built for transparent research</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A compact research prototype focused on evidence retrieval and traceability.</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            <div class="panel">
              <div class="evidence-title">How it works</div>
              <p class="small-muted">
                1. Extract text from a public PDF.<br>
                2. Split text into page-linked passages.<br>
                3. Retrieve relevant passages with TF-IDF similarity.<br>
                4. Select matching complete sentences and preserve page citations.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="panel">
              <div class="evidence-title">Important limitations</div>
              <p class="small-muted">
                This prototype uses lexical retrieval and extractive answer
                selection, not a generative language model. It may miss relevant
                evidence, and retrieved statements should be checked against
                the original report. A relevance score is not a truth score.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("**Primary source**")
    st.markdown(f"[{EUROPOL_TITLE}]({EUROPOL_URL})")
    st.stop()


# ---------------------------------------------------------------------
# Research workspace
# ---------------------------------------------------------------------
st.markdown('<div class="section-heading">Research workspace</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Start with a suggested question or write your own. Results stay grounded in the indexed report.</div>',
    unsafe_allow_html=True,
)

with st.form("research_form", clear_on_submit=False):
    preset = st.selectbox(
        "Question starter",
        ["Choose a suggested question…"] + SAMPLE_QUESTIONS,
        index=0,
        help="Selecting a suggestion will fill the question field after you submit.",
    )
    default_question = st.session_state.get("question_text", "")
    question = st.text_area(
        "Your research question",
        value=default_question,
        height=110,
        max_chars=1000,
        placeholder="For example: What cyber threats do hybrid threat actors pose?",
        help="Ask one clear question. Answers are based only on indexed report passages.",
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        top_k = st.slider(
            "Evidence passages",
            min_value=1,
            max_value=10,
            value=3,
            help="How many relevant passages to retrieve before selecting answer sentences.",
        )
    with c2:
        st.markdown(
            '<div class="small-muted" style="padding-top:31px;">'
            'Tip: ask about a specific threat, technique, impact, or type of data.'
            '</div>',
            unsafe_allow_html=True,
        )
    submitted = st.form_submit_button("⌕  Find evidence and answer", use_container_width=True)

if submitted:
    chosen_question = question.strip()
    if preset != "Choose a suggested question…" and not chosen_question:
        chosen_question = preset
    elif preset != "Choose a suggested question…" and chosen_question == default_question:
        # A newly selected preset should be useful even if the text box was untouched.
        chosen_question = preset

    if len(chosen_question) < 3:
        st.warning("Enter a question with at least 3 characters.")
    else:
        st.session_state["question_text"] = chosen_question
        with st.spinner("Searching the report and selecting traceable evidence…"):
            results = retriever.search(chosen_question, top_k=top_k)
            answer = generate_answer(chosen_question, results)
        st.session_state["last_question"] = chosen_question
        st.session_state["last_answer"] = answer
        st.session_state["last_results"] = results
        st.session_state["last_top_k"] = top_k


# ---------------------------------------------------------------------
# Results persist across reruns and remain visible while the user explores.
# ---------------------------------------------------------------------
if st.session_state.get("last_question"):
    st.markdown("---")
    st.markdown('<div class="section-heading">Research results</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Query: <strong>{html.escape(st.session_state["last_question"])}</strong></div>',
        unsafe_allow_html=True,
    )

    results = st.session_state.get("last_results", [])
    answer = st.session_state.get("last_answer", "")

    if results:
        st.markdown(
            f"""
            <div class="answer-panel">
              <div class="answer-label">✦ Evidence-based answer</div>
              <div class="answer-text">{html.escape(answer)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Evidence passages", len(results))
        r2.metric("Pages represented", len({r.chunk.page for r in results}))
        r3.metric("Retrieval method", "TF-IDF")
        st.markdown('<div class="section-heading">Source evidence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Open a passage to inspect the retrieved text and follow the original source.</div>',
            unsafe_allow_html=True,
        )

        for i, result in enumerate(results, start=1):
            chunk = result.chunk
            title = chunk.title or chunk.source
            with st.expander(
                f"Evidence {i}  ·  Page {chunk.page}  ·  Relevance {result.score:.3f}",
                expanded=(i == 1),
            ):
                st.markdown(
                    f"""
                    <div class="evidence-top">
                      <div class="evidence-title">{html.escape(title)}</div>
                      <span class="pill">PAGE {chunk.page}</span>
                    </div>
                    <div class="evidence-text">{html.escape(chunk.text)}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Local file: {chunk.source} · Relevance score: {result.score:.4f}")
                if chunk.url:
                    st.markdown(f"[Open the official source ↗]({chunk.url})")
    else:
        st.warning(
            "No sufficiently relevant passages were found in the indexed report. "
            "Try a more specific question or use one of the suggested questions."
        )
        st.markdown(
            f"""
            <div class="answer-panel">
              <div class="answer-label">No answer generated</div>
              <div class="answer-text">{html.escape(answer)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("---")
st.markdown(
    f"""
    <div class="footer">
      VERIFIABLE ANALYST COPILOT · RESEARCH PROTOTYPE<br>
      TF-IDF retrieval · Extractive answer selection · Page-level source traceability<br>
      Primary source: <a href="{EUROPOL_URL}" target="_blank">Europol IOCTA 2025</a>
    </div>
    """,
    unsafe_allow_html=True,
)
