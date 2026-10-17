"""Streamlit interface for the Verifiable Analyst Copilot."""

import streamlit as st

from src.ingest import ingest_directory
from src.rag import EvidenceRetriever


st.set_page_config(
    page_title="Verifiable Analyst Copilot",
    page_icon="🔎",
    layout="wide",
)

st.title("Verifiable Analyst Copilot")
st.caption(
    "Explore public cybercrime reports with source-traceable "
    "evidence retrieval."
)

st.info(
    "Prototype limitation: this application retrieves relevant "
    "passages. It does not generate a synthesized AI answer or "
    "verify the truth of retrieved claims."
)

REPORTS_DIRECTORY = "data/reports"


@st.cache_resource
def load_retriever():
    """Index PDF reports and cache the retrieval engine."""
    chunks = ingest_directory(REPORTS_DIRECTORY)

    for chunk in chunks:
        if chunk.source == "iocta_report.pdf":
            from dataclasses import replace

            index = chunks.index(chunk)
            chunks[index] = replace(
                chunk,
                title=(
                    "Steal, Deal and Repeat: How Cybercriminals "
                    "Trade and Exploit Your Data — IOCTA 2025"
                ),
                url=(
                    "https://www.europol.europa.eu/publication-events/"
                    "main-reports/steal-deal-and-repeat-how-"
                    "cybercriminals-trade-and-exploit-your-data"
                ),
            )

    return EvidenceRetriever(chunks)


try:
    retriever = load_retriever()
    st.success(f"Indexed {len(retriever.chunks)} report passages.")
except FileNotFoundError:
    st.error(
        "The reports directory was not found. Create "
        "'data/reports' and add a PDF report."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the reports: {exc}")
    st.stop()


with st.form("question_form"):
    question = st.text_area(
        "Ask a question about the indexed report",
        placeholder="How do cybercriminals acquire personal data?",
        max_chars=1000,
    )
    top_k = st.slider(
        "Maximum evidence passages",
        min_value=1,
        max_value=10,
        value=3,
    )
    submitted = st.form_submit_button("Find evidence")


if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        results = retriever.search(question, top_k=top_k)

        if not results:
            st.warning(
                "No sufficiently relevant evidence was found. "
                "Try rephrasing the question or consult the original report."
            )
        else:
            st.subheader(f"Retrieved evidence ({len(results)} passages)")

            for number, result in enumerate(results, start=1):
                chunk = result.chunk

                with st.expander(
                    f"{number}. {chunk.title} — PDF page {chunk.page}"
                ):
                    st.write(chunk.text)
                    st.caption(
                        f"Source file: {chunk.source} | "
                        f"Relevance score: {result.score:.3f}"
                    )

                    if chunk.url:
                        st.markdown(
                            f"[Open official source]({chunk.url})"
                        )

st.divider()
st.caption(
    "Research prototype | TF-IDF retrieval | "
    "Source: Europol IOCTA 2025"
)
