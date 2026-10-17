"""PDF ingestion with page-level source tracking."""

from pathlib import Path

from pypdf import PdfReader

from src.rag import DocumentChunk


def ingest_pdf(
    pdf_path: str | Path,
    title: str | None = None,
    url: str = "",
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Extract PDF text and return chunks with source metadata."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    reader = PdfReader(str(path))
    document_title = title or path.stem.replace("_", " ").replace("-", " ")
    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if not text:
            continue

        for start in range(0, len(text), step):
            passage = text[start : start + chunk_size].strip()

            if passage:
                chunks.append(
                    DocumentChunk(
                        text=passage,
                        source=path.name,
                        page=page_number,
                        title=document_title,
                        url=url,
                    )
                )

            if start + chunk_size >= len(text):
                break

    return chunks


def ingest_directory(
    directory: str | Path,
) -> list[DocumentChunk]:
    """Ingest all PDF files in a directory."""

    folder = Path(directory)

    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    all_chunks: list[DocumentChunk] = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        all_chunks.extend(ingest_pdf(pdf_path))

    return all_chunks
