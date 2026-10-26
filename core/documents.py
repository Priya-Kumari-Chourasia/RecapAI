"""PDF documents (papers, slides, notes) stored next to meetings in a project.

Text is extracted page by page, split into chunks that remember their page
number, and saved to the same Qdrant collections as meeting transcripts, so
one question can be answered from meetings and documents together, with
sources like "Bonsai paper · p. 4".
"""
import os
from datetime import date as date_cls
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import vector_store
from core.factcheck import run_fact_check
from core.llm import complete

CHUNK_CHARS = 900
SUMMARY_CHARS = 24000  # summarise the first ~24k characters; keeps Groq free-tier usage small


class NoTextInPdfError(RuntimeError):
    """The PDF has no extractable text (usually a scanned document)."""


def extract_pages(path: str) -> list[dict]:
    """[{"page": 1, "text": "..."}, ...] for pages that contain text."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            pages.append({"page": number, "text": text})
    return pages


def pdf_title(path: str) -> str:
    """Title from the PDF's metadata if it looks real, else the file name."""
    from pypdf import PdfReader

    try:
        meta_title = (PdfReader(path).metadata or {}).get("/Title")
    except Exception:  # noqa: BLE001 - broken metadata shouldn't block the upload
        meta_title = None
    meta_title = " ".join(str(meta_title or "").split())
    if 3 <= len(meta_title) <= 120 and not meta_title.lower().startswith(("microsoft word", "untitled")):
        return meta_title
    return os.path.splitext(os.path.basename(path))[0].replace("_", " ")


def chunk_pages(pages: list[dict], chunk_chars: int = CHUNK_CHARS) -> list[dict]:
    """Split each page separately, so every chunk belongs to exactly one page."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_chars, chunk_overlap=120)
    return [
        {"text": piece, "page": p["page"]}
        for p in pages
        for piece in splitter.split_text(p["text"])
    ]


def summarize_document(pages: list[dict]) -> str:
    text = "\n".join(f"[p. {p['page']}] {p['text']}" for p in pages)[:SUMMARY_CHARS]
    return complete(
        "You summarize documents (research papers, slides, reports, notes). Write 4-8 markdown bullet "
        "points covering what the document is about and its key points, methods, numbers and conclusions. "
        "Keep technical terms exact. Do not invent anything. Never ask the user questions.",
        text,
    )


def process_pdf(
    path: str,
    workspace: str,
    project: str = vector_store.DEFAULT_PROJECT,
    title: str | None = None,
    doc_date: str | None = None,
    progress=print,
) -> dict:
    progress("📄 Reading PDF…")
    pages = extract_pages(path)
    if not pages:
        raise NoTextInPdfError(
            "No text could be read from this PDF. It's probably a scanned document (images of pages); "
            "run it through OCR first, or upload a text-based PDF."
        )
    progress(f"📋 Summarising {len(pages)} page(s)…")
    summary = summarize_document(pages)
    record = {
        "workspace": workspace,
        "project": project,
        "source_id": vector_store.new_source_id(),
        "source_type": "pdf",
        "title": title or pdf_title(path),
        "date": doc_date or date_cls.today().isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin": "pdf",
        "origin_ref": os.path.basename(path),
        "page_count": len(pages),
        "word_count": sum(len(p["text"].split()) for p in pages),
        "summary": summary,
        "action_items": [],
        "key_decisions": [],
        "open_questions": [],
    }
    chunks = chunk_pages(pages)
    progress(f"🧠 Saving {len(chunks)} passages to your project memory…")
    vector_store.save_source(record, chunks)
    run_fact_check(record, progress)
    return record
