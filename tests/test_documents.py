import pytest
from tests.conftest import make_pdf

from core import documents, rag_engine, tracker, vector_store
from tests.test_memory import MEETING_1, add_to

PAPER = [
    "Bonsai builds a shallow and diverse label tree for extreme multi-label classification.",
    "",  # an image-only page: skipped
    "On EURLex-4K Bonsai reaches precision at 1 of 82.8 using a branching factor of 100.",
]


def test_extract_pages_keeps_page_numbers_and_skips_empty_pages(tmp_path):
    path = make_pdf(tmp_path / "bonsai.pdf", PAPER)
    pages = documents.extract_pages(path)
    assert [p["page"] for p in pages] == [1, 3]
    assert "branching factor of 100" in pages[1]["text"]


def test_title_from_metadata_or_file_name(tmp_path):
    assert documents.pdf_title(make_pdf(tmp_path / "x.pdf", ["hi"], title="Bonsai Diverse Label Trees")) == "Bonsai Diverse Label Trees"
    assert documents.pdf_title(make_pdf(tmp_path / "my_notes.pdf", ["hi"])) == "my notes"


def test_chunks_never_span_pages():
    pages = [{"page": 1, "text": "a " * 1200}, {"page": 2, "text": "b " * 50}]
    chunks = documents.chunk_pages(pages)
    assert {c["page"] for c in chunks} == {1, 2}
    assert all(set(c["text"].split()) == ({"a"} if c["page"] == 1 else {"b"}) for c in chunks)


def test_scanned_pdf_gives_a_clear_error(tmp_path, fake_llm, memory):
    path = make_pdf(tmp_path / "scan.pdf", ["", ""])
    with pytest.raises(documents.NoTextInPdfError, match="scanned"):
        documents.process_pdf(path, workspace="w", progress=lambda *_: None)


def test_pdf_and_meetings_answer_together_with_page_citations(tmp_path, fake_llm, memory):
    ws = vector_store.workspace_id("me")
    add_to(ws, "Thesis", MEETING_1, "2026-10-01")
    doc = documents.process_pdf(make_pdf(tmp_path / "bonsai.pdf", PAPER), ws, project="Thesis",
                                doc_date="2026-10-05", progress=lambda *_: None)
    assert doc["source_type"] == "pdf" and doc["page_count"] == 2

    # listed alongside meetings, but never treated as a meeting by the follow-up tracker
    types = sorted(s["source_type"] for s in vector_store.list_sources(ws, project="Thesis"))
    assert types == ["meeting", "pdf"]
    assert all(src["source_type"] == "meeting" for src, _ in tracker.open_items(ws, "x", "2099-01-01", "Thesis"))

    rag_engine.answer_question(ws, "What branching factor does Bonsai use?", project="Thesis")
    context = fake_llm.calls[-1][0]
    assert "Document (project: Thesis): bonsai · 05 Oct 2026 · p. 3" in context
    assert "Meeting (project: Thesis)" in context  # the meeting about the baseline is retrieved too

    hits = vector_store.search(ws, "branching factor", project="Thesis", source_ids=[doc["source_id"]])
    assert hits[0]["page"] == 3 and rag_engine.describe(hits[0]).endswith("p. 3")


def test_pdf_stays_in_its_project(tmp_path, fake_llm, memory):
    ws = vector_store.workspace_id("me")
    documents.process_pdf(make_pdf(tmp_path / "b.pdf", PAPER), ws, project="Thesis", progress=lambda *_: None)
    assert vector_store.search(ws, "Bonsai label tree", project="BrickMind") == []
    assert vector_store.list_sources(ws, project="BrickMind") == []
