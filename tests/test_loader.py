from app.services.document_loader import load_and_split_documents
from app.services.chunking import SectionAwareChunker, SectionDocument


def test_loader_splits_sample_documents_with_metadata():
    chunks = load_and_split_documents()

    assert chunks
    assert all("subject" in chunk.metadata for chunk in chunks)
    assert any(chunk.metadata["subject"] == "mathematics" for chunk in chunks)
    assert all(chunk.metadata["chunk_strategy"] == "section_aware_recursive" for chunk in chunks)
    assert any("section_path" in chunk.metadata for chunk in chunks)


def test_section_aware_chunking_preserves_pdf_numbered_sections():
    document = SectionDocument(
        page_content=(
            "1.1 Large Numbers Around Us\n"
            "One lakh is written as 100000.\n\n"
            "1.2 Comparing Numbers\n"
            "We compare numbers using place value."
        ),
        metadata={
            "source": "sample.pdf",
            "file_name": "sample.pdf",
            "file_type": ".pdf",
            "subject": "mathematics",
            "grade": 7,
            "page_number": 1,
        },
    )

    chunks = SectionAwareChunker(chunk_size=120, chunk_overlap=10).split(document)

    section_titles = {chunk.metadata["section_title"] for chunk in chunks}
    assert "Large Numbers Around Us" in section_titles
    assert "Comparing Numbers" in section_titles
    assert all(chunk.metadata["section_path"].startswith("mathematics") for chunk in chunks)
