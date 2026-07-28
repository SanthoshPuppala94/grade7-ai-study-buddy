from pathlib import Path

from app.services.document_loader import _build_pdf_document, load_and_split_documents
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
            "[Page 1]\n"
            "1.1 Large Numbers Around Us\n"
            "One lakh is written as 100000.\n\n"
            "[Page 2]\n"
            "1.2 Comparing Numbers\n"
            "We compare numbers using place value."
        ),
        metadata={
            "source": "sample.pdf",
            "file_name": "sample.pdf",
            "file_type": ".pdf",
            "subject": "mathematics",
            "grade": 7,
            "page_numbers": [1, 2],
            "related_images": [{"page_number": 2, "vision_caption": "place value chart"}],
        },
    )

    chunks = SectionAwareChunker(chunk_size=120, chunk_overlap=10).split(document)

    section_titles = {chunk.metadata["section_title"] for chunk in chunks}
    assert "Large Numbers Around Us" in section_titles
    assert "Comparing Numbers" in section_titles
    assert all(chunk.metadata["section_path"].startswith("mathematics") for chunk in chunks)
    comparing_chunk = next(chunk for chunk in chunks if chunk.metadata["section_title"] == "Comparing Numbers")
    assert comparing_chunk.metadata["page_number"] == 2
    assert comparing_chunk.metadata["related_images"][0]["vision_caption"] == "place value chart"


def test_pdf_loader_builds_one_full_document_before_section_chunking():
    extracted = {
        "source": "sample.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "1.1 Large Numbers Around Us\nOne lakh is written as 100000.",
                "images": [],
                "vector_drawing_count": 0,
            },
            {
                "page_number": 2,
                "text": "1.2 Comparing Numbers\nWe compare numbers using place value.",
                "images": [
                    {
                        "vision_caption": "place value chart",
                        "artifact_path": "artifacts/page_2_1.png",
                    }
                ],
                "vector_drawing_count": 1,
            },
        ],
    }

    document = _build_pdf_document(Path("sample.pdf"), extracted, "mathematics")

    assert document.metadata["loader_strategy"] == "full_pdf_then_section_split"
    assert document.metadata["page_numbers"] == [1, 2]
    assert document.metadata["related_images"][0]["page_number"] == 2
    assert "[Page 1]" in document.page_content
    assert "[Page 2]" in document.page_content
