from dataclasses import dataclass, field
from pathlib import Path

from app.config import SAMPLE_DOCS_DIR, TEXTBOOKS_DIR
from app.services.chunking import SectionAwareChunker, SectionDocument
from app.services.pdf_extractor import extract_pdf
from app.services.text_cleaning import clean_pdf_text


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


def load_documents() -> list[Document]:
    docs: list[Document] = []
    docs.extend(_load_markdown_samples())
    docs.extend(_load_textbook_pdfs())
    return docs


def load_and_split_documents() -> list[Document]:
    chunks: list[Document] = []
    for doc in load_documents():
        chunks.extend(split_document(doc))
    return _number_chunks(chunks)


def split_document(document: Document) -> list[Document]:
    section_document = SectionDocument(
        page_content=document.page_content,
        metadata=document.metadata,
    )
    return [
        Document(page_content=chunk.page_content, metadata=chunk.metadata)
        for chunk in SectionAwareChunker().split(section_document)
    ]


def _load_markdown_samples() -> list[Document]:
    docs = []
    for path in sorted(SAMPLE_DOCS_DIR.glob("*.md")):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": ".md",
                    "subject": _subject_from_name(path),
                    "grade": 7,
                    "loader": "MarkdownSampleLoader",
                },
            )
        )
    return docs


def _load_textbook_pdfs() -> list[Document]:
    docs = []
    for path in sorted(TEXTBOOKS_DIR.glob("*.pdf")):
        subject = _subject_from_name(path)
        extracted = extract_pdf(path, subject=subject, grade=7)
        docs.append(_build_pdf_document(path, extracted, subject))
    return docs


def _build_pdf_document(path: Path, extracted: dict, subject: str) -> Document:
    page_blocks: list[str] = []
    related_images: list[dict] = []
    vector_drawing_count = 0
    page_numbers: list[int] = []

    for page in extracted["pages"]:
        page_number = page["page_number"]
        page_numbers.append(page_number)
        vector_drawing_count += page["vector_drawing_count"]

        text_parts = [f"[Page {page_number}]", clean_pdf_text(page["text"])]
        for image in page["images"]:
            image_with_page = {**image, "page_number": page_number}
            related_images.append(image_with_page)
            text_parts.append(
                f"[Related textbook image on page {page_number}]\n"
                f"Caption: {image['vision_caption']}\n"
                f"Artifact: {image['artifact_path']}"
            )
        page_blocks.append("\n\n".join(part for part in text_parts if part.strip()))

    return Document(
        page_content="\n\n".join(block for block in page_blocks if block.strip()),
        metadata={
            "source": extracted["source"],
            "file_name": path.name,
            "file_type": ".pdf",
            "subject": subject,
            "grade": 7,
            "page_numbers": page_numbers,
            "loader": "PyMuPDFTextImageLoader",
            "loader_strategy": "full_pdf_then_section_split",
            "text_cleaning": "regex_noise_cleanup",
            "related_images": related_images,
            "vector_drawing_count": vector_drawing_count,
        },
    )


def _number_chunks(chunks: list[Document]) -> list[Document]:
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata = {**chunk.metadata, "chunk_index": index, "chunk_count": total}
    return chunks


def _subject_from_name(path: Path) -> str:
    name = path.stem.lower()
    if "math" in name or "ganita" in name:
        return "mathematics"
    if "science" in name:
        return "science"
    if "social" in name:
        return "social studies"
    if "english" in name:
        return "english"
    return "general"
