from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import SAMPLE_DOCS_DIR, TEXTBOOKS_DIR
from app.services.pdf_extractor import extract_pdf


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
    if document.metadata.get("file_type") == ".md":
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "chapter"), ("##", "section"), ("###", "topic")],
            strip_headers=False,
        ).split_text(document.page_content)
        chunks = []
        for section in sections:
            section.metadata = {**document.metadata, **section.metadata}
            chunks.extend(_recursive_splitter().split_documents([section]))
        return chunks
    return _recursive_splitter().split_documents([document])


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
        for page in extracted["pages"]:
            text_parts = [page["text"]]
            for image in page["images"]:
                text_parts.append(
                    f"[Related textbook image]\n"
                    f"Caption: {image['vision_caption']}\n"
                    f"Artifact: {image['artifact_path']}"
                )
            docs.append(
                Document(
                    page_content="\n\n".join(part for part in text_parts if part.strip()),
                    metadata={
                        "source": extracted["source"],
                        "file_name": path.name,
                        "file_type": ".pdf",
                        "subject": subject,
                        "grade": 7,
                        "page_number": page["page_number"],
                        "loader": "PyMuPDFTextImageLoader",
                        "related_images": page["images"],
                        "vector_drawing_count": page["vector_drawing_count"],
                    },
                )
            )
    return docs


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
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

