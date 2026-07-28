import re
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@dataclass
class SectionDocument:
    page_content: str
    metadata: dict


class SectionAwareChunker:
    """Split study material while preserving chapter/section metadata."""

    def __init__(self, chunk_size: int = 850, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, document: SectionDocument) -> list[SectionDocument]:
        file_type = document.metadata.get("file_type")
        if file_type == ".md":
            sections = self._split_markdown_sections(document)
        else:
            sections = self._split_textbook_sections(document)

        chunks: list[SectionDocument] = []
        for section in sections:
            split_parts = self._recursive_splitter.split_documents([section])
            for part_index, part in enumerate(split_parts, start=1):
                part.metadata = {
                    **section.metadata,
                    **part.metadata,
                    "section_chunk_index": part_index,
                    "chunk_strategy": "section_aware_recursive",
                }
                chunks.append(SectionDocument(page_content=part.page_content, metadata=part.metadata))
        return chunks

    def _split_markdown_sections(self, document: SectionDocument) -> list[SectionDocument]:
        markdown_sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "chapter"), ("##", "section"), ("###", "topic")],
            strip_headers=False,
        ).split_text(document.page_content)

        sections: list[SectionDocument] = []
        for index, section in enumerate(markdown_sections, start=1):
            metadata = {**document.metadata, **section.metadata}
            metadata.update(_section_metadata(metadata, fallback=f"markdown-section-{index}"))
            sections.append(SectionDocument(page_content=section.page_content, metadata=metadata))
        return sections

    def _split_textbook_sections(self, document: SectionDocument) -> list[SectionDocument]:
        parsed_sections = _detect_textbook_sections(document.page_content)
        sections: list[SectionDocument] = []
        for index, parsed in enumerate(parsed_sections, start=1):
            page_numbers = _extract_page_numbers(parsed["content"])
            related_images = _filter_related_images(
                document.metadata.get("related_images", []),
                page_numbers,
            )
            metadata = {
                **document.metadata,
                "section_title": parsed["title"],
                "section_number": parsed.get("number"),
                "section_index": index,
                "section_path": _build_section_path(document.metadata, parsed["title"]),
            }
            if page_numbers:
                metadata["page_numbers"] = page_numbers
                metadata["page_number"] = page_numbers[0]
            metadata["related_images"] = related_images
            sections.append(SectionDocument(page_content=parsed["content"], metadata=metadata))
        return sections or [document]


def _detect_textbook_sections(text: str) -> list[dict]:
    lines = text.splitlines()
    sections: list[dict] = []
    current_title = "page-content"
    current_number: str | None = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        heading = _classify_heading(stripped)
        if heading and current_lines and not _only_page_markers(current_lines):
            trailing_page_markers = _pop_trailing_page_markers(current_lines)
            sections.append(
                {
                    "title": current_title,
                    "number": current_number,
                    "content": "\n".join(current_lines).strip(),
                }
            )
            current_title = heading["title"]
            current_number = heading.get("number")
            current_lines = [*trailing_page_markers, line]
            continue
        if heading and (not current_lines or _only_page_markers(current_lines)):
            current_title = heading["title"]
            current_number = heading.get("number")
        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "title": current_title,
                "number": current_number,
                "content": "\n".join(current_lines).strip(),
            }
        )
    return [section for section in sections if section["content"]]


def _classify_heading(line: str) -> dict | None:
    if not line or len(line) > 120:
        return None

    numbered_match = re.match(r"^(\d+(?:\.\d+){0,3})\s+(.+)$", line)
    if numbered_match:
        return {"number": numbered_match.group(1), "title": numbered_match.group(2).strip()}

    chapter_match = re.match(r"^(chapter|unit)\s+\d+[:\-\s]*(.+)?$", line, flags=re.IGNORECASE)
    if chapter_match:
        title = chapter_match.group(2) or line
        return {"title": title.strip()}

    words = [word for word in re.split(r"\s+", line) if word]
    if 2 <= len(words) <= 8 and line.upper() == line and any(char.isalpha() for char in line):
        return {"title": line.title()}

    return None


def _only_page_markers(lines: list[str]) -> bool:
    non_empty = [line.strip() for line in lines if line.strip()]
    return bool(non_empty) and all(re.match(r"^\[Page\s+\d+\]$", line) for line in non_empty)


def _pop_trailing_page_markers(lines: list[str]) -> list[str]:
    trailing: list[str] = []
    while lines and not lines[-1].strip():
        trailing.insert(0, lines.pop())
    while lines and re.match(r"^\[Page\s+\d+\]$", lines[-1].strip()):
        trailing.insert(0, lines.pop())
        while lines and not lines[-1].strip():
            lines.pop()
    return trailing


def _section_metadata(metadata: dict, fallback: str) -> dict:
    title = metadata.get("topic") or metadata.get("section") or metadata.get("chapter") or fallback
    return {
        "section_title": title,
        "section_path": _build_section_path(metadata, title),
    }


def _build_section_path(metadata: dict, title: str) -> str:
    parts = [
        str(metadata.get("subject", "general")),
        str(metadata.get("chapter", "")).strip(),
        str(metadata.get("section", "")).strip(),
        str(metadata.get("topic", "")).strip(),
        str(title).strip(),
    ]
    return " > ".join(part for part in parts if part)


def _extract_page_numbers(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"\[Page\s+(\d+)\]", text)]


def _filter_related_images(images: list[dict], page_numbers: list[int]) -> list[dict]:
    if not page_numbers:
        return []
    page_set = set(page_numbers)
    return [image for image in images if image.get("page_number") in page_set]
