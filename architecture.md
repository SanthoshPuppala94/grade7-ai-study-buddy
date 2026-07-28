# Architecture - Grade 7 AI Study Buddy

## Business Problem

Grade 7 students need simple explanations grounded in their actual textbook material. They also need help understanding textbook diagrams and practicing concepts.

The system provides a RAG-based tutor that answers from indexed subject material instead of relying only on general LLM knowledge.

## High-Level Design

```text
Student
  -> FastAPI
  -> LangGraph Supervisor
  -> Subject-aware RAG retrieval
  -> Tutor / Quiz response
```

## Components

| Component | Responsibility |
| --- | --- |
| FastAPI | API layer for student questions |
| LangGraph Supervisor | Routes to Tutor Agent or Quiz Agent |
| Tutor Agent | Answers questions using retrieved textbook context |
| Quiz Agent | Generates practice questions |
| Document Loader | Loads markdown and PDF sources |
| Section-Aware Chunker | Preserves chapter, section, topic, page, and source metadata during chunking |
| PDF Extractor | Extracts text, images, and vector signals using PyMuPDF |
| Vision Captioner | Converts clean images to base64 and generates context-aware captions |
| Vector Store | Stores and searches embedded chunks |
| Guardrails | Blocks cheating requests and requires citations |

## Multimodal RAG Flow

```text
PDF page
  -> page text
  -> clean image artifacts
  -> base64
  -> context-aware image caption
  -> image embedding text
  -> section-aware chunking
  -> recursive fallback chunking
  -> embedding
  -> retrieval
  -> answer + related_images
```

## Chunking Design

The project uses a two-stage chunking strategy:

1. **Section-aware split**
   The loader first detects meaningful document sections. Markdown uses header-aware splitting. PDF textbook text uses common textbook heading patterns such as chapter titles, unit titles, uppercase lesson headings, and numbered sections like `1.1`, `2.3.1`, or `5.2`.

2. **Recursive fallback split**
   If a section is too large for retrieval, the system applies `RecursiveCharacterTextSplitter` within that section. This keeps each chunk small enough for embeddings and LLM prompts while preserving the original section metadata.

Every chunk stores section metadata including `section_title`, `section_path`, `section_number`, page number, source file, subject, grade, and `chunk_strategy`. In production, this metadata can be used for filtering, citations, debugging retrieval quality, and student-facing source references.

## Guardrails

- Student-safety guardrail blocks cheating/copying requests.
- Answers require citations when retrieved evidence exists.
- PDF image filters remove blank, oversized, tiny, and duplicate images.
- Image captioning uses page context so captions are not generic.
- Repository uses synthetic sample content and does not include copyrighted textbooks.

## Production Extensions

- Replace local hash embeddings with managed embeddings.
- Replace local deterministic captioner with a real vision model.
- Add ChromaDB/Pinecone/OpenSearch vector DB.
- Add authentication for student/teacher roles.
- Add curriculum metadata and grade-level answer tuning.
- Add evaluation with golden textbook Q&A datasets.
