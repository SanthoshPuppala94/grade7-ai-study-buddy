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
| Vector Store | Stores chunks and supports dense, sparse, and hybrid retrieval |
| Guardrails | Blocks cheating requests and requires citations |

## Multimodal RAG Flow

```text
PDF page
  -> page text
  -> regex text cleanup
  -> clean image artifacts
  -> base64
  -> context-aware image caption
  -> full PDF logical document
  -> image embedding text
  -> section-aware chunking
  -> recursive fallback chunking
  -> embedding
  -> dense / sparse / hybrid retrieval
  -> answer + related_images
```

## Retrieval Design

The retrieval layer supports:

1. **Dense retrieval**
   Query and chunks are embedded, then compared with cosine similarity. This is useful when the student asks a semantic question using different wording from the textbook.

2. **Sparse retrieval**
   Query and chunks are tokenized with regex and matched using keyword-frequency scoring. This is useful for exact terms such as `photosynthesis`, formulas, exercise numbers, chapter names, or production-style identifiers in enterprise RAG.

3. **Hybrid retrieval**
   Dense and sparse result lists are fused with reciprocal rank fusion. This gives a better balance because dense search captures meaning while sparse search protects exact-match terms.

Current local weighting:

```text
dense_weight = 0.6
sparse_weight = 0.4
```

The 60/40 split is a starting point. Dense is weighted slightly higher because student questions are usually semantic. Sparse still has enough weight to preserve exact terms, formulas, section names, and IDs. In production, these weights should be tuned using golden questions, retrieval traces, recall metrics, and answer citation quality.

The local project implements sparse scoring directly to avoid extra dependencies. In production, this layer can be replaced with OpenSearch BM25 plus vector search, Chroma/FAISS plus BM25, or a managed hybrid retrieval service.

## Chunking Design

The project uses a two-stage chunking strategy. For PDFs, PyMuPDF extracts text and images page by page. Before chunking, Python `re` cleanup removes unwanted extracted text such as repeated spaces, repeated blank lines, standalone page numbers, `Page N` labels, reprint footer lines, and copyright/footer noise. Then the loader merges the pages into one logical PDF document before chunking. Page markers such as `[Page 1]` are preserved so each section chunk can still carry page metadata and related images.

1. **Section-aware split**
   The loader first detects meaningful document sections. Markdown uses header-aware splitting. PDF textbook text uses common textbook heading patterns such as chapter titles, unit titles, uppercase lesson headings, and numbered sections like `1.1`, `2.3.1`, or `5.2`.

2. **Recursive fallback split**
   If a section is too large for retrieval, the system applies `RecursiveCharacterTextSplitter` within that section. This keeps each chunk small enough for embeddings and LLM prompts while preserving the original section metadata.

Every chunk stores section metadata including `section_title`, `section_path`, `section_number`, page number, source file, subject, grade, and `chunk_strategy`. In production, this metadata can be used for filtering, citations, debugging retrieval quality, and student-facing source references.

### Regex and Separator Rules

PDF section detection is regex-first. The system does not initially split PDFs by character count because that can break textbook sections across arbitrary boundaries. Instead, it scans the merged full-PDF text line by line and identifies section headings.

The heading detector supports:

| Heading type | Example | Detection approach |
| --- | --- | --- |
| Numbered section | `1.1 Large Numbers Around Us` | Regex for dotted section numbers |
| Nested section | `5.2.1 Water Cycle` | Regex for multi-level dotted numbers |
| Chapter or unit | `Chapter 1 Large Numbers` | Case-insensitive chapter/unit regex |
| Textbook title heading | `LARGE NUMBERS AROUND US` | Uppercase heading heuristic |

Representative regex patterns:

```python
r"^(\d+(?:\.\d+){0,3})\s+(.+)$"
r"^(chapter|unit)\s+\d+[:\-\s]*(.+)?$"
```

After the section-level split, the recursive fallback uses:

```python
chunk_size=1000
chunk_overlap=0
separators=["\n\n", "\n", ". ", " ", ""]
```

The fallback separator order is intentional. Paragraph breaks are preferred first, then line breaks, then sentence boundaries using `. `, then words, and finally characters. The sentence separator matters because some textbook sections can be much larger than the model input budget; `. ` allows the splitter to break a long section into more natural sentence-level chunks instead of cutting in the middle of a concept.

The project uses `chunk_size=1000` and `chunk_overlap=0` because the first pass already splits by document structure. With section-aware chunking, overlap is less critical than with naive fixed-size chunking. If retrieval evaluation later shows boundary-loss issues, overlap can be tuned per source type.

LangChain's default recursive splitter measures character length, not exact LLM tokens. For strict production token control, the same design can be moved to a token-aware splitter or a tokenizer-backed length function.

This means separators are used for chunk-size control, not for the primary section discovery. In interview terms: **regex identifies textbook structure; recursive separators control chunk size.**

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
