# Grade 7 AI Study Buddy

**Grade 7 AI Study Buddy** is a portfolio-ready multimodal RAG tutor for middle-school students. It lets students ask questions across Grade 7 subjects and receives textbook-grounded explanations, citations, related diagrams/images, and practice questions.

This project is designed as an original portfolio project inspired by classroom RAG experiments. It uses only synthetic sample content in the repository. Add textbook PDFs locally under `data/textbooks/` only if you have the right to use them.

## Why This Project

Students often need help understanding textbook concepts, diagrams, and practice questions. A normal chatbot may answer from general knowledge, but a RAG tutor grounds answers in the actual textbook content.

## Features

- FastAPI `/chat` endpoint
- LangGraph supervisor routing
- Tutor Agent for textbook-grounded answers
- Quiz Agent for practice questions
- Markdown and PDF document ingestion
- PyMuPDF PDF text extraction
- PDF image extraction with noise filtering
- Base64 image conversion
- Context-aware vision captioning boundary
- Section-aware chunking with recursive fallback
- Dense, sparse, and hybrid retrieval modes
- RAG evaluation with golden questions and optional DeepEval metrics
- Related image retrieval with RAG chunks
- Local deterministic embeddings for offline demos
- Citations and student-safety guardrails
- pytest test suite

## Architecture

```text
Student / UI
  -> FastAPI /chat
  -> LangGraph Supervisor
       -> Tutor Agent
       -> Quiz Agent
  -> RAG Retrieval
       -> Markdown/PDF loaders
       -> PyMuPDF text/image extraction
       -> Vision captioning for clean images
       -> Chunking
       -> Embeddings
       -> Vector search
  -> Answer + citations + related images + practice questions
```

## RAG Ingestion

```text
Textbook sources
  -> load markdown/PDF
  -> extract text and images from every PDF page
  -> clean unwanted PDF text with regex
  -> filter noisy images
  -> convert clean image to base64
  -> caption image with page context
  -> merge PDF pages into one logical document
  -> create image embedding text
  -> split by textbook sections/headings
  -> recursively split oversized sections
  -> embed chunks
  -> retrieve relevant evidence with dense/sparse/hybrid search
```

## Retrieval Modes

The project supports three retrieval modes in `StudyVectorStore`:

| Mode | How it works | Best for |
| --- | --- | --- |
| Dense | Converts query and chunks into embeddings, then uses cosine similarity | semantic questions and paraphrases |
| Sparse | Uses regex tokenization and keyword frequency scoring | exact textbook terms, chapter names, formulas, IDs |
| Hybrid | Combines dense and sparse results using reciprocal rank fusion | balanced retrieval for meaning plus exact terms |

Current hybrid weighting:

```text
dense_weight = 0.6
sparse_weight = 0.4
```

Dense has a slightly higher weight because most student questions are semantic and conversational. Sparse still receives strong weight so exact terms like `photosynthesis`, `chlorophyll`, exercise numbers, formulas, or enterprise-style IDs are not missed.

Local examples:

```python
StudyVectorStore().search("photosynthesis chlorophyll", retrieval_mode="sparse")
StudyVectorStore().search("plants make food from sunlight", retrieval_mode="dense")
StudyVectorStore().search("what is photosynthesis", retrieval_mode="hybrid")
```

In production, sparse retrieval can be implemented with BM25/OpenSearch, dense retrieval with embeddings and a vector DB, and hybrid retrieval with weighted fusion or reranking.

## RAG Evaluation

The project includes two evaluation layers:

1. **Offline golden-question checks**
   These run with normal `pytest` and do not require an LLM judge or API key. They verify that important questions retrieve expected textbook evidence.

2. **Optional DeepEval metrics**
   These run when `deepeval` is installed and `RUN_DEEPEVAL=1` is set. DeepEval can score answer relevancy, faithfulness, and contextual relevancy using an LLM judge.

Golden questions live in:

```text
evaluation/golden_questions.json
```

Run normal tests:

```bash
python -m pytest
```

Install optional DeepEval dependencies:

```bash
pip install -r requirements-eval.txt
```

Run DeepEval-enabled tests:

```bash
$env:RUN_DEEPEVAL="1"
python -m pytest tests/test_rag_evaluation.py
```

DeepEval metrics used:

| Metric | Purpose |
| --- | --- |
| `AnswerRelevancyMetric` | checks if the answer responds to the question |
| `FaithfulnessMetric` | checks if the answer is grounded in retrieved context |
| `ContextualRelevancyMetric` | checks if retrieved chunks are relevant to the question |

Evaluation configuration:

```text
judge_model = EVALUATION_JUDGE_MODEL or OPENAI_MODEL
default judge_model = gpt-4o-mini
metric_threshold = 0.7
```

The evaluation framework intentionally uses the same model configuration as the RAG/chat path unless `EVALUATION_JUDGE_MODEL` is set separately. This keeps local behavior explicit and avoids relying on hidden DeepEval defaults.

Production use:

```text
Change chunking / embedding / top-k / hybrid weights / prompt
  -> run golden question tests
  -> run DeepEval metrics
  -> compare scores
  -> promote only if retrieval and answer quality stay above threshold
```

## Section-Aware Chunking

The ingestion pipeline uses section-aware chunking before embedding. Markdown files are split by `#`, `##`, and `###` headers. For PDFs, PyMuPDF first extracts text and images from all pages, then the extracted text is cleaned with Python `re` rules to remove unwanted PDF noise. After cleanup, the loader merges the PDF into one logical document with page markers such as `[Page 1]`. After that, textbook text is split by detected chapter, unit, uppercase heading, and numbered section patterns such as `1.1 Large Numbers Around Us`.

Regex cleanup removes common extraction noise such as repeated spaces, repeated blank lines, standalone page numbers, `Page 12` labels, reprint footer lines, and copyright/footer text. This keeps the vector DB focused on useful textbook content instead of PDF formatting artifacts.

Each chunk carries metadata such as:

- `subject`
- `grade`
- `file_name`
- `page_number`
- `page_numbers`
- `section_title`
- `section_path`
- `section_number`
- `chunk_strategy`

This improves retrieval because the vector store can return the exact textbook section instead of a random character slice. It also handles sections that continue across page boundaries. If a section is still too large, the project applies `RecursiveCharacterTextSplitter` inside that section so semantic boundaries are preserved as much as possible.

### Regex vs Separator Strategy

The project does **not** use simple separators as the first step for PDF section detection. For PDFs, it first uses regex-based heading detection in `app/services/chunking.py`.

Supported heading patterns include:

```text
1.1 Large Numbers Around Us
2.3 Plant Nutrition
5.2.1 Water Cycle
Chapter 1 Large Numbers
Unit 2 Science Around Us
LARGE NUMBERS AROUND US
```

The main regex patterns are:

```python
r"^(\d+(?:\.\d+){0,3})\s+(.+)$"
r"^(chapter|unit)\s+\d+[:\-\s]*(.+)?$"
```

After sections are detected, oversized sections are split using LangChain `RecursiveCharacterTextSplitter` with fallback separators:

```python
chunk_size=1000
chunk_overlap=0
separators=["\n\n", "\n", ". ", " ", ""]
```

The separator order matters:

- `"\n\n"` keeps paragraphs together when possible.
- `"\n"` handles line-based textbook formatting.
- `". "` splits long paragraphs at sentence boundaries.
- `" "` falls back to word-level splitting.
- `""` is the final character-level fallback when no cleaner boundary exists.

Because the first split is already based on textbook structure, heavy overlap is usually less important than it would be with simple fixed-size chunking. The project uses `chunk_overlap=0` for this reason. In production, this can be tuned per subject or document type if evaluation shows that answers lose context at boundaries.

Important detail: LangChain `RecursiveCharacterTextSplitter` uses character length by default, not true LLM token count. In production, token-aware splitters can be used when strict model context-window control is required.

So the real flow is:

```text
Full PDF text
  -> regex detects textbook section headings
  -> section-level documents are created
  -> section metadata is attached
  -> recursive splitter runs only inside oversized sections using paragraph/line/sentence/word fallback separators
  -> chunks are embedded and stored for retrieval
```

## Image Captioning

The image captioner receives:

- image bytes converted to base64
- subject
- grade
- source file
- page number
- nearby page text

This makes captions context-aware. In production, replace the local deterministic captioner with a vision-capable model such as AWS Bedrock Claude vision or another approved multimodal model.

## Run Locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8010
```

Open:

```text
http://127.0.0.1:8010/docs
```

## Sample Questions

```text
What is one lakh?
Explain photosynthesis in simple words.
Give me practice questions about one lakh.
Why do plants need sunlight?
```

## Interview Positioning

> I built a multimodal RAG tutor for Grade 7 students. It indexes textbook content, extracts text and images from PDFs, captions images using a vision-model boundary with page context, and retrieves relevant chunks with citations. A LangGraph supervisor routes student questions to a Tutor Agent or Quiz Agent. This helped me demonstrate RAG, PDF ingestion, image captioning, vector search, agent routing, and student-safe guardrails in an education use case.

## GitHub / Portfolio Summary

> Multimodal RAG tutor for Grade 7 students using FastAPI, LangGraph, PyMuPDF, image captioning, vector search, citations, and quiz generation.
