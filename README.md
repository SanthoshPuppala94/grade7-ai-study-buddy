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
  -> extract page text
  -> extract images
  -> filter noisy images
  -> convert clean image to base64
  -> caption image with page context
  -> create image embedding text
  -> split by textbook sections/headings
  -> recursively split oversized sections
  -> embed chunks
  -> retrieve relevant evidence
```

## Section-Aware Chunking

The ingestion pipeline uses section-aware chunking before embedding. Markdown files are split by `#`, `##`, and `###` headers. PDF textbook text is split by detected chapter, unit, uppercase heading, and numbered section patterns such as `1.1 Large Numbers Around Us`.

Each chunk carries metadata such as:

- `subject`
- `grade`
- `file_name`
- `page_number`
- `section_title`
- `section_path`
- `section_number`
- `chunk_strategy`

This improves retrieval because the vector store can return the exact textbook section instead of a random character slice. If a section is still too large, the project applies `RecursiveCharacterTextSplitter` inside that section so semantic boundaries are preserved as much as possible.

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
