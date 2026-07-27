import math
from dataclasses import dataclass

from app.services.document_loader import load_and_split_documents
from app.services.embeddings import HashEmbedding


@dataclass
class Chunk:
    source: str
    text: str
    metadata: dict
    embedding: list[float]


class StudyVectorStore:
    def __init__(self):
        self.embedding_model = HashEmbedding()
        self._chunks: list[Chunk] = []

    def ensure_index(self) -> None:
        if self._chunks:
            return
        docs = load_and_split_documents()
        self._chunks = [
            Chunk(
                source=f"{doc.metadata['source']}#chunk-{doc.metadata['chunk_index']}",
                text=doc.page_content,
                metadata=doc.metadata,
                embedding=self.embedding_model.embed_query(doc.page_content),
            )
            for doc in docs
        ]

    def search(self, query: str, k: int = 4, subject: str | None = None) -> list[dict]:
        self.ensure_index()
        query_embedding = self.embedding_model.embed_query(query)
        scored = []
        for chunk in self._chunks:
            if subject and chunk.metadata.get("subject") != subject:
                continue
            scored.append(
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": _cosine(query_embedding, chunk.embedding),
                    "related_images": chunk.metadata.get("related_images", []),
                }
            )
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right)) / (
        math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)) or 1.0
    )

